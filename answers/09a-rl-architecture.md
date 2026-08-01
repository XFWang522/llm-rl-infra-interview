# 第 9A 章：RL Training Infra——架构与资源编排

[返回题目清单](../README.md#9-rl-training-infra不考算法推导)

## 87. RL 训练端到端数据流是什么？

典型路径：prompt dataset → rollout engine 生成 responses/trajectories → reward/verifier 打分 → actor/reference（以及需要时 critic）计算 token log-probs/value → 组装训练 batch → trainer backward/update → 新权重回流 rollout。

Prompt、文本、sample ID、版本号属于小控制/元数据；token IDs、logits/log-probs、rewards、masks、advantages 和权重属于大 tensor。Ray/RPC 适合控制面，大 tensor 应走 GPU collective、RDMA、shared memory 或分片存储，避免序列化进 Python object store。每条 sample 必须携带 policy version 与 lineage。

## 88. 各角色执行什么计算和并行？

Actor 同时做 rollout generate 和训练 forward/backward；Reference 通常只 forward 计算基准 log-prob；Reward/Verifier 可能是规则 CPU 服务或独立模型 forward；Critic 若存在则训练 value model。

Actor trainer 常用 FSDP/ZeRO/Megatron 3D/4D；rollout 用 TP/DP 的 vLLM/SGLang；Reference/Reward 只读，可用更激进的 TP、量化或副本扩展。不能让所有角色机械复用同一并行度：训练受 optimizer/activation 约束，生成受 KV cache 和 decode 并发约束。

## 89. 如何为角色分配 GPU？

先 profile 每个阶段的 service time 与显存，再使稳态产能匹配：`rollout_rate ≈ reward_rate ≈ train_consumption_rate`。静态配比简单稳定；动态伸缩根据 queue depth 调整，但模型加载/reshard 昂贵；自动 mapping 需满足 TP/PP group 的 gang 和拓扑。

还需决定角色共置、时分复用或独立资源池。目标是最大化 end-to-end goodput，而不是每个角色局部利用率。资源调整设置滞回与最小驻留时间，避免因短期队列波动频繁迁移。

## 90. RL 训练为何不是普通多模型训练？

普通多模型 pipeline 多为固定 DAG；RL 的数据由当前 policy 在线生成，模型更新又反过来改变数据分布，形成反馈环。每个逻辑角色内部还是分布式程序，边上常发生多对多 tensor 重分布。

额外问题包括生成长度长尾、训练/推理不同 layout、频繁权重同步、policy staleness、group 完整性、外部环境失败和异步一致性。调度错误不仅影响吞吐，还可能改变 on-policy 语义与收敛。

## 91. veRL/HybridFlow 的控制抽象解决什么？

Single controller 集中描述 RL 数据流和角色调用，而 worker group 封装每个角色内部的分布式 SPMD 程序；这样控制器不必逐 rank 编写通信逻辑。数据从一个 worker group 到另一个 group 时，由协议完成 dispatch/collect/reshard。

Ray 适合 actor 生命周期、placement group、RPC 和失败通知；全量权重/大 activation 不应经 Python 控制器中转。控制器还必须避免成为单点吞吐瓶颈，并让操作具有 step/version/idempotency 语义以支持恢复。

## 92. Trainer 与 Rollout 引擎如何组合？

FSDP/Megatron 和 vLLM/SGLang 各自创建 process groups、CUDA streams、allocator 和模型 layout。编排层需明确 rank namespace、GPU ownership、初始化顺序和独立/共享 NCCL communicator，防止所有 ranks 以不同顺序创建 groups 而 hang。

共置时还要管理显存：训练阶段释放/休眠 KV cache，rollout 阶段 offload optimizer/grad 或释放 activation；预留通信 workspace，避免 allocator 碎片。正确性测试应覆盖反复 train↔rollout 切换、权重版本和多轮故障恢复。

## 93. Colocate 与 Disaggregate 如何取舍？

Colocate 在同一 GPU 池时分复用：无需跨池传权重，资源不因阶段屏障长期空闲，适合小集群和严格 on-policy；但训练 states 与 KV cache 争显存，切换/offload 有成本，长 rollout 容易 OOM。

Disaggregate 使用独立 rollout/trainer 池：可分别选择硬件和并行度、并流水重叠；代价是全量权重同步、跨池网络、静态配比浪费和 staleness。选择应比较 `generation time、train time、weight-sync time、切换 time、峰值显存`，并验证收敛约束。

## 94. Hybrid Engine 如何做 Role Switching？

训练结束后确保 optimizer step 完成，将参数转换/暴露为 rollout layout，暂停或 offload optimizer/gradient，恢复推理 engine 与 KV cache；rollout 完成后清空/休眠 cache，恢复训练 states。模型权重可共享 storage、重新 materialize，或通过 GPU 内 collective reshard。

关键是生命周期：同步 streams、禁止悬挂 request、控制 allocator、保持 RNG/version 一致。Sleep/offload 节省显存但涉及 CPU/PCIe 开销；共享 storage 快但要求框架 dtype/layout 兼容。应测完整切换而非只测 generate/train。

## 95. Colocate 长 Response OOM 怎么查？

按时间记录 allocated/reserved，并标注 weights、optimizer、activation、KV blocks 和 sync buffer。若 KV 随 generated tokens 线性上升，是 cache capacity；若进入 backward 才峰值，是 activation；allocated 低而 reserved 高可能是碎片；切换后旧 engine 引用未释放则是生命周期泄漏。

做对照：限制 max tokens/KV utilization、关闭并发、启用 checkpoint、清空 sleep cache、改变 allocator。修复可采用更严格 admission、KV paging/offload、阶段间释放、降低 rollout TP 并发或转 disaggregate，不能只靠 `empty_cache()`。

## 96. Disaggregate 下如何动态调整 GPU 比例？

估计 rollout 每 GPU samples/s、trainer 每 GPU samples/s 和 reward rate，基于 queue depth/age 做闭环。增加 rollout GPU 前要考虑模型加载与 TP gang；减少时先 drain in-flight requests。trainer resize 还涉及 optimizer reshard，成本更高。

使用滞回、冷却时间和最小 gang 单位，目标保持有限 buffer 而非零队列。把 weight-sync 网络和 policy lag 加入代价函数；只根据瞬时 queue depth 会振荡，并可能生成大量很快过期的样本。

## 97. 训练 Layout 到推理 Layout 如何 Reshard？

建立 logical parameter metadata：全局 shape、轴语义、训练 shard offsets、推理 TP shard offsets。每个目标 rank 计算与源 shards 的重叠区间，形成 all-to-all/P2P 计划；相同 group 内连续 shards 可用 all-gather，复杂转置常需 all-to-all。

不能先 gather 到 rank 0：会产生单点显存和带宽。传输应分 layer/chunk 流水，边到达边加载，并处理 padding、tied weights、MoE experts、dtype conversion。结束后用版本 manifest、参数 checksum 或抽样 forward 验证所有 rollout ranks 完整一致。

## 98. Reference/Reward 何时共置？

共置适合模型较小、阶段不重叠、显存允许且希望减少跨节点 token/log-prob 传输；可在 actor 空闲窗口执行只读 forward。独立池适合 reward 模型大、服务时间长、需要独立扩缩容，或希望 rollout/reward/train 流水并发。

Reference 参数通常冻结，可量化、offload 或按需 materialize；Reward 还可能是远程服务。决策应看峰值显存、重复权重、跨池通信、stage critical path 与故障隔离。共置后局部 GPU 利用率提高，不代表端到端吞吐一定提高。
