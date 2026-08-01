# 第 9B 章：RL Training Infra——权重同步与异步 Pipeline

[返回题目清单](../README.md#93-权重同步与一致性)

## 99. 如何同步分片权重并避免 Rank 0 聚合？

核心是把同步建模为两个 distributed layouts 之间的 reshard，而不是“保存完整 checkpoint 再加载”。训练端发布 logical parameter manifest：参数名、全局 shape/dtype、源 shard offsets、版本；rollout 端给出目标 TP/EP shard offsets。编排器据此生成 source→destination transfer plan。

数据面由源 ranks 直接向目标 ranks 发送重叠 chunks，可用 NCCL send/recv、all-to-all 或 RDMA；按 layer/chunk 流水，目标端边接收边写入稳定 buffer。控制面只传 manifest、版本和完成确认。这样总带宽分散到所有 ranks，无 rank 0 的 `O(P)` 内存/网络热点。

发布不变量：同一 rollout replica 的所有参数必须来自同一个 committed policy version；仅当全部 shards 校验完成后原子切换 active version。失败时保留旧版本继续服务或整体重试，不能部分可见。

## 100. Ray、Shared Memory、NCCL 与 RDMA 如何选择？

- **Ray object store/RPC**：适合小对象、控制命令、placement 和 future；大权重会产生 serialization、CPU copy 和 object-store 压力。
- **CPU shared memory**：同节点不同进程可零/少拷贝共享 host buffer，适合 staging/offload，但仍需 H2D，受 NUMA 和 pinned-memory 限制。
- **NCCL**：GPU 集体/点对点传输，适合同一作业内已知 ranks，能利用 NVLink/RDMA；需要一致的 communicator 生命周期和调用顺序。
- **原生 RDMA/UCX 等**：适合解耦资源池、动态 endpoints 和定制传输，但内存注册、流控、重试与一致性都由系统承担。

成熟设计把 Ray 放控制面，tensor 走 NCCL/RDMA；同节点可走 CUDA IPC/NVLink。选择指标是端到端 sync critical path、CPU bounce bytes、峰值 staging memory 和故障可恢复性，而非单次带宽。

## 101. ZeRO-3/FSDP → TP 映射需要什么元数据？

至少需要 parameter 的 canonical name/ID、global shape、dtype、训练 flatten buffer 中的 global offsets、padding/alignment、目标 TP 切分轴与 offsets。还要描述 tied/shared weights、QKV fused layout、GQA head mapping、MoE expert ID、量化 scale 和可能的 transpose。

FSDP flat parameter 不能按本地 buffer 顺序直接广播给 TP rank；必须先恢复 logical tensor 坐标。版本升级时 parameter naming 也可能变化，因此 manifest 要带 schema 和模型结构 hash。

验证分三层：每 shard size/offset 完整覆盖且不重叠；全局 checksum 可由 shard checksum 组合；固定输入比较 trainer forward 与 rollout engine logits 的允许误差。

## 102. 全量、增量与 LoRA 同步如何取舍？

全量同步简单稳健，成本 `O(P)`，适合同步频率低或网络充足。增量同步可发送 changed blocks、delta 或低精度差分，但需要基版本、误差控制、周期性 full refresh 和断链恢复；dense optimizer update 通常让“只传变化参数”并不稀疏。

LoRA 只同步 adapter，成本与 rank/target modules 相关，适合 rollout base model 固定的 PEFT；必须保证 base model hash、adapter version、merge/unmerge 语义一致。

目标端先写 inactive buffer，完成后检查 manifest、byte count、checksum 和版本，再原子激活。不要用“RPC 返回成功”替代模型级一致性验证。

## 103. 如何发现 Rollout Worker 使用旧权重或错位 Shard？

每个 request/trajectory 记录 `(policy_version, rollout_replica, model_layout_hash)`；worker 只在完整加载后更新本地 active version。控制器比较期望版本与实际返回版本，并监控每版本样本比例和最大 lag。

错位 shard 往往不会崩溃，需要主动检测：同步后分层 checksum；固定 probe prompts 比较各 replicas 的 selected logits/token；周期性抽样与 trainer reference forward 对齐。若只检查最终文本，sampling 随机性会掩盖错误。

发现不一致时隔离整个 replica，撤销其未消费 samples，并保留 source/destination manifest、transfer trace 和 GPU 健康信息用于归因。

## 104. 如何让权重同步与 Generation Overlap？

双版本/双 buffer：active 权重服务当前 requests，inactive buffer 接收下一版本；传输完成后在 admission 边界切换，新请求使用新版本，旧请求可 drain。按 layer partial sync 若与 generation 同时读写同一 storage，会产生混合版本，必须通过 copy-on-write、pause/resume 或明确 partial-rollout 语义避免。

Overlap 的资源竞争也要计入：sync 占 NVLink/NIC/SM/HBM bandwidth，可能拖慢 decode。应限制 chunk/channel，优先关键 generation，比较 `generation slowdown + sync hidden ratio`。

严格 on-policy 场景通常选择 batch 边界 pause→sync→resume；允许 bounded staleness 时可双版本并发，但 sample 必须绑定实际版本。

## 105. Policy Version 与 Bounded Staleness 有什么作用？

Policy version 是一次 optimizer update 后可识别的不可变权重快照 ID。它让系统能计算 sample age、过滤过旧经验、重放故障批次并关联 reward/训练 step。

Bounded staleness 可定义为最大 version gap、最大 wall-clock age，或行为差异代理（例如 sampled policy 与 current policy 的 KL/ratio）。仅用队列长度不能保证策略差异小，因为不同 update 幅度不同。

系统应在 admission 和 trainer consume 两处执行上限；过期样本丢弃会浪费 rollout，需要反向调节 queue/资源。上限是算法-系统联合参数，必须用收敛与吞吐实验确定，不存在普适数字。

## 106. 同步 RL Pipeline 的 Bubble 从何而来？

同步 barrier 让阶段串行：trainer 等整批 rollout/reward；rollout GPU 在训练和权重同步时等待。令阶段时间为 `R`（rollout+reward）、`S`（sync/reshard）、`T`（train），独立资源池单轮 wall time 约 `R+S+T`，rollout 利用率约 `R/(R+S+T)`，trainer 约 `T/(R+S+T)`。

还存在 group barrier：一个 prompt 的多 responses 必须齐全才可计算 group-level 数据，最长 response 决定 ready time。优化前先用 trace 区分 stage bubble、group straggler 和资源内部低利用率。

可用 micro-batch pipelining、complete-group early dispatch、阶段重叠或共置时分复用减少 bubble，但必须保持所需版本一致性。

## 107. 异步 Pipeline 如何用 Bounded Queue 做 Backpressure？

Rollout producer 向 experience queue 写带版本的完整训练单元；trainer consumer 拉取。Queue 设容量、最大 age 和 per-version quota。达到 high watermark 时降低 admission/暂停 rollout；降到 low watermark 再恢复，使用滞回防振荡。

容量过大带来 GPU/CPU 内存和对象存储压力，更重要的是 sample policy lag 增大；容量太小则无法吸收 response 长尾，trainer 易饥饿。监控 queue depth 不够，还要看 age distribution、ready group 数、discard ratio 和 trainer starvation。

故障恢复时 queue item 要幂等、有唯一 sample/group ID，并区分 produced、rewarded、consumed、committed，防止重复训练。

## 108. 如何 On-Policy 地提前训练 Complete Groups？

若一个训练单元要求同一 prompt 的 `G` 条 responses，只要该 group 全部生成、reward/log-prob 完整，且使用同一冻结 policy version，就可 FIFO 发给 trainer，不必等待 batch 中其他 groups。

关键不变量：该 version 的 rollout 尚未完成时，trainer 更新不能让后续同一逻辑 batch 的 rollout 切到新权重。可保持 rollout side 冻结旧版本，trainer 处理已完成 groups；或双版本使旧请求继续 drain。调度器优先 admission “接近完成、能形成下一个 train micro-batch”的 frontier groups，避免碎片。

这减少 trainer wait，但增加 batch formation、版本生命周期和显存管理复杂度。

## 109. Partial Rollout 如何处理切权重与 KV Cache？

Partial rollout 暂停未完成序列，将 token history、sampling RNG/counter、stop state、environment state 和必要 KV 状态持久化。切权重后若继续沿用旧 KV，它由旧参数计算，与新权重不一致；选择包括：

1. 旧版本 worker 上继续完成，严格但占用双版本资源；
2. 新权重下从 token history 重算 KV，得到混合 policy trajectory；
3. 明确定义截断，已生成部分作为前缀，由新 policy 继续，并记录分段版本。

哪种可接受取决于算法。基础设施不能隐式混用，必须把 version boundary 编入 trajectory，并保证随机状态可审计。

## 110. Trainer 饥饿如何判断瓶颈？

构建每 sample 的分布式 trace：queue/admission、prefill、decode、tool/environment、reward、group-complete、transfer、trainer dequeue。看 critical-path 分位数而非平均值。

- sampling 瓶颈：decode GPU 饱和、tokens/s 低或长度增长；
- reward 瓶颈：completed responses 积压在 reward queue；
- 环境瓶颈：GPU 空闲但 tool/sandbox span 长；
- 调度瓶颈：总体容量够，但 ready groups 少、worker load skew 大。

调整资源前先确认训练消费率与数据生产率单位一致（samples、sequences 还是 tokens），否则会错误扩容。

## 111. 多级 Pipeline 如何做容量规划？

将 generation、reward、reference log-prob、actor train 视为服务站。对阶段 `i`，实测每 GPU/CPU 的 token 或 sample service rate `μi`、到达率 `λ`、长尾和 batch efficiency，配置满足稳定条件 `λ < capacity_i × μi`，并留故障/抖动余量。

Buffer 放在可解耦边界，保存最小必要 tensor；group-level 依赖处不能任意拆分。权重更新形成反馈 barrier，需要定义版本窗口和 drain 策略。瓶颈变化时优先动态 batch/调度，再调整资源池。

最终用 end-to-end goodput、GPU-hours/accepted sample、policy lag、discard rate 和收敛曲线评估，而不是把每阶段 GPU 利用率都优化到 100%。
