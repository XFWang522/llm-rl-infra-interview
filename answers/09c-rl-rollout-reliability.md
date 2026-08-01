# 第 9C 章：RL Training Infra——Rollout、环境与可靠性

[返回题目清单](../README.md#95-rolloutreward-与-agent-环境)

## 112. Response 长尾如何调度？

同步 static batch 的完成时间由最长 response 决定，短序列释放的 slot 无法复用。Continuous batching 让已完成 slot 立即接新请求；按历史/预测长度分桶降低同批方差；work stealing 把未开始请求移给空闲 worker。

正在 decode 的 request migration 需要转移 token state 与每层 KV cache，传输量大，只有剩余时间收益超过迁移成本才值得。更实用的是 prefill/decode 分离、chunked prefill、优先完成接近 group-ready 的请求，以及设置 max-token/cancel admission。

目标应是 trainer-ready groups/s 和尾部完成时间，而不只是 rollout tokens/s；过度偏爱短请求会造成长请求饥饿，需 aging/fairness。

## 113. Group Locality 如何影响系统？

同一 prompt 的多 responses 若在同一 rollout replica，可复用 prefix KV/cache，减少 prefill；但全部放一处会让单 worker 承担该 group 的长度长尾。分散到多 workers 提高并行度，却需跨 worker regroup token、reward、log-prob，并等待最慢成员。

调度可先在同一 prefix-cache domain 内分散到多个 decode slots，并让 group coordinator 维护成员状态。对 group-level 训练，只有完整/满足容错规则的 group 才进入 ready queue。选择依赖 prompt 长度、response 数、KV cache 容量和网络成本。

## 114. Reward/Verifier 的部署方式如何设计？

规则函数适合 CPU sandbox，本地、确定、便宜；GPU reward model 需 dynamic batching/TP；远程 judge 服务则受限流、网络和版本控制影响。统一接口应返回 sample ID、verifier version、score、错误类型和 trace。

请求必须幂等，重试复用 request ID；timeout 区分排队、执行和外部依赖。批处理按模型/shape/优先级，不能让一个慢 verifier 阻塞全部。失败策略（丢弃、默认分、重试、降级）会改变训练数据分布，必须显式记录并监控。

## 115. Reward 服务变慢如何端到端归因？

为 trajectory 建 trace ID，记录 rollout finish、reward enqueue/dequeue、batch formation、model compute、外部 RPC、result commit。指标包含 queue age、batch size、service time 分位数、错误/重试、GPU utilization 和按 verifier/version 分组。

若 compute 稳定而 queue 增长，是容量不足；batch formation 长说明流量碎片/调度；外部 span 长是依赖问题；少量 sample 拖慢需关联输入长度/工具类型。监控必须能从 trainer starvation 下钻到具体 sample，而非只有服务平均 QPS。

## 116. 多轮 Agent Rollout 的基础设施难点

Trajectory 不只是 token，还包含 observation、tool call、tool result、环境 snapshot、权限和时间。每个环境实例需要唯一 episode ID、step sequence 与幂等 action ID；sandbox 限制网络、文件、CPU/内存和时间。

不可重放副作用（发送消息、真实交易）不能直接用于训练环境，应使用模拟器、只读代理或事务性 dry-run。Timeout 分 agent think、tool queue、tool execution。环境和工具版本必须进入 lineage；仅保存对话文本不足以复现。

## 117. 部分环境 Step 失败如何处理？

先分类 deterministic task failure、transient infra failure、policy-caused invalid action。Transient 可在相同 state/action ID 下幂等重试；policy 错误应作为有效轨迹信号；环境损坏则从最近 snapshot 恢复或终止。

丢弃会产生选择偏差，默认惩罚会把 infra 故障误当 policy 行为。系统需把 terminal reason、retry count、有效 prefix 和环境证据传给数据策略，由算法明确决定截断/bootstrap。所有状态转换使用 append-only event log，便于审计和重建。

## 118. 超长 Response 有哪些联动影响？

Rollout KV cache 随 active tokens 线性增长，decode 时间和调度长尾增加；训练侧 activation/attention 成本上升，CP/checkpoint 需求变化；token tensor、log-prob 和权重版本生命周期延长。一个极长 sample 还会推迟 group ready，放大 policy lag。

系统需 token-budget admission、长度感知 batching、varlen kernels、按 token 而非 sequence 做负载/容量核算，并限制单 group 的最大在途资源。截断策略影响训练语义，不能由推理引擎静默执行。

## 119. 不同角色失败时恢复边界是什么？

Rollout worker 失败可重试未完成 request；若 sampling 必须可复现，保存 seed/counter 和版本。Reward worker 失败按 sample ID 幂等重算。Trainer rank 失败通常使整个同步 step/process group 失效，应从上一个 committed checkpoint 恢复，除非框架支持可靠局部重组。

Experience 的状态机至少包含 created、rolled-out、rewarded、queued、consumed、train-committed。只有 train commit 后才能最终回收；恢复时去重，避免同一 sample 更新两次。恢复粒度越细，元数据/一致性成本越高。

## 120. RL 吞吐如何分解？

按版本/step 记录 rollout admission/prefill/decode、environment、reward、reference/log-prob、experience transfer、weight reshard/sync、train forward/backward/optimizer，以及各阶段等待。

区分 service time 与 queue time，计算 critical path、overlap 和各资源池 goodput。端到端指标建议用 accepted training tokens/s、GPU-hours/committed sample、trainer starvation、rollout idle、sync exposed time 和 discard ratio。局部 tokens/s 提高但过期样本增加，整体可能更差。

## 121. RL 训练需要哪些额外指标？

除通用 MFU/显存/通信外，需要 response length 与完成时间分布、group-ready latency、queue depth/age、policy-version lag、weight-sync duration/failure、invalid/truncated/duplicate/discard samples、reward latency/error/distribution、环境/tool 成功率和 trainer data wait。

指标必须按 model/version、worker、task/domain 和 failure reason 分维度，但控制 cardinality；sample-level 细节进入 trace/log。算法指标与系统指标同时间轴展示，才能发现 reward 突变其实来自 verifier 降级或截断率变化。

## 122. Sample Lineage 如何设计？

使用不可变 sample/trajectory ID，关联 prompt dataset/version、policy version、rollout engine/config/seed、token sequence、environment events、reward/verifier version、所有 masks/log-probs、queue 状态、trainer step 和产出 checkpoint。

大 tensor 存对象存储，metadata 保存 content hash/URI；状态转换 append-only 并带幂等 key。隐私数据做访问控制和保留期限。系统应支持从异常 checkpoint 反查贡献 samples，也能从 sample 追到最终是否 committed，而非只记录“生成过”。

## 123. 设计 70B 千卡 RL Training 平台

先建立 workload model：prompt/response token 分布、每 prompt samples、reward 类型、目标 batch/tokens、允许 policy lag。资源分为 rollout、reward/environment、trainer；trainer 用 TP/PP/CP+DP/FSDP，rollout 用独立 TP replicas，按实测 `R:T` 服务率配比。

数据面：GPU-direct shard-to-shard 权重同步，experience 走分片 buffer/object store；控制面负责 placement、version、状态机。Disaggregate 支持流水，保留双版本 rollout；严格场景在版本边界 drain。长度感知 continuous batching、frontier-group priority 和 bounded queues 控制长尾。

可靠性：trainer step checkpoint，request/group 级幂等恢复，坏节点隔离；sample lineage 与 policy checksum 防 silent mismatch。容量预算必须列出 trainer states/activation、rollout weights/KV、sync staging、网络峰值以及 `R/S/T` critical path。

验收不是“能跑”：要求 loss/logit 对齐、故障注入、扩展效率、accepted tokens/s、policy lag、恢复时间和 GPU goodput SLO，并为 colocate/disaggregate、同步/异步提供可回退配置。

P7 级设计还应给出数字化示例：由 70B 参数 layout 推导 trainer/rollout 单副本卡数与同步字节量；用实测 response-token 分布估算 KV 容量和 group P99；以 `R/S/T` 时间决定资源比例；用 job MTBF、checkpoint cost 和 sample discard cost 估算故障预算。没有这些量化输入，架构图无法证明可行。
