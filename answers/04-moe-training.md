# 第 4 章：MoE 训练 Infra

[返回题目清单](../README.md#4-moe-训练-infra)

## 37. Expert Parallel 为什么需要 All-to-All？

Router 为每个 token 选择 top-k experts，而 expert 参数分布在不同 EP ranks。dispatch 阶段必须把 token hidden states 从原 rank 发到 expert 所在 rank；expert 计算后，combine 阶段再把输出送回原 token 位置。每个 rank 都可能与所有其他 rank 交换不同数量 token，因此语义是 all-to-all/all-to-all-v，而非 all-reduce。

完整路径通常是：router → 统计目标 rank/token 数 → permute/pack → all-to-all → grouped GEMM → reverse all-to-all → unpermute/加权。性能瓶颈不仅是网络，还包括排序、padding、内存拷贝和小 GEMM。若 EP 跨节点，应让拓扑 placement、通信 backend 与 expert 数共同设计。

## 38. Capacity factor、token dropping 和负载均衡损失

每个 expert 的容量常近似为 `capacity = capacity_factor × tokens × top_k / experts`。capacity factor 越大，越少丢 token，但 padding、显存和最慢 expert 时间增加。

- **Token dropping**：超出容量的 token 被跳过或走残差路径，可限制最坏负载，但改变训练语义。
- **Load-balancing loss**：鼓励 router 的概率质量和实际 token 数在 experts 间均衡；权重过大可能妨碍模型学习专业化。
- **Dropless**：不丢 token，保持语义，但必须支持变长通信和不等长 GEMM，长尾直接影响 step time。

面试时应同时讨论模型质量与系统吞吐，不能只说“调大 aux loss”。

## 39. 少数 Expert 持续过载如何定位？

分三层：

1. **路由层**：记录每层每 expert 的 token count、概率质量、top-k overlap、capacity overflow 和按数据域分布；检查 router collapse、初始化或 aux loss。
2. **通信层**：记录每个 peer 的 send/recv bytes、all-to-all 时间和等待时间，判断是否由单个目标 rank 主导。
3. **计算层**：比较 expert GEMM shape、kernel、SM 利用率和 padding；小而碎的 experts 可能 token 少但仍低效。

还要区分“expert 热”与“rank 热”：一个 rank 承载多个热门 expert 属于 placement 问题。可用 expert 重排、复制热门 expert、分层 all-to-all、动态 capacity 或 batch-aware routing 缓解。

## 40. Dropless MoE 如何处理变长 Token？

Dropless 不用固定容量裁掉 token，而是按 expert 统计真实数量，排序并压成连续 buffer。通信可使用 all-to-all-v 发送变长片段，接收端按 expert offset 执行 grouped GEMM，再逆变换。

主要取舍：

- padding 到统一容量实现简单、kernel shape 稳定，但浪费计算和显存；
- 完全变长减少浪费，但 prefix-sum、排序、通信元数据和大量小 GEMM 开销高；
- grouped GEMM 把多个 expert GEMM 合入一次调度，但极端不均衡仍受最大 expert 约束；
- token permutation 应 fusion，避免多次 HBM round trip。

实际优化目标是端到端 dispatch + compute + combine，而不是单独 GEMM TFLOPS。

## 41. TP 与 EP 如何组合？

TP 分片单个矩阵，EP 分布不同 experts。对 dense 层通常使用 TP；对 MoE experts 可降低 TP、增加 EP，让每个 expert 尽量位于较少 GPU 上，减少 expert GEMM 前后的 TP collective。

若一个 expert 太大放不进单卡，仍需 expert tensor parallel；这会让 token all-to-all 后再进入 TP group，通信更复杂。常见约束是令高频 EP all-to-all 尽量在高速域，或采用 hierarchical dispatch。

MoE 层与 dense 层不必共享同一布局，但布局切换需要 tensor reshard，可能吞掉收益。设计时应计算每层参数显存、激活 token 数、TP collective 与 EP all-to-all 的暴露时间。

## 42. DeepEP / Fused All-to-All 优化什么？

目标是把 token dispatch/combine 的 packing、跨 GPU 传输、接收计数和后续计算更紧密地流水化，减少 CPU 控制、额外拷贝和同步屏障。

- **高吞吐模式**：面向大 batch，追求链路带宽、分层 NVLink/RDMA 利用率和计算通信 overlap。
- **低延迟模式**：面向 decode 或小 token batch，减少启动次数、同步轮数和尾延迟，即使带宽利用率不是最高。

评估要包含 dispatch/combine latency、可 overlap 部分、SM 占用、workspace、最坏 peer 流量和对 GEMM 的干扰。通信 kernel 占满 SM 可能让“通信更快、总 step 更慢”。

## 43. MoE Tokens/s 周期性抖动如何诊断？

把每 step 的 expert token histogram、最大/平均 expert load、all-to-all bytes、collective 时间、grouped GEMM shapes 与数据 batch ID 对齐。

- 路由长尾：最大 expert load 与 step time 高相关，换节点仍复现。
- 网络拥塞：token 分布近似稳定，但特定链路/NIC 的吞吐、重传或等待异常。
- shape 变化：sequence packing 或动态 batch 令总 token/有效 token 周期变化，kernel 选择随 shape 跳变。
- 周期任务：checkpoint、evaluation、GC 或监控采集也可能伪装成 MoE 抖动。

先建立跨层 trace，再做固定输入、固定路由或单节点对照，避免只凭 NCCL 时间下结论。

## 44. 如何设计 MoE Placement？

输入包括 expert 参数大小、历史负载/共现、节点拓扑、故障域、GPU 显存和链路带宽。目标同时最小化最大 rank 计算、跨慢链路 token 流量和故障相关性。

可先把 EP group 放入 NVSwitch 或同 rail 域，再用加权 bin packing 分配 experts；高负载 expert 分散到不同 rank/节点，强共现 expert 避免集中。若支持复制热门 expert，router 还需选择副本并保证梯度同步。

动态迁移不能只看瞬时热度：迁移参数本身昂贵，会改变 optimizer/checkpoint shard。应使用滑动窗口、滞回阈值和安全切换点，并保留静态布局回退。
