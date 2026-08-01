# 第 3 章：Tensor / Pipeline / Context Parallel

[返回题目清单](../README.md#3-tensor--pipeline--context-parallel)

## 25. Column-parallel 与 row-parallel linear 如何切分？

线性层 `Y=XW`。Column parallel 按 `W` 的输出维切分：`W=[W1,...,Wp]`，每个 rank 算 `Yi=XWi`。若下游能继续消费分片输出，不需要立即通信；若需要完整 `Y`，做 all-gather。

Row parallel 按输入维切分：`W=[W1;...;Wp]`，输入也切为 `X=[X1,...,Xp]`，每个 rank 算局部部分积 `Yi=XiWi`，再 all-reduce 得到 `Y=ΣYi`。Megatron 通常把 column-parallel 与 row-parallel 成对使用，让中间分片保持本地，只在块末规约一次。

## 26. MLP 与 attention 如何做 TP？

MLP 的第一个投影按输出维切分，每个 rank 得到一部分 intermediate channel；激活函数本地执行；第二个投影按输入维切分并 all-reduce。这样中间大张量无需 gather。

Attention 中 Q/K/V 常按 head 维切分，每个 rank 独立计算部分 heads；输出投影用 row parallel 并规约。GQA 的 KV heads 少于 TP size 时需复制或重新分组，不能机械地一 rank 一个 KV head。

连续层之间的 reduce/gather 有时可通过保持分片 layout、使用 reduce-scatter + sequence parallel 消除。判断依据是下一个算子是否能直接消费当前 shard。

## 27. TP 为什么优先放在 NVLink 域内？

TP 几乎每个 Transformer block 都有 collective，消息频繁且处于关键路径，难以靠增加 gradient accumulation 降低频率。NVLink/NVSwitch 的带宽和延迟显著优于跨节点 RDMA，所以通常令 `TP size ≤ 每节点 GPU 数`。

跨节点 TP 会让每层 forward/backward 都等待网络，容易放大抖动。只有模型单层参数/activation 无法放入单节点、节点内 GPU 数不足，或跨节点互联极强时才考虑。拓扑映射应让 TP 使用最快链路，DP 使用较慢链路，PP 位于两者之间。

## 28. Pipeline bubble 如何推导？

GPipe 把一个 batch 分为 `m` 个 micro-batch，经过 `p` 个 stage；理想化每个 forward/backward stage 时间相等。仅看一方向流水，填充与排空共浪费 `p-1` 个时隙，有效比例约为：

`m / (m + p - 1)`，bubble ratio 为 `(p-1)/(m+p-1)`。

训练含 forward/backward，具体公式取决于 schedule，但结论一致：`m` 相对 `p` 越大，bubble 越小。真实系统还受 stage 不均衡、通信、重算和 optimizer step 影响，不能只套理想公式。

## 29. GPipe、1F1B、interleaved 与 zero-bubble 的差异

- **GPipe**：先跑完所有 forward，再跑 backward；实现简单，但保存大量 micro-batch activation。
- **1F1B**：warmup 后交替执行一个 forward 和一个 backward，降低峰值 activation；总 bubble 未必消失。
- **Interleaved 1F1B**：每个物理 rank 持有多个 virtual stages，缩短逻辑 stage 粒度，减少 bubble，但增加通信次数和调度复杂度。
- **Zero-bubble**：利用 backward 中 input-gradient 与 weight-gradient 的依赖差异重新排程，把空隙填入可延后执行的工作；要求更细的 autograd/optimizer 控制。

比较时应同时看 bubble、峰值显存、P2P 通信、参数版本语义和实现约束。

## 30. Virtual pipeline stage 的收益与代价

将每个物理 rank 的连续层拆为多个 chunk，例如 rank 0 同时持有逻辑 stage 0 和 4。micro-batch 在更细粒度的逻辑 stage 间交错，能减少 pipeline 空闲并改善层数不能均匀切分的问题。

代价是激活在 rank 间传输更频繁、调度状态更多、同一 rank 的多个 chunk 争用显存和 stream；chunk 太小还会让 P2P latency 主导。应基于每层时间、链路性能和 micro-batch 数选择 virtual size。

## 31. Pipeline stage 不均衡如何定位？

按 stage 分解 forward、backward、P2P、recompute 和 idle 时间，检查最慢 stage 是否稳定。常见根因包括 embedding/lm-head 参数大、首尾 stage 额外 loss、层 FLOPs 不等、MoE 路由倾斜、不同 checkpoint 策略和节点降速。

优化可按实测时间而非层数切分；把 embedding 与 head 拆分/共享；使用 virtual stages；对特殊层单独计权。吞吐由最慢 stage 决定，因此让平均负载相等不够，要降低最大 stage time 和方差。

## 32. Sequence Parallel 切分什么？

Megatron SP 通常在 TP group 内按 sequence 维切分原本复制的 activation，使 LayerNorm、Dropout 等逐 token 算子各 rank 只处理部分 token。TP block 边界常用 reduce-scatter 把结果变成 sequence shard，需要 TP 计算前再 all-gather 回所需 layout。

SP 主要节省 activation，不分片模型参数；它与 TP 共用 process group，避免 TP 后 activation 在各 rank 重复。必须正确处理 dropout RNG、残差和 layout 转换。

## 33. Context Parallel 与 Ring Attention

CP 将长序列 token 分到多个 rank，每个 rank 持有本地 Q/K/V。为了让本地 Q 看到全局 K/V，Ring Attention 在环上逐轮传递 KV block；每轮计算本地 Q 与当前 KV 的局部 attention，并用 online softmax 合并最大值、归一化因子和输出。

每 rank 的 KV/activation 显存约降为 `1/CP`，但引入 `CP-1` 轮通信。应把 KV 传输与 attention kernel overlap，并正确应用 causal mask。CP 解决长上下文 activation，和按 hidden/head 切分的 TP 是不同维度。

因 causal attention 不同 sequence block 的有效计算量不同，简单连续切分会造成 CP ranks 负载不均；常见 zig-zag/对称块映射把序列前后片段配对。Backward 还需传递/累加对应 KV gradients。GQA/MQA 虽减少 KV 大小，但不消除全上下文依赖；通信协议与 head layout 必须联合设计。

## 34. Ulysses 与 ring-based CP 如何取舍？

Ulysses 通过 all-to-all 在 sequence-sharded 与 head-sharded layout 间转换，使每 rank 拿到完整序列的一部分 heads，然后本地算 attention。它通信轮数少，但并行度受 attention head 数约束，all-to-all 对网络和不均匀消息敏感。

Ring 方案不要求一次全局转置，适合更大 CP size 和超长序列，可流式 overlap，但需要多轮点对点通信，latency 与实现复杂度更高。选择依据包括 head/KV-head 数、序列长度、拓扑、all-to-all 性能和 causal load balance。

## 35. 4D 并行的 world-size 与 rank group

通常满足：

`world_size = DP × TP × PP × CP`。

把 global rank 映射为四维坐标 `(d,t,p,c)`；某维 process group 固定其他坐标，只改变该维。例如 TP group 固定 `d,p,c`，遍历 `t`。创建 group 的顺序必须在所有 rank 一致，否则初始化会 hang。

数据 sampler 只跨 DP 区分样本；TP/CP ranks 处理同一样本的不同 tensor shard；PP ranks 处理不同层。若引入 EP，dense DP 与 expert DP 的定义可能改变，不能简单再乘一个维度。

CP 只切 activation、不切权重，因此 CP ranks 上存在权重副本；这些副本的 weight gradients 也必须规约。Megatron 常把 CP ranks 纳入相应 data-parallel gradient-reduction domain。只写 `DP×TP×PP×CP` 的乘积不足以定义正确通信，还必须明确每类参数使用哪个 gradient group。

## 36. 70B、128×8 H100 如何设计并行方案？

总 GPU 为 1024。一个合理起点例如 `TP=8, PP=8, CP=2, DP=8`，乘积为 1024：TP 完全位于单个 8-GPU NVSwitch 节点；PP 跨节点传 activation；CP 用于长上下文；DP 承担梯度/optimizer 分片。

这不是唯一答案。步骤应是：

1. 由单层参数、activation 和 kernel shape 决定最低 TP/CP。
2. 用层数与 stage 显存决定 PP，并按实测 layer time 平衡。
3. 剩余 GPU 给 DP，选择 DDP、distributed optimizer 或 FSDP。
4. 把高频 TP/CP/EP collective 映射到最快链路。
5. 用短跑测峰值显存、MFU、暴露通信和 stage imbalance，再搜索邻近配置。

若上下文较短可令 CP=1、增加 DP；若网络跨节点较弱可提高 PP、避免跨节点 TP。答案必须带上序列长度、micro-batch、网络拓扑等假设。

## 审校依据

- [Megatron Core Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)：CP 切分全部 sequence activation，而 SP 只覆盖部分逐 token activation。
- [Megatron parallel state](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.parallel_state.html)：CP 权重复制及 weight-gradient reduction group。
