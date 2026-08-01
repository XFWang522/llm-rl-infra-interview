# 第 2 章：数据并行、ZeRO 与 FSDP

[返回题目清单](../README.md#2-数据并行zero-与-fsdp)

## 13. DDP 的一次训练迭代发生什么？

DDP 在每个 rank 保存完整模型和 optimizer state，不负责切分输入；通常由 `DistributedSampler` 给各 rank 分配不同样本。

1. **初始化**：rank 0 的参数和 buffer 广播到其他 rank，保证初始状态一致。
2. **Forward**：各 rank 用本地 micro-batch 独立计算；默认可在 forward 前同步指定 buffer。
3. **Backward**：autograd 产生某个参数梯度后，DDP hook 将它标记 ready；一个 bucket 全部 ready 后异步发起 all-reduce。
4. **归一化**：all-reduce 得到梯度和，再除以 data-parallel world size，得到平均梯度。
5. **Optimizer step**：每个 rank 用相同梯度独立更新完整参数，因此参数继续一致。

all-reduce 不是等整个 backward 完成后才统一执行，而是按 bucket 尽早启动，以覆盖后续反向计算。使用 gradient accumulation 时，前几个 micro-step 应用 `no_sync()`，只在最后一次 backward 同步。

## 14. DDP reducer 如何构造 bucket？参数顺序为何重要？

Reducer 为参数注册 autograd hook，并把梯度按 dtype/device 和 bucket 大小组织到连续 buffer。Backward 时，某 bucket 内所有梯度 ready 后即可启动 collective。

Backward 的梯度通常按 forward 参数使用顺序的反向产生。如果 bucket 顺序与实际 ready 顺序不匹配，早已完成的梯度会等待 bucket 中最后一个参数，通信启动变晚，compute/communication overlap 下降。DDP 首轮可记录实际 ready 顺序并重建 bucket。

调优关注：

- bucket 太小：collective 次数多，latency 开销大；
- bucket 太大：启动晚，峰值 buffer 较大，overlap 变差；
- 不同 dtype 不会放在同一 bucket；
- `gradient_as_bucket_view` 可让 `param.grad` 直接成为 bucket view，减少一次梯度拷贝，但限制某些原地操作。

还要区分 DDP 版本与配置：`init_sync=True` 会在构造期校验 shape 并同步参数/buffer；`broadcast_buffers` 是 forward 期 buffer 同步，不是参数同步。注册后不得随意增删参数或改变各 rank 参数顺序，否则 reducer hook 与 collective 序列可能失配。

## 15. `find_unused_parameters` 为何有开销？static graph 优化什么？

启用 `find_unused_parameters=True` 时，DDP 每轮从 forward 输出遍历 autograd graph，提前找出不会收到梯度的参数，并将其标记 ready，防止 reducer 永久等待。这带来额外图遍历与同步逻辑。

如果计算图和参数使用集合每轮不变，可使用 static graph 模式：

- 不再每轮重复搜索 unused parameters；
- reducer 可复用固定的 hook/ready 顺序；
- 更容易支持固定的 reentrant backward/checkpoint 图。

若控制流导致不同 rank 使用不同参数，单纯打开该选项不能修复语义分歧；仍可能发生 collective 顺序不一致或不同梯度被错误平均。应保证各 rank 图一致，或显式设计同步逻辑。

## 16. Ring All-Reduce 如何工作？

将每个 rank 的 tensor 均分成 `N` 个 chunk，分两阶段：

1. **Reduce-scatter**：经过 `N-1` 轮，每轮向右发送一个 chunk、从左接收一个 chunk 并累加；结束后每个 rank 持有一个完成规约的 chunk。
2. **All-gather**：再经过 `N-1` 轮转发已规约 chunk；结束后每个 rank 拥有完整结果。

每个 rank 总发送量约为：

`2 × (N-1)/N × message_size`

大消息下它接近链路带宽最优，但需要 `2(N-1)` 个通信 step，小消息时 latency 随 rank 数增长，tree 算法通常更合适。实际 NCCL 还会切 channel、选择 LL/LL128/Simple protocol，并根据拓扑使用分层算法。

## 17. ZeRO-1/2/3 分别切分什么？

设 data-parallel size 为 `D`，每参数采用：参数 2 B、梯度 2 B、FP32 master weight 4 B、Adam `m/v` 8 B。

| 策略 | 每卡参数 | 每卡梯度 | 每卡 master + optimizer | 近似每参数字节 |
|---|---:|---:|---:|---:|
| DDP | 2 | 2 | 12 | 16 |
| ZeRO-1 | 2 | 2 | `12/D` | `4 + 12/D` |
| ZeRO-2 | 2 | `2/D` | `12/D` | `2 + 14/D` |
| ZeRO-3 | `2/D` | `2/D` | `12/D` | `16/D` |

具体实现中 master weight 属于 parameter shard 还是 optimizer shard、梯度 dtype、padding/alignment 都可能改变数字。Activation 并未被 ZeRO 自动解决。

Stage 越高，显存越省，但参数需要更频繁 all-gather，通信调度、prefetch 和 checkpoint 复杂度也更高。

## 18. Reduce-Scatter + All-Gather 为何等价于 All-Reduce？

All-reduce 的语义是让每个 rank 得到所有 rank 输入的规约结果。它可拆为：

1. reduce-scatter：按 chunk 完成规约，并把不同结果 shard 留在不同 rank；
2. all-gather：收集所有已规约 shard，拼成完整结果。

在 ring 实现中，all-reduce 本身通常就是这两个阶段。ZeRO/FSDP 的关键是：训练流程并不总需要第二阶段立即发生。例如梯度 reduce-scatter 后，每个 rank 只更新自己负责的 optimizer shard；参数可在真正执行对应模块 forward 前再按需 all-gather，从而避免长期保存完整副本。

通信量等价不代表性能必然相同；buffer layout、是否支持 overlap、消息切分、拓扑算法和额外拷贝都会影响端到端时间。

## 19. FSDP 在 forward/backward 中如何通信？

以 full shard 为例，每个 FSDP unit 的稳态只保存本 rank 参数 shard：

- **Forward 前**：all-gather 当前 unit 参数，得到可计算的完整参数；计算后 reshard，释放非本地部分。
- **Backward 前**：再次 all-gather 参数，用于计算输入梯度和权重梯度。
- **Backward 后**：对完整梯度执行 reduce-scatter；每个 rank 留下自己的梯度 shard。
- **Optimizer step**：本地更新对应参数/optimizer shard。

Forward/backward prefetch 会提前 all-gather 下一个 unit，以隐藏通信；但同时存活多个完整 unit 会提高峰值显存。`SHARD_GRAD_OP` 可在 forward 后保留完整参数到 backward，以减少一次 all-gather，代价是更多显存。

## 20. Wrapping policy 如何影响性能？

FSDP unit 是 all-gather/reshard 的基本边界。

- **unit 太大**：消息带宽效率高、collective 少，但完整参数驻留峰值大，all-gather 启动晚，难以精细 prefetch。
- **unit 太小**：峰值显存低、可早启动，但出现大量小 collective、Python/hook 开销和 allocator 压力。

Transformer 常以一层或若干层为 unit，并让 tied embedding、共享参数和特殊模块满足正确所有权。设计 policy 时要结合：每层参数量、网络启动延迟、可用显存、执行顺序、checkpoint 粒度和是否存在动态控制流。

最优 policy 应通过峰值显存、通信占比、暴露通信时间和 step time 联合验证，不能只以 unit 数量判断。

## 21. Full shard、shard-grad-op 与 hybrid sharding 如何取舍？

- **Full shard**：参数、梯度、optimizer state 全部分片；最省 model-state 显存，但 forward/backward 都可能需要参数 all-gather。
- **Shard-grad-op**：梯度和 optimizer state 分片，forward 后保留完整参数直到 backward；通信更少、显存更多，接近 ZeRO-2 的取舍。
- **Hybrid sharding**：节点内组成 shard group，节点间复制这些 shard group；参数 all-gather/reduce-scatter 主要限制在节点内高速域，跨节点 replica group 仍需要同步梯度。优势是减少昂贵跨节点分片通信的数据量/参与方式，**不应表述成同步频率天然降低**。

Hybrid 适合节点内快、节点间慢的层次化网络。以 PyTorch `HYBRID_SHARD` 为例，节点内执行 `FULL_SHARD`，节点间复制参数；跨节点梯度同步通常可与节点内 reduce-scatter 组合/分层执行。代价是跨节点复制 model states，显存收益小于全局 full shard，还需正确构造 shard/replica 二维 process groups。选择依据是单卡显存约束、节点内外带宽比、模型大小和 DP 副本需求。

## 22. ZeRO-3/FSDP 何时不值得？

它们降低长期驻留显存，却为每个 unit 引入参数 all-gather、梯度 reduce-scatter、reshard 和 buffer 管理。以下场景收益可能不高：

- 模型本来能轻松放入单卡，activation 才是主瓶颈；
- 网络慢或跨节点层级深，参数通信无法被计算隐藏；
- unit 很小、模型计算强度低，collective latency 占主导；
- 小 global batch 导致每次参数通信只服务很少 token；
- 动态图或共享参数使 prefetch/wrapping 难以优化。

判断应比较“暴露通信时间”而非 collective 总时间，并做 DDP、ZeRO-2、ZeRO-3/FSDP 的相同有效 batch 对照。若 activation 是瓶颈，应优先考虑 checkpoint、SP/CP/TP，而不是盲目提高 ZeRO stage。

## 23. 为什么只有部分 FSDP rank OOM？

常见根因：

1. **输入不均衡**：不同 rank 的实际 token/sequence 长度不同，activation 峰值不同。
2. **参数不均衡**：flatten/wrap 后 shard padding 或超大参数让某 rank 持有更多数据。
3. **通信时序**：forward/backward prefetch 使某 rank 同时驻留多个 unsharded unit。
4. **未释放引用**：日志、loss、hook 或异常路径保留计算图，仅发生在某些 rank。
5. **allocator 碎片**：allocated 相近但 reserved/最大连续块不同。
6. **rank 特殊工作**：rank 0 额外 checkpoint、评测、gather 或数据预处理。

排查时按 rank 记录 `allocated/reserved/peak`，关联当前 module、sample token 数和 collective 序号；关闭 prefetch、限制 batch shape、禁用 rank 0 特殊逻辑做对照。OOM snapshot 比只看 `nvidia-smi` 更有价值。

## 24. 如何设计 FSDP 配置推荐器？

输入至少包括：

- 模型 module tree、每个 module 参数/activation/计算量；
- GPU 显存与算力、节点内外拓扑和实测 collective 曲线；
- micro-batch、序列长度分布、dtype、checkpoint 策略；
- optimizer 状态格式、目标 global batch 和容许的重算比例。

候选空间包括 sharding strategy、wrap 边界、prefetch、reshard-after-forward、limit-all-gathers、mixed precision、activation checkpoint 和 hybrid group。

推荐过程可先用解析模型排除放不下的配置，再对少量候选做短 profile。目标函数不能只有 tokens/s，还应约束峰值显存、step time 方差、checkpoint 成本和故障恢复能力。输出应包含配置、预测瓶颈、置信度和回退方案；框架/驱动版本变化后需重新校准。

## 审校依据

- [PyTorch FSDP ShardingStrategy](https://docs.pytorch.org/docs/stable/fsdp.html)：`FULL_SHARD`、`SHARD_GRAD_OP`、`HYBRID_SHARD` 的参数生命周期与通信语义。
- [PyTorch FSDP2 communication grouping](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)：group 粒度、all-gather/reduce-scatter 和 overlap 边界。
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)：输入不由 DDP 自动切分、初始化同步、bucket/static-graph 配置语义。
