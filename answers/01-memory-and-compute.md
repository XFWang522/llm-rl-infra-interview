# 第 1 章：训练显存与计算量

[返回题目清单](../README.md#1-训练显存与计算量)

## 1. 参数、梯度和 Adam states 占多少显存？

设模型参数量为 `P`，先忽略 activation、临时 buffer、显存碎片和通信 workspace。

典型 BF16 混合精度 AdamW 的每参数显存为：

| 状态 | 每参数字节数 |
|---|---:|
| BF16 参数 | 2 B |
| BF16 或 FP32 梯度 | 2 B 或 4 B |
| FP32 master parameter | 4 B |
| FP32 一阶矩 `m` | 4 B |
| FP32 二阶矩 `v` | 4 B |

因此常见估算是 `16P` 或 `18P` bytes：

- 参数 2 B + 梯度 2 B + master weight 4 B + Adam states 8 B = **16 B/param**。
- 若梯度保留 FP32，则为 **18 B/param**。

框架实现并不完全相同。纯 BF16 optimizer、fused optimizer、是否保留 master weights、梯度 dtype 都会改变结果。面试时应先声明假设，再代数计算。例如 70B 模型按 16 B/param，仅 model states 就约 `1.12 TB`，还没有算 activation。

实际容量还需预留 CUDA context、NCCL buffer、allocator fragmentation、临时算子 workspace 和 checkpoint staging，通常不能把卡上标称显存用满。

## 2. 7B/70B 模型能否用 DDP 训练？

DDP 在每张卡复制完整 model states，因此增加 GPU 数量不会降低单卡 model-state 显存，只提高数据并行吞吐。

判断流程：

1. 用上一题的每参数字节数计算单卡 model states。
2. 估算 activation；它由 micro-batch、序列长度、hidden size、层数和 checkpoint 策略决定。
3. 加上通信 bucket、CUDA/NCCL context、临时 workspace 和安全余量。
4. 与单卡可用显存比较，而不是与集群总显存比较。

示例：7B 模型按 16 B/param，model states 约 112 GB，无法直接放入一张 80 GB GPU；70B 约 1.12 TB，更不可能用纯 DDP。需要 ZeRO/FSDP 分片、TP/PP，或结合 offload。若使用 8-bit optimizer、降低梯度精度或冻结大部分参数，数字会变化，但仍需明确逐项计算。

常见误区是说“8 张 80 GB 一共有 640 GB，所以能放下”。这只对真正分片的策略成立，对 DDP 不成立。

## 3. Activation 显存如何随训练配置变化？

粗略看，Transformer 主干 activation 与以下量近似成正比：

`micro_batch × sequence_length × hidden_size × layers`

还要乘以 dtype 字节数和每层保存张量的常数项。朴素 attention 若显式保存 attention matrix，还包含：

`micro_batch × heads × sequence_length² × layers`

关键点：

- **micro-batch** 决定单次 forward 保存多少样本；gradient accumulation steps 不直接线性增加同时存活的 activation。
- **sequence length** 对 MLP/投影 activation 近似线性，对朴素 attention matrix 为平方。
- **hidden size/layers** 增大通常线性增加保存量，但模型架构、GQA/MoE、并行策略会改变常数。
- **TP/SP/CP** 会分片部分 activation；PP 只让每个 stage 保存自己的层。
- **checkpointing** 用重算换保存量。

精确评估应以框架 memory snapshot 或 profiler 为准，并分别记录 allocated、reserved 和 peak memory。

## 4. Attention 为何平方增长？FlashAttention 改变了什么？

标准 attention 计算 `S = QKᵀ`，长度为 `L` 时，`S` 的形状是 `[batch, heads, L, L]`。若把 logits/probabilities 写入 HBM 并为 backward 保存，中间张量空间是 `O(L²)`。

FlashAttention 把 Q/K/V 分块加载到 SRAM，分块计算 `QKᵀ`，通过 online softmax 累积行最大值、归一化因子和输出，不把完整 `L×L` attention matrix 写回 HBM。因此：

- attention 中间结果的 HBM 存储从 `O(L²)` 降为近似 `O(L)`；
- 数学计算量仍是 `O(L²d)`，它没有消除 attention 的平方 FLOPs；
- 通过减少 HBM IO 和 kernel 往返获得加速。

回答时要区分“计算复杂度”和“显存/IO 复杂度”，这是常见追问。

## 5. Activation checkpointing 保存和重算什么？

正常 autograd 会保存 backward 所需的中间 activation。Checkpointing 只保留选定边界的输入（以及正确重放所需的 RNG 状态等），backward 到该区域时重新执行 forward，恢复中间值后再求梯度。

粒度选择：

- **整层 checkpoint**：省显存最多，实现简单，但重算 FLOPs 多。
- **按子模块 checkpoint**：如只重算 attention/MLP，可针对大 activation，工程更复杂。
- **按若干层分段**：减少边界保存量与重算次数之间折中。

代价不仅是理论上的额外 forward。重算会改变 kernel 调度、通信 overlap 和 pipeline 时序；如果某段包含 collective、随机算子或带副作用逻辑，需要保证重放正确。实际要比较端到端 step time，而不是只看显存下降。

## 6. Selective 与 full recomputation 如何取舍？

Full recomputation 通常以 Transformer layer 为单位全部重算，适用于显存压力极大、计算资源相对充足的场景。

Selective recomputation 只重算“保存成本高、重算成本低”的算子，例如某些 norm、dropout、attention 中间结果，而保留 GEMM 等高计算量结果。判断可用近似指标：

`saved_activation_bytes / recompute_FLOPs`

比值越高，越值得重算。还需考虑：

- FlashAttention 已减少 attention matrix 保存，原先策略可能不再最优。
- TP/SP 会改变每个 rank 的 activation 大小。
- selective 策略会增加 autograd 图和框架维护复杂度。
- 重算区域跨通信算子时，可能重复通信并破坏 overlap。

所以应在目标模型、并行配置和序列长度上 profile，而不是固定套用某个 layer 列表。

## 7. Gradient accumulation 的影响是什么？

若 data-parallel size 为 `D`，micro-batch 为 `m`，accumulation steps 为 `G`，则 global batch size 为：

`global_batch = D × m × G`

它的主要影响：

- 单次只保存一个 micro-batch 的 activation，因此可在显存不变时增大 global batch。
- 梯度 buffer 跨 micro-step 累加，直到 optimizer step 才更新参数。
- DDP 应在前 `G-1` 个 micro-step 使用 `no_sync()`，否则每次 backward 都 all-reduce，失去降低通信频率的收益。
- loss 通常要除以 `G`，或者在其他位置等价缩放，避免梯度被放大。
- global batch 改变优化语义，可能需要调整学习率、warmup 和 token-based scheduler。

累积能降低通信频率和 pipeline bubble，但 `G` 太大时参数更新变少、收敛可能变化，且一个 step 的故障重算成本增加。

## 8. BF16、FP16、FP32、FP8 的差异是什么？

| 格式 | 指数位 | 尾数位 | 主要特点 |
|---|---:|---:|---|
| FP32 | 8 | 23 | 范围和精度都高，成本大 |
| FP16 | 5 | 10 | 精度较好但动态范围小 |
| BF16 | 8 | 7 | 与 FP32 相近的动态范围，精度较低 |
| FP8 E4M3 | 4 | 3 | 精度相对高、范围小，常用于 forward |
| FP8 E5M2 | 5 | 2 | 范围更大，精度更低 |

FP16 的小指数范围容易让小梯度下溢，因此常用 dynamic loss scaling：放大 loss 后 backward，再在 optimizer step 前缩回并检查 Inf/NaN。BF16 与 FP32 指数位相同，通常不需要 loss scaling，但仍可能因算子或异常数据出现 NaN。

FP8 训练不能只把 dtype 改成 FP8。通常需要 per-tensor/per-channel scaling、amax history、合适的累积精度，并选择哪些 GEMM 使用 E4M3/E5M2；norm、reduction、optimizer 等敏感计算仍保留更高精度。

## 9. 训练出现 NaN/Inf 如何排查？

建议先找到“第一个坏 step、第一层坏 tensor”，而不是盲目降低学习率：

1. 固定 seed 和数据顺序，确认能否复现；记录 batch/sample ID。
2. 在 loss、gradient norm、参数 norm、AMP scale 上设置 finite check。
3. 用 forward/backward hook 或二分层范围，定位首次出现非有限值的算子。
4. 单卡、关闭 fused kernel/FlashAttention/`torch.compile` 对比，区分模型/数据与 kernel 问题。
5. 使用相同 batch 比较 FP32/BF16/FP16，检查 loss scaling、softmax、norm、除零和极端 logits。
6. 多卡独有时检查各 rank 输入、collective shape/order、通信错误和 silent hardware error。

数据异常通常与特定 batch 强相关；数值问题常可在高精度或更保守 kernel 下消失；通信/硬件问题更可能只在多卡、特定节点或特定拓扑出现。修复后要加入最小复现和数值回归测试。

## 10. 如何估算训练 FLOPs？

对 dense decoder-only Transformer，常用训练近似：

`training FLOPs ≈ 6 × P × T`

其中 `P` 是参数量，`T` 是训练 token 数。直觉是每个 token 的 forward 大约需要 `2P` FLOPs（每个权重参与一次乘加，乘和加计 2 FLOPs），backward 对输入梯度和权重梯度大约是 forward 的两倍，因此合计约 `6P`。

它会遗漏或弱化：

- attention 的 `L²` 项，长序列时不可忽略；
- embedding、loss、norm、激活函数等非参数主导算子；
- MoE 中总参数和每 token 激活参数不同；
- activation recomputation 带来的额外 forward；
- padding、无效 token 和 pipeline bubble；
- optimizer、通信和数据处理，它们影响时间但不一定计入模型 FLOPs。

容量规划应根据具体架构公式修正，再用 profiler 验证。

## 11. MFU、HFU 与 GPU utilization 有何不同？

- **GPU utilization**：采样窗口内 GPU 是否在执行 kernel。小而低效的 kernel 也能让它接近 100%。
- **MFU（Model FLOPs Utilization）**：模型有用 FLOPs / GPU 理论峰值 FLOPs / 时间。通常不把 checkpoint 重算等额外工作算作“有用模型 FLOPs”。
- **HFU（Hardware FLOPs Utilization）**：实际执行 FLOPs / 理论峰值 / 时间；重算也可能计入，因此采用 checkpointing 时 HFU 可高于 MFU。

GPU utilization 高而 MFU 低的常见原因：kernel shape 太小、memory-bound、低 Tensor Core 利用率、大量 elementwise kernel、通信 kernel 占满时间、dtype 未走目标 Tensor Core、pipeline stage 不均衡等。

因此性能判断至少同时看 step time/tokens/s、MFU、kernel timeline、Tensor Core 指标、HBM 带宽和 collective 时间。

## 12. 如何做训练容量与成本估算？

建议分四步：

1. **工作量**：根据模型结构和 token 数估算总训练 FLOPs，并加入 attention、MoE、重算和无效 token 修正。
2. **单卡有效吞吐**：不要直接使用理论峰值；用目标 dtype 峰值乘预期 MFU，最好来自相似模型的实测。
3. **集群时间**：`time = total_FLOPs / (GPU_count × effective_FLOPs_per_GPU)`，再加入 checkpoint、评测、故障恢复和维护窗口。
4. **成本与风险**：GPU-hour × 单价，加存储、网络、CPU 和工程余量；做乐观/基准/悲观三档。

还应反推：

- 每卡 micro-batch 和并行方案是否满足显存；
- global batch 和优化配置是否合理；
- 数据源能否持续提供目标 tokens/s；
- checkpoint 带宽和恢复时间是否符合有效训练时间目标；
- 在给定硬件 MTBF 下，千卡作业的故障损耗是多少。

一个可信估算必须同时满足 **算得完、放得下、喂得饱、坏了能恢复**。
