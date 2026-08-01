# 第 6 章：CUDA、算子与性能分析

[返回题目清单](../README.md#6-cuda算子与性能分析)

## 57. Thread、Block、Warp、SM 与 Occupancy

Kernel 启动 grid，grid 含 blocks，block 含 threads。硬件把 threads 每 32 个组成 warp；一个 block 只能驻留在一个 SM，多个 blocks/warps 可同时驻留。SM 的 warp scheduler 在 ready warps 间切换以隐藏访存延迟。

Occupancy 是活跃 warps 与 SM 理论最大 warps 的比值，受 threads/block、register/thread、shared memory/block 限制。高 occupancy 能提供更多延迟隐藏机会，但不是目标本身：增加 occupancy 可能迫使寄存器 spill 到 local memory，或降低每线程 tile，反而变慢。应结合 achieved occupancy、stall reason、带宽和指令吞吐判断。

## 58. Coalescing、Shared Memory 与 Bank Conflict

同一 warp 的 global-memory 访问若落在少量连续、对齐 cache lines，可合并成较少 memory transactions；跨 stride 或不对齐会放大传输。

Shared memory 是片上、显式管理的低延迟存储，常用于 block 内复用 tile。它分 banks；同一 warp 多线程访问同一 bank 的不同地址会串行形成 bank conflict，访问同一地址的 broadcast 是例外。常用 padding、改变 layout 或 vectorized access 消除冲突。

优化必须看实际 transaction、L2 hit 和 bank-conflict 指标，而非只检查源码索引“看起来连续”。

## 59. 如何用 Roofline 判断瓶颈？

算术强度 `AI = FLOPs / bytes_from_memory`。Roofline 给出性能上限：

`performance ≤ min(peak_compute, AI × memory_bandwidth)`。

点落在斜线段说明 memory-bound，应减少 HBM IO、fusion、提高复用；落在水平段说明 compute-bound，应提高 Tensor Core 利用率、并行性和指令效率。

需要使用实际层级的 bytes/FLOPs：L2/shared-memory roofline 与 HBM roofline不同。低于两条 roof 很远则可能受 launch latency、依赖、occupancy、分支或 shape 影响，不能简单归为 compute/memory bound。

## 60. GEMM Tiling 与 Double Buffering

GEMM 把 M/N/K 维切成 tiles。Thread block 将 A/B tile 从 HBM/L2 搬到 shared memory，warp 再搬到 registers/Tensor Core fragments，复用数据完成多次 MMA。

Double buffering 使用两套 buffer：计算 tile `i` 时异步预取 `i+1`，覆盖访存延迟。Tensor Core 要求合适 dtype、对齐和维度倍数；epilogue fusion 把 bias、activation、scaling 等写回前操作合并，减少 HBM round trip。

Tile 太大增加复用但消耗 registers/shared memory、降低并发；太小则数据复用差、指令开销高。选择依赖 shape 和 GPU 架构。

## 61. FlashAttention 与 Online Softmax

FlashAttention 分块读取 Q/K/V，在 SRAM 内算局部 score。对每个 query row 维护历史最大值 `m`、归一化和 `l`、未归一化输出 `o`；新 block 到来时用新的最大值重新缩放旧累积量，再合并当前 block，因此结果与全量 stable softmax 等价。

它避免把完整 attention matrix 写入 HBM，把 attention 中间存储降为近似线性，并通过 tiling 提高 IO efficiency。计算 FLOPs 仍为平方；causal mask、dropout、GQA 和 backward 需要专门 kernel 处理。

## 62. 常见 Fusion 省了什么？

- fused RMSNorm/LayerNorm：一次读取完成 reduction、归一化和 scale，减少中间写回与 launch。
- fused optimizer：批量处理许多参数，合并 unscale、finite check、update，降低大量小 kernel 和 CPU launch。
- fused cross entropy：避免物化完整 `[tokens,vocab]` 的额外概率/梯度中间量，可融合 max、sum-exp、target gather。

Fusion 的代价是代码路径和 shape/dtype 限制、编译时间、调试难度及数值差异。验证要包含端到端时间、峰值显存和 loss/gradient 对齐。

## 63. CUDA Graph 的收益和限制

CUDA Graph 捕获一段固定 GPU 工作 DAG，后续一次 replay 代替大量 CPU kernel launch，适合重复、launch-bound 的训练 step。

限制：捕获时的 kernel 拓扑、地址和大部分 shape 需稳定；allocator、动态控制流、CPU-GPU 同步、某些 collective 或 host callback 需 graph-safe。通常预分配静态 input/output buffer，为不同 shape/batch 建多个 graph bucket。

大 GEMM 主导时收益有限；小模型、大量融合前小 kernel 或 CPU launch 瓶颈时收益明显。

## 64. Nsight Systems 中 GPU 空洞怎么归因？

从空洞向上关联 CPU thread、CUDA API、stream event 和 dataloader：

- CPU 没有 launch：Python/数据加载/GC/锁/日志瓶颈；
- CPU 在同步 API：`.item()`、synchronize、blocking copy 或 allocator sync；
- 某 stream 等 event：错误依赖或 collective 未完成；
- 所有 ranks 同时停：checkpoint/evaluation；单 rank 先停则可能造成其他 rank collective 等待。

配合 NVTX 标注 training phases、PyTorch profiler 和系统 CPU/IO 指标。先定位第一个产生空洞的 rank和线程，不要把其他 rank 的 NCCL 等待当根因。

## 65. Tensor Core 利用率低如何排查？

检查 dtype 是否支持目标 Tensor Core，M/N/K 是否满足对齐和最小尺寸，layout/stride 是否触发额外 transpose，library 是否选择了合适 kernel。小 micro-batch、过高 TP 会把 GEMM 切得太小。

Nsight Compute 关注 Tensor Core pipe、achieved FLOPs、memory throughput、occupancy、stall reason；再做固定 shape microbenchmark，与 cuBLAS/CUTLASS 理论基线对比。若 GEMM 本身快但端到端低，问题可能在 elementwise、通信或 launch，不应继续调 GEMM。

## 66. 如何建立训练性能回归 CI？

选择代表性小/中模型、序列长度、并行配置和关键 kernel microbench。固定硬件池、功耗/时钟、镜像、数据与 warmup，重复多轮报告中位数和尾部，避免共享集群噪声。

门禁指标包括 tokens/s、step time 分解、MFU、峰值显存、collective exposed time 和数值对齐。阈值应来自历史方差，而非任意百分比。

回归后自动保存 profiler trace、commit/config/driver 信息，并支持二分。Microbenchmark 通过不能替代端到端；端到端异常也要能下钻到算子、通信或输入 pipeline。
