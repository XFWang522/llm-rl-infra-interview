# 第 7 章：数据 Pipeline 与 Checkpoint

[返回题目清单](../README.md#7-数据-pipeline-与-checkpoint)

## 67. 数据从对象存储到 GPU 的路径

典型路径是对象存储/分布式文件系统 → 节点本地 SSD cache → host page cache/用户态 buffer → DataLoader worker 解码与组 batch → pinned memory → PCIe/NVLink H2D → GPU input buffer。

每层都应流水化：后台下载下一 shard，本地 cache 按训练顺序预取；workers 并行解码；主进程维护 bounded prefetch queue；独立 CUDA stream 异步 H2D，并用 event 让计算等待。瓶颈判断要同时看远端吞吐、本地盘、CPU decode、queue depth、H2D 和 GPU idle。无限 prefetch 会挤占内存并放大故障重读，不是越大越好。

## 68. DataLoader 参数如何影响性能？

`num_workers` 增加并行读取/解码，但过多会争用 CPU、内存带宽、文件句柄和存储。`prefetch_factor` 决定每 worker 预先准备的 batch 数，过大增加 host memory 与陈旧数据。`pin_memory` 让 host buffer 可用于异步 DMA，通常加快 H2D，但 pinned memory 太多会伤害系统分页。

还要考虑 persistent workers、CPU affinity、NUMA locality、对象反序列化和 batch collate。调参时观察 GPU data-wait、worker queue、CPU/IO，而不是只测 DataLoader 单独吞吐。

## 69. GPU 周期性空闲如何归因？

给 dataloader fetch、H2D、forward/backward、checkpoint、evaluation 和日志打统一 trace。固定周期通常提示：每 N step checkpoint/eval、Python GC、日志 flush、dataset shard 边界或 cache miss。

若所有 ranks 同停，检查全局任务；若一个 rank 先停、其他 rank 卡在 collective，则根因常在该 rank 数据或 IO。关闭各周期功能做 A/B；记录 queue depth 与 sample/shard ID。不要把其他 ranks 显示的 NCCL 等待误判为通信问题。

## 70. Sequence Packing 如何正确实现？

把多个短样本拼入固定 token budget，减少 padding。关键是阻止不同样本互相 attention：使用 block-diagonal mask、varlen attention 的 cumulative sequence lengths，或 reset position IDs。loss mask 应只覆盖目标 token，并正确处理 BOS/EOS、prompt mask 和跨样本边界。

Packing 会改变 batch 内序列数与 token 数。应按有效 token 归一化 loss，而非固定按样本数；分布式 ranks 的 token load 也要平衡。测试要比较 packed/unpacked 的 token-level loss 和梯度。

## 71. 如何全局 Shuffle、按 Rank 切分并恢复？

先用确定性 seed 生成 epoch 内 shard permutation，再在 shard 内用可重放的 sample permutation/buffer shuffle。DP ranks 根据 global sample/token index 做无重叠 partition；TP/PP/CP ranks 必须消费同一批样本。

Checkpoint 保存 dataset version、epoch、shard permutation seed、global cursor、sampler state、packing buffer 状态及 worker RNG。恢复到不同 DP size 时，最好以全局 cursor/token ID 重新映射，而非直接恢复旧 rank 本地 offset。通过 sample ID 审计窗口验证不重不漏。

## 72. 严格恢复需要保存哪些状态？

除 model、optimizer 外，还包括 LR scheduler、gradient scaler、global step/consumed tokens、Python/NumPy/CPU/GPU RNG、每个并行 group RNG、dataloader/sampler/packing state、并行配置和框架版本。若在 gradient accumulation 中间恢复，还需梯度 buffer 与 micro-step；多数系统选择只在 optimizer-step 边界 checkpoint。

还要保存数据与代码/镜像身份、loss-scale history、EMA 等算法附加状态。验证不是“能 load”，而是从 checkpoint 续跑后若干 step 与未中断基线的 sample、loss 和参数在约定容差内一致。

## 73. Distributed Checkpoint 如何避免 Rank 0 瓶颈？

每 rank 直接写本地持有的 parameter/optimizer shards，生成小的 manifest 描述 logical tensor 到 storage chunks 的映射；不把完整模型 gather 到 rank 0。写入可分层到本地 SSD，再异步上传共享存储。

一致性采用临时目录/版本 ID：所有 shards 成功并校验后再原子发布 manifest/commit marker。失败 checkpoint 不可见。文件不能碎得过小，应聚合 chunks；也要限制并发避免压垮 metadata server 或对象存储。

## 74. 如何跨并行度恢复 Tensor Shard？

Checkpoint 记录 logical global tensor 的 shape、dtype、轴语义，以及每个 storage shard 对应的 global offsets。加载到新 TP/PP/FSDP 布局时，新 rank 根据目标 shard 计算所需区间，只读取/交换重叠 chunks，再组装本地 tensor。

PP 变化还需 layer-name 到 stage 的重新映射；TP 变化涉及行/列切分；optimizer state 必须与参数使用同一 logical mapping。不能依赖旧 rank 编号或文件名。转换要处理 padding、tied weights、MoE experts 和版本迁移，并做全局 checksum/小模型等价测试。

## 75. 异步 Checkpoint 如何保证一致？

Optimizer-step 边界先冻结一个逻辑快照。可把 GPU shards 异步拷到预分配 pinned CPU buffer，训练继续使用下一版本参数，后台线程/进程负责序列化和写盘。双缓冲避免覆盖尚未写完的数据。

必须限制同时在途快照数和额外 host/GPU 内存；后台失败要上报，不能仍更新“latest”指针。发布前校验所有 rank 的 step/version 一致。若 CPU copy 本身阻塞关键 stream，应分块、限速或使用本地持久化层。

## 76. 千卡作业的分层 Checkpoint 设计

本地 NVMe 带宽高，适合频繁临时 checkpoint 和节点内聚合，但节点故障会丢失；并行文件系统便于共享恢复但 metadata/带宽可能拥塞；对象存储耐久、弹性好但延迟高。

常见设计：每 rank 分片写节点本地 → 每节点聚合/限流 → 后台上传对象存储 → 全局 manifest 原子提交。保留最近的快速本地副本和较稀疏的耐久副本。checkpoint interval 根据故障率、写入成本和重算损失选择，并定期执行真实恢复演练，而不只验证文件存在。
