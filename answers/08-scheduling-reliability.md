# 第 8 章：调度、容错与可观测性

[返回题目清单](../README.md#8-调度容错与可观测性)

## 77. 为什么需要 Gang Scheduling？

同步分布式训练只有全部 workers 同时就绪才有进展；零散启动的 workers 占资源却等待 peers，可能形成死锁式碎片。Gang scheduling 对整组资源做原子分配/启动，保证 world size 和拓扑。

代价是大作业更难找到连续资源、排队时间长，并可能阻塞小作业。调度器需结合队列公平、backfill、reservation 和 topology-aware compaction。原子语义还应覆盖网络/NIC/本地盘等配套资源，而不只是 GPU 数量。

## 78. Placement 要考虑哪些约束？

硬约束包括 GPU 型号/显存、节点健康、world size、每节点卡数、网络 rail 和本地存储；软目标包括 NVLink 连续性、机架/故障域、碎片、公平和数据 locality。

TP/EP/CP 等高频通信组应放高速域，副本/DP 可跨故障域。bin packing 提高利用率但可能集中故障和热量；spread 提高隔离却增加通信。调度器应接收并行拓扑描述，而非把所有 GPU 当同质整数。

## 79. 抢占与最优 Checkpoint Interval

抢占前最好给作业 grace period 完成一致 checkpoint；无预告时依赖最近耐久副本。恢复还需重建 gang、相同或可转换的并行布局及数据 cursor。

经典近似中，checkpoint 成本为 `C`、平均故障间隔为 `MTBF`，最优间隔量级约 `sqrt(2×C×MTBF)`，更精确模型会加入恢复时间和 checkpoint 自身故障概率。抢占率、上传拥塞和增量 checkpoint 会改变参数。平台应以有效训练时间而非 GPU allocation time 衡量收益。

## 80. 弹性扩缩容如何保持语义？

在 optimizer-step 边界暂停，保存 global state，重建 process groups 并 reshard model/optimizer。DP size 变化时若 micro-batch 与 accumulation 不调，global batch 会变化；应保持目标 tokens/update，并决定是否按规则调整学习率。

Sampler 根据 consumed global samples/tokens 重新映射，避免重读或跳过。TP/PP 变化还需 checkpoint layout conversion。频繁弹性会付出 group initialization、reshard、cache warmup 和编译成本，因此应有最小驻留时间和收益阈值。

## 81. 如何降低千卡作业的故障损失？

规模扩大后，作业 MTBF 约随组件数下降。策略包括：入场/周期健康检查、坏节点隔离、快速故障传播、低开销增量/异步 checkpoint、自动重调度和局部恢复。

把总损失拆为检测时间、清理/排队、重启、load、重放和重新 warmup；逐项优化。若框架支持 elastic/local recovery，可只替换故障 worker，但同步状态、collective group 和 optimizer shard 必须重新一致。最终指标是 goodput：产生有效训练 token 的 GPU 时间占比。

## 82. GPU/NIC/节点健康检查怎么做？

入场检查覆盖 ECC/Xid、HBM、温度/功耗、PCIe/NVLink、GPU-GPU P2P、GPU-NIC RDMA、本地盘和时钟；使用小 GEMM、memtest、NCCL pair/all-reduce 验证功能和性能。

运行时根据硬错误和相对性能退化打健康分。节点隔离需有原因、证据和自动修复/人工复验流程，避免 flapping 节点反复回池。仅跑 `nvidia-smi` 不足以发现慢链路、silent corruption 或特定 GPU pair 问题。

## 83. Loss 正常但 Tokens/s 缓慢下降怎么查？

同时观察：step phase 时间、有效/padding tokens、GPU clock/power/temperature、HBM/SM、CPU/内存/GC、dataloader queue、存储延迟、collective 长尾和网络 counters。

常见慢性原因有内存泄漏/碎片、日志对象保留计算图、dataset shard 变慢、热降频、网络错误重传、checkpoint backlog 和动态 shape 漂移。按 rank 找最早变慢者，并把时间线与版本、节点和数据 ID 对齐；重启能恢复只说明状态累积，不等于找到根因。

## 84. Watchdog 如何区分 Slow、Hang、Crash 与 SDC？

每 rank 周期上报 step/micro-step、最后 collective sequence、CUDA event 和心跳。进程退出由 launcher 立即传播；心跳活着但进度不变是 hang；持续有进度但超过历史分位是 slow。

Silent data corruption 需额外信号：finite/norm checks、checksum、冗余小计算、loss/gradient 异常和硬件错误。Watchdog 超时时先触发现场转储（stack、flight recorder、NCCL/系统日志），再终止，避免只得到最终超时。策略要防 profiler/checkpoint 等合法长操作误报。

## 85. Experiment Lineage 如何设计？

每次运行生成不可变 run ID，关联 git commit/dirty diff、容器 digest、启动命令、完整配置、依赖/驱动、数据集 manifest、集群拓扑、随机种子、checkpoint 和评测结果。

Checkpoint manifest 记录父 checkpoint 与 consumed tokens；产物使用内容 hash，元数据服务只保存引用和状态转换。敏感配置应脱敏。目标是从任一模型反查“谁、用什么代码和数据、在哪些硬件上、经过哪些恢复”产生，并能自动构造复现实验。

## 86. 多租户平台如何兼顾公平与效率？

Quota 限制团队长期占用，优先级表达业务紧急度，fair-share 根据历史使用动态调整；backfill 用短作业填补大 gang 等待产生的空洞。成本核算应区分 allocated GPU-hours 与有效 goodput，并计入稀缺拓扑溢价。

隔离包括容器/权限、网络与存储带宽、MIG/整卡策略、故障域和日志数据。抢占必须配套 checkpoint SLA。调度决策应可解释、可审计，避免高优先级永久饿死其他队列；定期评估利用率、排队时间、SLO 和碎片的 Pareto 取舍。
