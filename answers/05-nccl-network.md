# 第 5 章：NCCL、网络与拓扑

[返回题目清单](../README.md#5-nccl网络与拓扑)

## 45. 常见 Collective 的语义与通信量

- **All-reduce**：规约所有输入并把完整结果发给所有 rank；用于 DDP 梯度。
- **Reduce-scatter**：规约后每 rank 只保留一个 shard；用于 FSDP/ZeRO 梯度。
- **All-gather**：收集各 rank shard，所有 rank 得到完整 tensor；用于 FSDP 参数 materialization。
- **All-to-all**：每个 rank 向每个目标发送不同 shard；用于 MoE token dispatch、layout transpose。

Ring all-reduce 每 rank 字节量约 `2(N-1)/N × M`；reduce-scatter 或 all-gather 各约 `(N-1)/N × M`。All-to-all 总字节可能近似 `M`，但存在多 peer、小消息和不均衡问题。通信量相同不代表时间相同，latency、拓扑、并发和拷贝路径同样重要。

## 46. Ring 与 Tree 如何选择？

用 `T≈α×steps + β×bytes` 理解：`α` 是每步启动延迟，`β` 是单位字节时间。

Ring 有 `O(N)` steps，但大消息时每条链路持续传输、带宽利用率高。Tree 深度 `O(logN)`，小消息下延迟更低；大消息可能受根部/链路并发和拓扑映射限制。分层集群还可节点内 reduce、节点间 tree/ring、节点内 broadcast。

NCCL 会根据消息大小、GPU 拓扑和可用网络选择算法/protocol。面试中不应绝对回答“ring 总是更快”，而要说明消息大小与 `α-β` 权衡。

## 47. PCIe、NVLink、NVSwitch、IB/RoCE 的角色

PCIe 连接 CPU、GPU、NIC，通用但 GPU-GPU 带宽通常低于 NVLink。NVLink 提供 GPU 间高速互联；NVSwitch 让节点内多 GPU 形成高双向带宽交换域。跨节点使用 InfiniBand 或 RoCE RDMA，通过 NIC 连接交换网络。

理想路径是节点内 collective 走 NVLink/NVSwitch，跨节点 GPU buffer 经 GPUDirect RDMA 直达 NIC。若 GPU 与 NIC 跨 NUMA/PCIe root complex，流量可能绕行 CPU interconnect，性能下降。并行维度 placement 必须匹配这套层次拓扑。

## 48. GPUDirect RDMA 绕过什么？

传统路径可能是 GPU → host staging buffer → kernel/network stack → NIC，接收端反向复制。GPUDirect RDMA 让 NIC DMA 直接读写 GPU memory，减少 CPU 参与和 host copy。

必要条件包括 GPU/NIC/驱动支持、peer-memory/DMABUF 路径、正确 PCIe 拓扑与内存注册。Pinned host memory 主要用于需要 host staging 时保证 DMA；BAR 映射让设备访问 GPU 地址窗口；IOMMU/ACS 配置可能改变 P2P 路径。启用环境变量不等于真正走 GDR，应通过 NCCL 日志、拓扑工具和带宽测试验证。

## 49. NCCL 如何选择拓扑、Channel 和 Protocol？

初始化时 NCCL 枚举 GPU、NIC、PCIe/NVLink 路径及带宽，构建 ring/tree 等通信图。一个 collective 可切为多个 channels 并行传不同 chunk，以利用多链路和更多 GPU blocks。

Protocol 常见 LL、LL128、Simple：低延迟协议适合小消息，Simple 更偏大消息带宽。Channel 太少用不满链路；太多会增加同步、buffer 和 SM 占用。算法选择还受拓扑、消息大小、rank 数、NIC rail 和环境配置影响。

排障时保存 NCCL topology/debug 日志并与正常基线比较，但不要长期在生产开启过量 debug 造成扰动。

## 50. `nccl-tests` 正常但训练通信慢怎么办？

`nccl-tests` 通常使用规则大 buffer、纯通信、固定 group；训练里消息大小、collective 顺序、计算竞争和 rank skew 完全不同。

检查：真实消息大小分布和次数；collective 是否等最慢 rank；通信 kernel 与 GEMM 是否争用 SM；bucket ready 是否过晚；多 process groups 是否互相串行；tensor 是否发生额外 contiguous/copy；实际 rank placement 是否与测试相同。

用 profiler 区分 collective duration 与 exposed communication。必要时从训练 trace 提取真实 collective replay，而不是继续重复理想 microbenchmark。

## 51. Collective Hang 的常见根因

最常见是各 rank 调用序列不一致：某 rank 少调用一次、进入不同 process group、tensor count/shape 不匹配，或异常后其他 rank 仍等待。其次是进程崩溃、CUDA asynchronous error、NIC/link 故障和超时配置。

定位方法：为每次 collective 记录 group、sequence number、op、shape、dtype 和调用栈；watchdog 汇报各 rank 最后进度；启用分布式 debug/flight recorder；检查最早异常 rank，而不是只看最终超时 rank。

修复要让控制流跨 rank 一致，在 collective 前验证元数据，并让进程失败能迅速传播，避免其余 ranks 无限等待。

## 52. 只有跨节点慢，如何检查网络？

先确认 GPU-NIC locality：NUMA、PCIe root、NIC rail 和 rank binding。再检查 RDMA device/GID 选择、RoCE MTU、PFC/ECN、丢包/重传、端口错误计数和交换机拥塞。

分层对照：单 GPU 单 NIC、节点内、两节点、逐步扩大节点；分别跑 host RDMA、GPU RDMA 和 NCCL。若 host 正常而 GPU 慢，聚焦 GDR/PCIe；若特定节点对慢，检查链路/rail；若规模扩大才慢，检查 oversubscription、拥塞控制和 collective topology。

设置 NIC/rail 环境变量前先证明自动选择错误，避免用配置掩盖坏链路。

## 53. 如何 Overlap 通信与计算？

条件是通信尽早有数据、在独立 stream 异步发起，且后续计算不依赖结果。DDP bucket、FSDP prefetch、TP 的 reduce-scatter 与 GEMM fusion 都利用此原则。

API 返回异步 handle 不代表物理重叠：默认 stream event、数据依赖、CPU launch 顺序、NCCL group、SM/NVLink 资源竞争都可能使其串行。应在 timeline 上测 `exposed_comm = step_time - compute_only_critical_path`。

通信占用过多 SM 时，即使时间轴重叠也会拖慢 GEMM；可调整 channel/CTA、chunk、优先级或将通信映射到专用能力。

## 54. NCCL Kernel 抢占 SM 怎么办？

NCCL 需要 GPU blocks 搬运/规约数据。Channels/CTAs 过多会占据 SM、寄存器和调度槽，与 Tensor Core kernel 竞争。

先用 Nsight 验证重叠区 GEMM 是否降速，再尝试减少 channels/CTA、改变 protocol/chunk、错开关键 GEMM、使用低 SM 通信能力或更好的硬件 offload。不能只看 collective 自身结束得更早；目标是 step time 最小。

同时确认网络是否真能受益于更多 channel。若瓶颈已在 NIC/交换机，增加 GPU 通信 blocks 只会加重竞争。

## 55. Topology-Aware Placement 怎么设计？

把硬件抽象为带权图：GPU、NIC、NUMA、节点、机架为顶点/层级，边权是实测带宽和延迟。把并行维度的通信频率/字节量作为需求图，再做映射。

通常 TP/CP/EP 放最快域，PP 跨较慢链路，DP 可跨更大故障域；但 MoE all-to-all 和长上下文 CP 可能改变优先级。调度器需 gang allocate 连续拓扑资源，并避免把作业放到降级链路。

目标还要考虑碎片、公平性和排队时间；等待完美拓扑过久可能不如接受次优 placement。应给出预测性能与最大等待阈值。

## 56. 如何持续监控 GPU 网络并关联训练性能？

采集 NIC/交换机端口吞吐、丢包、重传、ECN/PFC、错误计数，GPU-NIC PCIe/NVLink counters，以及训练侧每个 collective 的 group、bytes、duration、rank skew。

用 job/rank → GPU → NIC → switch port 的资产映射，把 step-time 异常与链路时间序列关联。建立小规模 canary collective 和节点健康分，发现退化后阻止新调度、迁移或隔离节点。

告警应关注相对基线和同组 rank 长尾，而不只是绝对带宽。保留拓扑、镜像、驱动/NCCL 版本，才能区分软件回归与硬件退化。
