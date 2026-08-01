# LLM Training Infra 面试问题清单

专注 **大模型训练基础设施**：分布式并行、训练框架、GPU/CUDA、集合通信、显存优化、集群调度、容错与性能工程。

不覆盖模型算法、Prompt/RAG、PPO/GRPO/DPO 数学推导，也不覆盖普通在线推理服务。**RL Training Infra 是核心范围之一**：重点讨论 rollout、训练、reward/verifier、权重同步、异构并行、调度和容错。

## 标签说明

- **[面经高频]**：公开训练 Infra 面经、面试题汇总中反复出现。
- **[岗位强相关]**：训练 Infra 官方 JD 明确要求，可高概率推导。
- **[源码深挖]**：用于区分“会用框架”和“理解实现”。
- **[故障诊断]**：以线上现象为入口，重点考察排查路径。
- **[系统设计]**：没有唯一答案，需要量化约束和权衡。

> 推荐回答结构：先给结论和公式，再画数据流/通信图，最后讲瓶颈、指标、故障案例与取舍。

答案正在逐题补充，进度见 [answers/README.md](answers/README.md)。

## 1. 训练显存与计算量

1. **[面经高频]** 训练一个参数量为 `P` 的 Transformer，参数、梯度、Adam states 分别占多少显存？混合精度下是否保留 FP32 master weights？
2. **[面经高频]** 给定 7B/70B 模型、GPU 数量与显存，判断能否用 DDP 训练，并给出计算过程。
3. **[面经高频]** activation 显存与 batch size、sequence length、hidden size、layer 数量是什么关系？
4. **[面经高频]** 为什么 attention activation 随序列长度呈平方增长？FlashAttention 改变了哪部分显存复杂度？
5. **[岗位强相关]** activation checkpointing 保存什么、重算什么？如何选择 checkpoint 粒度？
6. **[源码深挖]** selective activation recomputation 和 full recomputation 如何取舍？
7. **[面经高频]** gradient accumulation 如何影响显存、global batch size、通信频率和优化器语义？
8. **[面经高频]** BF16、FP16、FP32、FP8 的指数位和尾数位有什么差异？何时需要 loss scaling？
9. **[故障诊断]** 训练突然出现 NaN/Inf，如何区分数据异常、数值溢出、kernel bug 和通信错误？
10. **[面经高频]** 如何估算一个训练 step 的 FLOPs？`6 × parameters × tokens` 近似从何而来，有哪些遗漏？
11. **[面经高频]** MFU、HFU、GPU utilization 分别代表什么？为什么 GPU utilization 99% 仍可能很慢？
12. **[系统设计]** 给定模型、token 预算、集群规模和训练期限，做一次容量与成本估算。

## 2. 数据并行、ZeRO 与 FSDP

13. **[面经高频]** DDP 的 forward/backward/optimizer step 分别发生什么？梯度在何时 all-reduce？
14. **[源码深挖]** PyTorch DDP reducer 如何构造 bucket？参数注册顺序为什么影响 overlap？
15. **[源码深挖]** `find_unused_parameters` 为什么会带来额外开销？static graph 优化了什么？
16. **[面经高频]** ring all-reduce 如何工作？通信量和 world size 的关系是什么？
17. **[面经高频]** ZeRO-1/2/3 分别切分 optimizer、gradient、parameter 的哪些部分？逐项计算显存。
18. **[面经高频]** reduce-scatter + all-gather 与 all-reduce 的等价关系是什么？
19. **[面经高频]** FSDP 在 forward/backward 中何时 all-gather 参数、何时 reduce-scatter 梯度？
20. **[源码深挖]** FSDP wrapping policy 如何影响峰值显存、通信粒度和 prefetch 效果？
21. **[源码深挖]** `FULL_SHARD`、`SHARD_GRAD_OP`、hybrid sharding 的区别与适用拓扑是什么？
22. **[岗位强相关]** ZeRO-3/FSDP 的通信为何可能抵消显存收益？在什么规模下不值得使用？
23. **[故障诊断]** FSDP 某些 rank OOM、另一些 rank 正常，可能有哪些根因？
24. **[系统设计]** 设计一个 FSDP auto-wrap 与并行配置推荐器，需要收集哪些 profile 数据？

## 3. Tensor / Pipeline / Context Parallel

25. **[面经高频]** Megatron-LM 的 column-parallel 和 row-parallel linear 如何切分？各自需要什么 collective？
26. **[面经高频]** MLP 和 self-attention 如何做 Tensor Parallel？为什么某些相邻 collective 可以消掉？
27. **[面经高频]** TP 为什么通常优先放在 NVLink/NVSwitch 域内？跨节点 TP 的代价是什么？
28. **[面经高频]** Pipeline Parallel 的 micro-batch、stage 和 bubble 分别是什么？推导 GPipe bubble ratio。
29. **[面经高频]** 1F1B、interleaved 1F1B、zero-bubble pipeline 的调度差异是什么？
30. **[源码深挖]** virtual pipeline stage 如何减少 bubble？为什么会增加通信和实现复杂度？
31. **[故障诊断]** pipeline stage 负载不均衡时如何定位？embedding、loss、不同 layer FLOPs 怎么处理？
32. **[岗位强相关]** Sequence Parallel 切分什么张量？它与 Tensor Parallel 为什么经常一起使用？
33. **[岗位强相关]** Context Parallel 如何支持超长序列？Ring Attention 的 KV 交换流程是什么？
34. **[源码深挖]** Ulysses / all-to-all context parallel 与 ring-based 方案如何取舍？
35. **[面经高频]** DP × TP × PP × CP 的 world size 约束是什么？rank group 如何构造？
36. **[系统设计]** 给 70B 模型和 128 台 8×H100 集群设计 4D 并行方案，并解释拓扑映射。

## 4. MoE 训练 Infra

37. **[岗位强相关]** Expert Parallel 的 token dispatch 和 combine 为什么需要 all-to-all？
38. **[面经高频]** MoE 中 capacity factor、token dropping、load-balancing loss 分别影响什么？
39. **[故障诊断]** 某几个 expert 持续过载，如何从路由分布、通信和 kernel 三层定位？
40. **[源码深挖]** dropless MoE 如何处理不等长 token？padding、sorting 和 grouped GEMM 如何取舍？
41. **[岗位强相关]** TP 与 EP 为什么存在冲突或互补？MoE 层和 dense 层是否应使用相同并行布局？
42. **[源码深挖]** DeepEP / fused all-to-all 的优化目标是什么？低延迟与高吞吐模式有什么不同？
43. **[故障诊断]** MoE 训练 tokens/s 周期性抖动，如何判断是路由长尾、网络拥塞还是数据 shape 变化？
44. **[系统设计]** 设计一个 MoE placement 策略，使 expert、节点和故障域之间取得平衡。

## 5. NCCL、网络与拓扑

45. **[面经高频]** all-reduce、all-gather、reduce-scatter、all-to-all 的语义、通信量和典型使用场景是什么？
46. **[面经高频]** ring 与 tree collective 各适合大消息还是小消息？latency/bandwidth 模型如何估算？
47. **[岗位强相关]** PCIe、NVLink、NVSwitch、InfiniBand/RoCE 在训练数据路径中分别扮演什么角色？
48. **[面经高频]** GPUDirect RDMA 绕过了哪些拷贝？为什么需要 pinned memory、BAR 和 IOMMU 配合？
49. **[源码深挖]** NCCL 如何做 topology discovery、channel 划分和 algorithm/protocol 选择？
50. **[故障诊断]** `nccl-tests` 正常但真实训练 all-reduce 慢，下一步排查什么？
51. **[故障诊断]** collective hang 常见原因有哪些：rank 调用顺序、shape 不一致、进程退出、网络故障？
52. **[故障诊断]** 只有跨节点性能差，如何检查 NIC 亲和性、NUMA、GID、MTU、PFC/ECN 和 rail 配置？
53. **[岗位强相关]** 如何 overlap 通信与计算？为什么“异步 collective”不等于一定重叠？
54. **[源码深挖]** NCCL kernel 占用过多 SM 导致计算变慢，如何控制 channel/CTA 或使用低 SM 通信？
55. **[系统设计]** 设计训练作业的 topology-aware placement，避免 TP/EP 通信跨越慢链路。
56. **[系统设计]** 如何持续监控大规模 GPU 网络并把链路退化关联到训练 step time？

## 6. CUDA、算子与性能分析

57. **[面经高频]** CUDA thread/block/grid、warp、SM、occupancy 的关系是什么？occupancy 越高越好吗？
58. **[面经高频]** coalesced memory access、shared memory、bank conflict 分别是什么？
59. **[面经高频]** 一个 kernel 是 compute-bound 还是 memory-bound，如何用 Roofline 判断？
60. **[源码深挖]** GEMM 的 tiling、double buffering、Tensor Core 和 epilogue fusion 如何提高性能？
61. **[面经高频]** FlashAttention 为什么是 IO-aware？online softmax 如何保证数值稳定？
62. **[源码深挖]** fused LayerNorm/RMSNorm、fused optimizer、fused cross entropy 分别省了什么？
63. **[岗位强相关]** CUDA Graph 能减少什么开销？动态 shape、allocator 和 collective 带来哪些限制？
64. **[故障诊断]** 用 Nsight Systems 发现 GPU 中间有大量空洞，如何向 CPU launch、同步点、数据加载方向归因？
65. **[故障诊断]** Nsight Compute 显示 Tensor Core 利用率低，可能是 shape、dtype、layout 还是 kernel selection 问题？
66. **[系统设计]** 建立训练性能回归 CI：基准模型、噪声控制、指标阈值和归因流程怎么设计？

## 7. 数据 Pipeline 与 Checkpoint

67. **[岗位强相关]** 训练数据从对象存储到 GPU 的完整路径是什么？每一层如何 prefetch 和 cache？
68. **[面经高频]** DataLoader worker、pinned memory、prefetch factor 对吞吐和内存有什么影响？
69. **[故障诊断]** GPU 每隔固定时间空闲，如何判断是 dataloader、GC、checkpoint 还是日志系统？
70. **[岗位强相关]** sequence packing 如何减少 padding？如何正确处理 attention mask、position id 和 loss mask？
71. **[岗位强相关]** 数据如何做全局 shuffle、按 rank 切分和断点续训，保证不重不漏？
72. **[面经高频]** 一个可严格恢复的 checkpoint 需要保存哪些状态？为什么仅保存 model/optimizer 不够？
73. **[岗位强相关]** distributed checkpoint 如何避免 rank 0 聚合瓶颈？分片格式如何跨并行度恢复？
74. **[源码深挖]** Megatron distributed checkpoint / PyTorch DCP 如何把 logical tensor 映射到不同 shard？
75. **[系统设计]** 设计异步 checkpoint：如何保证一致性、限制额外显存，并处理后台写入失败？
76. **[系统设计]** 千卡作业每 30 分钟 checkpoint 一次，如何选择本地盘、并行文件系统和对象存储层级？

## 8. 调度、容错与可观测性

77. **[岗位强相关]** gang scheduling 为什么适合分布式训练？它带来什么碎片和排队问题？
78. **[岗位强相关]** bin packing、拓扑亲和、反亲和、GPU 型号和故障域如何共同影响 placement？
79. **[面经高频]** 抢占式调度如何与 checkpoint 协同？如何估算最优 checkpoint interval？
80. **[系统设计]** 作业弹性扩缩容时，global batch、学习率、dataloader 和 optimizer state 如何调整？
81. **[故障诊断]** 千卡作业平均每几小时遇到一次硬件故障，如何降低有效训练时间损失？
82. **[岗位强相关]** 如何做 GPU/NIC/节点健康检查，避免坏节点反复被调度？
83. **[故障诊断]** loss 正常但 tokens/s 慢慢下降，应该监控哪些训练、GPU、CPU、网络和存储指标？
84. **[系统设计]** 设计训练任务 watchdog：如何区分慢、hang、进程崩溃和 silent data corruption？
85. **[岗位强相关]** 如何记录 experiment lineage，使代码、镜像、配置、数据、checkpoint 和指标完全可追溯？
86. **[系统设计]** 多租户训练平台如何做 quota、公平调度、优先级、成本核算和隔离？

## 9. RL Training Infra（不考算法推导）

### 9.1 角色、数据流与资源映射

87. **[面经高频]** 画出 prompt → rollout → reward/verifier → log-prob → advantage → train 的端到端数据流；哪些边传小对象，哪些边传大 tensor？
88. **[岗位强相关]** Actor、Reference、Reward、Critic/Verifier 分别执行 generate、forward 还是 backward？各自适合什么并行策略？
89. **[系统设计]** 如何为不同角色分配 GPU？静态配比、动态伸缩与自动 device mapping 各自的优缺点是什么？
90. **[岗位强相关]** 为什么 RL 训练不是普通的多模型训练？多对多数据重分布和角色间依赖带来了哪些额外复杂度？
91. **[源码深挖]** veRL/HybridFlow 的 single-controller 与 worker group 抽象解决了什么问题？Ray 适合走控制面还是数据面？
92. **[系统设计]** FSDP/Megatron trainer 与 vLLM/SGLang rollout 如何组合，同时避免框架各自初始化 NCCL group 时冲突？

### 9.2 Colocate、Disaggregate 与混合引擎

93. **[面经高频]** rollout 与 training 共置（colocate）和分离（disaggregate）如何取舍？从显存、利用率、网络和 on-policy 语义分析。
94. **[岗位强相关]** hybrid engine 如何在同一批 GPU 上做 role switching？训练权重、optimizer state 与 KV cache 如何 pause/offload/resume？
95. **[故障诊断]** colocate 模式在长 response 下频繁 OOM，如何定位是 KV cache、训练 activation、碎片还是权重冗余？
96. **[系统设计]** disaggregate 模式下 rollout 和 trainer GPU 比例如何根据 generation time、train time 与队列积压动态调整？
97. **[源码深挖]** 从训练并行布局切换到推理 TP 布局时，参数如何 reshard？all-gather、all-to-all 与点对点传输如何选择？
98. **[岗位强相关]** 什么时候适合把 Reference/Reward 与 Actor 共置？什么时候必须独立资源池？

### 9.3 权重同步与一致性

99. **[面经高频]** 训练引擎与 rollout 引擎之间如何同步分片权重，避免 rank 0 聚合和 CPU bounce？
100. **[岗位强相关]** 权重同步走 Ray object store、CPU shared memory、NCCL broadcast 还是 RDMA？控制面与 tensor 数据面如何分离？
101. **[源码深挖]** trainer 使用 ZeRO-3/FSDP、rollout 使用 TP 时，建立 shard-to-shard 映射需要哪些元数据？
102. **[系统设计]** 全量权重、增量权重、LoRA adapter 同步分别适合什么场景？如何校验 rollout 已加载完整版本？
103. **[故障诊断]** 部分 rollout worker 使用旧权重或错位 shard，但作业没有崩溃，如何检测这种 silent inconsistency？
104. **[岗位强相关]** 同步流水线如何 overlap 权重传输与 generation？pause/resume 或 partial rollout 有哪些一致性风险？
105. **[系统设计]** 给每个 sample 记录 policy version 有什么作用？训练端允许多大的 bounded staleness？

### 9.4 同步、异步与 Pipeline

106. **[面经高频]** 同步 RL pipeline 为什么产生 rollout bubble 和 trainer bubble？画时间线并量化利用率。
107. **[岗位强相关]** 异步 rollout-training 如何用 bounded queue 做 backpressure？队列过大为什么可能影响收敛与内存？
108. **[系统设计]** 如何在保持 on-policy 的前提下，让完整 group 一生成完就进入训练，而不等待整个 rollout batch？
109. **[源码深挖]** partial rollout 如何处理中途切换权重、未完成序列、随机数状态与 KV cache？
110. **[故障诊断]** rollout 速度大幅波动导致 trainer 饥饿，如何判断瓶颈在 sampling、reward、环境还是调度？
111. **[系统设计]** generation、reward/verifier、reference log-prob 和 actor training 如何做多级流水与容量规划？

### 9.5 Rollout、Reward 与 Agent 环境

112. **[面经高频]** response 长度长尾如何造成 straggler？dynamic batching、sequence packing、work stealing 与 request migration 如何取舍？
113. **[岗位强相关]** 一条 prompt 需要生成多条 response 时，group locality 和跨 worker regroup 如何影响通信与等待？
114. **[系统设计]** reward/verifier 是本地函数、GPU 模型还是远程服务时，分别如何做 batching、超时、重试与幂等？
115. **[故障诊断]** reward 服务变慢但 rollout GPU 正常，如何设计 trace 将延迟归因到具体 sample 和服务阶段？
116. **[岗位强相关]** 多轮 Agent rollout 中，tool call、环境状态、sandbox、timeout 与不可重放副作用如何管理？
117. **[系统设计]** 一个 trajectory 中部分环境 step 失败，应该丢弃、重试还是截断？基础设施如何保存可审计状态？
118. **[岗位强相关]** 超长 response 对 KV cache、rollout 调度、训练 activation、通信和 checkpoint 有何联动影响？

### 9.6 容错、可观测性与端到端优化

119. **[系统设计]** rollout worker、reward worker、trainer rank 分别失败时，恢复边界应该是 request、group、batch 还是 step？
120. **[故障诊断]** RL 作业吞吐下降时，如何把端到端 step time 分解为 generation、reward、reshard、sync 和 train？
121. **[岗位强相关]** RL 训练需要额外监控哪些指标：response 长度分布、queue depth、policy lag、invalid sample、reward latency 还缺什么？
122. **[系统设计]** 如何做 sample lineage，使 prompt、response、reward、policy version、训练 step 和 checkpoint 可追溯？
123. **[系统设计]** 设计一个支持 70B、长链推理、多 verifier、千卡规模的 RL Training 平台，并给出资源模型和瓶颈预算。

## 10. Coding 与现场调试

124. **[面经高频]** 用 C++/Python 实现一个线程安全的 bounded blocking queue，支持 backpressure 和关闭。
125. **[训练 Infra 高频]** 实现 ring all-reduce 的模拟器，验证不同 rank 的 chunk 流转与最终结果。
126. **[训练 Infra 高频]** 给定参数列表和 bucket 上限，实现 DDP gradient bucketing。
127. **[训练 Infra 高频]** 给定 layer FLOPs 和显存，完成 pipeline stage 的负载均衡切分。
128. **[训练 Infra 高频]** 给定多 rank 日志，找出第一个 collective 序号或 tensor shape 不一致的位置。
129. **[训练 Infra 高频]** 实现一个 topology-aware GPU allocator，优先分配同节点/同 NVSwitch 域设备。
130. **[训练 Infra 高频]** 写一个简化训练循环，正确处理 gradient accumulation、AMP、clip、DDP `no_sync` 和 checkpoint。
131. **[RL Train Infra 高频]** 实现一个按 policy version 分桶的 bounded experience queue，支持过期淘汰和 backpressure。
132. **[现场调试]** 给一段会 hang/OOM/产生 NaN 的 PyTorch distributed 代码，现场定位并修复。

## 11. 高频系统设计题

### A. 千卡预训练平台

设计一个支持 1K～10K GPU 的预训练平台。至少覆盖：

1. 作业提交、镜像与配置管理。
2. gang scheduling 与 topology-aware placement。
3. DP/TP/PP/CP 自动配置或推荐。
4. 数据供给与分布式 checkpoint。
5. 指标、profiling、hang detection 和故障归因。
6. 抢占、恢复、坏卡隔离与训练有效时间。

### B. 训练变慢排障

某次代码升级后 step time 变慢 20%，loss 正常。要求给出：

1. 如何建立可信基线并排除数据差异。
2. 如何拆分 compute、collective、pipeline bubble、dataloader、checkpoint 时间。
3. Nsight Systems / PyTorch Profiler / NCCL 日志分别看什么。
4. 如何二分代码、配置、kernel、驱动和节点拓扑。
5. 如何把本次问题沉淀成自动性能回归检测。

### C. 大规模训练 Hang

作业运行数小时后随机 hang，无 Python 异常。要求给出：

1. watchdog 与 rank progress 如何设计。
2. 如何定位最后一个 collective 和对应源码调用点。
3. 如何区分 rank divergence、CUDA error、进程退出与网络故障。
4. 如何自动保留现场、隔离节点并恢复作业。

## 12. 项目深挖模板

训练 Infra 岗通常会沿简历项目连续追问。至少准备以下证据：

1. **规模**：模型、tokens、GPU 数、网络与训练时长。
2. **基线**：优化前 step time、tokens/s、MFU、峰值显存。
3. **定位**：用什么 profiler/指标证明瓶颈在哪里。
4. **改动**：修改了框架哪一层、通信/计算路径如何变化。
5. **收益**：端到端收益而非单 kernel microbenchmark。
6. **正确性**：loss 对齐、数值容差、故障注入和回归测试。
7. **代价**：显存、复杂度、稳定性、适用范围与回滚方案。

## 公开资料与面经线索

### 面经 / 岗位范围

- [AI Infra 面试汇总（牛客）](https://www.nowcoder.com/discuss/891334000239734784)：训练/推理框架、算力调度、数据工程与性能优化的综合面试方向。
- [AI Infra 面试收录](https://www.cnblogs.com/xmwblogs/p/19669357)：社媒面经线索聚合，用于交叉核对分布式训练、并行策略、通信和 coding 高频项。
- [小红书：大模型训练/压缩/推理 Infra 研发工程师](https://job.xiaohongshu.com/campus/position/17015?referer_code=C09UF8OCSZ4M)：明确要求多机多卡训练、TP/PP/ZeRO-3 动态协同和超长序列显存/通信优化。
- [小红书：AI Infra / 超大模型训练与推理框架](https://job.xiaohongshu.com/social/position/18875)：用于核对 GPU/NPU/PPU/CPU 异构与千卡框架方向。
- [阿里云：分布式基础平台 / AI Infra](https://www.nowcoder.com/jobs/detail/444262)：强调 Linux、GPU/网络/存储、profiling、调度、可观测性和故障复盘。

### 实现与系统资料

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Efficient Large-Scale Language Model Training with Megatron-LM](https://arxiv.org/abs/2104.04473)
- [DeepSpeed + Megatron 530B Training](https://arxiv.org/abs/2201.11990)
- [MegaScale: Scaling LLM Training to More Than 10,000 GPUs](https://arxiv.org/abs/2402.15627)
- [HybridFlow / veRL: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)
- [veRL 官方仓库](https://github.com/verl-project/verl)
- [OpenRLHF 文档](https://openrlhf.readthedocs.io/en/latest/)
- [OpenRLHF Performance Tuning](https://openrlhf.readthedocs.io/en/latest/performance.html)
- [PyTorch TorchRL：Weight Synchronization](https://pytorch.org/rl/stable/reference/collectors_weightsync.html)
- [NVIDIA Megatron FSDP 文档](https://docs.nvidia.com/nemo/megatron-bridge/0.4.1/training/megatron-fsdp.html)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

## 来源边界

小红书、脉脉原帖大量需要登录或 App 跳转，且搜索引擎摘要可能不完整。本文不复制受限原文，不虚构作者、日期或题目出处；“面经高频”表示在公开可检索面经、聚合摘要或多个岗位讨论中反复出现，不代表特定公司一定会问。

欢迎通过 Issue / PR 补充可公开核验的训练 Infra 面经。请勿提交候选人、面试官或内部系统的敏感信息。
