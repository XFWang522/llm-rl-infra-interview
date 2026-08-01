# LLM / RL Infra 面试问题清单

面向大模型训练、后训练（RLHF / DPO / GRPO）、Rollout、推理引擎和分布式系统岗位。

这份清单综合了小红书、脉脉等社媒面经线索、可公开检索的转载/聚合面经，以及目标岗位的官方 JD。社媒原帖常有登录限制且内容会变动，因此本文做如下区分：

- **面经高频**：在公开面经或其可检索摘要中明确出现，或在多份面经中反复出现。
- **JD 强相关**：由官方岗位职责直接推导出的高概率考点。
- **延伸追问**：为形成完整知识图谱而补充，不声称是某一篇帖子的原题。

> 建议准备方式：每题先用 2 分钟讲清核心，再准备 10 分钟版本，包含公式、系统图、瓶颈、指标和真实故障案例。

## 1. RLHF / RL 算法基础

1. **[面经高频]** 从 SFT、Reward Model 到 PPO，完整讲一遍经典 RLHF pipeline；每个阶段的输入输出是什么？
2. **[面经高频]** PPO 的 clipped objective 为什么能限制策略更新？clip range 过大或过小分别会怎样？
3. **[面经高频]** KL penalty 的作用是什么？固定 KL 系数和 adaptive KL controller 如何取舍？
4. **[面经高频]** Reward Model 如何训练？pairwise loss 如何推导？数据偏差、reward hacking 和过拟合怎么处理？
5. **[JD 强相关]** DPO 为什么不需要显式训练 Reward Model？它与 PPO 的目标有什么联系？
6. **[延伸追问]** GRPO 与 PPO 的差异是什么？不用 critic 的收益和代价分别是什么？
7. **[延伸追问]** RLOO、Reinforce++、DAPO、Dr.GRPO 分别解决了什么问题？
8. **[面经高频]** advantage 如何计算？GAE 中 `gamma` 和 `lambda` 对 bias/variance 有何影响？
9. **[延伸追问]** sequence-level reward 如何分配到 token？length bias 如何产生并被抑制？
10. **[延伸追问]** on-policy、off-policy 和 stale policy 在大模型 RL 中分别意味着什么？
11. **[延伸追问]** 为什么训练过程中 reward 上升但真实能力可能下降？如何设计防作弊评测？
12. **[延伸追问]** rule-based reward、model-based reward、process reward 如何组合？

## 2. RL 系统与数据流

13. **[JD 强相关]** 画出 Actor、Reference、Reward、Critic、Rollout Engine 的数据流和参数流。
14. **[JD 强相关]** colocated 与 disaggregated RL 架构如何取舍？哪些模型共享 GPU，哪些分开部署？
15. **[JD 强相关]** Rollout、reward 计算和 training 如何流水化？如何减少 GPU bubble？
16. **[JD 强相关]** 一批 prompt 生成长度差异很大时，straggler 如何拖慢整个 step？如何调度？
17. **[延伸追问]** dynamic batching、continuous batching、sequence packing 在 RL 训练中分别解决什么问题？
18. **[JD 强相关]** 训练引擎与推理引擎之间如何高效同步权重？全量参数、分片参数和增量更新如何选择？
19. **[延伸追问]** 权重同步时如何保证 rollout 使用的是一致版本？是否允许 bounded staleness？
20. **[延伸追问]** rollout 数据如何落盘、shuffle、去重和回放？如何保证 prompt 与 response 可追溯？
21. **[延伸追问]** 多轮 Agent RL 中环境状态、tool call、timeout 和不可重放外部副作用如何建模？
22. **[延伸追问]** 训练中途 worker 失败，如何做到 sample-level 或 step-level 恢复？
23. **[延伸追问]** 如何给 RL pipeline 做 backpressure，避免 rollout 把训练端或存储端压垮？
24. **[系统设计]** 设计一个支持 70B 模型、1K GPU、32K response 的 RL 训练平台。

## 3. 分布式训练与并行策略

25. **[面经高频]** DP、TP、PP、CP/SP、ZeRO/FSDP 各切分什么？通信算子分别是什么？
26. **[JD 强相关]** TP、PP、ZeRO-3 与 RL 多模型、多阶段流程如何动态组合？
27. **[面经高频]** ZeRO-1/2/3 分别切分 optimizer state、gradient、parameter 的哪部分？显存如何估算？
28. **[面经高频]** FSDP all-gather / reduce-scatter 在 forward/backward 的哪个阶段发生？
29. **[面经高频]** Pipeline Parallel 的 bubble 如何计算？1F1B、interleaved 1F1B、zero-bubble 有何区别？
30. **[面经高频]** Tensor Parallel 中列并行和行并行线性层为什么分别需要 all-gather / all-reduce？
31. **[JD 强相关]** 超长序列下 context parallel 如何切分 attention？Ring Attention 的通信模式是什么？
32. **[延伸追问]** MoE 的 expert parallel 如何实现？all-to-all 为什么容易成为瓶颈？
33. **[延伸追问]** MoE 负载不均衡如何观测和优化？capacity factor、aux loss、expert dropping 的影响是什么？
34. **[系统设计]** 给定模型参数量、序列长度和集群拓扑，如何搜索最优并行策略？
35. **[延伸追问]** 为什么 RL 阶段最优并行配置可能和 SFT 不同？
36. **[延伸追问]** H100 节点内 NVLink、节点间 RDMA 的带宽差异如何影响并行维度映射？

## 4. 显存、通信与性能优化

37. **[面经高频]** 训练一个 Transformer 时，参数、梯度、optimizer state、activation 各占多少显存？
38. **[面经高频]** activation checkpointing 为什么省显存但增加计算？应该在哪些层或算子上使用？
39. **[面经高频]** BF16、FP16、FP8 的数值范围和训练稳定性有什么差异？什么时候需要 loss scaling？
40. **[JD 强相关]** 超长 response 为什么同时放大显存、通信和负载不均衡问题？
41. **[延伸追问]** sequence packing 如何避免 padding 浪费？如何正确构造 position id 和 attention mask？
42. **[延伸追问]** FlashAttention 为什么是 IO-aware？online softmax 如何保证数值稳定？
43. **[延伸追问]** 如何把通信与计算 overlap？哪些 collective 最适合异步化？
44. **[面经高频]** NCCL all-reduce 变慢时如何定位：网络、拓扑、消息大小、rank 偏斜还是 GPU kernel？
45. **[延伸追问]** gradient accumulation 对吞吐、显存、优化器语义和通信频率的影响是什么？
46. **[延伸追问]** GPU 利用率很高但 MFU 很低，可能是什么原因？
47. **[系统设计]** 某次升级后 tokens/s 降低 20%，你会用哪些指标和 profiler 按什么顺序排查？
48. **[系统设计]** 训练偶发 hang，但没有 OOM 或报错，如何做分布式故障定位？

## 5. Rollout 与推理引擎

49. **[JD 强相关]** Prefill 和 Decode 的计算/访存特征有什么不同？为什么 Decode 更容易 memory-bound？
50. **[面经高频]** KV Cache 大小如何估算？GQA/MQA 为什么能降低 KV Cache？
51. **[延伸追问]** PagedAttention 如何降低显存碎片？block size 如何影响浪费和调度开销？
52. **[延伸追问]** continuous batching 的调度循环如何设计？吞吐和 TTFT/TPOT 如何权衡？
53. **[延伸追问]** prefix caching 在 RL rollout 中什么时候收益高？如何处理失效和多租户隔离？
54. **[延伸追问]** speculative decoding 为什么能加速？draft model 接受率受哪些因素影响？
55. **[JD 强相关]** Rollout 侧需要 deterministic replay 时，随机种子、采样算子和并行度变化如何处理？
56. **[延伸追问]** 推理引擎中的 CUDA Graph 有什么收益和约束？动态 shape 怎么处理？
57. **[系统设计]** 设计一个为 RL 训练服务的高吞吐 rollout 集群，支持优先级、取消、超时和弹性扩缩容。

## 6. 可靠性、可观测性与平台化

58. **[JD 强相关]** 多机多卡 RL 作业应该监控哪些指标？请覆盖算法、训练、推理、通信和集群层。
59. **[延伸追问]** checkpoint 应保存哪些状态，才能做到严格恢复：模型、优化器、scheduler、RNG、dataloader 还缺什么？
60. **[延伸追问]** checkpoint 写入很慢时，如何做异步、分片、增量和分层存储？
61. **[系统设计]** 某些 rank 频繁 OOM，但平均显存正常，如何定位长尾 sample 和显存碎片？
62. **[系统设计]** 如何设计作业抢占与恢复，使集群利用率提高但不破坏 RL 的 on-policy 语义？
63. **[延伸追问]** 如何设计 experiment lineage，确保模型、代码、数据、配置和评测结果可复现？
64. **[延伸追问]** 多租户平台如何做 GPU quota、优先级、公平调度和故障域隔离？

## 7. Coding / 基础系统题

65. **[面经高频]** 实现线程安全的 LRU Cache，并扩展 TTL、并发淘汰和统计指标。
66. **[面经高频]** 合并区间、Top-K、生产者消费者、阻塞队列等常见中等题。
67. **[Infra 高频]** 实现一个按 token 数组 batch 的调度器，使 padding 最小且满足最大 batch token 限制。
68. **[Infra 高频]** 给定多 rank 日志，找出第一个 collective 顺序不一致的位置。
69. **[Infra 高频]** 实现 consistent hashing，并讨论节点扩缩容时的数据迁移比例。
70. **[Infra 高频]** 用 Python/C++ 实现 ring buffer 或无界队列，并分析线程安全与 backpressure。
71. **[Infra 高频]** 手写 stable softmax / log-softmax，并解释数值稳定性。
72. **[Infra 高频]** 写一个简化版 gradient accumulation loop，正确处理 loss scaling、clip 和 optimizer step。

## 8. 项目深挖题

73. 你做过的最有效性能优化是什么？基线、profiling 证据、改动和收益分别是什么？
74. 如果吞吐提升 30% 但训练 loss 发生微小漂移，你如何证明改动可上线？
75. 讲一次分布式训练事故：症状、根因、为什么原监控没发现、长期修复是什么？
76. 你如何判断应该改算法、并行策略、kernel、通信库还是调度系统？
77. 如果重新设计当前项目，你会删除哪层抽象？为什么？
78. 如何把一次性的训练脚本演进成多团队可复用的平台，同时控制复杂度？

## 高频 Mock 组合

### 组合 A：RL 系统

1. 讲 PPO / GRPO。
2. 画 RL 数据流。
3. 解释 colocated 与 disaggregated 架构。
4. 设计 rollout-training pipeline。
5. 追问权重同步、staleness、容错和监控。

### 组合 B：分布式训练

1. 估算 70B 模型显存。
2. 设计 DP × TP × PP × CP。
3. 解释每一维通信。
4. 根据节点拓扑放置并行维度。
5. 排查吞吐下降或 collective hang。

### 组合 C：推理 / Rollout

1. 解释 Prefill / Decode。
2. 估算 KV Cache。
3. 讲 PagedAttention 与 continuous batching。
4. 设计长短请求混合调度。
5. 权衡吞吐、TTFT、TPOT、显存和公平性。

## 公开来源与边界

- [小红书 2026 校招：大模型训练/压缩/推理 Infra 研发工程师](https://job.xiaohongshu.com/campus/position/17015?referer_code=C09UF8OCSZ4M)：明确提到 RLHF/DPO、Rollout、Reward Model、多阶段 Pipeline、TP/PP/ZeRO-3 和超长时序瓶颈。
- [小红书社会招聘：AI Infra / 大模型训练与推理框架](https://job.xiaohongshu.com/social/position/18875)：用于核对超大模型、异构算力、分布式训练和推理方向。
- [牛客：算法实习面经—小红书多模态一面](https://www.nowcoder.com/feed/main/detail/bc194d18c3dc4842978980abb7e6a108)：公开摘要出现 Reward Model 训练难点、显存与训练速度优化等问题。
- [牛客：小红书大模型后训练一面](https://www.nowcoder.com/discuss/872533899736334336)：用于补充项目深挖、Agent 设计评估和 coding 的面试形式。
- [AI Infra 面试收录](https://www.cnblogs.com/xmwblogs/p/19669357)：聚合了社媒面经线索及 PPO、KL、MoE、分布式训练等高频方向。
- [小红书大模型面经公开整理](https://blog.csdn.net/2401_83173765/article/details/148292651)：用于交叉核对项目深挖、论文讲解和 coding 的面试结构。

### 关于小红书 / 脉脉原帖

两站的大量内容需要登录、App 跳转或未被搜索引擎完整索引。本文不复制受限原文，不为不可核验内容虚构作者、日期或帖子链接；“面经高频”只表示问题在公开可检索面经/摘要中出现或被多来源交叉印证，不代表特定公司一定会问。

## 如何贡献

欢迎提交 Issue / PR。新增题目时请标注：

1. 公司与岗位方向（可匿名）。
2. 时间范围。
3. 原题或回忆转述。
4. 可公开访问的来源链接（如有）。
5. 不包含面试官或候选人的个人信息。
