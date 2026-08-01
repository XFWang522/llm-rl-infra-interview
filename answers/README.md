# 答案索引

答案按主题拆分，题号与根目录 [README](../README.md) 保持一致。

## 回答标准

答案按资深 Infra 面试的五层结构审校：

1. **语义/不变量**：先说明系统必须保证什么，而不是先报框架名。
2. **数据与通信路径**：明确 tensor ownership、layout、collective/RPC 和同步点。
3. **资源与关键路径**：给出显存、通信量、复杂度或 service-rate 模型。
4. **故障与观测**：说明失败边界、幂等/恢复，以及用什么指标证明判断。
5. **取舍与适用条件**：区分理论可行、框架支持和生产值得，避免绝对化结论。

文档刻意不提供脱离版本的“唯一最佳参数”。FSDP、Megatron、NCCL、veRL/OpenRLHF 等实现细节会演进，涉及具体 API/能力矩阵的答案附有一手资料链接，并应以目标版本实测为准。

- [第 1 章：训练显存与计算量](01-memory-and-compute.md)（题 1～12，已完成）
- [第 2 章：数据并行、ZeRO 与 FSDP](02-ddp-zero-fsdp.md)（题 13～24，已完成）
- [第 3 章：Tensor / Pipeline / Context Parallel](03-model-parallel.md)（题 25～36，已完成）
- [第 4 章：MoE 训练 Infra](04-moe-training.md)（题 37～44，已完成）
- [第 5 章：NCCL、网络与拓扑](05-nccl-network.md)（题 45～56，已完成）
- [第 6 章：CUDA、算子与性能分析](06-cuda-performance.md)（题 57～66，已完成）
- [第 7 章：数据 Pipeline 与 Checkpoint](07-data-checkpoint.md)（题 67～76，已完成）
- [第 8 章：调度、容错与可观测性](08-scheduling-reliability.md)（题 77～86，已完成）
- 第 9 章：RL Training Infra（题 87～123，已完成）
  - [架构与资源编排](09a-rl-architecture.md)（题 87～98，已完成）
  - [权重同步与异步 Pipeline](09b-rl-sync-pipeline.md)（题 99～111，已完成）
  - [Rollout、Agent 环境、容错与优化](09c-rl-rollout-reliability.md)（题 112～123，已完成）
- [第 10 章：Coding 与现场调试](10-coding-debugging.md)（题 124～132，已完成）
