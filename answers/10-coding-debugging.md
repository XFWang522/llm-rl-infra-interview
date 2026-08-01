# 第 10 章：Coding 与现场调试

[返回题目清单](../README.md#10-coding-与现场调试)

## 124. 线程安全 Bounded Blocking Queue

状态包括固定 capacity、deque、mutex、`not_empty/not_full` condition variables 和 `closed`。`put` 在锁内用 `while size==capacity && !closed` 等待，醒来后重新检查谓词；入队后通知 `not_empty`。`get` 对空队列同理，取出后通知 `not_full`。

关闭时持锁设置 `closed=true` 并 `notify_all`：生产者立即失败；消费者可选择 drain 剩余元素后返回 EOF。不能用 `if` 代替 `while`，要处理 spurious wakeup。异常/取消必须用 RAII 释放锁。复杂度摊销 `O(1)`；进一步讨论公平性、批量 put/get、超时、指标和 backpressure 传播。

## 125. Ring All-Reduce 模拟器怎么写？

将每 rank tensor 均分 `N` chunks。Reduce-scatter 第 `s` 轮，rank `r` 发送 chunk `(r-s) mod N` 给右邻，接收左邻对应 chunk并累加；`N-1` 轮后每 rank 有一个 reduced chunk。All-gather 再转发 reduced chunks `N-1` 轮。

模拟器应使用“上一轮快照→下一轮状态”，不能在单线程循环中原地发送导致后处理 rank 读到本轮新值。测试 N=1、不能整除的 padding、不同规约操作和随机输入，与集中式 sum 比较。每 rank 通信量约 `2(N-1)M/N`，轮数 `2(N-1)`。

## 126. DDP Gradient Bucketing

输入参数 `(name,numel,dtype,device,ready_order)` 与 byte limit。按 dtype/device 分组，再按预期 backward ready order 顺序贪心装桶；单参数超过上限则独立成桶。维护每参数在 flat buffer 中的 offset/length，并按 alignment padding。

运行时 hook 将 gradient copy/view 到 bucket，递减 pending count；为零时异步 all-reduce。测试共享/unused 参数、不同 dtype、梯度 ready 乱序和 accumulation。仅按模型声明顺序装桶可能推迟通信；生产实现会用首轮 ready order 重建。

## 127. Pipeline Stage 负载均衡切分

给每层 profile 权重：forward、backward、recompute、通信和显存。要求连续层分给 `P` stages，并满足每 stage memory limit，目标最小化最大 stage time。

可用动态规划：`dp[i][p]` 表示前 i 层切 p 段的最小最大耗时，枚举最后切点 j，若区间显存可行，转移 `min max(dp[j][p-1], cost(j,i))`。复杂度 `O(P L²)`，可用前缀和和二分目标时间优化。Embedding/loss、tied weight 和 virtual stage 是额外约束；最终需加入 P2P 与真实 schedule 验证。

## 128. 如何从多 Rank 日志找 Collective Mismatch？

把每条记录规范成 `(process_group_id, sequence_no, op, shape, dtype, callsite, timestamp)`。按 group+sequence 聚合，检查是否缺 rank、op/shape/dtype 不同；第一个不一致序号通常比最终 timeout 更接近根因。

日志可能缺失/乱序，应依据 per-rank monotonic sequence，而非 wall clock；先验证 group membership。输出各 rank 最后一致调用和首次差异的源码栈。线上可用固定大小 flight recorder 环形缓冲，hang 时集中导出，避免全量日志开销。

## 129. Topology-Aware GPU Allocator

硬件建模为层次图：host→NUMA/PCIe root→NVSwitch→GPU，并关联 NIC/rail/健康分。请求描述 GPU 数和通信 group（如 TP=8、PP=4）。先枚举满足硬约束的 gang，再用代价函数评分：高频 group 跨慢链路成本、碎片、故障相关性和排队公平。

分配必须原子 compare-and-swap/事务，防并发 overcommit；释放幂等。测试异构 GPU、降级链路、并发请求、找不到完整 gang 和 backfill。P7 回答还应指出这是带约束图嵌入/装箱问题，通常使用启发式而非追求全局最优。

## 130. 正确的 AMP + Accumulation + DDP 训练循环

每个 optimizer step 先 `zero_grad`。循环 G 个 micro-batches，前 G-1 次在 `model.no_sync()` 中 forward；`scaled_loss = scaler.scale(loss/G)` 后 backward。最后一次触发 DDP sync，然后 `scaler.unscale_(optimizer)`，做 gradient clipping/finite check，再 `scaler.step(optimizer); scaler.update()`，最后 scheduler step。

Checkpoint 只在 optimizer-step 边界，保存 scaler、scheduler、RNG、sampler 和 consumed tokens。若发生 overflow，optimizer step 被跳过，scheduler 是否推进必须有明确语义。测试 G=1、overflow、最后不完整 accumulation、resume 与基线对齐。

## 131. 按 Policy Version 分桶的 Experience Queue

每项含 sample/group ID、policy version、created time、token bytes 和完整状态。Queue 以 version→deque 组织，同时维护总 item/byte 上限、最小可接受版本和 condition variables。

生产端在容量满时 backpressure；消费端只取已完整、未过期且满足 version policy 的 group，通常按最老可接受版本/FIFO。版本下限提升时批量淘汰旧 items并记录 wasted rollout tokens。状态变更用幂等 ID 和 commit marker，恢复避免重复训练。必须同时限制 item 数与 bytes，因为长 response 大小差异显著。

## 132. 现场排查 Hang、OOM、NaN 的方法

先分类并缩小首个异常点：

- **Hang**：各 rank 心跳/最后 collective sequence；找首个调用分歧或最早退出 rank，保存 stack、NCCL flight recorder、CUDA/Xid 和网络状态。
- **OOM**：记录每 rank allocated/reserved/peak、当前 module/sample tokens；区分 model states、activation、KV/sync buffer、引用泄漏和碎片。
- **NaN**：固定 seed/batch，finite hooks 二分首层；单卡/FP32/禁 fused kernel 对照，检查 AMP scale、输入和多卡差异。

现场修改遵循一次只变一个变量，先保留最小复现与证据，再修根因。验收包含功能/数值对齐、故障注入、性能回归和长期 soak；“加 timeout、减 batch、重启”只能用于止损，不能作为最终修复。
