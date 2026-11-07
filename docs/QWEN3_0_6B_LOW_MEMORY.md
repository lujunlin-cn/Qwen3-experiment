# Qwen3-0.6B：8 GB 显存预算与本地改造

2026-11-07：克隆机已关机。本轮完成本地代码与 CPU 数值检查，**未测新配置的 GPU 峰值或速度**。
下文的规划值是设备总显存预算，包含运行时余量；它们不是实测值或严格分配上界。
原始测量保留在 [基准报告](QWEN3_0_6B_BENCHMARK.md)。

| 阶段 | 原设备峰值 | 新配置规划值 | 本轮状态 |
|---|---:|---:|---|
| PPO rollout | 10.12 GiB | 约 6.9 GiB | 已实现，GPU 待测 |
| GRPO rollout | 18.10 GiB | 约 6.9 GiB | 已实现，GPU 待测 |
| DPO BF16 LoRA 训练 | 28.88 GiB | 约 6.7 GiB | 已实现，GPU 待测 |
| GRPO BF16 LoRA 训练 | 19.80 GiB | 约 5.6 GiB | 已实现，GPU 待测 |

这里 1 GiB = 1024 MiB；8 GB 十进制约为 7.45 GiB。规划值也低于这一容量，但余量较小，仍需实际 CUDA 验证。
先保留 BF16 策略精度；本次降显存不依赖 NF4，也没有缩短回答或减少 GRPO 的 G。

## Rollout：减少固定 KV cache 与并发

上一轮 runner 明确预留 PPO 8 GiB、GRPO 16 GiB KV cache，加上约 2.1 GiB 的权重、LoRA、运行时等开销，就超过了 8 GiB。
这些测量反映当时选择的推理配置，不是模型的最低显存要求。

04–06 Notebook 和 runner 统一提供以下默认配置：

```text
kv_cache_memory_bytes = 4 * 2**30
max_num_seqs = 2
max_num_batched_tokens = 1024
enable_chunked_prefill = True
enforce_eager = True
max_tokens = 16384
max_model_len = 20480
```

Qwen3-0.6B 的 28 层、8 个 KV heads、head_dim=128，BF16 KV 每 Token 为
`2 × 28 × 8 × 128 × 2 = 114688 bytes = 112 KiB`。
4 GiB 缓存理想容量约 37449 Token；两个「16384 输出 + 1024 Prompt」序列约需 3.72 GiB。
固定 6 道 Train Prompt 均远短于 1024 Token；更长 Prompt/上下文或块分配开销可能触发调度重算，此时可将并发降为 1。
GRPO 仍生成每题 8 个候选，由调度器分批运行；**G=8 不要求同时驻留 8 条序列**。

上一轮 DPO 已在 4 GiB KV cache 下测得 6273 MiB（6.13 GiB）设备峰值；新 PPO/GRPO 规划沿用该非 KV 开销，并额外留 0.75 GiB，合计约 6.88 GiB。
这是跨阶段外推，不能当作新实测。减少并发/预填充批量通常降低吞吐，也可能减少缓存抢占；具体速度需复测。
显式设置 `kv_cache_memory_bytes` 时，仅修改 `gpu_memory_utilization` 不能替代缩小这块缓存。

## 训练：分块计算输出投影及对齐检查

训练使用 `use_cache=False`，不分配生成阶段的 KV cache；应处理 Logits、Softmax 与激活，而不是继续减 KV。
词表大小为 151936，完整 `[1, 16384, 151936]` Logits 单份 BF16 为 4.64 GiB、FP32 为 9.27 GiB。
旧 DPO 在 Reference 评分中把整段 Logits 转 FP32 再做完整 LogSoftmax，产生多个大张量。
旧 GRPO 虽已分块 Softmax，但仍构造完整 Logits；行为 Logprob 对齐也走完整词表输出，抬高整个阶段峰值。

新增 [low_memory_logprobs.py](../utils/low_memory_logprobs.py) 先运行保留 PEFT LoRA 的 Qwen Decoder，再对 128 Token 的隐藏状态分块执行 LM Head 和完整词表 Softmax。
每块 BF16 Logits 约 37.1 MiB、FP32 约 74.2 MiB。投影与 Softmax 一起做 Activation Checkpoint，反传时重算，避免保留各块的大词表计算图。
Decoder 仍使用完整序列和原有 Gradient Checkpointing；没有对词表、Token、Mask 或梯度作采样/裁剪近似。

- DPO 的不可变 Reference 缓存和 Policy 两侧评分均使用新路径；Chosen/Rejected 的两个 Decoder 图仍可能同时保留，规划已包含这一开销。
- GRPO 的真实训练和 BF16/NF4 Logprob 对齐都使用新路径；G=8、原始行为 Logprob 分母、组奖励、累积窗口均保留。
- 全有效序列只使用 Decoder 的 Causal Mask，保留真正 Padding Mask；CUDA 上显式 BF16 autocast，以便使用高效 SDPA。
- 保存后的短 Prompt Logits 对比仍用于 Adapter 重载验证；其序列最多 32 Token，不会产生长序列词表张量。

按最大上下文 20480、micro-batch=1、rank=8 LoRA 规划，保守计入以下分项：

| 分项 | DPO | GRPO |
|---|---:|---:|
| BF16 基座权重（596049920 参数，Embedding 与 Head 共享） | 1.11 GiB | 1.11 GiB |
| 5046272 LoRA 参数的 FP32 权重、梯度和 Adam 两个矩 | 0.08 GiB | 0.08 GiB |
| 每层一份 Decoder Checkpoint 边界，DPO 计两条序列 | 2.19 GiB | 1.09 GiB |
| Decoder 重算与反传 Workspace 预留 | 1.50 GiB | 1.50 GiB |
| 分块输出投影与 Softmax Workspace 预留 | 0.35 GiB | 0.35 GiB |
| CUDA Context、Allocator 和其他开销预留 | 1.50 GiB | 1.50 GiB |
| 合计 | **6.72 GiB** | **5.63 GiB** |

前两项由固定配置及保存的 Adapter 计算，Checkpoint 项按 Tensor 形状计算；后三项是显式工程余量，依赖 CUDA Kernel、版本及分配器行为。
该预算要求高效 SDPA/Flash Attention，不能回退到保存完整 `T×T` Attention 的实现。若实测超过目标，先核对 Attention 后端与张量峰值，再考虑顺序 DPO 反传或 NF4。
直接改成 NF4 会改变相对 BF16 rollout 的策略分布，需要重做对齐与量化差异报告，不能视为只改显存。

## 验证与复跑

本地检查使用 Torch 2.11.0、Transformers 5.12.0、PEFT 0.21.2；与远端原训练的 Transformers 5.18.0 有版本差异。
4/5/6 Notebook 完整顺序 CPU 机制检查通过。新增检查覆盖实际 151936 维词表、FP32/BF16 输出投影、真实小型 Qwen3+PEFT、Padding、Gradient Checkpointing、DPO/GRPO Loss 与梯度、非零优化器更新。
这些固定 CPU 样例的最大数值/梯度差均为 0；LM Head Hook 确认 11 Token 的序列只投影 4/4/2 Token，未构造完整序列 Logits。
它们验证计算实现，不代表已经加载真实 0.6B 权重或验证了 CUDA 显存。

可复核文件：[CPU 数值检查](../data/processed/benchmarks/qwen3-0_6b/low_memory_20261107/cpu_numerical_checks.json)、[可重算显存预算](../data/processed/benchmarks/qwen3-0_6b/low_memory_20261107/memory_budget.json)。
Git 保存源码、固定模型配置、Manifest 和审计记录；模型与 Adapter 二进制权重保留在实验机器上。
预算脚本可用固定配置和 LoRA Manifest 计算参数量，无需权重；重新核验实际 Adapter 参数或复跑训练时需已有的 checkpoint 文件。

```bash
CUDA_VISIBLE_DEVICES='' .venv-low-memory/bin/python utils/check_low_memory.py
CUDA_VISIBLE_DEVICES='' .venv-low-memory/bin/python utils/budget_low_memory.py
```

有卡后使用新 Run ID 和原始 SFT 父策略，复用同一批真实回答对比训练显存：

```bash
export QWEN3_MODEL_ID=Qwen/Qwen3-0.6B
export QWEN3_MODEL_PATH=/root/autodl-tmp/huggingface/models/Qwen3-0.6B
export LOCAL_MODEL_PATH="$QWEN3_MODEL_PATH"
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# 每阶段新进程，先确认 GPU 空闲；新结果不覆盖旧结果。
.venv-pt-rl-cu128/bin/python utils/benchmark_training.py dpo_training --run-id qwen3_06b_lowmem_20261107 --source-run-id qwen3_06b_baseline_rl_20261107 --mode bf16_lora
.venv-pt-rl-cu128/bin/python utils/benchmark_training.py grpo_training --run-id qwen3_06b_lowmem_20261107 --source-run-id qwen3_06b_baseline_rl_20261107 --mode bf16_lora
.venv-vllm-cu128/bin/python utils/benchmark_training.py ppo_rollout --run-id qwen3_06b_lowmem_rollout_20261107 --source-run-id qwen3_06b_baseline_rl_20261107 --mode nf4_qlora
.venv-vllm-cu128/bin/python utils/benchmark_training.py grpo_rollout --run-id qwen3_06b_lowmem_rollout_20261107 --source-run-id qwen3_06b_baseline_rl_20261107 --mode bf16_lora
```

`--source-run-id` 只复制父 SFT Manifest；训练时另复制真实 Rollout 与其 Manifest，并核对 SHA256。原父 Adapter 通过原路径读取，不重复训练 SFT。
训练复用的旧 Rollout 与新生成的 Rollout 使用独立 Run ID，已有相同阶段产物拒绝覆盖。
`--kv-cache-gib`、`--max-num-seqs`、`--max-num-batched-tokens`、`--projection-chunk-size` 可显式调整，Protocol 记录参数及 helper SHA256。
