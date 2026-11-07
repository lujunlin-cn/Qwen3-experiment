# Qwen3-0.6B 三题型与训练资源基准

实测日期：2026-11-07。克隆机 SSH 端口为 30122，项目位于 `/root/autodl-tmp/Qwen3-experiment`。
Baseline 和四种训练资源 Smoke 均已完成，GPU 已释放。
后续 8 GB 本地改造与预算见 [低显存说明](QWEN3_0_6B_LOW_MEMORY.md)。本页保留原实测及其源码快照，新实现尚未复测 GPU，不能回写为原测试的显存结果。
设备自报 NVIDIA GeForce RTX 4080 SUPER / 32760 MiB；不能将结果解释为标准 16GB RTX 4080 或 RTX 5090 的性能。
模型为 `Qwen/Qwen3-0.6B`，Revision `c1899de289a04d12100db370d81485cdf75e47ca`。
容器 CPU 时间配额相当于 16 个逻辑核，内存上限约 62 GiB；主机枚举到的 128 CPU / 503 GiB 内存不能当作本容器的可用配额。

Baseline 使用原始官方 Prompt、官方 Thinking 模板、BF16 基座，不加载 Adapter。
数据版本为 `roman_gravity_equation_v1`，固定种子 42；校准与正式评测均取 Validation，二者 ID 互斥，Holdout 不用于预算选择。
正式采样设置为 temperature=0.6、top_p=0.95、top_k=20，max_model_len=20480。
统一 Verifier 为 `official_nvidia_metric_v3`：Numeral 检查标准 Roman 合法性与精确数值，数值题使用 `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`；截断回答一律失败。

| 题型 | max_tokens | 正确数 / 50 | Accuracy | 截断率 | 平均生成 Token |
|---|---:|---:|---:|---:|---:|
| Numeral System | 8192 | 20 | 40% | 16% | 3492.54 |
| Gravity | 16384 | 47 | 94% | 4% | 8020.88 |
| Equation Numeric | 16384 | 6 | 12% | 30% | 13494.40 |
| 合计 | 按题型 | 73 / 150 | 48.67% | 16.67% | 8335.94 |

Numeral 的 512/1024 初始预算不足，追加 4096/8192/16384 后按协议选择最小完整预算 8192。
Gravity 选择 16384；Equation 在 16384 仍有校准题截断，状态为 `budget_cap_unresolved`。
每类仅 3 道校准题，预算选择不能保证所有正式样本完整结束。不同采样批次的输出长度也不保证随预算单调。
正式评测的 25 条截断回答均记失败，不从 Thinking 中间结果补计分。
正常结束后仍判失败的数量为 Numeral 22 条、Gravity 1 条、Equation 29 条，低分也包含完整回答的错误。

预算校准实测如下，每个单元为「正确数 / 完整结束数」，分母均为固定的 3 题；这些题与正式 50 题评测互斥。
这是小样本预算诊断，不能解释为每个预算都有 50 题的 Accuracy 曲线。

| max_tokens | Numeral System | Gravity | Equation Numeric |
|---:|---:|---:|---:|
| 512 | 0 / 0 | 未测 | 未测 |
| 1024 | 1 / 1 | 未测 | 未测 |
| 4096 | 2 / 2 | 0 / 0 | 0 / 0 |
| 8192 | 3 / 3 | 2 / 2 | 0 / 0 |
| 16384 | 2 / 2 | 3 / 3 | 1 / 2 |

Baseline 总生成 1,250,391 Token，生成时间 3135.01 秒、吞吐 398.85 output tokens/s；含加载及 warmup 的 GPU 阶段 3152.92 秒。
设备显存采样峰值为 27,999 MiB。该数值包含按设备显存比例预留的 vLLM KV cache，**不代表模型最低运行显存**。
150 条输出的固定 ID、文件 SHA256、Token 数量与上限、结束状态及离线重评分均通过审计。

原始证据位于 [Baseline 目录](../data/processed/tasksets/roman_gravity_equation_v1/baseline/qwen3-0_6b_round1/)，
包括 `final_budget_decision.json`、`evaluation_status.json`、`evaluation_results.jsonl` 和 `evaluation_integrity_audit.json`。

训练资源测试使用 Run ID `qwen3_06b_baseline_rl_20261107`，三种方法从同一份 SFT Adapter 平行初始化。
SFT 是真实 Train 中每类 8 条、合计 24 条已独立复核轨迹的 Direct NF4 QLoRA，1 Epoch、micro-batch=1、gradient accumulation=2。
不使用合成数据；实际教师序列最长 288 Token，SFT 序列上限 1024，拒绝截断教师 Completion。
RL 固定每类两道 Train 题；DPO 每题 4 个候选，PPO 每题 1 个，GRPO 每题 G=8。
RL 均使用 Thinking、temperature=1、top_p=1、top_k=-1，输出上限 16384、上下文 20480。
Rollout 使用 vLLM 0.11.0、BF16、独立 SFT LoRA（rank=8）、`enforce_eager=True`，不是合并 LoRA 或 CUDA Graph 的吞吐测试。

| 阶段 | 样本 / 更新 | 阶段耗时 | 设备显存峰值 | 状态 |
|---|---|---:|---:|---|
| SFT NF4 QLoRA | 24 条、12 步 | 21.46 秒 | 3090 MiB | 完成，Adapter 实际重载通过 |
| DPO rollout BF16 | 24 条 | 2050.90 秒 | 6273 MiB | 完成 |
| DPO 训练 BF16 LoRA | 3 个偏好对、1 步 | 9.31 秒 | 29578 MiB | 完成，Adapter 实际重载通过 |
| PPO rollout BF16 | 6 条 | 935.73 秒 | 10359 MiB | 完成 |
| PPO 训练 NF4 QLoRA | 5 条有效、5 micro steps、1 步 | 32.43 秒 | 3362 MiB | 完成，Adapter 与 Value Head 实际重载通过 |
| GRPO rollout BF16 | 48 条、6 组 G=8 | 2565.22 秒 | 18537 MiB | 完成 |
| GRPO 训练 BF16 LoRA | 48 micro-batch、2 步 | 88.16 秒 | 20276 MiB | 完成，Adapter 实际重载通过 |

DPO rollout 的纯生成耗时约 2018 秒、吞吐 101.72 output tokens/s，配置为 4 GiB KV cache、max_num_seqs=8。
PPO 使用 8 GiB KV cache、max_num_seqs=4；GRPO 使用 16 GiB KV cache、max_num_seqs=8。
配置差异会影响速度和显存，不作为方法间的公平吞吐排名。
GRPO 先前的 8 GiB / 4 路并发未完成尝试保存在该 Run 的 `attempts/rollout_kv8_seqs4_interrupted/`，不计入正式耗时或合并为完整输出。
DPO 的 3 个真实偏好对来自 Numeral 2 题与 Equation 1 题，Gravity 两题所有候选都正确，无法构成偏好对。
Smoke 实际只对首个偏好对更新一次；较长 Equation 候选仍参与冻结 Reference Logprob 缓存，因此其开销包含在显存峰值中。
首对 chosen/rejected 为 1416/1863 Token；最长偏好对为 9207/11585 Token，未静默裁短。

PPO rollout 的纯生成耗时 913.32 秒，49,397 output tokens，吞吐 54.09 output tokens/s；6 条均正常结束。
Notebook 只让 Verifier 正确的完整回答进入 PPO 更新，因此实际使用 5/6 条，排除 1 条完整但答错的回答；这会改变训练样本分布。
五条训练序列总长度分别为 5599/4998/8660/8280/8913，micro-batch=1、chunk_size=128；LoRA 和 Value Head 均有实际非零参数更新。
每类一条真实轨迹共 27,515 Token 的行为 Logprob 对齐中，BF16 相对 vLLM 的平均绝对差为 0.016713、P99 为 0.137905；
NF4 相对 BF16 rollout 的平均绝对差为 0.225285、P99 为 2.164168、最大差为 7.458895。
上述工程门槛通过不意味着策略相同，未替换原始行为 Logprob 分母，也未把这些差值当作 KL。

GRPO 训练将整段 FP32 词表 LogSoftmax 改为 128 Token 分块，反传时重新计算完整词表 Softmax；BF16 Logits 仍按完整序列保留。
该修改减少 FP32 中间张量，不改变 Token、Mask、G=8 Advantage 或目标函数。
12 组 FP32/BF16、非连续 Tensor 与分块边界对照覆盖实际 151936 维词表，最大输出差为 0、最大一阶梯度差 2.39e-7；完整 Notebook CPU 机制检查通过。
GRPO rollout 保留生成开始时的旧源码快照，训练阶段使用修复后的新快照，各自通过 Protocol 的 SHA256 对应。
GRPO 的纯生成耗时 2539.79 秒，总计 377,943 output tokens，吞吐 148.81 output tokens/s。
真实奖励产生 3 个 Mixed、2 个 All-correct、1 个 All-wrong 组，Mixed Ratio=50%、Zero-Variance Ratio=50%；全部组使用统一 Verifier 计算二元奖励。
BF16 对齐检查覆盖 25,435 Token，平均绝对差 0.015845、P99 0.130555。
训练保留全部 48 条、377,943 个 Response Token，包含 2 条截断回答的零奖励；micro-batch=1、gradient accumulation=32，完成两个累积窗口。
所有组仍按 G=8 计算 Advantage，零方差组的 Advantage 为零；参数更新和 Adapter 的磁盘重载均通过核验。

Train rollout 的逐条 Verifier 结果如下，单元为「正确 / 总数（截断数）」；它们来自 SFT Adapter、6 个 Train Prompt 和 T=1，不能与 150 题基座 Validation 分数当作训练前后对照。

| 方法 | Numeral System | Gravity | Equation Numeric |
|---|---:|---:|---:|
| DPO | 5 / 8（0） | 8 / 8（0） | 2 / 8（1） |
| PPO | 2 / 2（0） | 2 / 2（0） | 1 / 2（0） |
| GRPO | 13 / 16（0） | 16 / 16（0） | 3 / 16（2） |

SFT、DPO/PPO/GRPO 状态分别保存在 `data/processed/{sft,dpo,ppo,grpo}/qwen3-0_6b/<Run ID>/`。
共享父策略与保存的 Adapter 位于 `data/processed/checkpoints/qwen3-0_6b/<Run ID>/`。
Git 中保存 Manifest 和 Adapter 配置，二进制 checkpoint 留在实验机器；离线逐参数审计需要这些本地权重文件。
阶段验收以最终 `*_status.json` 为准，`*_progress.json` 保留过程快照。
速度表报告整个 GPU 阶段（包括加载、对齐检查、保存和重载），不是稳定训练步吞吐；单轮 Smoke 也不能证明训练后的能力提升。
设备显存通过 `nvidia-smi memory.used` 周期采样，Baseline 间隔 0.25 秒、PPO 0.1 秒、其他阶段 0.5 秒，可能漏过更短瞬时峰值；
各阶段另存 PyTorch allocated peak。vLLM 在独立子进程执行，不能把父进程的 allocated=0 解释为推理不占显存。
PPO 现有 Notebook 强制 NF4，需单独报告相对 BF16 rollout 的量化漂移；PPO/GRPO 的 Reference KL 未实现，KL_COEF=0。
该测试沿用 Notebook 的数据筛选、奖励和优化器逻辑，不宣称标准 RL 算法的完整训练效果。

运行入口为 `utils/benchmark_training.py`，显式绑定新版任务集和统一 Baseline Verifier，GPU 开关在加载 Notebook 时保持关闭，随后只运行指定阶段。
旧 Notebook 的两个 Bit 专用 CPU 断言在 runner 中明确标记跳过；新版选题与 SFT 24 条来源审计另存结果。
这不代表 02–06 所有交互式数据入口已完成新版任务迁移。

统一审计产物为 [summary.json](../data/processed/benchmarks/qwen3-0_6b/20261107/summary.json)，其中 `audit_status=passed`、`all_training_completed=true`。
覆盖 150 条正式评测、33 条预算校准、78 条真实 RL Rollout，以及父策略指纹、来源、Token/Mask、逐条重评分、偏好对与组奖励、实际更新和磁盘重载。
三个 RL Adapter 相对父 SFT 的 392 个 LoRA 张量均发生非零变化；最大参数差分别约为 DPO 1.00e-5、PPO 1.00e-5、GRPO 2.00e-5。
各阶段使用的代码版本通过 Protocol 的 SHA256 对应到同目录下的 `sources/` 快照。

离线复核无需加载基座模型，只在 CPU 上读取保存的 Adapter Tensor：

```bash
.venv-pt-rl-cu128/bin/python utils/audit_benchmark_results.py
```

```bash
export QWEN3_MODEL_ID=Qwen/Qwen3-0.6B
export QWEN3_MODEL_PATH=/root/autodl-tmp/huggingface/models/Qwen3-0.6B
export LOCAL_MODEL_PATH="$QWEN3_MODEL_PATH"
export HF_HUB_OFFLINE=1
# 每阶段用新进程。结果禁止覆盖；复跑请使用新的 Run ID，先完成 SFT。
.venv-q-qlora-cu128/bin/python utils/benchmark_training.py sft --run-id <new_run> --mode nf4_qlora --max-tokens 16384
.venv-vllm-cu128/bin/python utils/benchmark_training.py dpo_rollout --run-id <new_run> --mode bf16_lora --max-tokens 16384 --kv-cache-gib 4 --max-num-seqs 8
.venv-pt-rl-cu128/bin/python utils/benchmark_training.py dpo_training --run-id <new_run> --mode bf16_lora --max-tokens 16384
.venv-vllm-cu128/bin/python utils/benchmark_training.py ppo_rollout --run-id <new_run> --mode nf4_qlora --max-tokens 16384 --kv-cache-gib 8 --max-num-seqs 4
.venv-q-qlora-cu128/bin/python utils/benchmark_training.py ppo_training --run-id <new_run> --mode nf4_qlora --max-tokens 16384
.venv-vllm-cu128/bin/python utils/benchmark_training.py grpo_rollout --run-id <new_run> --mode bf16_lora --max-tokens 16384 --kv-cache-gib 16 --max-num-seqs 8
.venv-pt-rl-cu128/bin/python utils/benchmark_training.py grpo_training --run-id <new_run> --mode bf16_lora --max-tokens 16384
```

SFT 数据准备：`benchmark_training.py prepare` 从新版真实 Train 计算并独立复核参考轨迹；现有清单包含 2943 条（Numeral 1260、Gravity 1277、Equation 406）。
实际推理环境为 Torch 2.8.0 / CUDA 12.8 / Transformers 4.57.1 / vLLM 0.11.0；
训练环境为 Torch 2.11.0 / CUDA 13.0 / Transformers 5.18.0 / PEFT 0.21.2，NF4 环境另含 bitsandbytes 0.50.2。
