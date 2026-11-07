# Qwen3-0.6B：8 小时课程实验

本轮只完成本地脚本、Notebook 接入和 CPU 验证。新配置的 GPU 峰值、吞吐、收敛时间和效果均待开机实测。旧结果见 [原基准报告](QWEN3_0_6B_BENCHMARK.md)，不能充当本轮结果。

## 固定的教学路线

| 方法 | 训练题型 | 展示重点 | 评测 |
|---|---|---|---|
| SFT | 罗马数字、Gravity、数学方程 | 主要展示罗马数字 | 三题型 |
| DPO | Gravity、数学方程 | Gravity 长答；方程正确率、截断和长度 | 三题型 |
| PPO | 仅数学方程 | 方程正确率、截断和长度 | 三题型，Gravity 只作评测 |
| GRPO | 仅数学方程 | 同上 | 同上 |

各 RL 方法独立从**同一个选定 SFT checkpoint** 出发，不串接 DPO→PPO→GRPO。训练和主要模型对照都固定原始题面；另在同一最终面板上评测公开先验，避免把换提示词误计成训练效果。

不能预先保证三题均提升、RL 必然提升，或只有 RL 才能改善长度/截断。短而正确的 SFT 教师轨迹本身也可能改善这些指标。报告保留所有真实结果，包括退步、没有偏好对、零组方差、超限及预算未完成，不筛选展示样本制造结论。

## 提示词与评测口径

- 罗马数字：保留原有题面，加 `1=I, 2=II, …, 9=IX, 10=X`，不注入目标答案。
- Gravity：正确物理量是位移 `d=½gt²`，反推 `g=2d/t²`；多条四舍五入观测使用 `g=2Σ(tᵢ²dᵢ)/Σ(tᵢ⁴)`。原题已经给出了正向公式，这次增加的是反推公式和估计方法；不假设 `g=9.8`。
- 数学方程：复用 Notebook 02 的全部 24 种公开候选规则、normal/flip、语义组和 join 例外；不提供当前隐藏映射。保留答案中的操作符前后缀，例如 `17/`，用 Notebook 00 当前验证器评分。
- 基线扫描输出上限 `512 / 2048 / 8192 / 16384`。主要对照固定为罗马数字 8192、Gravity/方程 16384；短上限另列，不能用降低上限造成的长度减少充当模型学习效果。
- 统一输出 `accuracy / truncation_rate / mean_output_tokens`，长度按实际 `completion_token_ids` 计数，包含 Thinking、结束 Token 及错误/截断样本，**不含输入**。另外给出答对样本平均输出长度、样本数和正确率 Wilson 95% 区间。
- 评测采用官方 Thinking 模板，`temperature=.6, top_p=.95, top_k=20`；同面板同 seed 同 cap。训练 rollout 用 `temperature=1, top_p=1, top_k=-1`，保留实际采样 Token 的原始 logprob；禁止用重算概率替换行为分母。

固定 Validation 调参面板每题 4 个、最终面板每题 8 个，seed=42，互不重叠；Holdout 不参与。小样本用于课程筛查，不能据此宣称统计显著或充分泛化。最终面板不参与 checkpoint 选择。脚本会先冻结数据、先验、源代码与哈希，恢复运行时校验；模型权重和 Tokenizer 也记录哈希。

## 时间预算与停止条件

| 阶段 | 上限 | 内容 |
|---|---:|---|
| 基线 | 45 分钟 | 主要对照、max_tokens 扫描 |
| 提示词 | 45 分钟 | 相同题目的先验对照 |
| SFT | 90 分钟 | 每题 64 条已验证 Train 轨迹，每轮一 epoch，最多 4 轮 |
| DPO | 90 分钟 | 每轮每题 4 道、每题 4 个候选，最多 6 轮 |
| PPO | 90 分钟 | 每轮 8 道方程、每题 1 个候选，最多 6 轮 |
| GRPO | 90 分钟 | 每轮 2 道方程、每题 G=8，最多 6 轮 |
| 最终复核 | 至少预留 30 分钟 | 两种提示词、5 个模型，使用前面节省的剩余时间 |

总计 8 小时；每种方法的预算包含进程启动、rollout、reference 评分、反传、保存、验证。每个 job 都有独立日志和时长，报告同时汇总 training / rollout / evaluation 和方法总耗时。

验证分数为任务宏平均 `accuracy - .15*truncation_rate - .05*mean_output_tokens/cap`。至少两次有效验证后，连续两次没有超过 0.01 的改进记为 `validation_plateau`，记录达到该条件及到最佳 checkpoint 的时间。这只是预设的小规模停止条件；`max_rounds`、`budget_exhausted` 和失败均**不称为收敛**。

RL 起始输出上限 4096；若 DPO 没有偏好信号、GRPO 组方差为零，或 PPO 没有正确结果且存在截断，下一轮预注册 Train 批次使用 8192，再到 16384。不制造正确样本或组方差。短答奖励为正确完整答案 `1-.15*output_tokens/cap`，错误或截断为 0。DPO 使用“正确优于错误/截断”和“正确短答优于正确长答”（至少短 64 Token 且 20%）两种偏好。

旧基线平均输出较长，低并发下不能保证全部大上限对照在 8 小时内完成。主要基线结束后会写 `pilot_budget_estimate.json`，按实测速率估计最终评测开销。脚本不悄悄减少样本/上限：未完成项会列出。若要调预算或样本数，应复制配置、使用新 Run 目录重新注册。

## 8 GB 显存控制

这里 **8 GB = 8,000,000,000 bytes ≈ 7.45 GiB**，比 8 GiB 更严格。

- vLLM：固定 3 GiB KV cache、并发 1、chunked prefill 512、eager，无 CUDA Graph；G=8 分批生成。每 Token KV 理论值 112 KiB，3 GiB 理想约能容纳 28086 Token，足够一条 20480 上下文，仍需实测调度与工作区。
- 训练：LoRA rank8、micro-batch=1、梯度累积、Decoder 非重入 checkpoint、`use_cache=False`；64 Token 一块计算完整词表投影及 Softmax，不构造整段 `[T,V]`。
- SFT 使用 NF4 QLoRA。DPO/PPO/GRPO 默认 BF16 LoRA，和 BF16 rollout 一致；这些是新课程 runner 的配置，不改变旧 PPO 冒烟入口的 NF4 契约。
- DPO 顺序重算并反传 chosen/rejected，精确链式法则给出原 DPO 梯度；冻结 SFT reference 用共享基座的独立 LoRA 评分，只缓存标量，不再并存两套模型或两侧 Decoder 图。
- PPO 保存并恢复 value head、Adam 状态；错误和截断样本也训练。`gamma=1, lambda=1`，按有限时域输出任务在预算结束处用零 bootstrap，不伪装观察到 EOS。PPO/GRPO 采用 Clip 目标，reference KL 系数为 0，不宣称实现了额外 reference-KL 正则。

监督进程开跑前确认选定 GPU 没有其他计算进程；每约 0.25 秒采样整张 GPU 的 `memory.used`，包含上下文、保留 KV 等，达到阈值即终止**本任务进程组**。另设 PyTorch allocator 上限为设备限额的 90%，避免只看 `max_memory_allocated` 漏掉 vLLM/驱动占用。失败、超时和监控失效都会保留日志。

显存报告写 `sampled_pass`：监控采样不是连续硬件上界证明，采样间的瞬时峰值仍不能排除。新设置只有在目标机器跑过对应实际训练/rollout 后才能报告通过；没有真实优化器更新的阶段会标明 `workload_exercised=false`。本地预算与更多背景见 [低显存说明](QWEN3_0_6B_LOW_MEMORY.md)。这套机制没有 1F1B 或多卡流水线。

## 明天开机后的命令

在克隆机仓库内执行，先取得本次提交。沿用已安装的 vLLM 和 QLoRA 环境；后者也能运行 BF16 LoRA。若环境路径不同，更换参数即可。

```bash
git pull --ff-only
python utils/course_run.py prepare --run-dir data/processed/course_runs/course8h_20261108

python utils/course_run.py run \
  --run-dir data/processed/course_runs/course8h_20261108 \
  --model /root/autodl-tmp/huggingface/models/Qwen3-0.6B \
  --inference-python /root/autodl-tmp/Qwen3-experiment/.venv-vllm-cu128/bin/python \
  --training-python /root/autodl-tmp/Qwen3-experiment/.venv-q-qlora-cu128/bin/python \
  --gpu 0

python utils/course_run.py report --run-dir data/processed/course_runs/course8h_20261108
```

`prepare` 与 `report` 不加载 CUDA。`run` 才会启动 GPU 工作。若控制进程正常中断，原命令可恢复已经完成的 job；已完成权重不会覆盖。每轮 Adam/value head 继续恢复，DPO reference 永远固定为最初 SFT。时间窗口从首次 `run` 开始，恢复不会把 8 小时重置。显存超限等失败记录不会在恢复时自动重跑以掩盖失败，应排查后另开 Run。运行中修改源代码、冻结面板、配置或模型会被拒绝。

输出：`report.md` 是可读表格，`report.json` 包含逐任务对照增量、逐阶段时长、显存验收和缺失项；`state.json`、各 job 的 `job.json / result.json / guard.json / worker.log / device_memory.jsonl` 可追溯。正式结果目录默认 Git 忽略，后续只挑选去除权重和优化器的大体积日志摘要提交。

Notebook 入口为 [07_Eight_Hour_Experiment.ipynb](../07_Eight_Hour_Experiment.ipynb)。核心 SFT/DPO/PPO/GRPO 目标及新增偏好、奖励定义仍在 03–06 教学 Notebook；工具负责调度、检查与 I/O，旧冒烟报告保持历史含义。

## 本地验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-low-memory/bin/python -m unittest discover -s tests -p test_course_experiment.py -v
```

覆盖数据隔离、输出长度口径、特殊方程答案、DPO 顺序反传的数值/梯度等价、SFT 分块梯度等价、偏好和奖励优先级、真实 tiny Qwen3＋PEFT 四方法连续两轮训练/保存/恢复，以及显存监控和超时终止（后两项通过模拟设备读数验证，不是 GPU 实测）。
