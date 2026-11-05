# 第一个 Notebook 实现说明

交付文件为 [`01_Baseline_Evaluation_and_Dataset_Preparation.ipynb`](01_Baseline_Evaluation_and_Dataset_Preparation.ipynb)。本轮只修订第一个 Notebook 的验证、合成数据和 Baseline 相关部分，没有新增教学用 `.py` 文件，也没有修改原始 `train.csv` 或第二个 Notebook。Notebook 默认 `RUN_BASELINE=False`；合成 Pilot 仍固定使用种子 42，每类请求 100 个单次候选。

## P0：数据正确性与可追踪性

- 官方数值答案和合成数值答案统一使用 `math.isclose(predicted, expected, rel_tol=1e-2, abs_tol=1e-5)`；Bit Manipulation 使用 8-bit 字符串精确匹配。
- Gravity Verifier 使用自己的浮点 OLS 拟合和答案计算；Equation Numeric Verifier 使用独立的 24-rule 枚举、全局约束和 `_numeric_verify_output()`；Bit Verifier 使用独立解析器、求值器和有限候选语义枚举。Verifier 不调用对应 Generator 的答案入口，也不调用 Equation Solver 作为证明路径。
- Generator 采用单次尝试，Solver 随后从公开 Prompt 计算答案和轨迹，Screening 再决定接受或拒绝。`synthetic_candidates.jsonl` 保留每次尝试的 `candidate_id`、`generation_status`、`generation_error`、`verification_status` 和 `rejection_reasons`；本次没有生成阶段异常，但 25 条筛查拒绝样本完整保存在 `synthetic_rejected.jsonl`。
- 负样本报告 `data/processed/verifier_negative_tests.json` 共 10 个案例，10/10 通过，覆盖错误答案、错误轨迹、非法示例输出、不可识别 Prompt、非法 Bit 表达式和重复 Prompt。
- 合成与真实数据交叉审计保存于 `data/processed/synthetic/synthetic_real_leakage_audit.{json,csv}`。本次 300 条候选与真实集合的精确 Prompt、规范化 Prompt、结构指纹均为 0 次匹配。结构指纹匹配只作为风险指标；审计明确记录未使用 Validation/Holdout 答案，且没有可可靠继承的原始 `source_row`。

## 官方数据主线

- `train.csv` 重新读取为 9,500 行，字段为 `source_row/id/prompt/answer/source_dataset`；无缺失、重复 ID 或完全重复 Prompt。原始文件 SHA256 为 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`。
- 精确模板分类得到 Gravity 1,597、Bit Manipulation 1,602、Equation Numeric 732。Equation Symbolic 823 被单独标记并排除，Unit Conversion、Encryption、Numeral System 也保留排除原因。
- 清洗结果保留原始 ID、Prompt、Answer、`source_row` 和来源信息。真实数据按任务分层、固定种子 42 做 80%/10%/10% IID 划分：Train 3,145、Validation 393、Holdout Test 393。精确 ID/Prompt 没有跨集合重复；规范化 Prompt 存在跨集合重复，因此 Notebook 不宣称 Structural OOD 无泄漏。

## P1：合成数据 Pilot

本次实际生成 300 条单次候选，全部完成独立 Solver/Verifier/Screening：

| task | candidates | accepted | rejected | acceptance |
|---|---:|---:|---:|---:|
| Gravity | 100 | 100 | 0 | 1.00 |
| Equation Numeric | 100 | 89 | 11 | 0.89 |
| Bit Manipulation | 100 | 86 | 14 | 0.86 |
| 合计 | 300 | 275 | 25 | 0.917 |

拒绝原因统计：Bit 为 `answer_verification_failed`/`answer_not_identifiable` 各 14 次；Equation Numeric 为前两项各 11 次，另有 10 次 `guess_answer_ambiguous`。`guess_answer_ambiguous` 不进入正式合格数据。Deduce/Guess 统计为 280/20；Guess 子类型为 `guess_answer_equivalent` 10、`guess_answer_ambiguous` 10，本 Pilot 没有 `guess_rule_unique` 样本。Equation 的 24 条规则均在规则覆盖统计中出现，且真实 Train 中观察到的 26 个 Query Operator 已全部纳入 Operator 池；`equation_operator_inventory.json` 记录没有被先验排除的真实符号。

Gravity 分布审计使用 Train 的 1,277 条题、5,180 个示例：每题示例数 3/4/5 的计数为 383/439/455；`t` 中位数 3.0、范围 1.0–5.0；OLS `g` 中位数约 12.123、范围约 4.909–19.574；距离小数位为 1 位 525 条、2 位 4,655 条。新 Generator 从 Train 的经验 `g_ols`、时间、示例数和小数位频率采样，独立采样每题隐藏 `g`，不固定 9.8，也不使用旧的 9.50–19.50 均匀网格。审计和新旧分布对比图分别为 `gravity_distribution_audit.json` 和 `gravity_distribution_comparison.png`。

合成产物位于 `data/processed/synthetic/`：

- `synthetic_candidates.jsonl`
- `synthetic_accepted.jsonl`
- `synthetic_rejected.jsonl`
- `synthetic_screening.csv`
- `synthetic_statistics.json`
- `generation_metadata.json`
- `synthetic_real_leakage_audit.json/csv`

## P1：NeMo Data Designer + 本地 vLLM

- 目标机为 AutoDL `connect.westd.seetacloud.com:48510`，项目和环境位于 `/root/autodl-tmp/Qwen3-experiment` 数据盘；模型位于 `/root/autodl-tmp/huggingface/models/Qwen3-4B-Thinking-2507`。
- AutoDL 数据盘上的 uv 环境包括 `.venv-pt-sft-cu128`、`.venv-q-qlora-cu128`、`.venv-pt-rl-cu128` 和 `.venv-vllm-cu128`；官方训练 Profile 已执行 `uv pip check`，vLLM Profile 实际使用 PyTorch `2.8.0+cu128`。模型权重已完整下载到数据盘；后续下载优先执行 `source /etc/network_turbo` 使用 AutoDL 学术加速，避免本地中转。
- 实际环境版本：vLLM `0.11.0`（`.venv-vllm-cu128`），Data Designer `0.10.0`（`.venv-data-designer`）。模型服务使用本地 OpenAI-compatible vLLM `http://127.0.0.1:8000/v1`，served model 为 `Qwen3-4B-Thinking-2507`，没有使用 `NVIDIA_API_KEY`、`build.nvidia.com`、OpenAI 或 OpenRouter。
- Data Designer 采用官方 `ModelProvider(provider_type="openai", endpoint=..., api_key=None)`、`LocalFileSeedSource`、`DataDesignerConfigBuilder.with_seed_dataset` 和 `DataDesigner.preview`。Preview Seed 严格分层为 Gravity/Equation Numeric/Bit 各 3 条，只发送 `task_type` 和 `prompt`，不发送 Ground Truth、Hidden Rule 或 Teacher Trace。
- 9 条 Seed 产生 9 条 Preview 记录，18 次本地请求全部成功；5 条通过独立 Solver/Verifier。Preview 仅保存在 `data_designer_preview.jsonl` 和 `data_designer_preview_screening.csv`，状态为 `preview_only_never_ground_truth`，没有合并到正式训练数据。

## P2：RTX 5090 完整 Baseline

在 RTX 5090 上使用本地模型和原始比赛 Prompt 对 Holdout Test 的 393 条样本完成完整推理，未注入课程先验、高级 Prompt、合成数据、SFT 或 RL。结果文件为 `data/processed/baseline_results.jsonl` 和 `data/processed/baseline_summary.json`：

- Overall sample-weighted Accuracy：`0.0152671756`
- Macro Accuracy：`0.0125`
- Gravity：160 条，Accuracy `0.00625`
- Equation Numeric：73 条，Accuracy `0.0`
- Bit Manipulation：160 条，Accuracy `0.03125`
- Answer Extraction Success Rate：各任务 `1.0`
- Invalid Output Rate：各任务 `0.0`
- 每条生成 512 tokens，`finish_reason=length`，截断率 `1.0`
- 总输出 tokens：`201,216`；总生成时间约 `28.612s`；吞吐约 `7,032.5 output tokens/s`
- vLLM 加载时间约 `16.372s`，`max_model_len=4096`，`max_new_tokens=512`
- `nvidia-smi` 后台采样峰值显存：`27,959 MiB`；初始 `3 MiB`，增量约 `27,956 MiB`。PyTorch allocator 的 `max_memory_allocated` 对 vLLM 为 `0 MiB`，因此报告以 nvidia-smi 设备采样值为准。

这组结果是当前模型配置和 512-token 截断设置下的真实基线，不代表后续训练效果。完整输出和失败样本均已保留；远端运行结束后 GPU 已释放，未终止其他用户进程。

## 未完成事项与限制

- 当前 Bit Generator 是课程规定的 8-bit 有限 DSL（NOT、SHL/SHR、ROTL/ROTR、AND/OR/XOR，最大深度 2），不宣称覆盖比赛全部位操作规则。
- Gravity 的经验分布来自当前 Train IID 划分；规范化 Prompt 的跨集合重复仍是数据审计风险，尚未获得可靠同源组键，不能据此声称无泄漏或完成 Structural OOD。
- Baseline 所有样本达到 `max_new_tokens=512`，后续若需比较完整思考链，应另行评估更合适的生成长度；本 Notebook 默认仍不自动运行 Baseline。
- 后续五个 Notebook 尚未开发。
