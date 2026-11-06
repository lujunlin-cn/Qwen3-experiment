# Nemotron SFT-RL Notebook 集成实现说明

本仓库交付六个阶段 Notebook 的集成修订；所有教学实现仍直接位于 Notebook Code Cell，没有新增教学用 `.py` 文件，也没有修改原始 `train.csv`。Notebook 默认关闭模型、vLLM 和 GPU 执行；模型路径统一通过 `QWEN3_MODEL_PATH` 等环境变量提供。

## 本轮六 Notebook 全链路集成修订（基于 `4c28ae2`）

- 统一模型接口：固定 `Qwen/Qwen3-4B-Thinking-2507`；Notebook 3 是唯一 SFT 初始化来源。正式 SFT 完成后写入 `data/processed/checkpoints/sft_manifest.json`，包含 Base Model、Tokenizer、Adapter、LoRA/QLoRA、数据版本、策略、Token Budget、Epoch 和 Seed。Notebook 4/5/6 通过 `PeftModel.from_pretrained()` 从该 Manifest 加载同一个 SFT Adapter，不把 Adapter 目录直接传给 `AutoModelForCausalLM.from_pretrained()`。
- Notebook 3：NF4、Double Quant、BF16 Compute 和 `prepare_model_for_kbit_training()` 已接入；Two-stage 真实执行 Recall Drill → Full Trace，并从 Stage 1 Adapter 继续。Smoke 固定 24 条训练轨迹、1 Epoch、NF4 QLoRA。
- Notebook 4：移除无条件 GPU 阻断；Rollout → 独立 Verifier → Chosen/Rejected → 手写 DPO Loss → Adapter 保存/重载入口已串通。Smoke 选 Equation Numeric/Bit 各 2 道 Train 题，每题 4 个候选；Chosen 和 Rejected 均排除截断回答。Reference Policy 与 Policy 使用同一个 SFT Adapter，Reference 冻结。
- Notebook 5：移除无条件 GPU 阻断；vLLM 行为 Logprob、Response Mask、EOS/Done、Outcome Reward、旧 Value、GAE、PPO-Clip 和 Value Head 入口已串通。Smoke 选 Equation Numeric/Bit 各 2 道 Train 题，采样参数为 Temperature 1.0、Top-p 1.0；Rollout 完成后释放 vLLM，再加载训练 Actor-Critic。
- Notebook 6：移除无条件 GPU 阻断；每个 Prompt 严格保留 G=8，Rollout 后执行独立 Verifier 并附加 Reward、Correct、Verifier Version。Mixed Group Ratio 为 0 时 fail closed，不人为制造方差。当前指标 `policy_logprob_change` 只表示行为策略到更新策略的 Logprob 变化，不冒充 Reference KL，因此 `KL_COEF=0`。
- 六个 Notebook 均已在 DGX Spark 顺序执行 CPU Smoke：JSON/AST 检查、数据接口、Label Mask/Loss、QLoRA 配置、Preference Pair、PPO GAE/Reward/Mask、GRPO Group/Verifier/Advantage 和 Manifest 检查均通过。报告文件分别为 `data/processed/baseline_cpu_smoke_report.json`、`data/processed/prompt_engineering/prompt_cpu_smoke_report.json`、`data/processed/sft/cpu_sft_smoke_report.json`、`data/processed/dpo/cpu_dpo_smoke_report.json`、`data/processed/ppo/cpu_ppo_smoke_report.json`、`data/processed/grpo/cpu_grpo_smoke_report.json` 和 `data/processed/grpo/cpu_readiness_audit.json`。
- 本轮未运行 Qwen3 权重、vLLM 或 GPU 训练；CPU Fixture 不作为模型效果结果。由于 CPU 阶段不伪造正式 Adapter，`data/processed/checkpoints/sft_manifest.json` 只有明天 SFT QLoRA Smoke 成功后才会出现；当前仅保存 `data/processed/sft/cpu_checkpoint_manifest.json`。
- 已清理公开 Notebook 和结果 JSON 中的机器专属路径，历史模型来源统一写为 `${QWEN3_MODEL_PATH}` 或环境相关占位符。

### RTX 5090 明日 Smoke 顺序

1. Notebook 1：Validation 每类 3 条 Baseline Smoke，确认生成长度和截断率。
2. Notebook 2：Validation 每类 3 条，P0–P5 共用相同样本和解码预算。
3. Notebook 3：官方 SFT/QLoRA Profile，`RUN_SFT=1`、`RUN_MODE=smoke`、NF4 QLoRA；检查 Adapter 保存、Manifest 和 Reload。
4. Notebook 4：官方 RL Profile，先 vLLM Rollout，再关闭 vLLM，使用各任务 2 题 × 4 候选构造偏好对并执行一次 DPO Update。
5. Notebook 5：重新启动独立 vLLM Rollout，关闭后加载同一 SFT Adapter，执行各任务 2 题的一次 PPO Rollout、GAE 和 Update。
6. Notebook 6：执行各任务 2 题、每题 G=8 的真实 Rollout 和 Readiness Audit；只有出现真实 Mixed Group 才执行一次 GRPO Update。

每一步记录样本数、Rollout 数、Reward、Loss、Peak GPU Memory、Wall Time、Adapter Path 和成功状态。三种 RL 方法均从 Notebook 3 的同一个 SFT Adapter 平行初始化；GPU 阶段若某一步失败，保留错误日志并停止该阶段，不用 CPU Fixture 替代。

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

- 远端验证使用 GPU 数据盘；项目、环境和模型均通过环境变量或官方 Profile 配置，不把机器专属路径写入教程。
- 远端 uv 环境按官方训练 Profile 分离管理；训练 Profile 和 vLLM Profile 均已完成依赖检查。模型下载使用目标机的官方网络加速和数据盘，不通过本地中转。
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
- 03--06 Notebook 的真实 Qwen3 GPU 训练、显存、吞吐和能力提升尚未运行；所有未运行项均保留为 `Not Run`，不能据 CPU Fixture 推断模型效果。

# 后训练 Notebook 实现记录（03--06）

## 03_SFT.ipynb

- 已写入 Direct Full-Trace、Joint、Two-stage（Memorization → Execution）和可选 Targeted Drill 的数据审计与手写 token-level Cross Entropy。
- CPU Smoke Test 通过：8 条可审计轨迹、Prompt/ Padding mask 不计入 Loss、Loss 有限且完成一次优化器更新；BF16 LoRA 与 NF4 QLoRA 仅提供配置入口，未在当前 DGX Spark 加载模型。

## 04_DPO.ipynb

- 已写入多回答 Rollout Schema、独立 Verifier 偏好对构造、Hard Negative 审计、冻结 Reference Policy 和手写 DPO Loss。
- CPU Smoke Test 通过：16 条 Fixture Rollout、4 个有效偏好对、DPO Loss 有限、Policy 有梯度；vLLM Rollout 和真实 DPO 更新未运行。

## 05_PPO.ipynb

- 已写入共享 Backbone + Value Head 的 Actor--Critic、终止状态不 bootstrap 的 GAE、PPO-Clip、Value Loss、Entropy 和 KL 诊断。
- CPU Smoke Test 通过：GAE 形状与 Mask、终止奖励、有限 Loss 和反向传播均通过；真实 Qwen3 PPO 未运行。

## 06_GRPO.ipynb

- 已写入 `G=8` Group Rollout 协议、All-correct/Mixed/All-wrong 分类、Group Advantage、Clipped Objective、KL 诊断、CausalLM 更新入口和 RL Readiness Audit。
- vLLM 入口保留 Prompt/Completion Token IDs、Response Mask、采样参数、策略版本、`finish_reason` 和行为策略 Logprob；Verifier 通过独立适配器注入 Reward，禁止用 Generator 或模型自评提供答案。
- 新增输入审计：缺字段、非法 Reward、行为 Logprob 长度错位、不完整 Group 均 fail closed；新增数值/Bit 错误答案负样本测试。
- CPU Smoke Test 通过，报告保存于 `data/processed/grpo/cpu_readiness_audit.json` 和 `data/processed/grpo/cpu_grpo_smoke_report.json`；Tiny Policy 与 CausalLM-shaped Update 均得到有限 Loss 和梯度。Fixture 明确覆盖三类 Group，All-correct 与 All-wrong 的零方差比例为 `2/3`；该数值只用于公式测试，不代表模型实验。
- `RUN_GRPO=False`，当前未运行 GPU Rollout、真实 Verifier、训练更新、显存和吞吐测量。RTX 5090 阶段应先用最佳 SFT Checkpoint 对 Equation Numeric/Bit 进行 `G=8` Smoke，再决定是否进入 Mixed Group 训练池。

## 第二轮审查修订（`docs/writing-block.md`）

- **SFT 梯度累积**：`run_sft_epoch` 现在累加每个 microbatch 的有效 Assistant Token Loss Sum，在窗口结束时按实际有效 Token 数缩放梯度；最后不足累积步数的窗口不再使用固定分母。CPU 机制测试覆盖 5 个 microbatch、累积步数 2 和不同有效 Token 数。
- **PPO 对齐**：明确隐藏状态 `t` 预测 token `t+1`。Policy Logprob、Value、Reward、Done、Response Mask 和 GAE 统一到动作轴；增加短序列人工 Value/Token 对齐测试。截断样本不声明 EOS，也不获得正确奖励。
- **统一 Verifier**：Notebook 04/05/06 使用 Notebook 01 的 `official_nvidia_metric_v2` marker-first、任务专用格式和数值容差；Rollout 记录 `verifier_version`，错误答案、非法格式和截断输出均有负样本测试。Notebook 01 的合成 Screening 记录 `synthetic_independent_verifier_v3`。
- **DPO Reference**：Policy 与 Reference 均从 SFT Manifest 的同一 Base Model/Adapter 初始化；Reference 冻结；标准 DPO 使用 Response Token Log Probability 求和；截断 Chosen/Rejected 均排除。
- **Adapter 交接**：SFT Manifest 增加 `policy_version`，DPO/PPO/GRPO Manifest 保存 Parent SFT Adapter、Base Model、Tokenizer 和自身 Policy Version。RL 入口要求 SFT Manifest 状态为 `completed` 且 Adapter 目录存在；不会把 DPO 输出作为 PPO/GRPO 默认初始化。
- **CPU 回归**：六个 Notebook 已重新通过 JSON/AST 和顺序 CPU Smoke；本轮新增的 SFT 累积窗口、DPO 序列长度/截断、PPO Value/Action 对齐、GRPO Mask/Verifier 机制测试均通过。SFT 机制测试显式固定在 CPU，即使主机可见 CUDA 设备也不会误占用 GPU。所有 Qwen3、vLLM、RTX 5090 训练和 Adapter 实际重载仍标记为 `Not Run`。

### 当前未解决问题

1. 仍未在 RTX 5090 上验证 QLoRA Adapter 被 Transformers 和 vLLM 交替加载的实际兼容性。
2. PPO/GRPO 当前没有实现 Reference Policy KL；仅保留行为策略 Logprob 变化诊断，KL 系数为 0。
3. Rollout 的真实 vLLM 0.11 Token IDs/Logprob 返回格式、显存峰值和吞吐尚待目标环境实测；若字段缺失，入口会 fail closed。
4. 规范化 Prompt 的跨集合重复仍是数据风险，原始数据没有可靠同源组键，Structural OOD 尚未宣称无泄漏。
