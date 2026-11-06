# Nemotron SFT-RL Notebook 集成实现说明

## 本轮 GPU 最终核验结果

- 本轮真实 vGPU Baseline 完成 150/150 条（Validation 每类 50）：Numeral 46/50（92%，截断 8%）、Gravity 34/50（68%，截断 32%）、Equation Numeric 19/50（38%，截断 24%）；Overall/Macro 均 66%，总体截断 21.33%。正式预算分别为 4096/8192/16384；峰值设备显存 27,659 MiB，生成吞吐约 312.89 output tokens/s，GPU 阶段约 56.73 分钟。118 条正常结束，32 条截断，不从 Thinking 中补计答案；未将该结果宣称为不受预算限制的能力。GPU 已释放。
- 交付的两个 Notebook 及当前并行修改后的工作区版本均通过 JSON/AST 和顺序 CPU 测试；150 条真实输出重新评分、Token IDs 长度/预算、文件哈希和固定 Validation ID 核验通过。本轮提交保留其他工作区修改，未将其混入提交。


## 2026-11-06：三档任务替换与独立 Baseline（本轮）

正式默认任务改为 Numeral System（`numeral_system`，Easy）、Gravity（Medium）、Equation Numeric（Hard）。Bit 保留高级扩展和历史记录，不进入新版数据或默认 Baseline。三类都计划参加 SFT；SFT 后再判断各类是否值得进入 RL，本轮未更改任何 RL 入口、算法或 Reward。02–06 和旧数据接口保持冻结，不自动读取本轮新版本。

- `utils/prepare_official_dataset.py` 提前只读处理官方 `train.csv`，分发 `real_cleaned.csv` 和 `official_preparation_manifest.json`。`00_Baseline_Diagnosis.ipynb` 只读取维护者已清洗数据，负责三集合划分、泄漏审计与 Validation Baseline。新产物位于 `data/processed/tasksets/roman_gravity_equation_v1/`。重新计算保留 3,905 条：Numeral 1,576、Gravity 1,597、Equation Numeric 732；Train/Validation/Holdout 为 3,123/391/391，种子 42。数量来自实际文件。
- Numeral 独立规范解析验证全部 6,222 条示例映射和 1,576 条目标答案；全部符合标准 Roman，观察范围 1–100。CPU 另穷举 1–3999 并覆盖非法/错值负样本。Numeric24 的符号前后缀保留，数值仍用官方容差；统一完成状态为 `official_nvidia_metric_v3`，截断回答不得取 Thinking 中间结果计分。
- 跨集合 ID、精确/规范化 Prompt 重复均为 0。结构指纹碰撞单独报告；Numeral 的 Validation 158/158 Query 映射曾在 Train 示例中出现，另有 62 道题的 Query 映射在自身示例出现。此 IID 划分不声称未见映射泛化或完全无泄漏。
- `01_Synthetic_Data_Engineering.ipynb` 默认关闭生成，按新版 Manifest 仅调度三类任务；独立版本目录保存小型 CPU Pilot，每类 5 条，15/15 通过，20/20 负样本与泄漏机制检查通过。该 Pilot 不进入 Baseline。Data Designer 本轮未运行。
- GPU 预算校准与每类 50 条 Validation 的正式结果、完整输出、Token IDs、完成状态和设备显存记录集中在新版目录的 `baseline/roman_round1/`。本轮职责拆分及验收见 [修改报告](NOTEBOOK_REFACTOR_REPORT.md)，本次没有重跑 GPU。以阶段状态文件为准，历史 512-Token 截断结果继续保留，不作为本轮模型能力结论。

以下为历史阶段记录，其旧任务集合、数据路径和未运行状态只适用于对应时间点。

本仓库当前交付 `00_Baseline_Diagnosis.ipynb`、可选的 `01_Synthetic_Data_Engineering.ipynb` 及原编号的 02–06。核心教学算法继续位于 Notebook 中；官方 CSV 的维护者清洗工具迁至 `utils/prepare_official_dataset.py`。原始 `train.csv` 保持不变。Notebook 默认关闭模型、vLLM 和 GPU 执行；模型路径统一通过 `QWEN3_MODEL_PATH` 等环境变量提供。

2026-11-06 的架构拆分与验收见 [NOTEBOOK_REFACTOR_REPORT.md](NOTEBOOK_REFACTOR_REPORT.md)。默认主线为 00 → 02 → 03 → 04–06；03 使用已验证真实 Train 轨迹，只有显式设置 `USE_SYNTHETIC_DATA=1` 才读取 `synthetic/audited/` 的 Accepted。下文保留拆分前的实验记录；其中历史 Notebook 01 同时承担的数据准备、Baseline 与合成职责，现分别由 00 和 01 承接。历史实验结果不代表本轮重跑了 GPU。

## 本轮真实 GPU Smoke 与截断诊断

本节补充此前 CPU 阶段和历史 RTX 5090 记录；本轮设备并非已验证的 RTX 5090。当前运行方法、逐阶段通过条件、硬件和阻塞见 [VGPU_SMOKE_GUIDE.md](docs/VGPU_SMOKE_GUIDE.md)，真实产物集中在 `data/processed/vgpu_smoke/` 与各算法的 `vgpu_round1/` 子目录。下面历史章节的 `Not Run` 表示当时状态，不能覆盖本节的新证据。

- Notebook 01：加入固定三道官方 Validation 题的预算诊断，原始基座与原始 Prompt，不加载合成数据或 Adapter。贪心 4096/8192 均三题截断；模型随附采样配置在 8192 下两题截断，Equation 正常结束但答错。因而 2048 预算不足有直接证据；合成训练不能解释基座现象，预算增加也不保证规则执行正确。完整原始输出与 Token IDs 均保留。
- 追加同题 16384 Token 对照：Gravity 在 8637 Token 正常结束且正确；Equation 仍在 7976 Token 结束、答错；Bit 用满 16384 Token，仍未结束 Thinking。区别记录预算不足、规则执行错误和未收敛搜索，不把这三道诊断题当作总体性能评测。
- Notebook 01/02：截断回答不再计正常答案；明确标记原始历史提取结果，离线重评分另存文件，不覆盖原始实验。01 的任意 CoT 数字回退改为明确答案标记或独立答案行，与 02/04/05/06 一致。CPU 回归覆盖“CoT 含正确 Boxed 数字但因 length 结束仍拒绝”。
- 更新后的提取与完成状态协议标记为 `official_nvidia_metric_v3`，数值容差保持不变；历史 GPU 原始记录保留实际 `v2` 标记。Baseline、Prompt 及 52 条 RL 原始回答的离线 v3 重评分单独保存，不冒充新一轮推理。
- Notebook 03：真实 Two-stage NF4 SFT 完成 24 条、1 Epoch，Stage 1/2 实际优化步 5/8，训练约 28.01 秒，设备显存峰值 5578 MiB；Stage 1 中间保存、最终 Adapter 重载和共享 Manifest 通过。后续发现预留 Recall 后的顺序截取令 Stage 2 缺 Bit；已改为 5/5/5 分层并补 CPU 回归。该修复不回写历史训练结果，也不替换第一轮 RL 已使用的父 Adapter。
- Notebook 04：16 条真实 Rollout 全部截断，偏好对为 0，状态 `blocked_no_preference_pairs`、优化步 0。Policy/Reference 的 GPU Update 和更新后 Adapter 重载仍未验证。
- Notebook 05：修复真实 Train 入口漏导入 pandas；保留首轮失败记录。修复 Transformers 5 `apply_chat_template()` 默认返回 `BatchEncoding` 的兼容处理，实际 Token IDs 与 vLLM 一致，没有放宽校验。真实 BF16/NF4 Actor-Value Head 前后向通过；4 条回答全截断，PPO 更新为 0。现有 RL 环境缺 bitsandbytes，NF4 诊断复用已有 Q-QLORA 环境，没有安装新依赖。
- Logprob 证据：4096 个真实行为 Token 上，BF16 跨引擎平均绝对差 0.011777、P99 0.131516；NF4 相对 BF16 平均差 0.121896、P99 1.251460、最大差 3.951513。工程门槛通过不等于策略相同；保留原始行为分母，量化差异单独报告。
- Notebook 04/05/06：独立 Rollout/Training 开关、JSONL/Manifest、Train 原始 ID/Prompt/答案核验、SFT 指纹和 Token/Mask 审计生效；新增筛查后的逐条 Rollout 持久化。预算可通过 `RL_MAX_NEW_TOKENS`/`RL_MAX_MODEL_LEN` 修改，默认仍为 2048/8192。GRPO 保持 G=8，Micro-batch=1、Gradient Accumulation=32；GPU 没有真实 Mixed Group 时不制造奖励、不执行 Update。
- 真实 Tokenizer 审计：33 条教师记录最长 398 Token，均含 Thinking 结束标记和 EOS，1024 上限没有截断教师 Completion。该检查不等于合成数据质量全面验证，也没有证明 24 条 Smoke 已教会规则。
- 现有合成产物重新独立验证 275/275 通过：Gravity 100、Equation 89、Bit 86；`synthetic_artifact_reverification.json` 逐条保留答案、轨迹、可识别性、来源状态与原文件 SHA256，没有修改或重生成既有数据。
- GRPO 真实执行 32 条、4 组 G=8 Rollout 与 Readiness，四组均 All-wrong，Mixed Ratio=0、Zero-Variance Ratio=1，更新次数 0。状态 `blocked_no_mixed_groups`，Rollout 约 456.57 秒，设备峰值 25841 MiB。全部截断的零奖励不代表正常完成后的真实正确率为 0。

GPU 阶段以每阶段状态 JSON 为准，不能把 Notebook 所有 Cell 执行结束视为训练成功。当前有效 DPO/PPO/GRPO Update、三种 RL 更新后重载、标准 RTX 4090 24GB 可运行性尚未通过。

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

## 第三轮教学颗粒度重构

- 六个 Notebook 保留原有公式、模型、数据、超参数和实验结果；仅按单一教学目标拆分代码单元，并在数据准备、核心算法、模型加载、训练调度、结果保存和测试之间增加 Markdown 说明。
- Notebook 01 将 Baseline 的 Chat Template、显存采样、单条结果记录、汇总保存和 vLLM 调度分开；合成数据按 Prior、Gravity、Bit、Equation、Screening、Leakage Audit、Data Designer 和 Pilot 导出分段。
- Notebook 03 将 Teacher Trace、Label Mask/Loss、LoRA/QLoRA、Manifest、训练调度和 CPU 测试分开；Notebook 04/05 分别拆分 Rollout、Adapter 加载、张量转换、训练更新和保存入口。
- Notebook 06 将 Rollout 审计、任务 Verifier、Reward/Batch、CausalLM Update、Adapter 交接和负样本测试分开，并修正第 9 节与第 10 节的教学顺序。
- 重构后所有 Code Cell 不超过 100 行；六个 Notebook JSON/AST 检查和顺序 CPU Smoke 均通过。GPU、vLLM 和真实 Qwen3 训练仍为 `Not Run`。

## vGPU Smoke 闭环修订（基于 b00cfb2ece）

- Baseline、Prompt、SFT、DPO、PPO、GRPO 的 GPU 入口都保留安全默认关闭状态；显式开启时先写入 GPU Preflight，捕获 OOM/异常并在 `finally` 释放模型和显存。阶段返回 `completed` 前必须确认预期产物存在。
- DPO/PPO/GRPO 均拆成可跨 Kernel 续跑的 Rollout/Training 阶段。Rollout 保存 JSONL 和阶段 Manifest；Training 只读取持久化 Rollout，不重新采样。训练阶段失败、无偏好对或无 Mixed Group 时返回阻塞状态并保留日志。
- SFT Stage 1 只保存中间 Adapter Manifest，不覆盖共享 `sft_manifest.json`；Stage 2/最终 Adapter 完成磁盘重载检查后才发布共享 Manifest。DPO/PPO/GRPO 从该 Manifest 平行初始化。
- PPO 的 Value Head 固定与基座设备对齐，输入 Hidden State 在 Value Head 前显式转换到 FP32；保存 PPO Adapter 时同时保存 Value Head，并在报告完成前实际重载。
- GRPO 保留每题 `G=8` 的组优势，在更新时使用 `GRPO_MICRO_BATCH_SIZE=1`、`GRPO_GRAD_ACCUMULATION_STEPS=32` 的微批次/梯度累积，避免把 32 条长回答一次性建图。当前 `policy_logprob_change` 仍不是 Reference KL，`KL_COEF=0`。
- 本地六 Notebook 的 JSON、nbformat、Code AST 与顺序 CPU Smoke 已检查；Code Cell 均不超过 100 行。目标机自报 RTX 4080 SUPER / 32760 MiB，虚拟化细节尚不能确认，不按 RTX 5090 预算推断。
