# 第二轮代码一致性审查与修复

审查基线 `5059a3e`；修改仅用于新协议与 CPU 回归，历史 Run/Checkpoint 不覆盖。当前交付不宣称 GPU Phase C/D 已通过。DPO 按最新决定仅保留 Gravity 正确短答/正确长答 DPO-B。

## 1. 正式定义与实际执行路径

`utils/course_worker.notebook_api` 从 Notebook 中提取被明确列出的定义，不执行整本 Notebook 的 GPU/Smoke 单元。它没有运行所有展示代码。因此 Notebook 中的历史 standalone 默认值不能代表正式训练；正式路径在 Notebook 新增第二轮单元中明确标注。

| 算法/数据 | Notebook 真正供 Runner 使用的定义 | Worker 职责 |
|---|---|---|
| SFT | 03 `encode_chat_example`、`collate_causal_examples`、`course_sft_loss_sum` | 模型I/O、按窗口有效Token均值缩放、优化与保存 |
| DPO | 04 `gravity_preference_pairs`、`course_dpo_backward` | 显式B模式、冻结初始SFT Reference、顺序反传 |
| PPO | 05 `compute_gae`、`align_ppo_action_axis`、`ppo_update`、`CausalLMValueHead` | 行为概率/Value冻结、全局adv统计、恢复 |
| GRPO | 06 `group_advantage`、`grpo_clip_loss`、`equation_reward_components` | 完整组、分量/方差落盘、sequence mean调度 |
| 独立过程验证 | 01 `audit_equation_process`、`audit_closed_completion` 与原独立Verifier | 从原始输出重新验证，保留证据 |

新的数学/教学定义直接写入 Notebook，`.py` 只做统一执行、数据处理、统计与测试。Notebook 05 通过维护者 API 复用 Notebook 06 的同一 Reward 函数，避免两套权重漂移。

## 2. 缺陷和风险清单

Cell 编号以本轮修改后 JSON 数组下标/函数名定位为准；新增单元可能改变后部下标，函数名是稳定定位依据。

| 编号 | 文件/函数或Cell | 具体问题与原因 | 修复与回归 |
|---|---|---|---|
| C01 | `03_SFT.ipynb` Cell24 `LORA_CONFIG`；`course_worker.load_policy` | Notebook dropout=.05，Runner=.0且训练时禁用dropout | Notebook统一0；保存真实 adapter rank/alpha/dropout/targets，不能只记录拟用配置 |
| C02 | `04_DPO.ipynb` `course_preference_pairs` | `not correct or truncated` 把截断答当正式负例 | 明确排除 length；新 `gravity_preference_pairs` 只保留正确完整、必要证明通过的双侧B对；测试错误/截断/跨Prompt/未验证样本 |
| C03 | `course_worker.train_dpo`、Notebook04 `course_dpo_backward` | β只取函数默认，配置无法显式控制 | 显式传入settings.beta；β=.05/.1/.3顺序反传与Dense loss/梯度一致 |
| C04 | `05_PPO.ipynb` Cell3；`course_worker.notebook_api` | Notebook γ=.99/λ=.95，Runner隐式1/1；长序列终端回报传播不同 | Notebook正式默认1/1；Runner从settings注入，冻结默认前完成；手算GAE/Return与action轴测试 |
| C05 | 同Notebook `course_ppo_outcome` | 清零 full-sequence 最后一个 Value 的写法易误导 | 删除冗余写入。该位置是最后action之后的state，既有GAE/mask已阻断bootstrap；**没有证据表明这行造成第一轮数值错误** |
| C06 | 同Notebook standalone `_generate_rollout_records` | `min(max_tokens,context_remaining)`静默减少输出预算 | 超context直接报错，不改变计划cap；正式Runner原已拒绝overflow |
| C07 | PPO `ppo_update`；`course_worker.train_on_policy` | 非零kl_coef会把Old Policy漂移当Reference KL罚项 | 禁止非零未实现Reference系数；Old−Current仅诊断，不声称SFT KL约束 |
| C08 | PPO/GRPO `ppo_clip_loss`/`grpo_clip_loss` | 在log ratio差上clamp到±20是未披露的梯度目标改动 | 使用原始exp ratio；非finite直接失败；保持PPO clip本身，CPU解析梯度测试 |
| C09 | GRPO `course_efficiency_reward`/Worker | 第一轮全错只有0，GRPO组内优势为0 | 不伪造奖励；新增独立F/S/P/L与P0–P3，显式 `reward_ablation` 才启用；全对/全错/mixed数值验证 |
| C10 | `course_worker.train_on_policy` GRPO归约 | 第一轮跨回答按Token加权，Notebook读者易误认为每回答等权 | 第二轮显式sequence_mean；不同长度回答先各自token mean再回答mean；历史token_mean不冒充同一目标 |
| C11 | `course_worker.optimizer_for`/`save_training` | 原有Adam恢复存在，但缺少累计步骤与额外元数据 | 保留原恢复；新增累计steps、trainer/value Hash、真实LR、实际LoRA；禁止跨算法恢复，PPO有Adam恢复时必须有Value Head |
| C12 | `course_run.Experiment.execute` | 调度循环结束即写finished，即使final预算耗尽 | 根据report写`status=incomplete`、missing列表，仅complete写finished；预算模拟确认10个final缺失不会标完成 |
| C13 | Notebook01 `build_round2_data_designer` | SDK结构化列不必然发送HTTP约束 | B显式response_format；实际SDK模拟观察9个Trace携带Schema；旧无约束CPU快照保留但停用 |
| C14 | `run_synthetic_smoke.capture_local_requests`/`attach_transport` | 历史仅请求Hash；结构化SDK值可能被修复/过滤 | 保存实际本地请求；验证原始响应与SDK对象一致后筛查原始文本，禁止修复JSON掩盖错误 |
| C15 | `utils/check_low_memory.py` | CPU检查输出路径指向历史实验证据 | 新增显式--output，存在则拒绝，不覆盖旧报告 |
| C16 | `tests/test_course_experiment.py` tiny PEFT恢复 | PEFT会自动选择可见GPU，名义CPU测试可能分配CUDA | 恢复显式torch_device=cpu，正式测试命令CUDA_VISIBLE_DEVICES为空；失败尝试日志保留 |

## 3. 十项指定风险逐项结论

1. **Dropout**：确认不一致，已修复并记录实际配置。rank8/alpha16一致；若恢复的Adapter自身不同，必须依据实载配置拒绝/重新注册，不能只看Notebook常数。
2. **截断Rejected**：确认，已排除；第二轮DPO-B两边都必须正确完整且必要证明通过。
3. **PPO奖励完整性**：第一轮course路径只对`correct && !truncated`给长度分；新协议还要求stop，长度分必须exact trace通过；原始完整/错误/截断都保留，不剔除失败再算均值。
4. **GRPO全错**：第一轮确为零优势；新协议只允许验证过的格式/过程提供小权重区分，仍可能全零。
5. **GRPO长度门控**：第一轮只在正确未截断生效；第二轮再加必要Trace完整验证，错误/截断/缺步骤长度分0。
6. **Reference KL描述**：现有部分文字易混淆；统一为未实现、coef0，Old-policy drift不是Reference KL。
7. **Adam与Value Head恢复**：Runner原来就支持，并非第一轮每轮重置。新增强制完整性检查与两轮数值恢复测试；实际optimizer.step计数不能用history轮数替代。
8. **final预算状态**：报告原可显示incomplete，但state.finished语义不准确；已修正并加回归。突然断电仍可能留下stale running，不能假称具备断电事务保证。
9. **只执行Smoke的定义**：确实存在历史standalone/Runner两条路径。新偏好/Reward已显式接入Worker；Notebook04旧正确/错误偏好GPU入口停用。第二轮审核配置仍需按算法独立注册，旧八小时调度器不接受该Draft，不冒充已经跑通第二轮完整调度。
10. **分块Loss/梯度**：当前CPU检查在完整151936词表FP32/BF16投影、真实小Qwen/PEFT、padding/无padding、checkpoint on/off上通过；7种检查记录最大差值0。不是对任意CUDA kernel逐bit等价的证明。

## 4. 配置权威来源与数据核对

本轮权威来源是 [configs/round2_design.json](../configs/round2_design.json) 审核稿；其schema故意与第一轮不同、`gpu_execution_enabled=false`。审核后每个算法/消融复制成独立不可变执行协议，记录git commit、dirty/source hashes、base/tokenizer revision、data/panel hashes、checkpoint parent、真实采样参数与LR。

| 项 | 第二轮拟用正式值 |
|---|---|
| LoRA | rank8、alpha16、dropout0；base冻结 |
| SFT parent | 第一轮选中r1，同一个adapter Hash，BF16加载 |
| DPO | Gravity/B；β.1；LR2e-6；累积4；2epochs；4候选 |
| PPO | Equation；LR1e-5；累积8；γ/λ1；clip.2；Value系数.5；Entropy系数0 |
| GRPO | Equation；LR1e-5；G8；累积32；population std；sequence_mean |
| Reward | P0/P1/P2/P3独立Run；权重1/.05/.20/.05，按消融门控 |
| Reference KL | 未实现，0 |
| Rollout | T1/top_p1/top_k−1，固定cap16384；不自适应减cap |
| Eval | greedy，同题同模板/contract/cap；所有输出Token含Thinking |

原始Train/Validation/Holdout Hash不变。每个真实rollout仍保留prompt/completion IDs、原始behavior logprobs、完成状态、policy fingerprint与官方verifier版本。新增证明verifier单独版本，不能替换官方评分。DPO序列sum、PPO action轴、SFT Prompt label=-100与窗口token mean均保持显式。

## 5. CPU 执行记录

环境：本地Linux aarch64；训练数值环境Torch2.11.0+cu130、Transformers5.12.0、PEFT0.21.2；CUDA可见性禁用。NeMo环境Python3.11/data-designer0.10.0。它们与远端GPU环境不完全相同，Phase C必须重新记录版本。

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 修复前课程回归 | 11 passed | `before_course_tests.log` |
| 课程端到端CPU小Qwen/PEFT、四方法各两轮 | 11 passed | `course_tests_final_v2.log` |
| 新数学/配对统计/状态与第二轮Worker集成回归 | 11 passed | `numerical_tests_final_v3.log` |
| 原NeMo SDK/严格Verifier回归 | 7 passed | `nemo_tests_final_v2.log` |
| R0/A/B/C真实SDK+模拟HTTP，独立部分过程奖励 | 2 passed | `nemo_arms_tests_final_v2.log` |
| NeMo环境解释器隔离 | 1 passed | `nemo_guard_tests_final.log` |
| 完整词表/decoder分块数值检查 | 7 cases passed | `dense_chunked_checks.json` |

最终日志与版本/源码Hash见 [CPU验收记录](../data/processed/round2/round2_20261110_phaseab_v1/cpu_validation.json)。测试失败尝试未删除：测试fixture命名冲突、Schema未上HTTP、CPU自动选GPU等原因均在后续修复并重跑。测试程序提供的玩具样本/模拟回复只证明数学与工程机制，不进入真实训练数据或模型质量表。

## 6. Phase C 前仍需注册的工程工作

审核后按独立Run注册DPO-B、PPO/GRPO各P0–P3，冻结执行环境和父权重；给每阶段预留验证时间并检查剩余预算；首次Schema constrained decode与真实长序列反传必须通过显存守卫。Draft已准备好Notebook数学定义与Worker分支，但未执行新协议的GPU调度/模型重载，不能称全流程已验收。
