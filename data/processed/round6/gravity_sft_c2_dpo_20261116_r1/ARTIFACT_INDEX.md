# Gravity C2 实验归档

最终状态：`no_preference_signal`，自动链已结束，当前没有训练或推理进程。

- 74条真实Train教师修复完成；实际最终回答证明74/74通过，全部数学Token参与监督。
- C2：2轮、20次真实Optimizer更新、107950监督Token；Checkpoint重载probe偏差0。
- C/C2 Greedy诊断26题：7/26 → 1/26；Tune48题：13/48 → 0/48。
- C2完整最终证明认证：诊断0/26、Tune0/48；不能宣称学会完整OLS。
- Readiness：208条初始+104条补充=312条，官方答案正确11，完整证明认证0，偏好对0；未达16对，未启动DPO。
- 未执行DPO重载、DPO Tune或Locked Final，不存在DPO收益。
- CPU实际97项测试通过。原C/R1、PPO/GRPO、历史实验文件及DPO Loss未改变。

最终报告：[DPO_SFT_C2_FINAL_REPORT.md](../../../../docs/DPO_SFT_C2_FINAL_REPORT.md)。

## 主要文件

| 文件 | 证据 |
|---|---|
| `registration.json` / `initial_state.json` | GPU前协议、来源、配置和保护Hash |
| `gravity_c2_teachers_74.jsonl` / `teacher_repair_audit.jsonl` | 逐条教师、实际序列化和数学审计 |
| `sft_train/` / `sft_reload.json` | 真实Loss、梯度、训练计数和重载 |
| `comparison/` / `C_vs_C2_*.json` | 原始评测、逐题比较和配对统计 |
| `rollout8/` / `rollout4/` | 全312条原始Completion、Token IDs和行为logprobs |
| `readiness*.json` / `preferences_*.jsonl` | 门槛与偏好筛选，空偏好文件是实际结果 |
| `checkpoint_archive_manifest.json` | 本地C2 Adapter/Optimizer文件Hash |
| `completion_hooks.jsonl` / `cpu_tests.log` | 自动阶段完成状态和CPU执行记录 |
| `final_status.json` / `final_integrity_check.json` | 最终停止状态与复核 |
| `source_before_repair/` | 修复前Notebook快照，保留历史行为证据 |
| `artifact_manifest.json` | 本次归档所有文件和报告Hash |

模型权重与Optimizer保留于`.autodl/gravity_sft_c2_dpo_20261116_r1/C2/`，Git提交仅含Hash与训练/评测记录。
本次失败结果如实保留；不自动增加Epoch、改变精度或降低Verifier标准。
