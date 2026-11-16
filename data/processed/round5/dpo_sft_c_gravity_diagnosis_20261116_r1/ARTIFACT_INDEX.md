# SFT C Gravity 专项诊断产物

真实GPU：26条Greedy；复用312条Sampling。权重更新0。

- `teacher_target_audit.jsonl`：实际R1 64条+C新增真实Train 10条，含独立Solver、序列化和Token Mask逐步骤检查。
- `teacher_audit_summary.json`：74条数学审计及真实训练Token数复现。
- `greedy/SFT_C_raw.jsonl`、`greedy/result.json`：26条新推理的原始回答、Token IDs及GPU资源。
- `greedy_scored.jsonl`、`sampling_diagnostic.jsonl`：原版Verifier逐条结果。
- `per_prompt_comparison.jsonl`、`comparison_summary.json`：逐题及分来源对照。
- `diagnosis_statistics.json`：按26题聚类的Bootstrap与配对分析。
- `reviewed_failure_cases.jsonl`、`failure_classification.json`：人工复核案例与可重叠失败分类。
- `registration.json`、`hypotheses.json`：Greedy前冻结协议及输入Hash。
- `cpu_tests.log`：GPU前54项通过；`cpu_final_tests.log`：离线修正后55项通过。
- `offline_analysis_amendment.json`：JSON tuple/list比较修正，未改Verifier或重生成。
- `attempts/offline_archive_comparison_v1/`、`failure_*.json`：完整保留首次离线失败与源码。
- `integrity_check.json`、`final_integrity_check.json`：历史、源码和权重隔离。
- `artifact_manifest.json`：最终文件及报告Hash。

最终报告：`docs/DPO_SFT_C_GRAVITY_DIAGNOSIS.md`。

7/26 Greedy答对，32/312 Sampling答对；两者完整证明认证均0。
74条教师数值正确且数学步骤均有监督，但实际最终通道全部只教授盒装答案。
先修复SFT输出合同及稳定OLS求解，再考虑DPO；本次未训练。
