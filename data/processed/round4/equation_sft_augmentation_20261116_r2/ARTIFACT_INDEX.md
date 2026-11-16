# 实验产物索引

最终有效实验为r2；相邻r1目录为保留的排除实验，不得混入效果统计。权重和Optimizer保存在Git外`.autodl/`，文件身份与重载证据由Checkpoint Manifest绑定。

- [完整效果报告](SFT_SYNTHETIC_AUGMENTATION_REPORT.md)
- [基线与实际数据划分](../../../../docs/SFT_AUGMENTATION_BASELINE_AUDIT.md)
- [原20条Preview审核](PREVIEW_AUDIT.md) / [逐题审核](preview_review.jsonl)
- [384条Equation验收](EQUATION_SYNTHETIC_100_REPORT.md) / [Accepted](equation_accepted.jsonl) / [Rejected](equation_rejected.jsonl)
- [来源审计](equation_lineage_audit.json) / [完整数学审计](equation_solver_audit.json) / [分布审计](equation_distribution_audit.json)
- [B训练数据](sft_synthetic_110.jsonl) / [C训练数据](sft_real_control_110.jsonl) / [实际纳入清单](sft_actual_inclusion_manifest.jsonl)
- [A逐题评分](sft_A_validation/rows.jsonl) / [B逐题评分](sft_B_validation/rows.jsonl) / [C逐题评分](sft_C_validation/rows.jsonl)
- [Wilson、精确配对检验与Bootstrap](sft_statistical_analysis.json)
- [训练曲线](sft_training_curves.png) / [B训练](sft_B_train/result.json) / [C训练](sft_C_train/result.json)
- [Checkpoint文件Hash及Optimizer重载](checkpoint_archive_manifest.json)
- [99项CPU回归实测日志](final_cpu_regression_99.log)
- [最终完整性检查](final_integrity_checks.json) / [最终状态](RUN_STATUS_FINAL.json)
- [源码绑定](final_source_binding.json) / [产物SHA256目录](artifact_archive_manifest.json)

数学证明Accepted、SFT格式通过、训练资格、实际进入本次隔离训练四个状态分开记录；384条只有100条Equation实际用于B训练。Gravity10条保持原字节。Roman30条只是原题前缀变体，不计入数学新题，不进入训练。默认SFT入口不变。

自动离线链的权限失败日志保留；修正UID后重新执行CPU评分成功，未重新生成450条模型输出。详见finalization_recovery.json。
