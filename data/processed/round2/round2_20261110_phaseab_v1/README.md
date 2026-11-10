# 第二轮 Phase A/B 证据索引

本目录没有第二轮真实 GPU 结果。所有测试中的 Mock/随机小模型只验证机制，不作为模型准确率或 NeMo 产量证据。DPO 正式设计只保留 Gravity DPO-B。

- `nemo_diagnosis/`：历史18个真实输出的离线诊断、参考复算、严格重筛及0 Accepted Manifest。
- `panels/`：从官方 Validation 冻结的 Tune/final，排除第一轮每类12题；Holdout不调参。
- `nemo_arms_v2/`：有效 R0/A/B/C 配置；`nemo_arms/`是发现Schema未上HTTP前的CPU快照，已停用，不得用于GPU。
- `research/`：原报告正文快照、上游源码/日志带行号片段与来源Hash。
- `cpu_validation_r1.json`：当前单一复合奖励与Gravity公式诊断的最终CPU测试、版本与源码Hash；`cpu_validation.json`是前一版设计快照，保留历史。
- `*_final*.log`、`dense_chunked_checks.json`：通过的回归证据；其他attempt与failed日志保留失败历史。

旧8小时协议的source hash不会改写。GPU开机及人工审核后才能注册独立C阶段Run，不能直接用本设计文件调用旧串行调度器。
