# SFT 增强实验基线与数据隔离审计

本轮 Run：`equation_sft_augmentation_20261116_r2`。起始提交 `a3f29dc81835a4136b14a6a2108e7f3e59e5fc51`。未跟踪的用户文件保留，未清理或提交。

## 真实 R1 身份

找到真实 Adapter：`.autodl/round2_20261114/setup/sft_r1_adapter/`，另有第一轮完整恢复备份。实际权重 SHA256 为 `d565b67a4e09d41b260b3929d5f3600b96e78ea8e7a8f3712dbfc749c190d371`，配置 SHA256 为 `1d6c4b00f50a4880d3f68967edf8e314026bb0df1de0b8d4aedf07fa3cd5962c`。二者均实际读取文件核对。

原始训练记录位于 `data/processed/benchmarks/qwen3-0_6b/real_gpu_20261109/recovered_course_r2/panels/sft.jsonl`，共192条，Roman/Gravity/Equation各64条，文件 SHA256 为 `8cb8bb0ead433931b938de6c60f6a0de3db2e6b5c81da9766417948ba7892d30`。真实 `result.json` 记录24次更新、22082监督Token、112.23秒。对应教师格式为自然语言 Full Trace 与 Boxed Answer；不是CPU `not_run` Manifest，也不是另一组24条实验。

基座 `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`。已补齐同revision基座文件，并与历史7个文件指纹逐一核对。权重SHA256：`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。

## 原始训练设置

NF4 QLoRA、r=8、alpha=16、dropout=0、七类线性投影、LR=2e-5、microbatch=1、累积8、max_length=4096、每轮完整192条。新增B/C均以同一个真实R1 Adapter在BF16基座上加载，使用相同BF16 LoRA设置；原始R1不更新。历史R1的NF4训练精度与本轮BF16增量训练精度区别明确记录，三组评测均使用同样BF16基座。

## 两套划分

真实R1协议明确绑定 `data/processed/tasksets/roman_gravity_equation_v1/dataset_manifest.json`，Train3123/Validation391/Holdout391。

- Train：`39f636ee4597f2851a69c959c1d078636eb1c10cc4823e36760152b488acdea7`
- Validation：`6f0267b11bd865c823c77538b559056646d749f62365bdc66da58be42bc53492`
- Holdout：`61219f80b68bb33712465a844bf277f02401a261d47ab678781c3ad87a732efc`

Notebook03当前默认的 `data/processed/dataset_manifest.json` 是包含Bit的另一套划分。即使Gravity/Equation各Split数量相同，逐行ID集合仍不同，不能混用。原始R1的192条ID/Prompt/Answer已逐条与实际Train核对。

20条旧Preview及本轮所有新候选均对两套Manifest的Train/Validation/Holdout重新交叉审计；正式效果面板只取真实R1对应的Validation。审计不调节生成器、不按Holdout重采样。结构相似只报告；不能据精确重复为零宣称不存在潜在家族暴露。

原始R1本身曾训练标准Roman映射，本轮Roman能力保持评测不构成映射OOD泛化证据。

机器可复核证据：本轮 `baseline_audit.json`、`preview_review.jsonl`、`equation_solver_audit.json`、`candidate_pool_freeze.json`、`sft_experiment_registration.json`。真实权重保存在Git外的`.autodl/`，Manifest与日志进Git。

## 固定R1教师质量的补充发现

对真实R1使用的64条Equation教师做当前精确审计：58条答案字符串与新的唯一Query输出完全一致，21条Trace包含至少一个不能精确解释全部示例的候选规则。旧答案仍可能按官方1%指标被判正确，这不代表中间证明正确。证据见`sft_r1_equation_teacher_quality_audit.json`。

这不改变R1模型身份；历史输入、Label与权重保持原样。B/C均使用新的精确合同，因此A vs增强组的差异同时包含增加训练、修复教师质量等因素。来源效应应主要比较同父权重、同预算的B/C；仍不能将单Seed试验泛化为全部数据源的因果结论。

原始官方答案64/64与新精确Solver一致；58/64指旧SFT教师真正输出的答案，六条差异由旧容差Solver引入，并非官方答案有错。该区分保留在逐题审计中，不能据此改动官方Label。
