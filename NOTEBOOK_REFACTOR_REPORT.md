# Notebook 架构拆分与数据准备职责调整

保留正式任务 **Numeral System → Gravity → Equation Numeric**。原合并 Notebook 已拆为 [00 Baseline Diagnosis](00_Baseline_Diagnosis.ipynb) 与完全可选的 [01 Synthetic Data Engineering](01_Synthetic_Data_Engineering.ipynb)，02–06 的编号及名称保留。本轮未添加学生 TODO，也未改变训练算法或模型路线。

## 最终职责

| 文件 | 职责 |
|---|---|
| [utils/prepare_official_dataset.py](utils/prepare_official_dataset.py) | 维护者读取官方 CSV，完成字段/重复核验、分类、筛选及 Numeral 映射核验，分发已清洗数据和 `official_preparation_manifest.json` |
| [00_Baseline_Diagnosis.ipynb](00_Baseline_Diagnosis.ipynb) | 核验已清洗数据；80/10/10 三集合划分；ID、精确/规范化 Prompt、结构指纹和 Numeral 答案暴露审计；保存 Manifest 和报告；Validation Baseline 与错误分析 |
| [01_Synthetic_Data_Engineering.ipynb](01_Synthetic_Data_Engineering.ipynb) | 可选的三任务合成、独立 Verifier、Identifiability、Screening 和三集合泄漏审计；保留历史 Bit 实现和 NeMo Preview |

00 不读取原始 `train.csv`，不包含题型分类器、官方数据过滤或逐条官方映射清洗循环，也不导入维护者工具。Baseline 所需的 Roman Verifier 和机制测试仍留在 Notebook 中。

## 验证

- 维护者工具在隔离目录复跑，`real_cleaned.csv`、`classification_audit.csv`、`numeral_mapping_audit.csv` 与既有版本逐字节相同。
- 在没有原始 `train.csv` 和 `utils` 目录的独立 Kernel 中执行完整 00，通过；再次独立运行仍通过，所有既有产物内容不变。
- 3,905 条真实数据划分为 Train **3,123**、Validation **391**、Holdout **391**。三个 CSV、Dataset Manifest、泄漏报告、统计及 Baseline Protocol 与原版本逐字节一致。
- Baseline 校准与正式样本均来自 Validation，ID 不相交；Holdout 不参与预算选择。
- 00 保留的 **28 个** Baseline/划分/审计函数经 AST 对比不变（仅重命名运行变量）；迁出的分类器算法保持一致。
- 01 默认跳过和显式生成均在独立 Kernel 中通过；生成 15 条候选，Accepted 必须同时具备 Verifier 与泄漏审计通过状态。
- 现有 03 的非合成默认路径及显式增强路径另行回归。默认路径在合成目录完全不存在时运行通过。

验收详情与源文件 SHA256 见 [architecture_refactor_validation.json](data/processed/architecture_refactor_validation.json)。本次职责调整未改写已有数据集、划分、Manifest、Baseline Protocol 及 GPU 实验产物，仅更新架构验收记录；未重新运行 GPU 推理、训练或 NeMo Preview。历史结果继续以各阶段状态文件为准。

## 使用说明

学生直接运行 00；维护者才需要运行 `python utils/prepare_official_dataset.py`。新版输入及产物位于 `data/processed/tasksets/roman_gravity_equation_v1/`。

01 完全可跳过。现有 03 默认继续使用原有非合成真实 Train 轨迹，仅 `USE_SYNTHETIC_DATA=1` 时读取其配置路径中通过审计的 Accepted。02–06 仍保留历史数据接口，尚未自动切换至 Numeral 新版数据；本轮没有把旧训练结果标作新版结果，也未提前进入教学练习改造。

## 01 补充：Numeral Query 映射暴露审计

01 新增三集合双向映射暴露、相同 Query 目标重复和题内暴露审计；逐候选明细、Candidates / Accepted 汇总及报告哈希一并保存。映射有效但已暴露时只作报告，与 00 口径一致；无效映射或无法解析的候选拒绝。Validation / Holdout 目标答案仅供审计，不参与生成、Solver 或调参。

新 Pilot 位于 `synthetic/query_exposure_v1/`。15 条候选的 Prompt、答案、轨迹、隐藏规则和接受结果与此前 Pilot 一致。5 条 Numeral Query 均在三个真实集合的示例中出现；Accepted 合成示例覆盖 Validation 的 28/158 条真实 Query、Holdout 的 21/158 条真实 Query。该结果不能宣称映射层面无泄漏。

独立 Kernel 的默认跳过、显式生成，以及混合有效/无效候选、全部无效和无 Numeral 候选等情形均通过 CPU 检查。没有有效合成 Query 时暴露率为 `null`；汇总数量和比例与逐条 CSV 证据一致。生成算法保持不变，历史数据与 GPU 产物保留；验收记录见 [numeral_query_exposure_validation.json](data/processed/numeral_query_exposure_validation.json)。
