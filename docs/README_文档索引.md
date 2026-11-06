# Nemotron SFT-RL 教程：文档索引

当前正式任务：**Numeral System（Easy）→ Gravity（Medium）→ Equation Numeric（Hard）**。模型为 `Qwen/Qwen3-4B-Thinking-2507`。

| Notebook | 职责 | 技术文档 |
|---|---|---|
| [00 Baseline Diagnosis](../00_Baseline_Diagnosis.ipynb) | 读取已清洗数据；80/10/10 划分、Manifest、泄漏审计、Validation 基线及错误分析 | [数据工程](01_数据工程.md) |
| [01 Synthetic Data Engineering](../01_Synthetic_Data_Engineering.ipynb)（完全可选） | Generator / Solver / 独立 Verifier / Screening、三集合交叉审计；保留历史 Bit 扩展及 NeMo Preview | [数据工程](01_数据工程.md) |
| [02 Prompt Engineering](../02_Prompt_Engineering.ipynb) | Prompt 对照 | [Prompt 工程](02_Prompt工程.md) |
| [03 SFT](../03_SFT.ipynb) | Direct / Joint / Two-stage，LoRA / QLoRA | [SFT](03_SFT.md)、[QLoRA](04_QLoRA.md) |
| [04 DPO](../04_DPO.ipynb) | 离线偏好与 DPO | [DPO](05_DPO.md) |
| [05 PPO](../05_PPO.ipynb) | Actor-Critic、GAE、PPO-Clip | [PPO](06_PPO.md) |
| [06 GRPO](../06_GRPO.ipynb) | Group Rollout、相对奖励及 Readiness | [GRPO](07_GRPO.md) |

00 的输入为维护者分发的 `data/processed/tasksets/roman_gravity_equation_v1/real_cleaned.csv` 和 `official_preparation_manifest.json`。
学生无需读取原始 CSV。官方 CSV 核验、分类、筛选和 Numeral 映射核验统一放在 [维护者工具](../utils/prepare_official_dataset.py)：

```bash
python utils/prepare_official_dataset.py
```

该命令只准备维护者数据，不执行划分、跨集合泄漏审计、模型推理或训练。00 的 Kernel 不导入该工具；它只通过磁盘读取准备好的数据。

00、01 的新产物均位于 `data/processed/tasksets/roman_gravity_equation_v1/`。02–06 的编号、名称和原有训练数据接口保留，尚未自动切换至新版三任务数据；不能将旧训练结果当作新版 Numeral 任务结果。

01 完全可跳过。现有 03 默认使用 `data/processed/real_train_traces.jsonl` 中的非合成真实 Train 轨迹，合成目录不存在也能执行；只有 `USE_SYNTHETIC_DATA=1` 才读取其配置路径中通过审计的 Accepted。本轮未改变训练算法或开展学生练习改造。

验证范围与产物说明见 [修改报告](../NOTEBOOK_REFACTOR_REPORT.md)。历史设计保留在 [设计记录](LeetCode_SFT_RL_Tutorial_Design_Record.md)；最新职责边界以本索引为准。
