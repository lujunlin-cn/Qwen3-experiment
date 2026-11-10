# Nemotron SFT-RL 教程：文档索引

当前正式任务：**Numeral System（Easy）→ Gravity（Medium）→ Equation Numeric（Hard）**。默认模型为 `Qwen/Qwen3-0.6B`（保留 Thinking）。150 题准确率与 Token 预算、SFT/DPO/PPO/GRPO 资源测试的当前结果见 [0.6B 基准报告](QWEN3_0_6B_BENCHMARK.md)；下载、路径与运行命令见 [0.6B 接入说明](QWEN3_0_6B_SETUP.md)。历史 4B 实测仅作为历史记录。

最新真实验证见 [2026-11-10 实验总结](GPU_VALIDATION_SUMMARY_20261110.md)：四种方法均完成真实更新，已测 rollout/train 峰值低于 8 GB；数学正确率尚未证明改善，GRPO 末轮耗尽时间预算，独立最终评测未完成。无卡模式下已取回原始日志与 checkpoint；完整流程状态为 incomplete。实现原理与 CPU 验证见 [低显存说明](QWEN3_0_6B_LOW_MEMORY.md)。

下一轮统一执行入口见 [8 小时实验协议](QWEN3_0_6B_8H_PROTOCOL.md)：包含 max_tokens 基线、公开先验、SFT 及三种 RL 的多轮训练/验证、仅输出 Token 统计、显存守护和耗时报告。PPO/GRPO 只训练数学方程，Gravity 仅评测。

Notebook 01 的首轮真实 GPU 合成测试已另行准备：[NeMo Data Designer Smoke](NEMO_DATA_DESIGNER_SMOKE.md)。每题型 3 条 Seed、两个生成列，预计 18 个输出；CPU 准备和真实 SDK 模拟接口联调不代表 GPU 验收，未经验证的 Trace 仅作 Preview。

| Notebook | 职责 | 技术文档 |
|---|---|---|
| [00 Baseline Diagnosis](../00_Baseline_Diagnosis.ipynb) | 读取已清洗数据；80/10/10 划分、Manifest、泄漏审计、Validation 基线及错误分析 | [数据工程](01_数据工程.md) |
| [01 Synthetic Data Engineering](../01_Synthetic_Data_Engineering.ipynb)（完全可选） | Generator / Solver / 独立 Verifier / Screening、三集合交叉审计；保留历史 Bit 扩展及 NeMo Preview | [数据工程](01_数据工程.md) |
| [02 Prompt Engineering](../02_Prompt_Engineering.ipynb) | Prompt 对照 | [Prompt 工程](02_Prompt工程.md) |
| [03 SFT](../03_SFT.ipynb) | Direct / Joint / Two-stage，LoRA / QLoRA | [SFT](03_SFT.md)、[QLoRA](04_QLoRA.md) |
| [04 DPO](../04_DPO.ipynb) | 离线偏好与 DPO | [DPO](05_DPO.md) |
| [05 PPO](../05_PPO.ipynb) | Actor-Critic、GAE、PPO-Clip | [PPO](06_PPO.md) |
| [06 GRPO](../06_GRPO.ipynb) | Group Rollout、相对奖励及 Readiness | [GRPO](07_GRPO.md) |
| [07 Eight Hour Experiment](../07_Eight_Hour_Experiment.ipynb) | 三题型正式实验的冻结、8 小时调度、恢复与统一报告 | [8 小时协议](QWEN3_0_6B_8H_PROTOCOL.md) |

00 的输入为维护者分发的 `data/processed/tasksets/roman_gravity_equation_v1/real_cleaned.csv` 和 `official_preparation_manifest.json`。
学生无需读取原始 CSV。官方 CSV 核验、分类、筛选和 Numeral 映射核验统一放在 [维护者工具](../utils/prepare_official_dataset.py)：

```bash
python utils/prepare_official_dataset.py
```

该命令只准备维护者数据，不执行划分、跨集合泄漏审计、模型推理或训练。00 的 Kernel 不导入该工具；它只通过磁盘读取准备好的数据。

00、01、02 使用 `data/processed/tasksets/roman_gravity_equation_v1/`。03–06 的原有交互式训练接口保留；07 的课程 runner 显式绑定新版三任务 Train、轨迹及 Notebook 00 Verifier，调用 03–06 的核心目标与新增奖励/偏好定义。`utils/benchmark_training.py` 保留为历史冒烟资源测试入口，不能用于判断多轮收敛或学习效果。

01 完全可跳过。现有 03 默认使用 `data/processed/real_train_traces.jsonl` 中的非合成真实 Train 轨迹，合成目录不存在也能执行；只有 `USE_SYNTHETIC_DATA=1` 才读取其配置路径中通过审计的 Accepted。07 则使用新版 taskset 内已经验证并冻结的真实轨迹，不需要重新执行 01。

验证范围与产物说明见 [修改报告](../NOTEBOOK_REFACTOR_REPORT.md)。历史设计保留在 [设计记录](LeetCode_SFT_RL_Tutorial_Design_Record.md)；最新职责边界以本索引为准。
