# 第一个 Notebook 实现说明

交付文件为 `01_Baseline_Evaluation_and_Dataset_Preparation.ipynb`。Notebook 默认只执行本地数据主线和合成 smoke test，`RUN_BASELINE=False`，不会下载 Qwen3 权重或伪造 GPU 结果。

原始 `train.csv` 重新读取后得到 9,500 行、`id/prompt/answer` 三列，无缺失、重复 ID 或完全重复 Prompt。精确模板分类得到 Gravity 1,597、Bit Manipulation 1,602、Equation Numeric 732；Equation Symbolic 823 被单独审计并排除。Unit Conversion、Encryption 和 Numeral System 也保留排除原因。

真实数据按任务分层、固定种子 42 做 80%/10%/10% IID 划分。精确 ID 和 Prompt 没有跨集合重复，但数字归一化后的 Prompt 有跨集合重复；由于缺少可靠规则实例/结构家族键，Notebook 不把当前划分宣称为 Structural OOD。

Baseline 代码遵循官方独立 vLLM Profile：原始 Prompt 进入 Qwen3 官方 Chat Template，统一抽取与题型验证，保存逐样本 JSONL 和按任务/宏平均指标。只有在目标 RTX 5090 和 vLLM 环境就绪后才打开 `RUN_BASELINE`。

合成 Pilot 只使用综述已确认的 Gravity 公式和 8-bit DSL。Equation Numeric 的候选规则数学定义尚未在设计记录中给出，因此生成器明确阻断；Bit DSL 的答案/轨迹 smoke 验证通过，但没有完整规则搜索器时不会宣称全局可识别。

## RTX 5090 环境准备（2026-11-05）

目标机为 AutoDL `connect.westd.seetacloud.com:48510`，项目与环境均位于 `/root/autodl-tmp/Qwen3-experiment` 数据盘。uv 可执行文件位于 `.uv/bin/uv`，普通 PyPI 包安装使用阿里源 `https://mirrors.aliyun.com/pypi/simple`。

已准备的隔离环境：

- `.venv-pt-sft-cu128`：官方 `PT-SFT-CUDA` 依赖集合；实际 PyTorch 为 `2.11.0+cu130`。
- `.venv-q-qlora-cu128`：官方 `Q-QLORA` 依赖集合，含 `bitsandbytes 0.50.2`；实际 PyTorch 为 `2.11.0+cu130`。
- `.venv-pt-rl-cu128`：官方 `PT-RL-CUDA` 依赖集合；实际 PyTorch 为 `2.11.0+cu130`。
- `.venv-vllm-cu128`：官方独立 vLLM 依赖集合，`vllm 0.11.0`、`torch 2.8.0+cu128`。

四个环境均通过 `uv pip check`，并确认可以看到 RTX 5090。训练 Profile 的 cu130 是因为目标机可访问的镜像当前提供该 PyTorch wheel；vLLM Profile 保持官方 cu128 组合。Baseline GPU 推理尚未执行：机器上已有 CMS 任务占用显存，待该任务结束后再按 Notebook 的 `RUN_BASELINE` 开关执行 smoke 测试。
