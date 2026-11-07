# Qwen3-0.6B 接入与基准运行说明

2026-11-07：根据课程模型缩小的决定，00–06 默认基座切换为 `Qwen/Qwen3-0.6B`。
保留官方 `enable_thinking=True` 聊天模板。该模型支持 Thinking / Non-thinking；本轮主线仍使用 Thinking。
已在端口 30122 的克隆机完成 150 题 GPU Baseline，以及 SFT、DPO、PPO、GRPO 的真实资源测试与保存后重载。
实测数据、配置及限制见 [0.6B 基准报告](QWEN3_0_6B_BENCHMARK.md)，以下保留安装与复跑说明。
8 GB 使用场景的最新代码与显存预算见 [低显存说明](QWEN3_0_6B_LOW_MEMORY.md)；该配置完成本地 CPU 验证，尚无新的 CUDA 峰值实测。

## 模型文件

远端项目：`/root/autodl-tmp/Qwen3-experiment`。
远端权重：`/root/autodl-tmp/huggingface/models/Qwen3-0.6B`。
固定 Hugging Face Revision：`c1899de289a04d12100db370d81485cdf75e47ca`。
BF16 权重文件 `model.safetensors` 为 1,503,300,328 bytes，SHA256：

```text
f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
```

下载工具需要 Python 3 与 curl，默认 4 路分段下载（可设 `--connections 1` 使用单连接）；下载全部 9 个模型/分词器/许可证/说明文件，校验 LFS SHA256 或 Git Blob SHA1，
并生成包含每个文件 SHA256 的 `download_manifest.json`。无 GPU 也可执行：

```bash
python3 utils/download_qwen3.py --destination /root/autodl-tmp/huggingface/models/Qwen3-0.6B --endpoint https://hf-mirror.com
```

其他机器默认下载到 `models/Qwen3-0.6B`，并可显式设置 `QWEN3_MODEL_PATH`。

## Notebook 配置与历史隔离

- 00–06 的 `QWEN3_MODEL_ID` 默认 `Qwen/Qwen3-0.6B`。优先识别项目下的 `models/<模型名>`，其次识别当前用户家目录下的 `autodl-tmp/huggingface/models/<模型名>`；路径优先级为 `QWEN3_MODEL_PATH` → `LOCAL_MODEL_PATH` → 自动目录 → Hub ID。00 的 GPU 阶段只接受已有本地目录，不临时下载。
- 00 默认 Run ID 为 `qwen3-0_6b_round1`，协议、校准和评测保存到 `data/processed/tasksets/roman_gravity_equation_v1/baseline/qwen3-0_6b_round1/`。已有同名阶段产物会拒绝覆盖。
- 数据版本、80/10/10 划分、种子 42、固定校准及评测 ID、Verifier 均沿用原协议。每类 3 条校准、50 条正式 Validation；Holdout 不参与预算选择。
- 02–06 的模型结果和 checkpoint 增加 `qwen3-0_6b` 目录层。SFT Manifest 的 Base Model ID 必须与当前模型一致；旧 4B Adapter 不能直接作为 0.6B 父策略。
- 00/01 当前三任务为 Numeral System / Gravity / Equation Numeric。02–06 仍保留旧版 Gravity / Equation Numeric / Bit 交互式数据入口，其全面迁移仍待处理。
- 旧 4B 数据、模型、结果及历史文档保留；旧实测显存、吞吐、准确率和 RL 状态均不代表 0.6B。

本次资源测试通过 `utils/benchmark_training.py` 显式绑定新版 Train 和统一 Verifier，未宣称 02–06 交互式数据入口已全面迁移。

## GPU 运行顺序

使用既有独立 vLLM Profile，无需改装训练环境：

```bash
cd /root/autodl-tmp/Qwen3-experiment
export QWEN3_MODEL_ID=Qwen/Qwen3-0.6B
export QWEN3_MODEL_PATH=/root/autodl-tmp/huggingface/models/Qwen3-0.6B
export LOCAL_MODEL_PATH="$QWEN3_MODEL_PATH"
export HF_HUB_OFFLINE=1
export BASELINE_RUN_ID=qwen3-0_6b_round2  # 本轮已完成 round1，复跑使用新 ID
```

仅 CPU 检查：

```bash
.venv-vllm-cu128/bin/python utils/run_baseline.py --phase off
```

使用新 Run ID，先核实设备及占用情况，再开始校准；已有结果禁止覆盖：

```bash
nvidia-smi
.venv-vllm-cu128/bin/python utils/run_baseline.py --phase calibration
```

原始 Numeral 预算为 512/1024，Gravity 和 Equation 为 4096/8192/16384；上下文统一为 20480。
只有本次 `budget_decision.json` 的 Numeral 为 `budget_cap_unresolved` 时，再执行：

```bash
.venv-vllm-cu128/bin/python utils/run_baseline.py --phase calibration_extension
```

随后在新进程中执行正式 150 题评测：

```bash
.venv-vllm-cu128/bin/python utils/run_baseline.py --phase evaluation
```

同样可在 Notebook 界面选择 vLLM Kernel，在配置 Cell 执行前设置对应 `BASELINE_PHASE`，再顺序运行全部 Cell。
不同阶段各用一个新进程/Kernel，确认自己的 GPU 占用释放后再开始下一阶段。
预算由当前模型重新校准；不复用旧 4B 的 `budget_decision.json`。
结果必须为 `completed`，并通过逐条 ID、Token 长度、输出重评分与 SHA256 审计；截断回答仍记失败。

## 当前验证边界

- 远端模型已完整下载：9 个文件，共 1,519,207,673 bytes；全部通过 Hub 校验。
- 接入阶段的真实 Config、Tokenizer 与 Thinking 模板在既有 vLLM / PT-SFT / Q-QLORA / PT-RL 四个环境离线检查通过（Transformers 4.57.1 / 5.22.1 / 5.18.0 / 5.18.0）；后续 GPU 实测的实际环境以基准报告和各阶段 Preflight 为准。
- 7 个 Notebook 的 JSON/AST 与完整顺序 CPU 检查通过；CPU 检查在独立副本中执行，模型推理与训练开关均关闭。
- 本机和远端 vLLM 环境的 00 CPU 检查均通过；3,905 条数据的既有划分及固定校准/评测 ID 保持一致。
- 7 个配置块均通过正常模型路径与错用 4B 路径的检查；0.6B 会拒绝旧 4B 的预算协议。

0.6B 的 GPU Baseline、vLLM 推理、SFT、DPO、PPO、GRPO 已实测完成，优化步数分别为 12/1/1/2；统一审计通过。
设备自报 RTX 4080 SUPER / 32760 MiB。DPO 的设备显存峰值达到 29578 MiB，不宣称标准 24GB 消费卡已验证完整流程；单轮资源 Smoke 也不证明训练后的能力提升。

接入阶段可审计记录：[验证报告](../data/processed/model_setup/qwen3-0_6b/validation_report.json)、[下载清单](../data/processed/model_setup/qwen3-0_6b/download_manifest.json)、[分词器兼容性](../data/processed/model_setup/qwen3-0_6b/tokenizer_profiles.json)。该阶段曾核对本地与远端的 7 个 Notebook 和 3 个工具源码 SHA256；后续基准阶段的来源另见运行 Protocol。
