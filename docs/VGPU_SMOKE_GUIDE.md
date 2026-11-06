> 架构更新（2026-11-06）：Baseline 已移至 `00_Baseline_Diagnosis.ipynb`；合成 Verifier 与 `RUN_TRAIN_ARTIFACT_AUDIT=1` 入口位于可选 `01_Synthetic_Data_Engineering.ipynb`，新复核报告保存于 `synthetic/audited/`。本指南中的历史 GPU 产物继续保留，不代表本轮重跑 GPU。

# 第一轮 GPU Smoke：运行入口与诊断记录

本轮设备由平台标称为 RTX 4080 32GB vGPU。实际查询得到 `NVIDIA GeForce RTX 4080 SUPER`、32760 MiB、Compute Capability 8.9，驱动 595.71.05，`Virtualization Mode: None`。这些信息不能确认物理改卡或平台虚拟化细节，也不能用于宣称 RTX 5090 性能。

## 截断诊断

诊断固定使用三道官方 Validation 原始 Prompt，未加载 SFT Adapter，没有使用合成数据或教师轨迹。因此，这组基座模型的截断不能由合成数据训练造成。

| 对照 | Gravity | Equation Numeric | Bit Manipulation |
|---|---|---|---|
| 贪心，4096 Token | length，未结束 Thinking | length，未结束 Thinking | length，未结束 Thinking |
| 贪心，8192 Token | length，未结束 Thinking | length，未结束 Thinking | length，未结束 Thinking |
| T=0.6 / top-p=0.95 / top-k=20，8192 Token | length，第 7706 个索引处出现 Thinking 结束标记 | 7976 Token 正常停止，答案 23，错误 | length，未结束 Thinking |
| 同样采样参数，16384 Token | 8637 Token 正常停止且正确 | 7976 Token 正常停止，答案 23，错误 | 16384 Token 耗尽，仍未结束 Thinking |

这里的索引从 0 开始。样本 ID 分别为 `00ed1836`、`04171e29`、`01e09228`。Equation 官方答案为 32；独立 Solver 和独立 Verifier 均得到三个合法全局规则分配、唯一 Query 输出 32。模型只反转操作数，忽略输出反转与跨运算符约束，因此正常停止仍然可能答错。

2048 Token 对这组题明显不足；8192 也尚不能保证正常结束。16384 对照进一步确认 Gravity 是预算不足，但 Equation 正常结束仍有规则错误，Bit 仍存在未收敛的规则搜索。不能仅凭耗尽预算将其判为无限循环。8192/16384 采样对照保持同题、同种子与解码参数，但扩展了必要的引擎上下文容量；Equation 的实际输出 Token IDs 在两档预算中完全一致，Gravity 的 8192 Token 前缀一致。采样对照一次改变整套解码配置，不能据此单独归因于 Temperature 或 top-p。三题诊断不能估计总体准确率。

诊断产物位于 `data/processed/vgpu_smoke/token_budget*_diagnostic/`，包含原始输出、实际 Token IDs、结束原因和 Thinking 结束位置；跨预算 Token 前缀核对见 `token_budget_comparison.json`。`equation_validation_diagnostic.json` 保存数值规则核查。

SFT 的真实 Tokenizer 审计覆盖 24 条 Full Trace 和 9 条 Recall Drill，最长完整序列 398 Token，均保留 Thinking 结束标记与 EOS，没有被 1024 Token 训练上限截断。这只能排除明显的长度和结束模板缺陷，不能证明全部合成轨迹质量或训练效果。

现有 275 条 Accepted 合成数据实际重新通过 Notebook 01 的独立 Verifier：Gravity 100/100、Equation 89/89、Bit 86/86，包含答案、轨迹、可识别性和来源检查。报告 `synthetic_artifact_reverification.json` 保留源文件 SHA256 和每条状态；没有重新生成或改写数据。该检查受当前有限课程 Prior 与比赛数值容差约束，不能证明完整比赛覆盖，也不能把短模板轨迹视为模型已经掌握规则的证据。可用 `RUN_TRAIN_ARTIFACT_AUDIT=1` 独立复跑。

## 六阶段入口与通过条件

所有开关默认关闭。在各自环境启动 Kernel **之前**设置变量；修改环境变量后需要从配置 Cell 重新运行。Notebook 最后一个执行 Cell 是正式入口。不要仅依据 Kernel 正常退出或“所有 Cell 执行结束”判断训练成功，应检查阶段 `status` 与 `optimizer_steps`。

```bash
export QWEN3_MODEL_PATH="<已存在的本地模型目录>"
export LOCAL_MODEL_PATH="$QWEN3_MODEL_PATH"
export GPU_SMOKE_RUN_ID="<本轮唯一标识>"
export RUN_MODE=smoke
export HF_HUB_OFFLINE=1
```

| 顺序 | 环境与开关 | 固定 Smoke 规模 | 通过条件 |
|---|---|---|---|
| 01 Baseline | 独立 vLLM Profile；`RUN_BASELINE=1` | Validation 每类 3 条 | 9 条完整记录；单独报告截断，全部截断不能视为可靠能力基线 |
| 02 Prompt | 独立 vLLM Profile；`RUN_GPU_EVAL=1` | 相同 Validation 每类 3 条，P0–P5 | 54 条记录、同预算；输出长度不足则不选择“最佳 Prompt” |
| 03 SFT | Q-QLORA 训练 Profile；`RUN_SFT=1` | 24 条，1 Epoch，NF4 | 真实优化步、最终 Adapter 保存并实际重载、最终共享 Manifest |
| 04 Rollout | 独立 vLLM Profile；`RUN_DPO_ROLLOUT=1` | Train 每类 2 题，每题 4 条 | 16 条真实 Token IDs/Logprob，完整 JSONL 与 SHA256 Manifest |
| 04 Training | 训练 Profile；`RUN_DPO_TRAINING=1` | 同一份 Rollout | 有未截断 Chosen/Rejected；一次 Update；Adapter 重载 |
| 05 Rollout | 独立 vLLM Profile；`RUN_PPO_ROLLOUT=1` | Train 每类 2 题 | 4 条真实 Rollout，与同一个 SFT 指纹关联 |
| 05 Training | 训练 Profile；`RUN_PPO_TRAINING=1` | 同一份 Rollout | Value Head 前后向、Token/Value 对齐；有效完整回答；一次 PPO Update 与 Adapter/Value Head 重载 |
| 06 Rollout | 独立 vLLM Profile；`RUN_GRPO_ROLLOUT=1` | Train 每类 2 题，每题 G=8 | 32 条、每组恰好 8 条，完整行为概率 |
| 06 Training | 训练 Profile；`RUN_GRPO_TRAINING=1` | 同一份 Rollout | 独立 Verifier 与真实 Readiness；Mixed Group 存在才更新；Adapter 重载 |

每次只开启一张 Notebook 的一个 GPU 阶段。Rollout 完成后保存文件、退出该 Kernel/进程，用 `nvidia-smi` 确认自身占用已释放，再切换训练环境。不得终止其他任务。

RL 初始化统一读取 `data/processed/checkpoints/${GPU_SMOKE_RUN_ID}/sft_manifest.json`，并校验 Adapter 文件指纹；DPO、PPO、GRPO 平行初始化，不互相接续。Stage 1 中间 Adapter 不发布共享 Manifest。不能手动修改 Manifest 后跳过指纹检查。

当前训练环境没有 bitsandbytes。BF16 RL 可设置 `RL_QUANTIZATION_MODE=bf16_lora`；NF4 检查使用现有 Q-QLORA 环境，不在 Notebook 安装依赖。RL 默认采样 T=1 / top-p=1 / top-k=-1，便于核对原始 Softmax 行为 Logprob；不能套用 Baseline 的采样参数却仍宣称概率分布不变。

## 预算、产物与恢复

Baseline/Prompt 初始预算仍为 2048，可通过各 Notebook 的配置修改。RL 支持 `RL_MAX_NEW_TOKENS`、`RL_MAX_MODEL_LEN`，默认 2048 / 8192。增大输出预算时必须同时保证 Prompt Token 数加输出预算不超过上下文；不要只提高 `max_model_len`，它不会自动提高 `max_tokens`。

预算诊断可单独开启 `RUN_TOKEN_DIAGNOSTIC=1`，保持 `RUN_BASELINE=0`；`TOKEN_DIAGNOSTIC_SAMPLING=1` 使用模型随附采样配置，`TOKEN_DIAGNOSTIC_BUDGETS` 控制预算，`TOKEN_DIAGNOSTIC_LABEL` 指定新的产物目录。该分支只读固定 Validation，不扩大样本、不注入先验。

GRPO 的 G=8 不变；默认 Micro-batch=1、Gradient Accumulation=32。组优势先用完整组计算，再拆模型前向。CPU 的 8 成员数值等价测试不代表已完成 32 条真实长序列的 GPU Update。

每个 RL 阶段在 `data/processed/{dpo,ppo,grpo}/${GPU_SMOKE_RUN_ID}/` 保存状态、错误和中间文件。Rollout Manifest 保留文件 SHA、SFT SHA、实际采样参数和量化模式；Training 拒绝错版本、缺字段、非 Train 来源或伪造 Fixture。没有偏好对、全截断、没有 Mixed Group 均属于明确阻塞，不能改成 `completed`。

仓库日志归档仅规范化行尾空白与终端进度条换行；远端原始日志保留。`log_archive_manifest.json` 记录规范化前后的 SHA256，JSONL 原始推理记录与指纹没有改写。

当前实测 vLLM 约使用 25–26 GiB 设备显存；该分配包含预留 KV Cache，不能外推到标准 24GB 卡。训练峰值另见阶段报告，不能以参数量估算代替。所有时间仅适用于本轮设备、环境与样本，不承诺整套教程六小时完成。

## 实际结果与限制

本轮原始 Baseline 9/9、Prompt 54/54 在 2048 Token 下截断。旧提取器曾从 CoT 抽数值，使 Answer Extraction Success Rate 虚高；原始记录保留，单独离线重评分，不将截断输出作为完成答案。

第一轮 Two-stage NF4 SFT 实际完成 24 条训练，Stage 1/2 分别 5/8 个优化步，训练约 28.01 秒，设备峰值 5578 MiB，Adapter 重载通过。但检查发现预留 Recall 后再次顺序截取 Full Trace，使 Stage 2 缺少 Bit。代码已修复为 5/5/5 分层，CPU 回归通过；该历史 Adapter 不冒充修复后的训练结果。三种 RL 的第一轮 Rollout 仍统一来自这个原始 SFT Adapter，避免中途更换父策略。

DPO 16/16 回答截断，偏好对为 0，真实状态为 `blocked_no_preference_pairs`，更新次数 0。PPO 4/4 截断，状态为 `blocked_all_responses_truncated`，更新次数 0；真实模型的 BF16 和 NF4 Value Head 前后向均通过，机制探针不计 PPO Update。

PPO 比较同一批真实行为 Token：BF16 平均绝对 Logprob 差 0.011777、P99 0.131516；NF4 平均差 0.121896、P99 1.251460、最大差 3.951513。NF4 改变了策略分布，不能因工程门槛通过就称两种策略相同。行为概率分母始终保留原始 vLLM 值；后续严谨的同策略实验优先 BF16，量化训练作为单独对照。

GRPO 实际生成 32 条，四组均为 All-wrong，Mixed Ratio=0、Zero-Variance Ratio=1，状态 `blocked_no_mixed_groups`、优化步 0。Rollout 约 456.57 秒、设备峰值 25841 MiB；Readiness 实际执行，未伪造奖励。当前 2048 Token 截断使这些零奖励无法区分“未完成”与“完成但答错”，因此不能据此断言模型真实成功概率为 0。

三个现有环境均实际通过 `uv pip check`。vLLM 环境是 vLLM 0.11.0 / Torch 2.8.0 / CUDA 12.8 / Transformers 4.57.1；训练环境是 Torch 2.11.0 / CUDA 13.0 / Transformers 5.18.0 / PEFT 0.21.2，Q-QLORA 另有 bitsandbytes 0.50.2。环境目录后缀 `cu128` 不代表实际安装 CUDA 版本，`pip check` 也不证明与官方 Profile 的锁定版本完全相同。

尚未通过的项目包括有效 DPO/PPO Update、真实 Mixed Group 下 GRPO Update、这三种 RL 更新后 Adapter 重载，以及标准 RTX 4090 24GB 可运行性。当前未完成全规模训练，不能宣称六阶段训练闭环全部通过，也没有新增 RTX 5090 实验结果。
