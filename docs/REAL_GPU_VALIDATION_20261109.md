# 2026-11-09 真实 GPU 验证

本轮验证已整理为 [实验验证总结](GPU_VALIDATION_SUMMARY_20261110.md)，以该总结为入口。
2026-11-10 已在无卡模式下取回原始产物，CPU 报告状态为 **incomplete**：GRPO
末轮因时间预算中断；独立最终面板仅留下 Base 2/24 的进度，没有完整结果。
四种方法均完成过真实参数更新，已测 rollout/train 峰值低于 8 GB；数学正确率
未证明改善。本页保留当时的准备、探针及中间结果，以下“运行中”是历史状态。

服务器报告 RTX 4080 SUPER，32760 MiB，驱动 595.71.05。采用整卡采样阈值
8,000,000,000 字节，约每 0.25 秒一次；这不是在 8 GB 实体卡上的兼容性证明，
也不能排除采样间的瞬时峰值。安装源为阿里镜像，三套隔离环境的实际版本见
[环境与探针证据](../data/processed/benchmarks/qwen3-0_6b/real_gpu_20261109/)。
原始 `train.csv` 的 SHA256 与本地一致；复用的七个 Qwen3-0.6B 模型文件均通过
既定版本哈希检查。与课程无关的旧 SGLang 源码和实验结果先完整备份到本地并
核验 SHA256，再删除旧目录、环境和无关模型缓存；清理明细见 `cleanup.json`。

## Notebook 01：NeMo Data Designer

Run：`nemo_qwen3_06b_gpu_smoke_20261109_r1`。真实 Data Designer 0.10.0 →
本地 vLLM 0.11.0 → Qwen3-0.6B，三任务各三条 Seed，收齐 18 个输出。

| 检查 | 实测结果 |
|---|---|
| HTTP 请求 | 18 次，0 次失败 |
| 输出 Token 总数 | 2743，仅输出 |
| 生成与验证墙钟时间 | 44.50 秒 |
| 含服务启动/退出的任务时间 | 71.09 秒 |
| 整卡采样峰值 | 3,808,428,032 字节 |
| Accepted / Preview | **0 / 18** |
| 内容验证状态 | **empty_accepted** |
| 合并到 SFT | 否 |

生成链路完整、显存采样通过，内容门槛未通过。Prompt 存在新增答案、漏题面、
改数字或添加分隔符；Trace 存在缺字段、错误 Roman 分解或其他未验证计算。
例如把 74 分成 `LII + XXII` 后输出 `LIIXXII`，即使各部分的数值之和是 74，
也不能作为规范 Roman 答案。最终答案碰巧正确但证明不完整的 Trace 同样留在
Preview。未放宽独立验证器或三集合审计，未将空 Accepted 改记为 passed。

首次专用服务启动未加载模型：启动器解析了虚拟环境 Python 的软链接，选中了
基础解释器。修复为保留虚拟环境路径，增加了实际执行解释器、检查 `sys.prefix`
的回归测试；首次失败及第二次真实运行的守卫记录均保留。

原始输出、逐项验证、泄漏审计与状态见
[该 Run 产物](../data/processed/tasksets/roman_gravity_equation_v1/synthetic/nemo_qwen3_06b_gpu_smoke_20261109_r1/)。

## 解码速度与显存探针

同一固定 Validation 题、Seed42、Thinking、1024 输出上限，两个独立进程。

| 配置 | 输出 Token | Token/s | 整卡采样峰值（字节） |
|---|---:|---:|---:|
| eager | 1024 | 47.98 | 5,141,168,128 |
| 单并发完整解码 CUDA Graph | 1024 | 256.68 | 5,195,694,080 |

1024 个输出 Token ID 完全一致；此探针约快 5.35 倍。只捕获 batch=1 的解码图，
额外图池约 0.05 GiB，不启用 Torch 编译。它不是解题效果证据，也不证明所有
输入的输出都等价；长序列、LoRA rollout 与反传另行接受完整阶段的监控。

此前 eager 全面板基线启动后因速度调整主动中断，日志与运行目录保留，不计为
已完成基线。新配置另开 Run，没有修改已注册运行的冻结面板或覆盖历史结果。

训练模式还修复了 Decoder checkpoint 的实际启用：反传期间保持 `train()`，
Dropout 模块单独关闭。回归测试验证 Decoder 确实重算且梯度与原计算一致。
课程测试 11 项通过，NeMo 测试 6 项通过，解释器隔离测试 1 项通过。

## 完整实验命令

### 20:20 中间结果

以下来自固定调参面板，每类四题，最终八题面板尚未评测。均使用 Thinking；
平均 Token 只计输出，包含错误与截断样本。完整过程仍在运行，不是最终验收。

| 题型 | 原始基线：正确率 / 截断率 / 平均输出 | 提示词先验 | SFT 第 1 轮 |
|---|---|---|---|
| Roman | 75% / 0% / 2137.50 | 100% / 0% / 1757.25 | 100% / 0% / 2225.50 |
| Gravity | 100% / 0% / 9407.25 | 50% / 0% / 7821.75 | 100% / 0% / 7882.75 |
| Equation | 25% / 25% / 12874.00 | 0% / 0% / 8696.00 | 25% / 0% / 9485.50 |

输出上限扫描 512/2048/8192/16384 均已完成，峰值约 5.20 GB。512 时三题型
全部截断且全部答错；2048 时 Gravity、Equation 同样全部截断且全部答错。
降低输出上限的效果不能算作模型学会简洁推理。

前两轮 SFT 均完成 192 条样本、22082 个 assistant Token、24 次优化器更新，
每轮纯训练约 112 秒，整卡采样峰值约 1.72 GB。实际 Tokenizer 检查确认所有
教师轨迹未截断，最长完整序列为 Roman 213 / Gravity 287 / Equation 269 Token。
训练记录确认 `training_mode=true / gradient_checkpointing_enabled=true /
use_cache=false`；第一轮 196 个 LoRA B 矩阵全部由零变为非零。

SFT 第 1 轮已经减少 Equation 的截断，不能再声称“只有 RL 才能改善截断”。
提示词先验在该小面板上也没有普遍提高正确率。原始中间记录见
`progress_20261109_2020.json`、`sft_token_budget.json` 和 `sft_update_audit.json`。

### 21:25 中间结果

SFT 共三轮达到预设调参平台期，选择第 1 轮，总耗时 1714.69 秒。三轮训练
损失为 2.751 / 2.029 / 1.396，分别进行 24 次更新；后两轮的验证正确率下降，
因此保留第 1 轮。训练最大整卡采样峰值约 1.724 GB。

DPO 同样三轮达到预设平台期，总耗时 3177.32 秒，选第 1 轮。rollout 峰值约
5.198 GB，真实训练峰值约 2.416 GB，均低于 8,000,000,000 字节。三轮共生成
96 个候选，只产生 9 对有效偏好、4 次优化器更新。每轮都保留了错误、截断
和未形成偏好的实际候选；没有补造正确样本。

选中 DPO 在调参面板上：Gravity 正确率 100%、平均输出 7750.25 Token；
Equation 正确率 25%、平均输出 10613.25 Token；两类均无截断。相较所选 SFT，
数学正确率未提高、数学输出更长，Gravity 输出略短，**尚未证明整体效果达标**。
独立最终面板仍未使用。PPO、GRPO 的显存与效果尚待实际运行。

DPO 总时间主要花在 rollout 和验证。各阶段明细、全部历史验证结果与所选权重
哈希见 `progress_20261109_2125.json`；当前不将调参平台期等同于充分收敛。

### 22:24 显存与训练信号进展

四种方法均已至少完成一次真实优化器更新；当前已完成任务的设备采样峰值如下。
GRPO 仍在继续，独立最终效果面板尚未评测，不能把本表当成全部实验结束。

| 方法 | rollout 峰值 GB | train 峰值 GB |
|---|---:|---:|
| SFT | 不适用 | 1.724 |
| DPO | 5.198 | 2.416 |
| PPO | 5.198 | 4.796 |
| GRPO（已完成部分） | 5.198 | 3.200 |

PPO 三轮达到预设调参平台期，总耗时 2512.66 秒。rollout 上限依次升级至
4096 / 8192 / 16384，第三轮获得 3/8 个正确且完整的候选。三轮真实训练
峰值分别约 2.389 / 3.102 / 4.796 GB；行为 logprob 对齐审计全部通过。
Math 调参正确率仍为 25%，按预设门槛选第 1 轮；该 checkpoint 的训练批次没有
正向正确性奖励，不能据此声称 PPO 学会了更准确地解题。

GRPO 第一轮组内奖励方差为零，正确返回 `no_group_signal`，没有伪造更新。
第二轮得到非零组内信号并完成真实反传，峰值 3.200 GB；后续轮次仍在运行。
完整中间记录见 `progress_20261109_2224.json`。

### 数学题可辨识性与 Trace 校验补充

重新按官方评分审计，全部 2943 条已接受教师记录仍通过原始标签评分，没有发现
官方评分意义下的错误教师答案。`3955` 与 `3956` 等差异属于官方 1% 数值容差，
不能仅按字符串不同将其判为标签错误；Solver 与独立 Verifier 在该例上一致。

现有 24 条规则与全局约束下，Train 的 586 道 Equation 中，406 道查询答案在
官方容差意义下唯一，另 180 道有多个官方评分也不等价的合法输出，原始答案是
其中之一。SFT 只使用前者；本轮已注册 RL 批次仍保留原 Train 分布，没有事后
删除困难题。调参面板的四道 Equation 中，有一道有多个合法输出。这里的结论
只针对现有公开规则集，不宣称覆盖所有可能的隐藏规则；最终面板未用于本次审计。
因此不能预设模型对全部数学题都能达到 100%。详见 `equation_identifiability_audit.json`。

监督中另发现并修复了 **NeMo Equation Trace** 的独立问题：旧 Trace 门槛把官方
最终答案容差也用在中间步骤，可能接受相差 1 的错误计算。现改为严格验证每个
中间输出及其与原例子的精确匹配，最终与标准结果评分仍沿用官方规则。CPU
反例确认旧版接受 `mul_m1` 的中间值 `882`，新版拒绝且保留正确结果 `881`；
NeMo 七项测试和课程十一项测试通过，Notebook 格式及每个代码单元不超过 100
行检查通过。原 Generator/Solver/Verifier 函数 AST 不变，SFT/RL 算法不变。
本地修复不会热改正在运行的 GPU 冻结源码；历史真实 NeMo 18 个输出仍全部
Preview，没有错误推理被晋升为 Accepted。证据见 `strict_trace_regression.json`。

按 [课程协议](QWEN3_0_6B_8H_PROTOCOL.md) 的固定面板、样本数、各阶段预算和
验证平台期规则运行。SFT 三题；DPO Gravity/Equation；PPO、GRPO 只训练
Equation，Gravity 仅评测；三种 RL 独立从同一个选定 SFT checkpoint 开始。

```bash
.venv-data-designer/bin/python utils/course_run.py prepare \
  --config configs/course_8h_cuda_graph.json \
  --run-dir data/processed/course_runs/course8h_20261109_r2_cuda_graph

.venv-data-designer/bin/python utils/course_run.py run \
  --config configs/course_8h_cuda_graph.json \
  --run-dir data/processed/course_runs/course8h_20261109_r2_cuda_graph \
  --model /root/autodl-tmp/huggingface/models/Qwen3-0.6B \
  --inference-python .venv-vllm-cu128/bin/python \
  --training-python .venv/bin/python --gpu 0
```
