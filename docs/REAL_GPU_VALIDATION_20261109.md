# 2026-11-09 真实 GPU 验证

完整课程实验运行中：`course8h_20261109_r2_cuda_graph`。当前不能声称 SFT/RL
效果达标、已经收敛，或全部 rollout/训练都通过 8 GB 门槛。后续结果按实测补充。

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
