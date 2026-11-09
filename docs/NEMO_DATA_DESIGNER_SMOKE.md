# Notebook 01：首轮 NeMo Data Designer GPU Smoke

2026-11-07 完成 CPU 准备、真实 SDK 配置编译和模拟接口联调。2026-11-09 已完成
Qwen3-0.6B → 本地 vLLM → NeMo 的首轮真实生成，18 个输出全部留在 Preview，
详见 [真实验证报告](REAL_GPU_VALIDATION_20261109.md)。现有 Generator、Solver、
Verifier 的实现保持不变，三集合 Leakage Audit 只增加输出目录参数，以便把
NeMo 结果与 Seed 产物分开保存；后续 SFT/RL 算法不变。

当前默认新 Run ID：`nemo_qwen3_06b_gpu_smoke_20261109_r2_exact`，固定配置见
[nemo_smoke.json](../configs/nemo_smoke.json)，用于加固后的 Equation Trace 门槛。
旧 CPU Run `nemo_qwen3_06b_gpu_smoke_20261108_r1` 和首轮真实 GPU Run
`nemo_qwen3_06b_gpu_smoke_20261109_r1` 均保留，不在新源码下复用旧冻结清单。
已存在的 Run 不能重新 prepare 或覆盖 GPU 结果；新一轮必须重新注册。

## 测试内容与验证门槛

| 项目 | 本轮规则 |
|---|---|
| Seed | Numeral System / Gravity / Equation Numeric，各 3 条；共 9 条 |
| 来源 | 原有确定性 Generator，分布来自 Train；经 Solver、独立 Verifier 和 Train/Validation/Holdout 交叉审计 |
| 模型输入 | 只有 `seed_id / task_type / prompt`；不提供 Seed 答案、隐藏规则或教师轨迹 |
| 模型输出 | 两个 LLM 列 `prompt_variant / trace_variant`；9 行 × 2 列，预计 18 个输出 |
| 模型配置 | Qwen3-0.6B，`temperature=.3, top_p=.9`，Seed42；本轮关闭 Thinking，生成直接可检查的输出 |
| 上限 | Prompt 列 1024、Trace 列 2048 输出 Token；一次只调度一个生成任务 |
| 入库 | 新的独立 `data_designer/accepted.jsonl`；不自动合并到原始 Seed Accepted 或后续 SFT |

**Prompt Variant** 第一轮采用可严格证明的保守措辞变化：在完整原题前添加以下等价引导句之一，原题逐字保留：

- `Solve the task described below.`
- `Use the given examples to answer the query below.`

这保留原数字、运算符、单位、公式、示例及 Query。任意改数字、改操作符、增加约束/答案/否定指令、缺行或自由改写均拒绝。随后对 Variant 本身重新运行原 Solver 和独立 Verifier，检查唯一性、答案与 Seed 一致。这里不宣称正则能够证明任意自然语言改写等价；自由改写可以在后续单独扩展验证范围。

**Trace Variant** 使用闭合 JSON 证明，验证每一个字段，不接受仅有正确最终答案的未经核验推理：

- Roman：Query、每个规范 Roman component 的值、各部分的和与拼接结果均验证。
- Gravity：逐条检查 `g_i=2d_i/t_i²`、独立 OLS 的 `g`、Query 时间及 `½gt²` 的未舍入结果；中间量容差 `rel_tol=1e-4, abs_tol=1e-5`，最终答案使用现有官方数值容差。
- Equation：验证所选规则名称、normal/flip 与语义组约束，重新计算所有示例，再算 Query；独立 24-rule Verifier 必须确认答案唯一。保留操作符前后缀等答案格式。

2026-11-09 加固：Equation Trace 的示例计算、Query 计算和最终输出之间使用精确
字符串比较，保留前后缀与前导零；再用独立算术实现严格筛选与全部原示例相符的
规则及唯一 Query 输出。官方最终答案的 1% 数值容差只用于与 Seed 标准结果评分，
不能用于放过错误中间计算。例如 `mul_m1` 下 `42(21` 必须得到 `881`，不能因为
最终答案正确或官方最终评分允许近似，就接受中间值 `882`。此修复没有改动原有
Generator、Solver、Verifier 定义或 SFT/RL 算法；历史 GPU Run 及其空 Accepted
结果保留。正在运行的课程实验使用冻结源码，修复用于新的 Smoke Run。

JSON 外的自由叙述、Markdown 包裹、重复键、额外字段、未知/错误规则、错误中间结果、缺少字段，以及单独返回的未验证 `reasoning_content` 都只进 Preview。原始模型返回保留，不用正确的 Solver 轨迹替换错误模型轨迹后宣称通过。

本地 HTTP 记录器透明转发 NeMo 的请求至 vLLM，保留实际返回、`finish_reason`、usage 和请求耗时；长度截断或返回无法对应生成列，也只进 Preview。18 是请求的输出单元数量，不保证 18 个都通过；重试次数单独计数。

## 状态与产物

原有 `accepted_manifest.json` 已修复：Accepted 为零时 `status=empty_accepted`，`all_accepted_verified_and_audited=false`。NeMo 自己的 Accepted Manifest 同样受非空约束。

NeMo `gpu_result.json / summary.json`：全部 18 个完整输出都验证、审计通过才 `passed`；部分通过为 `completed_with_previews`；全部未通过为 `empty_accepted`；运行错误为 `failed_runtime`。`transport_complete` 单独说明是否收齐全部 Seed 及两个生成列。CLI 对非 `passed` 返回非零，产物仍保留供诊断。

目录为 `data/processed/tasksets/roman_gravity_equation_v1/synthetic/<RUN_ID>/`：

| 路径 | 用途 |
|---|---|
| 根目录原有 `synthetic_*.jsonl`、Accepted Manifest、Leakage/映射暴露审计 | 确定性 Seed 的生成、拒绝原因和审计，保留旧流程 |
| `data_designer/public_seeds.jsonl` | 提交给 NeMo 的 9 条无答案 Seed |
| `data_designer/private_seeds.jsonl` | 本地验证所需的标准结果及来源，不进入 LLM 输入 |
| `data_designer/prepared_manifest.json` | CPU 准备状态、Seed/源码/数据哈希；不是 GPU 成功报告 |
| `data_designer/api_check.json` | 真实 Data Designer 包编译检查，模型请求数为 0 |
| `data_designer/compiled_config.json` | 可审阅的 NeMo 配置；运行时重新绑定当前主机路径 |
| `data_designer/raw_model_outputs.jsonl` | GPU 后才生成的原始两列及 NeMo trace 元数据 |
| `data_designer/http_requests.jsonl` | GPU 后才生成的实际请求响应、截断标记、Token usage、耗时 |
| `data_designer/candidates.jsonl / accepted.jsonl / preview.jsonl` | 按输出单元分开的 18 条候选及验证结果 |
| `data_designer/synthetic_real_leakage_audit.* / numeral_query_exposure.*` | NeMo 候选沿用三集合及 Roman 映射暴露审计 |
| `data_designer/accepted_manifest.json / summary.json / gpu_result.json` | 验证通过数量、预览数量、状态及墙钟耗时 |

Roman 映射暴露沿用原来的 `report_only` 政策；Accepted 不代表在有限 Roman 映射层面没有重合。Validation/Holdout 只参与隔离审计，不用于生成分布、Seed 选择或推理。

## 明天的启动步骤

在克隆机仓库目录先 `git pull --ff-only`。CPU 控制环境与现有 vLLM CUDA 环境分开，避免互相改变 Torch/CUDA 依赖：

```bash
uv venv .venv-data-designer --python 3.11
uv pip sync --python .venv-data-designer/bin/python configs/nemo_smoke_requirements.lock.txt
```

本次已准备的 Run 随 Git 提交，**不要重新 prepare 同名目录**。先复核哈希；`api-check` 若已有结果则拒绝覆盖，本地已完成的结果可直接读取：

```bash
.venv-data-designer/bin/python -c 'from utils.run_synthetic_smoke import verify_prepared_run; from pathlib import Path; print(verify_prepared_run(Path("data/processed/tasksets/roman_gravity_equation_v1/synthetic/nemo_qwen3_06b_gpu_smoke_20261109_r2_exact"))["status"])'
```

在一张空闲 GPU 上启动专用本地 vLLM（先核对 `nvidia-smi`，不要与明天的训练同时驻留）。以下命令只生成启动命令文本；将打印的命令在独立终端执行，或直接使用下面的显式命令：

```bash
.venv-data-designer/bin/python utils/run_synthetic_smoke.py server-command \
  --vllm-python .venv-vllm-cu128/bin/python \
  --model /root/autodl-tmp/huggingface/models/Qwen3-0.6B

CUDA_VISIBLE_DEVICES=0 .venv-vllm-cu128/bin/python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/huggingface/models/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B --host 127.0.0.1 --port 18081 \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
  --max-num-batched-tokens 512 --kv-cache-memory-bytes 2147483648 \
  --enforce-eager --enable-chunked-prefill --generation-config vllm
```

另一个终端运行真实 NeMo 请求。脚本先检查本地 `/v1/models` 中模型名，再验证所有冻结输入；结果不覆盖：

```bash
.venv-data-designer/bin/python utils/run_synthetic_smoke.py gpu
```

此命令才会使用模型。它不会自行启动、关闭或替换已有 vLLM 服务；结束后关闭专用服务再运行 SFT/RL，避免两套模型同时占显存。2 GiB KV cache 和低并发是本轮配置，实际显存待 GPU 测量，不能由配置直接宣布低于 8 GB。

若要另开一轮，使用新 ID（准备和编译仍无模型请求）：

```bash
.venv-data-designer/bin/python utils/run_synthetic_smoke.py prepare --run-id nemo_qwen3_06b_gpu_smoke_round2
.venv-data-designer/bin/python utils/run_synthetic_smoke.py api-check --run-id nemo_qwen3_06b_gpu_smoke_round2
.venv-data-designer/bin/python utils/run_synthetic_smoke.py gpu --run-id nemo_qwen3_06b_gpu_smoke_round2
```

2026-11-09 新增整卡显存守卫入口，适用于服务器上尚未启动 vLLM 的情况。先用上述
`prepare` 和 `api-check` 注册新 Run，再执行：

```bash
.venv-data-designer/bin/python utils/run_nemo_guarded.py \
  --run-id nemo_qwen3_06b_gpu_smoke_20261109_r1 \
  --model /root/autodl-tmp/huggingface/models/Qwen3-0.6B \
  --vllm-python .venv-vllm-cu128/bin/python \
  --output .autodl/real_gpu_20261109/nemo_guard
```

该入口要求 GPU 空闲、端点未被使用、输出目录不存在。它自行启动并关闭本次专用
vLLM，约每 0.25 秒记录整卡占用，达到 8,000,000,000 字节即终止自己的进程组。
`guard.json`、`device_memory.jsonl`、`server.log` 和 `worker.log` 保存设备峰值及原始日志。
这仍是采样上界检查；不能证明采样间绝无瞬时超限。部分输出留在 Preview 时，生成
CLI 会非零退出，需结合 NeMo `summary.json` 判断内容验证结果，不能将其改记为通过。

Notebook 01 也保留显式 `RUN_DATA_DESIGNER_SMOKE=1` 入口；必须设置本轮新 `SYNTHETIC_RUN_ID` 并先准备，关闭 `RUN_SYNTHETIC` 和旧 Preview 标志。CLI 是首轮推荐入口，避免交互式执行重写旁路诊断文件。

## 已完成的本地验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-data-designer/bin/python -m unittest discover -s tests -p test_nemo_smoke.py -v
```

测试使用真实 Data Designer 0.10.0 SDK、真实 HTTP 请求链路和独立验证器，但上游模型响应由测试服务构造，**不是 Qwen3 GPU 结果**。覆盖 9×2 的列生成、顺序 Seed 读取、JSON 完整性、错误中间结果、语义/数字/符号变化、三集合泄漏、重复/缺失/截断、未验证 reasoning、空 Accepted 状态及不覆盖既有产物。
