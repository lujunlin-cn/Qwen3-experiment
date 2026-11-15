# NVIDIA Hosted 合成数据：本机控制与严格验收

2026-11-15。独立于历史本地 Qwen/vLLM Smoke；不升级 `data-designer==0.10.0`，不运行 GPU 训练。数学、生成合同与筛选实现直接保留在 Notebook 01，Runner 只编排 SDK、冻结输入、记录传输与归档。

## 审查结论

已审查 Notebook 01、`run_synthetic_smoke.py`、`round2_nemo_gpu.py`、`nemo_smoke.json`、历史 Smoke/质量修复文档，以及 2026-11-14 的 `NEMO_RESULTS.md` 和 `nemo_stage_metrics.json`。R0/A/B/C 的 Private Seed **逐字节一致**，每份 SHA256：

`a5d35bb596a2d21161d4dd3e7bba25666c29b0770857cc73e01cb889c64f25e8`

历史 72 次生成仅 1 条 Accepted。B 组 9/9 Trace Schema 合法，但数学验证 0/9；不能把接口/结构通过称为数学质量通过。历史文档中“等待 GPU”的文字为当时记录，新 Runner 不修改历史结论或文件。

新实验冻结原来的 9 道题：每类 3 条，预计 9 个 Prompt Variant + 9 个 Trace Variant。Private Seed 含答案/隐藏规则/参考轨迹，只在本地读取；公开 Seed 严格投影为 `seed_id, task_type, prompt`。没有 Solver context 或隐藏答案进入模型输入。

## 官方 Provider 与参数

使用 SDK 已安装源码 `data_designer/config/utils/constants.py` 的 `PREDEFINED_PROVIDERS`、`PREDEFINED_PROVIDERS_MODEL_MAP`。不创建重复的 `ModelProvider`。显式 `ModelConfig(provider="nvidia")` 绑定以下配置：

| Alias | Model ID | temperature / top_p | 额外参数 | 输出上限 |
|---|---|---|---|---:|
| nvidia-text | nvidia/nemotron-3.5-lightning-30b-a3b | 1.0 / 0.95 | `chat_template_kwargs.enable_thinking=false` | 256 |
| nvidia-reasoning | nvidia/nemotron-3-super-120b-a12b | 1.0 / 0.95 | `reasoning_effort=medium` | 8192 |

前三列及 thinking/effort 为 SDK 默认，输出上限/600 秒超时是本实验工程预算，不宣称 SDK 默认。并发 1；禁止对话重启和自动修复轮次。底层 SDK 的传输重试保持原配置，但每一次真实 HTTP 尝试均单独计数。

Endpoint：`https://integrate.api.nvidia.com/v1`。认证仅通过环境变量 `NVIDIA_API_KEY` 和 SDK 的 `EnvironmentResolver`；不允许 PlaintextResolver 把缺失的变量名当密钥。若本机 YAML 覆盖了 NVIDIA 内置端点/认证，立即失败，不替换为自定义 Provider。

Hosted 服务不提供可固定的权重 revision，本轮如实记录此限制、请求/响应模型 ID、参数、源码 Hash、Git Commit 与冻结输入 Hash，不虚构 model revision。

## 两个生成合同

**Prompt Variant**：Lightning 只生成一句引导语，不复制原题。为防止自然语言暗加条件，新函数 `verify_nvidia_introduction` 使用明确的安全句表；表外句子即使看似合理也保留 Preview，不声称程序能证明任意语言等价。Python 直接拼接 `raw_introduction + "\n" + seed["prompt"]`，不 trim、不改换行。完整题面必须逐字保留，再由原 Solver 重新求解和独立 Verifier 复核。原 `verify_smoke_prompt` 定义不变。

**Trace Variant**：Super 使用原来的 `smoke_trace_schema(task)`，每类独立编译 `LLMStructuredColumnConfig`，并在实际 HTTP 中显式携带 `response_format=json_schema`、`strict=true`。Roman 位值、Gravity 全部 g_i/OLS/Query、Equation 全部规则/示例/Query 均由原 `verify_smoke_trace` 独立复算。数值精度与官方最终评分规则不变。

原始 HTTP `message.content` 才是筛选输入；SDK 的字典、JSON 修复结果都不能替换原始模型输出。拒绝重复键、额外字段、缺字段、非标准常量、外围解释、错误中间计算和截断。

`reasoning_content` 保存为未经认证的模型原始输出；本轮保守沿用历史门槛，存在该字段的候选只进 Preview，即使最终 JSON 数学正确也不能宣称整段推理已通过。API Smoke 可通过结构兼容测试，但不代表这个候选具备入库资格。

每条候选输出 `Seed → Generator → Solver → Verifier → Screening → Accepted` 独立阶段布尔值与拒绝原因，运行原有 Train/Validation/Holdout Leakage Audit 与 Roman 映射暴露报告。Accepted=0 时必须 `empty_accepted`，不能 `passed`。`eligible_for_sft=false`、`merge_into_sft=false`。

## 执行方式

在仓库根目录使用已有 `.venv-data-designer`。环境变量应由调用者所在进程注入；请勿把值放入命令行参数、Notebook、配置、Git 或日志。Runner 不读取密钥文件，不要求启动 vLLM。

```bash
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py prepare
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py api-smoke
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py run
```

- `prepare`：纯 CPU，检查四组输入字节/9 道题、原数学与泄漏门槛；编译三种任务配置；模型调用数为 0。无需 Key。
- `api-smoke`：必须先有环境变量。1 次 Lightning，reasoning 模型对三类原 Schema 各 1 次，共 4 次预期调用。当前实现只让模型检查并返回最小字段结构对象，明确标为 `API_SCHEMA_ONLY`、`api-schema-<task>`，不要求求解原题。这些接口 fixture 不属于冻结 9 Seed 的正式输出、不计质量指标、不进 Accepted。分别记录每模型结果；失败时保存具体 HTTP 错误，禁止降级解析/自动换模型。只证明接口与结构能力，数学质量此阶段不验收。
- `run`：要求同一 Run 的两个模型和三个 Schema 都通过 API Smoke。按 Seed 顺序以 SDK 每次处理一行、生成两个列，共 18 个预期输出。单条生成/解析失败保留 Preview 并继续其他冻结 Seed，避免一条失败跳过剩余全量；运行错误仍使总体 `failed_runtime`，不冒充通过。HTTP 400/401/403/404/410/422 等明确 Provider/Schema 不兼容仍停止实验。记录实际请求、失败、重试和 usage。

所有阶段拒绝覆盖。重新尝试时须用新 ID，并重新 prepare/API Smoke：

```bash
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py prepare --run-id nvidia_hosted_20261115_r2
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py api-smoke --run-id nvidia_hosted_20261115_r2
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py run --run-id nvidia_hosted_20261115_r2
```

## 传输、隐私与产物

Runner 对 SDK 所使用的 HTTPX 传输层做窄范围观察，原 HTTPS 传输直接请求 NVIDIA；没有本地代理或重复 Provider。只保存请求 JSON 与响应 JSON/状态，不读取或保存认证 Header；异常与 SDK 诊断先脱敏。遥测包含每次传输尝试的 HTTP 状态、原始响应、`finish_reason`、完整 usage 和耗时；按模型分别汇总。`completion_tokens` 包含服务报告的 reasoning token，不混入 `prompt_tokens`；保留 usage 的嵌套详情，服务未提供时不臆测。

本机测试只能证明本机至 NVIDIA 的网络，不证明 AutoDL 至 NVIDIA 的网络。本轮不使用 AutoDL GPU。SDK 附带的外部产品遥测在 CLI 中关闭，仅保存本地实验遥测。

CPU 联调发现 SDK 0.10.0 的单行 Preview 在生成完成后，统计报告把 Token 标准差 `pandas.NA` 转为 float 时抛出 `DataDesignerProfilingError`。Runner 只对这个明确的 `TypeError/NAType` 报告错误保存 `profiling_warning.json`，不升级或改写 SDK，不重发模型请求，也不修复 JSON。SDK 数据不可用时保存空 SDK 输出，正式候选仍严格来源于原始 HTTP，并经过完整 Schema、数学、泄漏门槛。其他生成/Schema/统计错误继续失败。

目录 `data/processed/round3/<RUN_ID>/`：

| 路径 | 内容 |
|---|---|
| `seed_manifest.json`, `config.json` | 同批 Seed 核对、源码/Git/文件 Hash、模型参数 |
| `private_seeds.jsonl`, `public_seeds.jsonl`, `seed_checks.json` | 本地冻结输入、公开投影、本地数学/泄漏验证 |
| `prepared/<task>/compiled_config.json` | 三类 SDK 配置，prepare 不调用模型 |
| `api_smoke/` | 4 个接口探测配置、原始 HTTP、SDK 输出/日志、逐模型结果与 summary |
| `generation/http_requests.jsonl`, `raw_model_outputs.jsonl` | 正式生成的原始响应及传输记录 |
| `generation/candidates.jsonl`, `accepted.jsonl`, `preview.jsonl` | 每条候选验证/拒绝原因，Accepted 与 Preview 分离 |
| `generation/accepted_manifest.json`, `summary.json`, `result.json` | 质量、传输、阶段计数、usage、耗时分别记录 |
| `generation/synthetic_real_leakage_audit.*`, `numeral_query_exposure.*` | 复用原来的三集合与映射暴露审计 |

## 测试与真实 API 状态

CPU 回归命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NEMO_TELEMETRY_ENABLED=false CUDA_VISIBLE_DEVICES='' \
  .venv-data-designer/bin/python -m unittest discover -s tests -p test_nvidia_synthetic_smoke.py -v
```

测试使用真实 SDK 和数学检查，HTTPS 上游由测试构造。它不能证明 NVIDIA 模型可用、真实 Schema 支持、速度、Token 成本或生成质量。真实执行状态须读取本轮 `api_smoke/summary.json` 和 `generation/result.json`；尚未生成这些产物时，一律为未验证，不用 CPU fixture 冒充。

本次实执行 CPU 结果：新增 12 项测试全部通过（4.142 秒），原 NeMo/第二轮/守卫回归 11 项全部通过（6.385 秒）。包括 18 次模拟生成、完整 prepare→4 次 API Smoke→18 次生成的模拟状态流、单模型失败拦截、无 Key 零请求、原始 JSON 严格验证、错误中间结果拒绝、全部示例检查、原独立 Verifier 不调用 Solver、三集合泄漏、空 Accepted、不覆盖和密钥脱敏。原始日志与首次失败调试记录保留于 `data/processed/round3/nvidia_cpu_regression_20261115_r1/`。

## 2026-11-15 真实 NVIDIA API 结果

已用调用者提供的环境变量执行本机真实 `api-smoke`，Run 为 `nvidia_hosted_20261115_r1`，执行源码提交 `dbebae3`。实际仅 2 次请求，无重试；正式生成受 API 门槛阻止，**没有生成 18 条正式候选，也没有数据质量通过结论**。

| 模型 | HTTP | finish_reason | 输出 Token | 请求耗时秒 | 已确认与未确认 |
|---|---:|---|---:|---:|---|
| Lightning | 200 | stop | 6 | 0.955 | 模型/当前凭证/本机网络可用；返回 `Answer the following question.`，此为接口探测，不计 Accepted |
| Super | 404 | 无 | 服务未提供 | 0.383 | 服务端 `Not Found`，Function 对当前 account 不可用；不能据此断言模型支持或不支持 JSON Schema |

Lightning 输入 245 Token、总计 251 Token；服务报告 reasoning_tokens=0。Super 无 usage，不能记为成功的零 Token 生成。API 阶段墙钟 1.916 秒，未继续 Gravity/Equation 的 Super 探测，也未运行训练或本地 vLLM。

具体服务错误保存在 `api_smoke/http_requests.jsonl`；SDK 将其归类为 `ModelNotFoundError`。这可能涉及模型部署/账户访问，现有证据不能唯一归因。禁止把 404 改写成 JSON Schema 已支持/不支持，也不自动改用其他模型。该 Run 保留失败状态，不重用或覆盖；服务可用后按上面的 `r2` 命令重新注册。

本轮未执行正式生成，因此没有伪造 `generation/accepted.jsonl`、Preview 或 Accepted Manifest；这些由真正执行的 `run` 阶段产出。CPU fixture 的 Accepted 18/18 仅证明门槛机制，不能替代真实生成结果。实际产物与控制台日志见 [本轮目录](../data/processed/round3/nvidia_hosted_20261115_r1/)。

## 用户选定 Ultra 与 Thinking：独立后续 Run

用户随后明确允许 Ultra/DeepSeek V4 Pro，并最终选定 Ultra、要求一定保留 Thinking。独立探测实测：

- `nvidia/nemotron-3-ultra-550b-a55b`：HTTP 200，Roman 原 Schema 与数学证明通过；531 输出 Token，其中服务报告 485 reasoning Token。这里只是单题能力探测，不等于三类质量验收。
- NVIDIA 官方 V4 Pro 页面重定向到 `deepseek-v4-pro-0813`，实测 `deepseek-ai/deepseek-v4-pro-0813` 返回 HTTP 410，服务明确表示该部署已于 2026-09-14 到期。没有把 Flash 当成 Pro。探测原始证据另存 `nvidia_reasoning_availability_20261115_r1/`。

新增显式配置 `configs/nvidia_synthetic_smoke_ultra.json`，新 Run `nvidia_hosted_20261115_ultra_r1`。原 Super 配置与失败日志保留，不存在自动回退。Lightning 参数不变；Ultra 保持原 Super 实验的 temperature=1.0、top_p=.95、reasoning_effort=medium、cap8192，模型更换是显式实验变量，**不声称这些是 Ultra 的官方默认参数**。

此配置 `require_reasoning_content=true`，API Smoke 必须实际观察三类响应的 Thinking；仅仅请求 `reasoning_effort` 不足以通过。正式生成逐候选检查 Thinking，缺失则拒绝；存在但未经认证时完整 Trace 仍留 Preview。闭合 JSON 数学结果与完整 Trace 的 Accepted 状态分别统计，不因 Thinking 的存在放宽验收。

```bash
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py prepare --config configs/nvidia_synthetic_smoke_ultra.json
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py api-smoke --config configs/nvidia_synthetic_smoke_ultra.json
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py run --config configs/nvidia_synthetic_smoke_ultra.json
```

上述 Ultra r1 实际在 Gravity API 探测返回 HTTP 200、`finish_reason=length`：8192 输出 Token 全部为 Thinking，无最终 content。Roman Schema/Thinking 通过，Gravity 未完成，因此 Equation 尚未探测，18 条正式生成没有启动。这是预算截断，不能归因于 Schema 不支持。

据此注册 `configs/nvidia_synthetic_smoke_ultra_longthinking.json` / `nvidia_hosted_20261115_ultra_r2_longthinking`：**只提高 Ultra 输出上限到 32768**，Seed、提示词、temperature、top_p、effort、Verifier 不变。不覆盖 r1；新 Run 重新经过 prepare/API Smoke。

```bash
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py prepare --config configs/nvidia_synthetic_smoke_ultra_longthinking.json
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py api-smoke --config configs/nvidia_synthetic_smoke_ultra_longthinking.json
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py run --config configs/nvidia_synthetic_smoke_ultra_longthinking.json
```

长预算 r2 的 Gravity 请求实际在约 340 秒后被服务端断开，SDK 开始自动重试；控制器随后中断仍在等待的重试并保留 `interruption.json` 与原始日志，未运行正式生成。旧捕获器只记录已结束尝试，因此中断清单明确还有至少一次未观察到最终状态的请求，不伪造总耗时或 Token 数。新代码也记录 Python 可观察到的取消异常。

这次暴露出旧 API Smoke 求解完整算术并不符合“最小接口探测”的目标。当前已改为上述 Schema-only 检查，正式数学题面与 Verifier 完全不变。新配置 `configs/nvidia_synthetic_smoke_ultra_lowthinking.json` / `nvidia_hosted_20261115_ultra_r3_lowthinking`，保持 cap32768，只把 Ultra 的 effort 从 medium 改为 low；依旧必须实测非空 Thinking，不能以关闭 Thinking 加速。low 是明确的工程实验参数，不声称为 NVIDIA 官方默认。

```bash
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py prepare --config configs/nvidia_synthetic_smoke_ultra_lowthinking.json
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py api-smoke --config configs/nvidia_synthetic_smoke_ultra_lowthinking.json
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py run --config configs/nvidia_synthetic_smoke_ultra_lowthinking.json
```

本轮用户要求 Smoke 通过后完成全量：此处全量按冻结输入协议为 **9 Seed × 2 = 18 个正式输出**。技术探测不能替代这 18 条的 Solver/Verifier/Leakage/Accepted 质量验收。最小探测与 low-effort 支持加入后的 14 项 CPU 回归已实际全部通过（3.315 秒），原始日志为 `tests_minimal_probe_final.log`。

r3 的最小 API Smoke 已真实通过（21.574 秒），之后正式生成了 3 道 Roman 的两列和 Gravity 第一道引导语；Roman 闭合 JSON 数学 3/3 通过，但 Thinking 未认证，不能进入完整 Trace Accepted。Gravity Trace 随后再次发生服务端断开，等待中的重试被控制器中断；完整 18 条测试此时尚未完成。原始记录继续保留。

进一步核查 [Ultra 官方说明](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b.md)，明确提供的是 `chat_template_kwargs.enable_thinking=true/false`。此前沿用 Super 的 `reasoning_effort=low` 仅是请求参数，**没有证据证明它能缩短 Ultra 的 Thinking**；不把它描述为已验证有效的强度控制。

因此注册 `configs/nvidia_synthetic_smoke_ultra_full.json` / `nvidia_hosted_20261115_ultra_r4_full`：显式 `enable_thinking=true`、cap16384，温度/采样不变；Gravity 的新 Prompt 明确七位有效数字是有限中间数表示、Query distance 指最终两位小数舍入之前的计算值，避免为了“未舍入”反复长除法。此修改不更改原数学算法、全部示例要求、官方最终评分或中间 `rel_tol=1e-4, abs_tol=1e-5`。这是多个工程纠正后的新配置，不作单一变量因果收益主张。

```bash
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py prepare --config configs/nvidia_synthetic_smoke_ultra_full.json
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py api-smoke --config configs/nvidia_synthetic_smoke_ultra_full.json
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py run --config configs/nvidia_synthetic_smoke_ultra_full.json
```
