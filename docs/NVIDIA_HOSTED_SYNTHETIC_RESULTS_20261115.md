# NVIDIA Hosted 合成数据全量质量测试：2026-11-15

**冻结的 9 条 Seed、18 个生成任务已全部执行和逐条筛查；整体数据质量未通过。** 最终为 9 条 Prompt Accepted、9 条完整 Trace Preview。没有把 Thinking、接口成功或正确末答自动解释成整段推理认证。

最终 Run：`nvidia_hosted_20261115_ultra_r4_full`。执行源码：`bf024db`。原始产物与离线复算见 [RESULTS.md](../data/processed/round3/nvidia_hosted_20261115_ultra_r4_full/RESULTS.md)、[逐条诊断](../data/processed/round3/nvidia_hosted_20261115_ultra_r4_full/per_output_diagnosis.jsonl)、[完整统计](../data/processed/round3/nvidia_hosted_20261115_ultra_r4_full/quality_breakdown.json)。

## 1. 全量结果

| 题型 | Prompt Accepted /3 | Trace Schema /3 | 闭合证明全项通过 /3 | 完整 Trace Accepted | Trace 截断 |
|---|---:|---:|---:|---:|---:|
| Numeral System | 3 | 3 | 3 | 0 | 0 |
| Gravity | 3 | 1 | 0 | 0 | 2 |
| Equation Numeric | 3 | 3 | 3 | 0 | 0 |
| 合计 | 9 | 7 | 6 | 0 | 2 |

18 次正式 HTTP 请求全部返回 200，失败 0、重试 0。16 条具有完整生成内容；Gravity 两条在 Thinking 阶段耗尽输出上限，没有最终 JSON。18 个输出单元均记录了 Verifier、Screening 与三集合审计结果，没有跳过难题或更换 Seed。

9 条 Ultra 响应全部实际包含非空 `reasoning_content`。Thinking 原文及服务报告的 reasoning Token 均保存。原 Verifier 只能认证闭合 JSON 的每个证明字段，不能认证任意自由 Thinking；依照本轮保守门槛，完整 Trace 全部保留 Preview。**这 0 条完整 Trace Accepted 不等于 9 条数学都算错；Roman/Equation 的 6 条闭合证明确实通过。**

Prompt Variant 由 Lightning 生成安全引导句，Python 原样拼接完整原题。9/9 通过引导语约束、题面逐字保留、Solver 重求解、独立 Verifier、答案不变与泄漏审计。Accepted 仅此 9 条，`eligible_for_sft=false`、`merge_into_sft=false`，没有自动进入 SFT。

## 2. Gravity 的具体失败位置

| Seed | finish_reason | 闭合 JSON | 程序诊断 |
|---|---|---|---|
| gravity-seed1 | length | 无 | 16384 输出 Token 用尽，SDK 无法解析最终证明 |
| gravity-seed2 | length | 无 | 16384 输出 Token 用尽，SDK 无法解析最终证明 |
| gravity-seed3 | stop | 有 | g_i、OLS、Query 距离、官方末答均通过；query_t 字段违反合同 |

第三条模型返回 `query_t="4.62s"`，原合同要求原数值字符串 `"4.62"`。末答 `163.35` 正确，未舍入 Query 距离也在原精度范围内；拒绝原因是字段值带了单位，不能笼统写成物理公式或数值全错。本轮没有自动去单位修复后冒充模型原答，也没有修改独立 Verifier。

中间值仍使用原 `rel_tol=1e-4, abs_tol=1e-5`，最终答案仍按官方评分。后续若修复，应单独测试生成提示中的 query_t 无单位要求、输出预算与计算表达方式；不得为了接受而改评分或跳过示例。

## 3. 模型、输入与耗时

| Alias | 实际模型 | 参数 |
|---|---|---|
| nvidia-text | nvidia/nemotron-3.5-lightning-30b-a3b | temperature=1.0、top_p=.95、enable_thinking=false、cap256 |
| nvidia-reasoning | nvidia/nemotron-3-ultra-550b-a55b | temperature=1.0、top_p=.95、enable_thinking=true、cap16384 |

使用 `data-designer==0.10.0` 官方内置 `nvidia` Provider、`https://integrate.api.nvidia.com/v1`。并发 1，认证仅从 `NVIDIA_API_KEY` 环境变量读取。Ultra 使用官方示例的 `chat_template_kwargs.enable_thinking=true`；不把 Super 的 reasoning_effort 参数描述为已验证有效的 Ultra 强度控制。

| 正式生成模型 | 请求 | 输入 Token | 输出 Token（含 Thinking） | 平均输出 Token | 服务报告 reasoning Token |
|---|---:|---:|---:|---:|---:|
| Lightning | 9 | 3068 | 65 | 7.222 | 0 |
| Ultra | 9 | 7865 | 76161 | 8462.333 | 75399 |
| 合计 | 18 | 10933 | 76226 | 4234.778 | 75399 |

平均输出包含错误和截断，不混入输入。正式生成墙钟 **1422.065 秒（23.70 分钟）**。之前的最小 API Smoke 为 4 次请求、28.934 秒、935 输出 Token；当前最终 Run 合计 22 次请求、77161 个已报告输出 Token。历史中断请求未返回的 usage 不计成零，也不混入本表。

9 Seed 与 2026-11-14 的 R0/A/B/C 逐字节一致，Private SHA256：

`a5d35bb596a2d21161d4dd3e7bba25666c29b0770857cc73e01cb889c64f25e8`

Public Seed 只有 `seed_id / task_type / prompt`。标准答案、隐藏规则和 Solver Trace 只用于本地检查，没有提交给模型。三集合 Leakage Audit **18/18 通过**，Train/Validation/Holdout 的 exact、normalized、source-id/source-row、结构匹配计数均为零；仍保留原 Roman 映射暴露报告政策。每类只有 3 Seed，不能外推一般质量或统计显著提升。Hosted 权重 revision 未由服务暴露，记录实际请求/响应模型 ID 与此限制，不虚构权重 Hash。

## 4. 状态、代码与 CPU 回归

原始状态保留：

- `all_seed_attempts_completed=true`，18 个输出单元全部尝试。
- `result.status=failed_runtime`：两次截断引发 SDK JSON 解析异常；其后仍继续其他 Seed。
- `quality.status=completed_with_previews`，`transport_complete=false`：两条截断不属于完整输出。
- `accepted_manifest.status=passed`：仅认证非空的 9 条 Accepted 子集；不是整体实验通过。

Accepted=0 的 CPU 回归仍要求 `empty_accepted`、`all_accepted_verified_and_audited=false`。没有把任一失败、部分结果或 Preview 改为整体 passed。

实际修改/新增：

- Notebook 01 追加两个 NVIDIA Cell，包含新 Prompt/Trace 合同、模型/列构造和筛选；原有 **94 个 Cell 完全不变**。
- `utils/run_nvidia_synthetic_smoke.py`：prepare、最小 API Smoke、正式生成、脱敏遥测、归档、单条失败后的全量继续。
- `configs/nvidia_synthetic_smoke*.json`：原 Super 与独立 Ultra 尝试分别冻结；当前使用 `nvidia_synthetic_smoke_ultra_full.json`。
- `tests/test_nvidia_synthetic_smoke.py` 与使用文档/索引；后续 SFT/RL 算法没有改动。

最新 CPU 测试：**新增 15/15 通过，5.287 秒；历史 NeMo/守卫/第二轮回归 11/11 通过，6.352 秒**。覆盖官方 Provider/模型映射、无 Key 零请求、脱敏、同批 Seed、公私投影、原题拼接、严格 JSON、错误数学、独立性、三集合泄漏、空 Accepted、不可覆盖、Thinking 门槛、逐条失败后仍全量继续。日志保留在 [CPU 回归目录](../data/processed/round3/nvidia_cpu_regression_20261115_r1/)。

SDK 0.10.0 的单行 Preview 统计报告有 `pandas.NA` 标准差错误。本轮仅对这个明确的后处理错误落盘 `profiling_warning.json`；模型请求由真实 SDK 发出，筛选始终使用原始 HTTP content，不利用 SDK 的解析/修复结果替换模型原答。其他错误保留并计入运行状态，没有升级依赖。

## 5. 历史尝试与复现

原 Super 返回 HTTP 404；独立 DeepSeek V4 Pro 探测返回 HTTP 410（服务声明该部署已到期）。随后按用户选择使用 Ultra。早期 Ultra 的 8192 截断、长请求断连、控制器中断和未完成的部分结果全部另存 Run，没有覆盖 2026-11-09/14 的历史数据。详细过程见 [接入与尝试记录](NVIDIA_HOSTED_SYNTHETIC_SMOKE.md)。不把不同 Prompt、Thinking、模型与预算下的接受率差解释为单一模型因素。

在有环境变量的进程中，以新 ID 复现同样的 9 Seed：

```bash
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py prepare --config configs/nvidia_synthetic_smoke_ultra_full.json --run-id nvidia_ultra_repeat_r1
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py api-smoke --config configs/nvidia_synthetic_smoke_ultra_full.json --run-id nvidia_ultra_repeat_r1
.venv-data-designer/bin/python utils/run_nvidia_synthetic_smoke.py run --config configs/nvidia_synthetic_smoke_ultra_full.json --run-id nvidia_ultra_repeat_r1
```

本次没有启动本地 vLLM、GPU 推理或 GPU 训练。当前交付是完整的合成数据质量测试与失败证据，不是 SFT 训练数据已全部验收。
