# NVIDIA 合成数据 V2：Proof 重审、SFT 预览与 Gravity 修复准备

**离线 V2 重审已完成：15 Accepted、3 Preview；其中 6 条数学 JSON Proof Accepted，已转换成 6 条通过格式审计的 SFT 预览。** 当前只有 3 条 Equation 与默认任务池兼容，仍待来源及入库审核；Training Eligible=0，实际进入 SFT=0。Gravity P16K/P32K 仅完成代码、配置编译和 CPU Mock，本次没有真实 NVIDIA API 请求、模型加载或 GPU 训练。

实现提交：`f67a481`。原 R4 执行提交：`bf024db57c5eaf0c93de6e164fb2806f449a75d4`。原始 Run 的 **101 个文件逐字节 Hash 均保持不变**；V1 的 `screen_nvidia_records()` 及 Notebook 03 的默认逻辑没有改动。

## 1. V1 为何得到 0/9 Trace Accepted

V1 把含有 `reasoning_content` 的整个回答视为验收对象：即使最终 JSON 通过数学检查，也会因 `unverified_reasoning_content` 留在 Preview。9 条 Ultra 回答均含 Thinking，因此完整 Trace 全部被挡住。这个历史结果保留，不能重写为 V1 已接受 6 条。

V2 改变的是**验收对象**，没有改变数学标准：

- `reasoning_content` 原文单独保存在 `thinking_audit.jsonl`，记录存在性、完整 usage、reasoning Token，标记 `unverified / audit_only / allowed_in_sft_completion=false`。有无 Thinking 均不决定最终 JSON 的数学接受。
- 最终 JSON 必须与成功原始 HTTP 响应逐字一致，原记录 `verified_response_match=true`，HTTP 200，`finish_reason=stop`，内容非空；严格 JSON、无重复键、无额外字段、原 Schema、全部独立数学检查与三集合 Audit 均通过。
- 不补 JSON，不从 Markdown 提取，不去单位，不因末答正确跳过中间过程，不用模型自评。

新增 Notebook 01 函数 `verify_nvidia_trace_v2()`、`nvidia_wire_check_v2()`、`screen_nvidia_records_v2()`。数学检查复用原 `verify_nvidia_trace()`、`verify_smoke_trace()` 及原独立算术实现；字段诊断也调用原有验证辅助函数。Gravity 精度保持 `rel_tol=1e-4, abs_tol=1e-5`，官方末答评分不变。

## 2. 对原始 18 输出的逐题重审

输入来自原 R4 的 `generation/raw_model_outputs.jsonl`，同时核对原 `generation/http_requests.jsonl`、Private/Public Seed 和原配置。无模型重新生成，无原始输出改写。

新目录：[nvidia_hosted_20261115_ultra_r4_reaudit_v2](../data/processed/round3/nvidia_hosted_20261115_ultra_r4_reaudit_v2/)。每条候选的完整字段审计见 [candidates.jsonl](../data/processed/round3/nvidia_hosted_20261115_ultra_r4_reaudit_v2/candidates.jsonl)。

| Seed | Prompt | 最终 JSON Proof | 具体验证结果 |
|---|---|---|---|
| numeral_system-seed1 | Accepted | Accepted | Query=100；100→C；分量规范、求和、拼接、示例与末答通过 |
| numeral_system-seed2 | Accepted | Accepted | Query=74；模型分量70→LXX、4→IV；求和与LXXIV通过 |
| numeral_system-seed3 | Accepted | Accepted | Query=13；10→X、3→III；求和与XIII通过 |
| gravity-seed1 | Accepted | Preview | `finish_reason=length`；没有最终 JSON，不能完成过程验证 |
| gravity-seed2 | Accepted | Preview | `finish_reason=length`；没有最终 JSON，不能完成过程验证 |
| gravity-seed3 | Accepted | Preview | 全部 g_i、OLS、Query 距离、官方末答通过；`query_t="4.62s"`不等于合同要求`"4.62"` |
| equation_numeric-seed1 | Accepted | Accepted | 4 运算符覆盖；全局约束、20/20 示例、Query、唯一性通过；末答`>8` |
| equation_numeric-seed2 | Accepted | Accepted | 4 运算符覆盖；全局约束、20/20 示例、Query、唯一性通过；末答`2` |
| equation_numeric-seed3 | Accepted | Accepted | 4 运算符覆盖；全局约束、20/20 示例、Query、唯一性通过；末答`73` |

| 题型 | Prompt Accepted | Trace Accepted | Trace Preview |
|---|---:|---:|---:|
| Roman | 3 | 3 | 0 |
| Gravity | 3 | 0 | 3 |
| Equation | 3 | 3 | 0 |
| 合计 | 9 | 6 | 3 |

新协议为 `nvidia-final-json-proof/v2`，Verifier 为 `nvidia-independent-closed-proof-v2`。[accepted_manifest.json](../data/processed/round3/nvidia_hosted_20261115_ultra_r4_reaudit_v2/accepted_manifest.json) 包含原 Run、原 Git Commit、原 raw/HTTP/Seed/配置 Hash、新协议、题型/变体统计、Thinking、Leakage、映射暴露政策和 `merge_into_sft=false`。

`accepted_manifest.status=passed` 仅认证 15 条已接受子集；**`run_status=completed_with_previews`、`all_quality_passed=false`**，整个 Run 不是完全通过。空 Accepted 的回归仍要求 `empty_accepted`，不存在空集 passed。

## 3. 泄漏与来源限制保持

18/18 候选通过现有三集合完整题目审计；exact/normalized/source-id/source-row/structure 匹配计数均为 0。这不能扩张解释为不存在任何暴露。

- 全部候选 `lineage_status=unavailable`。旧 Seed 没有逐条可证明的训练数据来源链，不能补写 Train-only 声明。
- Roman `numeral_query_exposure_policy=report_only`、`claims_no_mapping_leakage=false`。V2 Accepted 统计含同题 Prompt/Trace，Roman 的6个输出单元对应3道题；映射审计中它们的 Query 均出现在真实集合的示例中，并与部分真实目标映射重合。
- 现有 Generator 代码中的 Train 分布逻辑与逐条历史来源证明是不同证据。前者不补齐后者。不能因为完整题目不重复就声称 Roman 没有映射暴露。

因此 Roman 不自动进入 SFT；Equation 也先作为待审核候选，来源链与明确入库决策未完成前，不授予正式训练资格。

## 4. Verified JSON → 现有 SFT Full Trace

Notebook 03 新增独立 Section **NVIDIA Verified Structured Proof → SFT Full Trace**。入口为 `render_nvidia_verified_records()`，通过 `validate_v2_manifest()` 校验 Manifest、所有候选 Hash、原 Run 不变、原 Private/Public Seed、真实 HTTP 对应和三集合指纹，再对每条 Proof 重跑 V2 数学检查。

只读取 `variant_kind=trace_variant / verification_status=accepted` 的6条；不读取 Prompt Variant 为新题，不读取 Preview，不将 Thinking 当教师回答。然后从完整 Public Prompt 调用 Notebook 01 的原 Solver，按真实返回结构构造内部 trace：

| 题型 | Solver 结果映射 |
|---|---|
| Roman | `solver["trace"]["query_number"]`、`components` |
| Gravity | `solver["trace"]` 内的 `solver / g_estimates / g_ols / query_t / query_raw` |
| Equation | `candidates_by_operator → candidate_rules_by_operator`，以及 `query_outputs / valid_assignment_count / selected_assignment` |

真实数据没有 Gravity Accepted，所以不生成 Gravity SFT 预览；Gravity 映射仅在 CPU 合成测试 fixture 中验证。没有把该 fixture 当成真实模型收益或正式样本。

调用原 `_format_trace(sample)`、`make_trace_record(sample)`、`audit_trace_record(record)`；格式和原训练输入保持一致，来源覆盖为 `nvidia_verified_structured_proof_rendered`，明确标记 `model_autonomous_cot_claim=false`。模型提供结构化证据，Solver 重新计算，Python Renderer 生成自然语言。Solver 的规范表达可能与模型分解不同，例如74渲染为50+10+10+4；没有伪装成模型原始 CoT。

Roman 预览样例（原 Renderer 输出）：

```text
1. Check the standard Roman rules against the public examples.
2. Decompose 74 into [{'arabic': 50, 'roman': 'L'}, {'arabic': 10, 'roman': 'X'}, {'arabic': 10, 'roman': 'X'}, {'arabic': 4, 'roman': 'IV'}].
3. Concatenate the place-value components.
Final answer: \boxed{LXXIV}
```

Equation 预览样例：

```text
1. Parse every AB<operator>CD example.
2. Keep only public 24-rule candidates consistent with each example: {'>': ['flip_sub_prefix'], '+': ['flip_add'], '`': ['flip_mul'], '%': ['join_swap']}.
3. Apply the normal/flip family and semantic-group constraints; valid assignment count is 1.
4. Apply the selected candidates to the Query; valid Query outputs are ['>8'].
5. Keep the answer only when the verifier finds one unique Query output.
Final answer: \boxed{>8}
```

产物位于 [SFT Preview 目录](../data/processed/round3/nvidia_hosted_20261115_ultra_r4_sft_preview_v2/)：

- `nvidia_sft_rendered_preview.jsonl`：6 条独立预览。
- `nvidia_sft_render_audit.json`：每条 Prompt 一致、V2 通过、Solver 末答、Boxed、无隐藏规则/原始 Thinking、确定性和原 Audit 检查。
- `nvidia_sft_render_manifest.json`：输入/输出 Hash、来源角色、任务池与训练资格分离记录。

本地缓存 Tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca` 可用，使用 `local_files_only=true`，未加载模型。实际 Token 数与空白分词 `token_length_proxy` 分开保存：

| Seed | Completion Token | 按现有 Chat Template 的完整序列 Token | 实际监督 Token |
|---|---:|---:|---:|
| Roman 1 | 54 | 114 | 60 |
| Roman 2 | 95 | 153 | 101 |
| Roman 3 | 92 | 157 | 98 |
| Equation 1 | 116 | 387 | 122 |
| Equation 2 | 116 | 384 | 122 |
| Equation 3 | 122 | 397 | 128 |

6条均通过原 `audit_trace_record()`，实际完整序列均不超过默认1024上限，未截断教师。带本地 Tokenizer 的独立重复导出得到**逐字节相同的预览文件**，监督 Mask 也实测核对。Completion 中的 `reasoning` 是 Python Renderer 叙述，绝不是 Ultra 的 `reasoning_content`。

## 5. 四种状态明确分开

| 状态 | Roman | Gravity | Equation | 合计 |
|---|---:|---:|---:|---:|
| 数学 JSON Proof Accepted | 3 | 0 | 3 | 6 |
| SFT 格式 Audit Passed | 3 | 0 | 3 | 6 |
| 与当前任务池兼容的待审核候选 | 0 | 0 | 3 | 3 |
| SFT Training Eligible（已解除来源/入库限制） | 0 | 0 | 0 | 0 |
| 最终进入 SFT | 0 | 0 | 0 | 0 |

`build_full_trace_records()` 的实际任务 tuple 仍为 **Gravity、Equation、Bit Manipulation**。未修改 `full_trace_records`、`USE_SYNTHETIC_DATA`、原 `synthetic_accepted.jsonl`、训练配置或 Checkpoint，也未改变 DPO/PPO/GRPO。

## 6. Gravity 定向修复：仅 CPU 准备

新配置：[nvidia_gravity_targeted_v2.json](../configs/nvidia_gravity_targeted_v2.json)。新 Runner：[run_nvidia_gravity_targeted.py](../utils/run_nvidia_gravity_targeted.py)。Notebook 01 的独立 `build_nvidia_gravity_targeted_v2()` 复用 R4 原 builder，仅添加 Query 合同：

`query_t` 必须是原始 Query 数值字符串、没有单位、保留原小数精度；Correct `"4.62"`，Incorrect `"4.62s"`。禁止生成后自动删单位。

| 条件 | Run ID | Prompt | Max Tokens |
|---|---|---|---:|
| P16K | nvidia_gravity_v2_20261115_p16k_r1 | 同一修复后 Query 合同 | 16384 |
| P32K | nvidia_gravity_v2_20261115_p32k_r1 | 与P16K完全相同 | 32768 |

模型均为 Ultra，temperature=1.0、top_p=.95、`enable_thinking=true`；三条 Gravity Seed、原闭合 Schema、数学 Verifier 和精度均相同。P16K 对 R4 可描述观察到的 Prompt 差异；P16K/P32K 比较预算因素。小样本随机生成不支持强因果或一般接受率结论。

两条件已独立编译，产物位于 [Gravity CPU 准备目录](../data/processed/round3/nvidia_gravity_targeted_v2_20261115_prepare_r1/)，各自 Manifest 均为 `prepared_cpu_only / models_called=0`。当前真实 API 支持情况、Latency、Token 与接受率全部 **Not Run**。仅 Mock 已验证 32768 返回 HTTP400 时保留错误、停止该条件，不偷偷回退。后续显式 live 命令将分别归档 raw HTTP/Thinking/usage/finish/Schema/Verifier/Accepted/Preview，禁止混合条件。

工程门槛为某个条件的 **3/3完整最终Proof**：全部示例g_i、OLS、原Query时间、未舍入距离、末答、HTTP对应和stop均通过，并实际保留Thinking。Thinking仍为unverified审计材料。通过也只说明冻结三题通过，不代表一般接受率100%。

## 7. CPU 实执行验证

| 测试 | 实际结果 | 秒 |
|---|---:|---:|
| 新 V2/Renderer/Gravity真实SDK Mock | 14/14通过 | 2.612 |
| 原 NeMo/第二轮/守卫相关回归 | 11/11通过 | 6.164 |
| 原 NVIDIA V1回归 | 15/15通过 | 5.647 |
| 本地Tokenizer计数/重复导出集成 | 2/2通过 | 4.035 |
| 合计 | **42/42通过** | — |

原始日志见 [CPU验证目录](../data/processed/round3/nvidia_v2_cpu_validation_20261115_r1/)。新V2测试用 socket guard 禁止网络；Tokenizer集成另外禁止CUDA初始化，强制HF离线。Gravity使用真实SDK和Mock HTTPS返回，未调用NVIDIA；原NeMo回归仅本地模拟HTTP服务器，不是vLLM。

覆盖：Thinking有无不影响合格JSON、严格Schema/重复键/外围文本拒绝、正确末答错误过程拒绝、截断/HTTP失败/伪造对应拒绝、query_t单位/精度、Verifier不调用Solver、三类映射、Boxed/确定性、Manifest篡改/覆盖拦截、空Accepted、三集合重复、Roman不进任务池、原Notebook Cell不变、两条件仅预算不同、真实Tokenizer监督Mask与无Key零请求。

## 8. Expansion Gate 与复现

明确Gate见 [nvidia_synthetic_expansion_gate_v2.json](../configs/nvidia_synthetic_expansion_gate_v2.json)：

- Roman：先处理映射暴露政策和任务范围，暂不扩展或自动入库。
- Equation：已有3/3闭合证明证据，可准备下一轮10–20条设计；实际扩展仍需可证明的Train-only来源、新Seed与独立Run、原Generator/Solver/Verifier和三集合Audit。本次尚未授予自动扩展资格。
- Gravity：先真实执行P16K/P32K，达到冻结3/3完整证明门槛再讨论扩展。
- Prompt Variant：9/9通过，但不是9道新数学题，不扩大以凑量。

本次没有100–500条扩展，没有重复旧Seed凑数量，也没有补写不明来源。未来扩展必须记录逐题失败与实际Accepted Rate，不以删失败样本提高分母口径。

以下离线命令已实际执行；目标目录存在时会拒绝覆盖：

```bash
.venv-data-designer/bin/python utils/run_nvidia_v2.py reaudit
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  .venv-low-memory/bin/python utils/run_nvidia_v2.py render
.venv-data-designer/bin/python utils/run_nvidia_gravity_targeted.py prepare
```

后续Gravity真实执行需要另行安排；本次没有运行`run`，也没有消耗新的NVIDIA推理Token。所有SFT产物都为独立预览，`merge_into_sft=false`。
