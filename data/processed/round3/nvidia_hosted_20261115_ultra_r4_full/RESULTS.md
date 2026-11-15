# NVIDIA Hosted 全量 18 输出单元质量实测

**测试已完成；整体数据质量未通过。** 9 Seed × 2 个生成任务均执行，18 次 HTTP 全部返回，无重试。

Run：`nvidia_hosted_20261115_ultra_r4_full`；执行源码 `bf024db`。Lightning 生成引导语，Ultra 显式开启 Thinking。

| 题型 | Prompt Accepted /3 | Trace Schema /3 | 闭合证明全项通过 /3 | 完整 Trace Accepted | 截断 |
|---|---:|---:|---:|---:|---:|
| numeral_system | 3 | 3 | 3 | 0 | 0 |
| gravity | 3 | 1 | 0 | 0 | 2 |
| equation_numeric | 3 | 3 | 3 | 0 | 0 |

共 9 Accepted（仅 Prompt）、9 Preview（全部 Trace）。9 条 Ultra 响应均含真实 Thinking；JSON 证明检查不能认证任意自由 Thinking，因此完整 Trace 不进入 Accepted。

Gravity 前两条达到 16384 输出 Token 上限、没有最终 JSON。第三条全部数值及官方末答通过，但 query_t 返回 `4.62s`，原合同要求 `4.62`，没有自动去单位或放宽规则。

正式生成耗时 1422.065 秒；仅输出 Token 总计 76226，平均 4234.778，包含 Thinking、截断与错误，不含输入。

原始 runtime 状态仍为 `failed_runtime`（两次截断触发 SDK 解析错误），quality 为 `completed_with_previews`。Accepted Manifest 的 `passed` 只认证 9 条已接受子集，不表示 18 条或整个实验通过。

三集合泄漏审计 18/18 通过；保留原映射暴露政策。每任务仅 3 Seed，结果只作描述，不能证明一般接受率提升。未运行本地 GPU 推理或训练，Accepted/Preview 均未并入 SFT。

逐条根因见 `per_output_diagnosis.jsonl`，完整统计见 `quality_breakdown.json`。输入、原始请求/响应/Thinking、Verifier、Manifest 和失败记录均保留。
