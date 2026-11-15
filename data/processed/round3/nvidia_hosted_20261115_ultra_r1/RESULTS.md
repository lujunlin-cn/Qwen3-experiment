# NVIDIA Ultra 本机合成数据实测

同一批冻结 9 Seed；Lightning 引导语 + Ultra Thinking/闭合 JSON；无本地 GPU 推理或训练。

API Smoke：`failed_api_smoke`。正式生成：`not_started_api_gate_failed`。

| 题型 | Prompt 验证 /3 | Prompt Accepted | Trace Schema /3 | 闭合 JSON 数学 /3 | 完整 Trace Accepted | 未验证 Thinking Preview |
|---|---:|---:|---:|---:|---:|---:|
| numeral_system | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| gravity | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| equation_numeric | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |

闭合 JSON 的数学通过不能认证任意自由 Thinking。Thinking 原文完整保留；未验证的完整 Trace 不进入 Accepted。Accepted/Preview 不自动并入 SFT。每类仅 3 Seed，描述统计不代表一般质量或显著提升。

请求、响应、Thinking、全部 usage、逐候选验证与三集合泄漏审计见同目录原始文件；模型输入不含标准答案、隐藏规则或 Solver Trace。历史模型/Prompt 合同不同，不把接受率差归因于单一模型因素。
