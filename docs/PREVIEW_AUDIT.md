# 原有20条SFT Preview逐条审核

本轮重新执行公开Solver、独立Verifier、来源审计、两套Manifest交叉泄漏审计、现有SFT格式审计和真实Qwen3-0.6B Tokenizer。审计证据位于 `data/processed/round4/equation_sft_augmentation_20261116_r2/preview_review.jsonl`。

Gravity 10/10通过，保持历史JSONL记录和Completion字节不变。Equation 10/10数学、来源、格式均通过；旧Completion虽无错误规则，但仅简述候选和Assignment计数，没有逐例展示。正式增强候选使用新精确Solver重新计算，逐一展示示例和全部Assignment。旧文件不改写。

| Candidate | Task | Audit |
|---|---|---|
| gravity-final-001 | gravity | passed |
| gravity-final-002 | gravity | passed |
| gravity-final-003 | gravity | passed |
| gravity-final-004 | gravity | passed |
| gravity-final-005 | gravity | passed |
| gravity-final-006 | gravity | passed |
| gravity-final-007 | gravity | passed |
| gravity-final-008 | gravity | passed |
| gravity-final-009 | gravity | passed |
| gravity-final-010 | gravity | passed |
| equation_numeric-final-002 | equation_numeric | passed |
| equation_numeric-final-004 | equation_numeric | passed |
| equation_numeric-final-005 | equation_numeric | passed |
| equation_numeric-final-006 | equation_numeric | passed |
| equation_numeric-final-008 | equation_numeric | passed |
| equation_numeric-final-009 | equation_numeric | passed |
| equation_numeric-final-010 | equation_numeric | passed |
| equation_numeric-final-011 | equation_numeric | passed |
| equation_numeric-final-014 | equation_numeric | passed |
| equation_numeric-final-016 | equation_numeric | passed |

默认SFT入口未合并；110条专用实验数据与默认数据隔离。Token长度均由本地真实Tokenizer计算，按R1的4096完整序列上限验证，拒绝截断教师。
