# Gravity DPO-B：SFT C 完整实验报告

Run：`dpo_gravity_sft_c_20261116_r1`。最终状态：`failed`。

固定父 Adapter SHA256：`7033241d41efb086762df20789ce2b70764bb3a5c524e29b8d8f846d3254e3f4`。Base Revision：`c1899de289a04d12100db370d81485cdf75e47ca`。启动前全部 Base/Tokenizer 文件与历史 fingerprint 比对通过；Policy 和冻结 Reference 同为 SFT C，无回退。

## 数据与验收合同

训练题为原始16道 Gravity Train +10道保留 `synthetic_train_prior` 逐特征来源的合成题。原始 Prompt 不追加 JSON/答案/证明。独立 OLS 使用全部公开示例；最终自然语言证明中的 g、Query 时间、未舍入距离、物理关系和盒装答案按 rel_tol=1e-4、abs_tol=1e-5 检查；官方最终评分器保持不变。隐藏 Thinking 不参与证明验收，但计入总长度及 DPO Sequence Log Probability。不可解析或不完整回答记录为 unverifiable。Verifier 是保守的、有限语法的数学证书检查，未认证不等于数学错误。

Tune48和Locked Final100均复用历史冻结 Validation 面板；Locked Final不是官方 Holdout Test。未使用 Holdout 结果调整任何设置。两者可能被历史实验评估过，不能声称首次盲测。

## 实际 Rollout 与偏好数据

```json
{
  "status": "no_preference_signal",
  "rows": 208,
  "prompts": 26,
  "pairs": 0,
  "coverage_rate": 0.0,
  "official_correct_complete": 21,
  "natural_language_proof_passed": 0,
  "truncated": 0,
  "failure_reasons": {
    "official_final_answer_incorrect": 185,
    "final_answer_not_rounded_public_ols": 205,
    "full_trace_boxed_format_missing": 143,
    "missing_verifiable_physical_relation": 102,
    "all_example_ols_fit_not_demonstrated": 206,
    "fitted_g_missing_or_inaccurate": 206,
    "query_time_missing_or_incorrect": 108,
    "unrounded_query_distance_missing_or_inaccurate": 206,
    "incorrect_symbolic_arithmetic": 24,
    "unsupported_arithmetic": 73,
    "incorrect_intermediate_arithmetic": 84,
    "incorrect_numeric_equality_in_prose": 84,
    "altered_public_observation_in_proof": 31,
    "missing_or_multiple_numeric_boxed_answer": 70,
    "incorrect_reported_g_value": 56,
    "incorrect_additional_physical_relation": 1,
    "unsupported_math_syntax": 2,
    "empty_or_unclosed_final_section": 2
  },
  "pairs_by_prompt": {
    "c04634c0": 0,
    "2ce67ec9": 0,
    "8eecb224": 0,
    "6cfb1f29": 0,
    "d54f5b26": 0,
    "eba1891f": 0,
    "bdcccbab": 0,
    "ae5e65cf": 0,
    "0ab3d1de": 0,
    "736cdc47": 0,
    "42fee270": 0,
    "f19b29c6": 0,
    "76d51969": 0,
    "5fe169fc": 0,
    "16b45644": 0,
    "ff9540e2": 0,
    "gravity-final-001": 0,
    "gravity-final-002": 0,
    "gravity-final-003": 0,
    "gravity-final-004": 0,
    "gravity-final-005": 0,
    "gravity-final-006": 0,
    "gravity-final-007": 0,
    "gravity-final-008": 0,
    "gravity-final-009": 0,
    "gravity-final-010": 0
  },
  "proofs_by_prompt": {
    "c04634c0": 0,
    "2ce67ec9": 0,
    "8eecb224": 0,
    "6cfb1f29": 0,
    "d54f5b26": 0,
    "eba1891f": 0,
    "bdcccbab": 0,
    "ae5e65cf": 0,
    "0ab3d1de": 0,
    "736cdc47": 0,
    "42fee270": 0,
    "f19b29c6": 0,
    "76d51969": 0,
    "5fe169fc": 0,
    "16b45644": 0,
    "ff9540e2": 0,
    "gravity-final-001": 0,
    "gravity-final-002": 0,
    "gravity-final-003": 0,
    "gravity-final-004": 0,
    "gravity-final-005": 0,
    "gravity-final-006": 0,
    "gravity-final-007": 0,
    "gravity-final-008": 0,
    "gravity-final-009": 0,
    "gravity-final-010": 0
  },
  "total_completion_tokens": 209724,
  "thinking_included": true,
  "eos_included": true
}
```

初始每题8条；不足16对时全26题各补4条。Chosen最短、Rejected最长，双方均正确完整且独立证明通过，长度差>=64且>=长答20%，每题最多1对。所有错误、截断、不可验证回答均排除。

## 训练门槛与实际执行

实际 Optimizer Steps：0；未创建DPO权重，未执行Tune/Final模型比较。失败原因见下方和原始日志。

## 停止原因

RuntimeError('rollout4 failed: exit 1; see rollout4.log')

## 最终结论

未获得足够的严格正确长短偏好信号，本轮未达到真实DPO训练和效果验收目标。没有训练或评测结果可用于声称缩短输出、保持正确率或改进推理。

PPO/GRPO、默认SFT入口和SFT C保持不变；无Hosted API调用。失败日志、原始Completion/Token IDs/行为Logprobs、来源、偏好筛选及完成hook均保留。

CPU回归实测日志：`cpu_tests.log`。复现入口：`python utils/run_dpo_gravity_sft_c.py all`（当前Run已经存在，禁止覆盖；新实验需另行注册Run ID）。
