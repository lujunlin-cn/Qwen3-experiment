"""Recompute descriptive metrics from the raw, independent hosted experiment."""
import hashlib
import json
from pathlib import Path
from collections import Counter

root = Path(__file__).resolve().parent


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


api = json.loads((root / 'api_smoke/summary.json').read_text())
result = json.loads((root / 'generation/result.json').read_text()) if (root / 'generation/result.json').exists() else None
candidates = read_rows(root / 'generation/candidates.jsonl') if result else []
requests = read_rows(root / ('generation/http_requests.jsonl' if result else 'api_smoke/http_requests.jsonl'))
metrics = {}
for task in ('numeral_system', 'gravity', 'equation_numeric'):
    subset = [r for r in candidates if r['task_type'] == task]
    prompt = [r for r in subset if r['variant_kind'] == 'prompt_variant']
    trace = [r for r in subset if r['variant_kind'] == 'trace_variant']
    metrics[task] = {
        'expected_prompt_outputs': 3, 'expected_trace_outputs': 3,
        'prompt_verified': sum(r['verification'].get('verified', False) for r in prompt),
        'prompt_accepted': sum(r['verification_status'] == 'accepted' for r in prompt),
        'trace_schema_pass': sum(r['verification'].get('schema_valid', False) for r in trace),
        'closed_json_math_verified': sum(r['verification'].get('verified', False) for r in trace),
        'full_trace_accepted': sum(r['verification_status'] == 'accepted' for r in trace),
        'trace_unverified_thinking_preview': sum('unverified_reasoning_content' in r['rejection_reasons'] for r in trace),
        'missing_required_thinking': sum('missing_required_thinking' in r['rejection_reasons'] for r in trace),
        'rejection_reasons': dict(Counter(reason for r in subset for reason in r['rejection_reasons'])),
    }
usage = {}
for model in sorted({r['model'] for r in requests}):
    subset = [r for r in requests if r['model'] == model]
    usage[model] = {'requests': len(subset), 'failed_requests': sum(r['status'] != 'completed' for r in subset),
        'output_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in subset),
        'input_tokens': sum(r.get('usage', {}).get('prompt_tokens', 0) for r in subset),
        'reported_reasoning_tokens': sum(r.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens', 0) for r in subset),
        'thinking_responses': sum(bool((r.get('response', {}).get('choices') or [{}])[0].get('message', {}).get('reasoning_content')) for r in subset),
        'latency_seconds': sum(r['latency_seconds'] for r in subset)}
detail = {'api_status': api['status'], 'formal_generation_status': result['status'] if result else 'not_started_api_gate_failed',
          'metrics': metrics, 'request_usage': usage, 'request_scope': 'formal_generation' if result else 'api_smoke', 'all_usage_output_includes_thinking': True,
          'closed_json_math_is_not_verification_of_free_thinking': True, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
write_new(root / 'quality_breakdown.json', detail)
lines = ['# NVIDIA Ultra 本机合成数据实测', '',
         '同一批冻结 9 Seed；Lightning 引导语 + Ultra Thinking/闭合 JSON；无本地 GPU 推理或训练。', '',
         f"API Smoke：`{api['status']}`。正式生成：`{detail['formal_generation_status']}`。", '',
         '| 题型 | Prompt 验证 /3 | Prompt Accepted | Trace Schema /3 | 闭合 JSON 数学 /3 | 完整 Trace Accepted | 未验证 Thinking Preview |',
         '|---|---:|---:|---:|---:|---:|---:|']
for task, m in metrics.items():
    if not result:
        lines.append(f'| {task} | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |')
        continue
    lines.append(f"| {task} | {m['prompt_verified']} | {m['prompt_accepted']} | {m['trace_schema_pass']} | {m['closed_json_math_verified']} | {m['full_trace_accepted']} | {m['trace_unverified_thinking_preview']} |")
lines += ['', '闭合 JSON 的数学通过不能认证任意自由 Thinking。Thinking 原文完整保留；未验证的完整 Trace 不进入 Accepted。Accepted/Preview 不自动并入 SFT。每类仅 3 Seed，描述统计不代表一般质量或显著提升。', '',
          '请求、响应、Thinking、全部 usage、逐候选验证与三集合泄漏审计见同目录原始文件；模型输入不含标准答案、隐藏规则或 Solver Trace。历史模型/Prompt 合同不同，不把接受率差归因于单一模型因素。', '']
with (root / 'RESULTS.md').open('x') as stream:
    stream.write('\n'.join(lines))
print(json.dumps(detail, ensure_ascii=False, indent=2))
