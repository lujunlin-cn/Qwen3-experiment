"""Reproducible tensor accounting and explicit planning allowances, not a GPU test."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/processed/benchmarks/qwen3-0_6b/low_memory_20261107'
GIB = 2**30


def main():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config_path = BASE / 'model_config.json'
    config = json.loads(config_path.read_text())
    expected = json.loads((ROOT / 'data/processed/model_setup/qwen3-0_6b/download_manifest.json').read_text())['files']['config.json']['sha256']
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == expected
    with torch.device('meta'):
        model = Qwen3ForCausalLM(Qwen3Config.from_dict(config))
    parameters = model.num_parameters()
    tied = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    manifest_path = ROOT / 'data/processed/checkpoints/qwen3-0_6b/qwen3_06b_baseline_rl_20261107/sft_manifest.json'
    lora = json.loads(manifest_path.read_text())['lora']
    assert lora['bias'] == 'none'
    adapter_parameters = sum(lora['r'] * (module.in_features + module.out_features)
        for name, module in model.named_modules()
        if name.rsplit('.', 1)[-1] in lora['target_modules'])
    # Clones need only the config/manifest; verify the count against saved tensors
    # when the locally retained adapter checkpoint is available.
    lora_path = manifest_path.parent / 'sft_direct_nf4_qlora/adapter_model.safetensors'
    if lora_path.exists():
        from safetensors import safe_open
        with safe_open(lora_path, framework='pt', device='cpu') as f:
            assert adapter_parameters == sum(f.get_tensor(k).numel() for k in f.keys())
    layers, hidden, vocab = (config[k] for k in ['num_hidden_layers', 'hidden_size', 'vocab_size'])
    kv_bytes_per_token = 2 * layers * config['num_key_value_heads'] * config['head_dim'] * 2
    rows = []
    for length in (16384, 20480):
        rows.append({'tokens': length, 'bf16_dense_logits_gib': length * vocab * 2 / GIB,
                     'fp32_dense_logits_gib': length * vocab * 4 / GIB,
                     'one_bf16_checkpoint_boundary_per_layer_gib': layers * length * hidden * 2 / GIB,
                     'single_sequence_bf16_kv_gib': length * kv_bytes_per_token / GIB})
    # Explicit allowances: this is a planning envelope, NOT an upper-bound proof.
    # BF16 SDPA/Flash Attention, micro-batch 1, checkpointed decoder/head required.
    weights = parameters * 2 / GIB
    adapter = adapter_parameters * 16 / GIB  # FP32 weights+grads+two Adam moments
    checkpoint_one = layers * 20480 * hidden * 2 / GIB
    training = {}
    for name, graphs in [('dpo', 2), ('grpo', 1)]:
        components = {'base_bf16_weights_gib': weights, 'adapter_fp32_weights_grads_adam_gib': adapter,
            'retained_decoder_checkpoint_boundary_gib': checkpoint_one * graphs,
            'decoder_recompute_backward_workspace_allowance_gib': 1.50,
            'chunked_head_workspace_allowance_gib': 0.35,
            'cuda_context_allocator_other_allowance_gib': 1.50}
        training[name] = {'components': components, 'planning_total_gib': sum(components.values()),
                          'simultaneous_decoder_graphs': graphs, 'validation': 'GPU not run'}
    historical = ROOT / 'data/processed/benchmarks/qwen3-0_6b/20261107/summary.json'
    original = json.loads(historical.read_text())
    dpo_overhead = original['stages']['dpo_rollout']['peak_device_memory_mib'] / 1024 - 4
    rollout = {'kv_cache_gib': 4, 'max_num_seqs': 2, 'max_num_batched_tokens': 1024,
        'kv_cache_token_capacity_ideal': int(4 * GIB / kv_bytes_per_token),
        'two_16384_responses_plus_1024_prompt_tokens_each_kv_gib': 2 * (16384 + 1024) * kv_bytes_per_token / GIB,
        'previous_dpo_non_kv_device_peak_gib': dpo_overhead,
        'previous_dpo_peak_with_4gib_cache_gib': 4 + dpo_overhead,
        'additional_planning_margin_gib': .75,
        'planning_total_gib': 4 + dpo_overhead + .75,
        'validation': 'Extrapolated from historical DPO; new PPO/GRPO configuration not GPU tested'}
    payload = {'status': 'estimated_not_gpu_verified', 'config_sha256': expected,
        'base_parameter_count_from_meta_model': parameters, 'tied_embeddings': tied,
        'adapter_parameter_count': adapter_parameters,
        'kv_bytes_per_token_bf16': kv_bytes_per_token, 'sequence_accounting': rows,
        'projection_chunk_tokens': 128,
        'single_bf16_projection_chunk_gib': 128 * vocab * 2 / GIB,
        'single_fp32_projection_chunk_gib': 128 * vocab * 4 / GIB,
        'rollout': rollout, 'training': training,
        'device_gib_8': 8, 'decimal_gb_8_in_gib': 8e9 / GIB,
        'assumptions': ['BF16 model and efficient SDPA/Flash Attention, no quadratic attention fallback',
            'Micro-batch=1, rank=8 LoRA, decoder gradient checkpointing and projection checkpointing',
            'Full vocabulary exact softmax, output max_tokens=16384, max_model_len=20480',
            'DPO may retain chosen+rejected decoder graphs together; reference is a scalar cache',
            'GRPO preserves G=8 and processes one completion at a time',
            'Allowances are engineering estimates, not proven allocation bounds or GPU measurements',
            'No change to baseline accuracy claims or historical measured peaks']}
    path = BASE / 'memory_budget.json'
    path.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
