"""CPU numerical checks for projected logprobs, real Qwen3/PEFT and RL losses."""
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.low_memory_logprobs import causal_policy_logprobs, projected_token_logprobs


def dense(model, ids, mask):
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    return F.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)


def compare_gradients(reference, actual, atol=3e-6, rtol=3e-4):
    differences = []
    for (name, p), (name2, q) in zip(reference.named_parameters(), actual.named_parameters()):
        assert name == name2
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None, name
            torch.testing.assert_close(q.grad, p.grad, atol=atol, rtol=rtol, msg=name)
            differences.append(float((p.grad.float() - q.grad.float()).abs().max()))
    assert differences and max(float(p.grad.abs().max()) for p in actual.parameters() if p.grad is not None) > 0
    return max(differences)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(317)
    cases = []
    # The real vocabulary is included; the head is tied to embeddings below.
    for vocab, dtype, chunk in [(37, torch.float32, 1), (151936, torch.float32, 4),
                                (151936, torch.bfloat16, 4)]:
        h = torch.randn(2, 11, 16, dtype=dtype, requires_grad=True)
        head = torch.nn.Linear(16, vocab, bias=False, dtype=dtype)
        h2, head2 = h.detach().clone().requires_grad_(True), copy.deepcopy(head)
        ids = torch.randint(0, vocab, (2, 11))
        expected = F.log_softmax(head(h)[:, :-1].float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        actual = projected_token_logprobs(h2, head2, ids, chunk)
        weight = torch.randn_like(actual)
        (expected * weight).sum().backward()
        (actual * weight).sum().backward()
        tolerance = 0.04 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(h2.grad, h.grad, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(head2.weight.grad, head.weight.grad, atol=tolerance, rtol=tolerance)
        cases.append({'kind': 'projection', 'vocab': vocab, 'dtype': str(dtype), 'chunk': chunk,
                      'max_value_delta': float((actual - expected).abs().max().detach()),
                      'max_hidden_grad_delta': float((h2.grad - h.grad).abs().max())})

    config = Qwen3Config(vocab_size=71, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, tie_word_embeddings=True, attention_dropout=0.0, pad_token_id=0)
    config._attn_implementation = 'sdpa'
    model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(r=2, lora_alpha=4,
        lora_dropout=0.0, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                        'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM'))
    for name, parameter in model.named_parameters():
        if 'lora_B' in name:
            torch.nn.init.normal_(parameter, std=0.02)
    ids = torch.randint(1, 71, (2, 11))
    for checkpointing in (False, True):
        for padded in (False, True):
            reference, actual = copy.deepcopy(model), copy.deepcopy(model)
            mask = torch.ones_like(ids)
            if padded:
                mask[1, -3:] = 0
            response = mask[:, 1:].float()
            response[:, :3] = 0
            for m in (reference, actual):
                m.train()
                if checkpointing:
                    m.gradient_checkpointing_enable()
                    m.enable_input_require_grads()
            expected, result = dense(reference, ids, mask), causal_policy_logprobs(actual, ids, mask, 4)
            torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
            # DPO: sum over response Tokens, immutable chosen/rejected reference.
            frozen = (expected.detach() * response).sum(-1)
            r = (expected * response).sum(-1) - frozen
            a = (result * response).sum(-1) - frozen
            loss_r, loss_a = -F.logsigmoid(0.1 * (r[0] - r[1])), -F.logsigmoid(0.1 * (a[0] - a[1]))
            loss_r.backward(); loss_a.backward()
            grad_delta = compare_gradients(reference, actual)
            # Full GRPO clipped objective, including masked Prompt/padding.
            reference.zero_grad(); actual.zero_grad()
            expected, result = dense(reference, ids, mask), causal_policy_logprobs(actual, ids, mask, 4)
            old = expected.detach() + torch.randn_like(expected) * 0.2
            advantages = torch.tensor([1., -1.])[:, None]
            def objective(lp):
                ratio = (lp - old).exp()
                return -(torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages) * response).sum() / response.sum()
            loss_r, loss_a = objective(expected), objective(result)
            torch.testing.assert_close(loss_r, loss_a, atol=2e-6, rtol=2e-6)
            loss_r.backward(); loss_a.backward()
            grad_delta = max(grad_delta, compare_gradients(reference, actual))
            before = {n: p.detach().clone() for n, p in actual.named_parameters() if p.requires_grad}
            torch.optim.AdamW((p for p in actual.parameters() if p.requires_grad), lr=1e-5).step()
            update = max(float((p.detach() - before[n]).abs().max()) for n, p in actual.named_parameters() if n in before)
            assert update > 0
            cases.append({'kind': 'real_qwen3_peft', 'checkpointing': checkpointing, 'padding': padded,
                          'dpo_and_grpo_gradients': 'passed', 'max_gradient_delta': grad_delta,
                          'optimizer_max_parameter_update': update})
    # Enforce the structural memory guarantee on the real PEFT/Qwen path:
    # only decoder.forward may see the full sequence; the head sees <=chunk.
    model.eval()
    widths = []
    handle = model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: widths.append(args[0].shape[1]))
    with torch.no_grad():
        causal_policy_logprobs(model, ids, torch.ones_like(ids), 4)
    handle.remove()
    assert widths == [4, 4, 2], widths
    output = Path('data/processed/benchmarks/qwen3-0_6b/low_memory_20261107/cpu_numerical_checks.json')
    payload = {'status': 'passed', 'scope': 'CPU fixtures only; no pretrained weights or CUDA peak measurement',
               'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
               'peft': __import__('peft').__version__, 'cases': cases,
               'actual_head_projection_widths': widths}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
