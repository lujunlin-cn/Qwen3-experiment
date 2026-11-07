"""Exact full-vocabulary logprobs without full-sequence vocabulary logits.

The decoder still sees the entire causal sequence. Only the independent output
projection is chunked; checkpointing recomputes projection/softmax on backward.
No top-k approximation, token truncation or detached policy gradient is used.
"""
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _selected_logprobs(hidden, labels, head):
    logits = head(hidden).float()
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def projected_token_logprobs(hidden, head, input_ids, chunk_size=128):
    """Return [B,T-1]; hidden[t] predicts input_ids[t+1], including Prompt."""
    if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape or input_ids.size(1) < 2:
        raise ValueError('Hidden states and Token IDs must align as [B,T] with T>=2')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    values = []
    for start in range(0, hidden.size(1) - 1, chunk_size):
        end = min(start + chunk_size, hidden.size(1) - 1)
        h, labels = hidden[:, start:end], input_ids[:, start + 1:end + 1]
        # Include projection in the checkpoint so [B,T,V] is never retained.
        def block(h, labels):
            return _selected_logprobs(h, labels, head)
        if torch.is_grad_enabled() and (h.requires_grad or any(p.requires_grad for p in head.parameters())):
            value = checkpoint(block, h, labels, use_reentrant=False)
        else:
            value = block(h, labels)
        values.append(value)
    return torch.cat(values, dim=1)


def causal_policy_logprobs(model, input_ids, attention_mask=None, chunk_size=128):
    """Qwen causal decoder with PEFT layers active; TinyPolicy keeps CPU tests."""
    causal = model.get_base_model() if hasattr(model, 'get_base_model') else model
    # These injected PEFT projections are the same objects used by LM.forward.
    if not hasattr(causal, 'model') or not hasattr(causal, 'get_output_embeddings'):
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output if isinstance(output, torch.Tensor) else (output.logits if hasattr(output, 'logits') else output['logits'])
        return F.log_softmax(logits[:, :-1].float(), -1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    # All-valid masks need only the decoder causal mask. Preserve actual padding.
    mask = None if attention_mask is None or bool(torch.all(attention_mask == 1)) else attention_mask
    # k-bit preparation can leave norms/embeddings FP32: BF16 autocast lets SDPA
    # use efficient kernels and avoids large FP32 attention intermediates.
    context = torch.autocast('cuda', dtype=torch.bfloat16) if input_ids.device.type == 'cuda' else nullcontext()
    with context:
        hidden = causal.model(input_ids=input_ids, attention_mask=mask, use_cache=False,
                              output_hidden_states=False, return_dict=True).last_hidden_state
        return projected_token_logprobs(hidden, causal.get_output_embeddings(), input_ids, chunk_size)


def causal_sequence_logprob(model, input_ids, response_mask, chunk_size=128):
    if response_mask.shape != input_ids.shape:
        raise ValueError('Response mask and Token IDs must align')
    values = causal_policy_logprobs(model, input_ids, torch.ones_like(input_ids), chunk_size)
    return (values * response_mask[:, 1:].to(values.dtype)).sum(-1)
