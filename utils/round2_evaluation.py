"""CPU reporting only: matched per-question comparisons, never model selection on Holdout."""
from __future__ import annotations
import math
import random
import statistics


def wilson(successes, n, z=1.959963984540054):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError('Invalid binomial counts')
    p = successes / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return [max(0., center-radius), min(1., center+radius)]


def mcnemar_exact(lost, gained):
    """Two-sided exact binomial test conditional on discordant matched outcomes."""
    n = lost + gained
    if not n:
        return 1.0
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(lost, gained)+1)) / 2**n)


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    result = {}; previous = 0.
    for i, key in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered)-i)*pvalues[key]))
        result[key] = previous
    return result


def paired_mean_analysis(differences, draws=10000, seed=42):
    """Question-level bootstrap CI plus paired sign-flip randomization test."""
    if not differences or draws < 100:
        raise ValueError('Need matched observations and >=100 resamples')
    rng = random.Random(seed); n = len(differences)
    mean = statistics.mean(differences)
    bootstrap = sorted(statistics.mean(rng.choices(differences, k=n)) for _ in range(draws))
    extreme = sum(abs(sum(x*rng.choice((-1, 1)) for x in differences)/n) >= abs(mean)-1e-12
                  for _ in range(draws))
    return {'n': n, 'mean_after_minus_before': mean,
            'bootstrap95': [bootstrap[int(.025*draws)], bootstrap[min(draws-1, int(.975*draws))]],
            'sign_flip_pvalue': (extreme+1)/(draws+1), 'draws': draws, 'seed': seed,
            'note': 'Exploratory with small n; sign-flip requires exchangeability/symmetry under null.'}


def paired_comparison(before, after, draws=10000):
    """Reject duplicates, missing questions, and mismatched decoding/evaluation provenance."""
    def key(r):
        return (r['sample_id'], r['task_type'], r['variant'], r['max_tokens'], r.get('output_contract'))
    a = {key(r): r for r in before}; b = {key(r): r for r in after}
    if len(a) != len(before) or len(b) != len(after) or set(a) != set(b) or not a:
        raise ValueError('Need unique and identical evaluation questions/settings')
    for k in a:
        for field in ('prompt_token_ids', 'verifier_version', 'temperature', 'top_p', 'top_k', 'seed'):
            if field not in a[k] or field not in b[k] or a[k][field] != b[k][field]:
                raise ValueError(f'Unmatched evaluation provenance: {field}')
    results = {}
    for task in sorted({k[1] for k in a}):
        keys = sorted(k for k in a if k[1] == task); n = len(keys)
        lost = sum(a[k]['correct'] and not b[k]['correct'] for k in keys)
        gained = sum(not a[k]['correct'] and b[k]['correct'] for k in keys)
        more_trunc = sum(not a[k]['truncated'] and b[k]['truncated'] for k in keys)
        less_trunc = sum(a[k]['truncated'] and not b[k]['truncated'] for k in keys)
        both_good = [k for k in keys if a[k]['correct'] and b[k]['correct']]
        lengths = [len(b[k]['completion_token_ids']) - len(a[k]['completion_token_ids']) for k in keys]
        results[task] = {'n': n, 'lost_correct': lost, 'gained_correct': gained,
            'accuracy_delta': (gained-lost)/n, 'accuracy_mcnemar_pvalue': mcnemar_exact(lost, gained),
            'accuracy_before_wilson95': wilson(sum(bool(a[k]['correct']) for k in keys), n),
            'accuracy_after_wilson95': wilson(sum(bool(b[k]['correct']) for k in keys), n),
            'more_truncated': more_trunc, 'less_truncated': less_trunc,
            'truncation_delta': (more_trunc-less_trunc)/n,
            'truncation_mcnemar_pvalue': mcnemar_exact(more_trunc, less_trunc),
            'length': paired_mean_analysis(lengths, draws),
            'both_correct_n': len(both_good),
            'both_correct_length': paired_mean_analysis(
                [len(b[k]['completion_token_ids'])-len(a[k]['completion_token_ids']) for k in both_good], draws)
                if both_good else None,
            'per_question': [{'sample_id': k[0], 'correct_before': bool(a[k]['correct']),
                              'correct_after': bool(b[k]['correct']),
                              'truncated_before':bool(a[k]['truncated']),
                              'truncated_after':bool(b[k]['truncated']), 'token_delta': d}
                             for k, d in zip(keys, lengths)]}
    return results
