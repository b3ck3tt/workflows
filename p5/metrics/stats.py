"""Statistics for the equal-budget comparison (review §3.1). A *negative* result needs an EQUIVALENCE
test, not a failed difference test: with n=15 tasks and a tiny delta, Wilcoxon will never reject the null,
which is "absence of evidence", not "evidence of absence". We therefore report:
  - bootstrap CI on the mean paired delta (LLM - baseline);
  - TOST (two one-sided tests) against a PRE-REGISTERED equivalence margin (default 0.5 accuracy points):
    rejecting H0 means the LLM's advantage/disadvantage is bounded within +-margin (statistically equivalent).
Deltas are per-task paired differences (average the seeds within a task first).
"""
from __future__ import annotations
import numpy as np

DEFAULT_MARGIN = 0.005   # 0.5 accuracy points, pre-registered equivalence bound (PREREGISTRATION.md)


def bootstrap_ci(deltas, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    d = np.asarray(deltas, dtype=float)
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d))}


def tost(deltas, margin: float = DEFAULT_MARGIN):
    """Paired TOST for equivalence within +-margin. Returns the equivalence p-value (max of the two
    one-sided p-values); equivalent at 0.05 if p_equiv < 0.05. Falls back to a normal approximation if
    scipy is unavailable."""
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n < 2:
        return {"mean": float(d.mean()) if n else float("nan"), "p_equiv": float("nan"),
                "equivalent": False, "margin": margin, "n": n}
    m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n))
    if se == 0:
        eq = abs(m) < margin
        return {"mean": m, "p_equiv": 0.0 if eq else 1.0, "equivalent": eq, "margin": margin, "n": n}
    t_lo = (m - (-margin)) / se     # H0: mean <= -margin
    t_hi = (margin - m) / se        # H0: mean >=  margin
    try:
        from scipy import stats
        sf = lambda t: float(stats.t.sf(t, n - 1))
    except Exception:
        from math import erf, sqrt
        sf = lambda t: 0.5 * (1 - erf(t / sqrt(2)))   # normal approx
    p_equiv = max(sf(t_lo), sf(t_hi))
    return {"mean": m, "p_equiv": p_equiv, "equivalent": bool(p_equiv < 0.05), "margin": margin, "n": n}


def duplicate_concentration(dup_keys) -> float:
    """H4 (reformulated, review §3.4): are duplicates CONCENTRATED on a few genotypes (narrow revisiting)
    or SPREAD (broad)? Normalized entropy of the distribution of which genotypes get duplicated, in [0,1]
    (0 = all duplicates hit one genotype = concentrated; 1 = spread uniformly). `dup_keys` is the list of
    serialized genotypes that were duplicates."""
    import math
    from collections import Counter
    if len(dup_keys) < 2:
        return 0.0
    c = Counter(dup_keys); n = len(dup_keys)
    H = -sum((v / n) * math.log(v / n) for v in c.values())
    return float(H / math.log(len(c))) if len(c) > 1 else 0.0
