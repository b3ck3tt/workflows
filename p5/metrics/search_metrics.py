"""Search-quality metrics (zadani 2.3): normalized regret vs a per-task reference, and
evaluations-to-target. Regret needs a reference best per task; in the analysis the reference is the best
fitness found by any method/seed on that task (a fixed anchor computed once, then reused)."""
from __future__ import annotations


def nregret(best_fitness: float, ref_best: float, ref_worst: float) -> float:
    """Normalized regret in [0,1]: 0 = matched the reference best, 1 = at the reference worst."""
    rng = ref_best - ref_worst
    if rng <= 0:
        return 0.0
    return max(0.0, min(1.0, (ref_best - best_fitness) / rng))


def evals_to_target(generations: list[dict], target: float) -> int | None:
    """First cumulative eval count at which best fitness reached `target`; None if never."""
    for g in generations:
        if g["best"] >= target:
            return g["evals"]
    return None


def anytime_curve(generations: list[dict]) -> list[tuple[int, float]]:
    """(evals, best-so-far) trace for anytime-regret plots."""
    return [(g["evals"], g["best"]) for g in generations]
