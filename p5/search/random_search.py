"""Random-search baseline (zadani 1.5): sample genotypes uniformly at random under the grammar, keep the
best, at the same evaluation budget as the GA. No evolution — the null that any search must beat.
"""
from __future__ import annotations
import numpy as np
from . import representation as R


def random_search(evaluator, grammar, task_id: int, seed: int, max_evals: int) -> dict:
    rng = np.random.default_rng(seed)
    best_g, best_f, trace = None, -1.0, []
    while evaluator.n_evals < max_evals:
        g = R.random_genotype(grammar, rng)
        f = evaluator.evaluate(g, task_id)["accuracy"]
        if f > best_f:
            best_f, best_g = f, g
        trace.append({"evals": evaluator.n_evals, "best": best_f})
    return {"task_id": task_id, "seed": seed, "operator": "random_search",
            "best_fitness": float(best_f), "best_geno": R.serialize(best_g),
            "total_evals": evaluator.n_evals, "trace": trace}
