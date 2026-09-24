"""Random variation operator (baseline, zadani 1.5). Mutation + uniform crossover over the grammar.
Always returns a valid genotype (never None), so it also serves as the fallback for the LLM operator.
"""
from __future__ import annotations
import copy
from .base import Operator
from search import representation as R


class RandomOperator(Operator):
    name = "random"

    def __init__(self, grammar: dict, p_mutate_stage: float = 0.5):
        self.grammar = grammar
        self.p_stage = p_mutate_stage

    def mutate(self, g: dict, rng) -> dict:
        g = copy.deepcopy(g)
        stage = rng.choice(R.STAGES)
        if rng.random() < self.p_stage:
            # replace the whole stage (new option + fresh params)
            g[stage] = R._sample_stage(self.grammar, stage, rng)
        else:
            # tweak one parameter of the current option (or resample the stage if it has none)
            name = g[stage]["name"]; spec = self.grammar[stage][name]
            if spec:
                p = rng.choice(list(spec.keys()))
                g[stage].setdefault("params", {})[p] = R._sample_param(spec[p], rng)
                g[stage]["params"] = {k: v for k, v in g[stage]["params"].items() if v is not None}
            else:
                g[stage] = R._sample_stage(self.grammar, stage, rng)
        return g

    def crossover(self, a: dict, b: dict, rng) -> dict:
        return {st: copy.deepcopy((a if rng.random() < 0.5 else b)[st]) for st in R.STAGES}

    def propose(self, parents: list[dict], context: dict) -> dict:
        rng = context["rng"]
        if len(parents) >= 2 and rng.random() < 0.5:
            child = self.crossover(parents[0], parents[1], rng)
            if rng.random() < 0.5:
                child = self.mutate(child, rng)
        else:
            child = self.mutate(parents[0], rng)
        ok, _ = R.validate(child, self.grammar)
        return child if ok else R.random_genotype(self.grammar, rng)
