"""Generational GA with explicit per-generation stepping, for the Batch-API driver (run_phase3).

Same algorithm and same first-class metrics as search.ga.GA (zadani 1.7 / 2.3), but restructured so a
whole generation's offspring proposals can be produced in one batch: the driver calls
`gen_contexts()` to get every offspring's (parents, ctx) for the next generation, sends all their
suffixes to the Batch API together, then calls `apply(raws)` with the parsed candidates in order.

Proposals depend only on (pop, fit, rng) via tournament selection and the operator's mode coin-flip, not
on the partially-built next population, so batching them is exact. `apply` processes the raws in order and
builds the next population incrementally, so `dup_pop` (duplicate vs the population being formed) is
computed identically to the sequential GA. Metric definitions are kept byte-for-byte in sync with GA.

The operator is used only offline here: `op.build(parents, ctx)` makes the suffix (and sets ctx note
mode), `op.interpret(text, ctx)` parses one response. The LLM calls happen in the driver's batch.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from . import representation as R


def _tournament(fit, k, rng):
    idx = rng.choice(len(fit), size=min(k, len(fit)), replace=False)
    return int(idx[int(np.argmax([fit[i] for i in idx]))])


class BatchGA:
    def __init__(self, evaluator, operator, grammar, *, population=50, generations=30,
                 tournament_k=3, elitism=2, fallback=None, max_evals=None, max_generations=None, run_id="?"):
        self.ev, self.op, self.grammar = evaluator, operator, grammar
        self.population, self.generations = population, generations
        self.tournament_k, self.elitism = tournament_k, elitism
        self.fallback = fallback
        self.max_evals = max_evals
        # See GA: eval budget is primary, generations run up to this safety ceiling (default = generations).
        self.gen_ceiling = max_generations if max_generations is not None else generations
        self.run_id = run_id
        self._done = False

    # --- lifecycle -------------------------------------------------------------------------------
    def start(self, task_id: int, seed: int):
        """Offline: initial population + its evaluation + generation-0 snapshot."""
        self.task_id, self.seed = task_id, seed
        self.rng = np.random.default_rng(seed)
        self.pop = [R.random_genotype(self.grammar, self.rng) for _ in range(self.population)]
        self.fit = [self.ev.evaluate(g, task_id)["accuracy"] for g in self.pop]
        self.seen_full = {R.serialize(g) for g in self.pop}
        self.seen_struct = {R.structure_sig(g) for g in self.pop}
        self.events = []
        self.gens = []
        self._gen = 0
        self.best_so_far = max(self.fit)
        self.anytime = [{"evals": self.ev.n_evals, "best": float(self.best_so_far)}]
        self.dup_keys = []                 # serialized history-duplicate proposals (H4)
        self._snapshot(0)
        if self.max_evals and self.ev.n_evals >= self.max_evals:
            self._done = True

    def _snapshot(self, gi):
        from metrics.diversity import structural_diversity, structure_entropy
        order = np.argsort(self.fit)[::-1]; best = int(order[0])
        n_struct_total = R.n_structures(self.grammar)
        self.gens.append({"gen": gi, "best": float(self.fit[best]), "mean": float(np.mean(self.fit)),
                          "evals": self.ev.n_evals,
                          "struct_diversity": structural_diversity(self.pop),          # genotypic (edit dist)
                          "structure_entropy": structure_entropy(self.pop),            # structure-only
                          "struct_coverage": len(self.seen_struct) / n_struct_total,   # space coverage so far
                          "new_struct_this_gen": sum(1 for e in self.events
                                                     if e["gen"] == gi and e.get("structure") == "new_structure"),
                          "invalids": sum(1 for e in self.events if not e["valid"]),
                          "best_geno": R.serialize(self.pop[best])})

    @property
    def finished(self) -> bool:
        return self._done

    # --- one generation, split into request-building and result-applying -------------------------
    def gen_contexts(self) -> list[tuple]:
        """Build the next generation's offspring (parents, ctx). Consumes rng (tournaments + operator
        mode flips happen in op.build, called by the driver). Returns [] when the run is finished."""
        if self._done or self._gen >= self.gen_ceiling or \
                (self.max_evals is not None and self.ev.n_evals >= self.max_evals):
            self._done = True
            return []
        self._gen += 1
        order = list(np.argsort(self.fit)[::-1])
        self._elite_idx = order[:self.elitism]
        n_off = self.population - self.elitism
        if self.max_evals is not None:                     # never propose beyond the eval budget
            n_off = max(0, min(n_off, self.max_evals - self.ev.n_evals))
        if n_off <= 0:                                     # budget exhausted at a generation boundary
            self._done = True
            return []
        hist = [R.serialize(self.pop[i]) for i in order[:5]]
        ctxs = []
        for _ in range(n_off):
            p1 = _tournament(self.fit, self.tournament_k, self.rng)
            p2 = _tournament(self.fit, self.tournament_k, self.rng)
            ctx = {"grammar": self.grammar, "rng": self.rng, "generation": self._gen, "note": {},
                   "history": hist, "fitness": [self.fit[p1], self.fit[p2]], "run_id": self.run_id}
            ctxs.append(([self.pop[p1], self.pop[p2]], ctx))
        self._pending = ctxs
        return ctxs

    def _accept(self, child):
        return child is not None and R.validate(child, self.grammar)[0]

    def apply(self, raws: list):
        """Process this generation's parsed candidates (same order as gen_contexts). raws[i] is the
        genotype for offspring i or None (invalid/failed); the corresponding ctx note holds the reason."""
        new_pop = [self.pop[i] for i in self._elite_idx]
        new_fit = [self.fit[i] for i in self._elite_idx]
        for (parents, ctx), raw in zip(self._pending, raws):
            note = ctx["note"]
            valid = self._accept(raw)
            ev_rec = {"gen": self._gen, "valid": bool(valid), "parsed": note.get("parsed", valid)}
            if valid:
                sr = R.serialize(raw); sg = R.structure_sig(raw)
                ev_rec.update(dup_pop=sr in {R.serialize(x) for x in new_pop},
                              dup_hist=sr in self.seen_full,
                              structure=("hp_substitution" if sg in self.seen_struct else "new_structure"))
                if sr in self.seen_full:
                    self.dup_keys.append(sr)
                child = raw
            else:
                ev_rec["violation"] = note.get("violation", "unknown")
                child = (self.fallback.propose(parents, ctx) if self.fallback
                         else R.random_genotype(self.grammar, self.rng))
                if not self._accept(child):
                    child = R.random_genotype(self.grammar, self.rng)
            self.events.append(ev_rec)
            new_pop.append(child)
            cf = self.ev.evaluate(child, self.task_id)["accuracy"]
            new_fit.append(cf)
            if cf > self.best_so_far:
                self.best_so_far = cf
            self.anytime.append({"evals": self.ev.n_evals, "best": float(self.best_so_far)})
            self.seen_full.add(R.serialize(child)); self.seen_struct.add(R.structure_sig(child))
            if self.max_evals and self.ev.n_evals >= self.max_evals:
                break
        self.pop, self.fit = new_pop, new_fit
        self._snapshot(self._gen)
        if self._gen >= self.gen_ceiling or (self.max_evals and self.ev.n_evals >= self.max_evals):
            self._done = True

    # --- final result (identical schema to search.ga.GA.run) ------------------------------------
    def result(self) -> dict:
        def rate(pred):
            n = len(self.events)
            return (sum(1 for e in self.events if pred(e)) / n) if n else 0.0
        best = int(np.argmax(self.fit))
        valid_events = [e for e in self.events if e["valid"]]
        return {"task_id": self.task_id, "seed": self.seed,
                "operator": getattr(self.op, "name", "?"),
                "best_fitness": float(self.fit[best]), "best_geno": R.serialize(self.pop[best]),
                "total_evals": self.ev.n_evals, "proposals": len(self.events),
                "invalids": len(self.events) - len(valid_events),
                "fallbacks": len(self.events) - len(valid_events),
                "invalid_rate": rate(lambda e: not e["valid"]),
                "violation_breakdown": dict(Counter(e.get("violation", "") for e in self.events if not e["valid"])),
                "dup_pop_rate": rate(lambda e: e.get("dup_pop")),
                "dup_hist_rate": rate(lambda e: e.get("dup_hist")),
                "hp_substitution_rate": (sum(1 for e in valid_events if e["structure"] == "hp_substitution")
                                         / len(valid_events) if valid_events else 0.0),
                "new_structure_rate": (sum(1 for e in valid_events if e["structure"] == "new_structure")
                                       / len(valid_events) if valid_events else 0.0),
                "final_population": [R.serialize(x) for x in self.pop], "generations": self.gens,
                "anytime": self.anytime, "dup_keys": self.dup_keys}
