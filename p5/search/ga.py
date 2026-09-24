"""Generational GA over the pipeline space (zadani 1.7).

Operator-agnostic: it calls operator.propose(parents, context); if that returns None or an invalid
genotype, it falls back to a random proposal and logs the fallback. Budget is measured in genuine
evaluations (evaluator.n_evals), so baselines compare at an equal evaluation budget, not wall-clock.

Per-proposal measurement (zadani v2 2.3, first-class): for every operator proposal we record whether it
parsed, whether it is grammar-valid (+ violation type), whether it duplicates the current population or the
whole run history, and -- for valid proposals -- whether it only substitutes hyperparameters inside an
already-seen structure or introduces a genuinely new structure (Gurkan et al.). The operator may write a
failure reason into context["note"]; the GA aggregates everything.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from . import representation as R


def _tournament(fit, k, rng):
    idx = rng.choice(len(fit), size=min(k, len(fit)), replace=False)
    return int(idx[int(np.argmax([fit[i] for i in idx]))])


class GA:
    def __init__(self, evaluator, operator, grammar, *, population=20, generations=8,
                 tournament_k=3, elitism=2, fallback=None, max_evals=None, max_generations=None):
        self.ev, self.op, self.grammar = evaluator, operator, grammar
        self.population, self.generations = population, generations
        self.tournament_k, self.elitism = tournament_k, elitism
        self.fallback = fallback
        self.max_evals = max_evals
        # Stopping: run generations until the genuine-evaluation budget (max_evals) is spent, up to a
        # safety ceiling of max_generations. Default ceiling = generations -> old behaviour (stop at the
        # generation count). Phase 3 sets max_generations > generations so the EQUAL EVALUATION BUDGET is
        # the primary, binding stop for every method (C1: fixes GA methods under-spending the budget when
        # they propose duplicates, which don't consume a genuine evaluation).
        self.gen_ceiling = max_generations if max_generations is not None else generations

    def _accept(self, child):
        return child is not None and R.validate(child, self.grammar)[0]

    def run(self, task_id: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        pop = [R.random_genotype(self.grammar, rng) for _ in range(self.population)]
        fit = [self.ev.evaluate(g, task_id)["accuracy"] for g in pop]
        seen_full = {R.serialize(g) for g in pop}
        seen_struct = {R.structure_sig(g) for g in pop}
        events = []                        # one record per operator proposal (zadani 2.3)
        gens = []
        best_so_far = max(fit)
        anytime = [{"evals": self.ev.n_evals, "best": float(best_so_far)}]   # best-so-far vs budget
        dup_keys = []                      # serialized genotypes of history-duplicate proposals (H4)

        def rate(pred):
            n = len(events)
            return (sum(1 for e in events if pred(e)) / n) if n else 0.0

        n_struct_total = R.n_structures(self.grammar)

        def snapshot(gi):
            order = np.argsort(fit)[::-1]; best = int(order[0])
            from metrics.diversity import structural_diversity, structure_entropy
            gens.append({"gen": gi, "best": float(fit[best]), "mean": float(np.mean(fit)),
                         "evals": self.ev.n_evals,
                         "struct_diversity": structural_diversity(pop),          # genotypic (edit dist)
                         "structure_entropy": structure_entropy(pop),            # structure-only
                         "struct_coverage": len(seen_struct) / n_struct_total,   # space coverage so far
                         "new_struct_this_gen": sum(1 for e in events            # time-resolved novelty
                                                    if e["gen"] == gi and e.get("structure") == "new_structure"),
                         "invalids": sum(1 for e in events if not e["valid"]),
                         "best_geno": R.serialize(pop[best])})
        snapshot(0)

        gen = 0
        while gen < self.gen_ceiling and (self.max_evals is None or self.ev.n_evals < self.max_evals):
            gen += 1
            order = list(np.argsort(fit)[::-1])
            new_pop = [pop[i] for i in order[:self.elitism]]
            new_fit = [fit[i] for i in order[:self.elitism]]
            while len(new_pop) < self.population:
                p1, p2 = _tournament(fit, self.tournament_k, rng), _tournament(fit, self.tournament_k, rng)
                parents = [pop[p1], pop[p2]]
                note: dict = {}
                ctx = {"grammar": self.grammar, "rng": rng, "generation": gen, "note": note,
                       "history": [R.serialize(pop[i]) for i in order[:5]],
                       "fitness": [fit[p1], fit[p2]]}
                raw = self.op.propose(parents, ctx)
                valid = self._accept(raw)
                ev_rec = {"gen": gen, "valid": bool(valid), "parsed": note.get("parsed", valid)}
                if valid:
                    sr = R.serialize(raw); sg = R.structure_sig(raw)
                    ev_rec.update(dup_pop=sr in {R.serialize(x) for x in new_pop},
                                  dup_hist=sr in seen_full,
                                  structure=("hp_substitution" if sg in seen_struct else "new_structure"))
                    if sr in seen_full:
                        dup_keys.append(sr)
                    child = raw
                else:
                    ev_rec["violation"] = note.get("violation", "unknown")
                    child = (self.fallback.propose(parents, ctx) if self.fallback
                             else R.random_genotype(self.grammar, rng))
                    if not self._accept(child):
                        child = R.random_genotype(self.grammar, rng)
                events.append(ev_rec)
                new_pop.append(child)
                cf = self.ev.evaluate(child, task_id)["accuracy"]
                new_fit.append(cf)
                if cf > best_so_far:
                    best_so_far = cf
                anytime.append({"evals": self.ev.n_evals, "best": float(best_so_far)})
                seen_full.add(R.serialize(child)); seen_struct.add(R.structure_sig(child))
                if self.max_evals and self.ev.n_evals >= self.max_evals:
                    break
            pop, fit = new_pop, new_fit
            snapshot(gen)
            if self.max_evals and self.ev.n_evals >= self.max_evals:
                break

        best = int(np.argmax(fit))
        valid_events = [e for e in events if e["valid"]]
        return {"task_id": task_id, "seed": seed, "operator": getattr(self.op, "name", "?"),
                "best_fitness": float(fit[best]), "best_geno": R.serialize(pop[best]),
                "total_evals": self.ev.n_evals, "proposals": len(events),
                "invalids": len(events) - len(valid_events),
                "fallbacks": len(events) - len(valid_events),
                # zadani 2.3 first-class metrics:
                "invalid_rate": rate(lambda e: not e["valid"]),
                "violation_breakdown": dict(Counter(e.get("violation", "") for e in events if not e["valid"])),
                "dup_pop_rate": rate(lambda e: e.get("dup_pop")),
                "dup_hist_rate": rate(lambda e: e.get("dup_hist")),
                "hp_substitution_rate": (sum(1 for e in valid_events if e["structure"] == "hp_substitution")
                                         / len(valid_events) if valid_events else 0.0),
                "new_structure_rate": (sum(1 for e in valid_events if e["structure"] == "new_structure")
                                       / len(valid_events) if valid_events else 0.0),
                "final_population": [R.serialize(x) for x in pop], "generations": gens,
                "anytime": anytime, "dup_keys": dup_keys}
