"""LLM variation operator (zadani 2.1, 2.2).

Does BOTH mutation and crossover (crossover is an axis of the positioning vs LLaMEA and must not drop):
with two parents it recombines them (crossover) or mutates one (mutation), chosen by a probability, like
the random operator. Prompt = cached static prefix (instructions + grammar) + variable suffix (parents,
fitness, history, mode directive). Invalid/unparseable output -> one retry (same cached prefix) -> None,
so the GA falls back and logs it. The failure reason is written to context["note"] for the violation
breakdown (zadani 2.3). Every API call is charged and logged by the client's ledger.
"""
from __future__ import annotations
from .base import Operator
from search import representation as R
from llm.prompt import build_prefix, build_suffix


def _violation_category(reason: str) -> str:
    r = reason.lower()
    if "unknown option" in r: return "unknown_option"
    if "unknown param" in r:  return "unknown_param"
    if "out of range" in r:   return "out_of_range"
    if "not in choices" in r: return "bad_choice"
    if "not numeric" in r:    return "non_numeric"
    if "malformed" in r or "missing" in r: return "malformed"
    return "other"


class LLMOperator(Operator):
    name = "llm"

    def __init__(self, client, grammar: dict, *, include_fitness: bool = True,
                 include_history: bool = True, p_crossover: float = 0.5):
        self.client = client
        self.grammar = grammar
        self.include_fitness = include_fitness
        self.include_history = include_history
        self.p_crossover = p_crossover
        self.prefix = build_prefix(R.grammar_text(grammar))     # static -> prefix cache hits

    def build(self, parents: list[dict], context: dict) -> str:
        """Offline: pick mode (mutation/crossover) via context['rng'], set note['mode'], return the
        variable suffix. The static cacheable prefix is self.prefix. No API call -- used by both the
        synchronous path (propose) and the batch driver (run_phase3)."""
        rng = context["rng"]; note = context.get("note", {})
        crossover = len(parents) >= 2 and rng.random() < self.p_crossover
        mode = "crossover" if crossover else "mutation"
        used = parents[:2] if crossover else parents[:1]
        directive = ("\nRecombine the parent pipelines into ONE new valid candidate (crossover)."
                     if crossover else
                     "\nMutate the parent pipeline into ONE new valid candidate (mutation).")
        note["mode"] = mode
        return build_suffix(used, fitness=context.get("fitness"), history=context.get("history"),
                            include_fitness=self.include_fitness,
                            include_history=self.include_history) + directive

    def interpret(self, text: str | None, context: dict) -> dict | None:
        """Offline: parse+validate one raw LLM response, recording parsed/violation into note. Returns
        the genotype or None. text=None means the call failed (api_error)."""
        note = context.get("note", {})
        if text is None:
            note["parsed"] = False; note["violation"] = "api_error"; return None
        cand = R.parse(text)
        if cand is None:
            note["parsed"] = False; note["violation"] = "parse_error"; return None
        ok, why = R.validate(cand, self.grammar)
        if ok:
            note["parsed"] = True; return cand
        note["parsed"] = True; note["violation"] = _violation_category(why); return None

    def propose(self, parents: list[dict], context: dict) -> dict | None:
        note = context.get("note", {})
        base = self.build(parents, context)
        meta = {"run_id": context.get("run_id", "?"), "generation": context.get("generation", -1),
                "mode": note.get("mode")}
        for attempt in (0, 1):
            s = base if attempt == 0 else base + \
                "\nYour previous output was not valid under the grammar. Return ONE valid JSON object only."
            try:
                text, _ = self.client.propose_raw(self.prefix, s, attempt=attempt, **meta)
            except Exception:
                note["parsed"] = False; note["violation"] = "api_error"; return None
            cand = self.interpret(text, context)
            if cand is not None:
                return cand
        return None
