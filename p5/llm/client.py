"""Anthropic Messages client for the LLM operator: prefix caching, accounting, preflight cost cap.

Key resolution (P5 can later get its own workspace key without code changes): ANTHROPIC_API_KEY env,
then ~/.p5_anthropic_key, then ~/.anthropic_key.
"""
from __future__ import annotations
import os, time
from pathlib import Path
from .accounting import Ledger, call_cost, BudgetExceeded
from .prompt import system_block


def load_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    for name in (".p5_anthropic_key", ".anthropic_key"):
        p = Path.home() / name
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("No API key: set ANTHROPIC_API_KEY or write ~/.p5_anthropic_key")


# Minimum prefix tokens for the ephemeral cache to activate, per model family (measured empirically:
# Haiku 4.5 does not cache a 4090-token prefix but does cache a 4390-token one -> 4096). Below the
# threshold the cache silently no-ops (cache_creation = cache_read = 0), so cost estimates must NOT
# assume caching there.
def _cache_min_tokens(model: str) -> int:
    m = model.lower()
    if "haiku" in m:
        return 4096
    return 1024   # sonnet / opus


def _usage(msg) -> dict:
    u = msg.usage
    return {"inp": u.input_tokens, "out": u.output_tokens,
            "cw": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cr": getattr(u, "cache_read_input_tokens", 0) or 0}


class LLMClient:
    def __init__(self, model: str, ledger: Ledger, max_tokens: int = 1200, timeout: float = 120.0):
        import anthropic
        self.model = model
        self.ledger = ledger
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=load_key(), timeout=timeout, max_retries=2)

    def count_prefix_tokens(self, prefix: str, suffix: str) -> int:
        """Input-token count for one prompt (for preflight estimation)."""
        r = self.client.messages.count_tokens(
            model=self.model, system=system_block(prefix),
            messages=[{"role": "user", "content": suffix}])
        return r.input_tokens

    def estimate_run_cost(self, prefix: str, sample_suffix: str, n_calls: int,
                          est_out_tokens: int = 400) -> dict:
        """Preflight estimate (zadani 0.4): first call writes the prefix to cache, the rest read it.
        Returns {est_usd, in_tokens, ...} without spending anything beyond one count_tokens call."""
        price = self.ledger.price_for(self.model)
        n_in = self.count_prefix_tokens(prefix, sample_suffix)
        prefix_tok = self.count_prefix_tokens(prefix, "x")   # cacheable part (system block ~ prefix)
        cache_on = prefix_tok >= _cache_min_tokens(self.model)
        if cache_on:
            # prefix cached: first call writes it, the rest read it; only the prefix portion is cached.
            suffix_tok = max(0, n_in - prefix_tok)
            first = call_cost({"cw": prefix_tok, "inp": suffix_tok, "out": est_out_tokens}, price)
            rest = call_cost({"cr": prefix_tok, "inp": suffix_tok, "out": est_out_tokens}, price) * max(0, n_calls - 1)
            est = first + rest
        else:
            # below the model's cache threshold -> caching silently no-ops; every call pays full input.
            est = call_cost({"inp": n_in, "out": est_out_tokens}, price) * n_calls
        return {"est_usd": round(est, 4), "in_tokens": n_in, "prefix_tokens": prefix_tok,
                "cache_active": cache_on, "n_calls": n_calls,
                "est_out_tokens": est_out_tokens, "cap_usd": self.ledger.cap,
                "already_spent_usd": round(self.ledger.spent, 4)}

    def preflight(self, prefix: str, sample_suffix: str, n_calls: int, est_out_tokens: int = 400,
                  require_confirm: bool = True) -> dict:
        est = self.estimate_run_cost(prefix, sample_suffix, n_calls, est_out_tokens)
        print(f"[preflight] ~{est['in_tokens']} in-tok x {n_calls} calls -> est ${est['est_usd']} "
              f"(cap ${est['cap_usd']}, already spent ${est['already_spent_usd']})")
        if self.ledger.would_exceed(est["est_usd"]):
            raise BudgetExceeded(f"preflight estimate ${est['est_usd']} would exceed cap ${est['cap_usd']}")
        if require_confirm and os.environ.get("P5_ASSUME_YES") != "1":
            resp = input("[preflight] proceed? [y/N] ").strip().lower()
            if resp != "y":
                raise SystemExit("aborted at preflight")
        return est

    def propose_raw(self, prefix: str, suffix: str, **meta) -> tuple[str, dict]:
        """One operator call. Returns (text, usage); logs cost and enforces the cap via the ledger."""
        t0 = time.time()
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system_block(prefix),
            messages=[{"role": "user", "content": suffix}],
            thinking={"type": "disabled"})
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        usage = _usage(msg)
        self.ledger.record(self.model, usage, latency_s=round(time.time() - t0, 3), **meta)
        return text, usage
