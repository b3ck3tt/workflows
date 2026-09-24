"""OpenAI-compatible client for the open-weight model ablation (review §3.7, accepted C2): reproducibility
against a frozen closed model that drifts. DeepInfra (and most OpenAI-compatible endpoints) do NOT expose a
Batch API, so instead of the Anthropic Batches path we fire a generation's proposals CONCURRENTLY over a
thread pool. `ConcurrentBatchClient` exposes the SAME interface as llm.batch_client.BatchClient
(`estimate_usd`, `run_batch`) so run_phase3's lockstep driver uses it unchanged.

Key resolution: env DEEPINFRA_API_KEY / OPENAI_API_KEY, then ~/.deepinfra_key / ~/.openai_key (chmod 600).
No 50% batch discount here (calls are synchronous under the hood) -> ledger.record(batch=False).
"""
from __future__ import annotations
import os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .accounting import BudgetExceeded, call_cost


def load_openai_key() -> str:
    for env in ("DEEPINFRA_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(env):
            return os.environ[env]
    for name in (".deepinfra_key", ".openai_key"):
        p = Path.home() / name
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("No open-weight key: set DEEPINFRA_API_KEY or write ~/.deepinfra_key")


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)   # heuristic (no count_tokens endpoint); good enough for preflight


class OpenAICompatClient:
    def __init__(self, model: str, ledger, *, base_url: str | None = None, max_tokens: int = 1200,
                 timeout: float = 30.0, temperature: float = 0.7, max_retries: int = 2):
        self.model, self.ledger = model, ledger
        self.max_tokens, self.temperature = max_tokens, temperature
        self.base_url = base_url or "https://api.deepinfra.com/v1/openai"
        self.timeout, self.max_retries = timeout, max_retries
        self._client = None            # lazy: no key needed until the first real call (preflight-safe)

    @property
    def client(self):
        if self._client is None:
            import openai
            # low per-request timeout + few retries so a slow/hung endpoint fails fast to a fallback
            # rather than stalling a whole generation (learned from the Qwen pilot).
            self._client = openai.OpenAI(api_key=load_openai_key(), base_url=self.base_url,
                                         timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def propose_raw(self, prefix: str, suffix: str, **meta):
        t0 = time.time()
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            messages=[{"role": "system", "content": prefix}, {"role": "user", "content": suffix}])
        text = r.choices[0].message.content or ""
        u = r.usage
        usage = {"inp": getattr(u, "prompt_tokens", 0), "out": getattr(u, "completion_tokens", 0),
                 "cw": 0, "cr": 0}
        self.ledger.record(self.model, usage, batch=False, latency_s=round(time.time() - t0, 3), **meta)
        return text, usage


class ConcurrentBatchClient:
    """Drop-in for BatchClient over an OpenAI-compatible endpoint, via a thread pool (no server-side batch)."""
    def __init__(self, model: str, ledger, *, base_url: str | None = None, max_tokens: int = 1200,
                 temperature: float = 0.7, max_workers: int = 8, round_deadline: float = 300.0,
                 **_ignored):
        self.ledger = ledger
        self.max_workers = max_workers
        self.round_deadline = round_deadline    # hard cap per lockstep round; stragglers -> None (fallback)
        self.oc = OpenAICompatClient(model, ledger, base_url=base_url, max_tokens=max_tokens,
                                     temperature=temperature)
        self.model = model

    def estimate_usd(self, prefix: str, sample_suffix: str, n_calls: int, est_out: int = 200) -> float:
        price = self.ledger.price_for(self.model)
        n_in = _est_tokens(prefix) + _est_tokens(sample_suffix)
        return call_cost({"inp": n_in, "out": est_out}, price) * n_calls   # no batch discount

    def run_batch(self, prefix: str, items: list[tuple], **meta) -> dict:
        if not items:
            return {}
        import re
        bad = [cid for cid, _, _ in items if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", cid)]
        if bad:
            raise ValueError(f"invalid custom_id(s): {bad[:3]}")
        est = self.estimate_usd(prefix, items[0][1], len(items))
        if self.ledger.would_exceed(est):
            raise BudgetExceeded(
                f"concurrent pre-estimate ${est:.2f} + spent ${self.ledger.spent:.2f} would exceed "
                f"cap ${self.ledger.cap:.2f}; not submitting")
        out: dict = {cid: None for cid, _, _ in items}   # default None -> GA fallback if a call never returns

        def one(item):
            cid, suffix, m = item
            try:
                text, _ = self.oc.propose_raw(prefix, suffix, **{**meta, **m})
                return cid, text
            except Exception:
                return cid, None      # -> GA falls back to random (same as api_error)

        ex = ThreadPoolExecutor(max_workers=self.max_workers)
        futs = [ex.submit(one, it) for it in items]
        try:
            for fut in as_completed(futs, timeout=self.round_deadline):
                cid, text = fut.result()
                out[cid] = text
        except TimeoutError:
            pass   # round deadline hit: stragglers stay None (fallback) instead of stalling forever
        ex.shutdown(wait=False, cancel_futures=True)     # don't block on hung requests
        return out
