"""Pipeline evaluation harness (zadani 1.1): stratified CV accuracy of a genotype on a CC-18 task,
with a result cache keyed by hash(genotype + dataset + cv config). Repeated genotypes in the population
must not be recomputed.

A pipeline that fails on the data (e.g. an incompatible config) returns fitness 0.0 with an error string,
so the search treats it as a bad candidate rather than crashing.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, warnings, joblib, numpy as np
from .data import load_task
from search import representation as R

warnings.filterwarnings("ignore")


def _key(geno: dict, task_id: int, cv_folds: int, eval_seed: int, row_cap: int) -> str:
    payload = json.dumps({"g": R.serialize(geno), "t": task_id, "cv": cv_folds,
                          "s": eval_seed, "rc": row_cap}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def _run_cv(geno: dict, task_id: int, cv_folds: int, eval_seed: int, row_cap: int) -> dict:
    """Stratified-CV accuracy of one genotype (module-level so it can run in a worker process for the
    G2 timeout). Any failure -> accuracy 0 with an error string."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    try:
        X, y, nf = load_task(task_id)
        if row_cap and len(X) > row_cap:
            rng = np.random.default_rng(eval_seed)
            idx = rng.choice(len(X), row_cap, replace=False)
            X, y = X[idx], y[idx]
        pipe = R.to_sklearn(geno, seed=eval_seed, n_features=nf)
        cv = StratifiedKFold(cv_folds, shuffle=True, random_state=eval_seed)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        return {"accuracy": float(np.mean(scores)), "std": float(np.std(scores)), "error": None}
    except Exception as e:
        return {"accuracy": 0.0, "std": 0.0, "error": str(e)[:200]}


def _cv_proc(q, *args):
    q.put(_run_cv(*args))


class Evaluator:
    def __init__(self, cv_folds: int = 5, eval_seed: int = 42, row_cap: int = 5000,
                 cache_dir: str | Path | None = "runs/_evalcache", eval_timeout: float | None = None):
        self.cv_folds, self.eval_seed, self.row_cap = cv_folds, eval_seed, row_cap
        self.eval_timeout = eval_timeout      # per-eval wall-clock cap (G2 safety); None = no timeout
        # cache_dir=None -> IN-MEMORY cache only (per-run, cold at start). This is the correct mode for a
        # budget-measured run: a disk cache that survives a killed+resumed run would make a resumed arm
        # reuse warm results, silently under-counting n_evals (the budget unit) and distorting the
        # equal-evaluation-budget comparison. Duplicates within a run are still cached via _mem.
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        self.n_evals = 0          # real (non-cached) evaluations — the budget unit (zadani 1.5)

    def _cache_path(self, k: str) -> Path:
        return self.cache_dir / f"{k}.json"

    def evaluate(self, geno: dict, task_id: int) -> dict:
        k = _key(geno, task_id, self.cv_folds, self.eval_seed, self.row_cap)
        if k in self._mem:
            return self._mem[k]
        if self.cache_dir is not None:
            cp = self._cache_path(k)
            if cp.exists():
                r = json.loads(cp.read_text()); self._mem[k] = r; return r
        r = self._run(geno, task_id)
        self._mem[k] = r
        if self.cache_dir is not None:
            self._cache_path(k).write_text(json.dumps(r))
        self.n_evals += 1          # count only genuine computations
        return r

    def _run(self, geno: dict, task_id: int) -> dict:
        args = (geno, task_id, self.cv_folds, self.eval_seed, self.row_cap)
        if not self.eval_timeout:
            return _run_cv(*args)
        # G2 safety: run the CV in a worker process and hard-kill it if it exceeds eval_timeout
        # (a Python signal can't interrupt a long sklearn C call; a process can be terminated).
        import multiprocessing as mp, queue
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_cv_proc, args=(q,) + args, daemon=True)
        p.start()
        try:
            r = q.get(timeout=self.eval_timeout)
        except queue.Empty:
            r = {"accuracy": 0.0, "std": 0.0, "error": f"timeout>{self.eval_timeout}s"}
        finally:
            p.terminate(); p.join()
        return r
