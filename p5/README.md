# P5 — LLM as a variation operator in evolutionary AutoML pipeline search

Experimental infrastructure for the paper. Anonymous supplementary material (LNCS, double-blind).

## Anonymization (keep from the first commit)
No author names, institution, or absolute local paths in code, configs, or commit messages. Paths are
repo-relative; run outputs live in `runs/` (git-ignored). The repository is released as an anonymized link.

## Layout
```
config/     YAML run configs (one file = one experiment) + pricing + grammar
search/     GA, pipeline representation, operator interface
operators/  llm_operator.py, random_operator.py (shared interface propose(parents, context) -> candidate|None)
llm/        Anthropic client, prefix cache, cost accounting + hard budget cap
eval/       pipeline evaluation harness (CV, result cache)
metrics/    diversity, regret, validity
runs/       outputs / jsonl logs (never committed)
analysis/   scripts -> LaTeX tables (one script = one table)
tests/      phase-0 guards (prefix byte-identity, budget cap)
```

## Budget safety (Phase 0)
- `llm/accounting.py` — append-only jsonl ledger; per-call cost from `config/pricing.yaml`; hard cumulative
  cap that halts the run when exceeded.
- `llm/client.py` — Messages client with an ephemeral **prefix cache** (static instructions+grammar), a
  `preflight` cost estimate via `count_tokens` that refuses to start if the estimate would exceed the cap.
- Prefix must stay byte-identical across calls or the cache silently misses; `tests/test_phase0.py` guards it.

Run tests: `python tests/test_phase0.py`

Key resolution: `ANTHROPIC_API_KEY` env, then `~/.p5_anthropic_key`, then `~/.anthropic_key`.

## Status
- **Phase 0 (done):** repo skeleton, accounting ledger + hard cap, prefix cache + byte-identity test,
  preflight. `python tests/test_phase0.py`.
- **Phase 1 (core done):** representation over `config/grammar.yaml` (validate/serialize/parse/sample/build);
  eval harness `eval/` (CC-18 load+cache, stratified CV, result cache, `n_evals` budget counter); GA
  `search/ga.py` (tournament, elitism, eval-budget, generic fallback+logging); baselines RandomOperator
  and random search; 15 fixed CC-18 tasks in `config/datasets.yaml`. Smoke-tested live on task 11; offline
  tests `python tests/test_phase1.py`.
- **Phase 1 baselines still to wire (before Phase 3):** P3 warm-start (seed the GA population from the P3
  recommender order), P4 one-shot recommender, and one off-the-shelf system (TPOT or GAMA) for community
  calibration. The pilot decision gate only needs LLM-operator vs random-operator vs random-search at
  equal budget, which is ready.
- **Phase 2 (done, pilot awaiting preflight):** LLM operator `operators/llm_operator.py` (cached prefix,
  one retry, None-on-invalid -> GA logs fallback); metrics `metrics/diversity.py` (structural + MiniLM
  embedding) and `metrics/search_metrics.py` (nregret, evals-to-target); GA tracks invalid/duplicate rate.
  Pilot runner `run_pilot.py` preflights the LLM cost (est ~\$0.58 for the 3x3 pilot, cap \$15) and waits
  for confirmation. Run: `python run_pilot.py config/pilot.yaml` (or `P5_PREFLIGHT_ONLY=1` for estimate).
- **Decision gate (end Sept):** after the pilot, report invalid/duplicate rate, LLM-op vs random-op vs
  random-search at equal budget, diversity, and extrapolated full-run cost; do not start Phase 3 without
  consultation (zadani). Reports: `runs/pilot/DECISION_GATE.md` (easy), `runs/pilot_hard/DECISION_GATE_HARD.md`
  (hard tasks), `runs/pilot_hard_v2/DECISION_GATE_V2.md` (corrected operator prompt). Headline: at equal
  budget the frozen LLM operator gives no search advantage, with a clean structural-narrowness mechanism.

## Phase-3 configs and the Batch-API driver (prepared; GATED on the consultation)
- **Dataset set:** `config/datasets.yaml : cc18_phase3` — 15 CC-18 tasks, "set B + safeguard": a
  representative ceiling spread WITH headroom (0.56-0.95), REPLACING the near-ceiling smallest-15 (the pilot
  showed those cannot separate methods). Selection is by task-intrinsic learnability only (outcome-
  independent) and each task is tagged `decisive: structure|hp` from a variance decomposition; the **6
  hp-decisive tasks are a control subset** (the LLM's structural narrowness is not a handicap there) that
  defuses the circular-cherry-pick objection. Full rationale + analysis plan pre-registered in
  `PREREGISTRATION.md` (written before any Phase-3 run).
- **Two CO-PRIMARY models (not primary+ablation):** `config/phase3_base.yaml` (Haiku, closed) and
  `config/phase3_base_qwen.yaml` (Qwen2.5-72B, open-weight) — both 15×5, pop50/30g, full operator, equals.
  Cross-model consistency IS the robustness result. Single-axis ablations run on the Qwen co-primary
  (`config/phase3_qwen_abl_{nofitness,nohistory,mutonly,temperature}.yaml`, 15×3). `phase3_abl_sonnet_reduced`
  (15×2) is a closed-model-size cross-check. Each config is preflight- and hard-cap-gated. See
  `runs/pilot_qwen/QWEN_VS_HAIKU.md` for the pilot that motivated this.
- **Batch API (50% discount):** `run_phase3.py` advances all (task,seed) LLM runs in lockstep and sends one
  Batch-API request per generation carrying every run's offspring (~3600 requests x 30 rounds). Batched LLM
  cost is logged at the halved price (`Ledger.record(batch=True)`). Random-GA / random-search arms run
  offline (free). `search/ga_batch.py` (`BatchGA`) mirrors `search/ga.py` metrics exactly but steps a
  generation at a time; `llm/batch_client.py` wraps submit/poll/retrieve; `operators/llm_operator.py` gained
  offline `build()`/`interpret()` (used by both the synchronous and batch paths). Batch also survives flaky
  networks (submit-once + poll, no multi-hour socket hang).
- **Open-weight executor:** `run_phase3` picks the executor by model vendor — Anthropic → server-side Batch
  API (50% off); otherwise `llm/openai_client.py` `ConcurrentBatchClient` (OpenAI-compatible/DeepInfra) with
  the same `run_batch()` interface via a concurrent thread pool + per-round deadline (a hung request → random
  fallback, never stalls — a robustness bug the Qwen pilot caught and fixed). Needs `~/.deepinfra_key`.
- **Cost (batch/concurrent, from live preflight):** Haiku base ~\$171 (batch), Qwen base ~\$42, each Qwen
  ablation ~\$25, Sonnet check ~\$124, G2 ~\$90 → recommended program ≈ **\$530**. Haiku prefix (1889 tok) is
  below Haiku's empirical 4096-token cache threshold, so haiku does not cache; sonnet does; Qwen has no cache.
- **Larger space G2 (C3, secondary):** `config/grammar_g2.yaml` (585 structures vs 63; G1 ⊂ G2) +
  `config/phase3_g2.yaml` (5 tasks × 3 seeds, H7). `to_sklearn` compiles all G2 components; every G2
  genotype validates, builds, and fits (verified). Per-eval `eval_timeout` (process hard-kill) guards
  pathological pipelines (e.g. polynomial deg-3 + svc). Turns "bounded space" into a result.
- **Run order:** both co-primary bases (Haiku + Qwen) first, review, then the Qwen ablation suite, then the
  Sonnet size-check and the G2 secondary experiment.
  `P5_PREFLIGHT_ONLY=1 python run_phase3.py config/phase3_base.yaml` estimates without spending. Tests:
  `python tests/test_phase3.py` (offline BatchGA), plus a live 2-request batch smoke was verified.

## Fairness rule (equal evaluation budget)
The budget unit is a genuine (non-cached) pipeline evaluation (`Evaluator.n_evals`). To keep comparisons
fair, **each (method, task, seed) run gets its own `Evaluator` with its own `cache_dir`** — never share a
cache across methods, or a later method gets free cache hits and effectively a larger budget.

## Adopted zadani v2 (literature-updated)
- `POSITIONING.md` (phase 0): four-axis contribution + measurement program + what the paper does NOT
  claim + three nearest works (LLM-FE, LLaMEA, SemPipes).
- Metrics are first-class (2.3): added duplicate-vs-whole-history, invalid-violation breakdown, and the
  structural-novelty metric (hp-substitution vs new-structure, Gurkan et al.) in `search/ga.py`.
- LLM operator does mutation AND crossover (`operators/llm_operator.py`); crossover is a positioning axis.
- Decision gate changed: a well-measured diversity/validity collapse is publishable even if the LLM
  operator does not win; stop only on no-signal / inconsistent infra / over-budget.
- `BIBLIOGRAPHY.md` (phase 4) skeleton created; all bibliographic fields `TODO-verify` pending a
  web-verification pass (no from-memory citations).
