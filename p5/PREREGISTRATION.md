# Phase-3 pre-registration (dataset selection + analysis plan)

Written 2026-09-01, **before any Phase-3 run**, to fix the choices a reviewer would otherwise suspect
were made after seeing results. Nothing here depends on Phase-3 outcomes. The pilot (Phases 0-2,
`runs/pilot*/DECISION_GATE*.md`) is exploratory; Phase-3 is the confirmatory study this document governs.

## 1. Dataset selection criterion (fixed here)
Phase-3 uses `config/datasets.yaml : cc18_phase3` — 15 OpenML CC-18 tasks — instead of the zadani-1.4
"smallest 15" (`cc18_15`). Reason: the pilot showed the smallest tasks sit near ceiling (0.96-0.97), so
no method can separate from another and the equal-budget comparison (the paper's primary axis) is
uninformative there.

The 15 are selected by **task-intrinsic learnability only**, computed from historical OpenML sklearn-flow
evaluations (the `minimal_cache_cc18` bundle, ~3.78M runs), by two quantities:
- `ceiling` = 95th percentile of predictive accuracy over sklearn flows (best realistically achievable);
- `headroom` = ceiling - majority-class baseline (there is learnable signal iff this is well above 0).

Kept: ceiling in [0.56, 0.95] with real headroom, classes 2-10, sizes tractable (max n 9873, max f 1777;
the 60k-92k image tasks excluded for CV cost). **This criterion is a function of the task alone and is
independent of any LLM-vs-random-vs-AutoML outcome** — it cannot bias the comparison toward a conclusion
about the operator. Near-ceiling tasks (spambase 0.951, segment 0.937) are retained for range, so the set
is not "only tasks where methods differ".

## 2. The structure-decisive / hp-decisive split (the safeguard)
The LLM operator's measured weakness (pilot) is **structural narrowness**: ~90% of its proposals only
substitute hyperparameters inside an already-seen pipeline structure. A set chosen for headroom could, by
construction, over-represent tasks where *structure* choice is what matters — which would handicap the LLM
by design and make "the LLM has a structure problem" circular. To defuse this, every task is tagged
`decisive` by an **empirical variance decomposition** of the same historical evals:
- for each estimator class (the dominant structural axis), best-achievable (p95) and typical (p50) accuracy;
- `between_struct` = spread of best-achievable *across* estimator classes (gain from choosing structure);
- `within_hp` = median gain from HP tuning *within* a fixed structure;
- `structure` if between_struct >= within_hp, else `hp`.

Result: **9 structure-decisive, 6 hp-decisive** (cmc, cylinder-bands, mfeat-fourier, credit-approval,
phoneme, segment). hp-decisive tasks with headroom are genuinely rare in CC-18 — structure usually
dominates — so 9:6 is the most balanced split achievable, and that scarcity is itself a reportable finding.

The **6 hp-decisive tasks are the control subset**: on them the LLM's structural narrowness is *not* a
handicap. If the LLM shows no advantage even there, the negative result is not an artifact of task
selection. If it wins only there and loses on structure-decisive tasks, that is a weaker but honest claim.

## 3. Analysis plan (fixed here; updated per external review of 2026-09-02)
- **Primary budget = exactly equal genuine evaluations** (`Evaluator.n_evals`), never wall-clock. Every
  method (LLM-GA, random-GA, random-search) runs until it has spent the SAME number of genuine (non-cached)
  evaluations (GA methods run generations up to a safety ceiling until the budget is met; duplicates do not
  consume the budget). **Secondary axis: best-vs-proposals** (API calls / offspring), which exposes the
  real cost of an operator that duplicates or emits invalids. Both axes reported; the equal-genuine-eval
  axis is primary. (Fixes the confound that a GA proposing duplicates otherwise under-spends the budget.)
- **H1 is an EQUIVALENCE claim, tested with TOST**, not a difference test (a failed Wilcoxon is "absence of
  evidence"). **Pre-registered equivalence margin: ±0.005 accuracy (0.5 percentage points).** We report,
  per baseline, the per-task mean delta with a **bootstrap 95% CI** and the **TOST** equivalence verdict at
  ±0.005. Wilcoxon signed-rank is reported as secondary. Effect sizes always, not just p-values.
- **Report per-task**, not only aggregate; report the **structure-decisive (9)** and **hp-decisive (6)**
  subsets **separately**. The hp-decisive subset is a **descriptive plausibility control** (n=6, no
  statistical power claimed).
- **Operator characterization (2.3 metrics)**: invalid rate + violation types; duplicate rate vs population
  and vs history **plus duplicate concentration** (entropy of which genotypes are revisited — H4);
  **three diversity levels** — (i) structural (entropy of distinct pre|fs|est structures), (ii) genotypic
  (edit distance incl. HP), (iii) SNCS-D0 embedding; hp-substitution vs new-structure rate. Structural
  novelty and **space coverage are reported time-resolved (per generation / per evaluation)**, not only
  aggregated, because the base space has only **63 structures** and aggregate novelty saturates.
- **Model is a FIRST-CLASS axis, not an ablation (updated 2026-09-03 after the Qwen pilot).** The base and
  core mechanism are run on TWO CO-PRIMARY models at full scale (15 tasks × 5 seeds each): a closed
  frontier-small model (**Haiku**, via the Anthropic Batch API) and a large **open-weight** model
  (**Qwen2.5-72B**, via an OpenAI-compatible endpoint). Cross-model consistency IS the robustness result
  ("no equal-budget advantage, structural narrowness, on both"). A reduced closed-model-size check
  (Sonnet, 15×2) is retained as a secondary cross-check.
- **Ablations** (gated after base, one axis at a time), run on the **Qwen co-primary** (open, reproducible,
  cheap; 15 tasks × 3 seeds each): fitness-in-prompt on/off; population-history on/off; mutation-only vs
  mutation+crossover; **sampling temperature**. Plus a **prompt ablation** (no worked examples / an explicit
  instruction to propose new structures) to test whether the structural-narrowness signature is a property
  of the model or few-shot anchoring — code path to be added. (Worked-example structures are NOT pre-seeded
  into the "seen" set, so novelty already does not credit them for free; the ablation tests behavioural
  copying.)
- **Fitness = accuracy** (as in the pilot); balanced accuracy reported secondarily. Row-capping: datasets
  are capped at 5000 rows for a bounded per-evaluation cost (recorded in config).

## 4. What would change the conclusion (stated in advance)
- **H1 (no advantage) is overturned** if, across the 15 tasks with bootstrap CIs, LLM-GA's per-task mean
  best-accuracy exceeds a baseline by a margin whose 95% CI excludes zero on the equal-genuine-eval axis —
  including on the hp-decisive control subset.
- **H1 is positively supported as equivalence** if TOST rejects at ±0.005 (the LLM's advantage/disadvantage
  is bounded within half a point).
- **H2 (structural narrowness)** is judged a model property only if it **survives the prompt ablation** (no
  worked examples / explicit new-structure instruction); otherwise it is reported as prompt-induced.
- **H7 (space size, secondary):** the LLM-vs-random gap does not grow in the LLM's favour beyond the ±0.005
  margin when moving from the 63-structure space (G1) to the larger G2 space (§7).

## 5. Revised hypotheses (external review of 2026-09-02, accepted)
- **H1** (main) — no *practically significant* advantage of the LLM operator at equal genuine-eval budget;
  tested as **equivalence** within ±0.005 (TOST) + CI, not a difference test.
- **H2** (mechanism) — structural narrowness (high hp-substitution, low new-structure, low structural
  entropy at high genotypic diversity); **must survive the prompt ablation** to be claimed as a model
  property.
- **H3** (validity) — near-perfect grammar validity under a correct schema prompt.
- **H4** (self-repetition) — **reformulated**: the LLM revisits a *narrow, concentrated* region
  (low duplicate-entropy), whereas random variation duplicates more but *diffusely*. (The naive
  "LLM wastes more budget on duplicates" is FALSE — the pilot shows the opposite: LLM dup-vs-history 0.13
  vs random 0.37 — so it is dropped.)
- **H5** (control) — H1 holds on the 6 hp-decisive tasks; **descriptive plausibility check, n=6, no
  statistical power claimed**.
- **H6** (ablations) — fitness/history/crossover/temperature/model do not overturn H1.
- **H7** (space size, secondary) — the gap does not move in the LLM's favour beyond ±0.005 from G1
  (63 structures) to G2 (§7).

## 6. Accepted review actions
- Random operator: its ~37% duplicate rate (confirmed cold, not a cache artefact) comes from single-stage
  mutation + crossover of a converged elite population; it is **not** fixed but **documented**, and the
  equal-genuine-eval budget (which lets it run more generations) removes the fairness impact. Report both
  proposal and evaluation axes.
- Model ablation broadened to an open-weight model (reproducibility; a frozen closed model drifts).
- Classical AutoML: **TPOT only, as calibration with an explicit incomparable-space/budget caveat, or
  omitted**; no half-integrated GAMA before the deadline.
- Positioning: differentiate axis 2 explicitly from **Language Model Crossover (LMX)** — our novelty is the
  grammar-constrained space + the operator-behaviour measurement program, not crossover per se.
- Infrastructure detail (batch driver, cost ledger, cache) → **supplement / repository**, ≤2 sentences in
  the paper body.
- "Independent of any method's outcome" reworded to "independent of the outcomes of the methods **in this
  paper**" (historical CC-18 accuracies come from other methods' runs).

## 7. Secondary experiment G2 (larger space) — pre-registered design
- **Purpose:** turn the "bounded space" limitation into a result — does the LLM-vs-random gap change with
  space size? Keep G1 (63 structures) as PRIMARY (pilot/prompt/preregistration are bound to it, and a
  prompt-describable space is where the LLM *should* have an edge — that it does not is the stronger claim).
- **Construction:** extend, not replace — same genotype format, YAML grammar, prompt schema, operator
  interface, random operator, harness; change only the grammar block. Target G2 ≈ 1,560 structures
  (5 preprocessing × 4 optional feature-engineering × 6 feature-selection × 13 estimators); every G1
  genotype remains a valid G2 genotype (G1 ⊂ G2), enabling a direct G1-vs-G2 comparison.
- **Safety:** per-evaluation **timeout** (≈5× the G1 median) → fitness 0 with an error, like a crash;
  polynomial features restricted to low-dimensional datasets; row-cap possibly lowered (recorded).
- **Design:** 5 structure-decisive tasks (largest headroom) × 3 seeds × {LLM-GA, random-GA}, equal
  genuine-eval budget as the matching G1 runs; Haiku. Report Δ(LLM−random) on G1 vs G2 with CIs, and
  structural novelty / coverage **per generation** on G1 vs G2. Estimated **~$50–130 (batch)**, within the
  program reserve. If the primary G2 result shows signal, extend the model ablation to G2.

## 8. Two co-primary models — decision + Qwen pilot evidence (2026-09-03)
The single-closed-model risk (review §3.7) is resolved by making model a first-class axis: **Haiku (closed,
small) and Qwen2.5-72B (open, large) are co-primary**, run at full scale, and the finding is claimed only
where it holds on BOTH. A Qwen pilot on the 3 hard tasks (same protocol as the Haiku pilot) supports this:

| | Haiku | Qwen2.5-72B |
|---|---|---|
| LLM vs random-GA (per-task Δ) | −0.006 | −0.014 (1W/8L) |
| hp-substitution / new-structure | 0.916 / 0.084 | 0.949 / 0.051 |
| embedding diversity (SNCS-D0) | 0.019 | 0.013 |
| dup-vs-history / concentration | 0.126 / — | 0.380 / 0.920 |

Readings: the mechanism (structural narrowness, embedding collapse, concentrated self-repetition)
**replicates and is stronger on the larger open model**, and the no-advantage result is **more decisive**
on Qwen — so a bigger/open model does not rescue the operator, killing the "weak-model" objection. Qwen's
raw invalid rate (~0.04) is 100% network-straggler `api_error` (concurrent-client round-deadline
fallbacks), NOT model format errors; its true validity matches Haiku's.

Reproducibility caveat: DeepInfra serving (quantization/stack) can shift; full reproducibility requires the
open weights + exact revision, which we record. Still far better than a closed model that is deprecated.

Config set: `phase3_base.yaml` (Haiku 15×5) + `phase3_base_qwen.yaml` (Qwen 15×5) = co-primary bases;
`phase3_qwen_abl_*` (fitness/history/crossover/temperature, 15×3) = ablation suite on Qwen;
`phase3_abl_sonnet_reduced.yaml` (15×2) = closed-model-size check; `phase3_g2.yaml` = larger-space secondary.
Full-program batch cost ≈ Haiku base $171 + Qwen base $42 + 4 Qwen ablations ~$100 + Sonnet $124 + G2 ~$90
≈ **$530** (+ optional).
