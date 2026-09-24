# Reproducing the recommender-papers experiments

Canonical scripts behind the four companion papers (P1 method/CC18, P2 measurement/heterogeneity,
P3 warm-start/systems, P4 LLM-as-recommender). The core library is `../openml_flow.py`; these scripts
drive it and the ancillary analyses.

## Setup

**Base environment** (arm64/any, Python 3.11+):
```
pip install -r ../requirements.txt          # + lightgbm, optuna, shap, statsmodels, anthropic
```
- Run every script **from the repo root** (paths are relative: `scratch/…` for intermediates,
  `minimal_cache_cc18/…` for the OpenML disk cache, `results_recommender_*/` for per-run outputs).
- Create `./scratch` at the repo root (or `ln -s` it to an existing data dir); scripts read/write their
  intermediates there and the data-build scripts below seed the OpenML caches.
- OpenML access: `OPENML_API_KEY` env var (see `configure_openml` in `openml_flow.py`).

**auto-sklearn 2.0 environment** (P3 §4.0b only — needs a legacy stack; on Apple Silicon uses osx-64):
```
bash recommender/setup_asklearn_env.sh      # turnkey: creates env `asklearn64`, installs the
                                                        # pinned stack, patches pynisher, verifies import
```
This is fully scriptable (osx-64 conda env, pinned scikit-learn 0.24.2 / numpy 1.21 / scipy 1.7 / pyrfr /
swig, `auto-sklearn==0.15.0 --no-build-isolation`, `pandas==1.5.3`, and the macOS pynisher `RLIMIT_AS`
no-op patch). Purge with `conda env remove -n asklearn64 && conda clean -a` and recreate anytime by
re-running the script. auto-sklearn 2.0 = `autosklearn.experimental.askl2.AutoSklearn2Classifier`
(a class inside the package, not a separate release). Use `memory_limit=<positive int>` (None trips an
assert in 0.15).

## Data build (regenerates the caches from OpenML; slow, flaky endpoint — resumable)
- `longtail_full_scan.py` — scan all OpenML classification tasks, keep dense (>=50 flows) → 1,236-task long tail.
- `longtail_full_run.py` — assemble the long-tail supervised bundle (cached in `minimal_cache_cc18/longtail_*`).
- `longtail_qualities_full.py` — fetch dataset qualities (basic/landmarking/openml_full).
- `longtail_runtime.py` — fetch per-run `usercpu_time_millis` (P3 cost-aware).
- `longtail_census.py` — the sizing census (how many dense tasks exist).
CC18 bundle + flows are built by `openml_flow.load_cc18_from_openml` / `load_or_cache_flows`.

## Paper 1 — method / CC18
| script | produces |
|---|---|
| `fullpool.py` | §5.1b full-pool (deployment) ranking vs candidate-only (fix C) |
| `runs_confound.py` | §5.8 agg=max / #uploaded-runs confound (fix G) |
| `shap_alignment.py` | §5.9 quantitative SHAP-alignment metric (fix I; extra_trees TreeSHAP) |
| `cis_correction.py` | Table 1 bootstrap 95% CIs + Holm/BH correction (fix K) |
| `llm_pack_cc18.py` | §5.9 frozen-LLM baseline pack (family-level, complete CC18) |

## Paper 2 — measurement / heterogeneity
| script | produces |
|---|---|
| `als_showdown.py` | §5.6 text-LTR vs tuned ALS-PMF collaborative filter (fix B) |
| `factorial_canonical.py` | §5.5 factorial H×M interaction test on the canonical harness (fix E) |
| `heterogeneity_curve.py` | Fig: adaptation value vs task atypicality |
| `run_aggmean.py` | §5.8 agg=mean robustness |
| `run_toolkit_sweep.py` | cross-toolkit sweep (flow diversity != task diversity) |
| `run_enriched.py` | LLM-authored semantic text enrichment |

## Paper 3 — warm-start / systems
| script | produces |
|---|---|
| `warmstart_search.py` | real-search harness + per-task recommender/portfolio family orderings |
| `our_search_exp1.py` | wall-clock-budgeted Optuna search (Exp 1/2; env: `BUDGET`,`CONDS`,`SHARD`,`RESPATH`,`BASELINE`) |
| `asklearn_prep.py` | prep CC18 train/test arrays for auto-sklearn (base env) |
| `asklearn_exp1.py` | run AutoSklearn2Classifier at a budget (asklearn64 env; `BUDGET`,`SHARD`,`RESPATH`) |
| `analyze_realsearch.py` | Optuna real-search: evals/time-to-target, anytime regret |
| `analyze_asklearn.py` | Exp 1/2 at one budget: our-rec vs auto-sklearn 2.0 + order isolation |
| `analyze_crossover.py` | §4.0b matched 60s-vs-300s crossover vs auto-sklearn 2.0 |
| `fullpool_longtail.py` | §4.1b full-pool deployment on the long tail |
| `cost_aware_search.py` | §4.4 cost-aware ordering (accuracy-per-runtime) |
| `online_search.py` | §4.3 online within-task updating (null result) |

## Paper 4 — LLM-as-recommender
| script | produces |
|---|---|
| `llm_paper4.py` | main harness (build / dryrun / run <model> [N] / eval); cached-prefix Anthropic calls |
| `llm_sig.py` | paired significance vs baselines + model-tier tests |
| `llm_ablation.py` | thinking ablation (Sonnet think vs no-think) + per-task effort analysis |
Needs `ANTHROPIC_API_KEY` (env or `~/.anthropic_key`). Models: `claude-sonnet-5`, `claude-haiku-4-5-20251001`,
`claude-opus-4-8`, `claude-fable-5`. Claude-5 gotchas: no `temperature`; extract the `type=="text"` block
(thinking blocks precede it); `thinking={"type":"disabled"}` for lean output (or `adaptive`+`output_config.effort`).

## Notes
- Per-run outputs land in `results_recommender_*/` (gitignored, regenerable); paper drafts live in
  `papers/recommender/` (local). Multi-hour runs: thread-pin (`OMP_NUM_THREADS=1` …) and shard across
  cores; each parallel writer needs its own `RESPATH` (no shared-file joblib race).
- `random_state=42`, `cv_folds=5` are the standard defaults (see `../CLAUDE.md`).
