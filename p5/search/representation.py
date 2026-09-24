"""Pipeline representation over the grammar (zadani 1.2): validate, serialize, parse, sample, build.

A genotype is a structured dict:
    {"preprocessing":     {"name": str, "params": {...}},
     "feature_selection": {"name": str, "params": {...}},
     "estimator":         {"name": str, "params": {...}}}
The three operations the search needs: validate against the grammar, serialize to text (prompt/history/
hash), parse text back (from the LLM), plus random sampling and construction of an sklearn Pipeline.
"""
from __future__ import annotations
import json, re, math
import yaml

STAGES = ["preprocessing", "feature_selection", "estimator"]


def load_grammar(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def grammar_text(grammar: dict) -> str:
    """Deterministic text of the grammar for the (cached) prompt prefix."""
    return yaml.safe_dump(grammar, sort_keys=True, default_flow_style=False)


# ---------- sampling ----------
def _sample_param(spec: dict, rng):
    t = spec["type"]
    if t == "cat":
        return rng.choice(spec["choices"])
    if spec.get("optional") and rng.random() < 0.25:
        return None
    if t == "int":
        if spec.get("log"):
            lo, hi = math.log(float(spec["low"])), math.log(float(spec["high"]))
            return int(round(math.exp(rng.uniform(lo, hi))))
        return int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
    if t == "float":
        if spec.get("log"):
            lo, hi = math.log(float(spec["low"])), math.log(float(spec["high"]))
            return float(math.exp(rng.uniform(lo, hi)))
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    raise ValueError(f"unknown param type {t!r}")


def _sample_stage(grammar, stage, rng) -> dict:
    name = rng.choice(list(grammar[stage].keys()))
    spec = grammar[stage][name]
    params = {p: _sample_param(s, rng) for p, s in spec.items()}
    params = {k: v for k, v in params.items() if v is not None}   # drop optional-None
    return {"name": str(name), "params": params}


def random_genotype(grammar: dict, rng) -> dict:
    return {st: _sample_stage(grammar, st, rng) for st in STAGES}


# ---------- validation ----------
def validate(g: dict, grammar: dict) -> tuple[bool, str]:
    if not isinstance(g, dict):
        return False, "not a dict"
    for st in STAGES:
        if st not in g or not isinstance(g[st], dict) or "name" not in g[st]:
            return False, f"missing/malformed stage {st}"
        name = g[st]["name"]
        if name not in grammar[st]:
            return False, f"{st}: unknown option {name!r}"
        spec = grammar[st][name]
        params = g[st].get("params", {})
        if not isinstance(params, dict):
            return False, f"{st}.params not a dict"
        for p, v in params.items():
            if p not in spec:
                return False, f"{st}.{name}: unknown param {p!r}"
            s = spec[p]
            if s["type"] == "cat":
                if str(v) not in [str(c) for c in s["choices"]]:
                    return False, f"{st}.{p}={v!r} not in choices"
            elif s["type"] in ("int", "float"):
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return False, f"{st}.{p}={v!r} not numeric"
                if not (float(s["low"]) <= fv <= float(s["high"])):
                    return False, f"{st}.{p}={v} out of range [{s['low']},{s['high']}]"
    return True, "ok"


# ---------- serialize / parse ----------
def n_structures(grammar: dict) -> int:
    """Number of distinct (preprocessing, feature_selection, estimator) name-combinations the grammar
    allows -- the size of the STRUCTURE space (ignoring hyperparameters). Used for space-coverage."""
    n = 1
    for st in STAGES:
        n *= len(grammar[st])
    return n


def structure_sig(g: dict) -> str:
    """Structural signature = the three stage OPTION names, ignoring hyperparameter values.
    Two genotypes share a signature iff they use the same components; used to separate
    hyperparameter-substitution from genuinely new structure (Gurkan et al.)."""
    return "|".join(g[st]["name"] for st in STAGES)


def serialize(g: dict) -> str:
    return json.dumps(g, sort_keys=True, separators=(",", ":"))


def parse(text: str) -> dict | None:
    """Extract the first JSON object from LLM text, tolerant of code fences / preamble."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(json)?|```$", "", t, flags=re.MULTILINE).strip()
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{": depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------- build sklearn Pipeline ----------
def to_sklearn(g: dict, seed: int = 42, n_features: int | None = None):
    """Compile a genotype to an sklearn Pipeline. Handles both the G1 grammar and the larger G2 grammar
    (config/grammar_g2.yaml) -- the extra component names are a superset; G1 branches are unchanged."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
                                       PolynomialFeatures, KBinsDiscretizer)
    from sklearn.feature_selection import (SelectKBest, SelectPercentile, f_classif, mutual_info_classif,
                                           SelectFromModel, VarianceThreshold)
    from sklearn.decomposition import PCA
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier, GradientBoostingClassifier,
                                  AdaBoostClassifier)
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier

    def _clamp(v, lo=1):
        return max(lo, min(int(v), n_features)) if n_features is not None else int(v)

    pn = g["preprocessing"]["name"]; prp = g["preprocessing"].get("params", {})
    if pn == "none":            pre = "passthrough"
    elif pn == "standardize":   pre = StandardScaler()
    elif pn == "minmax":        pre = MinMaxScaler()
    elif pn == "robust":        pre = RobustScaler()
    elif pn == "quantile":      pre = QuantileTransformer(
        output_distribution=prp.get("output_distribution", "uniform"), random_state=seed)
    else:                       raise ValueError(f"unknown preprocessing {pn!r}")

    fs = g["feature_selection"]; fsp = fs.get("params", {}); fn = fs["name"]
    if fn == "none":                fsel = "passthrough"
    elif fn == "selectk":           fsel = SelectKBest(f_classif, k=_clamp(fsp.get("k", 10)))
    elif fn == "select_percentile": fsel = SelectPercentile(f_classif, percentile=int(fsp.get("percentile", 50)))
    elif fn == "mutual_info":       fsel = SelectKBest(mutual_info_classif, k=_clamp(fsp.get("k", 10)))
    elif fn == "pca":               fsel = PCA(n_components=_clamp(fsp.get("n", 5)), random_state=seed)
    elif fn == "model_based":       fsel = SelectFromModel(
        LogisticRegression(C=float(fsp.get("C", 1.0)), penalty="l1", solver="liblinear", max_iter=200))
    elif fn == "polynomial":        fsel = PolynomialFeatures(
        degree=int(fsp.get("degree", 2)), interaction_only=bool(fsp.get("interaction_only", False)),
        include_bias=False)
    elif fn == "binning":           fsel = KBinsDiscretizer(
        n_bins=int(fsp.get("n_bins", 5)), strategy=fsp.get("strategy", "uniform"), encode="ordinal")
    elif fn == "variance_threshold": fsel = VarianceThreshold(threshold=float(fsp.get("threshold", 0.0)))
    else:                           raise ValueError(f"unknown feature_selection {fn!r}")

    e = g["estimator"]; p = dict(e.get("params", {})); name = e["name"]
    mf = p.get("max_features")
    if mf is not None:
        try: mf = float(mf)      # "0.5" -> 0.5; keep "sqrt"/"log2" as strings
        except (TypeError, ValueError): pass
    _md = lambda: (int(p["max_depth"]) if p.get("max_depth") is not None else None)
    if name == "random_forest":
        clf = RandomForestClassifier(n_estimators=int(p.get("n_estimators", 200)), max_depth=_md(),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)), max_features=mf, random_state=seed, n_jobs=1)
    elif name == "extra_trees":
        clf = ExtraTreesClassifier(n_estimators=int(p.get("n_estimators", 200)), max_depth=_md(),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)), max_features=mf, random_state=seed, n_jobs=1)
    elif name == "hist_gbrt":
        clf = HistGradientBoostingClassifier(learning_rate=float(p.get("learning_rate", 0.1)),
            max_leaf_nodes=int(p.get("max_leaf_nodes", 31)), max_iter=int(p.get("max_iter", 100)),
            l2_regularization=float(p.get("l2_regularization", 1e-6)), random_state=seed)
    elif name == "logistic":
        clf = LogisticRegression(C=float(p.get("C", 1.0)), max_iter=500)
    elif name == "svc":
        clf = SVC(C=float(p.get("C", 1.0)), gamma=p.get("gamma", "scale"))
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=int(p.get("n_neighbors", 5)), weights=p.get("weights", "uniform"))
    elif name == "gaussian_nb":
        clf = GaussianNB()
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(int(p.get("hidden", 100)),), alpha=float(p.get("alpha", 1e-4)),
            learning_rate_init=float(p.get("lr_init", 1e-3)), max_iter=300, random_state=seed)
    elif name == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=int(p.get("n_estimators", 100)),
            learning_rate=float(p.get("learning_rate", 0.1)), max_depth=int(p.get("max_depth", 3)),
            random_state=seed)
    elif name == "adaboost":
        clf = AdaBoostClassifier(n_estimators=int(p.get("n_estimators", 50)),
            learning_rate=float(p.get("learning_rate", 1.0)), random_state=seed)
    elif name == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=_md(), min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            random_state=seed)
    elif name == "ridge":
        clf = RidgeClassifier(alpha=float(p.get("alpha", 1.0)))
    elif name == "sgd":
        clf = SGDClassifier(loss=p.get("loss", "hinge"), alpha=float(p.get("alpha", 1e-4)),
            penalty=p.get("penalty", "l2"), max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"unknown estimator {name!r}")
    return Pipeline([("pre", pre), ("fs", fsel), ("clf", clf)])
