"""Population diversity, two ways (zadani 2.3):
  - structural: mean pairwise edit distance over the pipeline structure (stage + params),
  - embedding:  mean pairwise cosine distance in the SNCS D0 embedding space.

The embedding metric reproduces SNCS D0 (DOI 10.1007/s42979-026-05295-9): the same
off-the-shelf `all-MiniLM-L6-v2` encoder applied to the same `cleanme()`-cleaned textual
rendering of a scikit-learn flow. SNCS embeds the OpenML flow text; we render each genotype
to its `str(to_sklearn(g))` sklearn repr and clean it identically, so distances are directly
comparable to that paper. Note `cleanme` strips all digits, so hyperparameter *values* do not
enter the embedding: this space measures structural / component-level variation (a within-
structure HP substitution moves ~0 here), which is exactly the intended reading.
Both take a list of genotypes and return a scalar in roughly [0, 1].
"""
from __future__ import annotations
import itertools, json, re, numpy as np
from search import representation as R

_MODEL = None
_EMB_CACHE: dict[str, np.ndarray] = {}

# SNCS cleanme(), verbatim (sncs_experiments/sncs_workflow_performance.ipynb, cell 4).
_CLEAN_MAP = str.maketrans({".": " ", "(": " ", ")": " ", "_": " ", ",": " ", "=": " "})


def cleanme(s: str) -> str:
    s1 = s.translate(_CLEAN_MAP)
    s1 = re.sub(r"\s[0-9a-fA-F]+\s", " ", s1)
    s1 = re.sub(r"\d+", "", s1)
    s1 = re.sub(r"TEST\S+", " ", s1)
    s1 = re.sub(r"\sCx\S+", " ", s1)
    return s1


def _pipeline_text(g: dict) -> str:
    """SNCS-comparable text for one genotype: cleaned sklearn-flow repr (fallback: cleaned JSON)."""
    try:
        return cleanme(str(R.to_sklearn(g)))
    except Exception:
        return cleanme(R.serialize(g))


def struct_distance(a: dict, b: dict) -> float:
    """0 = identical structure, 1 = every stage differs. Per stage: 1 if the option differs, else the
    fraction of differing parameter values."""
    d = 0.0
    for st in R.STAGES:
        na, nb = a[st]["name"], b[st]["name"]
        if na != nb:
            d += 1.0
        else:
            pa, pb = a[st].get("params", {}), b[st].get("params", {})
            keys = set(pa) | set(pb)
            if keys:
                d += sum(1 for k in keys if str(pa.get(k)) != str(pb.get(k))) / len(keys)
    return d / len(R.STAGES)


def structural_diversity(genos: list[dict]) -> float:
    """GENOTYPIC diversity: mean pairwise edit distance over the whole genotype (structure + HP values).
    High even when only hyperparameters differ -- see structure_entropy for structure-only diversity."""
    if len(genos) < 2:
        return 0.0
    ds = [struct_distance(a, b) for a, b in itertools.combinations(genos, 2)]
    return float(np.mean(ds))


def structure_entropy(genos: list[dict]) -> float:
    """STRUCTURAL diversity: normalized Shannon entropy of the (pre|fs|est) structure distribution,
    in [0,1] (0 = one structure, 1 = uniform over the population). Ignores hyperparameters, so a
    population that only tweaks HP within one structure scores ~0 here even if genotypic diversity is
    high -- this is the level that separates HP-jitter from genuine structural exploration."""
    import math
    from collections import Counter
    if len(genos) < 2:
        return 0.0
    c = Counter(R.structure_sig(g) for g in genos)
    n = len(genos)
    H = -sum((v / n) * math.log(v / n) for v in c.values())
    return float(H / math.log(n))


def _encoder():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")   # SNCS D0 encoder (off-the-shelf)
    return _MODEL


def _embed(texts: list[str]) -> np.ndarray:
    todo = [t for t in texts if t not in _EMB_CACHE]
    if todo:
        vecs = _encoder().encode(todo, normalize_embeddings=True, show_progress_bar=False)
        for t, v in zip(todo, vecs):
            _EMB_CACHE[t] = np.asarray(v, dtype=float)
    return np.stack([_EMB_CACHE[t] for t in texts])


def embedding_diversity(genos: list[dict]) -> float:
    """Mean pairwise cosine distance (1 - cos) of genotypes in the SNCS D0 embedding space."""
    if len(genos) < 2:
        return 0.0
    V = _embed([_pipeline_text(g) for g in genos])        # rows are unit vectors
    S = V @ V.T
    iu = np.triu_indices(len(genos), k=1)
    return float(np.mean(1.0 - S[iu]))


def diversity_report(serialized_pop: list[str]) -> dict:
    """Three levels (zadani 2.3 / review §3.6): structural (structure-only entropy), genotypic (edit
    distance incl. HP), embedding (SNCS-D0 cosine)."""
    genos = [json.loads(s) for s in serialized_pop]
    return {"structural": structure_entropy(genos),
            "genotypic": structural_diversity(genos),
            "embedding": embedding_diversity(genos)}
