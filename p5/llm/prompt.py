"""Prompt construction for the LLM variation operator, split into a static (cached) prefix and a
variable suffix (zadani 0.3).

The static prefix = instructions + pipeline-space (grammar) description. It must be byte-identical across
every call in a run so the ephemeral prefix cache actually hits; silent invalidation shows up only as
cache_read_input_tokens == 0 and is the most expensive bug in the project. The variable part -- parents,
their fitness, population history -- always goes in the user turn, after the cached prefix.
"""
from __future__ import annotations
import json

INSTRUCTIONS = (
    "You are a variation operator inside an evolutionary search over scikit-learn classification "
    "pipelines. You are given one or more parent pipelines (each with its fitness) and the grammar of "
    "the search space, and you must propose ONE new candidate pipeline that is likely to improve "
    "predictive accuracy on the target dataset while remaining valid under the grammar.\n\n"

    "OUTPUT FORMAT (strict).\n"
    "Return ONLY a single JSON object. No preamble, no explanation, no markdown code fences, no trailing "
    "text. The object has EXACTLY these three top-level keys (any order):\n"
    "  \"preprocessing\", \"feature_selection\", \"estimator\"\n"
    "There is NO top-level \"hyperparameters\" key. Each of the three values is ITSELF an object with "
    "exactly two keys:\n"
    "  \"name\"   -> a string naming one option that the grammar allows for that stage\n"
    "  \"params\" -> an object mapping that option's hyperparameters to values; use {} if it has none\n"
    "A stage is ALWAYS an object of the form {\"name\": ..., \"params\": ...}. A bare string such as\n"
    "  \"preprocessing\": \"standardize\"            <- INVALID (missing the object / params)\n"
    "is rejected; write instead\n"
    "  \"preprocessing\": {\"name\": \"standardize\", \"params\": {}}   <- correct.\n"
    "Do not add keys that the grammar does not define. Do not wrap the object in a list.\n\n"

    "READING THE GRAMMAR.\n"
    "Each stage lists its allowed option names. Each option lists its parameters as a spec with fields "
    "{type, low, high, log, choices, optional}:\n"
    "  - type int   -> an integer within [low, high]\n"
    "  - type float -> a real number within [low, high]. \"log: true\" only says values are spread on a "
    "log scale; the value you emit is still a plain number inside [low, high].\n"
    "  - type cat   -> a string equal to exactly one entry of \"choices\"\n"
    "  - optional: true -> you may omit that parameter entirely, or give it an in-range value\n"
    "An option whose grammar entry is empty ({}) takes no parameters, so its \"params\" MUST be {}. "
    "Every parameter you emit must obey its spec: an out-of-range value, an unknown parameter name, an "
    "unknown option name, or a category outside \"choices\" makes the whole candidate invalid and it is "
    "discarded (a wasted evaluation).\n\n"

    "HOW TO VARY.\n"
    "Prefer small, purposeful edits to a strong parent: change one estimator hyperparameter, swap a "
    "stage option, or add/drop feature selection. With two parents you may recombine their stages "
    "(take preprocessing from one, estimator from the other, etc.). Always return a complete, "
    "grammar-valid pipeline covering all three stages.\n\n"

    "COMMON MISTAKES TO AVOID.\n"
    "  1. Emitting a top-level \"hyperparameters\" key (there is none).\n"
    "  2. Writing a stage as a bare string instead of a {\"name\", \"params\"} object.\n"
    "  3. Putting a parameter on an option that does not define it (check the grammar).\n"
    "  4. Giving a numeric value outside [low, high], or a category not in \"choices\".\n"
    "  5. Omitting \"params\" (use {} when there are no parameters).\n"
    "  6. Adding prose, comments, or code fences around the JSON.\n\n"

    "EXAMPLES (each is a complete, valid object of exactly the required shape):\n"
    "{\"preprocessing\": {\"name\": \"standardize\", \"params\": {}}, \"feature_selection\": {\"name\": "
    "\"selectk\", \"params\": {\"k\": 20}}, \"estimator\": {\"name\": \"random_forest\", \"params\": "
    "{\"n_estimators\": 300, \"max_depth\": 12, \"min_samples_leaf\": 2, \"max_features\": \"sqrt\"}}}\n"
    "{\"preprocessing\": {\"name\": \"none\", \"params\": {}}, \"feature_selection\": {\"name\": \"pca\", "
    "\"params\": {\"n\": 15}}, \"estimator\": {\"name\": \"logistic\", \"params\": {\"C\": 2.5}}}\n"
    "{\"preprocessing\": {\"name\": \"minmax\", \"params\": {}}, \"feature_selection\": {\"name\": "
    "\"none\", \"params\": {}}, \"estimator\": {\"name\": \"knn\", \"params\": {\"n_neighbors\": 7, "
    "\"weights\": \"distance\"}}}\n"
    "{\"preprocessing\": {\"name\": \"standardize\", \"params\": {}}, \"feature_selection\": {\"name\": "
    "\"none\", \"params\": {}}, \"estimator\": {\"name\": \"hist_gbrt\", \"params\": {\"learning_rate\": "
    "0.05, \"max_leaf_nodes\": 63, \"max_iter\": 200, \"l2_regularization\": 0.001}}}\n"
    "{\"preprocessing\": {\"name\": \"none\", \"params\": {}}, \"feature_selection\": {\"name\": "
    "\"selectk\", \"params\": {\"k\": 10}}, \"estimator\": {\"name\": \"gaussian_nb\", \"params\": {}}}\n"
    "{\"preprocessing\": {\"name\": \"minmax\", \"params\": {}}, \"feature_selection\": {\"name\": "
    "\"pca\", \"params\": {\"n\": 8}}, \"estimator\": {\"name\": \"extra_trees\", \"params\": "
    "{\"n_estimators\": 200, \"min_samples_leaf\": 1, \"max_features\": \"log2\"}}}\n"
    "{\"preprocessing\": {\"name\": \"standardize\", \"params\": {}}, \"feature_selection\": {\"name\": "
    "\"selectk\", \"params\": {\"k\": 35}}, \"estimator\": {\"name\": \"svc\", \"params\": {\"C\": 3.0, "
    "\"gamma\": \"scale\"}}}\n\n"

    "Output ONLY the JSON object for your new candidate, matching the shape of the examples above, using "
    "only option names and parameters allowed by the grammar that follows.\n\n"
)


def build_prefix(grammar_text: str, instructions: str = INSTRUCTIONS) -> str:
    """The static, cacheable prefix. Deterministic in its inputs; no timestamps, no ordering nondeterminism.

    grammar_text should itself be produced deterministically (e.g. yaml.safe_dump(..., sort_keys=True)).
    """
    return instructions + "PIPELINE SPACE (grammar):\n" + grammar_text


def build_suffix(parents: list[dict], fitness: list[float] | None = None,
                 history: list[str] | None = None, *, include_fitness: bool = True,
                 include_history: bool = True) -> str:
    """The variable per-call part: parents, optional fitness, optional short population history."""
    parts = ["\n\nPARENTS:"]
    for i, p in enumerate(parents):
        line = f"  P{i}: {json.dumps(p, sort_keys=True)}"
        if include_fitness and fitness is not None and i < len(fitness):
            line += f"  (fitness={fitness[i]:.4f})"
        parts.append(line)
    if include_history and history:
        parts.append("RECENT POPULATION (structures seen):")
        parts.extend(f"  - {h}" for h in history)
    parts.append("Propose ONE new valid candidate as a single JSON object:")
    return "\n".join(parts)


def system_block(prefix: str) -> list[dict]:
    """Anthropic `system` param: one text block marked as an ephemeral cache prefix."""
    return [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}]
