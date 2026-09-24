#!/bin/bash
# Turnkey (re)creation of the auto-sklearn 2.0 environment used in Paper 3 §4.0b.
# macOS/Apple-Silicon: builds an osx-64 (Rosetta) conda env, since auto-sklearn's 2023
# stack has no arm64 wheels. Idempotent-ish: `conda env remove -n asklearn64` first to redo.
# Verify at the end that AutoSklearn2Classifier imports.
set -e
ENV=asklearn64

echo "== create osx-64 env =="
CONDA_SUBDIR=osx-64 conda create -n $ENV python=3.9 -y
conda run -n $ENV conda config --env --set subdir osx-64

echo "== install pinned old binary stack (conda-forge, osx-64) =="
CONDA_SUBDIR=osx-64 conda install -n $ENV -c conda-forge \
    "scikit-learn=0.24.2" "numpy=1.21" "scipy=1.7" pyrfr swig cython -y

echo "== auto-sklearn 0.15.0 (contains AutoSklearn2Classifier) =="
conda run -n $ENV pip install "auto-sklearn==0.15.0" --no-build-isolation

echo "== pin pandas<2 (2.x removed DataFrame.iteritems used by the askl2 selector) =="
conda run -n $ENV pip install "pandas==1.5.3"

echo "== patch pynisher: macOS does not support RLIMIT_AS setrlimit (crashes); make it a no-op =="
conda run -n $ENV python - <<'PY'
import pathlib, pynisher, re
p = pathlib.Path(pynisher.__file__).with_name("limit_function_call.py")
s = p.read_text()
needle = "resource.setrlimit(resource.RLIMIT_AS, (mem_in_b, mem_in_b))"
if "RLIMIT_AS local patch" not in s and needle in s:
    s = s.replace(needle,
        "try:\n            resource.setrlimit(resource.RLIMIT_AS, (mem_in_b, mem_in_b))\n"
        "        except (ValueError, OSError):\n            pass  # RLIMIT_AS local patch (macOS unsupported)")
    p.write_text(s); print("pynisher patched")
else:
    print("pynisher already patched or needle not found")
PY

echo "== verify =="
conda run -n $ENV python -c "import autosklearn; from autosklearn.experimental.askl2 import AutoSklearn2Classifier; print('OK auto-sklearn', autosklearn.__version__)"
echo "== done. Use memory_limit=<positive int> (None trips an assert in 0.15). =="
