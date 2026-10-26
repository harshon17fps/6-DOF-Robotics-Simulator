#!/usr/bin/env bash
# One-shot environment bootstrap: creates an isolated virtualenv in .venv and
# installs the production dependencies listed in requirements.txt.
#
# VTK (PyVista's rendering backend) ships binary wheels for CPython 3.10-3.14.
# If no compatible interpreter is on PATH, this script bootstraps `uv`
# (https://docs.astral.sh/uv/) and lets it fetch a standalone CPython 3.13.
set -euo pipefail

cd "$(dirname "$0")"
SUPPORTED_VERSIONS=("3.13" "3.12" "3.14" "3.11")

find_compatible_python() {
    local v
    for v in "${SUPPORTED_VERSIONS[@]}"; do
        if command -v "python$v" >/dev/null 2>&1; then
            echo "python$v"
            return 0
        fi
    done
    return 1
}

rm -rf .venv

if PY="$(find_compatible_python)"; then
    echo "==> Creating virtual environment with $($PY --version)"
    "$PY" -m venv .venv
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements.txt
else
    echo "==> No VTK-compatible Python (3.11-3.14) found on PATH; using uv"
    if command -v uv >/dev/null 2>&1; then
        UV="uv"
    else
        echo "==> Bootstrapping uv into .tools/"
        python3 -m venv .tools/uv-venv
        .tools/uv-venv/bin/python -m pip install --quiet --upgrade pip uv
        UV=".tools/uv-venv/bin/uv"
    fi
    echo "==> Creating virtual environment with CPython 3.13 (managed by uv)"
    "$UV" venv --quiet --seed --python 3.13 .venv
    "$UV" pip install --quiet --python .venv/bin/python -r requirements.txt
fi

echo "==> Verifying imports"
.venv/bin/python - <<'PY'
import sys
import numpy, scipy, pyvista, vtk
print(f"python {sys.version.split()[0]} | numpy {numpy.__version__} | "
      f"scipy {scipy.__version__} | pyvista {pyvista.__version__} | "
      f"vtk {vtk.vtkVersion.GetVTKVersion()}")
PY

echo "==> Done. Activate with: source .venv/bin/activate"
