#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON:-python3}"

if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
fi

pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

echo "[setup] Environment ready. Activate with: source $VENV_DIR/bin/activate"
