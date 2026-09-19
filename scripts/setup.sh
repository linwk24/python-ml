#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "[setup] Installing dependencies from requirements.txt..."

if command -v uv >/dev/null 2>&1; then
  echo "[setup] Using uv for installation"
  if [ -n "${PIP_TARGET:-}" ]; then
    echo "[setup] Deploy mode: installing to PIP_TARGET=$PIP_TARGET"
    uv pip install --no-cache --target "$PIP_TARGET" -r requirements.txt
  else
    echo "[setup] Devbox mode: installing to .venv"
    if [ ! -d ".venv" ]; then
      uv venv .venv --python 3.12
    fi
    uv pip install -r requirements.txt
  fi
elif [ -f "requirements.txt" ]; then
  echo "[setup] Fallback mode: using pip"
  pip install -r requirements.txt
else
  echo "[setup] Error: requirements.txt not found"
  exit 1
fi

echo "[setup] Creating models directory..."
mkdir -p models

echo "[setup] Installation complete."
