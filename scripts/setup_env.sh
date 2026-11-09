#!/bin/bash
# Build the project environment from scratch (macOS or Linux, Python 3.11+).
#   scripts/setup_env.sh
# Environment variables: VENV (default .venv), PYTHON (default python3)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"; PYTHON="${PYTHON:-python3}"
say() { printf '\n==> %s\n' "$*"; }
"$PYTHON" -c 'import sys; assert sys.version_info[:2] >= (3, 11), "Python 3.11+ required (tested on 3.13)"'
say "virtual environment: $VENV"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip >/dev/null
say "requirements"
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"
say "checking"
APP_PYTHON="$VENV/bin/python" "$ROOT/app" doctor

# ---- web app (React + Tailwind, built into src/app/server/static) -------------------------------------------------------------------
if command -v npm >/dev/null 2>&1; then
  ( cd "$ROOT/web" && npm install --silent && npm run build ) && echo "web app built"
else
  echo "npm not found: install Node.js to build the web app (cd web && npm install && npm run build)"
fi
