#!/bin/bash
# Launcher: runs the app CLI with the project environment. Override the interpreter with APP_PYTHON.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${APP_PYTHON:-$ROOT/.venv/bin/python}"; [ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="python3"
exec env PYTHONPATH="$ROOT/src" PYTHONWARNINGS=ignore "$PY" -m app "$@"
