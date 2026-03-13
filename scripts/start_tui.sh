#!/usr/bin/env bash
set -euo pipefail

INVOKE_DIR="$(pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$INVOKE_DIR}"

if [[ ! -x ".venv/Scripts/python.exe" ]]; then
  echo "Missing virtual environment at .venv/Scripts/python.exe"
  echo "Create it and install dependencies first."
  exit 1
fi

exec .venv/Scripts/python.exe -m app.tui --root "$WORKSPACE_ROOT"
