#!/usr/bin/env bash
set -euo pipefail

INVOKE_DIR="$(pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8011}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$INVOKE_DIR}"

if [[ ! -x ".venv/Scripts/python.exe" ]]; then
  echo "Missing virtual environment at .venv/Scripts/python.exe"
  echo "Create it and install dependencies first."
  exit 1
fi

if command -v ss >/dev/null 2>&1; then
  while ss -ltn "( sport = :$PORT )" | grep -q LISTEN; do
    PORT="$((PORT + 1))"
  done
fi

echo "Starting LocalAgentLLM on http://$HOST:$PORT"
echo "workspace_root=$WORKSPACE_ROOT"

exec .venv/Scripts/python.exe -m app.server_entry --host "$HOST" --port "$PORT" --workspace-root "$WORKSPACE_ROOT"
