#!/usr/bin/env bash
# Source this file from WSL: source ./scripts/activate_wsl.sh
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Use: source ./scripts/activate_wsl.sh"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/Scripts/python.exe"

if [[ ! -f "$VENV_PY" ]]; then
  echo "Virtual environment Python not found at: $VENV_PY"
  return 1
fi

python() {
  "$VENV_PY" "$@"
}

pip() {
  "$VENV_PY" -m pip "$@"
}

export LOCALAGENT_VENV_PY="$VENV_PY"
export VIRTUAL_ENV="$ROOT_DIR/.venv"

echo "WSL wrappers active: python/pip -> $VENV_PY"
