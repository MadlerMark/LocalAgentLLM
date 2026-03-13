#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_EXE="$ROOT_DIR/.venv/Scripts/python.exe"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8011}"
DEFAULT_ROOT="$(pwd)"

usage() {
  cat <<EOF
Usage: localagent <command> [options]

Commands:
  server [root]        Start API server with workspace root (default: current dir)
  tui [root]           Start interactive TUI with workspace root (default: current dir)
  ask <prompt> [root]  Run one prompt and print response
  ollama-start         Restart Ollama daemon with OLLAMA_VULKAN=1
  doctor               Check Ollama process, endpoint, and model availability
  gpu-doctor           Check GPU offload status and print remediation steps
  tune [prompts_file]  Run Human-in-the-Loop model scoring workflow
  cleanup              Stop LocalAgentLLM uvicorn processes
  help                 Show this message
EOF
}

require_python() {
  if [[ ! -x "$PY_EXE" ]]; then
    echo "Missing Python at $PY_EXE"
    exit 1
  fi
}

next_free_port() {
  local candidate="$1"
  if command -v ss >/dev/null 2>&1; then
    while ss -ltn "( sport = :$candidate )" | grep -q LISTEN; do
      candidate="$((candidate + 1))"
    done
  fi
  echo "$candidate"
}

start_server() {
  require_python
  local root="${1:-$DEFAULT_ROOT}"
  local p
  p="$(next_free_port "$PORT")"
  cd "$ROOT_DIR"
  echo "Starting LocalAgentLLM server on http://$HOST:$p"
  echo "workspace_root=$root"
  exec "$PY_EXE" -m app.server_entry --host "$HOST" --port "$p" --workspace-root "$root"
}

start_tui() {
  require_python
  local root="${1:-$DEFAULT_ROOT}"
  export WORKSPACE_ROOT="$root"
  cd "$ROOT_DIR"
  exec "$PY_EXE" -m app.tui --root "$WORKSPACE_ROOT"
}

ask_once() {
  require_python
  local prompt="${1:-}"
  local root="${2:-$DEFAULT_ROOT}"
  if [[ -z "$prompt" ]]; then
    echo "Usage: localagent ask <prompt> [root]"
    exit 1
  fi
  cd "$ROOT_DIR"
  exec "$PY_EXE" -m app.tui --root "$root" --ask "$prompt"
}

ollama_start() {
  if ! command -v powershell.exe >/dev/null 2>&1; then
    echo "powershell.exe not found"
    exit 1
  fi

  powershell.exe -NoProfile -Command '$p=(Get-Command ollama -ErrorAction SilentlyContinue).Source; if(-not $p){ Write-Output "OLLAMA_NOT_FOUND"; exit 1 }; Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue; $env:OLLAMA_VULKAN="1"; Start-Process -FilePath $p -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep -Seconds 2; & $p --version; & $p ps'
}

doctor() {
  echo "== LocalAgent Doctor =="
  echo "root_dir=$ROOT_DIR"
  echo "invocation_dir=$DEFAULT_ROOT"

  if command -v powershell.exe >/dev/null 2>&1; then
    local model_check
    model_check="${MODEL:-}"
    if [[ -z "$model_check" && -f "$ROOT_DIR/.env" ]]; then
      model_check="$(awk -F= '/^MODEL=/{print $2; exit}' "$ROOT_DIR/.env" | tr -d '\r')"
    fi
    model_check="${model_check:-qwen2.5-coder:1.5b}"
    echo "model_check=$model_check"
    echo "ollama_vulkan_expected=1"
    local ollama_path
    ollama_path="$(powershell.exe -NoProfile -Command "(Get-Command ollama -ErrorAction SilentlyContinue).Source" | tr -d '\r')"
    if [[ -n "$ollama_path" ]]; then
      echo "ollama_binary=$ollama_path"
    else
      echo "ollama_binary=NOT_FOUND"
    fi

    local proc_count
    proc_count="$(powershell.exe -NoProfile -Command "@(Get-Process ollama -ErrorAction SilentlyContinue).Count" | tr -d '\r')"
    echo "ollama_process_count=${proc_count:-0}"

    local tags_wsl
    tags_wsl="$(curl -sS http://127.0.0.1:11434/api/tags 2>/dev/null || true)"
    if [[ -n "$tags_wsl" ]]; then
      echo "ollama_api_wsl=reachable"
    else
      echo "ollama_api_wsl=unreachable"
    fi

    local tags_win
    tags_win="$(powershell.exe -NoProfile -Command "try { (Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -Method Get | ConvertTo-Json -Depth 6) } catch { '' }" | tr -d '\r')"
    if [[ -n "$tags_win" ]]; then
      echo "ollama_api_windows=reachable"
      echo "$tags_win" | head -c 500
      echo
    else
      echo "ollama_api_windows=unreachable"
    fi

    if [[ -x "$PY_EXE" ]]; then
      local rt_ok
      rt_ok="$($PY_EXE -c "from openai import OpenAI; c=OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='ollama'); r=c.chat.completions.create(model='$model_check', messages=[{'role':'user','content':'ok'}], max_tokens=2, temperature=0); print('ok' if r.choices else 'fail')" 2>/dev/null | tr -d '\r\n' || true)"
      if [[ "$rt_ok" == "ok" ]]; then
        echo "app_runtime_llm=reachable"
      else
        echo "app_runtime_llm=unreachable_or_model_missing"
      fi
    fi

    if [[ -n "$ollama_path" ]]; then
      echo "models:"
      powershell.exe -NoProfile -Command '$env:OLLAMA_VULKAN="1"; & (Get-Command ollama).Source list'
      echo "active_sessions:"
      powershell.exe -NoProfile -Command '$env:OLLAMA_VULKAN="1"; & (Get-Command ollama).Source ps'
    fi
  else
    echo "powershell.exe not found; unable to inspect Windows Ollama from WSL"
  fi
}

cleanup() {
  pkill -f "uvicorn app.main:app" || true
  sleep 1
  ps -ef | grep -E "uvicorn app.main:app|start_server.sh|localagent.sh" | grep -v grep || true
}

gpu_doctor() {
  if ! command -v powershell.exe >/dev/null 2>&1; then
    echo "powershell.exe not found"
    return
  fi

  echo "== GPU Doctor =="
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | Format-Table -AutoSize"
  echo
  powershell.exe -NoProfile -Command '$env:OLLAMA_VULKAN="1"; & (Get-Command ollama).Source --version; & (Get-Command ollama).Source ps'
  echo
  echo "Interpretation:"
  echo "- If PROCESSOR shows CPU only, Ollama is not offloading to GPU."
  echo "- If PROCESSOR shows a GPU split, offload is active."
  echo
  echo "Recommended remediation order for RX 9070 XT:"
  echo "1) Update AMD Adrenalin / AI stack to latest stable release."
  echo "2) Update Ollama to latest build that includes your AMD backend support."
  echo "3) Restart Ollama daemon with Vulkan: localagent ollama-start"
  echo "4) Re-test with: ollama run qwen2.5-coder:1.5b 'hello' && ollama ps"
  echo "5) If still CPU-only, use an alternate local runtime with explicit Vulkan/DirectML support and keep this app pointed to its OpenAI-compatible endpoint."
}

tune() {
  require_python
  cd "$ROOT_DIR"
  exec "$PY_EXE" -m app.hitl_tuner "$@"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  server) start_server "${1:-}" ;;
  tui) start_tui "${1:-}" ;;
  ask) ask_once "${1:-}" "${2:-}" ;;
  ollama-start) ollama_start ;;
  doctor) doctor ;;
  gpu-doctor) gpu_doctor ;;
  tune) tune "$@" ;;
  cleanup) cleanup ;;
  help|-h|--help) usage ;;
  *)
    echo "Unknown command: $cmd"
    usage
    exit 1
    ;;
esac
