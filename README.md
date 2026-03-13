# LocalAgentLLM

Local, high-performance agentic coding assistant with a browser chat UI.

## Features

- Local-first model runtime via OpenAI-compatible endpoint (default: Ollama).
- Agentic loop with tool calling for coding tasks:
	- read files
	- write files
	- list directories
	- run shell commands
- Workspace-root path guard so tool calls cannot escape project scope.
- FastAPI backend with simple web chat frontend.
- WSL-friendly path and command behavior.
- MCP-style tool runtime: typed argument schemas, validation, and structured execution results.
- Optional supervisor stage to generate a compact execution brief before worker tool use.
- Optional split-model supervision so planning can use a different model than tool execution.
- Scratchpad-backed execution loop with live TODO state snapshots in `steps` traces.
- Public reasoning summaries before tool actions (concise, no hidden chain-of-thought exposure).
- Advanced tools for complex edits and C++ workflows.

## C++ + Advanced Tooling

The agent now includes dedicated tools for C++ and richer file operations:

- `compile_cpp`: compile C++ source via `g++`/`clang++` (with WSL fallback on Windows)
- `run_binary`: execute compiled artifacts
- `search_files`: glob search across workspace
- `replace_in_file`: deterministic text replacement
- `move_path`: move/rename files and directories
- `delete_path`: safe deletion with explicit recursive flag for directories

For C++ requests, the runtime attempts autonomous validation by compiling written `.cpp/.cc/.cxx/.c` files, and running binaries when requested.

## Recommended Models (16GB VRAM)

 Qwen3 8B (default target)

 Supervisor emits structured TODO plans and tool strategy before execution.

## Quick Start (WSL)

1. Create and activate a Python environment.
2. Install dependencies.
3. Copy `.env.example` to `.env` and adjust values.
4. Start server.

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 in your browser.

### WSL Virtual Environment Activation

In WSL, do not run a Windows-style path like `c:\\Users\\...\\activate` directly.

This project uses a Windows venv (`.venv/Scripts/python.exe`). In WSL, use:

```bash
source ./scripts/activate_wsl.sh
```

That sets `python` and `pip` shell wrappers to the project venv interpreter.

### Scripted Launch

From WSL:

```bash
./scripts/start_server.sh
./scripts/start_tui.sh
```

From PowerShell:

```powershell
.\scripts\start_server.ps1
.\scripts\start_tui.ps1
```

Both scripts default `WORKSPACE_ROOT` to the directory where you invoke the script, so you can run from another folder and the agent tools will operate there.

### Global Launcher Command

Install once in WSL:

```bash
chmod +x ./scripts/localagent.sh ./scripts/install_global_launcher.sh
./scripts/install_global_launcher.sh
```

Then use from any directory:

```bash
localagent server
localagent tui
localagent ask "Do not use tools. Reply READY"
localagent ollama-start
localagent doctor
localagent gpu-doctor
localagent tune
localagent cleanup
```

If your host requires Vulkan mode for Ollama GPU offload, use `localagent ollama-start` to restart Ollama with `OLLAMA_VULKAN=1`.

`localagent tune` runs a Human-in-the-Loop scoring session so you can judge output quality and pick better model/settings based on your ratings.

Example with starter prompt set:

```bash
localagent tune --prompts prompts/coding_eval.txt --models qwen2.5-coder:1.5b
```

## Environment Variables

- `LLM_BASE_URL`: OpenAI-compatible local endpoint (default `http://localhost:11434/v1`)
- `LLM_API_KEY`: Any key your endpoint expects (for Ollama, `ollama` is fine)
- `MODEL`: Model ID served by your local runtime
- `FALLBACK_MODEL`: Worker fallback model if primary is unavailable
- `SUPERVISOR_MODEL`: Optional dedicated planning model (falls back to `MODEL` when empty)
- `SUPERVISOR_FALLBACK_MODEL`: Optional supervisor fallback model
- `WORKSPACE_ROOT`: Linux path under WSL for file/tool access control
- `MAX_AGENT_STEPS`: Max tool iterations per request
- `MAX_TOOL_ERRORS`: Stop after repeated tool failures
- `ENABLE_SUPERVISOR`: Enable/disable supervisor planning pass
- `COMMAND_TIMEOUT_S`: Shell command timeout for tool execution
- `WORKSPACE_ROOT`: Runtime tool root (scripts default this to your invocation directory)

## Extending Tools (Robust Path)

Tools are centralized in [app/tools.py](app/tools.py) using typed `pydantic` arg models and a registry.

1. Add a new args model (subclass of `ToolArgs`).
2. Implement async handler: `async def _my_tool(args: MyArgs) -> dict[str, Any]`.
3. Register it:

```python
registry.register(ToolSpec("my_tool", "Description", MyArgs, _my_tool))
```

That is all. The OpenAI tool schema is auto-generated from the args model, and execution includes:

- strict JSON decode
- schema validation
- structured error payloads

This is intentionally designed so new tools do not require regex parsing changes in the agent loop.

## Notes on Performance

- Keep `MAX_AGENT_STEPS` low for responsiveness.
- Use concise prompts to reduce token latency.
- Prefer smaller coding models when iterating quickly, switch to larger/MoE models for harder tasks.
