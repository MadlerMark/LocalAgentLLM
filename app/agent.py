import time
from typing import Any
import json
import re

from app.config import settings
from app.llm_client import LLMClient
from app.tools import registry, reset_workspace_root, set_workspace_root, tool_result_json

SYSTEM_PROMPT = """You are a high-performance coding agent running locally.
Use tools when needed for accurate codebase operations.
Keep responses concise, deterministic, and actionable.
When using tools, only call registered tools and follow their schemas exactly.
Prefer compile_python and run_python_file for Python validation/execution tasks.
Prefer compile_cpp and run_binary for C++ validation/execution tasks.
For requests to create, edit, compile, test, or run code, you must use tools rather than answering from memory.
Before tool calls, provide a brief public rationale sentence (no hidden chain-of-thought).
"""

SUPERVISOR_PROMPT = """You are the supervisor for a coding agent.
Think step-by-step privately, but output only strict JSON with this schema:
{
    "objective": "string",
    "todo": ["short task", "short task"],
    "constraints": ["constraint"],
    "tool_strategy": ["tool name order"],
    "done_when": "string"
}
Do not include markdown or extra text.
"""


class AgentRuntime:
    def __init__(self) -> None:
        self.llm = LLMClient()

    async def _supervisor_brief(self, user_message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        if not settings.enable_supervisor:
            return {}

        messages: list[dict[str, Any]] = [{"role": "system", "content": SUPERVISOR_PROMPT}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": user_message})
        resp = await self.llm.complete(
            messages=messages,
            tools=None,
            temperature=0.0,
            model=self._supervisor_model(),
            fallback_model=self._supervisor_fallback_model(),
        )
        msg = resp["choices"][0]["message"]
        plan = self._parse_supervisor_plan((msg.get("content") or "").strip()[:4000])
        if plan:
            return plan

        target_file = self._extract_target_code_file(user_message)
        return self._fallback_plan(user_message, target_file, self._should_require_tools(user_message))

    async def run(self, user_message: str, history: list[dict[str, str]], workspace_root: str | None = None) -> dict[str, Any]:
        token = None
        if workspace_root:
            token = set_workspace_root(workspace_root)

        try:
            require_tools = self._should_require_tools(user_message)
            target_file = self._extract_target_code_file(user_message)
            brief = await self._supervisor_brief(user_message, history) if require_tools else {}

            messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            if brief:
                todo_text = "\n".join(f"- {item}" for item in brief.get("todo", []))
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"Supervisor objective: {brief.get('objective', '')}\n"
                            f"Supervisor TODOs:\n{todo_text}\n"
                            f"Constraints: {', '.join(brief.get('constraints', []))}\n"
                            f"Tool strategy: {', '.join(brief.get('tool_strategy', []))}\n"
                            f"Done when: {brief.get('done_when', '')}\n"
                            "Follow this unless user intent conflicts."
                        ),
                    }
                )
            messages.extend(history)
            messages.append({"role": "user", "content": user_message})

            trace: list[dict[str, Any]] = []
            start = time.perf_counter()
            tool_errors = 0
            forced_tool_retry = False
            openai_tools = registry.to_openai_tools() if require_tools else None
            scratchpad = self._init_scratchpad(brief, user_message, target_file, require_tools) if require_tools else {}

            if brief:
                trace.append({"supervisor": brief})
            if require_tools:
                trace.append({"scratchpad": self._snapshot_scratchpad(scratchpad)})

            if require_tools and target_file:
                messages.append(
                    {
                        "role": "system",
                        "content": f"You must operate on the requested file: {target_file}. Do not substitute a different file.",
                    }
                )

            if require_tools:
                bootstrap_call_id = "bootstrap_list_dir"
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Bootstrapping workspace context.",
                        "tool_calls": [
                            {
                                "id": bootstrap_call_id,
                                "type": "function",
                                "function": {"name": "list_dir", "arguments": json.dumps({"path": "."})},
                            }
                        ],
                    }
                )
                bootstrap = await registry.execute("list_dir", {"path": "."})
                trace.append({"tool": "list_dir", "args": {"path": "."}, "result": bootstrap.payload, "ok": bootstrap.ok})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": bootstrap_call_id,
                        "content": tool_result_json(bootstrap.payload),
                    }
                )

            for step in range(settings.max_agent_steps):
                if require_tools:
                    messages.append({"role": "system", "content": self._scratchpad_prompt(scratchpad)})
                resp = await self.llm.complete(messages=messages, tools=openai_tools)
                msg = resp["choices"][0]["message"]
                tool_calls = msg.get("tool_calls") or []

                if require_tools and not tool_calls:
                    synthetic = self._extract_tool_call_from_content(msg.get("content") or "")
                    if synthetic is not None and registry.has_tool(synthetic["function"]["name"]):
                        tool_calls = [synthetic]

                if not tool_calls:
                    if require_tools and not any("tool" in s for s in trace) and not forced_tool_retry:
                        forced_tool_retry = True
                        messages.append(
                            {
                                "role": "system",
                                "content": "You must execute tools for this request. Do not answer until required tool calls are completed.",
                            }
                        )
                        continue

                    if require_tools and not self._coding_objective_completed(user_message, trace, target_file):
                        if target_file:
                            messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        "You have not completed the required coding objective yet. "
                                        f"Use tools to create/modify, compile, and run {target_file} as requested. "
                                        "Do not answer in plain text until done."
                                    ),
                                }
                            )
                        else:
                            messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        "You have not completed the required coding objective yet. "
                                        "Use tools to finish the requested code actions before answering."
                                    ),
                                }
                            )
                        continue

                    answer = self._normalize_answer(msg.get("content", ""))
                    return {
                        "answer": answer,
                        "steps": trace,
                        "latency_ms": int((time.perf_counter() - start) * 1000),
                        "iterations": step + 1,
                    }

                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content") or "",
                        "tool_calls": tool_calls,
                    }
                )

                for call in tool_calls:
                    fn_name = self._canonical_tool_name(call["function"]["name"])
                    raw_args = call["function"].get("arguments")
                    adjusted_args = self._enforce_target_file_args(fn_name, raw_args, target_file)
                    result = await registry.execute(fn_name, adjusted_args)

                    if not result.ok:
                        tool_errors += 1

                    trace.append({"tool": result.name, "args": result.args, "result": result.payload, "ok": result.ok})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": tool_result_json(result.payload),
                        }
                    )
                    if require_tools:
                        self._update_scratchpad_after_tool(
                            scratchpad=scratchpad,
                            tool_name=result.name,
                            ok=result.ok,
                            args=result.args,
                            payload=result.payload,
                            target_file=target_file,
                        )

                if require_tools and target_file:
                    trace, messages = await self._auto_complete_target_file_steps(
                        user_message,
                        trace,
                        messages,
                        target_file,
                        scratchpad,
                    )

                if require_tools:
                    trace, messages = await self._auto_validate_written_python_files(
                        user_message=user_message,
                        trace=trace,
                        messages=messages,
                        scratchpad=scratchpad,
                    )

                    trace, messages = await self._auto_validate_written_cpp_files(
                        user_message=user_message,
                        trace=trace,
                        messages=messages,
                        scratchpad=scratchpad,
                    )

                if require_tools:
                    trace.append({"scratchpad": self._snapshot_scratchpad(scratchpad)})

                if require_tools and self._coding_objective_completed(user_message, trace, target_file):
                    return {
                        "answer": self._build_completion_summary(user_message, trace, target_file),
                        "steps": trace,
                        "latency_ms": int((time.perf_counter() - start) * 1000),
                        "iterations": step + 1,
                    }

                if require_tools and self._is_repetitive_tool_loop(trace):
                    break

                if tool_errors >= settings.max_tool_errors:
                    if not require_tools:
                        fallback_messages = messages + [
                            {
                                "role": "system",
                                "content": "Tool calling is failing. Answer the user directly without tools.",
                            }
                        ]
                        fallback = await self.llm.complete(messages=fallback_messages, tools=None, temperature=0.0)
                        fallback_msg = fallback["choices"][0]["message"]
                        return {
                            "answer": self._normalize_answer(fallback_msg.get("content", "I could not complete this request.")),
                            "steps": trace,
                            "latency_ms": int((time.perf_counter() - start) * 1000),
                            "iterations": step + 1,
                        }

                    return {
                        "answer": "I hit repeated tool-call errors and stopped for safety. Please retry with a more specific request.",
                        "steps": trace,
                        "latency_ms": int((time.perf_counter() - start) * 1000),
                        "iterations": step + 1,
                    }

            if require_tools and target_file and not self._coding_objective_completed(user_message, trace, target_file):
                recovered = await self._recover_with_direct_codegen(
                    user_message=user_message,
                    history=history,
                    target_file=target_file,
                    trace=trace,
                    scratchpad=scratchpad,
                    target_iterations=settings.max_agent_steps,
                    start_time=start,
                )
                if recovered is not None:
                    return recovered

            return {
                "answer": "I hit the max tool-iteration limit before finishing.",
                "steps": trace,
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "iterations": settings.max_agent_steps,
            }
        finally:
            if token is not None:
                reset_workspace_root(token)

    def _should_require_tools(self, user_message: str) -> bool:
        text = user_message.lower()
        keywords = [
            "create",
            "write",
            "edit",
            "modify",
            "compile",
            "build",
            "run",
            "execute",
            "test",
            "file",
            "code",
            "c++",
            "cpp",
        ]
        return any(k in text for k in keywords)

    def _parse_supervisor_plan(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if not text:
            return {}

        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {}

        if not isinstance(obj, dict):
            return {}

        todo = obj.get("todo")
        constraints = obj.get("constraints")
        strategy = obj.get("tool_strategy")
        return {
            "objective": str(obj.get("objective", "")),
            "todo": [str(x) for x in todo] if isinstance(todo, list) else [],
            "constraints": [str(x) for x in constraints] if isinstance(constraints, list) else [],
            "tool_strategy": [str(x) for x in strategy] if isinstance(strategy, list) else [],
            "done_when": str(obj.get("done_when", "")),
        }

    def _extract_tool_call_from_content(self, content: str) -> dict[str, Any] | None:
        def to_tool(obj: dict[str, Any]) -> dict[str, Any] | None:
            if "name" in obj and "arguments" in obj and isinstance(obj["arguments"], dict):
                tool_name = self._canonical_tool_name(str(obj["name"]))
                return {
                    "id": "synthetic_tool_call",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(obj["arguments"])},
                }

            if "tool" in obj and "args" in obj and isinstance(obj["args"], dict):
                tool_name = self._canonical_tool_name(str(obj["tool"]))
                return {
                    "id": "synthetic_tool_call",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(obj["args"])},
                }

            return None

        text = content.strip()
        if not text:
            return None

        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: line-delimited JSON objects, choose first valid tool intent.
            for line in text.splitlines():
                candidate = line.strip().rstrip(",")
                if not candidate:
                    continue
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    tool = to_tool(parsed)
                    if tool is not None:
                        return tool
            return None

        if isinstance(obj, dict):
            return to_tool(obj)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    tool = to_tool(item)
                    if tool is not None:
                        return tool
        return None

    def _canonical_tool_name(self, name: str) -> str:
        aliases = {
            "create_file": "write_file",
            "edit_file": "write_file",
            "write_text_file": "write_file",
            "run_python": "run_python_file",
            "execute_python": "run_python_file",
            "compile_file": "compile_python",
            "compile_cxx": "compile_cpp",
            "build_cpp": "compile_cpp",
            "run_exe": "run_binary",
            "ls": "list_dir",
        }
        return aliases.get(name, name)

    def _coding_objective_completed(self, user_message: str, trace: list[dict[str, Any]], target_file: str | None) -> bool:
        text = user_message.lower()
        tool_steps = [s for s in trace if "tool" in s]
        if not tool_steps:
            return False

        def has_ok(name: str) -> bool:
            return any(step.get("tool") == name and step.get("ok") for step in tool_steps)

        wants_write = any(k in text for k in ["create", "write", "file"])
        wants_compile = "compile" in text
        wants_run = any(k in text for k in ["run", "execute"])

        write_done = has_ok("write_file")
        compile_done = has_ok("compile_python") or has_ok("compile_cpp")
        run_done = has_ok("run_python_file") or has_ok("run_command") or has_ok("run_binary")

        if target_file:
            target_norm = target_file.lower()

            def path_matches(step: dict[str, Any]) -> bool:
                args_path = str(step.get("args", {}).get("path", "")).replace("\\", "/").lower()
                if args_path.endswith("/" + target_norm) or args_path.endswith(target_norm):
                    return True
                result_path = str(step.get("result", {}).get("path", "")).replace("\\", "/").lower()
                return result_path.endswith("/" + target_norm) or result_path.endswith(target_norm)

            write_done = any(s.get("tool") == "write_file" and s.get("ok") and path_matches(s) for s in tool_steps)
            compile_done = any(s.get("tool") in {"compile_python", "compile_cpp"} and s.get("ok") and path_matches(s) for s in tool_steps)
            run_done = any(s.get("tool") in {"run_python_file", "run_command", "run_binary"} and s.get("ok") and path_matches(s) for s in tool_steps)

        if wants_compile and not compile_done:
            return False
        if wants_run and not run_done:
            return False
        if wants_write and not write_done:
            return False
        if wants_write and target_file and target_file.lower().endswith(".py") and not compile_done:
            return False
        if wants_write and target_file and self._is_cpp_file(target_file) and not compile_done:
            return False

        return write_done or compile_done or run_done

    def _build_completion_summary(self, user_message: str, trace: list[dict[str, Any]], target_file: str | None) -> str:
        tool_steps = [s for s in trace if "tool" in s]
        wrote = [s for s in tool_steps if s.get("tool") == "write_file" and s.get("ok")]
        compiled = [s for s in tool_steps if s.get("tool") in {"compile_python", "compile_cpp"} and s.get("ok")]
        ran = [
            s
            for s in tool_steps
            if s.get("ok") and s.get("tool") in {"run_python_file", "run_command", "run_binary"}
        ]

        parts: list[str] = []
        if wrote:
            path = wrote[-1].get("args", {}).get("path", "the requested file")
            parts.append(f"Created/updated {path}.")
        if compiled:
            parts.append("Compilation succeeded.")
        if ran:
            payload = ran[-1].get("result", {})
            stdout = str(payload.get("stdout", "")).strip()
            if stdout:
                parts.append(f"Program output: {stdout}")
            else:
                parts.append("Execution succeeded.")

        if parts:
            return " ".join(parts)
        return "Completed the requested tool actions."

    def _extract_target_code_file(self, user_message: str) -> str | None:
        match = re.search(r"([A-Za-z0-9_.-]+\.(?:py|cpp|cc|cxx|c|hpp|h))\b", user_message, re.IGNORECASE)
        if not match:
            return None
        return match.group(1)

    def _is_cpp_file(self, path: str) -> bool:
        lower = path.lower()
        return lower.endswith(".cpp") or lower.endswith(".cc") or lower.endswith(".cxx") or lower.endswith(".c")

    def _enforce_target_file_args(self, tool_name: str, raw_args: Any, target_file: str | None) -> Any:
        if not target_file:
            return raw_args
        if tool_name not in {"write_file", "compile_python", "run_python_file", "compile_cpp", "run_binary"}:
            return raw_args

        if isinstance(raw_args, dict):
            args_obj: dict[str, Any] = dict(raw_args)
        elif isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                if not isinstance(parsed, dict):
                    return raw_args
                args_obj = dict(parsed)
            except json.JSONDecodeError:
                return raw_args
        else:
            return raw_args

        path = args_obj.get("path")
        if isinstance(path, str) and path.strip():
            normalized = path.replace("\\", "/")
            if not normalized.lower().endswith(target_file.lower()):
                if "/" in normalized:
                    parent = normalized.rsplit("/", 1)[0]
                    args_obj["path"] = f"{parent}/{target_file}"
                else:
                    args_obj["path"] = target_file
        else:
            args_obj["path"] = target_file

        if tool_name == "write_file":
            # Keep target file content stable; avoid repeated append loops that corrupt syntax.
            args_obj["append"] = False

        return args_obj

    async def _auto_complete_target_file_steps(
        self,
        user_message: str,
        trace: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        target_file: str,
        scratchpad: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        text = user_message.lower()
        wants_compile = "compile" in text or "build" in text or target_file.lower().endswith(".py") or self._is_cpp_file(target_file)
        wants_run = any(k in text for k in ["run", "execute"])

        tool_steps = [s for s in trace if "tool" in s]

        def path_matches(step: dict[str, Any]) -> bool:
            args_path = str(step.get("args", {}).get("path", "")).replace("\\", "/").lower()
            if args_path.endswith("/" + target_file.lower()) or args_path.endswith(target_file.lower()):
                return True
            result_path = str(step.get("result", {}).get("path", "")).replace("\\", "/").lower()
            return result_path.endswith("/" + target_file.lower()) or result_path.endswith(target_file.lower())

        has_write = any(s.get("tool") == "write_file" and s.get("ok") and path_matches(s) for s in tool_steps)
        has_compile = any(s.get("tool") in {"compile_python", "compile_cpp"} and s.get("ok") and path_matches(s) for s in tool_steps)
        has_run = any(s.get("tool") in {"run_python_file", "run_binary"} and s.get("ok") and path_matches(s) for s in tool_steps)

        if not has_write:
            return trace, messages

        is_cpp_target = self._is_cpp_file(target_file)
        compile_tool = "compile_cpp" if is_cpp_target else "compile_python"
        run_tool = "run_binary" if is_cpp_target else "run_python_file"

        if wants_compile and not has_compile:
            compile_args = {"path": target_file}
            compile_result = await registry.execute(compile_tool, compile_args)
            trace.append(
                {
                    "tool": compile_tool,
                    "args": compile_args,
                    "result": compile_result.payload,
                    "ok": compile_result.ok,
                }
            )
            self._update_scratchpad_after_tool(
                scratchpad=scratchpad,
                tool_name=compile_tool,
                ok=compile_result.ok,
                args=compile_args,
                payload=compile_result.payload,
                target_file=target_file,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "auto_compile_target",
                    "content": tool_result_json(compile_result.payload),
                }
            )

        if wants_run and not has_run:
            run_path = target_file
            if is_cpp_target:
                for s in reversed(trace):
                    if s.get("tool") == "compile_cpp" and isinstance(s.get("result"), dict):
                        candidate = str(s["result"].get("output", "")).strip()
                        if candidate:
                            run_path = candidate
                            break
            run_args = {"path": run_path, "args": []}
            run_result = await registry.execute(run_tool, run_args)
            trace.append(
                {
                    "tool": run_tool,
                    "args": run_args,
                    "result": run_result.payload,
                    "ok": run_result.ok,
                }
            )
            self._update_scratchpad_after_tool(
                scratchpad=scratchpad,
                tool_name=run_tool,
                ok=run_result.ok,
                args=run_args,
                payload=run_result.payload,
                target_file=target_file,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "auto_run_target",
                    "content": tool_result_json(run_result.payload),
                }
            )

        return trace, messages

    async def _auto_validate_written_python_files(
        self,
        user_message: str,
        trace: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        scratchpad: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        wants_run = any(k in user_message.lower() for k in ["run", "execute"])
        tool_steps = [s for s in trace if "tool" in s]

        written_targets: list[str] = []
        for step in tool_steps:
            if step.get("tool") != "write_file" or not step.get("ok"):
                continue
            path = str(step.get("args", {}).get("path", ""))
            if not path.lower().endswith(".py"):
                continue
            normalized = path.replace("\\", "/")
            if normalized not in written_targets:
                written_targets.append(normalized)

        for target in written_targets:
            has_compile_ok = any(
                s.get("tool") == "compile_python"
                and s.get("ok")
                and self._path_matches_target(s.get("args", {}).get("path"), target.rsplit("/", 1)[-1])
                for s in tool_steps
            )

            if not has_compile_ok:
                compile_result = await registry.execute("compile_python", {"path": target})
                trace.append(
                    {
                        "tool": "compile_python",
                        "args": {"path": target},
                        "result": compile_result.payload,
                        "ok": compile_result.ok,
                    }
                )
                self._update_scratchpad_after_tool(
                    scratchpad=scratchpad,
                    tool_name="compile_python",
                    ok=compile_result.ok,
                    args={"path": target},
                    payload=compile_result.payload,
                    target_file=target.rsplit("/", 1)[-1],
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": "auto_compile_written",
                        "content": tool_result_json(compile_result.payload),
                    }
                )

            if wants_run:
                has_run_ok = any(
                    s.get("tool") == "run_python_file"
                    and s.get("ok")
                    and self._path_matches_target(s.get("args", {}).get("path"), target.rsplit("/", 1)[-1])
                    for s in tool_steps
                )
                if not has_run_ok:
                    run_result = await registry.execute("run_python_file", {"path": target, "args": []})
                    trace.append(
                        {
                            "tool": "run_python_file",
                            "args": {"path": target, "args": []},
                            "result": run_result.payload,
                            "ok": run_result.ok,
                        }
                    )
                    self._update_scratchpad_after_tool(
                        scratchpad=scratchpad,
                        tool_name="run_python_file",
                        ok=run_result.ok,
                        args={"path": target, "args": []},
                        payload=run_result.payload,
                        target_file=target.rsplit("/", 1)[-1],
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": "auto_run_written",
                            "content": tool_result_json(run_result.payload),
                        }
                    )

        return trace, messages

    async def _auto_validate_written_cpp_files(
        self,
        user_message: str,
        trace: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        scratchpad: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        wants_run = any(k in user_message.lower() for k in ["run", "execute"])
        tool_steps = [s for s in trace if "tool" in s]

        written_targets: list[str] = []
        for step in tool_steps:
            if step.get("tool") != "write_file" or not step.get("ok"):
                continue
            path = str(step.get("args", {}).get("path", ""))
            if not self._is_cpp_file(path):
                continue
            normalized = path.replace("\\", "/")
            if normalized not in written_targets:
                written_targets.append(normalized)

        for target in written_targets:
            base = target.rsplit("/", 1)[-1]
            has_compile_ok = any(
                s.get("tool") == "compile_cpp"
                and s.get("ok")
                and self._path_matches_target(s.get("args", {}).get("path"), base)
                for s in tool_steps
            )

            if not has_compile_ok:
                compile_result = await registry.execute("compile_cpp", {"path": target})
                trace.append(
                    {
                        "tool": "compile_cpp",
                        "args": {"path": target},
                        "result": compile_result.payload,
                        "ok": compile_result.ok,
                    }
                )
                self._update_scratchpad_after_tool(
                    scratchpad=scratchpad,
                    tool_name="compile_cpp",
                    ok=compile_result.ok,
                    args={"path": target},
                    payload=compile_result.payload,
                    target_file=base,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": "auto_compile_cpp_written",
                        "content": tool_result_json(compile_result.payload),
                    }
                )

            if wants_run:
                run_path = target
                for s in reversed(trace):
                    if s.get("tool") == "compile_cpp" and isinstance(s.get("result"), dict):
                        candidate = str(s["result"].get("output", "")).strip()
                        if candidate and self._path_matches_target(s.get("args", {}).get("path"), base):
                            run_path = candidate
                            break
                has_run_ok = any(
                    s.get("tool") == "run_binary"
                    and s.get("ok")
                    and self._path_matches_target(s.get("args", {}).get("path"), run_path.rsplit("/", 1)[-1])
                    for s in tool_steps
                )
                if not has_run_ok:
                    run_result = await registry.execute("run_binary", {"path": run_path, "args": []})
                    trace.append(
                        {
                            "tool": "run_binary",
                            "args": {"path": run_path, "args": []},
                            "result": run_result.payload,
                            "ok": run_result.ok,
                        }
                    )
                    self._update_scratchpad_after_tool(
                        scratchpad=scratchpad,
                        tool_name="run_binary",
                        ok=run_result.ok,
                        args={"path": run_path, "args": []},
                        payload=run_result.payload,
                        target_file=base,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": "auto_run_cpp_written",
                            "content": tool_result_json(run_result.payload),
                        }
                    )

        return trace, messages

    def _supervisor_model(self) -> str:
        return settings.supervisor_model.strip() or settings.model

    def _supervisor_fallback_model(self) -> str:
        return settings.supervisor_fallback_model.strip() or settings.fallback_model

    def _fallback_plan(self, user_message: str, target_file: str | None, require_tools: bool) -> dict[str, Any]:
        todo: list[str] = []
        text = user_message.lower()

        if target_file:
            todo.append(f"Create or update {target_file}.")
        if "compile" in text or (target_file and target_file.endswith(".py")):
            todo.append("Compile the Python file to validate syntax.")
        if "build" in text or (target_file and self._is_cpp_file(target_file)):
            todo.append("Compile the C++ file to validate build success.")
        if "run" in text or "execute" in text:
            todo.append("Run the program and capture output.")
        if require_tools and not todo:
            todo.append("Inspect workspace and use tools to complete the task.")

        return {
            "objective": user_message[:280],
            "todo": todo,
            "constraints": ["Stay inside workspace root"],
            "tool_strategy": [
                "list_dir",
                "search_files",
                "read_file",
                "replace_in_file",
                "write_file",
                "compile_python",
                "run_python_file",
                "compile_cpp",
                "run_binary",
            ],
            "done_when": "All requested file operations are completed and validated.",
        }

    def _init_scratchpad(
        self,
        brief: dict[str, Any],
        user_message: str,
        target_file: str | None,
        require_tools: bool,
    ) -> dict[str, Any]:
        plan = brief or self._fallback_plan(user_message, target_file, require_tools)
        todo_items = []
        for item in plan.get("todo", []):
            todo_items.append({"task": str(item), "done": False})

        return {
            "objective": str(plan.get("objective", user_message[:280])),
            "target_file": target_file,
            "todo": todo_items,
            "notes": [],
        }

    def _snapshot_scratchpad(self, scratchpad: dict[str, Any]) -> dict[str, Any]:
        # Roundtrip through JSON to avoid mutable references in trace snapshots.
        return json.loads(json.dumps(scratchpad))

    def _scratchpad_prompt(self, scratchpad: dict[str, Any]) -> str:
        pending = [item["task"] for item in scratchpad.get("todo", []) if not item.get("done")]
        completed = [item["task"] for item in scratchpad.get("todo", []) if item.get("done")]
        notes = scratchpad.get("notes", [])[-5:]
        payload = {
            "objective": scratchpad.get("objective", ""),
            "target_file": scratchpad.get("target_file"),
            "pending_todo": pending,
            "completed_todo": completed,
            "recent_notes": notes,
        }
        return (
            "Scratchpad state (authoritative). Keep working until pending_todo is empty and objective is satisfied. "
            f"State JSON: {json.dumps(payload, ensure_ascii=True)}"
        )

    def _update_scratchpad_after_tool(
        self,
        scratchpad: dict[str, Any],
        tool_name: str,
        ok: bool,
        args: dict[str, Any],
        payload: dict[str, Any],
        target_file: str | None,
    ) -> None:
        if not isinstance(args, dict):
            args = {}
        if not isinstance(payload, dict):
            payload = {}

        note = f"{tool_name}:{'ok' if ok else 'fail'}"
        path = args.get("path")
        if isinstance(path, str) and path.strip():
            note = f"{note}:{path}"
        scratchpad.setdefault("notes", []).append(note)
        scratchpad["notes"] = scratchpad["notes"][-30:]

        if tool_name == "write_file" and ok:
            if target_file is None or self._path_matches_target(args.get("path"), target_file):
                self._mark_todo_done(scratchpad, ["create", "write", "update", "file"])

        if tool_name in {"compile_python", "compile_cpp"} and ok:
            if target_file is None or self._path_matches_target(args.get("path"), target_file):
                self._mark_todo_done(scratchpad, ["compile", "syntax", "validate"])

        if tool_name in {"run_python_file", "run_command", "run_binary"} and ok:
            if tool_name == "run_command" or target_file is None or self._path_matches_target(args.get("path"), target_file):
                self._mark_todo_done(scratchpad, ["run", "execute", "output"])

    def _path_matches_target(self, value: Any, target_file: str) -> bool:
        if not isinstance(value, str):
            return False
        normalized = value.replace("\\", "/").lower()
        target_norm = target_file.lower()
        return normalized.endswith("/" + target_norm) or normalized.endswith(target_norm)

    def _mark_todo_done(self, scratchpad: dict[str, Any], keywords: list[str]) -> None:
        for item in scratchpad.get("todo", []):
            task = str(item.get("task", "")).lower()
            if any(keyword in task for keyword in keywords):
                item["done"] = True

    def _is_repetitive_tool_loop(self, trace: list[dict[str, Any]]) -> bool:
        tool_steps = [s for s in trace if "tool" in s]
        if len(tool_steps) < 8:
            return False
        last = tool_steps[-8:]
        names = [str(s.get("tool", "")) for s in last]
        unique = set(names)
        failures = sum(1 for s in last if not s.get("ok"))
        # Detect oscillation patterns like compile/run loops that keep failing.
        return len(unique) <= 2 and failures >= 3

    async def _recover_with_direct_codegen(
        self,
        user_message: str,
        history: list[dict[str, str]],
        target_file: str,
        trace: list[dict[str, Any]],
        scratchpad: dict[str, Any],
        target_iterations: int,
        start_time: float,
    ) -> dict[str, Any] | None:
        if not target_file.lower().endswith(".py"):
            return None

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Generate complete Python source code for the requested task. "
                    "Output only raw Python code, no markdown fences, no explanations."
                ),
            }
        ]
        messages.extend(history[-6:])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Task: {user_message}\n"
                    f"Target file: {target_file}\n"
                    "Return complete, runnable Python code."
                ),
            }
        )

        code_resp = await self.llm.complete(messages=messages, tools=None, temperature=0.0)
        code_msg = code_resp["choices"][0]["message"].get("content", "")
        code = self._extract_code_content(code_msg)
        if not code.strip():
            return None

        write_result = await registry.execute("write_file", {"path": target_file, "content": code, "append": False})
        trace.append({"tool": "write_file", "args": {"path": target_file, "append": False}, "result": write_result.payload, "ok": write_result.ok})
        self._update_scratchpad_after_tool(scratchpad, "write_file", write_result.ok, {"path": target_file, "append": False}, write_result.payload, target_file)

        compile_result = await registry.execute("compile_python", {"path": target_file})
        trace.append({"tool": "compile_python", "args": {"path": target_file}, "result": compile_result.payload, "ok": compile_result.ok})
        self._update_scratchpad_after_tool(scratchpad, "compile_python", compile_result.ok, {"path": target_file}, compile_result.payload, target_file)

        run_result = await registry.execute("run_python_file", {"path": target_file, "args": []})
        trace.append({"tool": "run_python_file", "args": {"path": target_file, "args": []}, "result": run_result.payload, "ok": run_result.ok})
        self._update_scratchpad_after_tool(scratchpad, "run_python_file", run_result.ok, {"path": target_file, "args": []}, run_result.payload, target_file)

        trace.append({"scratchpad": self._snapshot_scratchpad(scratchpad)})

        if self._coding_objective_completed(user_message, trace, target_file):
            return {
                "answer": self._build_completion_summary(user_message, trace, target_file),
                "steps": trace,
                "latency_ms": int((time.perf_counter() - start_time) * 1000),
                "iterations": target_iterations,
            }
        return None

    def _extract_code_content(self, content: str) -> str:
        text = (content or "").strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return text

    def _normalize_answer(self, content: str) -> str:
        text = (content or "").strip()
        if not text:
            return text

        candidate = text
        if candidate.startswith("```") and candidate.endswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()

        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return text

        if isinstance(obj, (int, float, bool)):
            return str(obj)
        if obj is None:
            return "null"
        if isinstance(obj, str) and obj.strip():
            return obj.strip()

        if isinstance(obj, dict):
            args = obj.get("arguments")
            if isinstance(args, dict):
                msg = args.get("message") or args.get("text")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            ans = obj.get("answer")
            if isinstance(ans, str) and ans.strip():
                return ans.strip()

        return text
