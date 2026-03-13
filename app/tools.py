import asyncio
import json
import os
import re
import shutil
import shlex
import sys
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field, ValidationError

from app.config import normalize_workspace_root, settings


_workspace_root_ctx: ContextVar[str] = ContextVar("workspace_root", default=str(Path(settings.workspace_root).resolve()))


class ToolArgs(BaseModel):
    pass


class ReadFileArgs(ToolArgs):
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=300, ge=1)


class WriteFileArgs(ToolArgs):
    path: str
    content: str
    append: bool = False


class ListDirArgs(ToolArgs):
    path: str = "."


class RunCommandArgs(ToolArgs):
    command: str
    timeout_s: int | None = Field(default=None, ge=1)
    shell: str = Field(default="powershell")


class CompilePythonArgs(ToolArgs):
    path: str


class RunPythonFileArgs(ToolArgs):
    path: str
    args: list[str] = Field(default_factory=list)


class SearchFilesArgs(ToolArgs):
    pattern: str = "**/*"
    max_results: int = Field(default=200, ge=1, le=2000)


class ReplaceInFileArgs(ToolArgs):
    path: str
    search: str
    replace: str
    max_replacements: int = Field(default=0, ge=0)


class MovePathArgs(ToolArgs):
    src: str
    dst: str
    overwrite: bool = False


class DeletePathArgs(ToolArgs):
    path: str
    recursive: bool = False


class CompileCppArgs(ToolArgs):
    path: str
    output: str = ""
    std: str = "c++20"
    extra_args: list[str] = Field(default_factory=list)


class RunBinaryArgs(ToolArgs):
    path: str
    args: list[str] = Field(default_factory=list)
    timeout_s: int | None = Field(default=None, ge=1)


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[ToolArgs]
    handler: Callable[[ToolArgs], Awaitable[dict[str, Any]]]


@dataclass
class ToolExecutionResult:
    ok: bool
    name: str
    args: dict[str, Any]
    payload: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._specs[spec.name] = spec

    def has_tool(self, name: str) -> bool:
        return name in self._specs

    def to_openai_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for spec in self._specs.values():
            schema = spec.args_model.model_json_schema()
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": schema,
                    },
                }
            )
        return tools

    def _decode_raw_arguments(self, raw_args: Any) -> dict[str, Any]:
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            text = raw_args.strip()
            if not text:
                return {}
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("Tool arguments JSON must decode to an object")
            return parsed
        raise ValueError(f"Unsupported tool arguments type: {type(raw_args).__name__}")

    async def execute(self, name: str, raw_args: Any) -> ToolExecutionResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolExecutionResult(
                ok=False,
                name=name,
                args={},
                payload={"error": f"Unknown tool: {name}"},
            )

        try:
            decoded = self._decode_raw_arguments(raw_args)
        except (json.JSONDecodeError, ValueError) as exc:
            return ToolExecutionResult(
                ok=False,
                name=name,
                args={},
                payload={"error": f"Invalid tool arguments: {exc}", "raw_arguments": str(raw_args)[:1200]},
            )

        try:
            validated = spec.args_model.model_validate(decoded)
        except ValidationError as exc:
            return ToolExecutionResult(
                ok=False,
                name=name,
                args=decoded,
                payload={"error": "Tool argument validation failed", "details": exc.errors()},
            )

        try:
            result = await spec.handler(validated)
            ok = True
            if isinstance(result, dict):
                exit_code = result.get("exit_code")
                if isinstance(exit_code, int) and exit_code != 0:
                    ok = False
            return ToolExecutionResult(
                ok=ok,
                name=name,
                args=validated.model_dump(),
                payload=result,
            )
        except Exception as exc:
            return ToolExecutionResult(
                ok=False,
                name=name,
                args=validated.model_dump(),
                payload={"error": str(exc)},
            )


def _workspace_root() -> Path:
    return Path(_workspace_root_ctx.get()).resolve()


def set_workspace_root(path: str) -> Token:
    target = normalize_workspace_root(path)
    if not target.exists() or not target.is_dir():
        raise ValueError(f"Workspace root is not a directory: {path}")
    return _workspace_root_ctx.set(str(target))


def reset_workspace_root(token: Token) -> None:
    _workspace_root_ctx.reset(token)


def _resolve_path(raw: str) -> Path:
    root = _workspace_root()
    normalized_raw = raw
    if raw.startswith("/mnt/") or re.match(r"^[a-zA-Z]:[\\/]", raw):
        normalized_raw = str(normalize_workspace_root(raw))

    candidate_path = Path(normalized_raw)
    candidate = (root / normalized_raw).resolve() if not candidate_path.is_absolute() else candidate_path.resolve()
    if not str(candidate).startswith(str(root)):
        raise ValueError(f"Path escapes workspace root: {raw}")
    return candidate


async def _read_file(args: ReadFileArgs) -> dict[str, Any]:
    target = _resolve_path(args.path)
    text = target.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = max(args.start_line, 1)
    end = max(args.end_line, start)
    segment = lines[start - 1 : end]
    joined = "\n".join(segment)
    return {"path": str(target), "content": joined[: settings.max_file_chars], "total_lines": len(lines)}


async def _write_file(args: WriteFileArgs) -> dict[str, Any]:
    target = _resolve_path(args.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with target.open(mode, encoding="utf-8") as f:
        f.write(args.content)
    return {"path": str(target), "bytes_written": len(args.content.encode("utf-8")), "append": args.append}


async def _list_dir(args: ListDirArgs) -> dict[str, Any]:
    target = _resolve_path(args.path)
    if not target.is_dir():
        raise ValueError(f"Not a directory: {args.path}")
    entries = []
    for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
        entries.append({"name": child.name, "type": "dir" if child.is_dir() else "file"})
    return {"path": str(target), "entries": entries}


async def _run_command(args: RunCommandArgs) -> dict[str, Any]:
    timeout = args.timeout_s or settings.command_timeout_s

    shell = args.shell.lower()
    if shell == "powershell":
        cmd = ["powershell.exe", "-NoProfile", "-Command", args.command]
    elif shell == "cmd":
        cmd = ["cmd.exe", "/c", args.command]
    else:
        return {"exit_code": 2, "stdout": "", "stderr": f"Unsupported shell: {args.shell}"}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(_workspace_root()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"exit_code": 124, "stdout": "", "stderr": f"Timed out after {timeout}s"}

    stdout = out.decode("utf-8", errors="replace")[: settings.max_file_chars]
    stderr = err.decode("utf-8", errors="replace")[: settings.max_file_chars]
    return {"exit_code": proc.returncode, "stdout": stdout, "stderr": stderr}


async def _compile_python(args: CompilePythonArgs) -> dict[str, Any]:
    target = _resolve_path(args.path)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "py_compile",
        str(target),
        cwd=str(_workspace_root()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return {
        "path": str(target),
        "exit_code": proc.returncode,
        "stdout": out.decode("utf-8", errors="replace")[: settings.max_file_chars],
        "stderr": err.decode("utf-8", errors="replace")[: settings.max_file_chars],
    }


async def _run_python_file(args: RunPythonFileArgs) -> dict[str, Any]:
    target = _resolve_path(args.path)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(target),
        *args.args,
        cwd=str(_workspace_root()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return {
        "path": str(target),
        "exit_code": proc.returncode,
        "stdout": out.decode("utf-8", errors="replace")[: settings.max_file_chars],
        "stderr": err.decode("utf-8", errors="replace")[: settings.max_file_chars],
    }


async def _search_files(args: SearchFilesArgs) -> dict[str, Any]:
    root = _workspace_root()
    matches: list[dict[str, Any]] = []
    count = 0
    for candidate in root.glob(args.pattern):
        count += 1
        if len(matches) >= args.max_results:
            break
        kind = "dir" if candidate.is_dir() else "file"
        rel = candidate.relative_to(root).as_posix()
        matches.append({"path": rel, "type": kind})

    return {
        "root": str(root),
        "pattern": args.pattern,
        "returned": len(matches),
        "scanned": count,
        "entries": matches,
    }


async def _replace_in_file(args: ReplaceInFileArgs) -> dict[str, Any]:
    target = _resolve_path(args.path)
    text = target.read_text(encoding="utf-8")
    original_count = text.count(args.search)
    if original_count == 0:
        return {"path": str(target), "replacements": 0, "matched": 0}

    if args.max_replacements > 0:
        updated = text.replace(args.search, args.replace, args.max_replacements)
    else:
        updated = text.replace(args.search, args.replace)

    target.write_text(updated, encoding="utf-8")
    replacements = original_count if args.max_replacements == 0 else min(original_count, args.max_replacements)
    return {"path": str(target), "replacements": replacements, "matched": original_count}


async def _move_path(args: MovePathArgs) -> dict[str, Any]:
    src = _resolve_path(args.src)
    dst = _resolve_path(args.dst)
    if not src.exists():
        raise ValueError(f"Source does not exist: {args.src}")
    if dst.exists() and not args.overwrite:
        raise ValueError(f"Destination exists (set overwrite=true): {args.dst}")
    if dst.exists() and args.overwrite:
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {"src": str(src), "dst": str(dst), "overwrote": args.overwrite}


async def _delete_path(args: DeletePathArgs) -> dict[str, Any]:
    target = _resolve_path(args.path)
    if not target.exists():
        return {"path": str(target), "deleted": False, "reason": "not_found"}
    if target.is_dir():
        if not args.recursive:
            raise ValueError("Refusing to delete directory without recursive=true")
        shutil.rmtree(target)
        return {"path": str(target), "deleted": True, "type": "dir"}
    target.unlink()
    return {"path": str(target), "deleted": True, "type": "file"}


def _to_wsl_path(path: Path) -> str:
    raw = str(path)
    if re.match(r"^[a-zA-Z]:\\", raw):
        drive = raw[0].lower()
        tail = raw[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{tail}"
    return raw


async def _compile_cpp(args: CompileCppArgs) -> dict[str, Any]:
    target = _resolve_path(args.path)
    if args.output.strip():
        out_path = _resolve_path(args.output)
    else:
        suffix = ".exe" if os.name == "nt" else ".out"
        out_path = target.with_suffix(suffix)

    cmd_base = ["-std=" + args.std, str(target), "-o", str(out_path), *args.extra_args]

    async def try_local(compiler: str) -> tuple[int, str, str] | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                compiler,
                *cmd_base,
                cwd=str(_workspace_root()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return None
        out, err = await proc.communicate()
        return (
            proc.returncode,
            out.decode("utf-8", errors="replace")[: settings.max_file_chars],
            err.decode("utf-8", errors="replace")[: settings.max_file_chars],
        )

    local = await try_local("g++") or await try_local("clang++")
    if local is not None:
        code, out, err = local
        return {"path": str(target), "output": str(out_path), "exit_code": code, "stdout": out, "stderr": err}

    if os.name == "nt":
        wsl_target = shlex.quote(_to_wsl_path(target))
        wsl_out = shlex.quote(_to_wsl_path(out_path))
        extra = " ".join(shlex.quote(x) for x in args.extra_args)
        cmd = f"(command -v g++ >/dev/null && g++ -std={shlex.quote(args.std)} {wsl_target} -o {wsl_out} {extra}) || (command -v clang++ >/dev/null && clang++ -std={shlex.quote(args.std)} {wsl_target} -o {wsl_out} {extra})"
        proc = await asyncio.create_subprocess_exec(
            "wsl.exe",
            "bash",
            "-lc",
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return {
            "path": str(target),
            "output": str(out_path),
            "exit_code": proc.returncode,
            "stdout": out.decode("utf-8", errors="replace")[: settings.max_file_chars],
            "stderr": err.decode("utf-8", errors="replace")[: settings.max_file_chars],
        }

    return {
        "path": str(target),
        "output": str(out_path),
        "exit_code": 127,
        "stdout": "",
        "stderr": "No C++ compiler found (g++/clang++).",
    }


async def _run_binary(args: RunBinaryArgs) -> dict[str, Any]:
    target = _resolve_path(args.path)
    timeout = args.timeout_s or settings.command_timeout_s

    async def run_local(exec_path: str, argv: list[str]) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            exec_path,
            *argv,
            cwd=str(_workspace_root()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return (124, "", f"Timed out after {timeout}s")
        return (
            proc.returncode,
            out.decode("utf-8", errors="replace")[: settings.max_file_chars],
            err.decode("utf-8", errors="replace")[: settings.max_file_chars],
        )

    try:
        code, out, err = await run_local(str(target), args.args)
        return {"path": str(target), "exit_code": code, "stdout": out, "stderr": err}
    except (FileNotFoundError, PermissionError):
        pass

    if os.name == "nt":
        wsl_bin = shlex.quote(_to_wsl_path(target))
        argv = " ".join(shlex.quote(a) for a in args.args)
        cmd = f"{wsl_bin} {argv}".strip()
        proc = await asyncio.create_subprocess_exec(
            "wsl.exe",
            "bash",
            "-lc",
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return {"path": str(target), "exit_code": 124, "stdout": "", "stderr": f"Timed out after {timeout}s"}
        return {
            "path": str(target),
            "exit_code": proc.returncode,
            "stdout": out.decode("utf-8", errors="replace")[: settings.max_file_chars],
            "stderr": err.decode("utf-8", errors="replace")[: settings.max_file_chars],
        }

    return {"path": str(target), "exit_code": 126, "stdout": "", "stderr": "Binary is not executable."}


registry = ToolRegistry()
registry.register(ToolSpec("read_file", "Read file content by line range.", ReadFileArgs, _read_file))
registry.register(ToolSpec("write_file", "Write text to file in workspace.", WriteFileArgs, _write_file))
registry.register(ToolSpec("list_dir", "List files and folders under a path.", ListDirArgs, _list_dir))
registry.register(ToolSpec("run_command", "Run shell command in workspace root.", RunCommandArgs, _run_command))
registry.register(ToolSpec("search_files", "Find files using a glob pattern relative to workspace root.", SearchFilesArgs, _search_files))
registry.register(ToolSpec("replace_in_file", "Replace text in a file with optional replacement limit.", ReplaceInFileArgs, _replace_in_file))
registry.register(ToolSpec("move_path", "Move or rename a file or directory inside workspace.", MovePathArgs, _move_path))
registry.register(ToolSpec("delete_path", "Delete a file or directory (recursive for directories).", DeletePathArgs, _delete_path))
registry.register(ToolSpec("compile_python", "Compile an existing Python file to validate syntax; use write_file first if missing.", CompilePythonArgs, _compile_python))
registry.register(ToolSpec("run_python_file", "Run a Python file with optional args.", RunPythonFileArgs, _run_python_file))
registry.register(ToolSpec("compile_cpp", "Compile a C++ source file (g++/clang++, with WSL fallback on Windows).", CompileCppArgs, _compile_cpp))
registry.register(ToolSpec("run_binary", "Run a compiled binary with optional arguments.", RunBinaryArgs, _run_binary))


def tool_result_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=True)
