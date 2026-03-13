import argparse
import asyncio
from pathlib import Path

from app.agent import AgentRuntime
from app.config import normalize_workspace_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LocalAgentLLM TUI")
    parser.add_argument("--root", default=str(Path.cwd()), help="Workspace root used for tool calls")
    parser.add_argument("--ask", default="", help="Run one prompt and exit")
    return parser.parse_args()


def _help_text() -> str:
    return (
        "Commands:\n"
        "  /help                Show this help\n"
        "  /root <path>         Change workspace root for tool calls\n"
        "  /pwd                 Show current workspace root\n"
        "  /clear               Clear chat history\n"
        "  /quit                Exit TUI\n"
    )


async def _run_tui(initial_root: str) -> None:
    agent = AgentRuntime()
    history: list[dict[str, str]] = []
    active_root = str(normalize_workspace_root(initial_root))

    print("LocalAgentLLM TUI")
    print(f"workspace_root={active_root}")
    print("Type /help for commands.")

    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            print("\nbye")
            return

        if not user:
            continue

        if user == "/quit":
            print("bye")
            return
        if user == "/help":
            print(_help_text())
            continue
        if user == "/pwd":
            print(active_root)
            continue
        if user == "/clear":
            history.clear()
            print("history cleared")
            continue
        if user.startswith("/root "):
            candidate = user[len("/root ") :].strip()
            try:
                resolved = normalize_workspace_root(candidate)
                if not resolved.exists() or not resolved.is_dir():
                    raise ValueError("not a directory")
                active_root = str(resolved)
                print(f"workspace_root={active_root}")
            except Exception as exc:
                print(f"root error: {exc}")
            continue

        try:
            result = await agent.run(user_message=user, history=history, workspace_root=active_root)
            answer = result.get("answer", "")
            print(f"agent> {answer}")
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": answer})
        except Exception as exc:
            print(f"agent error: {exc}")


async def _run_once(root: str, prompt: str) -> None:
    agent = AgentRuntime()
    active_root = str(normalize_workspace_root(root))
    result = await agent.run(user_message=prompt, history=[], workspace_root=active_root)
    print(result.get("answer", ""))


def main() -> None:
    args = _parse_args()
    if args.ask.strip():
        asyncio.run(_run_once(args.root, args.ask.strip()))
        return
    asyncio.run(_run_tui(args.root))


if __name__ == "__main__":
    main()
