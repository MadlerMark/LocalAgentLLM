import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human-in-the-loop model tuner")
    parser.add_argument("--prompts", help="Path to prompts text file (one prompt per line)")
    parser.add_argument("--models", default="", help="Comma-separated model ids")
    parser.add_argument("--out", default="", help="Output json file path")
    return parser.parse_args()


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return [
            "List top-level files in the current workspace and briefly describe each.",
            "Given a bug report, propose a minimal debugging plan.",
            "Write a concise Python function that reads JSON safely with error handling.",
        ]

    content = Path(path).read_text(encoding="utf-8")
    prompts = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not prompts:
        raise ValueError("No prompts found in prompts file")
    return prompts


def load_models(arg_models: str) -> list[str]:
    if arg_models.strip():
        return [m.strip() for m in arg_models.split(",") if m.strip()]

    env_model = os.getenv("MODEL", "qwen2.5-coder:1.5b").strip()
    return [env_model]


def ask_score() -> tuple[int, str]:
    while True:
        raw = input("Score 1-5 (5 best): ").strip()
        if raw in {"1", "2", "3", "4", "5"}:
            notes = input("Notes (optional): ").strip()
            return int(raw), notes
        print("Please enter 1,2,3,4,5")


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    models = load_models(args.models)

    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    client = OpenAI(base_url=base_url, api_key=api_key)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or f"hitl_results_{ts}.json"

    records: list[dict] = []

    print("HITL tuning session")
    print(f"base_url={base_url}")
    print(f"models={models}")
    print(f"prompts={len(prompts)}")

    for p_idx, prompt in enumerate(prompts, start=1):
        print("\n" + "=" * 72)
        print(f"Prompt {p_idx}/{len(prompts)}")
        print(prompt)

        for model in models:
            print("\n" + "-" * 72)
            print(f"Model: {model}")
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=700,
                )
                answer = (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                answer = f"[ERROR] {exc}"

            print(answer)
            score, notes = ask_score()
            records.append(
                {
                    "prompt": prompt,
                    "model": model,
                    "answer": answer,
                    "score": score,
                    "notes": notes,
                }
            )

    Path(out).write_text(json.dumps(records, indent=2, ensure_ascii=True), encoding="utf-8")

    by_model: dict[str, list[int]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r["score"])

    print("\nSummary")
    for model, scores in by_model.items():
        avg = sum(scores) / len(scores)
        print(f"{model}: avg={avg:.2f} n={len(scores)}")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
