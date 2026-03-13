import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LocalAgentLLM server launcher")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8011, type=int)
    parser.add_argument("--workspace-root", default=os.getcwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["WORKSPACE_ROOT"] = args.workspace_root

    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
