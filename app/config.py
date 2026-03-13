from pathlib import Path
import os
import re

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Local Agent LLM"
    host: str = "127.0.0.1"
    port: int = 8000

    model: str = "qwen3:8b"
    fallback_model: str = "qwen2.5-coder:1.5b"
    supervisor_model: str = ""
    supervisor_fallback_model: str = ""
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    temperature: float = 0.15
    max_tokens: int = 1200
    llm_timeout_s: int = 60
    max_agent_steps: int = 8
    max_tool_errors: int = 3
    enable_supervisor: bool = True

    workspace_root: str = Field(default_factory=lambda: str(Path.cwd()))
    command_timeout_s: int = 20
    max_file_chars: int = 30000


settings = Settings()


def normalize_workspace_root(raw: str) -> Path:
    # Fix Windows-mangled WSL form: C:\mnt\c\... -> C:\...
    mangled = re.match(r"^[a-zA-Z]:\\mnt\\([a-zA-Z])\\(.*)$", raw)
    if mangled:
        drive = mangled.group(1).upper()
        tail = mangled.group(2)
        raw = f"{drive}:\\{tail}"

    # Convert WSL path -> Windows path when running Windows Python.
    if os.name == "nt" and raw.startswith("/mnt/"):
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw)
        if match:
            drive = match.group(1).upper()
            tail = match.group(2).replace("/", "\\")
            raw = f"{drive}:\\{tail}"

    # Convert Windows path -> WSL path when running Linux Python.
    if os.name != "nt" and re.match(r"^[a-zA-Z]:\\", raw):
        drive = raw[0].lower()
        tail = raw[2:].replace("\\", "/").lstrip("/")
        raw = f"/mnt/{drive}/{tail}"

    return Path(raw).resolve()


settings.workspace_root = str(normalize_workspace_root(settings.workspace_root))
