from openai import AsyncOpenAI
import asyncio

from app.config import settings


class LLMClient:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        temperature: float | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
    ) -> dict:
        payload_base: dict = {
            "messages": messages,
            "temperature": settings.temperature if temperature is None else temperature,
            "max_tokens": settings.max_tokens,
        }
        if tools is not None:
            payload_base["tools"] = tools
            payload_base["tool_choice"] = tool_choice or "auto"

        primary_model = model or settings.model
        secondary_model = fallback_model if fallback_model is not None else settings.fallback_model

        async def run_with_model(model_name: str) -> dict:
            payload = {"model": model_name, **payload_base}
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(**payload),
                timeout=settings.llm_timeout_s,
            )
            return resp.model_dump()

        try:
            return await run_with_model(primary_model)
        except TimeoutError as exc:
            raise RuntimeError(f"LLM request timed out after {settings.llm_timeout_s}s") from exc
        except Exception as exc:
            msg = str(exc).lower()
            fallbackable = "not found" in msg or "unknown model" in msg or "model" in msg
            if fallbackable and secondary_model and secondary_model != primary_model:
                return await run_with_model(secondary_model)
            raise
