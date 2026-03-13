from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import AgentRuntime
from app.config import settings


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    workspace_root: str | None = None


class ChatResponse(BaseModel):
    answer: str
    iterations: int
    latency_ms: int
    steps: list[dict]


app = FastAPI(title=settings.app_name)
agent = AgentRuntime()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "model": settings.model,
        "supervisor_model": settings.supervisor_model or settings.model,
        "supervisor_fallback_model": settings.supervisor_fallback_model or settings.fallback_model,
        "base_url": settings.llm_base_url,
        "workspace_root": str(settings.workspace_root),
        "supervisor_enabled": settings.enable_supervisor,
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    history = [{"role": msg.role, "content": msg.content} for msg in req.history]
    result = await agent.run(user_message=req.message, history=history, workspace_root=req.workspace_root)
    return ChatResponse(**result)
