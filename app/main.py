"""FastAPI application — the REST surface for the RUKU agent."""
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .agent import AutonomousAgent
from .config import settings
from .llm import build_llm
from .models import AgentRequest, AgentResponse
from .store import JobStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="RUKU — Autonomous Document Agent",
    version="1.0.0",
    description="Turns a natural-language request into an autonomously planned Word document.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = JobStore()
llm = build_llm(settings)
agent = AutonomousAgent(llm, store, settings)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _safe_filename(title: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
    return (keep[:60] or "document") + ".docx"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    with open(os.path.join(_STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": settings.provider_name,
        "model": settings.groq_model if settings.provider_name == "groq" else None,
    }


@app.post("/agent", response_model=AgentResponse)
def run_agent(req: AgentRequest) -> AgentResponse:
    try:
        return agent.run(req.request, req.session_id)
    except Exception as exc:
        logging.exception("agent run failed")
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}")


@app.get("/jobs/{document_id}", response_model=AgentResponse)
def get_job(document_id: str) -> AgentResponse:
    record = store.get(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="document not found")
    return record["response"]


@app.get("/download/{document_id}")
def download(document_id: str) -> FileResponse:
    record = store.get(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="document not found")
    path = record["path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=410, detail="file no longer available")
    return FileResponse(
        path, media_type=_DOCX_MIME, filename=_safe_filename(record["response"].title)
    )
