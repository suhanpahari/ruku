"""Pydantic schemas + request validation / guardrails."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import settings


class AgentRequest(BaseModel):
    """Incoming request. Validation here is the first guardrail on POST /agent."""

    request: str = Field(..., description="Natural language request for a document")
    session_id: Optional[str] = Field(default=None, description="Optional client session id")

    @field_validator("request")
    @classmethod
    def _validate_request(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("`request` must not be empty")
        if len(v) < 5:
            raise ValueError("`request` is too short to plan a meaningful document")
        if len(v) > settings.max_request_chars:
            raise ValueError(
                f"`request` exceeds the {settings.max_request_chars} character limit"
            )
        return v


class PlanStep(BaseModel):
    """A single item in the agent's autonomously generated TODO list."""

    id: str
    title: str
    kind: str
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    detail: Optional[str] = None


class AgentResponse(BaseModel):
    document_id: str
    provider: str
    document_type: str
    title: str
    summary: str
    assumptions: List[str] = []
    plan: List[PlanStep] = []
    reflection_notes: List[str] = []
    download_url: str
    created_at: str
