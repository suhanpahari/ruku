"""End-to-end tests that run fully offline (no API key required)."""
import zipfile

import pytest

from app.agent import AutonomousAgent
from app.config import settings
from app.llm import build_llm
from app.models import AgentRequest
from app.store import JobStore


def _make_agent(tmp_path) -> AutonomousAgent:
    settings.groq_api_key = ""  # force the deterministic offline provider
    settings.output_dir = str(tmp_path)
    return AutonomousAgent(build_llm(settings), JobStore(), settings)


def test_standard_request_produces_valid_docx(tmp_path):
    agent = _make_agent(tmp_path)
    resp = agent.run(
        "Write a business proposal for a mobile app that helps small cafes run loyalty programs."
    )

    record = agent.store.get(resp.document_id)
    assert record is not None
    path = record["path"]

    # A .docx is a zip; assert it exists, is non-empty, and is a valid archive.
    assert zipfile.is_zipfile(path)
    assert len(resp.plan) >= 5
    assert all(step.status == "done" for step in resp.plan)
    assert resp.document_type == "proposal"
    assert resp.download_url == f"/download/{resp.document_id}"


def test_complex_request_records_assumptions(tmp_path):
    agent = _make_agent(tmp_path)
    resp = agent.run(
        "We have a client meeting next week — maybe put together a project plan or a short "
        "report. Budget is tight and the timeline isn't set. Decide what makes sense."
    )
    # Ambiguous input should trigger the agent to state assumptions.
    assert resp.assumptions, "expected the agent to record assumptions for an ambiguous request"
    assert any("reflection" in n.lower() for n in resp.reflection_notes)


def test_request_validation_guardrails():
    with pytest.raises(Exception):
        AgentRequest(request="")
    with pytest.raises(Exception):
        AgentRequest(request="hi")  # too short
