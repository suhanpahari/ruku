"""The autonomous agent orchestrator: plan → execute → reflect → assemble."""
import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from .config import Settings
from .document import build_document
from .llm import ResilientLLM
from .models import AgentResponse
from .planner import build_plan_steps, set_status
from .store import JobStore

log = logging.getLogger("ruku.agent")


class AutonomousAgent:
    def __init__(self, llm: ResilientLLM, store: JobStore, settings: Settings) -> None:
        self.llm = llm
        self.store = store
        self.settings = settings

    def run(self, request: str, session_id: Optional[str] = None) -> AgentResponse:
        doc_id = uuid4().hex[:12]
        self.llm.reset()
        log.info("run %s: %s", doc_id, request[:80])

        # 1. Plan — the agent decides document type, outline and assumptions.
        plan = self.llm.plan(request)
        steps = build_plan_steps(plan)
        set_status(steps, "interpret", "done")
        set_status(steps, "outline", "done")

        # 2. Execute — draft each planned section.
        drafted: List[dict] = []
        section_names = [s["name"] for s in plan["sections"]]
        for i, sec in enumerate(plan["sections"]):
            step_id = f"draft_{i}"
            set_status(steps, step_id, "running")
            content = self.llm.draft_section(
                request, plan["document_type"], sec["name"], section_names[:i]
            )
            drafted.append(
                {
                    "name": sec["name"],
                    "paragraphs": content["paragraphs"],
                    "bullets": content["bullets"],
                }
            )
            set_status(steps, step_id, "done")

        # 3. Reflect — self-check and revise weak/missing sections.
        set_status(steps, "reflect", "running")
        reflection = self.llm.reflect(request, plan["document_type"], drafted)
        revisions_applied = self._apply_revisions(drafted, reflection.get("revised_sections", []))
        notes = list(reflection.get("notes", []))
        notes.append(f"Applied {revisions_applied} revision(s) during the reflection pass.")
        set_status(steps, "reflect", "done", detail=f"{revisions_applied} revision(s) applied")

        # 4. Assemble — build the .docx.
        set_status(steps, "assemble", "running")
        path = build_document(
            doc_id=doc_id,
            plan=plan,
            sections=drafted,
            assumptions=plan["assumptions"],
            reflection_notes=notes,
            request=request,
            out_dir=self.settings.output_dir,
        )
        set_status(steps, "assemble", "done")

        provider = self._provider_label()
        summary = (
            f"Generated a {plan['document_type']} titled “{plan['title']}” with "
            f"{len(drafted)} sections. Recorded {len(plan['assumptions'])} assumption(s) and "
            f"applied {revisions_applied} reflection revision(s). Provider: {provider}."
        )

        response = AgentResponse(
            document_id=doc_id,
            provider=provider,
            document_type=plan["document_type"],
            title=plan["title"],
            summary=summary,
            assumptions=plan["assumptions"],
            plan=steps,
            reflection_notes=notes,
            download_url=f"/download/{doc_id}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.store.save(doc_id, response, path)
        return response

    @staticmethod
    def _apply_revisions(drafted: List[dict], revised_sections: List[dict]) -> int:
        applied = 0
        for rev in revised_sections:
            for section in drafted:
                if section["name"].lower() == str(rev.get("name", "")).lower():
                    if rev.get("paragraphs"):
                        section["paragraphs"] = rev["paragraphs"]
                    if rev.get("bullets"):
                        section["bullets"] = rev["bullets"]
                    applied += 1
                    break
        return applied

    def _provider_label(self) -> str:
        base = self.settings.provider_name
        if self.llm.fallbacks and base != "offline":
            return f"{base}→offline(fallback)"
        return base
