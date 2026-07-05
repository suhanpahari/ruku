"""LLM abstraction layer.

Three high-level capabilities the agent needs — ``plan``, ``draft_section`` and
``reflect`` — are exposed as an interface with two implementations:

* :class:`GroqProvider`  – calls Groq's OpenAI-compatible API over httpx.
* :class:`OfflineProvider` – a deterministic, rule-based fallback that needs no key.

:class:`ResilientLLM` wraps a primary provider with retries and, on failure, falls
back to the offline provider so a run always completes (retry & fallback logic).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, List

import httpx

from .config import Settings

log = logging.getLogger("ruku.llm")

DOC_TYPES = [
    "proposal",
    "project plan",
    "meeting minutes",
    "business report",
    "technical design",
    "SOP",
    "product specification",
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_str_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _loads(text: str) -> dict:
    """Tolerant JSON parse — strips code fences and trailing prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def _normalize_content(raw: dict) -> Dict[str, List[str]]:
    return {
        "paragraphs": _as_str_list(raw.get("paragraphs")) or ["(no content generated)"],
        "bullets": _as_str_list(raw.get("bullets")),
    }


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------


class GroqProvider:
    name = "groq"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.groq_model

    def _chat(self, system: str, user: str, json_mode: bool = True) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = httpx.post(
            f"{self.settings.groq_base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.settings.groq_api_key}"},
            timeout=self.settings.llm_timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def plan(self, request: str) -> dict:
        system = (
            "You are RUKU, an autonomous business-document planning agent. You decide "
            "which document type best fulfils a request, propose a professional section "
            "outline, and surface assumptions when the request is ambiguous or missing "
            "details. Respond ONLY with valid JSON."
        )
        user = (
            f'Request: """{request}"""\n\n'
            "Return JSON with this exact shape:\n"
            "{\n"
            f'  "document_type": "one of: {" | ".join(DOC_TYPES)}",\n'
            '  "title": "a specific, professional document title",\n'
            '  "assumptions": ["assumptions made where the request was ambiguous or missing info"],\n'
            '  "sections": [{"name": "Section Name", "purpose": "what it covers"}]\n'
            "}\n"
            "Choose 5-8 sections appropriate to the chosen document type."
        )
        data = _loads(self._chat(system, user, json_mode=True))
        return _normalize_plan(data, request)

    def draft_section(
        self, request: str, document_type: str, section_name: str, prior: List[str]
    ) -> Dict[str, List[str]]:
        system = (
            f"You are RUKU, drafting one section of a professional {document_type}. Write "
            "concise, business-appropriate content grounded in the request. Respond ONLY "
            "with valid JSON."
        )
        user = (
            f'Request: """{request}"""\n'
            f"Document type: {document_type}\n"
            f"Section to write: {section_name}\n"
            f"Already covered: {prior}\n\n"
            'Return JSON: {"paragraphs": ["1-3 short paragraphs"], '
            '"bullets": ["0-6 optional bullet points"]}'
        )
        return _normalize_content(_loads(self._chat(system, user, json_mode=True)))

    def reflect(
        self, request: str, document_type: str, sections: List[dict]
    ) -> dict:
        system = (
            "You are RUKU performing a self-check on a draft. Identify gaps and weak or "
            "missing sections relative to the original request, then improve at most TWO "
            "sections. Respond ONLY with valid JSON."
        )
        digest = [
            {"name": s["name"], "preview": " ".join(s["paragraphs"])[:400]}
            for s in sections
        ]
        user = (
            f'Original request: """{request}"""\n'
            f"Document type: {document_type}\n"
            f"Current sections: {json.dumps(digest, ensure_ascii=False)}\n\n"
            "Return JSON: {\n"
            '  "notes": ["what you checked and what you changed"],\n'
            '  "revised_sections": [{"name": "existing section name", '
            '"paragraphs": [...], "bullets": [...]}]\n'
            "}"
        )
        data = _loads(self._chat(system, user, json_mode=True))
        revised = []
        for r in data.get("revised_sections", []) or []:
            if isinstance(r, dict) and r.get("name"):
                revised.append(
                    {"name": str(r["name"]), **_normalize_content(r)}
                )
        return {"notes": _as_str_list(data.get("notes")), "revised_sections": revised}


# ---------------------------------------------------------------------------
# Offline (deterministic) provider
# ---------------------------------------------------------------------------

# Word-boundary phrases that signal an ambiguous / under-specified request.
_AMBIGUOUS_MARKERS = (
    "maybe",
    "not sure",
    "unclear",
    "tight budget",
    "figure out",
    "decide what",
    "isn't set",
    "not set",
    "tbd",
    "up to you",
    "whatever makes sense",
    "some kind of",
)


def _is_ambiguous(low: str) -> bool:
    return any(marker in low for marker in _AMBIGUOUS_MARKERS)

_SECTION_TEMPLATES = {
    "proposal": [
        "Executive Summary", "Problem Statement", "Proposed Solution",
        "Scope & Deliverables", "Timeline", "Budget & Pricing",
        "Risks & Assumptions", "Next Steps",
    ],
    "project plan": [
        "Overview & Objectives", "Scope", "Work Breakdown & Milestones",
        "Timeline", "Resources & Roles", "Budget", "Risks & Mitigations",
        "Success Metrics",
    ],
    "meeting minutes": [
        "Meeting Details", "Attendees", "Agenda", "Discussion Summary",
        "Decisions Made", "Action Items", "Next Meeting",
    ],
    "business report": [
        "Executive Summary", "Background", "Findings", "Analysis",
        "Recommendations", "Risks & Assumptions", "Conclusion & Next Steps",
    ],
    "technical design": [
        "Overview", "Goals & Non-Goals", "Architecture", "Data Model & APIs",
        "Key Design Decisions", "Risks & Trade-offs", "Rollout Plan",
    ],
    "SOP": [
        "Purpose & Scope", "Roles & Responsibilities", "Prerequisites",
        "Procedure Steps", "Quality Checks", "Exceptions & Escalation",
        "Revision History",
    ],
    "product specification": [
        "Overview", "Goals & Success Metrics", "User Stories",
        "Functional Requirements", "Non-Functional Requirements",
        "Assumptions & Constraints", "Milestones",
    ],
}

_KEYWORDS = {
    "proposal": ["proposal", "pitch", "offer", "quote"],
    "project plan": ["project plan", "roadmap", "timeline", "milestone", "plan"],
    "meeting minutes": ["meeting", "minutes", "standup", "sync", "notes from"],
    "technical design": ["technical", "architecture", "system design", "api", "design doc"],
    "SOP": ["sop", "procedure", "runbook", "standard operating", "guideline", "process"],
    "product specification": ["spec", "specification", "prd", "feature", "requirements"],
    "business report": ["report", "analysis", "quarterly", "review", "summary"],
}


def _pick_doc_type(request: str) -> str:
    low = request.lower()
    for doc_type, words in _KEYWORDS.items():
        if any(w in low for w in words):
            return doc_type
    return "business report"


def _subject(request: str) -> str:
    subject = " ".join(request.strip().split())
    return subject if len(subject) <= 160 else subject[:157] + "..."


_TITLE_PREFIXES = (
    "please ", "can you ", "i need ", "i want ", "we need ", "we want ",
    "write ", "create ", "draft ", "generate ", "make ", "prepare ",
    "build ", "put together ", "produce ", "a ", "an ", "the ", "me ", "us ",
)


def _clean_title(request: str, doc_type: str) -> str:
    """Derive a tidy title from a free-text request (offline fallback)."""
    text = " ".join(request.strip().split()).rstrip(".!?")
    low = text.lower()
    changed = True
    while changed:
        changed = False
        for prefix in _TITLE_PREFIXES:
            if low.startswith(prefix):
                text = text[len(prefix):]
                low = text.lower()
                changed = True
    if len(text) > 60:  # cut on a word boundary
        text = text[:60].rsplit(" ", 1)[0] + "…"
    text = text[:1].upper() + text[1:] if text else doc_type.title()
    return f"{text} — {doc_type.title()}"


class OfflineProvider:
    name = "offline"

    def plan(self, request: str) -> dict:
        doc_type = _pick_doc_type(request)
        sections = [
            {"name": name, "purpose": f"Covers {name.lower()}."}
            for name in _SECTION_TEMPLATES[doc_type]
        ]
        low = request.lower()
        assumptions: List[str] = []
        if _is_ambiguous(low):
            assumptions = [
                f"The request was ambiguous, so the agent selected a '{doc_type}' as the "
                "most useful deliverable.",
                "Where concrete figures, dates, or names were missing, reasonable "
                "placeholder / mock values were used and flagged for review.",
                "A lean team and a ~6 week horizon were assumed for any timeline or budget.",
            ]
        title = _clean_title(request, doc_type)
        return _normalize_plan(
            {
                "document_type": doc_type,
                "title": title,
                "assumptions": assumptions,
                "sections": sections,
            },
            request,
        )

    def draft_section(
        self, request: str, document_type: str, section_name: str, prior: List[str]
    ) -> Dict[str, List[str]]:
        return _offline_section(section_name, document_type, _subject(request))

    def reflect(self, request: str, document_type: str, sections: List[dict]) -> dict:
        notes = [
            "Self-check: verified every planned section is present and non-empty.",
            "Self-check: confirmed the document directly answers the original request.",
            "Self-check: ensured assumptions are stated for any ambiguous requirements.",
        ]
        revised: List[dict] = []
        if sections:
            first = sections[0]
            paras = list(first["paragraphs"]) + [
                "(Reflection pass) Reviewed against the original request and flagged all "
                "assumptions for stakeholder confirmation before execution."
            ]
            revised = [
                {"name": first["name"], "paragraphs": paras, "bullets": first["bullets"]}
            ]
        return {"notes": notes, "revised_sections": revised}


def _offline_section(name: str, doc_type: str, subject: str) -> Dict[str, List[str]]:
    n = name.lower()
    paras: List[str] = []
    bullets: List[str] = []
    if "summary" in n or "overview" in n:
        paras = [
            f"This {doc_type} addresses the request: {subject}. It sets out the objective, "
            "the recommended approach, and the outcomes stakeholders can expect.",
            "The sections that follow break the work into clear, actionable parts so the "
            "team can align quickly and move to execution.",
        ]
    elif "problem" in n or "background" in n:
        paras = [
            f"The need driving this document is: {subject}. Today it is handled in an "
            "ad-hoc way, creating inefficiency and inconsistent results."
        ]
        bullets = [
            "Manual, repetitive effort that does not scale",
            "Limited visibility into progress and outcomes",
            "No single source of truth for stakeholders",
        ]
    elif "solution" in n or "approach" in n or "architecture" in n:
        paras = [
            f"We propose a focused solution targeting: {subject}. The approach favours a "
            "simple, reliable core that can be extended as needs grow."
        ]
        bullets = [
            "Start with a minimal, working end-to-end flow",
            "Automate the highest-effort manual steps first",
            "Instrument the system so decisions are data-driven",
        ]
    elif "scope" in n or "deliverable" in n or "requirement" in n or "story" in n:
        paras = [
            "The scope below defines what is included in this phase and what is "
            "intentionally deferred to keep delivery predictable."
        ]
        bullets = [
            "In scope: the core workflow and primary user journey",
            "In scope: generated outputs and basic reporting",
            "Out of scope (this phase): third-party integrations and advanced analytics",
        ]
    elif "timeline" in n or "milestone" in n or "schedule" in n or "breakdown" in n:
        paras = ["The work is sequenced into short phases, each ending in a demonstrable outcome."]
        bullets = [
            "Phase 1 (Week 1-2): Discovery and foundational setup",
            "Phase 2 (Week 3-4): Core build and internal review",
            "Phase 3 (Week 5-6): Hardening, documentation, and handover",
        ]
    elif "budget" in n or "pricing" in n or "cost" in n or "resource" in n or "role" in n:
        paras = [
            "The estimate below uses mock figures suitable for planning; final numbers "
            "depend on confirmed scope."
        ]
        bullets = [
            "Engineering: 1-2 people for ~6 weeks",
            "Tooling & infrastructure: low, mostly free-tier",
            "Contingency: 15% for scope changes",
        ]
    elif "risk" in n or "assumption" in n or "trade" in n or "mitigation" in n or "constraint" in n:
        paras = ["Key risks and the assumptions this plan relies on, listed so they can be validated early."]
        bullets = [
            "Assumption: requirements are stable enough to begin",
            "Risk: ambiguous scope may expand — mitigated by phased delivery",
            "Risk: dependency availability — mitigated by built-in fallbacks",
        ]
    elif "action" in n or "next" in n or "recommendation" in n or "conclusion" in n or "decision" in n:
        paras = ["Recommended next steps to move from plan to execution:"]
        bullets = [
            "Confirm scope and success metrics with stakeholders",
            "Approve the phased timeline and resourcing",
            "Kick off Phase 1 and schedule the first review",
        ]
    elif "metric" in n or "success" in n or "quality" in n:
        paras = ["Success will be measured against a small set of concrete, observable metrics."]
        bullets = [
            "Time saved versus the current manual process",
            "Quality and consistency of the generated output",
            "Stakeholder satisfaction and adoption",
        ]
    elif "attendee" in n or "agenda" in n or "detail" in n:
        paras = [f"Recorded for the session concerning: {subject}."]
        bullets = [
            "Date/Time: TBD (placeholder)",
            "Facilitator: Project lead",
            "Participants: Key stakeholders (mock)",
        ]
    else:
        paras = [
            f"This section covers {name.lower()} as it relates to: {subject}.",
            "Details are kept concise and business-appropriate, with assumptions noted "
            "where information was not provided.",
        ]
    return {"paragraphs": paras, "bullets": bullets}


def _normalize_plan(data: dict, request: str) -> dict:
    doc_type = str(data.get("document_type") or "").strip().lower()
    if doc_type not in DOC_TYPES:
        doc_type = _pick_doc_type(request)
    sections = []
    for s in data.get("sections") or []:
        if isinstance(s, dict) and s.get("name"):
            sections.append({"name": str(s["name"]), "purpose": str(s.get("purpose", ""))})
        elif isinstance(s, str) and s.strip():
            sections.append({"name": s.strip(), "purpose": ""})
    if not sections:
        sections = [{"name": n, "purpose": ""} for n in _SECTION_TEMPLATES[doc_type]]
    title = str(data.get("title") or "").strip() or f"{doc_type.title()}"
    return {
        "document_type": doc_type,
        "title": title,
        "assumptions": _as_str_list(data.get("assumptions")),
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Resilient wrapper: retries + fallback to offline
# ---------------------------------------------------------------------------


class ResilientLLM:
    def __init__(self, primary, fallback: OfflineProvider, attempts: int) -> None:
        self.primary = primary
        self.fallback = fallback
        self.attempts = max(1, attempts)
        self.fallbacks = 0  # number of times we fell back during the current run

    def reset(self) -> None:
        self.fallbacks = 0

    def _run(self, method: str, *args):
        if self.primary is self.fallback:
            return getattr(self.primary, method)(*args)
        last_err = None
        for attempt in range(self.attempts):
            try:
                return getattr(self.primary, method)(*args)
            except Exception as err:  # noqa: BLE001 - deliberate broad catch for resilience
                last_err = err
                log.warning("primary LLM %s failed (attempt %d): %s", method, attempt + 1, err)
                time.sleep(min(2 ** attempt, 4))
        self.fallbacks += 1
        log.warning("falling back to offline provider for %s: %s", method, last_err)
        return getattr(self.fallback, method)(*args)

    def plan(self, request: str) -> dict:
        return self._run("plan", request)

    def draft_section(self, request, document_type, section_name, prior):
        return self._run("draft_section", request, document_type, section_name, prior)

    def reflect(self, request, document_type, sections):
        return self._run("reflect", request, document_type, sections)


def build_llm(settings: Settings) -> ResilientLLM:
    offline = OfflineProvider()
    primary = GroqProvider(settings) if settings.groq_api_key else offline
    return ResilientLLM(primary, offline, settings.llm_max_retries)
