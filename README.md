# RUKU — Autonomous Document Agent

An autonomous AI agent that takes a natural-language request, **plans its own TODO list**,
executes each step, **self-checks (reflection)**, and produces a polished **Microsoft Word
(.docx)** business document. Exposed as a FastAPI service with a small web UI.

Built for the *Python AI Engineer — Autonomous Agents* 60-minute build challenge.

---

## What it does

`POST /agent {"request": "..."}` →

1. **Plan** — the agent decides the best document type (proposal, project plan, report,
   SOP, technical design, spec, meeting minutes), builds a section outline, and records
   assumptions when the request is ambiguous.
2. **Execute** — drafts each section in turn.
3. **Reflect** — a self-check pass reviews the draft against the original request and
   revises weak/missing sections.
4. **Assemble** — renders a structured `.docx` (title, metadata table, headings, bullets,
   assumptions + self-check appendix).
5. **Return** — JSON with the generated plan, per-step status, summary, and a download URL.

## The mandatory engineering improvement: Multi-step planning + reflection

- **What:** the agent generates an explicit, ordered TODO list (`interpret → outline →
  draft each section → reflect → assemble`), executes it step by step, then runs a
  **reflection/self-check** pass that revises the draft.
- **Why:** business documents fail when a section is missing, thin, or ignores an
  ambiguous requirement. A single-shot generation can't catch that; an explicit plan plus
  a review pass makes the agent's reasoning inspectable and self-correcting.
- **How it improves the agent:** every run returns the plan and the revisions applied, so
  output quality is higher and the autonomous decision-making is transparent and debuggable.

> Robustness bonus: LLM calls are wrapped in **retry + automatic fallback** to a
> deterministic offline provider, so a run always completes even if Groq is unavailable.

## Extra feature: web UI + `/download/{id}`

A dependency-free single-page UI (`static/index.html`) submits a request, renders the
agent's live TODO list and self-check notes, and downloads the `.docx` — making the
deployed Render URL a self-contained demo.

---

## Run locally

```bash
# from the ruku/ folder, using the medimg conda env
conda run -n medimg pip install -r requirements.txt
conda run -n medimg uvicorn app.main:app --reload
# open http://localhost:8000
```

No API key is required — the agent runs on its offline provider by default.
To use the real LLM, copy `.env.example` → `.env` and set a free **Groq** key
(https://console.groq.com). Provider is auto-selected based on the key's presence.

### API examples

```bash
curl -X POST localhost:8000/agent -H "Content-Type: application/json" \
  -d '{"request":"Write a business proposal for a café loyalty app"}'

curl -OJ localhost:8000/download/<document_id>   # fetch the .docx
curl localhost:8000/health                        # {"status":"ok","provider":"..."}
```

## Tests

```bash
conda run -n medimg python -m pytest -q      # fully offline, no key needed
```

---

## Two test inputs

- **Standard:** *"Write a business proposal for a mobile app that helps small cafés run
  loyalty programs."* → the agent picks `proposal` and drafts 8 sections.
- **Complex / ambiguous:** *"We have a client meeting next week — maybe put together a
  project plan or a short report. Budget is tight and the timeline isn't set. Decide what
  makes sense and draft it."* → the agent chooses a document type, **states its
  assumptions**, and self-plans the rest.

## Deploy to Render

1. Push this folder to a Git repo.
2. Create a **Web Service** from the repo — `render.yaml` is auto-detected.
3. Set `GROQ_API_KEY` in the Render dashboard (optional; without it the app still runs
   offline). Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

---

## Architecture

```
app/
  main.py       FastAPI routes + static UI  (POST /agent, /download/{id}, /jobs/{id}, /health)
  agent.py      Orchestrator: plan → execute → reflect → assemble
  planner.py    Builds the explicit TODO list (PlanStep list)
  llm.py        GroqProvider + OfflineProvider + ResilientLLM (retry & fallback)
  document.py   python-docx renderer
  models.py     Pydantic schemas + request guardrails
  store.py      In-memory job store
  config.py     Env-driven settings + provider auto-select
static/index.html   Web UI (extra feature)
tests/test_agent.py End-to-end offline tests
```

### Debugging insight
Groq's chat completions occasionally wrap JSON in prose or code fences. The tolerant
`_loads()` helper in `llm.py` strips fences and extracts the outermost `{...}` before
parsing — fixing intermittent `JSONDecodeError`s without disabling JSON mode.

### Tradeoff: Autonomous planning vs. deterministic workflow
The agent plans its own outline (flexible, handles novel/ambiguous requests) but every
step is bounded and observable, and falls back to deterministic templates. This trades a
little raw autonomy for **predictability, testability, and a demo that never breaks** —
the right call for a 60-minute build that must run reliably on a free host.
