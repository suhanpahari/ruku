"""Turn a plan dict into the agent's explicit TODO list (multi-step planning)."""
from typing import List

from .models import PlanStep


def build_plan_steps(plan: dict) -> List[PlanStep]:
    """Materialise an ordered TODO list from a plan.

    Order mirrors execution: interpret -> outline -> draft each section ->
    reflect (self-check) -> assemble the final .docx.
    """
    sections = plan.get("sections", [])
    steps: List[PlanStep] = [
        PlanStep(
            id="interpret",
            title=f"Interpret request → {plan['document_type']}",
            kind="interpret",
        ),
        PlanStep(
            id="outline",
            title=f"Build outline ({len(sections)} sections)",
            kind="outline",
        ),
    ]
    for i, sec in enumerate(sections):
        steps.append(
            PlanStep(
                id=f"draft_{i}",
                title=f"Draft section: {sec['name']}",
                kind="draft",
                detail=sec.get("purpose") or None,
            )
        )
    steps.append(PlanStep(id="reflect", title="Self-check & revise (reflection)", kind="reflect"))
    steps.append(PlanStep(id="assemble", title="Assemble .docx document", kind="assemble"))
    return steps


def set_status(steps: List[PlanStep], step_id: str, status: str, detail: str = None) -> None:
    for step in steps:
        if step.id == step_id:
            step.status = status
            if detail is not None:
                step.detail = detail
            return
