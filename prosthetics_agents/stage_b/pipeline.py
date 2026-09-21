"""Stage B orchestrator.

    Intake -> Biomech -> Recommendation -> Safety gate -> Documentation
                             ^                |
                             +--- revise -----+      (bounded loop)
                                              +--> escalate -> Documentation (escalation mode)

The orchestrator owns the ``PatientRecord``; agents are stateless workers that
each fill one section. Every hand-off, verdict and LLM call is written to the
run's event log.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from crewai import Crew, Process, Task
from crewai.events import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent

from prosthetics_agents.core.contracts import MessageKind, PatientRecord, RevisionEntry
from prosthetics_agents.core.eventlog import RUNS_DIR, EventLog
from prosthetics_agents.core.llm_registry import describe_roles

from . import crew as crew_mod

ORCH = "orchestrator"
HUMAN = "clinical_team"

_active_log: EventLog | None = None


@crewai_event_bus.on(LLMCallCompletedEvent)
def _on_llm_call(_source: Any, event: LLMCallCompletedEvent) -> None:
    """Record every model invocation (model, role, tokens) into the active run log."""
    if _active_log is None:
        return
    usage = event.usage or {}
    _active_log.emit(
        sender=getattr(event, "agent_role", None) or "?",
        recipient="llm",
        kind=MessageKind.LLM_CALL,
        summary=event.model or "",
        payload={
            "model": event.model,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    )


def _run_task(task: Task):
    """Run one task in its own single-agent crew and return the pydantic output."""
    out = Crew(agents=[task.agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    if out.pydantic is None:
        raise RuntimeError(f"Task returned no structured output. Raw: {out.raw[:500]}")
    return out.pydantic


def run_case(
    referral_text: str,
    case_id: str = "adhoc",
    max_revisions: int = 2,
    runs_dir: Path = RUNS_DIR,
) -> tuple[PatientRecord, EventLog]:
    global _active_log
    record = PatientRecord(referral_text=referral_text)
    log = EventLog(record.run_id, runs_dir)
    _active_log = log
    agents = crew_mod.build_agents()
    t0 = time.perf_counter()

    log.emit(ORCH, "*", MessageKind.RUN_STARTED, f"case {case_id}",
             {"case_id": case_id, "models": describe_roles(crew_mod.ROLES)})

    # 1. Intake
    record.intake = _run_task(crew_mod.intake_task(agents["intake"], record))
    log.emit("Intake Agent", "Biomechanical Analysis Agent", MessageKind.HANDOFF,
             f"{record.intake.amputation_level}, {record.intake.reported_k_level}, "
             f"{len(record.intake.red_flags)} red flag(s)",
             {"section": "intake", "red_flags": record.intake.red_flags})

    # 2. Biomech
    record.biomech = _run_task(crew_mod.biomech_task(agents["biomech"], record))
    log.emit("Biomechanical Analysis Agent", "Recommendation Agent", MessageKind.HANDOFF,
             f"confirmed {record.biomech.confirmed_k_level}, load {record.biomech.load_class}",
             {"section": "biomech"})

    # 3. Recommendation -> 4. Safety gate, bounded revise loop
    feedback: list[str] = []
    attempt = 0
    while True:
        attempt += 1
        record.recommendation = _run_task(
            crew_mod.recommendation_task(agents["recommendation"], record, feedback)
        )
        log.emit("Recommendation Agent", "Safety and Review Agent", MessageKind.HANDOFF,
                 f"{record.recommendation.device_type}: "
                 + ", ".join(c.catalog_id for c in record.recommendation.components),
                 {"section": "recommendation", "components":
                     [c.catalog_id for c in record.recommendation.components]},
                 attempt=attempt)

        record.safety = _run_task(crew_mod.safety_task(agents["safety"], record))
        record.history.append(
            RevisionEntry(attempt=attempt, recommendation=record.recommendation, safety=record.safety)
        )
        verdict = record.safety.verdict
        critical = [i.message for i in record.safety.issues if i.severity == "critical"]

        if verdict == "approved":
            log.emit("Safety and Review Agent", "Documentation Agent", MessageKind.VERDICT,
                     "approved", {"verdict": verdict}, attempt=attempt)
            break
        if verdict == "escalate":
            log.emit("Safety and Review Agent", HUMAN, MessageKind.ESCALATION,
                     "; ".join(critical) or record.safety.rationale,
                     {"verdict": verdict, "issues": critical}, attempt=attempt)
            log.emit("Safety and Review Agent", "Documentation Agent", MessageKind.VERDICT,
                     "escalate", {"verdict": verdict}, attempt=attempt)
            break
        # revise
        if attempt > max_revisions:
            log.emit("Safety and Review Agent", HUMAN, MessageKind.ESCALATION,
                     f"revision budget exhausted after {attempt} attempts",
                     {"verdict": "escalate", "issues": critical}, attempt=attempt)
            record.safety.verdict = "escalate"
            record.safety.escalate_to_human = True
            record.safety.rationale += f" Revision budget ({max_revisions}) exhausted."
            break
        feedback = list(record.safety.required_changes)
        log.emit("Safety and Review Agent", "Recommendation Agent", MessageKind.REVISION_REQUEST,
                 "; ".join(feedback), {"required_changes": feedback}, attempt=attempt)

    # 5. Documentation — only reachable through the gate
    record.document = _run_task(crew_mod.documentation_task(agents["documentation"], record))
    log.emit("Documentation Agent", ORCH, MessageKind.HANDOFF, "clinical document ready",
             {"section": "document", "escalated": record.safety.verdict == "escalate"})

    elapsed = time.perf_counter() - t0
    log.emit(ORCH, "*", MessageKind.RUN_FINISHED,
             f"{record.safety.verdict} after {attempt} attempt(s), {elapsed:.1f}s",
             {"verdict": record.safety.verdict, "attempts": attempt, "elapsed_s": round(elapsed, 2)})
    log.save_json("record.json", record.model_dump(mode="json"))
    _active_log = None
    return record, log
