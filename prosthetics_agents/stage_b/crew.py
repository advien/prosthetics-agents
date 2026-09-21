"""Stage B agents and task builders (CrewAI).

Five agents, five roles, each owning one section of the ``PatientRecord``.
Tasks are built per step from a fresh record snapshot, so the orchestrator
(``pipeline.py``) stays the single owner of state.
"""

from __future__ import annotations

import json
from typing import Any

from crewai import Agent, Task
from pydantic import BaseModel

from prosthetics_agents.core.contracts import (
    BiomechAssessment,
    ClinicalDocument,
    DeviceRecommendation,
    PatientIntake,
    PatientRecord,
    SafetyReview,
)
from prosthetics_agents.core.llm_registry import llm_for, spec_for
from prosthetics_agents.core.mock_llm import (
    FEEDBACK_CLOSE,
    FEEDBACK_OPEN,
    RECORD_CLOSE,
    RECORD_OPEN,
    MockLLM,
    extract_feedback,
    extract_record,
)

from . import rules
from .tools import catalog_search, verify_component_ratings

ROLES = ["intake", "biomech", "recommendation", "safety", "documentation"]

DISCLAIMER_LINE = (
    "This is a research prototype, not a certified clinical tool; "
    "output is reviewed by a certified prosthetist/orthotist."
)


# ── Agents ───────────────────────────────────────────────────────────────────


def _agent(role_key: str, role: str, goal: str, backstory: str, tools: list | None = None) -> Agent:
    llm = llm_for(role_key)
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory + " " + DISCLAIMER_LINE,
        llm=llm,
        # tools need the ReAct loop; the mock answers in one shot, so skip them there
        tools=[] if spec_for(role_key).is_mock else (tools or []),
        verbose=False,
        allow_delegation=False,
        max_iter=6,
    )


def build_agents() -> dict[str, Agent]:
    return {
        "intake": _agent(
            "intake",
            "Intake Agent",
            "Turn a free-text referral into a complete, structured patient intake record.",
            "Clinical intake coordinator at a prosthetics & orthotics clinic. Extracts facts "
            "exactly as stated, never invents missing data, flags every safety-relevant "
            "statement (wounds, infection, uncontrolled comorbidities, cognition) as a red flag.",
        ),
        "biomech": _agent(
            "biomech",
            "Biomechanical Analysis Agent",
            "Assess functional level (K0-K4), load class and gait/suspension considerations.",
            "Rehabilitation biomechanist. Confirms or adjusts the reported K-level from evidence, "
            "reasons about lever arms, load, skin and dexterity, and lists contraindications.",
        ),
        "recommendation": _agent(
            "recommendation",
            "Recommendation Agent",
            "Select a device type, socket, suspension and catalog components that best serve "
            "the patient's function and goals.",
            "Senior prosthetist/orthotist. Uses the component catalog, explains every choice, "
            "and when the safety gate sends the work back, applies the required changes.",
            tools=[catalog_search],
        ),
        "safety": _agent(
            "safety",
            "Safety and Review Agent",
            "Block unsafe or non-compliant recommendations before they reach documentation.",
            "Clinical governance reviewer. Verifies component ratings deterministically, checks "
            "red flags and contraindications, and escalates to the clinical team when in doubt. "
            "Never approves a recommendation with an unresolved critical issue.",
            tools=[verify_component_ratings],
        ),
        "documentation": _agent(
            "documentation",
            "Documentation Agent",
            "Write the clinical justification and device specification for the record and insurer.",
            "Clinical documentation specialist. Writes precise, traceable justifications and "
            "always includes the escalation status and the research-prototype disclaimer.",
        ),
    }


# ── Tasks ────────────────────────────────────────────────────────────────────


def _record_block(record: PatientRecord) -> str:
    return f"{RECORD_OPEN}\n{json.dumps(record.snapshot(), ensure_ascii=False, indent=1)}\n{RECORD_CLOSE}"


def _task(agent: Agent, description: str, model: type[BaseModel]) -> Task:
    return Task(
        description=description,
        expected_output=f"A single JSON object matching the {model.__name__} schema. No prose.",
        agent=agent,
        output_pydantic=model,
    )


def intake_task(agent: Agent, record: PatientRecord) -> Task:
    return _task(
        agent,
        "Structure the following referral into a PatientIntake. Copy numbers exactly. "
        "Put wounds, infection, poorly controlled diabetes, cognitive issues or cardiac "
        "limitations into red_flags. Estimate reported_k_level from described activity "
        "(K1 household, K2 limited community, K3 unlimited community / variable cadence, "
        "K4 high-impact).\n\nReferral:\n" + record.referral_text + "\n\n" + _record_block(record),
        PatientIntake,
    )


def biomech_task(agent: Agent, record: PatientRecord) -> Task:
    return _task(
        agent,
        "Using the intake section of the record below, produce a BiomechAssessment: confirm "
        "or adjust the K-level with a rationale, classify load (light <75 kg, standard "
        "75-125 kg, heavy >125 kg), and list gait, suspension and contraindication "
        "considerations.\n\n" + _record_block(record),
        BiomechAssessment,
    )


def recommendation_task(agent: Agent, record: PatientRecord, feedback: list[str]) -> Task:
    fb = ""
    if feedback:
        fb = (
            "\n\nThe Safety and Review Agent rejected the previous recommendation. Apply every "
            f"required change:\n{FEEDBACK_OPEN}\n" + "\n".join(f"- {f}" for f in feedback) + f"\n{FEEDBACK_CLOSE}"
        )
    return _task(
        agent,
        "Using intake and biomech sections of the record below, produce a DeviceRecommendation. "
        "Choose device_type, socket, suspension and components from the catalog (use the "
        "catalog_search tool per category with the patient's K-level and weight). Use catalog "
        "ids exactly. Give a rationale per component and an overall confidence 0-1."
        + fb + "\n\n" + _record_block(record),
        DeviceRecommendation,
    )


def safety_task(agent: Agent, record: PatientRecord) -> Task:
    return _task(
        agent,
        "Review the recommendation in the record below. Run verify_component_ratings on all "
        "component ids with the patient's weight and confirmed K-level. Any red flag in intake "
        "=> verdict 'escalate' with escalate_to_human true. Any critical component/contraindication "
        "issue => verdict 'revise' with concrete required_changes. Otherwise 'approved'.\n\n"
        + _record_block(record),
        SafetyReview,
    )


def documentation_task(agent: Agent, record: PatientRecord) -> Task:
    return _task(
        agent,
        "Write the ClinicalDocument for the record below: summary, insurer-style justification "
        "tied to K-level and biomechanics, device specification listing every component with "
        "catalog id, follow-up plan. If safety verdict is 'escalate', fill escalation_note with "
        "the critical issues and state that no fitting proceeds. Always include the disclaimer.\n\n"
        + _record_block(record),
        ClinicalDocument,
    )


# ── Mock handlers (offline mode) ─────────────────────────────────────────────


def _rec_from(text: str) -> PatientRecord:
    data = extract_record(text)
    return PatientRecord.model_validate(data)


def _h_intake(text: str, _model: Any) -> BaseModel:
    rec = _rec_from(text)
    return rules.mock_intake(rec.referral_text)


def _h_biomech(text: str, _model: Any) -> BaseModel:
    rec = _rec_from(text)
    return rules.assess_biomech(rec.intake)


def _h_recommend(text: str, _model: Any) -> BaseModel:
    rec = _rec_from(text)
    return rules.recommend(rec.intake, rec.biomech, extract_feedback(text))


def _h_safety(text: str, _model: Any) -> BaseModel:
    rec = _rec_from(text)
    return rules.review_safety(rec.intake, rec.biomech, rec.recommendation)


def _h_docs(text: str, _model: Any) -> BaseModel:
    return rules.write_document(_rec_from(text))


MockLLM.register("intake", _h_intake)
MockLLM.register("biomech", _h_biomech)
MockLLM.register("recommendation", _h_recommend)
MockLLM.register("safety", _h_safety)
MockLLM.register("documentation", _h_docs)
