"""Message and state contracts shared by every stage.

Two ideas live here:

* ``AgentMessage`` — the envelope every inter-agent hand-off is logged as.
  Stage B (CrewAI), C (LangGraph) and A (AutoGen) all emit the same envelope,
  so one visualiser reads all three.
* ``PatientRecord`` — the shared state object for Stage B. Each agent owns one
  section and only writes there; the orchestrator holds the whole record.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Inter-agent envelope ─────────────────────────────────────────────────────


class MessageKind(StrEnum):
    HANDOFF = "handoff"  # agent finished, passes its section on
    VERDICT = "verdict"  # safety gate decision
    REVISION_REQUEST = "revision_request"  # gate sends work back
    ESCALATION = "escalation"  # human-in-the-loop required
    LLM_CALL = "llm_call"  # model invocation (for cost/latency traces)
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"


class AgentMessage(BaseModel):
    msg_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sender: str
    recipient: str
    kind: MessageKind
    attempt: int = 1
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


# ── Stage B: patient record sections ─────────────────────────────────────────

KLevel = Literal["K0", "K1", "K2", "K3", "K4"]
AmputationLevel = Literal["transtibial", "transfemoral", "partial_foot", "none"]
DeviceType = Literal[
    "transtibial_prosthesis",
    "transfemoral_prosthesis",
    "partial_foot_prosthesis",
    "afo",
    "kafo",
]


class ResidualLimb(BaseModel):
    length: Literal["short", "medium", "long"] | None = None
    condition: str = ""
    skin_issues: list[str] = Field(default_factory=list)


class PatientIntake(BaseModel):
    """Owner: Intake Agent. Structured from free-text referral."""

    patient_id: str
    age: int
    sex: Literal["F", "M", "other"] | None = None
    weight_kg: float
    height_cm: float | None = None
    amputation_level: AmputationLevel
    side: Literal["left", "right", "bilateral"] | None = None
    etiology: str = ""
    time_since_amputation_months: int | None = None
    reported_k_level: KLevel | None = None
    residual_limb: ResidualLimb = Field(default_factory=ResidualLimb)
    comorbidities: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    lifestyle: str = ""
    prior_device: str = ""
    orthotic_indication: str = ""  # for AFO/KAFO cases: e.g. "drop foot, no spasticity"
    red_flags: list[str] = Field(default_factory=list)


class BiomechAssessment(BaseModel):
    """Owner: Biomechanical Analysis Agent."""

    confirmed_k_level: KLevel
    k_level_rationale: str
    load_class: Literal["light", "standard", "heavy"]
    gait_considerations: list[str] = Field(default_factory=list)
    suspension_considerations: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)


class Component(BaseModel):
    category: str
    catalog_id: str
    name: str
    rationale: str


class DeviceRecommendation(BaseModel):
    """Owner: Recommendation Agent."""

    device_type: DeviceType
    socket_design: str = ""
    suspension: str = ""
    interface: str = ""
    components: list[Component] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class SafetyIssue(BaseModel):
    severity: Literal["info", "warning", "critical"]
    message: str


class SafetyReview(BaseModel):
    """Owner: Safety / Review Agent — the gate before documentation."""

    verdict: Literal["approved", "revise", "escalate"]
    issues: list[SafetyIssue] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)
    escalate_to_human: bool = False
    rationale: str


class ClinicalDocument(BaseModel):
    """Owner: Documentation Agent."""

    summary: str
    justification: str
    device_specification: str
    follow_up_plan: str
    escalation_note: str = ""
    disclaimer: str


class RevisionEntry(BaseModel):
    attempt: int
    recommendation: DeviceRecommendation
    safety: SafetyReview


class PatientRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    referral_text: str
    intake: PatientIntake | None = None
    biomech: BiomechAssessment | None = None
    recommendation: DeviceRecommendation | None = None
    safety: SafetyReview | None = None
    document: ClinicalDocument | None = None
    history: list[RevisionEntry] = Field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        """Serialisable view handed to agents (never the history — keeps prompts short)."""
        return self.model_dump(mode="json", exclude={"history"}, exclude_none=True)
