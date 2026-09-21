"""End-to-end through CrewAI with the mock LLM — control flow, not model quality."""

import json
from pathlib import Path

import pytest

from prosthetics_agents.core.contracts import MessageKind
from prosthetics_agents.stage_b import pipeline

CASES_DIR = Path(__file__).resolve().parents[1] / "data" / "cases"


def _referral(stem_part: str) -> tuple[str, str]:
    p = next(CASES_DIR.glob(f"*{stem_part}*.json"))
    case = json.loads(p.read_text(encoding="utf-8"))
    return case["case_id"], case["referral_text"]


@pytest.fixture
def runs_dir(tmp_path):
    return tmp_path / "runs"


def _kinds(log):
    return [m.kind for m in log.messages]


def test_happy_path_single_attempt(runs_dir):
    cid, text = _referral("transtibial_k3")
    record, log = pipeline.run_case(text, cid, runs_dir=runs_dir)
    assert record.safety.verdict == "approved"
    assert len(record.history) == 1
    assert record.document is not None and "prototype" in record.document.disclaimer.lower()
    assert MessageKind.REVISION_REQUEST not in _kinds(log)
    assert (runs_dir / record.run_id / "events.jsonl").exists()
    assert (runs_dir / record.run_id / "record.json").exists()


def test_revise_loop_runs_twice_then_approves(runs_dir):
    cid, text = _referral("heavy")
    record, log = pipeline.run_case(text, cid, runs_dir=runs_dir)
    assert record.safety.verdict == "approved"
    assert len(record.history) == 2
    assert record.history[0].safety.verdict == "revise"
    kinds = _kinds(log)
    assert kinds.count(MessageKind.REVISION_REQUEST) == 1
    # documentation only after the gate said yes
    verdict_idx = max(i for i, k in enumerate(kinds) if k == MessageKind.VERDICT)
    doc_idx = next(i for i, m in enumerate(log.messages) if m.sender == "Documentation Agent")
    assert doc_idx > verdict_idx


def test_red_flags_escalate_to_human(runs_dir):
    cid, text = _referral("redflag")
    record, log = pipeline.run_case(text, cid, runs_dir=runs_dir)
    assert record.safety.verdict == "escalate"
    esc = [m for m in log.messages if m.kind == MessageKind.ESCALATION]
    assert esc and esc[0].recipient == pipeline.HUMAN
    assert record.document.escalation_note.startswith("ESCALATED")


def test_revision_budget_exhaustion_escalates(runs_dir, monkeypatch):
    """If the recommender never fixes the issue, the gate must not loop forever."""
    from prosthetics_agents.stage_b import rules

    original = rules.recommend
    monkeypatch.setattr(rules, "recommend", lambda i, b, feedback=None: original(i, b, None))
    cid, text = _referral("heavy")
    record, log = pipeline.run_case(text, cid, max_revisions=1, runs_dir=runs_dir)
    assert record.safety.verdict == "escalate"
    assert record.safety.escalate_to_human
    assert len(record.history) == 2
    assert _kinds(log)[-2] == MessageKind.ESCALATION or MessageKind.ESCALATION in _kinds(log)
