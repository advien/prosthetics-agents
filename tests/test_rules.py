import json
from pathlib import Path

import pytest

from prosthetics_agents.stage_b import rules

CASES = sorted((Path(__file__).resolve().parents[1] / "data" / "cases").glob("*.json"))


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_intake_matches_expected(path):
    case = _load(path)
    intake = rules.mock_intake(case["referral_text"])
    assert intake.amputation_level == case["expected"]["amputation_level"]
    assert intake.reported_k_level == case["expected"]["k_level"]
    assert intake.weight_kg > 0 and intake.age > 0


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_first_pass_safety_verdict(path):
    """Verdict of the first recommendation pass (before any revision)."""
    case = _load(path)
    intake = rules.mock_intake(case["referral_text"])
    bio = rules.assess_biomech(intake)
    rec = rules.recommend(intake, bio)
    review = rules.review_safety(intake, bio, rec)
    expected = case["expected"]["verdict"]
    if expected == "escalate":
        assert review.verdict == "escalate" and review.escalate_to_human
    else:
        assert review.verdict in ("approved", "revise")
    assert rec.device_type == case["expected"]["device_type"]


def test_weight_rating_is_caught_then_fixed():
    case = _load(next(p for p in CASES if "heavy" in p.stem))
    intake = rules.mock_intake(case["referral_text"])
    bio = rules.assess_biomech(intake)
    rec1 = rules.recommend(intake, bio)
    review1 = rules.review_safety(intake, bio, rec1)
    assert review1.verdict == "revise"
    assert any("exceeds rating" in i.message for i in review1.issues)

    rec2 = rules.recommend(intake, bio, feedback=review1.required_changes)
    review2 = rules.review_safety(intake, bio, rec2)
    assert review2.verdict == "approved"
    assert {c.catalog_id for c in rec2.components} >= {"KN-MPK-HD", "FT-ESAR-HD"}


def test_verify_components_flags_over_prescription():
    from prosthetics_agents.core.contracts import Component, DeviceRecommendation

    rec = DeviceRecommendation(
        device_type="transfemoral_prosthesis",
        components=[Component(category="knee", catalog_id="KN-MPK", name="MPK", rationale="")],
        confidence=0.9,
        rationale="",
    )
    issues = rules.verify_components(rec, weight_kg=80, k_level="K1")
    assert any(i.severity == "critical" and "over-prescription" in i.message for i in issues)
