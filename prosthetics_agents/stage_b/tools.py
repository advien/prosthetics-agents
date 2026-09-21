"""Deterministic tools exposed to the LLM agents (CrewAI ``@tool``)."""

from __future__ import annotations

import json

from crewai.tools import tool

from prosthetics_agents.core.contracts import Component, DeviceRecommendation

from . import catalog, rules


@tool("catalog_search")
def catalog_search(category: str, k_level: str, weight_kg: float) -> str:
    """Search the component catalog. category: foot|knee|liner|suspension|socket|afo|kafo.
    k_level: K0..K4. weight_kg: patient weight. Returns JSON list of components valid
    for that K-level and weight rating, with ids, names, tags and notes."""
    return json.dumps(catalog.candidates(category, k_level, weight_kg), ensure_ascii=False)


@tool("verify_component_ratings")
def verify_component_ratings(component_ids: list[str], weight_kg: float, k_level: str) -> str:
    """Hard-check a list of catalog component ids against patient weight and K-level.
    Returns JSON list of issues (severity, message). Empty list means all ratings pass."""
    rec = DeviceRecommendation(
        device_type="transtibial_prosthesis",
        components=[Component(category="?", catalog_id=cid, name=cid, rationale="") for cid in component_ids],
        confidence=1.0,
        rationale="verification",
    )
    issues = rules.verify_components(rec, weight_kg, k_level)
    return json.dumps([i.model_dump() for i in issues], ensure_ascii=False)
