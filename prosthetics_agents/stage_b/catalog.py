"""Component catalog access. Deterministic — no LLM involved."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "catalog" / "components.json"

K_RANK = {"K0": 0, "K1": 1, "K2": 2, "K3": 3, "K4": 4}


@lru_cache(maxsize=1)
def load_catalog() -> list[dict[str, Any]]:
    with CATALOG_PATH.open(encoding="utf-8") as f:
        return json.load(f)["components"]


def by_id(catalog_id: str) -> dict[str, Any] | None:
    return next((c for c in load_catalog() if c["id"] == catalog_id), None)


def fits_k(component: dict[str, Any], k_level: str) -> bool:
    return K_RANK[component["k_min"]] <= K_RANK[k_level] <= K_RANK[component["k_max"]]


def fits_weight(component: dict[str, Any], weight_kg: float) -> bool:
    return weight_kg <= component["weight_limit_kg"]


def candidates(
    category: str,
    k_level: str,
    weight_kg: float | None = None,
    tag: str | None = None,
) -> list[dict[str, Any]]:
    """Components of a category valid for the K-level (and weight / tag if given)."""
    out = []
    for c in load_catalog():
        if c["category"] != category or not fits_k(c, k_level):
            continue
        if weight_kg is not None and not fits_weight(c, weight_kg):
            continue
        if tag is not None and tag not in c["tags"]:
            continue
        out.append(c)
    return out
