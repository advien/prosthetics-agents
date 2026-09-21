"""Per-role LLM selection, driven entirely by environment variables.

The point: which model serves which agent is an engineering decision, not a
hard-coded constant. Cheap models for structuring/templating, stronger models
where reasoning matters, the strongest on the safety gate.

Resolution order for a role ``SAFETY``:

    LLM_SAFETY  ->  LLM_DEFAULT  ->  "mock"

Two layers of configuration, kept apart on purpose:

* ``.env``               — secrets (API keys). Never committed.
* ``profiles/<name>.env`` — routing only: which model serves which role.
                            Committed; switching provider = switching profile:
                            ``--profile ollama`` or ``LLM_PROFILE=ollama``.

Model strings use CrewAI's ``provider/model`` form. Every provider CrewAI
supports natively works here; the ones we care about (see CONNECTIONS.md):

    mock                                   no network, deterministic rules
    anthropic/claude-sonnet-5              ANTHROPIC_API_KEY
    openai/gpt-...                         OPENAI_API_KEY
    gemini/gemini-2.5-flash                GEMINI_API_KEY   (free tier)
    openrouter/<org>/<model>               OPENROUTER_API_KEY (free models exist)
    ollama/<model>                         local, no key
    hosted_vllm/<model>                    any OpenAI-compatible URL — Groq, Cloudflare
                                           Workers AI, your own VPS. Needs
                                           LLM_<ROLE>_BASE_URL (or VLLM_BASE_URL)
                                           and LLM_<ROLE>_API_KEY (or VLLM_API_KEY).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

MOCK = "mock"
PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"

load_dotenv()  # secrets


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.env"))


def load_profile(name: str | None = None) -> str:
    """Apply a routing profile on top of the environment. Returns the profile name.

    Profile values override anything already set, so ``--profile`` wins over a
    stale ``LLM_DEFAULT`` in ``.env``. With no name and no ``LLM_PROFILE``, the
    environment is left as is (which resolves to mock unless ``.env`` says otherwise).
    """
    name = name or os.getenv("LLM_PROFILE")
    if not name:
        return "env"
    path = PROFILES_DIR / f"{name}.env"
    if not path.exists():
        raise FileNotFoundError(
            f"profile '{name}' not found; available: {', '.join(available_profiles())}"
        )
    # a profile is authoritative for routing: drop routing vars, keep *_API_KEY secrets
    for key in list(os.environ):
        if key.startswith("LLM_") and key != "LLM_PROFILE" and not key.endswith("_API_KEY"):
            del os.environ[key]
    load_dotenv(path, override=True)
    llm_for.cache_clear()
    return name


@dataclass(frozen=True)
class LLMSpec:
    role: str
    model: str
    base_url: str | None = None
    api_key: str | None = None

    @property
    def is_mock(self) -> bool:
        return self.model == MOCK

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0] if "/" in self.model else self.model


def spec_for(role: str) -> LLMSpec:
    key = role.upper()
    model = os.getenv(f"LLM_{key}") or os.getenv("LLM_DEFAULT") or MOCK
    return LLMSpec(
        role=role,
        model=model.strip(),
        base_url=os.getenv(f"LLM_{key}_BASE_URL") or os.getenv("LLM_BASE_URL"),
        api_key=os.getenv(f"LLM_{key}_API_KEY") or os.getenv("LLM_API_KEY"),
    )


@lru_cache(maxsize=None)
def llm_for(role: str):
    """Return a CrewAI-compatible LLM object for the role (cached per role)."""
    spec = spec_for(role)
    if spec.is_mock:
        from .mock_llm import MockLLM

        return MockLLM(model="mock/rules")

    from crewai import LLM

    kwargs: dict = {"model": spec.model}
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
    if spec.api_key:
        kwargs["api_key"] = spec.api_key
    return LLM(**kwargs)


def describe_roles(roles: list[str]) -> dict[str, str]:
    """Role -> model string, for the run header and the event log."""
    return {r: spec_for(r).model for r in roles}
