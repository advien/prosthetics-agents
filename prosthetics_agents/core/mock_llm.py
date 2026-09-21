"""Deterministic stand-in for an LLM.

Lets the whole multi-agent pipeline run with no API key, in CI, and in tests.
Each stage registers a handler per agent role; the handler receives the
prompt text and the expected response model and returns an instance of it.

The mock never "reasons" — it delegates to explicit rules. That is the point:
the pipeline's control flow (hand-offs, gate, revision loop, escalation) is
tested independently of model quality.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, ClassVar

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel

Handler = Callable[[str, type[BaseModel] | None], BaseModel | str]

RECORD_OPEN, RECORD_CLOSE = "<<RECORD>>", "<</RECORD>>"
FEEDBACK_OPEN, FEEDBACK_CLOSE = "<<FEEDBACK>>", "<</FEEDBACK>>"


def extract_block(text: str, open_tag: str, close_tag: str) -> str | None:
    m = re.search(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), text, re.S)
    return m.group(1).strip() if m else None


def extract_record(text: str) -> dict[str, Any]:
    block = extract_block(text, RECORD_OPEN, RECORD_CLOSE)
    return json.loads(block) if block else {}


def extract_feedback(text: str) -> list[str]:
    block = extract_block(text, FEEDBACK_OPEN, FEEDBACK_CLOSE)
    return [line.strip("- ").strip() for line in block.splitlines() if line.strip()] if block else []


class MockLLM(BaseLLM):
    llm_type: str = "mock"
    _handlers: ClassVar[dict[str, Handler]] = {}
    calls: int = 0

    @classmethod
    def register(cls, role_keyword: str, handler: Handler) -> None:
        cls._handlers[role_keyword.lower()] = handler

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model: type[BaseModel] | None = None,
    ):
        self.calls += 1
        role = (getattr(from_agent, "role", "") or "").lower()
        text = messages if isinstance(messages, str) else "\n".join(
            str(m.get("content", "")) for m in messages if isinstance(m, dict)
        )
        handler = next((h for kw, h in self._handlers.items() if kw in role), None)
        if handler is None:
            raise RuntimeError(f"MockLLM: no handler registered for agent role '{role}'")
        result = handler(text, response_model)
        return result.model_dump_json() if isinstance(result, BaseModel) else str(result)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 32_000
