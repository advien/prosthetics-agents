"""JSONL event log of inter-agent communication.

One file per run: ``runs/<run_id>/events.jsonl``. Every line is an
``AgentMessage``. This is the raw material for the communication-graph
visualisation — the thing that actually *shows* multi-agent behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import AgentMessage, MessageKind

RUNS_DIR = Path("runs")


class EventLog:
    def __init__(self, run_id: str, runs_dir: Path = RUNS_DIR) -> None:
        self.run_id = run_id
        self.dir = runs_dir / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._messages: list[AgentMessage] = []

    def emit(
        self,
        sender: str,
        recipient: str,
        kind: MessageKind,
        summary: str = "",
        payload: dict[str, Any] | None = None,
        attempt: int = 1,
    ) -> AgentMessage:
        msg = AgentMessage(
            run_id=self.run_id,
            sender=sender,
            recipient=recipient,
            kind=kind,
            summary=summary,
            payload=payload or {},
            attempt=attempt,
        )
        self._messages.append(msg)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg.model_dump_json() + "\n")
        return msg

    @property
    def messages(self) -> list[AgentMessage]:
        return list(self._messages)

    def save_json(self, name: str, data: Any) -> Path:
        """Store an arbitrary artifact (final record, document) next to the log."""
        path = self.dir / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path) -> list[AgentMessage]:
        with path.open(encoding="utf-8") as f:
            return [AgentMessage.model_validate_json(line) for line in f if line.strip()]
