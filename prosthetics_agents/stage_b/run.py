"""CLI: run one case (or all) through the Stage B crew.

    python -m prosthetics_agents.stage_b.run --case case_01_transtibial_k3
    python -m prosthetics_agents.stage_b.run --all
    python -m prosthetics_agents.stage_b.run --referral "Patient P-9, 50-year-old ..."

With no LLM_* variables set, everything runs on the deterministic mock.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prosthetics_agents.core.contracts import MessageKind, PatientRecord
from prosthetics_agents.core.eventlog import EventLog
from prosthetics_agents.core.llm_registry import available_profiles, describe_roles, load_profile

CASES_DIR = Path(__file__).resolve().parents[2] / "data" / "cases"
console = Console()

KIND_STYLE = {
    MessageKind.HANDOFF: "cyan",
    MessageKind.VERDICT: "green",
    MessageKind.REVISION_REQUEST: "yellow",
    MessageKind.ESCALATION: "bold red",
    MessageKind.LLM_CALL: "dim",
    MessageKind.RUN_STARTED: "dim",
    MessageKind.RUN_FINISHED: "bold",
}


def render(record: PatientRecord, log: EventLog, show_llm: bool = False) -> None:
    table = Table(title=f"Inter-agent messages — run {log.run_id}", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("from")
    table.add_column("→ to")
    table.add_column("kind")
    table.add_column("att", justify="right")
    table.add_column("summary")
    for i, m in enumerate(log.messages, 1):
        if m.kind == MessageKind.LLM_CALL and not show_llm:
            continue
        table.add_row(str(i), m.sender, m.recipient, f"[{KIND_STYLE[m.kind]}]{m.kind}[/]",
                      str(m.attempt), m.summary)
    console.print(table)

    s = record.safety
    verdict_style = {"approved": "green", "revise": "yellow", "escalate": "red"}[s.verdict]
    body = (
        f"[bold]Verdict:[/] [{verdict_style}]{s.verdict.upper()}[/]  "
        f"(attempts: {len(record.history)})\n[bold]Rationale:[/] {s.rationale}\n"
    )
    if s.issues:
        body += "\n".join(f"  [{'red' if i.severity == 'critical' else 'yellow'}]{i.severity}[/] {i.message}"
                          for i in s.issues) + "\n"
    r = record.recommendation
    body += f"\n[bold]Device:[/] {r.device_type}  socket: {r.socket_design or '-'}  suspension: {r.suspension or '-'}\n"
    body += "\n".join(f"  • {c.category:<11} {c.catalog_id:<14} {c.name}" for c in r.components)
    body += f"\n\n[bold]Document summary:[/] {record.document.summary}"
    if record.document.escalation_note:
        body += f"\n[red]{record.document.escalation_note}[/]"
    console.print(Panel(body, title=f"Result — {record.intake.patient_id}", border_style=verdict_style))
    console.print(f"[dim]artifacts: {log.dir / 'events.jsonl'}, {log.dir / 'record.json'}[/]")


def load_case(case_id: str) -> dict:
    path = CASES_DIR / f"{case_id}.json"
    if not path.exists():
        raise SystemExit(f"case not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--case", help="case id from data/cases (without .json)")
    g.add_argument("--all", action="store_true", help="run every case in data/cases")
    g.add_argument("--referral", help="free-text referral to run ad hoc")
    ap.add_argument("--profile", choices=available_profiles(), default=None,
                    help="routing profile from profiles/ (or set LLM_PROFILE); default: .env / mock")
    ap.add_argument("--max-revisions", type=int, default=2)
    ap.add_argument("--show-llm", action="store_true", help="include LLM call rows in the table")
    args = ap.parse_args()

    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    profile = load_profile(args.profile)

    from . import pipeline  # after env is set

    console.print(Panel(
        "\n".join(f"{k:<15} {v}" for k, v in describe_roles(pipeline.crew_mod.ROLES).items()),
        title=f"LLM per role — profile: {profile}", border_style="dim",
    ))

    if args.referral:
        jobs = [("adhoc", args.referral)]
    elif args.case:
        c = load_case(args.case)
        jobs = [(c["case_id"], c["referral_text"])]
    else:
        jobs = [(c["case_id"], c["referral_text"])
                for c in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(CASES_DIR.glob("*.json")))]

    for case_id, text in jobs:
        console.rule(f"[bold]{case_id}")
        record, log = pipeline.run_case(text, case_id, max_revisions=args.max_revisions)
        render(record, log, show_llm=args.show_llm)


if __name__ == "__main__":
    main()
