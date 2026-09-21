# prosthetics-agents

Multi-agent systems for prosthetics & orthotics, built as three stages of increasing
complexity — each on a different agent framework, each closer to the hardware:

| Stage | Domain | Framework | What it demonstrates | Status |
|---|---|---|---|---|
| **B** | Clinical fitting: referral → device recommendation → safety gate → documentation | **CrewAI** | role-based agents, shared state, deterministic safety gate, bounded revision loop, human escalation | MVP runs (mock + real LLMs) |
| **C** | Fabrication: 3D scan → parametric CAD → scheduling → QA | **LangGraph** | explicit state graph with a QA→CAD feedback edge, tool-use over files | planned |
| **A** | Real-time prosthesis control (simulated EMG/IMU) | **AutoGen** | fast control loop with a safety supervisor on the command path, slow adaptation loop, a classical ML model as a tool | planned |

Plan and comparison of the three directions: [docs/PLAN.md](docs/PLAN.md).
Block diagrams (overview + one per stage): [docs/agent-blueprints.html](docs/agent-blueprints.html).
External services, keys, and what is free: [CONNECTIONS.md](CONNECTIONS.md).

> Research prototype. Not a certified clinical tool. Every output is meant to be
> reviewed by a certified prosthetist/orthotist and the treating physician.

## Stage B — clinical fitting crew

```
referral text
   │
   ▼
Intake ──► Biomech ──► Recommendation ──► Safety / Review ──► Documentation
                            ▲                 │
                            └── revise ───────┤   (bounded: max_revisions)
                                              └── escalate ──► clinical team
```

Five CrewAI agents, each owning one section of a shared `PatientRecord`. The
orchestrator holds the record and routes; agents are stateless workers. The
Safety agent is the only path to Documentation: it verifies component weight
ratings and K-level ranges **deterministically** (a tool, not the model's
arithmetic), sends the recommendation back with concrete required changes,
or escalates to a human when red flags are present.

Every hand-off, verdict, revision request, escalation and LLM call is written
to `runs/<run_id>/events.jsonl` as an `AgentMessage` — the same envelope all
three stages will use, so one visualiser serves all of them.

### Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[crewai,dev]"
.venv/Scripts/python -m prosthetics_agents.stage_b.run --all        # mock: no key, no network
.venv/Scripts/python -m pytest -q
```

Copy `.env.example` to `.env` to run on real models. The model is chosen **per role**
(`LLM_INTAKE`, `LLM_SAFETY`, …), so cheap models do structuring and templating while
the strongest model sits on the safety gate; providers can be mixed freely
(Anthropic, OpenAI, Gemini, OpenRouter, Ollama, any OpenAI-compatible URL).

### Cases

Five synthetic referrals in `data/cases/`, chosen to exercise every path:

| Case | Path exercised |
|---|---|
| `case_01_transtibial_k3` | happy path, approved first pass |
| `case_02_transfemoral_k2_vascular` | short limb, fragile skin, low dexterity → stance-control knee, pin-lock + auxiliary belt |
| `case_03_afo_dropfoot` | orthotic branch → dynamic carbon AFO |
| `case_04_redflag_infection` | unhealed wound, uncontrolled diabetes → **escalate** to clinical team |
| `case_05_transfemoral_k3_heavy` | 138 kg user, standard MPK/foot over rating → **revise** → heavy-duty components → approved |

### Layout

```
prosthetics_agents/
  core/          Stage 0: contracts (AgentMessage, PatientRecord), JSONL event log,
                 per-role LLM registry, deterministic MockLLM
  stage_b/       CrewAI agents & tasks, catalog, clinical rules, tools, orchestrator, CLI
data/catalog/    simplified generic component catalog (K-levels, weight ratings)
data/cases/      synthetic referrals with expected outcomes (eval ground truth)
runs/            per-run artifacts (gitignored)
docs/            plan, diagrams
```

### Why an orchestrator on top of CrewAI

CrewAI's sequential process is linear. The safety gate needs a conditional
back-edge (revise) and an exit to a human (escalate), so the loop lives in
`stage_b/pipeline.py` and each task runs as its own single-agent crew against
a fresh snapshot of the record. That is exactly the shape LangGraph expresses
natively — which is why Stage C moves to it.

## Roadmap

- [x] Stage 0: contracts, event log, per-role LLM registry, mock mode
- [x] Stage B MVP: five agents, gate loop, escalation, five cases, tests
- [ ] Stage B: first real-LLM runs, eval harness (crew vs single prompt) on the golden cases
- [ ] Stage B: HTML timeline / communication graph from `events.jsonl`
- [ ] Stage C on LangGraph (scan → CAD → schedule → QA)
- [ ] Stage A on AutoGen (simulated control loop + ML intent classifier service)
