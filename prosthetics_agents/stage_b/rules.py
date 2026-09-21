"""Deterministic clinical heuristics for Stage B.

Used in two places:

1. ``MockLLM`` — so the whole crew runs offline with no API key, exercising
   every path (approve / revise / escalate) with reproducible output.
2. Tools and the deterministic verifier that real LLM agents call — the
   Safety gate does not *trust* the model's arithmetic on weight ratings or
   K-level ranges; it checks them here.

This is a research prototype. The heuristics are simplified and are not a
clinical guideline.
"""

from __future__ import annotations

import re

from prosthetics_agents.core.contracts import (
    BiomechAssessment,
    ClinicalDocument,
    Component,
    DeviceRecommendation,
    PatientIntake,
    PatientRecord,
    ResidualLimb,
    SafetyIssue,
    SafetyReview,
)

from .catalog import K_RANK, by_id, candidates

DISCLAIMER = (
    "Research prototype output. Not a certified clinical decision-support tool. "
    "Every recommendation must be reviewed by a certified prosthetist/orthotist "
    "and the treating physician before any clinical use."
)

RED_FLAG_PATTERNS: list[tuple[str, str]] = [
    (r"dehiscence|purulent|active infection|unhealed|open wound", "unhealed wound / active infection"),
    (r"poorly controlled diabetes|hba1c\s*(9|1\d)", "poorly controlled diabetes"),
    (r"cognitive impairment|dementia", "cognitive impairment"),
    (r"limited exercise tolerance|heart failure|unstable angina", "cardiovascular limitation"),
]


# ── Intake (mock only: a regex stand-in for the LLM's text structuring) ──────


def _first(pattern: str, text: str, flags=re.I) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1) if m else None


def mock_intake(referral_text: str) -> PatientIntake:
    t = referral_text
    low = t.lower()

    if "transfemoral" in low:
        level = "transfemoral"
    elif "transtibial" in low:
        level = "transtibial"
    elif "partial foot" in low:
        level = "partial_foot"
    else:
        level = "none"

    side = None
    m = re.search(r"\b(left|right|bilateral)\b[^.]*?(amputation|foot drop|drop foot)", low)
    if m:
        side = m.group(1)

    # activity level, from strongest signal down
    if re.search(r"competitive|athlete|high-impact", low):
        k = "K4"
    elif re.search(r"unlimited|stairs and ramps|hiking|variable speed|jog|all day", low):
        k = "K3"
    elif re.search(r"short (community )?distances|mailbox|garden|limited community", low):
        k = "K2"
    elif re.search(r"transfer|within the home|household|indoors", low):
        k = "K1"
    else:
        k = "K2"

    limb_len = None
    m = re.search(r"residual limb (short|medium|long)", low)
    if m:
        limb_len = m.group(1)

    skin = []
    if "fragile skin" in low:
        skin.append("fragile distal skin")
    if re.search(r"dehiscence|purulent", low):
        skin.append("wound dehiscence with discharge")

    comorb = []
    for pat, name in [
        (r"type 2 diabetes|diabetic", "type 2 diabetes"),
        (r"hypertension", "hypertension"),
        (r"kidney disease", "chronic kidney disease"),
        (r"ischaemic heart|ischemic heart|coronary", "ischaemic heart disease"),
        (r"peripheral arterial", "peripheral arterial disease"),
        (r"reduced grip", "reduced grip strength"),
    ]:
        if re.search(pat, low):
            comorb.append(name)

    red = [name for pat, name in RED_FLAG_PATTERNS if re.search(pat, low)]

    etiology = ""
    for pat, name in [
        (r"accident|trauma", "trauma"),
        (r"arterial disease|vascular", "vascular"),
        (r"diabetic foot", "diabetic"),
        (r"tumou?r|sarcoma", "oncologic"),
    ]:
        if re.search(pat, low):
            etiology = name
            break

    goals = []
    for pat in [r"return to ([^.,]+)", r"wants to ([^.,]+)", r"goal:? ([^.]+)"]:
        for g in re.findall(pat, t, re.I):
            goals.append(g.strip())

    orth = ""
    if level == "none":
        parts = []
        if "drop" in low:
            parts.append("drop foot")
        parts.append("no spasticity" if "no spasticity" in low else "spasticity present")
        parts.append(
            "no medio-lateral instability"
            if "no medio-lateral instability" in low
            else "instability present"
        )
        orth = ", ".join(parts)

    months = _first(r"(\d+)\s*months? ago", t)
    weeks = _first(r"(\d+)\s*weeks? ago", t)
    years = _first(r"(\d+)\s*years? ago", t)
    tsa = int(months) if months else (int(years) * 12 if years else (0 if weeks else None))

    return PatientIntake(
        patient_id=_first(r"Patient ([A-Z]-\d+)", t) or "unknown",
        age=int(_first(r"(\d+)-year-old", t) or 0),
        sex="M" if " male" in low else ("F" if "female" in low else None),
        weight_kg=float(_first(r"(\d+(?:\.\d+)?)\s*kg", t) or 0),
        height_cm=float(_first(r"(\d+)\s*cm", t) or 0) or None,
        amputation_level=level,
        side=side,
        etiology=etiology,
        time_since_amputation_months=tsa,
        reported_k_level=k,
        residual_limb=ResidualLimb(length=limb_len, skin_issues=skin),
        comorbidities=comorb,
        goals=goals[:3],
        prior_device=_first(r"(temporary prosthesis[^.]*|prosthesis with [^.]*)", t) or "",
        orthotic_indication=orth,
        red_flags=red,
    )


# ── Biomechanical assessment ─────────────────────────────────────────────────


def assess_biomech(intake: PatientIntake) -> BiomechAssessment:
    k = intake.reported_k_level or "K2"
    rationale = f"Reported activity consistent with {k}."
    if intake.age >= 70 and intake.etiology == "vascular" and K_RANK[k] > 2:
        k, rationale = "K2", "Age and vascular etiology: capped at K2 pending gait trial."
    if "cardiovascular limitation" in intake.red_flags and K_RANK[k] > 1:
        k, rationale = "K1", "Cardiovascular limitation restricts sustained ambulation."

    w = intake.weight_kg
    load = "light" if w < 75 else ("standard" if w <= 125 else "heavy")

    gait, susp, contra = [], [], []
    if intake.amputation_level == "transfemoral":
        gait.append("Knee stability in early stance is the primary safety concern.")
        if intake.residual_limb.length == "short":
            gait.append("Short lever arm: reduced socket control, higher rotational forces.")
            susp.append("Auxiliary suspension (belt/lanyard) likely required.")
        if intake.residual_limb.length == "long":
            gait.append("Long lever arm: good control; polycentric/MPK knees fit well.")
    if intake.amputation_level == "transtibial":
        gait.append("Protect tibial crest and fibular head from pressure.")
    if intake.etiology in ("vascular", "diabetic") or intake.residual_limb.skin_issues:
        gait.append("Fragile skin: minimise shear, prefer total-surface-bearing with gel liner.")
    if load == "heavy":
        gait.append("Heavy-duty component ratings required (>125 kg).")
    if "reduced grip strength" in intake.comorbidities:
        susp.append("Avoid suspension that needs fine dexterity to don (suction valve).")
        contra.append("Suction suspension with manual valve (dexterity).")
    if K_RANK[k] >= 3 and not intake.residual_limb.skin_issues:
        susp.append("Stable limb volume: suction or elevated vacuum feasible.")
    if intake.amputation_level == "none" and intake.orthotic_indication:
        gait.append(f"Orthotic indication: {intake.orthotic_indication}.")

    return BiomechAssessment(
        confirmed_k_level=k,
        k_level_rationale=rationale,
        load_class=load,
        gait_considerations=gait,
        suspension_considerations=susp,
        contraindications=contra,
    )


# ── Recommendation ───────────────────────────────────────────────────────────


def _pick(category: str, k: str, weight: float | None, prefer_tags: list[str] | None = None):
    """First catalog candidate matching preferred tags, else any candidate."""
    pool = candidates(category, k, weight)
    for tag in prefer_tags or []:
        for c in pool:
            if tag in c["tags"]:
                return c
    return pool[0] if pool else None


def recommend(
    intake: PatientIntake, biomech: BiomechAssessment, feedback: list[str] | None = None
) -> DeviceRecommendation:
    """Function-first recommender.

    Deliberately does NOT filter by weight rating unless the safety gate has
    sent feedback about it — separation of concerns: the recommender optimises
    function, the gate enforces constraints. This is what makes the revise
    loop observable in mock mode.
    """
    feedback = feedback or []
    k = biomech.confirmed_k_level
    weight = intake.weight_kg if any("weight" in f.lower() for f in feedback) else None
    fragile = bool(intake.residual_limb.skin_issues) or intake.etiology in ("vascular", "diabetic")
    low_dex = "reduced grip strength" in intake.comorbidities
    comps: list[Component] = []
    alts: list[str] = []

    def add(c, why):
        if c:
            comps.append(Component(category=c["category"], catalog_id=c["id"], name=c["name"], rationale=why))

    if intake.amputation_level == "none":
        ind = intake.orthotic_indication
        if "no spasticity" in ind and "no medio-lateral instability" in ind and K_RANK[k] >= 2:
            afo = by_id("AFO-CARBON")
            why = "Flaccid drop foot, no spasticity/instability, active user: dynamic carbon AFO."
            alts.append("AFO-HINGED if dorsiflexion range for stairs becomes limiting")
        elif "spasticity present" in ind or "instability present" in ind:
            afo = by_id("AFO-SOLID")
            why = "Spasticity or medio-lateral instability: rigid ankle control."
        else:
            afo = by_id("AFO-HINGED")
            why = "Hinged AFO with plantarflexion stop."
        add(afo, why)
        return DeviceRecommendation(
            device_type="afo",
            components=comps,
            alternatives=alts,
            confidence=0.8,
            rationale=why,
        )

    if intake.amputation_level == "transtibial":
        socket = by_id("SK-TSB" if fragile else "SK-PTB")
        add(socket, "Total-surface bearing protects fragile skin." if fragile else "Standard PTB design for healthy limb.")
        if fragile:
            add(by_id("LN-CUSH"), "Cushion liner for skin protection.")
            add(by_id("SP-SLEEVE"), "Sleeve suspension avoids distal pin loading.")
            suspension = "sleeve"
        elif K_RANK[k] >= 3 and not low_dex:
            add(by_id("LN-SEAL"), "Seal-in liner for suction suspension.")
            add(by_id("SP-SUCTION"), "Suction: better limb control for active user.")
            suspension = "suction"
            alts.append("SP-VACUUM elevated vacuum for best volume control")
        else:
            add(by_id("LN-PIN"), "Pin liner: simple, robust.")
            add(by_id("SP-SHUTTLE"), "Shuttle lock: easy donning.")
            suspension = "pin-lock"
        foot = _pick("foot", k, weight, ["community", "uneven-terrain"])
        add(foot, f"Foot class for {k}.")
        return DeviceRecommendation(
            device_type="transtibial_prosthesis",
            socket_design=socket["name"] if socket else "",
            suspension=suspension,
            interface="gel liner",
            components=comps,
            alternatives=alts,
            confidence=0.85 if not intake.red_flags else 0.55,
            rationale=f"{k} transtibial user; {biomech.load_class} load class.",
        )

    # transfemoral
    long_limb = intake.residual_limb.length == "long"
    socket = by_id("SK-SUBI" if (K_RANK[k] >= 3 and long_limb) else "SK-IC")
    add(socket, "Sub-ischial comfort socket for active user with long limb." if socket and socket["id"] == "SK-SUBI" else "Ischial containment for medio-lateral stability.")
    if K_RANK[k] <= 1:
        knee = by_id("KN-STANCE")
        why = "Stance-control knee: maximum stability for household ambulator."
        alts.append("KN-LOCK manual-locking knee if stance control is insufficient")
    elif K_RANK[k] == 2:
        knee = by_id("KN-STANCE" if intake.residual_limb.length == "short" else "KN-POLY")
        why = "Stance-control knee for short limb / fall-risk user." if intake.residual_limb.length == "short" else "Polycentric knee: stance stability with toe clearance."
    else:
        wants_mpk = re.search(r"stumble|fall", " ".join(intake.goals) + " " + intake.prior_device, re.I) or K_RANK[k] >= 3
        knee = _pick("knee", k, weight, ["stumble-recovery"]) if wants_mpk else _pick("knee", k, weight, ["variable-cadence"])
        why = "Microprocessor knee: stumble recovery and variable cadence." if knee and "MPK" in knee["id"] else "Hydraulic knee for variable cadence."
        alts.append("KN-HYD hydraulic knee if MPK not reimbursed")
    add(knee, why)
    if intake.residual_limb.length == "short" or low_dex:
        add(by_id("LN-PIN"), "Pin liner: no dexterity needed.")
        add(by_id("SP-SHUTTLE"), "Shuttle lock.")
        add(by_id("SP-BELT"), "Auxiliary belt for short limb.")
        suspension = "pin-lock + auxiliary belt"
    else:
        add(by_id("LN-SEAL"), "Seal-in liner.")
        add(by_id("SP-SUCTION"), "Suction suspension for active user.")
        suspension = "suction"
    foot = _pick("foot", k, weight, ["community", "knee-stability", "uneven-terrain"])
    add(foot, f"Foot class for {k}.")
    return DeviceRecommendation(
        device_type="transfemoral_prosthesis",
        socket_design=socket["name"] if socket else "",
        suspension=suspension,
        interface="gel liner",
        components=comps,
        alternatives=alts,
        confidence=0.8 if not intake.red_flags else 0.5,
        rationale=f"{k} transfemoral user; {biomech.load_class} load class.",
    )


# ── Safety gate — deterministic verifier ─────────────────────────────────────


def verify_components(rec: DeviceRecommendation, weight_kg: float, k_level: str) -> list[SafetyIssue]:
    """Hard checks every recommendation must pass, regardless of who produced it."""
    issues: list[SafetyIssue] = []
    for comp in rec.components:
        c = by_id(comp.catalog_id)
        if c is None:
            issues.append(SafetyIssue(severity="critical", message=f"{comp.catalog_id}: not in catalog."))
            continue
        if weight_kg > c["weight_limit_kg"]:
            issues.append(SafetyIssue(
                severity="critical",
                message=f"{c['id']} rated to {c['weight_limit_kg']} kg; patient weight {weight_kg:g} kg exceeds rating.",
            ))
        if K_RANK[k_level] < K_RANK[c["k_min"]]:
            issues.append(SafetyIssue(
                severity="critical",
                message=f"{c['id']} requires {c['k_min']}+; patient is {k_level} (over-prescription).",
            ))
        if K_RANK[k_level] > K_RANK[c["k_max"]]:
            issues.append(SafetyIssue(
                severity="warning",
                message=f"{c['id']} is rated up to {c['k_max']}; patient is {k_level} (under-prescription).",
            ))
    return issues


def review_safety(intake: PatientIntake, biomech: BiomechAssessment, rec: DeviceRecommendation) -> SafetyReview:
    issues: list[SafetyIssue] = []
    changes: list[str] = []

    for flag in intake.red_flags:
        issues.append(SafetyIssue(severity="critical", message=f"Red flag: {flag}."))
    comp_issues = verify_components(rec, intake.weight_kg, biomech.confirmed_k_level)
    issues.extend(comp_issues)
    for i in comp_issues:
        if i.severity == "critical":
            changes.append("Replace component: " + i.message)
    for contra in biomech.contraindications:
        if "suction" in contra.lower() and rec.suspension and "suction" in rec.suspension:
            issues.append(SafetyIssue(severity="critical", message=f"Contraindicated: {contra}"))
            changes.append("Change suspension to pin-lock or lanyard (dexterity).")
    if rec.confidence < 0.5:
        issues.append(SafetyIssue(severity="warning", message="Low recommender confidence."))

    escalate = bool(intake.red_flags) or rec.confidence < 0.5
    if escalate:
        verdict = "escalate"
        rationale = "Red flags or low confidence: fitting decision requires the clinical team."
    elif changes:
        verdict = "revise"
        rationale = "Recommendation violates hard constraints; returned for revision."
    else:
        verdict = "approved"
        rationale = "No red flags; all components within K-level and weight ratings."
    return SafetyReview(
        verdict=verdict,
        issues=issues,
        required_changes=changes,
        escalate_to_human=escalate,
        rationale=rationale,
    )


# ── Documentation ────────────────────────────────────────────────────────────


def write_document(record: PatientRecord) -> ClinicalDocument:
    i, b, r, s = record.intake, record.biomech, record.recommendation, record.safety
    assert i and b and r and s
    comps = "\n".join(f"- {c.category}: {c.name} [{c.catalog_id}] — {c.rationale}" for c in r.components)
    summary = (
        f"{i.age}-year-old {i.sex or ''} patient, {i.weight_kg:g} kg, {i.amputation_level} "
        f"({i.side or 'n/a'}), etiology {i.etiology or 'n/a'}; functional level {b.confirmed_k_level}."
    ).replace("  ", " ")
    justification = (
        f"{b.k_level_rationale} Load class: {b.load_class}. "
        + " ".join(b.gait_considerations)
        + f" Recommended {r.device_type.replace('_', ' ')}: {r.rationale}"
    )
    esc = ""
    if s.verdict == "escalate":
        esc = "ESCALATED FOR HUMAN REVIEW. " + "; ".join(x.message for x in s.issues if x.severity == "critical")
    follow = (
        "Fitting review at 2 weeks (skin check, socket fit), gait training referral, "
        "volume reassessment at 3 months."
        if s.verdict == "approved"
        else "Resolve escalation items before any fitting; re-refer when cleared."
    )
    return ClinicalDocument(
        summary=summary,
        justification=justification,
        device_specification=f"Socket: {r.socket_design or 'n/a'}; suspension: {r.suspension or 'n/a'}.\n{comps}",
        follow_up_plan=follow,
        escalation_note=esc,
        disclaimer=DISCLAIMER,
    )
