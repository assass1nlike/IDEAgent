"""Generation operators (free / repair / pivot) and the rejected-memory consolidation prompt
for the Yield archive. Operator text is passed to IdeatorAgent.open() in the critic-challenge
slot as the directive for one free single-shot generation."""
from __future__ import annotations

from typing import Any

from ideagent.signature import SemanticSignature

SOUND_BY_CONSTRUCTION = (
    "SOUNDNESS COMES FROM CONSTRUCTION. Strongly prefer a mechanism whose correctness is "
    "establishable by design -- a well-defined estimator, a provable bound or invariant, a "
    "conservation/consistency property, or a reduction to a known-correct procedure -- over one "
    "whose success depends on the emergent empirical behavior of a learned component. The most "
    "common fatal flaw here is a mechanism whose core signal quietly degenerates (all-zero, "
    "uninformative, or circular) under a realistic case; before finalizing, name the step most "
    "likely to degenerate and show it does not."
)
NON_OBVIOUSNESS_PRESSURE = (
    "NON-OBVIOUSNESS IS A FLOOR, NOT A BOX TO CHECK. A superficially strong score (even 80/100) "
    "does not establish that the central move is harder to anticipate than another strong idea. "
    "Identify the most obvious competent approach to the same failure mode, then make the proposed "
    "causal lever genuinely discontinuous from that approach. Extra components, implementation "
    "detail, jargon, and combinations of familiar techniques do not create non-obviousness. The "
    "conceptual leap must remain clear after all decorative details are removed, and it must remain "
    "sound rather than borrowing surprise from a flaw."
)
# Backward-compatible private name.
_SOUND_BY_CONSTRUCTION = SOUND_BY_CONSTRUCTION
_RESTATE = (
    "Respond with your COMPLETE idea -- full motivation, mechanism, and justification -- not a delta."
)

REPAIR_CRITIC_SYSTEM = (
    "You are the repair critic in a research-ideation search. The controller only calls you for "
    "a candidate that is semantically diverse and close to passing its quality gates. Diagnose "
    "the stated defect against the COMPLETE idea. Return ONE precise, actionable challenge that "
    "the ideator can answer by revising the same core idea. Do not suggest abandoning the problem "
    "or switching mechanism families. Do not praise, summarize, score, or provide several options. "
    "When soundness is at issue, identify the exact degeneracy, circularity, missing invariant, or "
    "unjustified causal step and demand a construction-level repair. Treat a high non-obviousness "
    "score as provisional: explicitly protect the hard-to-anticipate central move from being "
    "conventionalized by the repair."
)

NOVELTY_BRANCH_CRITIC_SYSTEM = (
    "You are the novelty-branch critic in a research-ideation search. The supplied parent may be a "
    "strong idea, but the next candidate must become a NEW mechanism lineage, not a polished or "
    "decorated version. Identify the parent's most predictable causal assumption or intervention, "
    "then return ONE precise challenge requiring a genuinely different causal diagnosis, central "
    "lever, or source of decisive information. Do not propose merely another application, dataset, "
    "loss term, module, verifier, or combination of familiar components. The branch must remain "
    "technically sound and independently actionable. Do not praise, summarize, score, or give a "
    "menu of options."
)

RELATIVE_NB_SYSTEM = (
    "You are a blinded adjudicator comparing a PARENT research idea and its proposed POLISHED "
    "revision on NON-OBVIOUSNESS only. The labels IDEA_A and IDEA_B are arbitrary positions. Judge "
    "which central research move would be harder for a knowledgeable reviewer to anticipate "
    "independently. Ignore length, clarity, implementation detail, soundness, feasibility, writing "
    "polish, and coined terminology. A revision that merely elaborates or conventionalizes the same "
    "mechanism is not more non-obvious. Return strict JSON only with winner (A, B, or tie), "
    "confidence (low, medium, or high), and a concise rationale."
)

RELATIVE_NB_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "relative_non_obviousness",
        "schema": {
            "type": "object",
            "properties": {
                "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "rationale": {"type": "string"},
            },
            "required": ["winner", "confidence", "rationale"],
            "additionalProperties": False,
        },
    },
}


def build_repair_critic_user(
    idea_text: str,
    *,
    defect: str,
    feedback_notes: list[str],
    soundness_rationales: list[str],
    accepted: list[SemanticSignature],
) -> str:
    archive = "\n".join(f"- {s.compact()}" for s in accepted) or "(none yet)"
    feedback = "\n".join(f"- {x}" for x in feedback_notes if x.strip()) or "(none)"
    soundness = "\n".join(f"- {x}" for x in soundness_rationales if x.strip()) or "(none)"
    return (
        f"COMPLETE CANDIDATE IDEA:\n{idea_text.strip()}\n\n"
        f"FAILED OR WEAK GATE:\n{defect.strip()}\n\n"
        f"QUALITY-EVALUATOR EVIDENCE:\n{feedback}\n\n"
        f"SOUNDNESS-PANEL OBJECTIONS:\n{soundness}\n\n"
        "ACCEPTED ARCHIVE SIGNATURES (the repair must remain distinct from these):\n"
        f"{archive}\n\nReturn exactly one focused repair challenge."
    )


def build_novelty_branch_critic_user(
    idea_text: str,
    *,
    feedback_notes: list[str],
    accepted: list[SemanticSignature],
) -> str:
    archive = "\n".join(f"- {s.compact()}" for s in accepted) or "(none yet)"
    feedback = "\n".join(f"- {x}" for x in feedback_notes if x.strip()) or "(none)"
    return (
        f"COMPLETE PARENT IDEA:\n{idea_text.strip()}\n\n"
        f"QUALITY-EVALUATOR EVIDENCE:\n{feedback}\n\n"
        "ACTIVE AND HISTORICAL MECHANISM SIGNATURES (the branch must be distinct):\n"
        f"{archive}\n\n"
        f"{NON_OBVIOUSNESS_PRESSURE}\n\n"
        "Return exactly one focused novelty-branch challenge."
    )


def build_relative_nb_user(idea_a: str, idea_b: str) -> str:
    return (
        "<UNTRUSTED_IDEA_A>\n" + idea_a.strip() + "\n</UNTRUSTED_IDEA_A>\n\n"
        "<UNTRUSTED_IDEA_B>\n" + idea_b.strip() + "\n</UNTRUSTED_IDEA_B>\n\n"
        "Compare only the non-obviousness of the central research move. Return strict JSON."
    )


def _avoid_block(accepted: list[SemanticSignature], rejected_summary: str) -> str:
    lines: list[str] = []
    if accepted:
        lines.append(
            "ALREADY IN THE ARCHIVE — your idea must attack a DIFFERENT problem or use a "
            "DIFFERENT core mechanism than every one of these (no duplicates):"
        )
        lines += [f"  - {s.compact()}" for s in accepted]
    if rejected_summary.strip():
        lines.append(
            "COMPACT SEARCH MEMORY — do not repeat previously qualified mechanisms, rejected "
            "families, fatal flaws, or mechanisms explicitly reserved for repair:"
        )
        lines.append(rejected_summary.strip())
    return "\n".join(lines)


def op_free(accepted: list[SemanticSignature], rejected_summary: str) -> str:
    avoid = _avoid_block(accepted, rejected_summary)
    head = (
        "Propose a NOVEL, non-obvious, and SOUND research idea for this problem space. It must "
        "not be a direct combination, extension, or ablation of the background papers. You are "
        "free to choose any problem and any mechanism (it may combine several parts of the "
        "pipeline) -- just make it genuinely distinct from what already exists."
    )
    return "\n\n".join(
        p for p in [head, avoid, NON_OBVIOUSNESS_PRESSURE, _SOUND_BY_CONSTRUCTION, _RESTATE]
        if p.strip()
    )


def op_free_baseline(prior_generated: list[SemanticSignature]) -> str:
    """The same free-search directive with judgment-neutral baseline memory.

    A no-evaluator baseline cannot truthfully call its prior outputs accepted or rejected.  It
    therefore receives every earlier compact signature under a neutral label.  The research
    criteria, sound-by-construction pressure, and complete-restatement rule are identical to
    :func:`op_free`; only the provenance label differs so no hidden quality signal is invented.
    """

    avoid = ""
    if prior_generated:
        avoid = (
            "PREVIOUSLY GENERATED IDEAS — no quality judgment is implied. Your next idea must "
            "attack a DIFFERENT problem or use a DIFFERENT core mechanism than every one of "
            "these (no duplicates):\n"
            + "\n".join(f"  - {signature.compact()}" for signature in prior_generated)
        )
    head = (
        "Propose a NOVEL, non-obvious, and SOUND research idea for this problem space. It must "
        "not be a direct combination, extension, or ablation of the background papers. You are "
        "free to choose any problem and any mechanism (it may combine several parts of the "
        "pipeline) -- just make it genuinely distinct from what already exists."
    )
    return "\n\n".join(
        part for part in [
            head, avoid, NON_OBVIOUSNESS_PRESSURE, _SOUND_BY_CONSTRUCTION, _RESTATE
        ] if part.strip()
    )


def op_repair(idea_text: str, defect: str) -> str:
    return "\n\n".join([
        "Here is a promising idea that fell just short. Fix its SPECIFIC defect below while "
        "keeping its non-obvious core and its problem -- do not abandon the idea or switch to a "
        "different mechanism. Repair only what the defect names.",
        f"IDEA:\n{idea_text.strip()}",
        f"DEFECT TO FIX (from an independent panel):\n{defect.strip()}",
        NON_OBVIOUSNESS_PRESSURE,
        _SOUND_BY_CONSTRUCTION,
        _RESTATE,
    ])


def op_critic_revision(
    idea_text: str, critic_challenge: str, *, mode: str,
) -> str:
    """Reconstruct a persistent lineage for one budgeted revision step.

    The parent may have been generated many search steps ago, so its original in-memory chat is
    no longer available. Supplying the complete parent and focused critique preserves the lineage
    without keeping every old conversation in model context.
    """
    if mode == "accepted_refine":
        goal = (
            "This parent already passes every acceptance gate. POLISH the same mechanism to improve "
            "only the deficits named in the focused critic challenge while preserving its full "
            "conceptual leap. Feasibility is non-gating and must not be optimized unless the "
            "challenge explicitly names it. Do not conventionalize the idea merely to make it "
            "easier to explain or implement."
        )
    else:
        goal = (
            "This is a promising, diverse mechanism that narrowly missed acceptance. Repair the "
            "specific weakness without abandoning its problem or switching mechanism families."
        )
    return "\n\n".join([
        goal,
        f"COMPLETE PARENT IDEA:\n{idea_text.strip()}",
        f"FOCUSED CRITIC CHALLENGE:\n{critic_challenge.strip()}",
        NON_OBVIOUSNESS_PRESSURE,
        _SOUND_BY_CONSTRUCTION,
        _RESTATE,
    ])


def op_novelty_branch(
    idea_text: str,
    critic_challenge: str,
    *,
    accepted: list[SemanticSignature],
    rejected_summary: str,
) -> str:
    avoid = _avoid_block(accepted, rejected_summary)
    return "\n\n".join(part for part in [
        (
            "Use the parent only as intellectual context. Produce a NEW research direction with a "
            "different causal diagnosis, central intervention, or source of decisive information. "
            "It must remain meaningful if the parent's core mechanism is removed. Adding modules, "
            "applications, implementation detail, or safeguards to the parent is not a branch."
        ),
        f"COMPLETE PARENT IDEA:\n{idea_text.strip()}",
        f"NOVELTY-BRANCH CRITIC CHALLENGE:\n{critic_challenge.strip()}",
        avoid,
        NON_OBVIOUSNESS_PRESSURE,
        SOUND_BY_CONSTRUCTION,
        _RESTATE,
    ] if part.strip())


def op_pivot(accepted: list[SemanticSignature], rejected_summary: str, stuck_on: str) -> str:
    avoid = _avoid_block(accepted, rejected_summary)
    head = (
        "The search keeps landing on the same territory: "
        f"{stuck_on.strip() or 'a mechanism/problem already covered or already shown unsound'}. "
        "PIVOT HARD: propose an idea with a FUNDAMENTALLY different problem framing OR a "
        "different core mechanism / causal lever than that territory — not a variation of it."
    )
    return "\n\n".join(
        p for p in [head, avoid, NON_OBVIOUSNESS_PRESSURE, _SOUND_BY_CONSTRUCTION, _RESTATE]
        if p.strip()
    )


# ── rejected-memory consolidation (keeps prompt memory sub-linear) ──
CONSOLIDATION_SYSTEM = (
    "You maintain a COMPRESSED memory of rejected research ideas. Given the current summary and a "
    "batch of newly rejected idea signatures (each with its defect), produce an updated, SHORTER "
    "summary. Merge ideas into mechanism FAMILIES, note recurring DUPLICATE patterns, and list "
    "the recurring FATAL FLAWS. Do not include full idea text — only families, patterns, and "
    "flaws, terse enough to fit a prompt. Drop one-off noise; keep what a generator must avoid."
)
CONSOLIDATION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "rejected_summary",
        "schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}


def build_consolidation_user(existing_summary: str, pending: list[str]) -> str:
    return (
        f"CURRENT SUMMARY:\n{existing_summary.strip() or '(empty)'}\n\n"
        f"NEWLY REJECTED (signature :: defect):\n" + "\n".join(f"- {p}" for p in pending)
        + "\n\nReturn the updated compressed summary."
    )
