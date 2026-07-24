"""Hierarchical semantic diversity evaluation.

For each topic, diversity is evaluated in three auditable levels:

1. Each idea is independently decomposed into a multi-valued semantic signature.
2. Each pair receives either eight independent calls--one per signature axis--or one
   joint call covering all eight axes, using four semantic relations (same, variant,
   related, distinct).  A separate assimilation call sees only that frozen evidence
   and assigns an interpretable relationship class. Code maps that class
   deterministically to a 0--9 diversity score.
3. Code derives set-level statistics, exact distinct capacity, clone/family clusters,
   and a backwards-compatible severity label.  No LLM writes the group verdict.

Non-obviousness (whether an idea would surprise a knowledgeable reviewer) is a
per-idea quality judgment, not a pairwise diversity one -- it lives in
quality_eval.py alongside soundness, feasibility, and significance.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

from ideagent.console import print_warning
from ideagent.diversity_metrics import (
    AXIS_RELATION_TO_DISTANCE,
    RELATIONSHIP_CLASS_TO_SCORE,
    maximum_distinct_subset,
    maximum_similarity_cluster,
    pair_score_lookup,
)


AXIS_CALL_MODE_INDEPENDENT = "independent"
AXIS_CALL_MODE_JOINT = "joint_axes"
AXIS_CALL_MODES = (AXIS_CALL_MODE_INDEPENDENT, AXIS_CALL_MODE_JOINT)
DEFAULT_AXIS_CALL_MODE = AXIS_CALL_MODE_JOINT
EVALUATION_VERSIONS = {
    AXIS_CALL_MODE_INDEPENDENT: "hierarchical-semantic-v4-lineage-aware-independent-axes",
    AXIS_CALL_MODE_JOINT: "hierarchical-semantic-v4-lineage-aware-joint-axes-frozen-class",
}
# Backwards-compatible name for callers that mean the current default mode.
EVALUATION_VERSION = EVALUATION_VERSIONS[DEFAULT_AXIS_CALL_MODE]
DISTINCT_THRESHOLD = 6
NEAR_CLONE_MAX = 2
MECHANISM_FAMILY_MAX = 5

_AXES = (
    "failure_mode",
    "causal_diagnosis",
    "intervention",
    "signal_source",
    "intervention_locus",
    "objective",
    "eval_regime",
    "assumption_class",
)
_CORE_AXES = ("failure_mode", "causal_diagnosis", "intervention")
_AXIS_RELATIONS = tuple(AXIS_RELATION_TO_DISTANCE)
_RELATIONSHIP_CLASSES = tuple(RELATIONSHIP_CLASS_TO_SCORE)

_AXIS_DEFINITIONS: dict[str, str] = {
    "failure_mode": "The concrete problem or failure the idea is trying to solve.",
    "causal_diagnosis": "The claimed reason that failure occurs.",
    "intervention": "The central causal lever or method, not its coined name.",
    "signal_source": "The evidence or measurable information the method relies on.",
    "intervention_locus": "Where the method acts in the system or workflow.",
    "objective": "The outcome or tradeoff being optimized.",
    "eval_regime": "The experimental setting required to test the idea.",
    "assumption_class": "The important claims about the world or model that must hold.",
}


@dataclass
class IdeaSignature:
    idea_idx: int
    semantic_core: str
    failure_mode: list[str] = field(default_factory=list)
    causal_diagnosis: list[str] = field(default_factory=list)
    intervention: list[str] = field(default_factory=list)
    signal_source: list[str] = field(default_factory=list)
    intervention_locus: list[str] = field(default_factory=list)
    objective: list[str] = field(default_factory=list)
    eval_regime: list[str] = field(default_factory=list)
    assumption_class: list[str] = field(default_factory=list)


@dataclass
class AxisAssessment:
    axis: str
    differ: bool
    note: str
    relation: str = "same"
    semantic_distance: int = 0


@dataclass
class PairwiseScore:
    idea_i: int
    idea_j: int
    axis_assessments: list[AxisAssessment] = field(default_factory=list)
    score: int = 0
    rationale: str = ""
    relationship_class: str = "semantic_duplicate"


def _pair_score_from_cache(value: dict[str, Any]) -> PairwiseScore:
    return PairwiseScore(
        idea_i=int(value["idea_i"]),
        idea_j=int(value["idea_j"]),
        axis_assessments=[AxisAssessment(**a) for a in value["axis_assessments"]],
        score=int(value["score"]),
        rationale=str(value.get("rationale", "")),
        relationship_class=str(value.get("relationship_class", "semantic_duplicate")),
    )


@dataclass
class GroupReport:
    topic_id: str
    n_ideas: int
    pairwise_scores: list[PairwiseScore] = field(default_factory=list)
    avg_pairwise_score: float = 0.0
    stable_core_severity: str = "S0"
    dominant_stable_core: str = ""
    most_repeated_axis: str = ""
    most_diverse_axis: str = ""
    short_explanation: str = ""
    verdict: str = ""
    evaluation_version: str = EVALUATION_VERSION
    axis_call_mode: str = DEFAULT_AXIS_CALL_MODE
    idea_signatures: list[IdeaSignature] = field(default_factory=list)
    min_pairwise_score: int = 0
    non_distinct_pair_count: int = 0
    non_distinct_pair_rate: float = 0.0
    near_clone_pair_count: int = 0
    near_clone_pair_rate: float = 0.0
    nearest_neighbor_scores: list[dict[str, int]] = field(default_factory=list)
    distinct_capacity_d6: int = 0
    distinct_capacity_d6_idea_indices: list[int] = field(default_factory=list)
    largest_near_clone_cluster: dict[str, Any] = field(default_factory=dict)
    largest_mechanism_family_cluster: dict[str, Any] = field(default_factory=dict)
    axis_relation_summary: dict[str, Any] = field(default_factory=dict)
    skipped_same_lineage_pair_count: int = 0


_SIGNATURE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "semantic_idea_signature",
        "schema": {
            "type": "object",
            "properties": {
                "semantic_core": {"type": "string"},
                **{
                    axis: {"type": "array", "items": {"type": "string"}}
                    for axis in _AXES
                },
            },
            "required": ["semantic_core", *_AXES],
            "additionalProperties": False,
        },
    },
}

_AXIS_RELATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "single_axis_semantic_relationship",
        "schema": {
            "type": "object",
            "properties": {
                "relation": {"type": "string", "enum": list(_AXIS_RELATIONS)},
                "note": {"type": "string"},
            },
            "required": ["relation", "note"],
            "additionalProperties": False,
        },
    },
}

_JOINT_AXIS_RELATIONS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "joint_axis_semantic_relationships",
        "schema": {
            "type": "object",
            "properties": {
                axis: {
                    "type": "object",
                    "properties": {
                        "relation": {"type": "string", "enum": list(_AXIS_RELATIONS)},
                        "note": {"type": "string"},
                    },
                    "required": ["relation", "note"],
                    "additionalProperties": False,
                }
                for axis in _AXES
            },
            "required": list(_AXES),
            "additionalProperties": False,
        },
    },
}

_PAIR_CLASS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "pair_relationship_class",
        "schema": {
            "type": "object",
            "properties": {
                "relationship_class": {
                    "type": "string",
                    "enum": list(_RELATIONSHIP_CLASSES),
                },
                "rationale": {"type": "string"},
            },
            "required": ["relationship_class", "rationale"],
            "additionalProperties": False,
        },
    },
}

SIGNATURE_EXTRACTION_SYSTEM_PROMPT = """\
You decompose one research idea into an auditable semantic signature. Do not evaluate its
quality, novelty, feasibility, or relation to any other idea. Describe what the idea actually
claims, not its terminology.

Return a concise one-sentence semantic_core of the form: because [causal diagnosis], use
[central intervention] with [signal/locus] to address [failure]. Then return ordered lists for
the eight axes supplied in the output schema. The first list item is the primary component;
later items are genuine secondary components. Lists may be empty when the idea does not specify
an axis. Preserve compound ideas: an idea may use multiple mechanisms, signals, or loci. Do not
force it into a single category and do not invent missing details.

OUTPUT: strict JSON matching the schema and nothing else.\
"""

AXIS_RELATION_SYSTEM_PROMPT = """\
You judge ONE semantic axis for TWO research ideas. You receive the untouched idea texts,
independently extracted signatures, one axis name, and that axis's definition. Judge only that
axis. Do not infer, discuss, or score any other axis, overall diversity, novelty, or quality.

Use semantic roles rather than lexical overlap:
  same     -- equivalent primary concept or operational role.
  variant  -- the same underlying concept with a meaningful implementation modification.
  related  -- different concepts in the same broader family, or partially overlapping composites.
  distinct -- genuinely different causal or operational concepts.

For compound ideas, compare primary roles and the complete composition. Sharing one secondary
component does not make two ideas the same. In particular, a shared signal source or evaluation
setting cannot by itself imply mechanism duplication.

Write one concise, neutral note referring to "one idea" and "the other idea" rather than A/B.
OUTPUT strict JSON with relation and note only.\
"""

JOINT_AXIS_RELATIONS_SYSTEM_PROMPT = """\
You judge ALL EIGHT semantic axes for TWO research ideas in one call. You receive the untouched
idea texts, independently extracted signatures, and definitions for every axis. Judge every axis
semantically, but do not assign an overall diversity score or relationship class and do not judge
novelty or quality.

For each axis use exactly one relation:
  same     -- equivalent primary concept or operational role.
  variant  -- the same underlying concept with a meaningful implementation modification.
  related  -- different concepts in the same broader family, or partially overlapping composites.
  distinct -- genuinely different causal or operational concepts.

Use semantic roles rather than lexical overlap. For compound ideas, compare primary roles and the
complete composition. Sharing one secondary component does not make two ideas the same. A shared
signal source or evaluation setting cannot by itself imply mechanism duplication.

Return one concise, neutral note per axis, referring to "one idea" and "the other idea" rather
than A/B. Treat each axis as a distinct judgment even though all are returned together. OUTPUT
strict JSON keyed by all eight axis names, with relation and note for each, and nothing else.\
"""

PAIR_SCORE_SYSTEM_PROMPT = """\
You assimilate EIGHT semantic-axis judgments produced in a completed earlier stage. The evidence
is frozen: do not redo, override, or reinterpret any individual axis. Select one overall
relationship class using only the supplied relations and notes.

The class is converted to 0--9 by code:
  semantic_duplicate (0): same research claim and substantive mechanism.
  cosmetic_rewrite (1): only wording, presentation, or naming differs.
  minor_variant (2): same core mechanism and causal story with a small operational change.
  meaningful_variant (3): meaningful modification, but still the same research direction.
  sibling_direction (4): one core component changes; recognizable common direction remains.
  partially_distinct (5): substantial change, but mechanism-family/core overlap remains.
  distinct_direction (6): separately actionable direction with a genuinely different central
      mechanism or causal route; contextual differences alone never qualify.
  clearly_distinct (7): mechanism, diagnosis, and evidence path are clearly separated.
  highly_distinct (8): only the broad problem area or domain substantially connects the ideas.
  orthogonal (9): essentially all substantive components differ; reserve this for rare cases.

CORE AXES are failure_mode, causal_diagnosis, and intervention. Supporting axes explain how the
core is instantiated. Scores/classes 0--2 are rediscoveries, 3--5 are one broader research family,
and 6--9 are separately countable directions. A class of 6 or higher requires substantive core
separation; differences only in objective, locus, evaluation, or wording are insufficient.

Write a short rationale naming the decisive frozen evidence. OUTPUT strict JSON with
relationship_class and rationale only.\
"""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    candidate = match.group(0) if match else stripped
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    result = [str(item).strip() for item in value if str(item).strip()]
    return result


def _parse_signature(raw: str, *, idx: int) -> IdeaSignature | None:
    obj = _extract_json_object(raw)
    if obj is None:
        return None
    semantic_core = str(obj.get("semantic_core", "")).strip()
    if not semantic_core:
        return None
    values: dict[str, list[str]] = {}
    for axis in _AXES:
        parsed = _string_list(obj.get(axis))
        if parsed is None:
            return None
        values[axis] = parsed
    return IdeaSignature(idea_idx=idx, semantic_core=semantic_core, **values)


def _grossly_inconsistent(pair_class: str, assessments: list[AxisAssessment]) -> bool:
    """Reject only unambiguous class/axis contradictions and allow nuanced cases."""

    score = RELATIONSHIP_CLASS_TO_SCORE[pair_class]
    relations = {a.axis: a.relation for a in assessments}
    core = [relations[axis] for axis in _CORE_AXES]
    if score == 0 and any(a.relation != "same" for a in assessments):
        return True
    if score <= 2 and relations["intervention"] == "distinct":
        return True
    if score >= 6 and all(r in {"same", "variant"} for r in core):
        return True
    if score == 9 and any(relations[axis] != "distinct" for axis in _CORE_AXES):
        return True
    return False


def _parse_axis_relationship(raw: str, *, axis: str) -> AxisAssessment | None:
    obj = _extract_json_object(raw)
    if obj is None:
        return None
    relation = str(obj.get("relation", "")).strip().lower()
    if relation not in AXIS_RELATION_TO_DISTANCE:
        return None
    distance = AXIS_RELATION_TO_DISTANCE[relation]
    note = str(obj.get("note", "")).strip()
    if not note:
        return None
    return AxisAssessment(
        axis=axis,
        relation=relation,
        semantic_distance=distance,
        differ=distance >= AXIS_RELATION_TO_DISTANCE["related"],
        note=note,
    )


def _parse_joint_axis_relationships(raw: str) -> list[AxisAssessment] | None:
    obj = _extract_json_object(raw)
    if obj is None or set(obj) != set(_AXES):
        return None
    assessments: list[AxisAssessment] = []
    for axis in _AXES:
        payload = obj.get(axis)
        if not isinstance(payload, dict):
            return None
        parsed = _parse_axis_relationship(json.dumps(payload), axis=axis)
        if parsed is None:
            return None
        assessments.append(parsed)
    return assessments


def _parse_relationship_class(
    raw: str,
    *,
    idea_i: int,
    idea_j: int,
    axis_assessments: list[AxisAssessment],
) -> PairwiseScore | None:
    obj = _extract_json_object(raw)
    if obj is None:
        return None
    relationship_class = str(obj.get("relationship_class", "")).strip()
    if relationship_class not in RELATIONSHIP_CLASS_TO_SCORE:
        return None
    if {assessment.axis for assessment in axis_assessments} != set(_AXES):
        return None
    ordered = sorted(axis_assessments, key=lambda assessment: _AXES.index(assessment.axis))
    if _grossly_inconsistent(relationship_class, ordered):
        return None
    return PairwiseScore(
        idea_i=idea_i,
        idea_j=idea_j,
        axis_assessments=ordered,
        relationship_class=relationship_class,
        score=RELATIONSHIP_CLASS_TO_SCORE[relationship_class],
        rationale=str(obj.get("rationale", "")).strip(),
    )


_EVAL_KEYS = (
    "idea_summary", "core_claim", "mechanism", "problem_targeted",
    "expected_effect", "source_gap",
)


def _render_idea_block(label: str, idea: dict[str, Any]) -> str:
    lines = [f"<{label}>"]
    idea_text = str(idea.get("idea_text", "")).strip()
    if idea_text:
        lines.extend([idea_text, f"</{label}>"])
        return "\n".join(lines)
    summary = idea.get("final_idea_core")
    if summary and isinstance(summary, dict):
        for key in _EVAL_KEYS:
            val = str(summary.get(key, "")).strip()
            if val:
                lines.append(f"{key.upper()}: {val}")
            else:
                print_warning(f"[diversity_eval] {label}: missing final_idea_core key '{key}'")
    else:
        lines.append("(no structured data available)")
    lines.append(f"</{label}>")
    return "\n".join(lines)


def _build_signature_user_message(idea: dict[str, Any]) -> str:
    definitions = "\n".join(f"- {axis}: {_AXIS_DEFINITIONS[axis]}" for axis in _AXES)
    return (
        f"{_render_idea_block('IDEA', idea)}\n\nAXIS DEFINITIONS:\n{definitions}\n\n"
        "Decompose this idea. Return strict JSON and nothing else."
    )


def _signature_payload(signature: IdeaSignature) -> dict[str, Any]:
    return {
        "semantic_core": signature.semantic_core,
        **{axis: getattr(signature, axis) for axis in _AXES},
    }


def _build_axis_user_message(
    idea_a: dict[str, Any],
    idea_b: dict[str, Any],
    signature_a: IdeaSignature,
    signature_b: IdeaSignature,
    axis: str,
) -> str:
    if random.random() < 0.5:
        first = (idea_a, signature_a)
        second = (idea_b, signature_b)
    else:
        first = (idea_b, signature_b)
        second = (idea_a, signature_a)
    return (
        f"{_render_idea_block('IDEA_A', first[0])}\n"
        f"<SIGNATURE_A>{json.dumps(_signature_payload(first[1]), ensure_ascii=False)}</SIGNATURE_A>\n\n"
        f"{_render_idea_block('IDEA_B', second[0])}\n"
        f"<SIGNATURE_B>{json.dumps(_signature_payload(second[1]), ensure_ascii=False)}</SIGNATURE_B>\n\n"
        f"AXIS: {axis}\nDEFINITION: {_AXIS_DEFINITIONS[axis]}\n\n"
        f"Judge only the '{axis}' axis. Return strict JSON and nothing else."
    )


def _build_joint_axis_user_message(
    idea_a: dict[str, Any],
    idea_b: dict[str, Any],
    signature_a: IdeaSignature,
    signature_b: IdeaSignature,
) -> str:
    if random.random() < 0.5:
        first = (idea_a, signature_a)
        second = (idea_b, signature_b)
    else:
        first = (idea_b, signature_b)
        second = (idea_a, signature_a)
    definitions = "\n".join(f"- {axis}: {_AXIS_DEFINITIONS[axis]}" for axis in _AXES)
    return (
        f"{_render_idea_block('IDEA_A', first[0])}\n"
        f"<SIGNATURE_A>{json.dumps(_signature_payload(first[1]), ensure_ascii=False)}</SIGNATURE_A>\n\n"
        f"{_render_idea_block('IDEA_B', second[0])}\n"
        f"<SIGNATURE_B>{json.dumps(_signature_payload(second[1]), ensure_ascii=False)}</SIGNATURE_B>\n\n"
        f"AXIS DEFINITIONS:\n{definitions}\n\n"
        "Judge all eight axes, but do not assign an overall relationship class or score. "
        "Return strict JSON and nothing else."
    )


def _build_pair_class_user_message(axis_assessments: list[AxisAssessment]) -> str:
    ordered = sorted(axis_assessments, key=lambda assessment: _AXES.index(assessment.axis))
    lines = [
        f"- {assessment.axis}: {assessment.relation} — {assessment.note}"
        for assessment in ordered
    ]
    return (
        "FROZEN AXIS JUDGMENTS:\n"
        + "\n".join(lines)
        + "\n\nSelect the overall relationship class without revising the axis evidence. "
        "Return strict JSON and nothing else."
    )


def detect_format(record: dict[str, Any]) -> str:
    if "items" in record:
        return "agentic"
    raise ValueError(
        f"Cannot auto-detect JSONL format; record has keys: {list(record.keys())}. "
        "Expected 'items' key (agentic/one_shot format)."
    )


def extract_ideas_from_agentic_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    ideas = []
    for idx, item in enumerate(record.get("items", [])):
        metadata = item.get("source_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        ideas.append({
            "idx": idx,
            "idea_text": item.get("idea_text"),
            "final_idea_core": item.get("final_idea_core"),
            "lineage_id": metadata.get("lineage_id") or item.get("lineage_id"),
            "parent_id": metadata.get("parent_id") or item.get("parent_id"),
        })
    return ideas


def _same_lineage_pair(idea_i: dict[str, Any], idea_j: dict[str, Any]) -> bool:
    lineage_i = str(idea_i.get("lineage_id") or "").strip()
    lineage_j = str(idea_j.get("lineage_id") or "").strip()
    return bool(lineage_i and lineage_i == lineage_j)


def _excluded_lineage_score(idea_i: dict[str, Any], idea_j: dict[str, Any]) -> PairwiseScore:
    """Structural edge used for selection, never a semantic-diversity judgment."""
    idx_i, idx_j = int(idea_i["idx"]), int(idea_j["idx"])
    return PairwiseScore(
        idea_i=idx_i,
        idea_j=idx_j,
        axis_assessments=[],
        score=0,
        rationale=(
            "Diversity evaluation skipped: both records are accepted versions of the same "
            "mechanism lineage and cannot be selected together."
        ),
        relationship_class="same_lineage_excluded",
    )


def _axis_summaries(pair_scores: list[PairwiseScore]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for axis in _AXES:
        values = [
            assessment
            for pair in pair_scores
            for assessment in pair.axis_assessments
            if assessment.axis == axis
        ]
        counts = {relation: 0 for relation in _AXIS_RELATIONS}
        for value in values:
            counts[value.relation] += 1
        total = len(values)
        summary[axis] = {
            "counts": counts,
            "rates": {
                relation: round(counts[relation] / total, 4) if total else 0.0
                for relation in _AXIS_RELATIONS
            },
            "mean_semantic_distance_0_3": round(
                mean(v.semantic_distance for v in values), 4
            ) if values else 0.0,
        }
    return summary


def _deterministic_group_fields(
    ideas: list[dict[str, Any]], pair_scores: list[PairwiseScore],
    *, excluded_pair_keys: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    excluded_pair_keys = excluded_pair_keys or set()
    indices = [int(idea["idx"]) for idea in ideas]
    lookup = pair_score_lookup(pair_scores)
    metric_pairs = [
        pair for pair in pair_scores
        if (min(pair.idea_i, pair.idea_j), max(pair.idea_i, pair.idea_j))
        not in excluded_pair_keys
    ]
    scores = [pair.score for pair in metric_pairs]
    n_pairs = len(scores)
    non_distinct = [score for score in scores if score < DISTINCT_THRESHOLD]
    near_clone = [score for score in scores if score <= NEAR_CLONE_MAX]
    # A lineage edge is score 0 for portfolio compatibility (so parent and child cannot both
    # be selected), but score 9 for semantic diagnostics so it cannot manufacture clone/family
    # severity. Nearest-neighbour values likewise use only cross-lineage comparisons.
    metric_lookup = dict(lookup)
    for idea_i, idea_j in excluded_pair_keys:
        metric_lookup[(idea_i, idea_j)] = 9
        metric_lookup[(idea_j, idea_i)] = 9
    nearest = [
        {
            "idea_idx": idx,
            "score": min(
                (
                    metric_lookup[(min(idx, other), max(idx, other))]
                    for other in indices
                    if other != idx
                    and (min(idx, other), max(idx, other)) not in excluded_pair_keys
                ),
                default=0,
            ),
        }
        for idx in indices
    ]
    distinct = maximum_distinct_subset(indices, lookup, min_score=DISTINCT_THRESHOLD)
    clone_cluster = maximum_similarity_cluster(
        indices, metric_lookup, max_score=NEAR_CLONE_MAX
    )
    family_cluster = maximum_similarity_cluster(
        indices, metric_lookup, max_score=MECHANISM_FAMILY_MAX
    )
    axis_summary = _axis_summaries(metric_pairs)
    repeated_axis = min(
        _AXES, key=lambda axis: axis_summary[axis]["mean_semantic_distance_0_3"]
    )
    diverse_axis = max(
        _AXES, key=lambda axis: axis_summary[axis]["mean_semantic_distance_0_3"]
    )

    if len(clone_cluster) >= 4:
        severity = "S4"
    elif len(clone_cluster) >= 3 or len(family_cluster) >= max(4, (len(indices) + 1) // 2):
        severity = "S3"
    elif len(clone_cluster) >= 2 or len(family_cluster) >= 3:
        severity = "S2"
    elif len(family_cluster) >= 2:
        severity = "S1"
    else:
        severity = "S0"

    if len(clone_cluster) >= 2:
        dominant = "Mutually near-clone ideas: " + ", ".join(
            str(idx + 1) for idx in clone_cluster
        )
    elif len(family_cluster) >= 2:
        dominant = "Mutually related mechanism-family ideas: " + ", ".join(
            str(idx + 1) for idx in family_cluster
        )
    else:
        dominant = ""
    verdicts = {
        "S0": "No mechanism-family overlap",
        "S1": "Isolated family overlap",
        "S2": "Moderate family reuse",
        "S3": "Strong mechanism-family concentration",
        "S4": "Repeated semantic template",
    }
    return {
        "stable_core_severity": severity,
        "dominant_stable_core": dominant,
        "most_repeated_axis": repeated_axis,
        "most_diverse_axis": diverse_axis,
        "short_explanation": (
            f"Mean cross-lineage pairwise diversity is "
            f"{mean(scores) if scores else 0.0:.3f}; "
            f"{len(non_distinct)}/{n_pairs} pairs are below D={DISTINCT_THRESHOLD} and "
            f"{len(near_clone)}/{n_pairs} are near-clones (D<={NEAR_CLONE_MAX}). "
            f"The exact mutually distinct capacity at D={DISTINCT_THRESHOLD} is "
            f"{len(distinct)}/{len(indices)}."
        ),
        "verdict": verdicts[severity],
        "min_pairwise_score": min(scores) if scores else 0,
        "non_distinct_pair_count": len(non_distinct),
        "non_distinct_pair_rate": round(len(non_distinct) / n_pairs, 4) if n_pairs else 0.0,
        "near_clone_pair_count": len(near_clone),
        "near_clone_pair_rate": round(len(near_clone) / n_pairs, 4) if n_pairs else 0.0,
        "nearest_neighbor_scores": nearest,
        "distinct_capacity_d6": len(distinct),
        "distinct_capacity_d6_idea_indices": distinct,
        "largest_near_clone_cluster": {
            "size": len(clone_cluster), "idea_indices": clone_cluster,
        },
        "largest_mechanism_family_cluster": {
            "size": len(family_cluster), "idea_indices": family_cluster,
        },
        "axis_relation_summary": axis_summary,
        "skipped_same_lineage_pair_count": len(excluded_pair_keys),
    }


class TopicalDiversityEvaluator:
    def __init__(
        self,
        *,
        client: Any,
        gen_kwargs: dict[str, Any],
        max_parsing_retries: int = 2,
        max_workers: int = 10,
        axis_call_mode: str = DEFAULT_AXIS_CALL_MODE,
        cache_path: Path | None = None,
        escalation_client: Any | None = None,
        escalation_after_attempts: int = 5,
    ) -> None:
        axis_call_mode = str(axis_call_mode).strip().lower()
        if axis_call_mode not in AXIS_CALL_MODES:
            choices = ", ".join(AXIS_CALL_MODES)
            raise ValueError(f"axis_call_mode must be one of: {choices}; got {axis_call_mode!r}")
        self.client = client
        self.gen_kwargs = gen_kwargs
        self.max_parsing_retries = max(0, int(max_parsing_retries))
        self.max_workers = max(1, int(max_workers))
        self.axis_call_mode = axis_call_mode
        # After this many failed attempts on the primary model, switch to
        # escalation_client for the remaining retries. None disables escalation.
        # Independent of the thinking-effort escalation below -- both can fire.
        self.escalation_client = escalation_client
        self.escalation_after_attempts = max(1, int(escalation_after_attempts))
        self.evaluation_version = EVALUATION_VERSIONS[axis_call_mode]
        # Sub-topic checkpoint: every successfully computed signature/NB score/axis
        # judgment/pair classification is appended here as soon as it succeeds, so a
        # later retry of the same topic (e.g. after one hard pair exhausted retries)
        # reuses everything that already worked instead of redoing all of it. Cleared
        # for a topic once evaluate() fully succeeds for it (see _clear_cache).
        self.cache_path = cache_path
        self._cache_lock = threading.Lock()

    def _load_cache(self, topic_id: str) -> dict[str, dict[Any, Any]]:
        cache: dict[str, dict[Any, Any]] = {"signature": {}, "axis": {}, "pair": {}}
        if self.cache_path is None or not self.cache_path.exists():
            return cache
        with self.cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("topic_id") != topic_id:
                    continue
                kind = record["kind"]
                key = record["key"]
                value = record["value"]
                if kind == "signature":
                    cache["signature"][int(key)] = IdeaSignature(**value)
                elif kind == "axis":
                    cache["axis"][tuple(key)] = [AxisAssessment(**a) for a in value]
                elif kind == "pair":
                    cache["pair"][tuple(key)] = _pair_score_from_cache(value)
        return cache

    def _append_cache(self, topic_id: str, kind: str, key: Any, value: Any) -> None:
        if self.cache_path is None:
            return
        record = {"topic_id": topic_id, "kind": kind, "key": key, "value": value}
        line = json.dumps(record, ensure_ascii=False)
        with self._cache_lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _clear_cache(self, topic_id: str) -> None:
        """Drop this topic's checkpoint entries once it has fully succeeded -- the real
        output now has it, and --resume skips it before evaluate() runs again, so
        there's nothing left to ever read these back for."""
        if self.cache_path is None or not self.cache_path.exists():
            return
        with self._cache_lock:
            with self.cache_path.open(encoding="utf-8") as handle:
                remaining = [
                    line for line in handle
                    if line.strip() and json.loads(line).get("topic_id") != topic_id
                ]
            with self.cache_path.open("w", encoding="utf-8") as handle:
                handle.writelines(remaining)

    def _stage_gen_kwargs(self, stage: str) -> dict[str, Any]:
        """Apply optional stage-specific generation overrides.

        Keys such as ``axis_thinking_effort`` are controller settings and must not
        leak into provider requests.  The unprefixed values remain fallbacks.
        """

        stages = ("signature", "axis", "joint_axis", "pair")
        overridable = ("thinking_effort", "max_new_tokens")
        kwargs = dict(self.gen_kwargs)
        overrides: dict[str, Any] = {}
        for configured_stage in stages:
            for name in overridable:
                value = kwargs.pop(f"{configured_stage}_{name}", None)
                if configured_stage == stage and value is not None:
                    overrides[name] = value
        kwargs.update(overrides)
        return kwargs

    def _call_with_retry(self, messages, response_format, parser, *, stage: str):
        kwargs = {
            **self._stage_gen_kwargs(stage),
            "response_format": response_format,
        }
        total_attempts = 1 + self.max_parsing_retries
        # Past the halfway point, a stage still failing at its configured effort is more
        # likely a systematic misjudgment than random formatting noise (e.g. pair
        # classification's answer has to cross-check against frozen axis evidence) --
        # more retries at the same effort won't fix that. Escalate to high for the back
        # half of the attempt budget instead. Proportional to max_parsing_retries (not a
        # fixed count) so it stays "the back half" whatever that's tuned to.
        escalate_after = max(1, total_attempts // 2)
        last_failure = "no attempts were made"
        for attempt in range(total_attempts):
            call_kwargs = kwargs if attempt < escalate_after else {**kwargs, "thinking_effort": "high"}
            use_escalation_client = (
                self.escalation_client is not None
                and attempt >= self.escalation_after_attempts
            )
            client = self.escalation_client if use_escalation_client else self.client
            try:
                raw = client.generate(messages, **call_kwargs)
            except Exception as exc:
                # generate() already retries transiently (backoff) before raising, so a
                # raise here means every one of those attempts failed the same way (e.g.
                # Claude returning a thinking-only response with no text block) -- treat
                # it as one failed attempt of this outer loop rather than aborting the
                # whole pair/topic on the first miss.
                last_failure = f"{type(exc).__name__}: {exc}"
                continue
            result = parser(raw)
            if result is not None:
                return result
            snippet = raw.strip().replace("\n", " ")[:300]
            last_failure = f"parser rejected response: {snippet!r}"
        print_warning(f"[diversity_eval] stage '{stage}' exhausted {total_attempts} attempts -- last failure: {last_failure}")
        return None

    def _run_pool(self, tasks: list[Any], workers: int, desc: str):
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [executor.submit(fn, *args) for fn, *args in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
                yield future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def extract_signature(self, idea: dict[str, Any]) -> IdeaSignature | None:
        messages = [
            {"role": "system", "content": SIGNATURE_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": _build_signature_user_message(idea)},
        ]
        return self._call_with_retry(
            messages, _SIGNATURE_RESPONSE_FORMAT,
            lambda raw: _parse_signature(raw, idx=int(idea["idx"])),
            stage="signature",
        )

    def evaluate_axis(
        self,
        idea_a: dict[str, Any], idea_b: dict[str, Any],
        signature_a: IdeaSignature, signature_b: IdeaSignature,
        axis: str,
    ) -> AxisAssessment | None:
        messages = [
            {"role": "system", "content": AXIS_RELATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_axis_user_message(
                    idea_a, idea_b, signature_a, signature_b, axis
                ),
            },
        ]
        return self._call_with_retry(
            messages, _AXIS_RELATION_RESPONSE_FORMAT,
            lambda raw: _parse_axis_relationship(raw, axis=axis),
            stage="axis",
        )

    def evaluate_axes_joint(
        self,
        idea_a: dict[str, Any], idea_b: dict[str, Any],
        signature_a: IdeaSignature, signature_b: IdeaSignature,
    ) -> list[AxisAssessment] | None:
        messages = [
            {"role": "system", "content": JOINT_AXIS_RELATIONS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_joint_axis_user_message(
                    idea_a, idea_b, signature_a, signature_b
                ),
            },
        ]
        return self._call_with_retry(
            messages, _JOINT_AXIS_RELATIONS_RESPONSE_FORMAT,
            _parse_joint_axis_relationships,
            stage="joint_axis",
        )

    def classify_pair(
        self,
        *,
        idea_i: int,
        idea_j: int,
        axis_assessments: list[AxisAssessment],
    ) -> PairwiseScore | None:
        messages = [
            {"role": "system", "content": PAIR_SCORE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_pair_class_user_message(axis_assessments),
            },
        ]
        return self._call_with_retry(
            messages, _PAIR_CLASS_RESPONSE_FORMAT,
            lambda raw: _parse_relationship_class(
                raw, idea_i=idea_i, idea_j=idea_j,
                axis_assessments=axis_assessments,
            ),
            stage="pair",
        )

    def evaluate(self, *, topic_id: str, ideas: list[dict[str, Any]]) -> GroupReport | None:
        n = len(ideas)
        if n < 2:
            print_warning(f"[diversity_eval] topic {topic_id}: need >=2 ideas, got {n}; skipping.")
            return None

        # Everything already computed for this topic in a prior, partially-failed run is
        # reused instead of redone -- only the genuine gaps get a fresh call.
        cache = self._load_cache(topic_id)
        signatures: dict[int, IdeaSignature] = dict(cache["signature"])

        def _signature(pos: int):
            result = self.extract_signature(ideas[pos])
            if result is None:
                raise RuntimeError(
                    f"[diversity_eval] topic {topic_id}: idea {pos} signature failed after retries."
                )
            self._append_cache(topic_id, "signature", result.idea_idx, asdict(result))
            return result

        signature_tasks = [
            (_signature, pos) for pos in range(n) if int(ideas[pos]["idx"]) not in signatures
        ]
        if signature_tasks:
            for result in self._run_pool(
                signature_tasks, max(1, min(self.max_workers, len(signature_tasks))),
                f"{topic_id} ({len(signature_tasks)} signatures remaining)",
            ):
                if result is not None:
                    signatures[result.idea_idx] = result

        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        excluded_pairs = {
            (i, j) for i, j in all_pairs if _same_lineage_pair(ideas[i], ideas[j])
        }
        pairs = [pair for pair in all_pairs if pair not in excluded_pairs]
        axis_results: dict[tuple[int, int], list[AxisAssessment]] = {}
        for i, j in pairs:
            idx_i, idx_j = int(ideas[i]["idx"]), int(ideas[j]["idx"])
            axis_results[(i, j)] = list(cache["axis"].get((idx_i, idx_j), []))

        def _axis(i: int, j: int, axis: str):
            idx_i, idx_j = int(ideas[i]["idx"]), int(ideas[j]["idx"])
            result = self.evaluate_axis(
                ideas[i], ideas[j], signatures[idx_i], signatures[idx_j], axis
            )
            if result is None:
                raise RuntimeError(
                    f"[diversity_eval] topic {topic_id}: pair ({idx_i},{idx_j}) "
                    f"axis '{axis}' failed after retries."
                )
            return (i, j), result

        if self.axis_call_mode == AXIS_CALL_MODE_INDEPENDENT:
            pending_pairs = [pair for pair in pairs if not axis_results[pair]]
            axis_tasks = [(_axis, i, j, axis) for i, j in pending_pairs for axis in _AXES]
            if axis_tasks:
                for pair, assessment in self._run_pool(
                    axis_tasks,
                    max(1, min(self.max_workers, len(axis_tasks))),
                    f"{topic_id} ({len(pending_pairs)} pairs x {len(_AXES)} independent axes remaining)",
                ):
                    axis_results[pair].append(assessment)
                    if len(axis_results[pair]) == len(_AXES):
                        i, j = pair
                        idx_i, idx_j = int(ideas[i]["idx"]), int(ideas[j]["idx"])
                        self._append_cache(
                            topic_id, "axis", [idx_i, idx_j],
                            [asdict(a) for a in axis_results[pair]],
                        )
        else:
            def _joint_axes(i: int, j: int):
                idx_i, idx_j = int(ideas[i]["idx"]), int(ideas[j]["idx"])
                result = self.evaluate_axes_joint(
                    ideas[i], ideas[j], signatures[idx_i], signatures[idx_j]
                )
                if result is None:
                    raise RuntimeError(
                        f"[diversity_eval] topic {topic_id}: pair ({idx_i},{idx_j}) "
                        "joint axis evaluation failed after retries."
                    )
                self._append_cache(
                    topic_id, "axis", [idx_i, idx_j], [asdict(a) for a in result]
                )
                return (i, j), result

            joint_tasks = [(_joint_axes, i, j) for i, j in pairs if not axis_results[(i, j)]]
            if joint_tasks:
                for pair, assessments in self._run_pool(
                    joint_tasks,
                    max(1, min(self.max_workers, len(joint_tasks))),
                    f"{topic_id} ({len(joint_tasks)} joint eight-axis calls remaining)",
                ):
                    axis_results[pair] = list(assessments)

        def _classify(i: int, j: int):
            idx_i, idx_j = int(ideas[i]["idx"]), int(ideas[j]["idx"])
            assessments_by_axis = {
                assessment.axis: assessment for assessment in axis_results[(i, j)]
            }
            result = self.classify_pair(
                idea_i=idx_i, idea_j=idx_j,
                axis_assessments=[assessments_by_axis[axis] for axis in _AXES],
            )
            if result is None:
                raise RuntimeError(
                    f"[diversity_eval] topic {topic_id}: pair ({idx_i},{idx_j}) "
                    "classification failed after retries."
                )
            self._append_cache(topic_id, "pair", [idx_i, idx_j], asdict(result))
            return result

        pair_scores: list[PairwiseScore] = []
        classify_tasks = []
        for i, j in pairs:
            idx_i, idx_j = int(ideas[i]["idx"]), int(ideas[j]["idx"])
            cached_pair = cache["pair"].get((idx_i, idx_j))
            if cached_pair is not None:
                pair_scores.append(cached_pair)
            else:
                classify_tasks.append((_classify, i, j))
        if classify_tasks:
            pair_scores.extend(self._run_pool(
                classify_tasks,
                max(1, min(self.max_workers, len(classify_tasks))),
                f"{topic_id} ({len(classify_tasks)} frozen-evidence classifications remaining)",
            ))
        pair_scores.extend(
            _excluded_lineage_score(ideas[i], ideas[j]) for i, j in excluded_pairs
        )
        pair_scores.sort(key=lambda p: (p.idea_i, p.idea_j))
        judged_pair_scores = [
            pair for pair in pair_scores
            if pair.relationship_class != "same_lineage_excluded"
        ]
        avg = mean(pair.score for pair in judged_pair_scores) if judged_pair_scores else 0.0
        excluded_pair_keys = {
            (min(int(ideas[i]["idx"]), int(ideas[j]["idx"])),
             max(int(ideas[i]["idx"]), int(ideas[j]["idx"])))
            for i, j in excluded_pairs
        }
        group = _deterministic_group_fields(
            ideas, pair_scores, excluded_pair_keys=excluded_pair_keys
        )
        ordered_signatures = [signatures[int(idea["idx"])] for idea in ideas]
        self._clear_cache(topic_id)
        return GroupReport(
            topic_id=topic_id,
            n_ideas=n,
            pairwise_scores=pair_scores,
            avg_pairwise_score=round(avg, 4),
            evaluation_version=self.evaluation_version,
            axis_call_mode=self.axis_call_mode,
            idea_signatures=ordered_signatures,
            **group,
        )


class GeminiTopicalDiversityEvaluator(TopicalDiversityEvaluator):
    async def _acall_with_retry(self, messages, response_format, parser, *, stage: str):
        kwargs = {
            **self._stage_gen_kwargs(stage),
            "response_format": response_format,
        }
        for _ in range(1 + self.max_parsing_retries):
            result = parser(await self.client.generate_async(messages, **kwargs))
            if result is not None:
                return result
        return None

    async def _evaluate_async(
        self, *, topic_id: str, ideas: list[dict[str, Any]]
    ) -> GroupReport | None:
        n = len(ideas)
        if n < 2:
            print_warning(f"[diversity_eval] topic {topic_id}: need >=2 ideas, got {n}; skipping.")
            return None
        semaphore = asyncio.Semaphore(self.max_workers)

        async def throttled(coro):
            async with semaphore:
                return await coro

        async def signature(idea):
            messages = [
                {"role": "system", "content": SIGNATURE_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_signature_user_message(idea)},
            ]
            result = await self._acall_with_retry(
                messages, _SIGNATURE_RESPONSE_FORMAT,
                lambda raw: _parse_signature(raw, idx=int(idea["idx"])),
                stage="signature",
            )
            if result is None:
                raise RuntimeError(
                    f"[diversity_eval] topic {topic_id}: idea {idea['idx']} signature failed."
                )
            return result

        signature_results = await asyncio.gather(*[throttled(signature(idea)) for idea in ideas])
        signatures = {sig.idea_idx: sig for sig in signature_results}
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        excluded_pairs = {
            (i, j) for i, j in all_pairs if _same_lineage_pair(ideas[i], ideas[j])
        }
        pairs = [pair for pair in all_pairs if pair not in excluded_pairs]

        async def evaluate_axis(i: int, j: int, axis: str):
            idea_i, idea_j = ideas[i], ideas[j]
            idx_i, idx_j = int(idea_i["idx"]), int(idea_j["idx"])
            messages = [
                {"role": "system", "content": AXIS_RELATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_axis_user_message(
                        idea_i, idea_j, signatures[idx_i], signatures[idx_j], axis
                    ),
                },
            ]
            result = await self._acall_with_retry(
                messages, _AXIS_RELATION_RESPONSE_FORMAT,
                lambda raw: _parse_axis_relationship(raw, axis=axis),
                stage="axis",
            )
            if result is None:
                raise RuntimeError(
                    f"[diversity_eval] topic {topic_id}: pair ({idx_i},{idx_j}) "
                    f"axis '{axis}' failed."
                )
            return (i, j), result

        axis_results: dict[tuple[int, int], list[AxisAssessment]] = {
            pair: [] for pair in pairs
        }
        if self.axis_call_mode == AXIS_CALL_MODE_INDEPENDENT:
            axis_outputs = await asyncio.gather(*[
                throttled(evaluate_axis(i, j, axis))
                for i, j in pairs for axis in _AXES
            ])
            for pair, assessment in axis_outputs:
                axis_results[pair].append(assessment)
        else:
            async def evaluate_axes_joint(i: int, j: int):
                idea_i, idea_j = ideas[i], ideas[j]
                idx_i, idx_j = int(idea_i["idx"]), int(idea_j["idx"])
                messages = [
                    {"role": "system", "content": JOINT_AXIS_RELATIONS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _build_joint_axis_user_message(
                            idea_i, idea_j, signatures[idx_i], signatures[idx_j]
                        ),
                    },
                ]
                result = await self._acall_with_retry(
                    messages, _JOINT_AXIS_RELATIONS_RESPONSE_FORMAT,
                    _parse_joint_axis_relationships,
                    stage="joint_axis",
                )
                if result is None:
                    raise RuntimeError(
                        f"[diversity_eval] topic {topic_id}: pair ({idx_i},{idx_j}) "
                        "joint axis evaluation failed."
                    )
                return (i, j), result

            joint_outputs = await asyncio.gather(*[
                throttled(evaluate_axes_joint(i, j)) for i, j in pairs
            ])
            for pair, assessments in joint_outputs:
                axis_results[pair].extend(assessments)

        async def classify(i: int, j: int):
            idx_i, idx_j = int(ideas[i]["idx"]), int(ideas[j]["idx"])
            assessments_by_axis = {
                assessment.axis: assessment for assessment in axis_results[(i, j)]
            }
            assessments = [assessments_by_axis[axis] for axis in _AXES]
            messages = [
                {"role": "system", "content": PAIR_SCORE_SYSTEM_PROMPT},
                {"role": "user", "content": _build_pair_class_user_message(assessments)},
            ]
            result = await self._acall_with_retry(
                messages, _PAIR_CLASS_RESPONSE_FORMAT,
                lambda raw: _parse_relationship_class(
                    raw, idea_i=idx_i, idea_j=idx_j,
                    axis_assessments=assessments,
                ),
                stage="pair",
            )
            if result is None:
                raise RuntimeError(
                    f"[diversity_eval] topic {topic_id}: pair ({idx_i},{idx_j}) "
                    "classification failed."
                )
            return result

        pair_scores = list(await asyncio.gather(*[
            throttled(classify(i, j)) for i, j in pairs
        ])) if pairs else []
        pair_scores.extend(
            _excluded_lineage_score(ideas[i], ideas[j]) for i, j in excluded_pairs
        )
        pair_scores.sort(key=lambda p: (p.idea_i, p.idea_j))
        judged_pair_scores = [
            pair for pair in pair_scores
            if pair.relationship_class != "same_lineage_excluded"
        ]
        avg = mean(pair.score for pair in judged_pair_scores) if judged_pair_scores else 0.0
        excluded_pair_keys = {
            (min(int(ideas[i]["idx"]), int(ideas[j]["idx"])),
             max(int(ideas[i]["idx"]), int(ideas[j]["idx"])))
            for i, j in excluded_pairs
        }
        group = _deterministic_group_fields(
            ideas, pair_scores, excluded_pair_keys=excluded_pair_keys
        )
        return GroupReport(
            topic_id=topic_id,
            n_ideas=n,
            pairwise_scores=pair_scores,
            avg_pairwise_score=round(avg, 4),
            evaluation_version=self.evaluation_version,
            axis_call_mode=self.axis_call_mode,
            idea_signatures=[signatures[int(idea["idx"])] for idea in ideas],
            **group,
        )

    def evaluate(self, *, topic_id: str, ideas: list[dict[str, Any]]) -> GroupReport | None:
        return asyncio.run(self._evaluate_async(topic_id=topic_id, ideas=ideas))
