#!/usr/bin/env python3
"""Collapse an evaluated agentic version pool to one representative per fresh seed.

The input pool is evaluated *before* this step, so every accepted parent and accepted child has
external quality scores and cross-lineage diversity judgments. Within each seed, accepted
versions are the eligible alternatives; the seed's logged fallback is used only when no version
ever passed the internal acceptance gates. The output contains exactly one item per fresh seed
and reindexed quality/diversity reports, so downstream Top-5 evaluation sees the same N as the
sequential-memory baseline without counting parent and child as separate discoveries.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ideagent.diversity_eval import (
    AxisAssessment,
    PairwiseScore,
    _deterministic_group_fields,
)
from ideagent.diversity_metrics import pair_score_lookup


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _quality_by_idea(record: dict[str, Any]) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for score in record.get("scores", []):
        result.setdefault(int(score["idea_idx"]), {})[str(score["metric"])] = int(
            score["score"]
        )
    return result


def _pair_from_dict(record: dict[str, Any]) -> PairwiseScore:
    assessments = [
        AxisAssessment(
            axis=str(assessment["axis"]),
            differ=bool(assessment.get("differ", False)),
            note=str(assessment.get("note", "")),
            relation=str(assessment.get("relation", "same")),
            semantic_distance=int(assessment.get("semantic_distance", 0)),
        )
        for assessment in record.get("axis_assessments", [])
    ]
    return PairwiseScore(
        idea_i=int(record["idea_i"]),
        idea_j=int(record["idea_j"]),
        axis_assessments=assessments,
        score=int(record["score"]),
        rationale=str(record.get("rationale", "")),
        relationship_class=str(record.get("relationship_class", "semantic_duplicate")),
    )


def _selection_objective(
    selection: tuple[int, ...],
    *,
    quality: dict[int, dict[str, int]],
    pair_lookup: dict[tuple[int, int], int],
    nb_min: int,
    soundness_min: int,
    clarity_min: int,
    diversity_min: int,
) -> tuple[Any, ...]:
    passes = []
    total_q = 0.0
    feasibility = 0
    for idx in selection:
        metrics = quality[idx]
        soundness = int(metrics["soundness"])
        clarity = int(metrics["mechanism_clarity_specificity"])
        nb = int(metrics["nonobviousness"])
        passes.append(
            nb >= nb_min and soundness >= soundness_min and clarity >= clarity_min
        )
        total_q += 0.4 * nb + 0.4 * soundness + 0.2 * clarity
        feasibility += int(metrics["feasibility"])

    pair_scores = [
        int(pair_lookup[(left, right)])
        for left, right in itertools.combinations(selection, 2)
    ]
    distinct_pairs = sum(score >= diversity_min for score in pair_scores)
    min_pair = min(pair_scores) if pair_scores else 9
    # Yield-aligned and deterministic: preserve the most gate-passing representatives first,
    # then the broadest D-qualified coverage, then weakest-pair separation and core Q.
    return (
        sum(passes),
        distinct_pairs,
        min_pair,
        round(total_q, 9),
        sum(pair_scores),
        feasibility,
        tuple(-idx for idx in selection),
    )


def select_topic(
    ideas_record: dict[str, Any],
    diversity_record: dict[str, Any],
    quality_record: dict[str, Any],
    *,
    nb_min: int = 6,
    soundness_min: int = 6,
    clarity_min: int = 6,
    diversity_min: int = 6,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    topic_id = str(ideas_record["topic_id"])
    if str(diversity_record.get("topic_id")) != topic_id:
        raise ValueError(f"{topic_id}: diversity topic mismatch")
    if str(quality_record.get("topic_id")) != topic_id:
        raise ValueError(f"{topic_id}: quality topic mismatch")

    items = ideas_record.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{topic_id}: lineage pool has no items")
    n_pool = len(items)
    if int(diversity_record.get("n_ideas", -1)) != n_pool:
        raise ValueError(f"{topic_id}: diversity report does not cover the full version pool")
    if int(quality_record.get("n_ideas", -1)) != n_pool:
        raise ValueError(f"{topic_id}: quality report does not cover the full version pool")

    quality = _quality_by_idea(quality_record)
    required = {
        "nonobviousness", "soundness", "mechanism_clarity_specificity", "feasibility",
        "significance",
    }
    for idx in range(n_pool):
        if not required.issubset(quality.get(idx, {})):
            raise ValueError(f"{topic_id}: incomplete external quality scores for idea {idx}")

    grouped: dict[int, list[int]] = {}
    for idx, item in enumerate(items):
        metadata = item.get("source_metadata")
        if not isinstance(metadata, dict) or metadata.get("seed_index") is None:
            raise ValueError(f"{topic_id}: idea {idx} has no source_metadata.seed_index")
        grouped.setdefault(int(metadata["seed_index"]), []).append(idx)

    expected = int(
        ideas_record.get("fresh_seed_budget")
        or ideas_record.get("n_seed_slots")
        or len(grouped)
    )
    seed_indices = sorted(grouped)
    if seed_indices != list(range(1, expected + 1)):
        raise ValueError(
            f"{topic_id}: expected seed groups 1..{expected}, got {seed_indices}"
        )

    options: list[list[int]] = []
    option_basis: dict[int, str] = {}
    for seed_index in seed_indices:
        accepted = [
            idx for idx in grouped[seed_index]
            if bool((items[idx].get("source_metadata") or {}).get("accepted_version"))
        ]
        if accepted:
            options.append(accepted)
            option_basis[seed_index] = "all_accepted_parent_child_versions"
        else:
            fallbacks = [
                idx for idx in grouped[seed_index]
                if bool(
                    (items[idx].get("source_metadata") or {}).get(
                        "primary_seed_representative"
                    )
                )
            ]
            if len(fallbacks) != 1:
                raise ValueError(
                    f"{topic_id}: seed {seed_index} needs exactly one non-accepted fallback"
                )
            options.append(fallbacks)
            option_basis[seed_index] = "nonaccepted_seed_fallback"

    pair_lookup = pair_score_lookup(diversity_record.get("pairwise_scores", []))
    for left, right in itertools.combinations(range(n_pool), 2):
        if (left, right) not in pair_lookup:
            raise ValueError(f"{topic_id}: missing diversity pair ({left}, {right})")

    best_selection: tuple[int, ...] | None = None
    best_objective: tuple[Any, ...] | None = None
    combinations_considered = 0
    for selection in itertools.product(*options):
        combinations_considered += 1
        objective = _selection_objective(
            selection,
            quality=quality,
            pair_lookup=pair_lookup,
            nb_min=nb_min,
            soundness_min=soundness_min,
            clarity_min=clarity_min,
            diversity_min=diversity_min,
        )
        if best_objective is None or objective > best_objective:
            best_selection = tuple(selection)
            best_objective = objective
    assert best_selection is not None and best_objective is not None

    old_to_new = {old: new for new, old in enumerate(best_selection)}
    selected_items = []
    for new_idx, (seed_index, old_idx) in enumerate(zip(seed_indices, best_selection)):
        item = copy.deepcopy(items[old_idx])
        metadata = dict(item.get("source_metadata") or {})
        metadata.update({
            "lineage_pool_source_idea_idx": old_idx,
            "lineage_pool_source_idea_number": old_idx + 1,
            "lineage_selection_basis": option_basis[seed_index],
            "lineage_alternative_source_indices": list(options[new_idx]),
        })
        item["source_metadata"] = metadata
        selected_items.append(item)

    pair_records_by_key = {
        (min(int(pair["idea_i"]), int(pair["idea_j"])),
         max(int(pair["idea_i"]), int(pair["idea_j"]))): pair
        for pair in diversity_record.get("pairwise_scores", [])
    }
    selected_pair_dicts: list[dict[str, Any]] = []
    for old_i, old_j in itertools.combinations(best_selection, 2):
        source = copy.deepcopy(pair_records_by_key[(min(old_i, old_j), max(old_i, old_j))])
        source["idea_i"] = old_to_new[old_i]
        source["idea_j"] = old_to_new[old_j]
        selected_pair_dicts.append(source)
    selected_pair_dicts.sort(key=lambda pair: (pair["idea_i"], pair["idea_j"]))
    pair_objects = [_pair_from_dict(pair) for pair in selected_pair_dicts]
    excluded_pair_keys = {
        (int(pair["idea_i"]), int(pair["idea_j"]))
        for pair in selected_pair_dicts
        if pair.get("relationship_class") == "same_lineage_excluded"
    }
    group = _deterministic_group_fields(
        [{"idx": idx} for idx in range(expected)],
        pair_objects,
        excluded_pair_keys=excluded_pair_keys,
    )
    judged_scores = [
        int(pair["score"])
        for pair in selected_pair_dicts
        if pair.get("relationship_class") != "same_lineage_excluded"
    ]
    selected_signatures = []
    source_signatures = diversity_record.get("idea_signatures", [])
    if source_signatures:
        if len(source_signatures) != n_pool:
            raise ValueError(f"{topic_id}: incomplete idea signatures")
        for new_idx, old_idx in enumerate(best_selection):
            signature = copy.deepcopy(source_signatures[old_idx])
            signature["idea_idx"] = new_idx
            selected_signatures.append(signature)

    selected_diversity = {
        "topic_id": topic_id,
        "n_ideas": expected,
        "pairwise_scores": selected_pair_dicts,
        "avg_pairwise_score": round(mean(judged_scores), 4) if judged_scores else 0.0,
        "evaluation_version": diversity_record.get("evaluation_version"),
        "axis_call_mode": diversity_record.get("axis_call_mode"),
        "idea_signatures": selected_signatures,
        "source_version_pool_n": n_pool,
        "source_version_pool_indices": list(best_selection),
        **group,
    }

    selected_quality_scores = []
    for score in quality_record.get("scores", []):
        old_idx = int(score["idea_idx"])
        if old_idx not in old_to_new:
            continue
        selected = copy.deepcopy(score)
        selected["idea_idx"] = old_to_new[old_idx]
        selected_quality_scores.append(selected)
    selected_quality_scores.sort(key=lambda score: (int(score["idea_idx"]), str(score["metric"])))
    selected_quality = {
        **{key: value for key, value in quality_record.items() if key not in {"n_ideas", "scores"}},
        "topic_id": topic_id,
        "n_ideas": expected,
        "scores": selected_quality_scores,
        "source_version_pool_n": n_pool,
        "source_version_pool_indices": list(best_selection),
    }
    selected_ideas = {
        "topic_id": topic_id,
        "generation_mode": "agentic_external_best_version_per_seed",
        "n_generated": expected,
        "n_requested": expected,
        "source_version_pool_n": n_pool,
        "source_version_pool_indices": list(best_selection),
        "lineage_selection": {
            "objective_order": [
                "quality_gate_pass_count",
                f"cross_seed_pair_count_at_D>={diversity_min}",
                "minimum_cross_seed_diversity",
                "total_weighted_Q",
                "total_pairwise_diversity",
                "feasibility_exact_final_tiebreak",
            ],
            "quality_weights": {"non_obviousness": 0.4, "soundness": 0.4, "clarity": 0.2},
            "thresholds": {
                "non_obviousness": nb_min,
                "soundness": soundness_min,
                "clarity": clarity_min,
                "diversity": diversity_min,
            },
            "combinations_considered": combinations_considered,
            "winning_objective": list(best_objective[:-1]),
            "fallback_policy": (
                "use all accepted versions for a seed; only if none exists, use its logged "
                "primary representative so the output remains exactly seed-budget matched"
            ),
        },
        "items": selected_items,
    }
    return selected_ideas, selected_diversity, selected_quality


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one externally evaluated parent/child representative per fresh seed."
    )
    parser.add_argument("--ideas", type=Path, required=True)
    parser.add_argument("--diversity", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output-ideas", type=Path, required=True)
    parser.add_argument("--output-diversity", type=Path, required=True)
    parser.add_argument("--output-quality", type=Path, required=True)
    parser.add_argument("--nb-min", type=int, default=6)
    parser.add_argument("--soundness-min", type=int, default=6)
    parser.add_argument("--clarity-min", type=int, default=6)
    parser.add_argument("--diversity-min", type=int, default=6)
    args = parser.parse_args()

    idea_records = {str(record["topic_id"]): record for record in _read_jsonl(args.ideas)}
    diversity_records = {
        str(record["topic_id"]): record for record in _read_jsonl(args.diversity)
    }
    quality_records = {str(record["topic_id"]): record for record in _read_jsonl(args.quality)}
    topics = sorted(idea_records)
    if set(topics) != set(diversity_records) or set(topics) != set(quality_records):
        raise SystemExit("Ideas, diversity, and quality inputs must contain identical topic IDs")

    selected_ideas = []
    selected_diversity = []
    selected_quality = []
    for topic_id in topics:
        ideas, diversity, quality = select_topic(
            idea_records[topic_id], diversity_records[topic_id], quality_records[topic_id],
            nb_min=args.nb_min,
            soundness_min=args.soundness_min,
            clarity_min=args.clarity_min,
            diversity_min=args.diversity_min,
        )
        selected_ideas.append(ideas)
        selected_diversity.append(diversity)
        selected_quality.append(quality)

    _write_jsonl(args.output_ideas, selected_ideas)
    _write_jsonl(args.output_diversity, selected_diversity)
    _write_jsonl(args.output_quality, selected_quality)
    print(
        f"Selected exactly one representative per fresh seed for {len(topics)} topic(s).\n"
        f"  ideas: {args.output_ideas}\n"
        f"  diversity: {args.output_diversity}\n"
        f"  quality: {args.output_quality}"
    )


if __name__ == "__main__":
    main()
