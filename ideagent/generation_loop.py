"""Fixed-budget scheduled search over free ideas, repairs, and accepted refinements.

Every ideator output consumes one global search step. The first phase forces free exploration;
later steps sample from state-dependent action weights, redistributing unavailable actions.
Acceptance uses hard diversity/non-obviousness/clarity/soundness gates, then a phase-dependent
weighted quality score for archive competition. Promising rejected lineages and accepted ideas
can each be revisited up to their configured limits without retaining old model conversations.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ideagent.agent_prompts import IDEATOR_SYSTEM_PROMPT, build_feedback_system_prompt
from ideagent.agents import CriticAgent, IdeatorAgent, QualityAgent, evaluate_soundness_multi
from ideagent.console import print_candidate_result
from ideagent.data import (
    build_background_context,
    load_raw_topic_groups,
    parse_paper,
    truncate_by_tokens_rough,
)
from ideagent.experiment_logging import (
    ExperimentLogger,
    set_trace_context,
    text_sha256,
    write_run_manifest,
    write_trace_summary,
)
from ideagent.diversity_judge import DIVERSITY_JUDGE_SYSTEM, DiversityVerdict, judge_diversity
from ideagent.quality_eval import SOUNDNESS_EVAL_SYSTEM_PROMPT
from ideagent.signature import extract_signature, signature_to_final_idea_core
from ideagent.archive import (
    ACCEPTED, KEEP_ACCEPTED, QUALIFIED_RETIRED, REPAIR_QUEUED, REPAIR_RETAINED,
    REJECT_DUPLICATE, REJECT_EVAL_FAIL, REJECT_UNSOUND, REPLACED,
    Candidate, QualityWeights, RefinementPriorityWeights, RepairPriorityWeights,
    Archive, Thresholds,
)
from ideagent.prompts import (
    CONSOLIDATION_RESPONSE_FORMAT, CONSOLIDATION_SYSTEM, build_consolidation_user,
    NOVELTY_BRANCH_CRITIC_SYSTEM, RELATIVE_NB_RESPONSE_FORMAT, RELATIVE_NB_SYSTEM,
    REPAIR_CRITIC_SYSTEM, build_relative_nb_user,
    build_repair_critic_user, op_critic_revision, op_free, op_pivot,
)
from ideagent.utils import append_jsonl, create_jsonl, prepare_output_dir, read_jsonl


@dataclass
class Config:
    input_jsonl: str
    output_dir: str
    max_background_tokens: int | None = None
    resume: bool = False    # False = explicit restart: wipes output_dir (60s warning first)
    init_topic_idx: int = 0  # 0-indexed: skip this many topics before starting
    end_topic_idx: int | None = None  # 0-indexed, exclusive: stop before this topic; None = run
                              # to the end of the input list. Combine with init_topic_idx for
                              # parallel runs each covering a disjoint slice of the same
                              # output_dir. final_idea_cores.jsonl is auto-suffixed by init_topic_idx
                              # (e.g. final_idea_cores_from5.jsonl) so slices never race a shared file.
    # Number of independent free seed generations. Repair/polish drafts are auxiliary and do
    # not consume this budget, but they are separately bounded and fully logged.
    max_budget: int = 10
    forced_exploration_steps: int = 0  # legacy scheduler compatibility; immediate mode is used
    # thresholds
    soundness_n: int = 5
    soundness_floor: float = 60.0
    soundness_fatal_threshold: int = 3
    # A lone severe vote creates a repairable dispute. Only this many severe votes create a
    # hard fatal consensus. None means a strict panel majority.
    soundness_fatal_min_votes: int | None = None
    nb_min: int = 60
    clarity_min: int = 60
    # Reporting threshold only: never used by acceptance, Q, repair admission, or replacement.
    feasibility_actionable_min: int = 60
    diversity_D: int = 55
    repair_margin: int = 20
    repair_diversity_D: int = 45
    # memory limits
    accepted_capacity: int = 10
    # Legacy scheduler compatibility only. Immediate-lineage processing does not stop or switch
    # behavior at this target; it always completes exactly max_budget free seeds.
    accepted_target: int | None = None
    repair_capacity: int = 1
    repair_attempt_limit: int = 1
    accepted_refinement_limit: int = 2
    # Shared across repair and accepted polishing. This—not the two individual limits added
    # together—bounds auxiliary ideator drafts for one free seed.
    auxiliary_attempt_limit: int = 2
    novelty_branch_limit: int = 0
    # Soft aspiration targets for immediately polishing an accepted seed. These are not gates;
    # acceptance continues to use nb_min/soundness_floor/clarity_min and never feasibility.
    press_nb_target: int = 90
    press_soundness_target: float = 80.0
    press_clarity_target: int = 80
    press_feasibility_target: int = 0
    rejected_pending_cap: int = 20
    consolidate_every: int = 12
    signature_char_cap: int = 240
    history_field_char_cap: int = 120
    # State-dependent schedule after forced exploration.
    before_full_free: float = 0.45
    before_full_repair: float = 0.35
    before_full_novelty_branch: float = 0.20
    before_full_accepted_refine: float = 0.0
    after_full_free: float = 0.15
    after_full_repair: float = 0.30
    after_full_novelty_branch: float = 0.35
    after_full_accepted_refine: float = 0.20
    # When repair is empty, its probability moves to novelty branch/free, never polishing.
    repair_fallback_branch_fraction: float = 0.70
    relative_nb_max_retries: int = 2
    # Gate-passing ideas are ranked by Q; early search favors hard-to-invent novelty.
    early_nb_weight: float = 0.70
    early_soundness_weight: float = 0.20
    early_clarity_weight: float = 0.10
    late_nb_weight: float = 0.40
    late_soundness_weight: float = 0.40
    late_clarity_weight: float = 0.20
    # Hierarchical within-pool selection priorities. The outer scheduler remains fixed.
    repair_priority_novelty_weight: float = 0.50
    repair_priority_closeness_weight: float = 0.35
    repair_priority_remaining_weight: float = 0.15
    repair_closeness_soundness_weight: float = 0.55
    repair_closeness_nb_weight: float = 0.15
    repair_closeness_clarity_weight: float = 0.30
    refinement_priority_novelty_weight: float = 0.60
    refinement_priority_headroom_weight: float = 0.25
    refinement_priority_remaining_weight: float = 0.15
    refinement_headroom_soundness_weight: float = 0.65
    refinement_headroom_clarity_weight: float = 0.35
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_budget < 1:
            raise ValueError("max_budget must be >= 1")
        if not 0 <= self.forced_exploration_steps <= self.max_budget:
            raise ValueError("forced_exploration_steps must be between 0 and max_budget")
        if self.soundness_n < 1:
            raise ValueError("soundness_n must be >= 1")
        for name in (
            "soundness_floor", "nb_min", "clarity_min", "diversity_D",
            "repair_diversity_D", "feasibility_actionable_min", "press_nb_target",
            "press_soundness_target", "press_clarity_target", "press_feasibility_target",
        ):
            if not 0 <= float(getattr(self, name)) <= 100:
                raise ValueError(f"{name} must be between 0 and 100")
        if not 0 <= self.soundness_fatal_threshold <= 9:
            raise ValueError("soundness_fatal_threshold must be between 0 and 9")
        if self.soundness_fatal_min_votes is None:
            self.soundness_fatal_min_votes = self.soundness_n // 2 + 1
        if not 1 <= self.soundness_fatal_min_votes <= self.soundness_n:
            raise ValueError("soundness_fatal_min_votes must be between 1 and soundness_n")
        if self.repair_margin < 0:
            raise ValueError("repair_margin must be >= 0")
        if self.accepted_capacity < 1 or self.repair_capacity < 1 or self.rejected_pending_cap < 1:
            raise ValueError("accepted, repair, and rejected capacities must be positive")
        if (
            self.accepted_target is not None
            and not 1 <= self.accepted_target <= self.accepted_capacity
        ):
            raise ValueError("accepted_target must be between 1 and accepted_capacity")
        if (
            self.repair_attempt_limit < 1
            or self.accepted_refinement_limit < 0
            or self.auxiliary_attempt_limit < 0
            or self.novelty_branch_limit < 0
        ):
            raise ValueError(
                "repair_attempt_limit must be positive; auxiliary, accepted-refinement, and "
                "novelty-branch limits may be zero"
            )
        if self.consolidate_every < 1:
            raise ValueError("consolidate_every must be >= 1")
        if self.signature_char_cap < 16:
            raise ValueError("signature_char_cap must be >= 16")
        if self.history_field_char_cap < 32:
            raise ValueError("history_field_char_cap must be >= 32")
        for name, values in {
            "before_full": (
                self.before_full_free, self.before_full_repair,
                self.before_full_novelty_branch, self.before_full_accepted_refine,
            ),
            "after_full": (
                self.after_full_free, self.after_full_repair,
                self.after_full_novelty_branch, self.after_full_accepted_refine,
            ),
        }.items():
            if any(v < 0 for v in values) or abs(sum(values) - 1.0) > 1e-6:
                raise ValueError(f"{name} schedule probabilities must be non-negative and sum to 1")
        if not 0 <= self.repair_fallback_branch_fraction <= 1:
            raise ValueError("repair_fallback_branch_fraction must be between 0 and 1")
        if self.relative_nb_max_retries < 0:
            raise ValueError("relative_nb_max_retries must be non-negative")
        self.early_weights()
        self.late_weights()
        self.repair_priority_weights()
        self.refinement_priority_weights()
        if self.init_topic_idx < 0:
            raise ValueError("init_topic_idx must be >= 0")
        if self.end_topic_idx is not None and self.end_topic_idx <= self.init_topic_idx:
            raise ValueError("end_topic_idx must be > init_topic_idx")

    def thresholds(self) -> Thresholds:
        return Thresholds(
            soundness_floor=self.soundness_floor,
            soundness_fatal_threshold=self.soundness_fatal_threshold,
            soundness_fatal_min_votes=int(self.soundness_fatal_min_votes),
            nb_min=self.nb_min, clarity_min=self.clarity_min,
            diversity_D=self.diversity_D, repair_margin=self.repair_margin,
            repair_diversity_D=self.repair_diversity_D,
        )

    def early_weights(self) -> QualityWeights:
        return QualityWeights(
            self.early_nb_weight, self.early_soundness_weight, self.early_clarity_weight,
        )

    def late_weights(self) -> QualityWeights:
        return QualityWeights(
            self.late_nb_weight, self.late_soundness_weight, self.late_clarity_weight,
        )

    def repair_priority_weights(self) -> RepairPriorityWeights:
        return RepairPriorityWeights(
            novelty=self.repair_priority_novelty_weight,
            gate_closeness=self.repair_priority_closeness_weight,
            remaining_attempts=self.repair_priority_remaining_weight,
            soundness_deficit=self.repair_closeness_soundness_weight,
            non_obviousness_deficit=self.repair_closeness_nb_weight,
            clarity_deficit=self.repair_closeness_clarity_weight,
        )

    def refinement_priority_weights(self) -> RefinementPriorityWeights:
        return RefinementPriorityWeights(
            novelty=self.refinement_priority_novelty_weight,
            repair_headroom=self.refinement_priority_headroom_weight,
            remaining_attempts=self.refinement_priority_remaining_weight,
            soundness_headroom=self.refinement_headroom_soundness_weight,
            clarity_headroom=self.refinement_headroom_clarity_weight,
        )


def _score_map(rubric_assessment: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rubric_assessment or []:
        rid = r.get("id")
        if rid:
            out[rid] = max(0, min(100, int(r.get("score", 0))))
    return out


def _soundness_panel_status(
    samples: list[int], *, fatal_threshold: int, fatal_min_votes: int,
) -> tuple[bool, bool, int]:
    """Return (hard_fatal, disputed, severe_vote_count).

    A partial/empty panel still fails closed through eval_ok; an empty panel is also marked fatal.
    A non-consensus severe objection is preserved as a repairable dispute rather than ignored or
    allowed directly through the acceptance gate.
    """

    if not samples:
        return True, False, 0
    votes = sum(score <= fatal_threshold for score in samples)
    return votes >= fatal_min_votes, 0 < votes < fatal_min_votes, votes


def _sig_to_core(cand: Candidate) -> dict[str, Any]:
    return signature_to_final_idea_core(
        cand.signature, idea_text=cand.idea_text, episode_id=cand.id
    )


def _candidate_metrics(cand: Candidate, weights: QualityWeights) -> dict[str, Any]:
    return {
        "non_obviousness": cand.non_obviousness,
        "mechanism_clarity": cand.mechanism_clarity,
        "soundness_100": round(cand.soundness_100, 3),
        "feasibility": cand.feasibility,
        "diversity_score": cand.diversity_score,
        "weighted_quality": round(cand.weighted_quality(weights), 6),
    }


def _compact_archive_state(archive: Archive) -> dict[str, Any]:
    return {
        **archive.stats(),
        "search_steps_completed": archive.search_steps_completed,
        "action_counts": dict(archive.action_counts),
        "accepted_ids": list(archive.accepted),
        "repair_ids": [cand.id for cand in archive.repair_archive],
        "retired_qualified_ids": [str(item.get("id", "")) for item in archive.retired_qualified],
        "qualified_lineage_ids": sorted(archive.qualified_lineages),
        "rejected_pending_count": len(archive.rejected_pending),
        "rejected_pending": list(archive.rejected_pending),
        "rejected_families_summary": archive.rejected_families_summary,
        "generation_memory_summary": archive.generation_memory_summary(),
    }


def _archive_transition(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def changes(key: str) -> tuple[list[str], list[str]]:
        old, new = set(before.get(key, [])), set(after.get(key, []))
        return sorted(new - old), sorted(old - new)

    accepted_added, accepted_removed = changes("accepted_ids")
    repair_added, repair_removed = changes("repair_ids")
    qualified_added, qualified_removed = changes("qualified_lineage_ids")
    retired_added, retired_removed = changes("retired_qualified_ids")
    return {
        "accepted_added": accepted_added,
        "accepted_removed": accepted_removed,
        "repair_added": repair_added,
        "repair_removed": repair_removed,
        "qualified_lineages_added": qualified_added,
        "qualified_lineages_removed": qualified_removed,
        "retired_qualified_added": retired_added,
        "retired_qualified_removed": retired_removed,
        "active_yield_delta": int(after.get("active_yield", 0)) - int(before.get("active_yield", 0)),
        "discovery_yield_delta": int(after.get("discovery_yield", 0)) - int(before.get("discovery_yield", 0)),
        "rejected_pending_delta": (
            int(after.get("rejected_pending_count", 0))
            - int(before.get("rejected_pending_count", 0))
        ),
        "consolidated_summary_changed": (
            before.get("rejected_families_summary") != after.get("rejected_families_summary")
        ),
        "generation_memory_changed": (
            before.get("generation_memory_summary") != after.get("generation_memory_summary")
        ),
    }


def _analysis_summary(
    *, topic_id: str, rows: list[dict[str, Any]], archive: Archive
) -> dict[str, Any]:
    by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    revision_effects: list[dict[str, Any]] = []
    acceptance_timeline: list[dict[str, Any]] = []
    accepted_lifecycle: dict[str, dict[str, Any]] = {}

    for row in rows:
        by_lineage[str(row["lineage_id"])].append(row)
        by_operator[str(row["operator"])].append(row)
        if row.get("parent_id"):
            revision_effects.append({
                "step": row["gen"],
                "operator": row["operator"],
                "lineage_id": row["lineage_id"],
                "parent_id": row["parent_id"],
                "child_id": row["id"],
                "before": row.get("parent_metrics"),
                "after": row.get("metrics"),
                "delta": row.get("metric_delta_from_parent"),
                "outcome": row["outcome"],
                "archive_transition": row.get("archive_transition"),
            })
        transition = row.get("archive_transition") or {}
        if transition.get("accepted_added") or transition.get("accepted_removed"):
            change = {
                "step": row["gen"],
                "candidate_id": row["id"],
                "operator": row["operator"],
                "outcome": row["outcome"],
                "accepted_added": transition.get("accepted_added", []),
                "accepted_removed": transition.get("accepted_removed", []),
                "active_yield": row["archive_after"]["active_yield"],
            }
            acceptance_timeline.append(change)
            for accepted_id in transition.get("accepted_added", []):
                lifecycle = accepted_lifecycle.setdefault(accepted_id, {})
                lifecycle.update({
                    "idea_id": accepted_id,
                    "entered_at_step": row["gen"],
                    "entered_by_operator": row["operator"],
                    "entered_with_outcome": row["outcome"],
                    "lineage_id": row["lineage_id"] if accepted_id == row["id"] else None,
                })
            for removed_id in transition.get("accepted_removed", []):
                if row["operator"] == "accepted_refine" and row["outcome"] == REPLACED:
                    exit_reason = "accepted_refinement_replaced"
                elif removed_id in row.get("duplicate_ids", []):
                    exit_reason = "active_duplicate_replaced"
                else:
                    exit_reason = "active_capacity_replaced"
                lifecycle = accepted_lifecycle.setdefault(removed_id, {"idea_id": removed_id})
                lifecycle.update({
                    "exited_at_step": row["gen"],
                    "exit_reason": exit_reason,
                    "replaced_by": (
                        row["id"] if row["id"] in transition.get("accepted_added", []) else None
                    ),
                })

    metric_names = (
        "non_obviousness", "mechanism_clarity", "soundness_100",
        "feasibility", "diversity_score", "weighted_quality",
    )
    operator_summary: dict[str, Any] = {}
    for operator, operator_rows in by_operator.items():
        operator_summary[operator] = {
            "count": len(operator_rows),
            "accepted_or_replaced": sum(
                row["outcome"] in {ACCEPTED, REPLACED} for row in operator_rows
            ),
            "mean_metrics": {
                name: round(
                    sum(float(row["metrics"][name]) for row in operator_rows) / len(operator_rows),
                    6,
                )
                for name in metric_names
            },
        }

    lineage_trajectories: dict[str, Any] = {}
    final_active_ids = set(archive.accepted)
    for lineage_id, lineage_rows in by_lineage.items():
        ordered = sorted(lineage_rows, key=lambda row: int(row["gen"]))
        same_lineage_revisions = sum(
            row.get("operator") in {"repair", "accepted_refine"} for row in ordered
        )
        lineage_trajectories[lineage_id] = {
            "n_candidates": len(ordered),
            "n_revisions": same_lineage_revisions,
            "n_same_lineage_revisions": same_lineage_revisions,
            "n_parent_informed_novelty_branches": sum(
                row.get("operator") == "novelty_branch" for row in ordered
            ),
            "ever_qualified": lineage_id in archive.qualified_lineages,
            "final_active": any(row["id"] in final_active_ids for row in ordered),
            "steps": [
                {
                    "step": row["gen"], "candidate_id": row["id"],
                    "parent_id": row.get("parent_id"), "revision": row["revision"],
                    "operator": row["operator"], "outcome": row["outcome"],
                    "metrics": row["metrics"],
                }
                for row in ordered
            ],
        }

    outcomes = Counter(str(row["outcome"]) for row in rows)
    rubric_ids = sorted({
        str(rubric.get("id"))
        for row in rows
        for rubric in row.get("rubric_assessment", [])
        if rubric.get("id")
    })
    score_series: dict[str, list[dict[str, Any]]] = {
        metric: [
            {"step": row["gen"], "score": float(row["metrics"][metric])}
            for row in rows
        ]
        for metric in metric_names
    }
    for rubric_id in rubric_ids:
        series = []
        for row in rows:
            scores = _score_map(row.get("rubric_assessment", []))
            if rubric_id in scores:
                series.append({"step": row["gen"], "score": scores[rubric_id]})
        score_series[f"rubric:{rubric_id}"] = series

    def trend(series: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(point["score"]) for point in series]
        if not values:
            return {"n": 0, "series": []}
        split = max(1, len(values) // 2)
        first = values[:split]
        second = values[split:] or values[-1:]
        xs = [float(point["step"]) for point in series]
        xbar, ybar = sum(xs) / len(xs), sum(values) / len(values)
        denom = sum((x - xbar) ** 2 for x in xs)
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, values)) / denom if denom else 0.0
        return {
            "n": len(values),
            "first_half_mean": round(sum(first) / len(first), 6),
            "second_half_mean": round(sum(second) / len(second), 6),
            "second_minus_first": round(
                sum(second) / len(second) - sum(first) / len(first), 6
            ),
            "ols_slope_per_search_step": round(slope, 6),
            "series": series,
        }

    lower_is_better_trends = {
        "rubric:prior_episode_overlap", "rubric:source_boundedness/narrowness",
    }
    metric_and_rubric_trends = {}
    for name, series in score_series.items():
        summary = trend(series)
        summary["direction"] = "lower_is_better" if name in lower_is_better_trends else "higher_is_better"
        metric_and_rubric_trends[name] = summary
    gate_names = sorted({
        gate_name for row in rows for gate_name in row.get("gate_results", {})
    })
    gate_pass_counts = {
        gate_name: sum(row.get("gate_results", {}).get(gate_name) is True for row in rows)
        for gate_name in gate_names
    }
    cumulative_progress: list[dict[str, Any]] = []
    running_metric_sums = {name: 0.0 for name in metric_names}
    running_outcomes: Counter[str] = Counter()
    running_gates: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        for name in metric_names:
            running_metric_sums[name] += float(row["metrics"][name])
        running_outcomes[str(row["outcome"])] += 1
        for gate_name, passed in row.get("gate_results", {}).items():
            if passed is True:
                running_gates[gate_name] += 1
        cumulative_progress.append({
            "step": row["gen"],
            "mean_metrics_to_date": {
                name: round(running_metric_sums[name] / index, 6) for name in metric_names
            },
            "gate_passes_to_date": dict(running_gates),
            "outcomes_to_date": dict(running_outcomes),
            "active_yield": row["archive_after"]["active_yield"],
            "discovery_yield": row["archive_after"]["discovery_yield"],
            "active_actionable_yield": row["archive_after"]["active_actionable_yield"],
            "discovery_actionable_yield": row["archive_after"]["discovery_actionable_yield"],
        })

    accepted_lifecycle_rows = sorted(
        accepted_lifecycle.values(), key=lambda item: (item.get("entered_at_step", 10**9), item["idea_id"])
    )
    modified_reasons = {"accepted_refinement_replaced", "active_duplicate_replaced"}
    relative_nb_rows = [
        row["relative_nb_comparison"] for row in rows
        if (row.get("relative_nb_comparison") or {}).get("performed")
    ]
    return {
        "topic_id": topic_id,
        "search_steps": len(rows),  # backward-compatible: total evaluated drafts
        "free_seed_budget_used": sum(
            row.get("counts_toward_free_budget") is True for row in rows
        ),
        "total_candidates_evaluated": len(rows),
        "auxiliary_candidates_evaluated": sum(
            row.get("counts_toward_free_budget") is not True for row in rows
        ),
        "outcome_counts": dict(outcomes),
        "operator_counts": dict(Counter(str(row["operator"]) for row in rows)),
        "repair_attempts": sum(row["operator"] == "repair" for row in rows),
        "repair_conversions": sum(
            row["operator"] == "repair" and row["outcome"] in {ACCEPTED, REPLACED}
            for row in rows
        ),
        "novelty_branch_attempts": sum(
            row["operator"] == "novelty_branch" for row in rows
        ),
        "novelty_branch_discoveries": sum(
            row["operator"] == "novelty_branch"
            and row["outcome"] in {ACCEPTED, REPLACED, QUALIFIED_RETIRED}
            for row in rows
        ),
        "accepted_refinement_attempts": sum(
            row["operator"] == "accepted_refine" for row in rows
        ),
        "accepted_refinement_replacements": sum(
            row["operator"] == "accepted_refine" and row["outcome"] == REPLACED
            for row in rows
        ),
        "accepted_refinement_kept_parent": sum(
            row["operator"] == "accepted_refine" and row["outcome"] == KEEP_ACCEPTED
            for row in rows
        ),
        "accepted_refinement_relative_nb": {
            "comparisons_performed": len(relative_nb_rows),
            "child_wins": sum(row.get("outcome") == "child" for row in relative_nb_rows),
            "ties": sum(row.get("outcome") == "tie" for row in relative_nb_rows),
            "parent_wins": sum(row.get("outcome") == "parent" for row in relative_nb_rows),
            "position_consistent": sum(
                row.get("order_consistent") is True for row in relative_nb_rows
            ),
            "parse_failures": sum(
                row.get("parse_failed") is True for row in relative_nb_rows
            ),
        },
        "soundness_panel": {
            "fatal_consensus_candidates": sum(row.get("soundness_fatal") is True for row in rows),
            "disputed_candidates": sum(row.get("soundness_disputed") is True for row in rows),
        },
        "accepted_modifications": [
            effect for effect in revision_effects
            if effect["operator"] == "accepted_refine" and effect["outcome"] == REPLACED
        ],
        "accepted_archive_analysis": {
            "n_accepted_entries_ever": len(accepted_lifecycle_rows),
            "n_accepted_later_removed": sum(
                "exited_at_step" in item for item in accepted_lifecycle_rows
            ),
            "n_accepted_modified_or_superseded_same_mechanism": sum(
                item.get("exit_reason") in modified_reasons for item in accepted_lifecycle_rows
            ),
            "n_accepted_capacity_displaced": sum(
                item.get("exit_reason") == "active_capacity_replaced"
                for item in accepted_lifecycle_rows
            ),
            "n_final_active": len(archive.accepted),
            "lifecycles": accepted_lifecycle_rows,
        },
        "revision_effects": revision_effects,
        "acceptance_timeline": acceptance_timeline,
        "gate_pass_counts": gate_pass_counts,
        "metric_and_rubric_trends": metric_and_rubric_trends,
        "cumulative_progress": cumulative_progress,
        "operator_metric_summary": operator_summary,
        "lineage_trajectories": lineage_trajectories,
        "per_step_metrics": [
            {
                "step": row["gen"], "candidate_id": row["id"],
                "seed_index": row.get("seed_index"),
                "counts_toward_free_budget": row.get("counts_toward_free_budget"),
                "lineage_id": row["lineage_id"], "revision": row["revision"],
                "operator": row["operator"], "outcome": row["outcome"],
                "metrics": row["metrics"], "gate_results": row["gate_results"],
                "active_yield": row["archive_after"]["active_yield"],
                "discovery_yield": row["archive_after"]["discovery_yield"],
            }
            for row in rows
        ],
        "final_archive_stats": archive.stats(),
        "final_active_ids": list(archive.accepted),
    }


class GenerationLoop:
    def __init__(
        self, *, config: Config, ideator: IdeatorAgent, critic: CriticAgent,
        quality: QualityAgent,
        steno_client: Any, steno_gen_kwargs: dict[str, Any],
        judge_client: Any, judge_gen_kwargs: dict[str, Any],
        experiment_logger: ExperimentLogger | None = None,
    ) -> None:
        self.c = config
        self.ideator = ideator
        self.critic = critic
        self.quality = quality
        self.steno_client = steno_client
        self.steno_gen_kwargs = steno_gen_kwargs
        self.judge_client = judge_client
        self.judge_gen_kwargs = judge_gen_kwargs
        self.experiment_logger = experiment_logger
        self.last_lineage_representatives: list[Candidate] = []
        self.last_representative_outcomes: dict[int, str] = {}

    # ── one candidate: signature + quality + soundness (fail closed) ──
    def _evaluate(
        self, idea_text: str, *, idea_id: str, gen: int, operator: str,
        topic_id: str, background: str, parent_id: str | None = None,
        lineage_id: str = "", revision: int = 0, repair_attempts: int = 0,
        accepted_refinements_used: int = 0, auxiliary_attempts_used: int = 0,
        seed_index: int = 0,
    ) -> Candidate:
        set_trace_context(
            self.steno_client, topic_id=topic_id, candidate_id=idea_id,
            search_step=gen, operator=operator, stage="signature_extraction",
        )
        sig = extract_signature(
            idea_text, idea_id=idea_id, client=self.steno_client,
            gen_kwargs=self.steno_gen_kwargs, char_cap=self.c.signature_char_cap,
        )
        set_trace_context(
            self.quality.client, topic_id=topic_id, candidate_id=idea_id,
            search_step=gen, operator=operator, stage="quality_rubrics",
        )
        self.quality.reset_for_topic(prior_final_idea_cores=[])
        self.quality.set_log_context(topic_id=topic_id, target_paper_id=idea_id)
        vq = self.quality.evaluate(prev_critic_turn="", latest_ideator_turn=idea_text, current_final_idea_core=None)
        rubrics = vq.get("rubric_assessment") or []
        scores = _score_map(rubrics)
        feedback_notes = []
        for rubric in rubrics:
            rid = str(rubric.get("id", "")).strip()
            if rid not in {"non_obviousness", "mechanism_clarity", "feasibility"}:
                continue
            evidence = str(rubric.get("evidence", "")).strip()
            pressure = str(rubric.get("critic_pressure", "")).strip()
            note = f"{rid}={scores.get(rid, 0)}"
            if evidence:
                note += f"; evidence: {evidence}"
            if pressure:
                note += f"; requested repair: {pressure}"
            feedback_notes.append(note)

        # Topic-wide (not per-candidate id) so the static SOUNDNESS_EVAL_SYSTEM_PROMPT
        # prefix actually gets reused across every candidate generated for this topic.
        soundness_cache_key = hashlib.sha256(f"{topic_id}/soundness".encode()).hexdigest()
        set_trace_context(
            self.quality.client, topic_id=topic_id, candidate_id=idea_id,
            search_step=gen, operator=operator, stage="soundness_panel",
        )
        sresults = evaluate_soundness_multi(
            idea_text, client=self.quality.client, gen_kwargs=self.quality.gen_kwargs, n=self.c.soundness_n,
            prompt_cache_key=soundness_cache_key,
        )
        ssamples = [r["score"] for r in sresults]
        smean = mean(ssamples) if ssamples else 0.0
        s100 = smean * 100.0 / 9.0
        fatal, disputed, fatal_votes = _soundness_panel_status(
            ssamples,
            fatal_threshold=self.c.soundness_fatal_threshold,
            fatal_min_votes=int(self.c.soundness_fatal_min_votes),
        )
        worst = sorted(sresults, key=lambda r: r["score"])[:2]
        rationale = worst[0]["rationale"] if worst else ""
        rationales = [str(r.get("rationale", "")).strip() for r in worst if str(r.get("rationale", "")).strip()]

        # Partial panels/rubric parses are not trustworthy enough for threshold decisions.
        # Feasibility is retained for analysis but is not an acceptance gate. A missing optional
        # feasibility score must not fail-close an otherwise complete gate evaluation.
        required_scores = {"non_obviousness", "mechanism_clarity"}
        eval_ok = (
            sig is not None
            and len(ssamples) == self.c.soundness_n
            and required_scores.issubset(scores)
        )
        return Candidate(
            id=idea_id, idea_text=idea_text,
            signature=sig if sig is not None else _placeholder_sig(idea_id),
            soundness_100=s100, soundness_samples=ssamples, soundness_fatal=fatal,
            soundness_fatal_votes=fatal_votes, soundness_disputed=disputed,
            soundness_rationale=rationale, soundness_rationales=rationales,
            soundness_panel_results=sresults,
            non_obviousness=scores.get("non_obviousness", 0),
            mechanism_clarity=scores.get("mechanism_clarity", 0),
            feasibility=scores.get("feasibility", 0),
            generation=gen, operator=operator, seed_index=seed_index,
            feedback_notes=feedback_notes,
            parent_id=parent_id, lineage_id=lineage_id or idea_id, revision=revision,
            repair_attempts=repair_attempts,
            accepted_refinements_used=accepted_refinements_used, eval_ok=eval_ok,
            auxiliary_attempts_used=auxiliary_attempts_used,
            rubric_assessment=rubrics,
        )

    def _judge(
        self, cand: Candidate, archive: Archive, *, topic_id: str,
        exclude_accepted_id: str | None = None,
        exclude_historical_lineage_id: str | None = None,
    ) -> DiversityVerdict:
        if not cand.eval_ok:
            return DiversityVerdict(parsed=False)
        set_trace_context(
            self.judge_client, topic_id=topic_id, candidate_id=cand.id,
            search_step=cand.generation, operator=cand.operator, stage="diversity_judgment",
        )
        judge_cache_key = hashlib.sha256(f"{topic_id}/judge".encode()).hexdigest()
        return judge_diversity(
            cand.signature, accepted=archive.accepted_signatures(exclude_id=exclude_accepted_id),
            historical=archive.historical_signatures(
                exclude_lineage_id=exclude_historical_lineage_id,
            ),
            rejected_summary=archive.rejected_summary(),
            historical_field_char_cap=self.c.history_field_char_cap,
            client=self.judge_client, gen_kwargs=self.judge_gen_kwargs,
            prompt_cache_key=judge_cache_key,
        )

    def _relative_nb_comparison(
        self, *, parent: Candidate, child: Candidate, topic_id: str,
    ) -> dict[str, Any]:
        """Position-balanced NB non-regression check for a replaceable polish child."""

        judgments: list[dict[str, Any]] = []
        for orientation in ("parent_first", "child_first"):
            idea_a, idea_b = (
                (parent.idea_text, child.idea_text)
                if orientation == "parent_first"
                else (child.idea_text, parent.idea_text)
            )
            parsed: dict[str, Any] | None = None
            raw = ""
            attempts = 0
            for attempts in range(1, self.c.relative_nb_max_retries + 2):
                set_trace_context(
                    self.quality.client,
                    topic_id=topic_id, candidate_id=child.id,
                    search_step=child.generation, operator=child.operator,
                    parent_id=parent.id, lineage_id=parent.lineage_id,
                    stage=f"relative_nb_{orientation}",
                )
                raw = self.quality.client.generate(
                    [
                        {"role": "system", "content": RELATIVE_NB_SYSTEM},
                        {"role": "user", "content": build_relative_nb_user(idea_a, idea_b)},
                    ],
                    response_format=RELATIVE_NB_RESPONSE_FORMAT,
                    **self.quality.gen_kwargs,
                )
                start, end = (raw or "").find("{"), (raw or "").rfind("}")
                if start < 0 or end < start:
                    continue
                try:
                    obj = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    continue
                winner = str(obj.get("winner", "")).strip()
                confidence = str(obj.get("confidence", "")).strip().lower()
                rationale = str(obj.get("rationale", "")).strip()
                if winner in {"A", "B", "tie"} and confidence in {
                    "low", "medium", "high"
                } and rationale:
                    parsed = {
                        "winner": winner, "confidence": confidence,
                        "rationale": rationale,
                    }
                    break
            if parsed is None:
                # Fail closed for archive replacement, but preserve both complete raw calls on disk.
                canonical = "parent"
                parsed = {"winner": "parse_fail", "confidence": "low", "rationale": ""}
            elif parsed["winner"] == "tie":
                canonical = "tie"
            elif orientation == "parent_first":
                canonical = "parent" if parsed["winner"] == "A" else "child"
            else:
                canonical = "child" if parsed["winner"] == "A" else "parent"
            judgments.append({
                "orientation": orientation,
                "parsed": parsed["winner"] != "parse_fail",
                "raw_winner": parsed["winner"],
                "canonical_winner": canonical,
                "confidence": parsed["confidence"],
                "rationale": parsed["rationale"],
                "raw_response": raw,
                "parse_attempts": attempts,
            })

        canonical = [judgment["canonical_winner"] for judgment in judgments]
        order_consistent = canonical[0] == canonical[1]
        parse_failed = any(not judgment["parsed"] for judgment in judgments)
        # Valid position disagreement is a conservative tie. A malformed guard call cannot
        # authorize destructive archive replacement, so parsing failure keeps the parent.
        outcome = "parent" if parse_failed else (canonical[0] if order_consistent else "tie")
        return {
            "performed": True,
            "outcome": outcome,
            "child_not_worse": outcome != "parent",
            "order_consistent": order_consistent,
            "parse_failed": parse_failed,
            "judgments": judgments,
        }

    def _weights_for_seed(self, seed_index: int) -> QualityWeights:
        """Use the preregistered early/late Q schedule without an archive-size target."""

        return (
            self.c.early_weights()
            if seed_index <= max(1, self.c.max_budget // 2)
            else self.c.late_weights()
        )

    def _press_deficits(self, candidate: Candidate) -> list[str]:
        deficits: list[str] = []
        if candidate.non_obviousness < self.c.press_nb_target:
            deficits.append(
                f"non-obviousness {candidate.non_obviousness}<{self.c.press_nb_target}"
            )
        if candidate.soundness_100 < self.c.press_soundness_target:
            deficits.append(
                f"soundness {candidate.soundness_100:.1f}<{self.c.press_soundness_target:.1f}"
            )
        if candidate.mechanism_clarity < self.c.press_clarity_target:
            deficits.append(
                f"mechanism clarity {candidate.mechanism_clarity}<{self.c.press_clarity_target}"
            )
        if (
            self.c.press_feasibility_target > 0
            and candidate.feasibility < self.c.press_feasibility_target
        ):
            deficits.append(
                f"non-gating feasibility {candidate.feasibility}<{self.c.press_feasibility_target}"
            )
        return deficits

    def _needs_immediate_polish(self, candidate: Candidate) -> bool:
        return (
            candidate.accepted_refinements_used < self.c.accepted_refinement_limit
            and candidate.auxiliary_attempts_used < self.c.auxiliary_attempt_limit
            and bool(self._press_deficits(candidate))
        )

    def _revision_feedback_notes(self, candidate: Candidate) -> list[str]:
        """Keep feasibility evidence in logs, but hide it from critics unless opted in."""
        if self.c.press_feasibility_target > 0:
            return list(candidate.feedback_notes)
        return [
            note for note in candidate.feedback_notes
            if not note.lstrip().lower().startswith("feasibility=")
        ]

    def _fresh_directive(
        self, last_outcome: str | None, last_verdict: DiversityVerdict | None,
        archive: Archive,
    ) -> tuple[str, str]:
        acc, rej = archive.accepted_signatures(), archive.generation_memory_summary()
        if last_outcome == REJECT_DUPLICATE:
            stuck = last_verdict.rejected_family if last_verdict else ""
            return "pivot", op_pivot(acc, rej, stuck or "a mechanism already covered")
        if last_outcome == REJECT_UNSOUND and last_verdict and last_verdict.matches_rejected_family:
            return "pivot", op_pivot(acc, rej, last_verdict.rejected_family or "a known-bad family")
        return "free", op_free(acc, rej)

    def _maybe_consolidate(self, archive: Archive) -> None:
        if not archive.needs_consolidation(self.c.consolidate_every):
            return
        user = build_consolidation_user(archive.rejected_families_summary, archive.rejected_pending)
        raw = self.judge_client.generate(
            [{"role": "system", "content": CONSOLIDATION_SYSTEM}, {"role": "user", "content": user}],
            response_format=CONSOLIDATION_RESPONSE_FORMAT, **self.judge_gen_kwargs,
        )
        import re
        m = re.search(r"\{.*\}", raw or "", re.S)
        summary = ""
        if m:
            try:
                summary = str(json.loads(m.group(0)).get("summary", "")).strip()
            except json.JSONDecodeError:
                summary = ""
        if summary:
            archive.set_consolidated_summary(summary)

    @staticmethod
    def _write_candidate(
        cand_log: Any, cand: Candidate, verdict: DiversityVerdict, outcome: str,
        *, weights: QualityWeights, critic_challenge: str = "",
        parent_selection_priority: float | None = None,
        parent: Candidate | None = None,
        directive: str = "",
        relative_nb_comparison: dict[str, Any] | None = None,
        scheduler_decision: dict[str, Any] | None = None,
        memory_before: dict[str, Any] | None = None,
        archive_before: dict[str, Any] | None = None,
        archive_after: dict[str, Any] | None = None,
        thresholds: Thresholds | None = None,
    ) -> dict[str, Any]:
        metrics = _candidate_metrics(cand, weights)
        parent_metrics = _candidate_metrics(parent, weights) if parent is not None else None
        metric_delta = (
            {
                key: round(float(metrics[key]) - float(parent_metrics[key]), 6)
                for key in metrics
            }
            if parent_metrics is not None else None
        )
        gate_results = {
            "eval_ok": cand.eval_ok,
            "non_obviousness_pass": (
                cand.non_obviousness >= thresholds.nb_min if thresholds else None
            ),
            "mechanism_clarity_pass": (
                cand.mechanism_clarity >= thresholds.clarity_min if thresholds else None
            ),
            "soundness_pass": cand.passes_soundness(thresholds) if thresholds else None,
            "soundness_not_fatal": not cand.soundness_fatal,
            "soundness_disputed": cand.soundness_disputed,
            "soundness_fatal_votes": cand.soundness_fatal_votes,
            "soundness_fatal_min_votes": (
                thresholds.soundness_fatal_min_votes if thresholds else None
            ),
            "quality_pass": cand.passes_quality(thresholds) if thresholds else None,
            "all_quality_gates_pass": cand.passes_all(thresholds) if thresholds else None,
            "diversity_parsed": verdict.parsed,
            "not_duplicate": (not verdict.is_duplicate) if verdict.parsed else None,
            "diversity_pass": (
                verdict.parsed and thresholds is not None
                and verdict.diversity_score >= thresholds.diversity_D
            ),
            "feasibility_actionable": (
                cand.feasibility
                >= int((archive_after or {}).get("feasibility_actionable_min", 60))
            ),
            "repair_eligible": (
                cand.near_thresholds(thresholds) and verdict.parsed
                and not verdict.is_duplicate
                and verdict.diversity_score >= thresholds.repair_diversity_D
                if thresholds else None
            ),
        }
        gate_results["full_acceptance_gate_pass"] = all(
            gate_results.get(name) is True
            for name in (
                "eval_ok", "non_obviousness_pass", "mechanism_clarity_pass",
                "soundness_pass", "diversity_parsed", "not_duplicate", "diversity_pass",
            )
        )
        before = archive_before or {}
        after = archive_after or {}
        row = {
            "id": cand.id, "gen": cand.generation, "revision": cand.revision,
            "seed_index": cand.seed_index,
            "counts_toward_free_budget": cand.operator in {"free", "pivot"},
            "parent_id": cand.parent_id, "lineage_id": cand.lineage_id,
            "operator": cand.operator, "outcome": outcome,
            "repair_attempts": cand.repair_attempts,
            "accepted_refinements_used": cand.accepted_refinements_used,
            "auxiliary_attempts_used": cand.auxiliary_attempts_used,
            "novelty_branches_used": cand.novelty_branches_used,
            "eval_ok": cand.eval_ok,
            "non_obviousness": cand.non_obviousness,
            "mechanism_clarity": cand.mechanism_clarity,
            "soundness_100": round(cand.soundness_100, 1),
            "soundness_samples": cand.soundness_samples,
            "soundness_fatal": cand.soundness_fatal,
            "soundness_fatal_votes": cand.soundness_fatal_votes,
            "soundness_disputed": cand.soundness_disputed,
            "soundness_rationale": cand.soundness_rationale,
            "soundness_rationales": cand.soundness_rationales,
            "soundness_panel_results": cand.soundness_panel_results,
            "rubric_assessment": cand.rubric_assessment,
            "feedback_notes": cand.feedback_notes,
            "feasibility": cand.feasibility,
            "feasibility_delta_from_parent": cand.feasibility_delta_from_parent,
            "feasibility_pressure": cand.feasibility_pressure,
            "diversity_parsed": verdict.parsed,
            "diversity_score": verdict.diversity_score,
            "duplicate_ids": verdict.duplicate_ids,
            "historical_duplicate_ids": verdict.historical_duplicate_ids,
            "closest": verdict.closest,
            "matches_rejected_family": verdict.matches_rejected_family,
            "rejected_family": verdict.rejected_family,
            "defect": cand.defect,
            "ranking_weights": {
                "non_obviousness": weights.non_obviousness,
                "soundness": weights.soundness,
                "clarity": weights.clarity,
            },
            "weighted_quality": round(cand.weighted_quality(weights), 3),
            "metrics": metrics,
            "parent_metrics": parent_metrics,
            "metric_delta_from_parent": metric_delta,
            "relative_nb_comparison": relative_nb_comparison or {"performed": False},
            "gate_results": gate_results,
            "parent_selection_priority": (
                round(parent_selection_priority, 6)
                if parent_selection_priority is not None else None
            ),
            "critic_challenge": critic_challenge,
            "generation_directive": directive,
            "scheduler_decision": scheduler_decision or {},
            "memory_before": memory_before or {},
            "archive_before": before,
            "archive_after": after,
            "archive_transition": _archive_transition(before, after),
            "thresholds": asdict(thresholds) if thresholds is not None else {},
            "parent_full_record": (
                Archive._candidate_json(parent, include_text=True)
                if parent is not None else None
            ),
            "signature": cand.signature.to_json(), "idea_text": cand.idea_text,
        }
        cand_log.write(json.dumps(row, ensure_ascii=False) + "\n")
        cand_log.flush()
        return row

    @staticmethod
    def _print_candidate(cand: Candidate, verdict: DiversityVerdict, outcome: str,
                         archive: Archive) -> None:
        step = f"g{cand.generation:03d}" + (f".r{cand.revision}" if cand.revision else "")
        print_candidate_result(
            step=step,
            operator=cand.operator,
            non_obviousness=cand.non_obviousness,
            clarity=cand.mechanism_clarity,
            soundness=cand.soundness_100,
            diversity_score=verdict.diversity_score,
            outcome=outcome,
            active_yield=archive.yield_count(),
            discovery_yield=archive.discovery_yield(),
            repair_archive_size=len(archive.repair_archive),
        )

    def run_topic(self, *, topic_id: str, background: str, out_dir: Path) -> Archive:
        archive = Archive(
            accepted_capacity=self.c.accepted_capacity,
            repair_capacity=self.c.repair_capacity,
            repair_attempt_limit=self.c.repair_attempt_limit,
            accepted_refinement_limit=self.c.accepted_refinement_limit,
            rejected_pending_cap=self.c.rejected_pending_cap,
            history_field_char_cap=self.c.history_field_char_cap,
            feasibility_actionable_min=self.c.feasibility_actionable_min,
            repair_priority_weights=self.c.repair_priority_weights(),
            refinement_priority_weights=self.c.refinement_priority_weights(),
        )
        t = self.c.thresholds()
        cand_log_path = out_dir / f"candidates_{topic_id}.jsonl"
        snapshot_log_path = out_dir / f"archive_snapshots_{topic_id}.jsonl"
        analysis_path = out_dir / f"analysis_{topic_id}.json"

        last_outcome: str | None = None
        last_verdict: DiversityVerdict | None = None
        action_counts = {
            "free": 0, "pivot": 0, "repair": 0,
            "novelty_branch": 0, "accepted_refine": 0,
        }
        trace_rows: list[dict[str, Any]] = []
        representatives: dict[int, Candidate] = {}
        representative_outcomes: dict[int, str] = {}
        free_count = 0
        total_candidate_count = 0
        pending_action: str | None = None
        pending_parent_id: str | None = None
        pending_lineage_id: str | None = None
        pending_seed_index = 0

        if self.experiment_logger is not None:
            self.experiment_logger.log(
                "topic_started", topic_id=topic_id, mode="yield_qd",
                background=background, config=asdict(self.c), thresholds=asdict(t),
            )

        with (
            cand_log_path.open("w", encoding="utf-8") as cand_log,
            snapshot_log_path.open("w", encoding="utf-8") as snapshot_log,
        ):
            snapshot_log.write(json.dumps({
                "topic_id": topic_id, "step": 0, "event": "initial_archive",
                "state": archive.to_json(),
            }, ensure_ascii=False) + "\n")
            snapshot_log.flush()
            while free_count < self.c.max_budget or pending_action is not None:
                total_candidate_count += 1
                step = total_candidate_count
                idea_id = f"i{step:03d}"
                archive_before = _compact_archive_state(archive)
                memory_before = {
                    "accepted_signatures": [s.to_json() for s in archive.accepted_signatures()],
                    "historical_signatures": [s.to_json() for s in archive.historical_signatures()],
                    "repair_signatures": [c.signature.to_json() for c in archive.repair_archive],
                    "rejected_summary": archive.rejected_summary(),
                    "generation_memory_summary": archive.generation_memory_summary(),
                }
                requested_action = pending_action
                requested_parent_id = pending_parent_id
                requested_lineage_id = pending_lineage_id
                if requested_action is None:
                    action = "free"
                    free_count += 1
                    seed_index = free_count
                else:
                    action = requested_action
                    seed_index = pending_seed_index
                pending_action = None
                pending_parent_id = None
                pending_lineage_id = None
                pending_seed_index = 0
                scheduler_decision = {
                    "policy": "immediate_lineage_v1",
                    "chosen_action": action,
                    "free_seed_budget": self.c.max_budget,
                    "free_budget_used_before_candidate": (
                        free_count - 1 if action == "free" else free_count
                    ),
                    "free_budget_used_after_candidate": free_count,
                    "counts_toward_free_budget": action == "free",
                    "pending_parent_id": requested_parent_id,
                    "pending_lineage_id": requested_lineage_id,
                }
                weights = self._weights_for_seed(seed_index)
                parent: Candidate | None = None
                critic_challenge = ""
                parent_selection_priority: float | None = None
                relative_nb_comparison: dict[str, Any] = {"performed": False}

                if action == "free":
                    operator, directive = self._fresh_directive(
                        last_outcome, last_verdict, archive,
                    )
                    action_counts[operator] += 1
                    lineage_id = idea_id
                    revision = repair_attempts = accepted_refinements_used = 0
                    auxiliary_attempts_used = 0
                elif action == "repair":
                    parent = next(
                        (
                            candidate for candidate in archive.repair_archive
                            if candidate.id == requested_parent_id
                            or candidate.lineage_id == requested_lineage_id
                        ),
                        None,
                    )
                    if parent is None:
                        raise RuntimeError(
                            "Immediate repair parent disappeared before its bounded repair chain"
                        )
                    else:
                        parent_selection_priority = parent.repair_priority(
                            t, self.c.repair_attempt_limit, archive.repair_priority_weights,
                        )
                        archive.remove_repair(parent.id)
                        operator = "repair"
                        action_counts[operator] += 1
                        set_trace_context(
                            getattr(self.critic, "client", None),
                            topic_id=topic_id, candidate_id=idea_id,
                            search_step=step, operator=operator, parent_id=parent.id,
                            lineage_id=parent.lineage_id, stage="focused_repair_critic",
                        )
                        self.critic.set_log_context(topic_id=topic_id, target_paper_id=idea_id)
                        critic_challenge = self.critic.focused_review(
                            system_prompt=REPAIR_CRITIC_SYSTEM,
                            user_prompt=build_repair_critic_user(
                                parent.idea_text, defect=parent.defect,
                                feedback_notes=self._revision_feedback_notes(parent),
                                soundness_rationales=parent.soundness_rationales,
                                accepted=archive.accepted_signatures(),
                            ),
                        )
                        directive = op_critic_revision(
                            parent.idea_text, critic_challenge, mode="repair",
                        )
                        lineage_id = parent.lineage_id
                        revision = parent.revision + 1
                        repair_attempts = parent.repair_attempts + 1
                        accepted_refinements_used = 0
                        auxiliary_attempts_used = parent.auxiliary_attempts_used + 1
                else:  # accepted_refine = same-lineage accepted polishing
                    parent = archive.accepted.get(str(requested_parent_id))
                    if parent is None and requested_lineage_id is not None:
                        parent = next(
                            (
                                candidate for candidate in archive.accepted.values()
                                if candidate.lineage_id == requested_lineage_id
                            ),
                            None,
                        )
                    if parent is None:
                        raise RuntimeError(
                            "Immediate polish parent disappeared before its bounded refinement chain"
                        )
                    else:
                        parent_selection_priority = parent.refinement_priority(
                            self.c.accepted_refinement_limit,
                            archive.refinement_priority_weights,
                        )
                        operator = "accepted_refine"
                        action_counts[operator] += 1
                        press_deficits = self._press_deficits(parent)
                        defect = (
                            "The idea passes every acceptance gate but remains below one or more "
                            "soft aspiration targets. Increase its weighted quality without "
                            "losing its mechanism or any gate: "
                            f"Q={weights.non_obviousness:.2f}*NB + "
                            f"{weights.soundness:.2f}*soundness + "
                            f"{weights.clarity:.2f}*clarity; current NB={parent.non_obviousness}, "
                            f"soundness={parent.soundness_100:.1f}, clarity={parent.mechanism_clarity}."
                            " The NB threshold is only a floor: challenge whether the conceptual "
                            "leap is genuinely hard to anticipate, and do not trade it away for "
                            "conventional implementation polish. A separate swapped-order NB "
                            "comparison will block replacement if the child becomes more obvious."
                        )
                        defect += "\n\nSOFT PRESS TARGETS (not acceptance gates):\n- " + "\n- ".join(
                            press_deficits
                        )
                        if self.c.press_feasibility_target > 0 and parent.feasibility_pressure:
                            defect += "\n\nNON-GATING FEASIBILITY PRESSURE:\n" + parent.feasibility_pressure
                        set_trace_context(
                            getattr(self.critic, "client", None),
                            topic_id=topic_id, candidate_id=idea_id,
                            search_step=step, operator=operator, parent_id=parent.id,
                            lineage_id=parent.lineage_id, stage="focused_accepted_refinement_critic",
                        )
                        self.critic.set_log_context(topic_id=topic_id, target_paper_id=idea_id)
                        critic_challenge = self.critic.focused_review(
                            system_prompt=REPAIR_CRITIC_SYSTEM,
                            user_prompt=build_repair_critic_user(
                                parent.idea_text, defect=defect,
                                feedback_notes=self._revision_feedback_notes(parent),
                                soundness_rationales=parent.soundness_rationales,
                                accepted=archive.accepted_signatures(exclude_id=parent.id),
                            ),
                        )
                        directive = op_critic_revision(
                            parent.idea_text, critic_challenge, mode="accepted_refine",
                        )
                        lineage_id = parent.lineage_id
                        revision = parent.revision + 1
                        repair_attempts = 0
                        accepted_refinements_used = parent.accepted_refinements_used + 1
                        auxiliary_attempts_used = parent.auxiliary_attempts_used + 1

                # Every path below makes one ideator call and one full evaluation. Only a free
                # seed consumes the free-search budget; auxiliary drafts remain separately
                # bounded/logged. Persistent revisions reconstruct their lineage from the
                # complete local parent text plus the focused critic challenge.
                set_trace_context(
                    getattr(self.ideator, "client", None),
                    topic_id=topic_id, candidate_id=idea_id,
                    search_step=step, operator=operator,
                    parent_id=parent.id if parent is not None else None,
                    lineage_id=lineage_id, revision=revision, stage="candidate_generation",
                )
                self.ideator.set_log_context(topic_id=topic_id, target_paper_id=idea_id)
                idea_text = self.ideator.open(
                    background=background, prior_cores_block=None,
                    critic_opening_challenge=directive,
                )
                cand = self._evaluate(
                    idea_text, idea_id=idea_id, gen=step, operator=operator,
                    topic_id=topic_id, background=background,
                    parent_id=parent.id if parent is not None else None,
                    lineage_id=lineage_id, revision=revision,
                    repair_attempts=repair_attempts,
                    accepted_refinements_used=accepted_refinements_used,
                    auxiliary_attempts_used=auxiliary_attempts_used,
                    seed_index=seed_index,
                )
                verdict = self._judge(
                    cand, archive, topic_id=topic_id,
                    exclude_accepted_id=(parent.id if operator == "accepted_refine" and parent else None),
                    exclude_historical_lineage_id=(
                        parent.lineage_id
                        if operator == "accepted_refine" and parent else None
                    ),
                )
                if operator == "accepted_refine" and parent is not None:
                    replaceable_before_nb = (
                        cand.eval_ok and verdict.parsed and cand.passes_all(t)
                        and not verdict.is_duplicate
                        and verdict.diversity_score >= t.diversity_D
                        and cand.outranks(parent, weights)
                    )
                    if replaceable_before_nb:
                        relative_nb_comparison = self._relative_nb_comparison(
                            parent=parent, child=cand, topic_id=topic_id,
                        )
                    outcome = archive.consider_accepted_refinement(
                        cand, parent_id=parent.id, verdict=verdict, t=t, weights=weights,
                        nb_not_worse=bool(
                            relative_nb_comparison.get("child_not_worse", True)
                        ),
                    )
                else:
                    outcome = archive.consider(
                        cand, verdict, t, weights,
                        repair_parent=(parent if operator == "repair" else None),
                    )

                # Maintain exactly one externally evaluable representative for each free seed.
                # A failed auxiliary draft never overwrites its still-better parent.
                if operator in {"free", "pivot"}:
                    representatives[seed_index] = cand
                elif operator == "repair":
                    repair_representative = next(
                        (
                            candidate for candidate in archive.repair_archive
                            if candidate.seed_index == seed_index
                        ),
                        None,
                    )
                    if outcome in {ACCEPTED, REPLACED}:
                        representatives[seed_index] = cand
                    elif repair_representative is not None:
                        representatives[seed_index] = repair_representative
                elif operator == "accepted_refine":
                    if outcome == REPLACED:
                        representatives[seed_index] = cand
                    elif parent is not None:
                        representatives[seed_index] = parent
                representative_outcomes[seed_index] = outcome

                # Finish this lineage before spending the next free seed. Fatal consensus and
                # diversity duplicates are discarded immediately. Non-fatal repair candidates
                # exhaust only their own K; accepted candidates are polished only while a soft
                # aspiration remains below target.
                repair_next = next(
                    (
                        candidate for candidate in archive.repair_archive
                        if candidate.seed_index == seed_index
                    ),
                    None,
                )
                if (
                    outcome in {REPAIR_QUEUED, REPAIR_RETAINED}
                    and repair_next is not None
                    and repair_next.repair_attempts < self.c.repair_attempt_limit
                    and repair_next.auxiliary_attempts_used < self.c.auxiliary_attempt_limit
                ):
                    pending_action = "repair"
                    pending_parent_id = repair_next.id
                    pending_lineage_id = repair_next.lineage_id
                    pending_seed_index = seed_index
                else:
                    accepted_next = next(
                        (
                            candidate for candidate in archive.accepted.values()
                            if candidate.seed_index == seed_index
                        ),
                        None,
                    )
                    if accepted_next is not None and self._needs_immediate_polish(accepted_next):
                        pending_action = "accepted_refine"
                        pending_parent_id = accepted_next.id
                        pending_lineage_id = accepted_next.lineage_id
                        pending_seed_index = seed_index

                set_trace_context(
                    self.judge_client, topic_id=topic_id, candidate_id=idea_id,
                    search_step=step, operator=operator,
                    stage="rejected_memory_consolidation_if_needed",
                )
                self._maybe_consolidate(archive)
                archive.action_counts = dict(action_counts)
                archive.free_generations_completed = free_count
                archive.total_candidates_evaluated = total_candidate_count
                archive.search_steps_completed = free_count
                archive_after = _compact_archive_state(archive)
                row = self._write_candidate(
                    cand_log, cand, verdict, outcome, weights=weights,
                    critic_challenge=critic_challenge,
                    parent_selection_priority=parent_selection_priority,
                    parent=parent,
                    directive=directive,
                    relative_nb_comparison=relative_nb_comparison,
                    scheduler_decision=scheduler_decision,
                    memory_before=memory_before,
                    archive_before=archive_before,
                    archive_after=archive_after,
                    thresholds=t,
                )
                trace_rows.append(row)
                snapshot_record = {
                    "topic_id": topic_id, "step": step, "candidate_id": idea_id,
                    "operator": operator, "outcome": outcome,
                    "transition": row["archive_transition"],
                    "state": archive.to_json(),
                }
                snapshot_log.write(json.dumps(snapshot_record, ensure_ascii=False) + "\n")
                snapshot_log.flush()
                if self.experiment_logger is not None:
                    self.experiment_logger.log(
                        "search_step_completed", topic_id=topic_id, **row
                    )
                self._print_candidate(cand, verdict, outcome, archive)
                last_outcome, last_verdict = outcome, verdict

        archive.action_counts = action_counts
        archive.free_generations_completed = free_count
        archive.total_candidates_evaluated = total_candidate_count
        archive.search_steps_completed = free_count
        self.last_lineage_representatives = [
            representatives[seed_index]
            for seed_index in range(1, self.c.max_budget + 1)
        ]
        self.last_representative_outcomes = dict(representative_outcomes)
        representative_path = out_dir / f"lineage_representatives_{topic_id}.jsonl"
        with representative_path.open("w", encoding="utf-8") as handle:
            for representative in self.last_lineage_representatives:
                payload = Archive._candidate_json(representative, include_text=True)
                accepted_version_ids = [
                    candidate.id for candidate in archive.accepted_version_pool.values()
                    if candidate.lineage_id == representative.lineage_id
                    and candidate.seed_index == representative.seed_index
                ]
                accepted_lineage_version_ids = [
                    candidate.id for candidate in archive.accepted_version_pool.values()
                    if candidate.lineage_id == representative.lineage_id
                ]
                payload["final_internal_outcome"] = representative_outcomes.get(
                    representative.seed_index
                )
                payload["active_at_search_end"] = representative.id in archive.accepted
                payload["accepted_version_ids"] = accepted_version_ids
                payload["accepted_lineage_version_ids"] = accepted_lineage_version_ids
                payload["selected_from_parent_child_pool"] = len(accepted_version_ids) > 1
                payload["lineage_selection_policy"] = (
                    "current accepted version after gated weighted-Q replacement; failed or "
                    "non-improving children never displace their parent"
                )
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        # Optional external-analysis pool. It contains every version that was ever accepted,
        # plus one fallback for a seed whose lineage never reached acceptance. Parent/child
        # records share lineage_id, so the diversity evaluator skips their mutual pair while
        # still comparing both versions against every foreign lineage.
        evaluation_pool_by_id: dict[str, Candidate] = dict(archive.accepted_version_pool)
        for representative in self.last_lineage_representatives:
            evaluation_pool_by_id.setdefault(representative.id, representative)
        evaluation_pool = sorted(
            evaluation_pool_by_id.values(),
            key=lambda candidate: (
                candidate.seed_index, candidate.generation, candidate.revision, candidate.id,
            ),
        )
        evaluation_pool_path = out_dir / f"lineage_evaluation_pool_{topic_id}.jsonl"
        evaluation_pool_record = {
            "topic_id": topic_id,
            "generation_mode": "agentic_lineage_version_pool",
            "fresh_seed_budget": self.c.max_budget,
            "n_seed_slots": len(self.last_lineage_representatives),
            "n_version_candidates": len(evaluation_pool),
            "selection_constraint": (
                "at most one version per lineage; paper-facing primary file already contains "
                "one internally selected representative per fresh seed"
            ),
            "items": [
                {
                    "idea_id": candidate.id,
                    "idea_text": candidate.idea_text,
                    "final_idea_core": _sig_to_core(candidate),
                    "source_metadata": {
                        "seed_index": candidate.seed_index,
                        "lineage_id": candidate.lineage_id,
                        "operator": candidate.operator,
                        "revision": candidate.revision,
                        "parent_id": candidate.parent_id,
                        "accepted_version": candidate.id in archive.accepted_version_pool,
                        "active_at_search_end": candidate.id in archive.accepted,
                        "primary_seed_representative": any(
                            candidate.id == representative.id
                            for representative in self.last_lineage_representatives
                        ),
                    },
                }
                for candidate in evaluation_pool
            ],
        }
        evaluation_pool_path.write_text(
            json.dumps(evaluation_pool_record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (out_dir / f"archive_{topic_id}.json").write_text(json.dumps(archive.to_json(), indent=2))
        analysis = _analysis_summary(topic_id=topic_id, rows=trace_rows, archive=archive)
        analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.experiment_logger is not None:
            self.experiment_logger.log(
                "topic_completed", topic_id=topic_id, mode="yield_qd",
                analysis_summary=analysis, final_archive=archive.to_json(),
            )
        return archive


def _placeholder_sig(idea_id: str):
    from ideagent.signature import SemanticSignature
    return SemanticSignature(id=idea_id, problem="", core_mechanism="", novel_move="", key_assumption="", expected_effect="")


def run_search(
    *, config: Config, ideator: IdeatorAgent, critic: CriticAgent,
    quality: QualityAgent,
    steno_client: Any, steno_gen_kwargs: dict[str, Any],
    judge_client: Any, judge_gen_kwargs: dict[str, Any],
    experiment_logger: ExperimentLogger | None = None,
) -> dict[str, Any]:
    out_dir = Path(config.output_dir)
    prepare_output_dir(out_dir, resume=config.resume, init_topic_idx=config.init_topic_idx)
    if experiment_logger is not None:
        models = {
            "ideator": getattr(ideator.client, "model_id", ""),
            "critic": getattr(critic.client, "model_id", ""),
            "feedback_and_soundness": getattr(quality.client, "model_id", ""),
            "steno": getattr(steno_client, "model_id", ""),
            "judge": getattr(judge_client, "model_id", ""),
        }
        fairness_contract = {
            "experimental_unit": "one topic",
            "independent_free_seed_budget": config.max_budget,
            "candidate_output_budget": (
                "variable: free seeds plus bounded immediate repair/polish drafts"
            ),
            "one_idea_per_ideator_call": True,
            "ideator_calls_per_completed_topic": (
                "variable and logged; at most free_seed_budget * "
                "(1 + auxiliary_attempt_limit)"
            ),
            "maximum_ideator_drafts_per_topic": (
                config.max_budget * (1 + config.auxiliary_attempt_limit)
            ),
            "stopping_rule": (
                "exactly max_budget independent free seeds; each seed's immediate bounded "
                "repair/polish chain completes before the next seed; accepted_target does not stop"
            ),
            "background_builder": "build_background_context",
            "posthoc_evaluation_field": "items[].idea_text",
            "treatment": [
                "selective evaluator-informed compact memory",
                "quality and diversity gates",
                "immediate repair critic and bounded same-lineage repair",
                "immediate accepted-idea same-mechanism polishing against soft aspiration targets",
                "one shared per-seed auxiliary allowance across repair and accepted polishing",
                "position-swapped relative non-obviousness non-regression check",
                "majority consensus for hard-fatal soundness with disputed cases repairable",
                "deterministic free-seed then immediate-lineage controller",
            ],
            "not_used": [
                "embeddings", "predefined diversity loci", "adaptive opportunity scheduler",
                "probabilistic action scheduler", "novelty branching",
            ],
            "evaluation_order": (
                "signature, rubric evaluation, all soundness samples, then diversity judgment; "
                "no staged early rejection; relative non-obviousness comparison is run only for "
                "an otherwise replacement-eligible accepted refinement"
            ),
        }
        manifest = {
            "mode": "yield_qd",
            "config": asdict(config),
            "thresholds": asdict(config.thresholds()),
            "models": models,
            "sampling": {
                "ideator": getattr(ideator, "gen_kwargs", {}),
                "critic": getattr(critic, "gen_kwargs", {}),
                "feedback_and_soundness": getattr(quality, "gen_kwargs", {}),
                "steno": steno_gen_kwargs,
                "judge": judge_gen_kwargs,
            },
            "prompts": {
                "ideator_system_prompt": IDEATOR_SYSTEM_PROMPT,
                "ideator_system_prompt_sha256": text_sha256(IDEATOR_SYSTEM_PROMPT),
                "feedback_system_prompt_no_prior": build_feedback_system_prompt([]),
                "soundness_system_prompt": SOUNDNESS_EVAL_SYSTEM_PROMPT,
                "diversity_judge_system_prompt": DIVERSITY_JUDGE_SYSTEM,
                "repair_critic_system_prompt": REPAIR_CRITIC_SYSTEM,
                "novelty_branch_critic_system_prompt": NOVELTY_BRANCH_CRITIC_SYSTEM,
                "relative_nb_system_prompt": RELATIVE_NB_SYSTEM,
                "consolidation_system_prompt": CONSOLIDATION_SYSTEM,
            },
            "fairness_contract": fairness_contract,
            "storage_policy": {
                "model_context": (
                    "free/pivot calls receive bounded signatures and compressed failure memory; "
                    "repair/refinement/novelty-branch calls receive the selected full parent plus "
                    "a focused critique; novelty branches create new lineages"
                ),
                "disk": (
                    "every full candidate, parent, prompt, output, rubric, soundness sample, diversity "
                    "verdict, scheduler draw, memory state, archive transition, and archive snapshot"
                ),
            },
        }
        manifest_path = write_run_manifest(
            out_dir, logger=experiment_logger, manifest=manifest
        )
        experiment_logger.log(
            "run_started", mode="yield_qd", config=asdict(config),
            models=models, manifest_path=str(manifest_path), fairness_contract=fairness_contract,
        )
    # Suffixed by init_topic_idx (blank when 0, i.e. the default single-run case) so that
    # parallel runs covering disjoint topic slices against the same output_dir each get
    # their own file instead of append-racing a shared one.
    _tag = f"_from{config.init_topic_idx}" if config.init_topic_idx else ""
    final_idea_cores_path = out_dir / f"final_idea_cores{_tag}.jsonl"

    loop = GenerationLoop(
        config=config, ideator=ideator, critic=critic, quality=quality,
        steno_client=steno_client, steno_gen_kwargs=steno_gen_kwargs,
        judge_client=judge_client, judge_gen_kwargs=judge_gen_kwargs,
        experiment_logger=experiment_logger,
    )

    records: list[dict[str, Any]] = []
    completed_topic_ids: set[str] = set()
    if config.resume and final_idea_cores_path.exists():
        for rec in read_jsonl(final_idea_cores_path):
            tid = str(rec.get("topic_id", ""))
            if tid:
                items = rec.get("items")
                recorded_budget = rec.get("free_seed_budget")
                if (
                    recorded_budget != config.max_budget
                    or not isinstance(items, list)
                    or len(items) != config.max_budget
                ):
                    raise ValueError(
                        f"Refusing to resume incompatible topic {tid}: expected "
                        f"free_seed_budget={config.max_budget} and exactly "
                        f"{config.max_budget} seed representatives. Use a new output_dir or "
                        "set resume=false."
                    )
                completed_topic_ids.add(tid)
                records.append(rec)
        print(f"Resume: {len(completed_topic_ids)} topic(s) already done, skipping.", flush=True)
    else:
        create_jsonl(final_idea_cores_path)

    groups = load_raw_topic_groups(config.input_jsonl)[config.init_topic_idx:config.end_topic_idx]
    for group in groups:
        topic_id = str(group["topic_id"])
        if topic_id in completed_topic_ids:
            print(f"Topic {topic_id}: already completed, skipping.", flush=True)
            continue

        background = build_background_context([parse_paper(p) for p in group["background_papers"]])
        if config.max_background_tokens:
            background = truncate_by_tokens_rough(background, config.max_background_tokens)
        print(f"\n=== Yield search — topic {topic_id} ===", flush=True)
        archive = loop.run_topic(topic_id=topic_id, background=background, out_dir=out_dir)
        print(
            f"--- topic {topic_id}: ActiveYield={archive.yield_count()} "
            f"DiscoveryYield={archive.discovery_yield()} "
            f"ActiveActionable={archive.active_actionable_yield()} "
            f"DiscoveryActionable={archive.discovery_actionable_yield()} "
            f"(F>={config.feasibility_actionable_min}) / free_seeds={config.max_budget} "
            f"total_drafts={archive.total_candidates_evaluated} ---",
            flush=True,
        )
        record = {
            "topic_id": topic_id,
            "active_yield": archive.yield_count(),
            "discovery_yield": archive.discovery_yield(),
            "active_actionable_yield": archive.active_actionable_yield(),
            "discovery_actionable_yield": archive.discovery_actionable_yield(),
            "feasibility_actionable_min": config.feasibility_actionable_min,
            "accepted_target_legacy_unused": config.accepted_target,
            "free_seed_budget": config.max_budget,
            "search_budget": config.max_budget,  # backward-compatible alias
            "search_steps_completed": archive.search_steps_completed,
            "total_candidates_evaluated": archive.total_candidates_evaluated,
            "auxiliary_candidates_evaluated": (
                archive.total_candidates_evaluated - archive.free_generations_completed
            ),
            # Keep the untouched proposal beside the compact compatibility core. Post-hoc
            # comparisons must evaluate this same full-text field for every method; otherwise
            # steno/core compression differences become an experimental confound.
            "items": [
                {
                    "idea_id": c.id,
                    "idea_text": c.idea_text,
                    "final_idea_core": _sig_to_core(c),
                    "source_metadata": {
                        "seed_index": c.seed_index,
                        "lineage_id": c.lineage_id,
                        "operator": c.operator,
                        "revision": c.revision,
                        "parent_id": c.parent_id,
                        "active_at_search_end": c.id in archive.accepted,
                        "final_internal_outcome": loop.last_representative_outcomes.get(
                            c.seed_index
                        ),
                        "accepted_version_ids": [
                            candidate.id
                            for candidate in archive.accepted_version_pool.values()
                            if candidate.lineage_id == c.lineage_id
                            and candidate.seed_index == c.seed_index
                        ],
                        "accepted_lineage_version_ids": [
                            candidate.id
                            for candidate in archive.accepted_version_pool.values()
                            if candidate.lineage_id == c.lineage_id
                        ],
                        "selected_from_parent_child_pool": sum(
                            candidate.lineage_id == c.lineage_id
                            and candidate.seed_index == c.seed_index
                            for candidate in archive.accepted_version_pool.values()
                        ) > 1,
                        "lineage_selection_policy": (
                            "current accepted version after gated weighted-Q replacement; "
                            "at most one version of a lineage is eligible for Top-K"
                        ),
                    },
                }
                for c in loop.last_lineage_representatives
            ],
        }
        records.append(record)
        append_jsonl(final_idea_cores_path, record)

    print(f"\nDone. {len(records)} topic(s). Output dir: {out_dir}", flush=True)
    result = {"records": records}
    if experiment_logger is not None:
        experiment_logger.log(
            "run_completed", mode="yield_qd", topic_records=len(records), records=records,
        )
        summary_path = write_trace_summary(out_dir, logger=experiment_logger)
        experiment_logger.log("trace_summary_written", path=str(summary_path))
    return result
