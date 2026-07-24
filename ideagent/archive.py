"""Bounded active archive and persistent memories for Yield search.

The model-facing accepted archive is capped, but discovery Yield is not: qualified ideas that
leave the active store are retained as compact history. Borderline, non-fatal mechanisms live in
a persistent repair archive. Full candidate text remains local; prompts receive only signatures,
defects, and compressed failure/retired-history summaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isclose
from typing import Any

from ideagent.diversity_judge import DiversityVerdict
from ideagent.signature import SemanticSignature

ACCEPTED = "accepted"
REPLACED = "replaced"
QUALIFIED_RETIRED = "qualified_retired"
KEEP_ACCEPTED = "keep_accepted"
REJECT_DUPLICATE = "reject_duplicate"
REPAIR_QUEUED = "repair_queued"
REPAIR_RETAINED = "repair_retained"
REJECT_UNSOUND = "reject_unsound"
REJECT_EVAL_FAIL = "reject_eval_fail"


@dataclass(frozen=True)
class QualityWeights:
    non_obviousness: float
    soundness: float
    clarity: float

    def __post_init__(self) -> None:
        values = (self.non_obviousness, self.soundness, self.clarity)
        if any(v < 0 for v in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("QualityWeights must be non-negative and sum to 1")


EARLY_WEIGHTS = QualityWeights(0.70, 0.20, 0.10)
LATE_WEIGHTS = QualityWeights(0.40, 0.40, 0.20)


@dataclass(frozen=True)
class RepairPriorityWeights:
    novelty: float = 0.50
    gate_closeness: float = 0.35
    remaining_attempts: float = 0.15
    soundness_deficit: float = 0.55
    non_obviousness_deficit: float = 0.15
    clarity_deficit: float = 0.30

    def __post_init__(self) -> None:
        outer = (self.novelty, self.gate_closeness, self.remaining_attempts)
        inner = (
            self.soundness_deficit,
            self.non_obviousness_deficit,
            self.clarity_deficit,
        )
        if (
            any(v < 0 for v in outer + inner)
            or abs(sum(outer) - 1.0) > 1e-6
            or abs(sum(inner) - 1.0) > 1e-6
        ):
            raise ValueError("Repair priority outer and deficit weights must each sum to 1")


@dataclass(frozen=True)
class RefinementPriorityWeights:
    novelty: float = 0.60
    repair_headroom: float = 0.25
    remaining_attempts: float = 0.15
    soundness_headroom: float = 0.65
    clarity_headroom: float = 0.35

    def __post_init__(self) -> None:
        outer = (self.novelty, self.repair_headroom, self.remaining_attempts)
        inner = (self.soundness_headroom, self.clarity_headroom)
        if (
            any(v < 0 for v in outer + inner)
            or abs(sum(outer) - 1.0) > 1e-6
            or abs(sum(inner) - 1.0) > 1e-6
        ):
            raise ValueError("Refinement priority outer and headroom weights must each sum to 1")


DEFAULT_REPAIR_PRIORITY = RepairPriorityWeights()
DEFAULT_REFINEMENT_PRIORITY = RefinementPriorityWeights()


@dataclass
class Thresholds:
    soundness_floor: float = 60.0
    soundness_fatal_threshold: int = 3
    soundness_fatal_min_votes: int = 3
    nb_min: int = 60
    clarity_min: int = 60
    diversity_D: int = 55
    repair_margin: int = 20
    repair_diversity_D: int = 45


@dataclass
class Candidate:
    id: str
    idea_text: str
    signature: SemanticSignature
    soundness_100: float
    soundness_samples: list[int]
    soundness_fatal: bool
    soundness_rationale: str
    non_obviousness: int
    mechanism_clarity: int
    feasibility: int
    generation: int
    operator: str
    seed_index: int = 0
    soundness_rationales: list[str] = field(default_factory=list)
    soundness_panel_results: list[dict[str, Any]] = field(default_factory=list)
    soundness_fatal_votes: int = 0
    soundness_disputed: bool = False
    feedback_notes: list[str] = field(default_factory=list)
    rubric_assessment: list[dict[str, Any]] = field(default_factory=list)
    parent_id: str | None = None
    lineage_id: str = ""
    revision: int = 0
    repair_attempts: int = 0
    accepted_refinements_used: int = 0
    auxiliary_attempts_used: int = 0
    novelty_branches_used: int = 0
    diversity_score: int = 0
    eval_ok: bool = True
    defect: str = ""
    feasibility_delta_from_parent: int | None = None
    feasibility_pressure: str = ""

    def __post_init__(self) -> None:
        if not self.lineage_id:
            self.lineage_id = self.id

    def weighted_quality(self, weights: QualityWeights) -> float:
        return (
            weights.non_obviousness * self.non_obviousness
            + weights.soundness * self.soundness_100
            + weights.clarity * self.mechanism_clarity
        )

    def outranks(self, other: "Candidate", weights: QualityWeights) -> bool:
        """Core Q decides; feasibility breaks only an exact floating-point tie."""
        own_q = self.weighted_quality(weights)
        other_q = other.weighted_quality(weights)
        if not isclose(own_q, other_q, rel_tol=0.0, abs_tol=1e-9):
            return own_q > other_q
        return self.feasibility > other.feasibility

    def passes_soundness(self, t: Thresholds) -> bool:
        return (
            self.soundness_100 >= t.soundness_floor
            and not self.soundness_fatal
            and not self.soundness_disputed
        )

    def passes_quality(self, t: Thresholds) -> bool:
        return self.non_obviousness >= t.nb_min and self.mechanism_clarity >= t.clarity_min

    def passes_all(self, t: Thresholds) -> bool:
        return self.passes_soundness(t) and self.passes_quality(t)

    def near_thresholds(self, t: Thresholds) -> bool:
        # Any non-consensus severe objection deserves its immediate bounded repair chain,
        # even when the panel mean is farther than the ordinary numeric margin.
        sound_near = (
            self.soundness_disputed
            or self.soundness_100 >= t.soundness_floor - t.repair_margin
        )
        quality_near = (
            self.non_obviousness >= t.nb_min - t.repair_margin
            and self.mechanism_clarity >= t.clarity_min - t.repair_margin
        )
        return not self.passes_all(t) and not self.soundness_fatal and sound_near and quality_near

    def defect_text(self, t: Thresholds) -> str:
        defects: list[str] = []
        if self.soundness_fatal:
            defects.append(
                "soundness fatal-panel consensus "
                f"({self.soundness_fatal_votes}/{len(self.soundness_samples)} votes <= "
                f"{t.soundness_fatal_threshold}): {self.soundness_rationale}".strip()
            )
        else:
            if self.soundness_disputed:
                defects.append(
                    "soundness panel dispute "
                    f"({self.soundness_fatal_votes}/{len(self.soundness_samples)} severe votes <= "
                    f"{t.soundness_fatal_threshold}; repair or re-check the cited failure): "
                    f"{self.soundness_rationale}".strip()
                )
            if self.soundness_100 < t.soundness_floor:
                defects.append(
                    f"soundness {self.soundness_100:.0f}<{t.soundness_floor:.0f}: "
                    f"{self.soundness_rationale}".strip()
                )
        if self.non_obviousness < t.nb_min:
            defects.append(f"non-obviousness {self.non_obviousness}<{t.nb_min}")
        if self.mechanism_clarity < t.clarity_min:
            defects.append(f"mechanism clarity {self.mechanism_clarity}<{t.clarity_min}")
        return "; ".join(defects)

    def repair_priority(
        self, t: Thresholds, attempt_limit: int,
        policy: RepairPriorityWeights = DEFAULT_REPAIR_PRIORITY,
    ) -> float:
        """Upside score for choosing one mechanism from the repair archive."""
        scale = max(float(t.repair_margin), 1.0)
        clip = lambda x: max(0.0, min(1.0, x))
        d_sound = clip((t.soundness_floor - self.soundness_100) / scale)
        d_nb = clip((t.nb_min - self.non_obviousness) / scale)
        d_clarity = clip((t.clarity_min - self.mechanism_clarity) / scale)
        closeness = 1.0 - (
            policy.soundness_deficit * d_sound
            + policy.non_obviousness_deficit * d_nb
            + policy.clarity_deficit * d_clarity
        )
        remaining = clip(1.0 - self.repair_attempts / max(attempt_limit, 1))
        novelty = clip(self.non_obviousness / 100.0)
        return (
            policy.novelty * novelty
            + policy.gate_closeness * closeness
            + policy.remaining_attempts * remaining
        )

    def refinement_priority(
        self, attempt_limit: int,
        policy: RefinementPriorityWeights = DEFAULT_REFINEMENT_PRIORITY,
    ) -> float:
        """Upside score for choosing one already-accepted idea to refine."""
        clip = lambda x: max(0.0, min(1.0, x))
        headroom = (
            policy.soundness_headroom * clip(1.0 - self.soundness_100 / 100.0)
            + policy.clarity_headroom * clip(1.0 - self.mechanism_clarity / 100.0)
        )
        remaining = clip(1.0 - self.accepted_refinements_used / max(attempt_limit, 1))
        novelty = clip(self.non_obviousness / 100.0)
        return (
            policy.novelty * novelty
            + policy.repair_headroom * headroom
            + policy.remaining_attempts * remaining
        )


class Archive:
    def __init__(
        self, *, accepted_capacity: int = 10, repair_capacity: int = 10,
        repair_attempt_limit: int = 1, accepted_refinement_limit: int = 2,
        rejected_pending_cap: int = 20,
        history_field_char_cap: int = 120,
        feasibility_actionable_min: int = 60,
        repair_priority_weights: RepairPriorityWeights = DEFAULT_REPAIR_PRIORITY,
        refinement_priority_weights: RefinementPriorityWeights = DEFAULT_REFINEMENT_PRIORITY,
    ) -> None:
        self.accepted: dict[str, Candidate] = {}
        self.repair_archive: list[Candidate] = []
        # Backward-compatible alias used by older analysis/tests.
        self.repair_queue = self.repair_archive
        self.retired_qualified: list[dict[str, Any]] = []
        self._historical_signatures: dict[str, SemanticSignature] = {}
        self._historical_lineages: dict[str, str] = {}
        self.qualified_lineages: set[str] = set()
        self.qualified_lineage_feasibility: dict[str, int] = {}
        # Full local accepted-version memory. Prompts still receive compact signatures only.
        # Replaced parents remain here with explicit edges to their accepted children.
        self.accepted_version_pool: dict[str, Candidate] = {}
        self.accepted_parent_child_edges: list[dict[str, str]] = []
        self.action_counts: dict[str, int] = {}
        self.free_generations_completed = 0
        self.total_candidates_evaluated = 0
        self.search_steps_completed = 0
        self.rejected_families_summary = ""
        self.rejected_pending: list[str] = []
        self.accepted_capacity = accepted_capacity
        self.repair_capacity = repair_capacity
        self.repair_attempt_limit = repair_attempt_limit
        self.accepted_refinement_limit = accepted_refinement_limit
        self.rejected_pending_cap = rejected_pending_cap
        self.history_field_char_cap = history_field_char_cap
        self.feasibility_actionable_min = feasibility_actionable_min
        self.repair_priority_weights = repair_priority_weights
        self.refinement_priority_weights = refinement_priority_weights

    def _sync_repair_alias(self) -> None:
        self.repair_queue = self.repair_archive

    # ── reads / selection ──────────────────────────────────────────────────────────────
    def yield_count(self) -> int:
        return len(self.accepted)

    def discovery_yield(self) -> int:
        return len(self.qualified_lineages)

    def active_actionable_yield(self) -> int:
        return sum(
            c.feasibility >= self.feasibility_actionable_min
            for c in self.accepted.values()
        )

    def discovery_actionable_yield(self) -> int:
        return sum(
            feasibility >= self.feasibility_actionable_min
            for feasibility in self.qualified_lineage_feasibility.values()
        )

    def accepted_signatures(self, *, exclude_id: str | None = None) -> list[SemanticSignature]:
        return [c.signature for cid, c in self.accepted.items() if cid != exclude_id]

    def historical_signatures(
        self, *, exclude_lineage_id: str | None = None,
    ) -> list[SemanticSignature]:
        return [
            sig for sid, sig in self._historical_signatures.items()
            if self._historical_lineages[sid] != exclude_lineage_id
        ]

    def rejected_summary(self) -> str:
        parts = []
        if self.rejected_families_summary.strip():
            parts.append(self.rejected_families_summary.strip())
        if self.rejected_pending:
            parts.append("recent: " + " ;; ".join(self.rejected_pending[-self.rejected_pending_cap :]))
        return "\n".join(parts)

    def generation_memory_summary(self) -> str:
        """Compact search memory for free generation; never exposes full idea text."""
        parts: list[str] = []
        rejected = self.rejected_summary()
        if rejected:
            parts.append("REJECTED / KNOWN-BAD FAMILIES:\n" + rejected)
        if self._historical_signatures:
            history = "\n".join(
                sig.mechanism_compact(self.history_field_char_cap)
                for sig in self._historical_signatures.values()
            )
            parts.append(
                "PREVIOUSLY QUALIFIED MECHANISMS (never count these as new discoveries):\n"
                + history
            )
        if self.repair_archive:
            reserved = "\n".join(
                c.signature.mechanism_compact(self.history_field_char_cap)
                for c in self.repair_archive
            )
            parts.append(
                "PROMISING MECHANISMS RESERVED FOR REPAIR (do not independently rediscover):\n"
                + reserved
            )
        return "\n".join(parts)

    # ── compact memory writes ──────────────────────────────────────────────────────────
    def _record_memory(self, cand: Candidate, note: str) -> None:
        self.rejected_pending.append(f"{cand.signature.compact()} :: {note}")
        if len(self.rejected_pending) > self.rejected_pending_cap:
            self.rejected_pending = self.rejected_pending[-self.rejected_pending_cap :]

    def _remember_qualified(self, cand: Candidate) -> None:
        self.qualified_lineages.add(cand.lineage_id)
        self.qualified_lineage_feasibility[cand.lineage_id] = max(
            cand.feasibility,
            self.qualified_lineage_feasibility.get(cand.lineage_id, 0),
        )

    def _remember_accepted_version(
        self, cand: Candidate, *, parent: Candidate | None = None, reason: str,
    ) -> None:
        self.accepted_version_pool[cand.id] = cand
        if parent is None:
            return
        self.accepted_version_pool[parent.id] = parent
        edge = {
            "lineage_id": cand.lineage_id,
            "parent_id": parent.id,
            "child_id": cand.id,
            "reason": reason,
        }
        if edge not in self.accepted_parent_child_edges:
            self.accepted_parent_child_edges.append(edge)

    @staticmethod
    def _record_feasibility_change(cand: Candidate, parent: Candidate) -> None:
        delta = cand.feasibility - parent.feasibility
        cand.feasibility_delta_from_parent = delta
        if delta < 0:
            cand.feasibility_pressure = (
                f"Feasibility fell by {-delta} points ({parent.feasibility}->{cand.feasibility}) "
                "in the last accepted same-lineage replacement. Treat this as refinement "
                "pressure, never as an acceptance gate: improve implementation/evaluation "
                "practicality without sacrificing core Q or changing the mechanism."
            )

    def _retire_qualified(self, cand: Candidate, reason: str) -> None:
        if cand.id in self._historical_signatures:
            return
        self._historical_signatures[cand.id] = cand.signature
        self._historical_lineages[cand.id] = cand.lineage_id
        self.retired_qualified.append({
            "id": cand.id,
            "lineage_id": cand.lineage_id,
            "reason": reason,
            # Disk-only provenance. generation_memory_summary() reads the separate compact
            # signature map, so retaining this text does not increase any later model prompt.
            "idea_text": cand.idea_text,
            "signature": cand.signature.to_json(),
            "non_obviousness": cand.non_obviousness,
            "mechanism_clarity": cand.mechanism_clarity,
            "soundness_100": round(cand.soundness_100, 1),
            "soundness_samples": cand.soundness_samples,
            "soundness_panel_results": cand.soundness_panel_results,
            "feasibility": cand.feasibility,
            "rubric_assessment": cand.rubric_assessment,
        })

    def _has_foreign_historical_duplicate(self, verdict: DiversityVerdict) -> bool:
        """True unless every historical match belongs to an actively matched lineage."""
        if not verdict.historical_duplicate_ids:
            return False
        active_lineages = {
            self.accepted[cid].lineage_id
            for cid in verdict.duplicate_ids
            if cid in self.accepted
        }
        return any(
            self._historical_lineages.get(hid) not in active_lineages
            for hid in verdict.historical_duplicate_ids
        )

    def _push_repair(self, cand: Candidate, t: Thresholds, weights: QualityWeights) -> None:
        self.repair_archive = [c for c in self.repair_archive if c.lineage_id != cand.lineage_id]
        self.repair_archive.append(cand)
        if len(self.repair_archive) > self.repair_capacity:
            ordered = sorted(
                self.repair_archive,
                key=lambda c: (
                    c.repair_priority(
                        t, self.repair_attempt_limit, self.repair_priority_weights,
                    ),
                    c.weighted_quality(weights),
                    -c.generation,
                ),
                reverse=True,
            )
            self.repair_archive = ordered[: self.repair_capacity]
            for evicted in ordered[self.repair_capacity :]:
                self._record_memory(evicted, "repair_capacity_evicted")
        self._sync_repair_alias()

    def remove_repair(self, candidate_id: str) -> None:
        self.repair_archive = [c for c in self.repair_archive if c.id != candidate_id]
        self._sync_repair_alias()

    def needs_consolidation(self, every: int) -> bool:
        return len(self.rejected_pending) >= every

    def set_consolidated_summary(self, summary: str) -> None:
        self.rejected_families_summary = summary.strip()
        self.rejected_pending = []

    # ── admission ──────────────────────────────────────────────────────────────────────
    def consider(
        self, cand: Candidate, verdict: DiversityVerdict, t: Thresholds,
        weights: QualityWeights = EARLY_WEIGHTS, *, repair_parent: Candidate | None = None,
    ) -> str:
        cand.diversity_score = verdict.diversity_score
        if not cand.eval_ok or not verdict.parsed:
            # An evaluator/parser failure must fail closed, but it must not accidentally erase a
            # persistent promising lineage that was temporarily removed for this repair attempt.
            if repair_parent is not None:
                repair_parent.repair_attempts = cand.repair_attempts
                repair_parent.auxiliary_attempts_used = cand.auxiliary_attempts_used
                if repair_parent.repair_attempts < self.repair_attempt_limit:
                    self._push_repair(repair_parent, t, weights)
            self._record_memory(cand, "eval_fail")
            return REJECT_EVAL_FAIL

        # Previously qualified mechanisms have already contributed to Discovery Yield. They can
        # never re-enter as a fresh lineage merely because their full idea left the active store.
        if self._has_foreign_historical_duplicate(verdict):
            if repair_parent is not None and cand.repair_attempts < self.repair_attempt_limit:
                repair_parent.repair_attempts = cand.repair_attempts
                repair_parent.auxiliary_attempts_used = cand.auxiliary_attempts_used
                self._push_repair(repair_parent, t, weights)
                self._record_memory(cand, "repair_child_historical_duplicate_parent_retained")
                return REPAIR_RETAINED
            self._record_memory(cand, "historical_qualified_rediscovery")
            return REJECT_DUPLICATE

        if cand.passes_all(t):
            if not verdict.is_duplicate and verdict.diversity_score >= t.diversity_D:
                self._remember_qualified(cand)
                if len(self.accepted) < self.accepted_capacity:
                    self.accepted[cand.id] = cand
                    self._remember_accepted_version(cand, reason="accepted")
                    return ACCEPTED

                weakest_id, weakest = min(
                    self.accepted.items(),
                    key=lambda item: (
                        item[1].weighted_quality(weights), item[1].feasibility,
                    ),
                )
                if cand.outranks(weakest, weights):
                    self._retire_qualified(weakest, "active_capacity_replaced")
                    del self.accepted[weakest_id]
                    self.accepted[cand.id] = cand
                    self._remember_accepted_version(cand, reason="active_capacity_replaced")
                    return REPLACED
                self._retire_qualified(cand, "active_capacity_not_selected")
                return QUALIFIED_RETIRED

            if len(verdict.duplicate_ids) == 1:
                dup_id = verdict.duplicate_ids[0]
                incumbent = self.accepted.get(dup_id)
                if incumbent is not None and cand.outranks(incumbent, weights):
                    cand.lineage_id = incumbent.lineage_id
                    cand.accepted_refinements_used = incumbent.accepted_refinements_used
                    cand.novelty_branches_used = incumbent.novelty_branches_used
                    self._record_feasibility_change(cand, incumbent)
                    self._remember_qualified(cand)
                    self._retire_qualified(incumbent, "active_duplicate_replaced")
                    del self.accepted[dup_id]
                    self.accepted[cand.id] = cand
                    self._remember_accepted_version(
                        cand, parent=incumbent, reason="active_duplicate_replaced",
                    )
                    return REPLACED
                if repair_parent is not None and cand.repair_attempts < self.repair_attempt_limit:
                    repair_parent.repair_attempts = cand.repair_attempts
                    repair_parent.auxiliary_attempts_used = cand.auxiliary_attempts_used
                    self._push_repair(repair_parent, t, weights)
                    self._record_memory(cand, "repair_child_duplicate_parent_retained")
                    return REPAIR_RETAINED
                self._record_memory(cand, "duplicate")
                return REJECT_DUPLICATE
            if repair_parent is not None and cand.repair_attempts < self.repair_attempt_limit:
                repair_parent.repair_attempts = cand.repair_attempts
                repair_parent.auxiliary_attempts_used = cand.auxiliary_attempts_used
                self._push_repair(repair_parent, t, weights)
                self._record_memory(cand, "repair_child_low_diversity_parent_retained")
                return REPAIR_RETAINED
            self._record_memory(cand, "duplicate_multi_or_low_diversity")
            return REJECT_DUPLICATE

        if cand.near_thresholds(t) and not verdict.is_duplicate and verdict.diversity_score >= t.repair_diversity_D:
            cand.defect = cand.defect_text(t)
            if repair_parent is not None:
                # An attempted repair consumes lineage budget even if the child regresses.
                attempts = cand.repair_attempts
                if attempts >= self.repair_attempt_limit:
                    self._record_memory(cand, "repair_attempt_limit_exhausted")
                    return REJECT_UNSOUND
                parent_copy = repair_parent
                parent_copy.repair_attempts = attempts
                parent_copy.auxiliary_attempts_used = cand.auxiliary_attempts_used
                chosen = (
                    cand if cand.outranks(parent_copy, weights)
                    else parent_copy
                )
                chosen.repair_attempts = attempts
                chosen.auxiliary_attempts_used = cand.auxiliary_attempts_used
                if chosen is parent_copy:
                    chosen.defect = repair_parent.defect
                self._push_repair(chosen, t, weights)
                return REPAIR_RETAINED
            self._push_repair(cand, t, weights)
            return REPAIR_QUEUED

        # A failed repair does not erase its still-promising parent until the lineage budget ends.
        if repair_parent is not None and cand.repair_attempts < self.repair_attempt_limit:
            repair_parent.repair_attempts = cand.repair_attempts
            repair_parent.auxiliary_attempts_used = cand.auxiliary_attempts_used
            self._push_repair(repair_parent, t, weights)
            self._record_memory(cand, "repair_child_regressed_parent_retained")
            return REPAIR_RETAINED

        note = "rejected_family" if verdict.matches_rejected_family else cand.defect_text(t) or "low_quality"
        self._record_memory(cand, note)
        return REJECT_UNSOUND

    def consider_accepted_refinement(
        self, cand: Candidate, *, parent_id: str, verdict: DiversityVerdict,
        t: Thresholds, weights: QualityWeights, nb_not_worse: bool = True,
    ) -> str:
        parent = self.accepted.get(parent_id)
        if parent is None:
            self._record_memory(cand, "accepted_refinement_parent_missing")
            return REJECT_EVAL_FAIL

        if parent.accepted_refinements_used >= self.accepted_refinement_limit:
            self._record_memory(cand, "accepted_refinement_limit_exhausted")
            return KEEP_ACCEPTED

        used = parent.accepted_refinements_used + 1
        parent.accepted_refinements_used = used
        cand.accepted_refinements_used = used
        parent.auxiliary_attempts_used = max(
            parent.auxiliary_attempts_used, cand.auxiliary_attempts_used
        )
        cand.auxiliary_attempts_used = parent.auxiliary_attempts_used
        cand.novelty_branches_used = parent.novelty_branches_used
        cand.lineage_id = parent.lineage_id
        cand.diversity_score = verdict.diversity_score

        valid = (
            cand.eval_ok and verdict.parsed and cand.passes_all(t)
            and not verdict.is_duplicate and verdict.diversity_score >= t.diversity_D
        )
        if valid and nb_not_worse and cand.outranks(parent, weights):
            self._record_feasibility_change(cand, parent)
            self._remember_qualified(cand)
            self._retire_qualified(parent, "accepted_refinement_replaced")
            del self.accepted[parent_id]
            self.accepted[cand.id] = cand
            self._remember_accepted_version(
                cand, parent=parent, reason="accepted_refinement_replaced",
            )
            return REPLACED

        reason = (
            "accepted_refinement_relative_nb_worse"
            if valid and not nb_not_worse
            else "accepted_refinement_not_better_or_failed_gate"
        )
        self._record_memory(cand, reason)
        return KEEP_ACCEPTED

    # ── reporting ──────────────────────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        return {
            "free_generations_completed": self.free_generations_completed,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "auxiliary_candidates_evaluated": (
                self.total_candidates_evaluated - self.free_generations_completed
            ),
            "active_yield": self.yield_count(),
            "discovery_yield": self.discovery_yield(),
            "active_actionable_yield": self.active_actionable_yield(),
            "discovery_actionable_yield": self.discovery_actionable_yield(),
            "feasibility_actionable_min": self.feasibility_actionable_min,
            "repair_archive": len(self.repair_archive),
            "retired_qualified": len(self.retired_qualified),
            "historical_qualified": len(self._historical_signatures),
            "accepted_versions_retained": len(self.accepted_version_pool),
            "accepted_parent_child_edges": len(self.accepted_parent_child_edges),
            "rejected_pending": len(self.rejected_pending),
            "has_consolidated_summary": bool(self.rejected_families_summary),
        }

    @staticmethod
    def _candidate_json(c: Candidate, *, include_text: bool = False) -> dict[str, Any]:
        out = {
            "id": c.id, "lineage_id": c.lineage_id, "operator": c.operator,
            "seed_index": c.seed_index,
            "generation": c.generation, "revision": c.revision, "parent_id": c.parent_id,
            "repair_attempts": c.repair_attempts,
            "accepted_refinements_used": c.accepted_refinements_used,
            "auxiliary_attempts_used": c.auxiliary_attempts_used,
            "novelty_branches_used": c.novelty_branches_used,
            "eval_ok": c.eval_ok,
            "non_obviousness": c.non_obviousness,
            "mechanism_clarity": c.mechanism_clarity,
            "soundness_100": round(c.soundness_100, 1), "feasibility": c.feasibility,
            "soundness_samples": c.soundness_samples,
            "soundness_fatal": c.soundness_fatal,
            "soundness_fatal_votes": c.soundness_fatal_votes,
            "soundness_disputed": c.soundness_disputed,
            "soundness_rationale": c.soundness_rationale,
            "soundness_rationales": c.soundness_rationales,
            "soundness_panel_results": c.soundness_panel_results,
            "rubric_assessment": c.rubric_assessment,
            "feedback_notes": c.feedback_notes,
            "diversity_score": c.diversity_score, "defect": c.defect,
            "feasibility_delta_from_parent": c.feasibility_delta_from_parent,
            "feasibility_pressure": c.feasibility_pressure,
            "signature": c.signature.to_json(),
        }
        if include_text:
            out["idea_text"] = c.idea_text
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "active_yield": self.yield_count(),
            "discovery_yield": self.discovery_yield(),
            "active_actionable_yield": self.active_actionable_yield(),
            "discovery_actionable_yield": self.discovery_actionable_yield(),
            "feasibility_actionable_min": self.feasibility_actionable_min,
            "accepted_capacity": self.accepted_capacity,
            "history_field_char_cap": self.history_field_char_cap,
            "qualified_lineage_ids": sorted(self.qualified_lineages),
            "qualified_lineage_feasibility": self.qualified_lineage_feasibility,
            "search_steps_completed": self.search_steps_completed,
            "free_generations_completed": self.free_generations_completed,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "auxiliary_candidates_evaluated": (
                self.total_candidates_evaluated - self.free_generations_completed
            ),
            "action_counts": self.action_counts,
            "selection_priority_weights": {
                "repair": asdict(self.repair_priority_weights),
                "accepted_refinement": asdict(self.refinement_priority_weights),
            },
            "accepted": [self._candidate_json(c, include_text=True) for c in self.accepted.values()],
            "accepted_lineage_memory": {
                lineage_id: {
                    "current_active_id": next(
                        (
                            candidate.id for candidate in self.accepted.values()
                            if candidate.lineage_id == lineage_id
                        ),
                        None,
                    ),
                    "versions": [
                        self._candidate_json(candidate, include_text=True)
                        for candidate in self.accepted_version_pool.values()
                        if candidate.lineage_id == lineage_id
                    ],
                    "parent_child_edges": [
                        edge for edge in self.accepted_parent_child_edges
                        if edge["lineage_id"] == lineage_id
                    ],
                }
                for lineage_id in sorted({
                    candidate.lineage_id
                    for candidate in self.accepted_version_pool.values()
                })
            },
            "repair_archive": [self._candidate_json(c, include_text=True) for c in self.repair_archive],
            "historical_qualified": self.retired_qualified,
            "retired_qualified": self.retired_qualified,
            "rejected_families_summary": self.rejected_families_summary,
            "rejected_pending": list(self.rejected_pending),
            "generation_memory_summary": self.generation_memory_summary(),
        }
