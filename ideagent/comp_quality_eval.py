"""Comprehensive quality evaluator.

Scores the same 5 metrics as quality_eval.py (non-obviousness, soundness,
mechanism_clarity_specificity, feasibility, significance) but against the ideator's full
final response text for each idea, instead of the compressed final_idea_core summary.

Reads responses/<topic_id>/ep<episode_id>/ideator.jsonl, where episode_id = idea_idx + 1
(confirmed against final_idea_cores.jsonl's own final_idea_core.episode_id field --
episode folder "ep1" holds items[0], "ep2" holds items[1], etc.), and takes the
LATEST-`timestamp` "open"/"respond" record's text -- the ideator's final, fully-refined
response after all critic rounds in that episode -- as that idea's full content. Each
round's ideator.jsonl also holds a "core" event record (that round's extracted
final_idea_core) interleaved with the text records; those are excluded here since they
don't carry idea text. Some files are the concatenation of an earlier run's records plus
a later (e.g. resumed) run's records appended after, where `sequence` numbering does not
necessarily stay monotonic across the join, so `timestamp` (not `sequence`) is the only
reliable way to find the true most-recent record.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ideagent.console import print_warning
from ideagent.quality_eval import (
    AdditionalQualityEvaluator,
    GeminiAdditionalQualityEvaluator,
    _METRIC_LABELS,
)
from ideagent.utils import read_jsonl


def discover_topics(responses_root: Path) -> list[tuple[str, int]]:
    """List (topic_id, n_ideas) pairs directly from the responses/ directory tree --
    topic_id = each subdirectory name; n_ideas = the highest episode-folder number
    found under it (episode folders are 1-based: "ep1".."ep<n_ideas>")."""
    topics = []
    for topic_dir in sorted(responses_root.iterdir()):
        if not topic_dir.is_dir():
            continue
        episode_ids = [
            int(d.name[2:]) for d in topic_dir.iterdir()
            if d.is_dir() and d.name.startswith("ep") and d.name[2:].isdigit()
        ]
        if not episode_ids:
            continue
        topics.append((topic_dir.name, max(episode_ids)))
    return topics


def _record_timestamp(record: dict[str, Any]) -> datetime:
    try:
        return datetime.fromisoformat(str(record.get("timestamp", "")))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def load_final_ideator_text(responses_root: Path, topic_id: str, idea_idx: int) -> str | None:
    """responses/<topic_id>/ep<idea_idx + 1>/ideator.jsonl -> latest-timestamp text.

    Only "open"/"respond" records carry idea text; "core" records (that round's
    extracted final_idea_core, logged alongside) are excluded from the search."""
    path = responses_root / topic_id / f"ep{idea_idx + 1}" / "ideator.jsonl"
    if not path.exists():
        return None
    records = [r for r in read_jsonl(path) if r.get("event") in ("open", "respond")]
    if not records:
        return None
    last = max(records, key=_record_timestamp)
    text = str(last.get("text", "")).strip()
    return text or None


def extract_ideas_from_responses(
    responses_root: Path, topic_id: str, n_ideas: int
) -> list[dict[str, Any]]:
    ideas = []
    for idx in range(n_ideas):
        text = load_final_ideator_text(responses_root, topic_id, idx)
        if text is None:
            print_warning(
                f"[comp_quality_eval] {topic_id}: no final ideator response for idea "
                f"{idx} (episode {idx + 1}); skipping idea."
            )
            continue
        ideas.append({"idx": idx, "text": text})
    return ideas


def extract_ideas_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the common untouched full-text field used by all new generation methods."""

    ideas: list[dict[str, Any]] = []
    for idx, item in enumerate(record.get("items", [])):
        if not isinstance(item, dict):
            continue
        text = str(item.get("idea_text", "")).strip()
        if text:
            ideas.append({"idx": idx, "text": text})
    return ideas


def _build_comp_user_message(idea: dict[str, Any], metric: str) -> str:
    return (
        f"<IDEA>\n{idea['text']}\n</IDEA>\n\n"
        f"Score this research idea's {_METRIC_LABELS[metric]}. "
        "Return strict JSON and nothing else."
    )


class ComprehensiveQualityEvaluator(AdditionalQualityEvaluator):
    def _build_idea_user_message(self, idea: dict[str, Any], metric: str) -> str:
        return _build_comp_user_message(idea, metric)


class GeminiComprehensiveQualityEvaluator(GeminiAdditionalQualityEvaluator):
    def _build_idea_user_message(self, idea: dict[str, Any], metric: str) -> str:
        return _build_comp_user_message(idea, metric)
