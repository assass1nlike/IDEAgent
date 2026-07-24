"""Batched structured diversity judgment over active and historical qualified mechanisms."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ideagent.signature import SemanticSignature

DIVERSITY_JUDGE_SYSTEM = (
    "You are a strict diversity judge for a research-idea archive. Compare the CANDIDATE idea "
    "against (1) ACTIVE ACCEPTED ideas, (2) HISTORICAL QUALIFIED mechanisms that previously "
    "passed every gate but left the active store, and (3) REJECTED memory. Judge by substance -- "
    "the problem attacked and core mechanism -- not surface wording.\n\n"
    "For active and historical qualified ideas, identify the closest entries and whether the "
    "candidate shares the same problem and core mechanism without a substantive new contribution. "
    "Put duplicate active ids in duplicate_ids and duplicate historical ids in "
    "historical_duplicate_ids. Historical mechanisms must never be presented as new discoveries. "
    "Also decide whether the candidate re-treads a known-bad rejected family. Give diversity "
    "0-100 against both qualified pools."
)


def _response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "diversity_verdict",
            "schema": {
                "type": "object",
                "properties": {
                    "closest": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "same_problem": {"type": "boolean"},
                                "same_mechanism": {"type": "boolean"},
                                "substantive_difference": {"type": "boolean"},
                            },
                            "required": ["id", "same_problem", "same_mechanism", "substantive_difference"],
                            "additionalProperties": False,
                        },
                    },
                    "duplicate_ids": {"type": "array", "items": {"type": "string"}},
                    "historical_duplicate_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "diversity_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "matches_rejected_family": {"type": "boolean"},
                    "rejected_family": {"type": "string"},
                },
                "required": [
                    "closest", "duplicate_ids", "historical_duplicate_ids", "diversity_score",
                    "matches_rejected_family", "rejected_family",
                ],
                "additionalProperties": False,
            },
        },
    }


@dataclass
class DiversityVerdict:
    parsed: bool
    diversity_score: int = 0
    duplicate_ids: list[str] = field(default_factory=list)
    historical_duplicate_ids: list[str] = field(default_factory=list)
    matches_rejected_family: bool = False
    rejected_family: str = ""
    closest: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_duplicate(self) -> bool:
        return bool(self.duplicate_ids or self.historical_duplicate_ids)


def _first_json(raw: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_diversity(
    candidate: SemanticSignature,
    *,
    accepted: list[SemanticSignature],
    historical: list[SemanticSignature],
    rejected_summary: str,
    historical_field_char_cap: int,
    client: Any,
    gen_kwargs: dict[str, Any],
    prompt_cache_key: str | None = None,
) -> DiversityVerdict:
    """One batched call. With an empty archive there is nothing to compare against -> fully novel.

    prompt_cache_key should be constant for the whole topic (e.g. f"{topic_id}/judge"). Note
    the CANDIDATE block sits first in the user message and changes every call, so only the
    static system prompt actually benefits from caching here -- still worth a consistent key
    since it costs nothing and every call shares that system message."""
    if not accepted and not historical and not rejected_summary.strip():
        return DiversityVerdict(parsed=True, diversity_score=100)

    accepted_block = "\n".join(s.compact() for s in accepted) or "(none yet)"
    historical_block = (
        "\n".join(s.mechanism_compact(historical_field_char_cap) for s in historical)
        or "(none yet)"
    )
    rejected_block = rejected_summary.strip() or "(none yet)"
    user = (
        f"CANDIDATE:\n{candidate.compact()}\n\n"
        f"ACCEPTED IDEAS (archive):\n{accepted_block}\n\n"
        "HISTORICAL QUALIFIED MECHANISMS (previously counted; never new again):\n"
        f"{historical_block}\n\n"
        f"REJECTED MEMORY (known-bad families / duplicate patterns / fatal flaws):\n{rejected_block}\n\n"
        "Return the structured diversity verdict. Only use ids from ACCEPTED IDEAS in duplicate_ids. "
        "Only use ids from HISTORICAL QUALIFIED MECHANISMS in historical_duplicate_ids."
    )
    call_kwargs = dict(gen_kwargs)
    if prompt_cache_key:
        call_kwargs["prompt_cache_key"] = prompt_cache_key
    obj = _first_json(
        client.generate(
            [{"role": "system", "content": DIVERSITY_JUDGE_SYSTEM}, {"role": "user", "content": user}],
            response_format=_response_format(),
            **call_kwargs,
        )
    )
    if obj is None:
        return DiversityVerdict(parsed=False)  # fail closed
    try:
        score = int(obj["diversity_score"])
        raw_dup = obj["duplicate_ids"]
        raw_historical_dup = obj["historical_duplicate_ids"]
        raw_closest = obj["closest"]
        matches_rejected = obj["matches_rejected_family"]
        rejected_family = obj["rejected_family"]
        if (
            not isinstance(raw_dup, list)
            or not isinstance(raw_historical_dup, list)
            or not isinstance(raw_closest, list)
            or not isinstance(matches_rejected, bool)
            or not isinstance(rejected_family, str)
        ):
            raise TypeError
        accepted_ids = {s.id for s in accepted}
        historical_ids = {s.id for s in historical}
        dup = [str(i) for i in raw_dup if str(i) in accepted_ids]
        historical_dup = [str(i) for i in raw_historical_dup if str(i) in historical_ids]
        return DiversityVerdict(
            parsed=True,
            diversity_score=max(0, min(100, score)),
            duplicate_ids=dup,
            historical_duplicate_ids=historical_dup,
            matches_rejected_family=matches_rejected,
            rejected_family=rejected_family.strip(),
            closest=raw_closest,
        )
    except (KeyError, TypeError, ValueError):
        return DiversityVerdict(parsed=False)
