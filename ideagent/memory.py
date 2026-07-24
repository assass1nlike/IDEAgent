"""Idea core extraction for the agentic ideation system.

Every extraction produces the idea's FULL core in one call, directly from the current
idea text -- there is no separate lightweight per-round core and no history-aggregation
synthesis step. The same field set is used whether this is round 1 or the final round;
the last round's core is simply used as the episode's final result."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

from ideagent.console import print_warning


_FINAL_IDEA_CORE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "final_idea_core",
        "schema": {
            "type": "object",
            "properties": {
                "core_claim":       {"type": "string"},
                "idea_summary":     {"type": "string"},
                "mechanism":        {"type": "string"},
                "problem_targeted": {"type": "string"},
                "source_gap":       {"type": "string"},
                "expected_effect":  {"type": "string"},
                "major_keywords":   {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                "generic_theme":    {"type": "string"},
                "parent_theme":     {"type": "string"},
            },
            "required": [
                "core_claim", "idea_summary", "mechanism", "problem_targeted", "source_gap",
                "expected_effect", "major_keywords", "generic_theme", "parent_theme",
            ],
            "additionalProperties": False,
        },
    },
}


class MemoryManager:
    """Runs full idea core extraction."""

    def __init__(
        self,
        *,
        steno_client: Any,
        gen_kwargs: dict[str, Any],
        max_parsing_failed_retries: int | None = None,
    ) -> None:
        self.steno_client = steno_client
        self.gen_kwargs = gen_kwargs
        self.max_parsing_failed_retries = max(0, int(max_parsing_failed_retries or 0))

    def _generate_and_parse(
        self,
        messages: list[dict[str, str]],
        parser: Callable[[str], Any],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        attempts = 1 + self.max_parsing_failed_retries
        gen_kwargs = dict(self.gen_kwargs)
        if response_format is not None:
            gen_kwargs["response_format"] = response_format
        for _ in range(attempts):
            raw = self.steno_client.generate(messages, **gen_kwargs)
            parsed = parser(raw)
            if parsed is not None:
                return parsed
        return None

    def extract_final_idea_core(
        self,
        *,
        topic_id: str,
        target_paper_id: str,
        round_no: int,
        background_context: str,
        latest_ideator_turn: str,
    ) -> "FinalIdeaCore":
        """Extract the idea's full core directly from the current idea text -- every field
        the final artifact needs, in one call, read fresh from this text alone. Called every
        round; the last round's result is simply used as the episode's final core, with no
        separate history-aggregation step."""
        from ideagent.agent_prompts import build_final_idea_core_messages

        episode_id = target_paper_id
        messages = build_final_idea_core_messages(
            background_context=background_context,
            latest_ideator_turn=latest_ideator_turn,
        )
        parsed = self._generate_and_parse(
            messages,
            lambda raw: final_idea_core_from_obj(_extract_json(raw), episode_id=episode_id),
            response_format=_FINAL_IDEA_CORE_RESPONSE_FORMAT,
        )
        if parsed is not None:
            return parsed
        print_warning(
            f"steno core extraction failed for {topic_id}:{target_paper_id}:r{round_no}; "
            "using fallback."
        )
        return FinalIdeaCore(
            episode_id=episode_id,
            core_claim=latest_ideator_turn.strip()[:500] or "(empty ideator response)",
            idea_summary="",
            mechanism="",
            problem_targeted="",
            source_gap="",
            expected_effect="",
            major_keywords=[],
            generic_theme="",
            parent_theme="",
        )


@dataclass
class FinalIdeaCore:
    """The idea's core. Same field set every time it's extracted -- mid-round or as the
    episode's final result -- there is no separate lightweight/rich distinction anymore."""

    episode_id: str            # 1-based index within the topic
    core_claim: str            # one-line claim: what the idea claims to do right now
    idea_summary: str          # 4-5 structured sentences describing the idea as it stands
    mechanism: str             # the central causal lever
    problem_targeted: str      # the specific problem this idea addresses
    source_gap: str            # plain-language background limitation/tension this idea attacked
    expected_effect: str       # the expected measurable or qualitative outcome
    major_keywords: list[str]  # 5 key terms, concepts, named techniques
    generic_theme: str         # specific territory this idea occupies (leaf level)
    parent_theme: str          # branch this idea belongs to (3-8 words, conference-track granularity)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------

def final_idea_core_from_obj(
    item: Any,
    *,
    episode_id: str,
) -> "FinalIdeaCore | None":
    if not isinstance(item, dict):
        return None
    core_claim = str(item.get("core_claim", "")).strip()
    if not core_claim:
        return None
    return FinalIdeaCore(
        episode_id=str(item.get("episode_id", episode_id)).strip() or episode_id,
        core_claim=core_claim,
        idea_summary=str(item.get("idea_summary", "")).strip(),
        mechanism=str(item.get("mechanism", "")).strip(),
        problem_targeted=str(item.get("problem_targeted", "")).strip(),
        source_gap=str(item.get("source_gap", "")).strip(),
        expected_effect=str(item.get("expected_effect", "")).strip(),
        major_keywords=_string_list(item.get("major_keywords"), limit=5),
        generic_theme=str(item.get("generic_theme", "")).strip(),
        parent_theme=str(item.get("parent_theme", "")).strip(),
    )


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value[:limit] if str(item).strip()]


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    for candidate in (stripped, _first_bracket(stripped)):
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _first_bracket(text: str) -> str | None:
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start is None:
        return None
    opener = text[start]
    closer = "]" if opener == "[" else "}"
    depth = 0
    for j in range(start, len(text)):
        if text[j] == opener:
            depth += 1
        elif text[j] == closer:
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    return None
