"""Compact structured semantic signature of an idea.

The signature is the ONLY representation of an idea sent to models during the search (full
candidate text lives on disk). Five fields, each capped, so accepted/rejected memory stays
compact and prompt cost does not grow with idea length.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any

SIGNATURE_FIELDS = ("problem", "core_mechanism", "novel_move", "key_assumption", "expected_effect")

SIGNATURE_SYSTEM = (
    "Extract a COMPACT structured semantic signature of the research idea below. Be terse and "
    "concrete -- each field one tight sentence, no hedging, no restating the prompt. Fields:\n"
    "- problem: the specific problem the idea targets.\n"
    "- core_mechanism: the central causal lever / how it works (the actual method).\n"
    "- novel_move: the one non-obvious step that distinguishes it from prior work.\n"
    "- key_assumption: the single load-bearing assumption its soundness depends on.\n"
    "- expected_effect: the concrete measurable/qualitative outcome it predicts."
)

SIGNATURE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "signature",
        "schema": {
            "type": "object",
            "properties": {f: {"type": "string"} for f in SIGNATURE_FIELDS},
            "required": list(SIGNATURE_FIELDS),
            "additionalProperties": False,
        },
    },
}


@dataclass
class SemanticSignature:
    id: str
    problem: str
    core_mechanism: str
    novel_move: str
    key_assumption: str
    expected_effect: str

    def compact(self) -> str:
        return (
            f"[{self.id}] problem: {self.problem} | mechanism: {self.core_mechanism} | "
            f"novel_move: {self.novel_move} | assumption: {self.key_assumption} | "
            f"effect: {self.expected_effect}"
        )

    def mechanism_compact(self, char_cap: int = 120) -> str:
        """Small fingerprint for persistent semantic deduplication.

        Historical accepted ideas need problem/mechanism identity, not their full assumptions and
        predicted effects. Capping each retained field keeps worst-case prompt growth bounded while
        preserving the information used by the novelty judge.
        """
        return (
            f"[{self.id}] problem: {_cap(self.problem, char_cap)} | "
            f"mechanism: {_cap(self.core_mechanism, char_cap)} | "
            f"novel_move: {_cap(self.novel_move, char_cap)}"
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def signature_to_final_idea_core(
    signature: SemanticSignature,
    *,
    idea_text: str,
    episode_id: str | None = None,
) -> dict[str, Any]:
    """Render the same evaluator-compatible core used by Yield/QD outputs.

    The untouched proposal remains in ``idea_text`` at the item level and is what new evaluators
    should score. This compact object exists for legacy tools and is intentionally derived from
    the same signature representation across generation methods.
    """

    return {
        "episode_id": episode_id or signature.id,
        "core_claim": signature.novel_move,
        "idea_summary": idea_text,
        "mechanism": signature.core_mechanism,
        "problem_targeted": signature.problem,
        "source_gap": signature.key_assumption,
        "expected_effect": signature.expected_effect,
        "major_keywords": [],
        "generic_theme": "",
        "parent_theme": "",
    }


def _cap(text: str, char_cap: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= char_cap else text[: char_cap - 1].rstrip() + "…"


def _first_json(raw: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def extract_signature(
    idea_text: str,
    *,
    idea_id: str,
    client: Any,
    gen_kwargs: dict[str, Any],
    char_cap: int = 240,
) -> SemanticSignature | None:
    """Extract a capped signature. Returns None on parse failure so the caller can fail closed
    (a candidate whose signature can't be extracted is never accepted)."""
    messages = [
        {"role": "system", "content": SIGNATURE_SYSTEM},
        {"role": "user", "content": idea_text.strip()},
    ]
    obj = _first_json(client.generate(messages, response_format=SIGNATURE_RESPONSE_FORMAT, **gen_kwargs))
    if not all(obj.get(f) for f in SIGNATURE_FIELDS):
        return None
    return SemanticSignature(
        id=idea_id,
        **{f: _cap(obj[f], char_cap) for f in SIGNATURE_FIELDS},
    )
