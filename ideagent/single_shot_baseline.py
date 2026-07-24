"""Response-format helper shared by the single-call baseline pipeline.

The generator receives the same background context and substantive ideation criteria used by
the agentic/QD arm, then returns all N ideas in one structured response.  There is deliberately
no critic, evaluator, archive, repair, refinement, or second generator turn.
"""
from __future__ import annotations

from typing import Any


def single_shot_response_format(n_ideas: int) -> dict[str, Any]:
    """A dynamic schema that makes the one response contain exactly ``n_ideas`` items."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"single_shot_research_ideas_{n_ideas}",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ideas": {
                        "type": "array",
                        "minItems": n_ideas,
                        "maxItems": n_ideas,
                        "items": {
                            "type": "object",
                            "properties": {"idea_text": {"type": "string"}},
                            "required": ["idea_text"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["ideas"],
                "additionalProperties": False,
            },
        },
    }
