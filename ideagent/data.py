"""Input parsing for the agentic ideation system.

Reads raw topic groups (each topic = N background papers + M target papers) and renders the
background context block shown to the agents. This is the only data layer the agentic loop
needs; the δ-Mem preprocessed-episode loaders are intentionally not carried over here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ideagent.utils import read_jsonl


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    introduction: str | None = None
    conclusion: str | None = None
    full_text: str | None = None


def _require_string(record: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = record.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"Expected non-empty string field `{key}`")
    return value


def _optional_string(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Paper field `{key}` must be a string when present")
    return value


def parse_paper(record: dict[str, Any], *, title_optional: bool = False) -> Paper:
    if not isinstance(record, dict):
        raise ValueError("Paper entries must be objects")
    title = record.get("title", "")
    if title_optional and title is None:
        title = ""
    if not isinstance(title, str):
        raise ValueError("Paper title must be a string")
    return Paper(
        paper_id=_require_string(record, "paper_id"),
        title=title,
        abstract=_require_string(record, "abstract"),
        introduction=_optional_string(record, "introduction"),
        conclusion=_optional_string(record, "conclusion"),
        full_text=_optional_string(record, "full_text"),
    )


def paper_training_text(paper: Paper) -> str:
    if paper.full_text and paper.full_text.strip():
        return paper.full_text.strip()
    pieces = []
    if paper.title.strip():
        pieces.append(f"Title: {paper.title.strip()}")
    if paper.abstract.strip():
        pieces.append(f"Abstract: {paper.abstract.strip()}")
    if paper.introduction and paper.introduction.strip():
        pieces.append(f"Introduction:\n{paper.introduction.strip()}")
    if paper.conclusion and paper.conclusion.strip():
        pieces.append(f"Conclusion:\n{paper.conclusion.strip()}")
    return "\n\n".join(pieces)


def truncate_by_tokens_rough(text: str, max_tokens: int | None) -> str:
    if max_tokens is None or max_tokens <= 0:
        return text
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def build_background_context(
    background_papers: list[Paper],
    *,
    max_background_tokens: int | None = None,
) -> str:
    if not background_papers:
        raise ValueError("background_papers must be non-empty")
    rendered: list[str] = ["<BACKGROUND>"]
    token_budget = max_background_tokens
    for index, paper in enumerate(background_papers, start=1):
        text = paper_training_text(paper)
        if token_budget is not None and token_budget > 0:
            per_paper_budget = max(1, token_budget // len(background_papers))
            text = truncate_by_tokens_rough(text, per_paper_budget)
        rendered.extend(
            [
                f"[Paper {index}]",
                f"Paper ID: {paper.paper_id}",
                f"Title: {paper.title.strip()}" if paper.title.strip() else "Title:",
                "Text:",
                text,
                "",
            ]
        )
    rendered.append("</BACKGROUND>")
    return "\n".join(rendered)


def load_raw_topic_groups(
    path: str | Path,
) -> list[dict[str, Any]]:
    groups = read_jsonl(path)
    for group in groups:
        _require_string(group, "topic_id")
        background = group.get("background_papers")
        targets = group.get("target_papers")
        if not isinstance(background, list):
            raise ValueError("Each topic group must contain background_papers list")
        if not background:
            raise ValueError("background_papers must be non-empty")
        if not isinstance(targets, list) or not targets:
            raise ValueError("Each topic group must contain non-empty target_papers list")
        for paper in background:
            parse_paper(paper)
        for paper in targets:
            parse_paper(paper, title_optional=True)
    return groups
