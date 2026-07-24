"""Console formatting helpers for agentic ideation runs.

Rich is optional: when installed, debug/model/progress messages are shown in TRL-like
panels. Without it, the previous plain-text style is preserved.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text as RichText
except ModuleNotFoundError:  # pragma: no cover - exercised implicitly in fallback tests
    Console = None
    Panel = None
    RichText = None


_MODEL_STYLES = {
    "IDEATOR": "bold green",
    "CRITIC": "bold magenta",
    "FEEDBACK": "bold yellow",
    "OPENING CRITIC": "bold magenta",
    "FINAL IDEA": "bold blue",
    "ABSTRACT": "bold cyan",
}


def rich_available() -> bool:
    return Console is not None and Panel is not None


def _console(*, stderr: bool = False):
    if Console is None:
        return None
    return Console(file=sys.stderr if stderr else sys.stdout)


def _panel(title: str, body: str, *, border_style: str = "cyan", stderr: bool = False) -> None:
    if rich_available():
        _console(stderr=stderr).print(
            Panel(body, title=title, border_style=border_style, expand=True)
        )
        return
    stream = sys.stderr if stderr else sys.stdout
    print(f"[{title}] {body}", file=stream, flush=True)


def print_debug_event(message: str) -> None:
    _panel("Debug", message, border_style="dim cyan")


def print_model_output(label: str, text: str) -> None:
    title = f"Model Output - {label}"
    body = (text or "").strip() or "(empty output)"
    style = _MODEL_STYLES.get(label.split(" (", 1)[0], "bold white")
    if rich_available():
        # Wrap in RichText so square-bracket sequences in model output are not
        # parsed as Rich markup (e.g. "[/SCOPE 1]" would otherwise crash).
        _console(stderr=False).print(
            Panel(RichText(body), title=title, border_style=style, expand=True)
        )
        return
    indented = textwrap.indent(body, "        ")
    print(f"[debug]   {label}:\n{indented}", flush=True)


def print_feedback_guidance(
    *,
    label: str,
    verdict: dict[str, Any],
    guidance_text: str,
) -> None:
    """Print the critic-facing feedback signal without requiring raw model-output logging."""
    mode = str(verdict.get("critic_guidance_mode", "none")).strip() or "none"
    lines = [
        f"mode: {mode}",
    ]
    hint = str(verdict.get("hint_for_critic", "")).strip()
    expected = str(verdict.get("expected_ideator_direction", "")).strip()
    remark = str(verdict.get("critic_positive_remark", "")).strip()
    if hint:
        lines.extend(["", "hint_for_critic:", textwrap.indent(hint, "  ")])
    if remark:
        lines.extend(["", "critic_positive_remark:", textwrap.indent(remark, "  ")])
    if expected:
        lines.extend(["", "expected_ideator_direction:", textwrap.indent(expected, "  ")])
    if guidance_text.strip() and not any((hint, remark, expected)):
        lines.extend(["", "critic_private_payload:", textwrap.indent(guidance_text.strip(), "  ")])
    _panel(f"Feedback Guidance - {label}", "\n".join(lines), border_style="bold yellow")


def print_feedback_scores(
    *,
    label: str,
    verdict: dict[str, Any],
) -> None:
    """Print every feedback score axis for a round, independent of guidance emission."""
    lines = [
        f"phase: {verdict.get('discussion_phase', '(unknown)')}",
        f"guidance_mode: {verdict.get('critic_guidance_mode', 'none')}",
    ]
    bottleneck = str(verdict.get("primary_bottleneck", "")).strip()
    if bottleneck:
        lines.append(f"primary_bottleneck: {bottleneck}")
    rubrics = verdict.get("rubric_assessment", [])
    if isinstance(rubrics, list) and rubrics:
        lines.extend(["", "rubric_feedback:"])
        for item in rubrics:
            if not isinstance(item, dict):
                continue
            rubric_id = str(item.get("id", "")).strip()
            if not rubric_id:
                continue
            feedback_text = str(
                item.get("critic_pressure") or item.get("evidence") or ""
            ).strip()
            if not feedback_text:
                feedback_text = "(no textual feedback)"
            lines.append(f"  {rubric_id}: {feedback_text} [{item.get('score', '(unknown)')}]")
    _panel(f"Feedback Scores - {label}", "\n".join(lines), border_style="bold yellow")


def print_soundness_scores(
    *,
    label: str,
    soundness_results: list[dict[str, Any]],
    fatal_threshold: int,
) -> None:
    """Print every independent sample from the standalone soundness judge for a round,
    styled to match print_feedback_scores since it's the second half of the same round's
    evaluation (feedback's rubrics being the first half)."""
    lines = [f"fatal_threshold: <= {fatal_threshold}/9"]
    if soundness_results:
        lines.extend(["", "soundness_samples:"])
        for i, r in enumerate(soundness_results, 1):
            score = r.get("score", "(unknown)")
            rationale = str(r.get("rationale", "")).strip() or "(no textual feedback)"
            tag = " [FATAL]" if isinstance(score, int) and score <= fatal_threshold else ""
            lines.append(f"  ({i}) {rationale} [{score}/9]{tag}")
    else:
        lines.extend(["", "soundness_samples: (none parsed)"])
    _panel(f"Feedback Scores - {label}", "\n".join(lines), border_style="bold yellow")


def print_item_result(item: dict[str, Any]) -> None:
    summary = item.get("final_idea_core") or {}
    episode_id = summary.get("episode_id", "?")
    title = f"Result - episode {episode_id}"
    body = "\n".join(
        [
            f"core_claim: {summary.get('core_claim', '(none)')}",
            f"generic_theme: {summary.get('generic_theme', '(none)')}",
            f"parent_theme: {summary.get('parent_theme', '(none)')}",
            f"mechanism: {summary.get('mechanism', '(none)')}",
        ]
    )
    _panel(title, body, border_style="bold green")


def print_run_complete(*, mode: str, record_count: int, memory_dir: str | Path) -> None:
    _panel(
        f"{mode} complete",
        f"topics: {record_count}\nmemory_dir: {memory_dir}",
        border_style="bold green",
    )


def print_warning(message: str) -> None:
    _panel("Warning", message, border_style="bold yellow", stderr=True)
