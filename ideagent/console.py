"""Console formatting helpers for agentic ideation runs.

Rich is optional: when installed, debug/model/progress messages are shown in TRL-like
panels. Without it, the previous plain-text style is preserved.
"""
from __future__ import annotations

import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text as RichText
except ModuleNotFoundError:  # pragma: no cover - exercised implicitly in fallback tests
    Console = None
    Panel = None
    RichText = None


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


_OUTCOME_STYLES = {
    "accepted": "bold green",
    "replaced": "bold green",
    "repair_queued": "bold yellow",
    "repair_retained": "bold yellow",
    "keep_accepted": "dim white",
    "qualified_retired": "dim white",
    "reject_duplicate": "bold red",
    "reject_unsound": "bold red",
    "reject_eval_fail": "bold red",
}


def print_candidate_result(
    *,
    step: str,
    operator: str,
    non_obviousness: int,
    clarity: int,
    soundness: float,
    diversity_score: int,
    outcome: str,
    active_yield: int,
    discovery_yield: int,
    repair_archive_size: int,
) -> None:
    """One line per evaluated candidate. Colored by outcome so a scrolling run is scannable
    at a glance: green = entered/stayed in the active archive, yellow = sent to repair,
    dim = accepted-but-unchanged bookkeeping, red = rejected outright."""
    style = _OUTCOME_STYLES.get(outcome, "white")
    line = (
        f"[{step}] {operator:<13} nb={non_obviousness:>3} cl={clarity:>3} "
        f"snd={soundness:5.1f} div={diversity_score:>3} -> {outcome:<16} "
        f"ACTIVE={active_yield} DISC={discovery_yield} (repair={repair_archive_size})"
    )
    if rich_available():
        _console(stderr=False).print(RichText(line, style=style))
        return
    print(line, flush=True)


def print_warning(message: str) -> None:
    _panel("Warning", message, border_style="bold yellow", stderr=True)
