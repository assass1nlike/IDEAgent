"""Shared JSONL I/O and config-loading helpers used across the runners and evaluators."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single record to a JSONL file (file must already exist)."""
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_jsonl(path: str | Path) -> None:
    """Create (or truncate) an output JSONL file, ensuring parent dirs exist."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")


def prepare_output_dir(
    output_dir: str | Path,
    *,
    resume: bool,
    init_topic_idx: int = 0,
    wait_seconds: int = 60,
) -> None:
    """resume=True: leave the directory alone (caller's own completed-topic-skip logic
    handles the rest); just ensure it exists.

    resume=False means an explicit RESTART: the entire output_dir (final_idea_cores*.jsonl
    AND responses/) is wiped and generation starts clean. Since this is destructive, warn
    on screen and wait `wait_seconds` before deleting, so a run started with resume=False
    by mistake can still be cancelled (Ctrl+C).

    init_topic_idx > 0 (0-indexed: skips the first init_topic_idx topics) combined with
    resume=False is a sharper danger than an ordinary restart -- it usually means one of
    several parallel runs, each covering a different topic slice against the same
    output_dir, was accidentally started with resume=false. Wiping output_dir here erases
    every OTHER slice's completed topics too, not just this run's own. Gets an extra
    explicit warning and doubles the wait before deleting anything."""
    output_dir = Path(output_dir)
    if resume:
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        mid_slice = init_topic_idx > 0
        effective_wait = wait_seconds * 2 if mid_slice else wait_seconds
        message = (
            f"[RESTART WARNING] resume=false: {output_dir} will be COMPLETELY WIPED "
            f"(final_idea_cores*.jsonl and responses/) in {effective_wait} seconds. "
            "Press Ctrl+C now to cancel."
        )
        if mid_slice:
            message += (
                f"\n[RESTART WARNING] init_topic_idx={init_topic_idx} (0-indexed -- skips "
                f"the first {init_topic_idx} topic(s) in the input list) is set, which "
                "usually means this is one of several parallel runs covering different "
                "topic slices against the same output_dir. Restarting here will erase "
                "every OTHER slice's completed topics too, not just this run's own."
            )
        print(message, file=sys.stderr)
        time.sleep(effective_wait)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return {} if loaded is None else dict(loaded)
    except ModuleNotFoundError:
        pass

    # Small fallback parser for a simple nested mapping when PyYAML is unavailable.
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Cannot parse config line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root
