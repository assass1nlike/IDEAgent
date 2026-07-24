"""Asynchronous JSONL logging for raw agent model responses."""
from __future__ import annotations

import json
import queue
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


AGENT_RESPONSE_ROLES = ("ideator", "critic", "feedback")


class AsyncResponseLogger:
    """Append raw role outputs to JSONL files without blocking generation.

    Config supports two layouts:
    - legacy per-role files: {"ideator": "...", "critic": "...", "feedback": "..."}
    - scoped tree: {"path": "logs/responses"} -> path/topic/target/role.jsonl
    """

    def __init__(
        self,
        paths: Mapping[str, str | Path | None] | None = None,
        *,
        root_path: str | Path | None = None,
    ) -> None:
        self.paths = self._normalize_paths(paths or {})
        self.root_path = self._normalize_root_path(root_path)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._sequence = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "AsyncResponseLogger":
        config = config or {}
        root_path = config.get("path") or config.get("root") or config.get("dir")
        return cls(config, root_path=root_path)

    @property
    def enabled(self) -> bool:
        return bool(self.paths or self.root_path)

    def __enter__(self) -> "AsyncResponseLogger":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("response logger is already closed")
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="agent-response-jsonl-writer",
                daemon=True,
            )
            self._thread.start()

    def log(
        self,
        *,
        role: str,
        event: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._role_enabled(role):
            return
        metadata_dict = dict(metadata or {})
        with self._lock:
            if self._closed:
                return
            self._sequence += 1
            sequence = self._sequence
        path = self._path_for_record(role=role, metadata=metadata_dict)
        if path is None:
            return
        self.start()
        self._queue.put(
            {
                "_path": path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence": sequence,
                "role": role,
                "event": event,
                "text": text,
                "metadata": metadata_dict,
            }
        )

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is None:
            return
        self._queue.put(None)
        self._queue.join()
        thread.join()

    @staticmethod
    def _normalize_paths(paths: Mapping[str, str | Path | None]) -> dict[str, Path]:
        normalized: dict[str, Path] = {}
        for role in AGENT_RESPONSE_ROLES:
            raw = paths.get(role)
            if raw is None:
                continue
            path_text = str(raw).strip()
            if not path_text:
                continue
            normalized[role] = Path(path_text).expanduser()
        return normalized

    @staticmethod
    def _normalize_root_path(path: str | Path | None) -> Path | None:
        if path is None:
            return None
        path_text = str(path).strip()
        if not path_text:
            return None
        return Path(path_text).expanduser()

    def _role_enabled(self, role: str) -> bool:
        if role not in AGENT_RESPONSE_ROLES:
            return False
        return role in self.paths or self.root_path is not None

    def _path_for_record(self, *, role: str, metadata: Mapping[str, Any]) -> Path | None:
        if self.root_path is not None:
            topic_id = self._safe_component(str(metadata.get("topic_id", "_unknown")))
            target_id = self._safe_component(str(metadata.get("target_paper_id", "_unknown")))
            return self.root_path / topic_id / f"ep{target_id}" / f"{role}.jsonl"
        return self.paths.get(role)

    @staticmethod
    def _safe_component(raw: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
        cleaned = cleaned.strip("._")
        return cleaned or "_unknown"

    def _writer_loop(self) -> None:
        handles_by_path: dict[Path, Any] = {}
        disabled_paths: set[Path] = set()

        while True:
            record = self._queue.get()
            try:
                if record is None:
                    return
                role = str(record.get("role", ""))
                path = record.pop("_path", None)
                if not isinstance(path, Path) or path in disabled_paths:
                    continue
                try:
                    if path not in handles_by_path:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        handles_by_path[path] = path.open("a", encoding="utf-8")
                    handle = handles_by_path[path]
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    handle.flush()
                except OSError as exc:
                    disabled_paths.add(path)
                    self._warn(
                        f"response logger failed for role {role!r} at {path}; disabling that path: {exc}"
                    )
            finally:
                self._queue.task_done()
                if record is None:
                    for handle in handles_by_path.values():
                        handle.close()

    @staticmethod
    def _warn(message: str) -> None:
        print(f"[ideagent] warning: {message}", file=sys.stderr)
