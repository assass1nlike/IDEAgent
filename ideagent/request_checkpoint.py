"""Durable successful API responses for deterministic request replay."""
from __future__ import annotations

from collections import defaultdict
import functools
import contextvars
import importlib
import os
import inspect
import copy
import sys
from importlib.metadata import distributions
import hashlib
import json
from pathlib import Path
import sqlite3
import threading


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class RequestJournal:
    def __init__(self, path, identity, *, resume=False):
        import fcntl

        self.path = Path(path)
        if resume and not self.path.exists():
            raise ValueError("This run has no request checkpoint; start a new run with checkpoint-enabled code")
        if not resume and self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._process_lock = self.path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self._process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._process_lock.close()
            raise RuntimeError("Another process is using this request checkpoint") from None
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS protocol (identity TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT NOT NULL, occurrence INTEGER NOT NULL, request TEXT NOT NULL, response TEXT, PRIMARY KEY(key, occurrence))')
        self.db.execute('CREATE TABLE IF NOT EXISTS responses (key TEXT NOT NULL, occurrence INTEGER NOT NULL, position INTEGER NOT NULL, response TEXT NOT NULL, PRIMARY KEY(key, occurrence, position))')
        previous = self.db.execute('SELECT identity FROM protocol').fetchone()
        expected = canonical(identity)
        if previous and previous[0] != expected:
            self.close()
            raise ValueError("Resume configuration differs: input, source, environment, model, endpoint, seed and generation parameters must match")
        if previous is None:
            self.db.execute('INSERT INTO protocol VALUES (?)', (expected,))
        self.db.commit()
        self.lock = threading.Lock()
        self.occurrences = defaultdict(int)

    def reserve(self, request):
        serialized = canonical(request)
        key = hashlib.sha256(serialized.encode()).hexdigest()
        with self.lock:
            occurrence = self.occurrences[key]
            self.occurrences[key] += 1
            row = self.db.execute('SELECT response FROM calls WHERE key=? AND occurrence=?', (key, occurrence)).fetchone()
            if row is None:
                self.db.execute('INSERT INTO calls VALUES (?, ?, ?, NULL)', (key, occurrence, serialized))
                self.db.commit()
            return (key, occurrence), json.loads(row[0]) if row and row[0] is not None else None

    def complete(self, ticket, response):
        with self.lock:
            self.db.execute('UPDATE calls SET response=? WHERE key=? AND occurrence=?', (canonical(response), *ticket))
            self.db.commit()

    def raw_response(self, ticket, position):
        with self.lock:
            row = self.db.execute('SELECT response FROM responses WHERE key=? AND occurrence=? AND position=?', (*ticket, position)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if value["class"] == "dict":
            return value["data"]
        cls = getattr(importlib.import_module(value["module"]), value["class"])
        return cls.model_validate(value["data"])

    def save_response(self, ticket, position, response):
        value = {"module": type(response).__module__, "class": type(response).__name__,
                 "data": response if isinstance(response, dict) else response.model_dump(mode="json")}
        with self.lock:
            self.db.execute('INSERT INTO responses VALUES (?, ?, ?, ?)', (*ticket, position, canonical(value)))
            self.db.commit()

    def close(self):
        self.db.close()
        self._process_lock.close()


_current = None


def configure(journal):
    global _current
    _current = journal


def current():
    return _current


_active_request = contextvars.ContextVar("checkpoint_request", default=None)


def request_response(send):
    """Persist the SDK response before text extraction, including parsing retries."""
    active = _active_request.get()
    if active is None:
        return send()
    journal, ticket, cursor = active
    response = journal.raw_response(ticket, cursor[0])
    if response is None:
        response = send()
        cursor[1] = False
        journal.save_response(ticket, cursor[0], response)
    cursor[0] += 1
    return response


async def request_response_async(send):
    active = _active_request.get()
    if active is None:
        return await send()
    journal, ticket, cursor = active
    response = journal.raw_response(ticket, cursor[0])
    if response is None:
        response = await send()
        cursor[1] = False
        journal.save_response(ticket, cursor[0], response)
    cursor[0] += 1
    return response


def checkpointed(fn):
    """Journal one backend request, outside its existing transient-error retry loop."""
    def prepare(client, messages, kwargs):
        journal = current()
        if journal is None:
            return None, None, None
        ticket, response = journal.reserve({
            "backend": type(client).__name__, "model": client.model_id,
            "role": getattr(client, "checkpoint_role", ""),
            "method": fn.__name__, "messages": messages, "kwargs": kwargs,
        })
        if response is not None:
            client._set_response_metadata({**(response["metadata"] or {}), "checkpoint_replay": True})
        return journal, ticket, response

    def complete(client, journal, ticket, output):
        if journal is not None:
            active = _active_request.get()
            if active is not None and active[2][0] and active[2][1]:
                client._set_response_metadata({**(client.get_last_response_metadata() or {}), "checkpoint_replay": True})
            journal.complete(ticket, {"output": output, "metadata": client.get_last_response_metadata()})
        return output

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_call(client, messages, **kwargs):
            journal, ticket, response = prepare(client, messages, kwargs)
            if response is not None:
                return response["output"]
            token = _active_request.set((journal, ticket, [0, True]) if journal is not None else None)
            try:
                output = await fn(client, messages, **kwargs)
                return complete(client, journal, ticket, output)
            finally:
                _active_request.reset(token)
        return async_call

    @functools.wraps(fn)
    def call(client, messages, **kwargs):
        journal, ticket, response = prepare(client, messages, kwargs)
        if response is not None:
            return response["output"]
        token = _active_request.set((journal, ticket, [0, True]) if journal is not None else None)
        try:
            output = fn(client, messages, **kwargs)
            return complete(client, journal, ticket, output)
        finally:
            _active_request.reset(token)
    return call


def prepare_checkpoint(config, root, *, resume=False):
    """Validate a run and put reconstructed artifacts in a separate attempt directory."""
    source_root = Path(__file__).resolve().parents[1]
    protocol = copy.deepcopy(config)
    protocol["data"].pop("output_dir", None)
    input_file = Path(protocol["data"].pop("input_jsonl"))
    identity = {
        "version": 1, "config": protocol,
        "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
        "python": sys.version,
        "provider_environment": {key: hashlib.sha256(os.environ[key].encode()).hexdigest()
                                 for key in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "GOOGLE_GENAI_USE_VERTEXAI",
                                             "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION") if key in os.environ},
        "packages": sorted((d.metadata["Name"], d.version) for d in distributions()),
        "source": {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in ("ideagent", "scripts") for p in sorted((source_root / folder).glob("*.py"))},
    }
    journal = RequestJournal(Path(root) / "requests.sqlite", identity, resume=resume)
    try:
        attempts = Path(root) / "attempts"
        number = max((int(p.name) for p in attempts.glob("*") if p.is_dir() and p.name.isdigit()), default=0) + 1
        attempt = attempts / f"{number:04d}"
        attempt.mkdir(parents=True)
        config = copy.deepcopy(config)
        config["data"]["output_dir"] = str(attempt)
        # Each attempt is new; preserve its config and trace while rebuilding all topics.
        config.setdefault("run", {})["resume"] = True
        (attempt / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        return journal, config
    except BaseException:
        journal.close()
        raise
