"""Post-hoc quality evaluator for generated research ideas.

Scores five independent per-idea metrics -- non-obviousness, soundness, mechanism
clarity/specificity, feasibility, and significance -- each 0-9. Every (idea, metric)
pair is one independent LLM call; no metric's prompt ever sees another metric's score,
so a halo effect on one axis (e.g. "this sounds important, so it must be sound too")
cannot leak into another. All N*5 calls run concurrently in a single flat pool.

Scores the ideator's full final response text for each idea, not the compressed
final_idea_core summary. Reads responses/<topic_id>/ep<episode_id>/ideator.jsonl, where
episode_id = idea_idx + 1 (confirmed against final_idea_cores.jsonl's own
final_idea_core.episode_id field -- episode folder "ep1" holds items[0], "ep2" holds
items[1], etc.), and takes the LATEST-`timestamp` "open"/"respond" record's text -- the
ideator's final, fully-refined response after all critic rounds in that episode -- as
that idea's full content. Each round's ideator.jsonl also holds a "core" event record
(that round's extracted final_idea_core) interleaved with the text records; those are
excluded here since they don't carry idea text. Some files are the concatenation of an
earlier run's records plus a later (e.g. resumed) run's records appended after, where
`sequence` numbering does not necessarily stay monotonic across the join, so `timestamp`
(not `sequence`) is the only reliable way to find the true most-recent record.

Diversity (whether generated ideas differ from one another) is a separate, pairwise
concern handled by diversity_eval.py -- not mixed in here.
"""
from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ideagent.console import print_warning
from ideagent.utils import read_jsonl


_METRICS = (
    "nonobviousness", "soundness", "mechanism_clarity_specificity", "feasibility",
    "significance",
)

_METRIC_LABELS: dict[str, str] = {
    "nonobviousness": "non-obviousness",
    "soundness": "soundness",
    "mechanism_clarity_specificity": "mechanism clarity/specificity",
    "feasibility": "feasibility",
    "significance": "significance",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityScore:
    idea_idx: int
    metric: str
    score: int          # 0-9
    rationale: str = ""


@dataclass
class QualityReport:
    topic_id: str
    n_ideas: int
    scores: list[QualityScore] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# JSON schema for structured output (shared across all five metrics)
# ─────────────────────────────────────────────────────────────────────────────

_QUALITY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "quality_score",
        "schema": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "score":     {"type": "integer", "minimum": 0, "maximum": 9},
            },
            "required": ["rationale", "score"],
            "additionalProperties": False,
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

NONOBVIOUSNESS_EVAL_SYSTEM_PROMPT = """\
You are evaluating the non-obviousness of a single research idea.

Your goal is to estimate whether the core research move would surprise a knowledgeable
reviewer — not whether the writing sounds sophisticated.

Score on a 0–9 scale:

  0 — Obvious / trivial
      Direct application of a standard fix; the first thing a reviewer would propose.

  1 — Near-trivial variant
      A known fix with a cosmetic tweak or a different hyperparameter; no new reasoning.

  2 — Straightforward extension
      Slightly modifies an existing method or combines familiar components in the expected way.

  3 — Predictable combination
      Joins two known ideas whose pairing is natural; a reviewer agrees without surprise.

  4 — Mildly non-obvious
      Has an interesting proxy, constraint, or framing, but the core move remains fairly standard.

  5 — Moderately non-obvious
      Identifies a real failure mode and picks a non-default mechanism, but the pairing is still
      reachable by careful incremental reasoning.

  6 — Solidly non-obvious
      Meaningful failure mode plus a mechanism that is not the obvious solution; a reviewer would
      pause before agreeing it is the right move.

  7 — Notably non-obvious
      A surprising problem-mechanism pairing, signal source, or intervention point that is
      technically meaningful rather than merely novel-sounding.

  8 — Highly non-obvious
      A novel causal diagnosis or mechanism acting at a point reviewers rarely consider; hard to
      reach without a genuine insight.

  9 — Exceptional / field-reframing
      Changes how the problem itself is conceptualized. Simple to explain once stated, but very
      hard to reach through incremental reasoning.

Base the score on the core move: problem-mechanism pairing, causal diagnosis, and intervention.
Do not reward complexity, length, or jargon. Do not score soundness, feasibility, significance,
or diversity relative to the other generated ideas. OUTPUT strict JSON with rationale and integer
score only.\
"""

SOUNDNESS_EVAL_SYSTEM_PROMPT = """\
You are evaluating the technical/logical soundness of a single research idea.

Your goal is to judge whether the proposed mechanism would actually work as claimed — whether \
the causal chain from mechanism to claimed effect is logically and technically valid — not \
whether the idea is novel, easy to build, or important.

Score on a 0–9 scale:

  0 — Fatally flawed
      The mechanism contradicts itself or relies on something unavailable in its own setting.
      Example: a training-time fix that assumes access to labels only available at test time.

  1 — Mostly unsound
      The core causal claim doesn't follow from its own premises without unstated extra leaps.

  2 — Significant gap
      The mechanism is plausible in isolation but rests on an assumption very likely false or untested.

  3 — Shaky
      Reasoning holds in the common case but ignores an obvious failure mode of its own mechanism.

  4 — Mostly sound with a caveat
      The core logic holds, but at least one non-trivial edge case or confound is unaddressed.

  5 — Sound with minor gaps
      The mechanism is logically coherent; a few secondary assumptions are untested but plausible.

  6 — Solid
      The causal chain from mechanism to claimed effect is clear, internally consistent, and \
its assumptions are reasonable and explicitly stated.

  7 — Rigorous
      The reasoning anticipates and addresses its own likely failure modes or confounds.

  8 — Highly rigorous
      Plausible alternative explanations for the claimed effect are explicitly considered and \
ruled out by the mechanism's own design.

  9 — Airtight
      The mechanism follows validly from well-justified premises with no exploitable logical or \
technical gap; a skeptical reviewer would find no wedge to attack the core reasoning.

SCORING RULES:
  - Judge only the internal logical/technical validity of the mechanism.
  - Do not reward novelty, penalise conventionality, or factor in feasibility or importance.
  - Do not reward dense or jargon-heavy phrasing, and do not penalise plain language — judge \
the logic, not how sophisticated the writing sounds.
  - A very safe, unoriginal idea can still score low here if its own logic is broken; a highly \
novel idea can score high here if its reasoning is airtight.

OUTPUT: strict JSON with "rationale" (one or two sentences explaining what makes the mechanism \
sound or unsound) and "score" (integer 0–9). Nothing else.\
"""

MECHANISM_CLARITY_SPECIFICITY_EVAL_SYSTEM_PROMPT = """\
You are evaluating the mechanism clarity/specificity of a single research idea.

Your goal is to judge how concretely and unambiguously the mechanism is specified — whether \
someone else could implement it from the description without inventing missing pieces — not \
whether the mechanism is correct or good.

Score on a 0–9 scale:

  0 — Empty / hand-wavy
      No actual mechanism given, just a goal restated as a method.
      Example: "we will make the model more robust" with no stated method.

  1 — Vague direction
      Names a technique family without saying what changes or how.

  2 — Underspecified
      Identifies a component to modify but not what signal, rule, or update it uses.

  3 — Sketch-level
      The mechanism's general shape is stated but key steps are asserted rather than defined.

  4 — Partially specified
      Most of the mechanism is concrete; one central step remains hand-wavy or unresolved.

  5 — Adequately specified
      The mechanism's main steps are all named and connected, though exact operationalization \
(thresholds, formulas) is left open.

  6 — Well specified
      Each step is concrete enough that an implementer would only need to choose minor free parameters.

  7 — Precisely specified
      Inputs, the transformation applied, and outputs are all stated; no structural ambiguity remains.

  8 — Fully operational
      Could be implemented directly from the description; only routine engineering choices are left.

  9 — Unambiguous and complete
      Every step, signal, and decision point is explicit enough that two independent implementers \
would build functionally the same thing.

SCORING RULES:
  - Judge only the concreteness/precision of the description, not whether the mechanism is \
correct (soundness) or practical to build (feasibility).
  - Reward precision, not length: a short precise description outscores a long vague one.
  - Coined terminology or technical-sounding jargon is not specificity — if the mechanism's \
actual steps are still vague once the jargon is stripped away, score it low.

OUTPUT: strict JSON with "rationale" (one or two sentences explaining what is or isn't \
concretely specified) and "score" (integer 0–9). Nothing else.\
"""

FEASIBILITY_EVAL_SYSTEM_PROMPT = """\
You are evaluating the feasibility of a single research idea.

Your goal is to judge how practical it would be to actually build and test this idea given \
realistic constraints (data, compute, existing tooling, time) — not whether it is correct or \
important.

Score on a 0–9 scale:

  0 — Practically impossible
      Requires resources, data, or capabilities that don't exist and aren't obtainable.
      Example: ground-truth access to a model's internal "intentions."

  1 — Extremely demanding
      Would require large, dedicated infrastructure or data collection beyond a typical project's scope.

  2 — Very demanding
      Feasible only with substantial new tooling, data collection, or compute beyond standard \
academic resources.

  3 — Demanding
      Needs meaningful new infrastructure (e.g. a new training pipeline or annotated dataset) \
not implied by the idea itself.

  4 — Moderately demanding
      Buildable with standard resources but requires nontrivial engineering effort or a \
moderately large compute budget.

  5 — Feasible with effort
      Implementable with existing tools/data with a reasonable, if nontrivial, engineering effort.

  6 — Fairly feasible
      Mostly reuses existing infrastructure/data; one or two components need custom work.

  7 — Feasible
      Could be built and tested by a small team using standard, available tools and data with \
modest effort.

  8 — Highly feasible
      Implementable quickly by adapting existing pipelines with minor modification.

  9 — Immediately actionable
      Testable essentially off-the-shelf, with data/tools/compute already at hand.

SCORING RULES:
  - Judge only the practicality of building/testing the idea as described.
  - Do not factor in whether the idea is correct (soundness) or important (significance) — \
an idea can be highly feasible yet unsound, or highly sound yet infeasible.
  - Do not let dense or technical-sounding phrasing make an idea seem more (or less) feasible \
than it actually is — judge the real resource/engineering requirements, not the writing style.

OUTPUT: strict JSON with "rationale" (one or two sentences explaining what makes this feasible \
or infeasible) and "score" (integer 0–9). Nothing else.\
"""

SIGNIFICANCE_EVAL_SYSTEM_PROMPT = """\
You are evaluating the significance of a single research idea.

Your goal is to judge how much it would matter if the idea worked exactly as claimed — to the \
specific problem, and to the broader field — not whether it is currently feasible or already proven.

Score on a 0–9 scale:

  0 — Negligible
      Even complete success would not meaningfully change anything a practitioner or researcher \
cares about.

  1 — Marginal
      A small, local improvement noticeable only in narrow or synthetic settings.

  2 — Minor
      A modest improvement on a narrow problem with little downstream consequence.

  3 — Limited
      Addresses a real but narrow problem; success would matter only to a small niche.

  4 — Moderate
      Meaningfully improves a recognized problem, though the problem itself is not central to the field.

  5 — Notable
      Success would meaningfully move a problem that a nontrivial part of the field actively cares about.

  6 — Significant
      Success would change how a meaningful subset of practitioners approach a known, important problem.

  7 — Important
      Success would resolve or substantially advance a problem widely recognized as a major open issue.

  8 — Major
      Success would shift how the field thinks about the underlying failure mode itself, beyond \
just improving numbers.

  9 — Field-defining
      Success would open or redefine a research direction, changing what subsequent work in the \
area targets.

SCORING RULES:
  - Judge hypothetical impact assuming the mechanism works as claimed; do not penalise for \
feasibility or current-stage uncertainty (those are separate axes).
  - Do not reward significance-by-association: citing a "big problem" without the mechanism \
actually addressing its core difficulty does not raise this score.
  - Do not let grandiose or elaborate phrasing inflate the perceived impact — judge the actual \
hypothetical consequence of success, not how the claim is worded.

OUTPUT: strict JSON with "rationale" (one or two sentences explaining the scope of impact if \
successful) and "score" (integer 0–9). Nothing else.\
"""

_METRIC_SYSTEM_PROMPTS: dict[str, str] = {
    "nonobviousness": NONOBVIOUSNESS_EVAL_SYSTEM_PROMPT,
    "soundness": SOUNDNESS_EVAL_SYSTEM_PROMPT,
    "mechanism_clarity_specificity": MECHANISM_CLARITY_SPECIFICITY_EVAL_SYSTEM_PROMPT,
    "feasibility": FEASIBILITY_EVAL_SYSTEM_PROMPT,
    "significance": SIGNIFICANCE_EVAL_SYSTEM_PROMPT,
}


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    candidate = match.group(0) if match else stripped
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_quality_score(raw: str, *, idx: int, metric: str) -> QualityScore | None:
    obj = _extract_json_object(raw)
    if obj is None:
        return None
    try:
        score = int(float(obj.get("score", -1)))
    except (TypeError, ValueError):
        return None
    if score < 0 or score > 9:
        return None
    return QualityScore(
        idea_idx=idx,
        metric=metric,
        score=score,
        rationale=str(obj.get("rationale", "")).strip(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Idea-text discovery (full ideator response, not the compressed final_idea_core)
# ─────────────────────────────────────────────────────────────────────────────

def discover_topics(responses_root: Path) -> list[tuple[str, int]]:
    """List (topic_id, n_ideas) pairs directly from the responses/ directory tree --
    topic_id = each subdirectory name; n_ideas = the highest episode-folder number
    found under it (episode folders are 1-based: "ep1".."ep<n_ideas>")."""
    topics = []
    for topic_dir in sorted(responses_root.iterdir()):
        if not topic_dir.is_dir():
            continue
        episode_ids = [
            int(d.name[2:]) for d in topic_dir.iterdir()
            if d.is_dir() and d.name.startswith("ep") and d.name[2:].isdigit()
        ]
        if not episode_ids:
            continue
        topics.append((topic_dir.name, max(episode_ids)))
    return topics


def _record_timestamp(record: dict[str, Any]) -> datetime:
    try:
        return datetime.fromisoformat(str(record.get("timestamp", "")))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def load_final_ideator_text(responses_root: Path, topic_id: str, idea_idx: int) -> str | None:
    """responses/<topic_id>/ep<idea_idx + 1>/ideator.jsonl -> latest-timestamp text.

    Only "open"/"respond" records carry idea text; "core" records (that round's
    extracted final_idea_core, logged alongside) are excluded from the search."""
    path = responses_root / topic_id / f"ep{idea_idx + 1}" / "ideator.jsonl"
    if not path.exists():
        return None
    records = [r for r in read_jsonl(path) if r.get("event") in ("open", "respond")]
    if not records:
        return None
    last = max(records, key=_record_timestamp)
    text = str(last.get("text", "")).strip()
    return text or None


def extract_ideas_from_responses(
    responses_root: Path, topic_id: str, n_ideas: int
) -> list[dict[str, Any]]:
    ideas = []
    for idx in range(n_ideas):
        text = load_final_ideator_text(responses_root, topic_id, idx)
        if text is None:
            print_warning(
                f"[quality_eval] {topic_id}: no final ideator response for idea "
                f"{idx} (episode {idx + 1}); skipping idea."
            )
            continue
        ideas.append({"idx": idx, "text": text})
    return ideas


def extract_ideas_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the common untouched full-text field used by all new generation methods."""

    ideas: list[dict[str, Any]] = []
    for idx, item in enumerate(record.get("items", [])):
        if not isinstance(item, dict):
            continue
        text = str(item.get("idea_text", "")).strip()
        if text:
            ideas.append({"idx": idx, "text": text})
    return ideas


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_idea_user_message(idea: dict[str, Any], metric: str) -> str:
    return (
        f"<IDEA>\n{idea['text']}\n</IDEA>\n\n"
        f"Score this research idea's {_METRIC_LABELS[metric]}. "
        "Return strict JSON and nothing else."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class QualityEvaluator:
    def __init__(
        self,
        *,
        client: Any,
        gen_kwargs: dict[str, Any],
        max_parsing_retries: int = 2,
        max_workers: int = 10,
        escalation_client: Any | None = None,
        escalation_after_attempts: int = 5,
    ) -> None:
        self.client = client
        self.gen_kwargs = gen_kwargs
        self.max_parsing_retries = max(0, int(max_parsing_retries))
        self.max_workers = max(1, int(max_workers))
        # After this many failed attempts on the primary model, switch to
        # escalation_client for the remaining retries. None disables escalation.
        self.escalation_client = escalation_client
        self.escalation_after_attempts = max(1, int(escalation_after_attempts))

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        parser: Any,
    ) -> Any:
        gen_kwargs = {**self.gen_kwargs}
        if response_format is not None:
            gen_kwargs["response_format"] = response_format
        for attempt in range(1 + self.max_parsing_retries):
            use_escalation = (
                self.escalation_client is not None
                and attempt >= self.escalation_after_attempts
            )
            client = self.escalation_client if use_escalation else self.client
            try:
                raw = client.generate(messages, **gen_kwargs)
            except Exception:
                # A raised exception (e.g. a thinking-only reply with no text content)
                # is a failed attempt like any other -- keep retrying/escalating rather
                # than letting it propagate and abort the whole eval run.
                continue
            result = parser(raw)
            if result is not None:
                return result
        return None

    def _run_pool(self, tasks: list[Any], workers: int, desc: str):
        """Run (fn, *args) tasks concurrently, yielding results as they complete.
        Each (idea, metric) call is independent, so a single failure does not abort
        the rest -- unlike diversity_eval's axis/pair calls, which are chained."""
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [executor.submit(fn, *args) for fn, *args in tasks]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=desc, leave=False
            ):
                yield future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _build_idea_user_message(self, idea: dict[str, Any], metric: str) -> str:
        return _build_idea_user_message(idea, metric)

    def evaluate_metric(self, idea: dict[str, Any], metric: str) -> QualityScore | None:
        """Score a single metric for one idea (one of the N*5 independent calls)."""
        idx = idea["idx"]
        messages = [
            {"role": "system", "content": _METRIC_SYSTEM_PROMPTS[metric]},
            {"role": "user", "content": self._build_idea_user_message(idea, metric)},
        ]
        return self._call_with_retry(
            messages,
            _QUALITY_RESPONSE_FORMAT,
            lambda raw: _parse_quality_score(raw, idx=idx, metric=metric),
        )

    def evaluate(
        self,
        *,
        topic_id: str,
        ideas: list[dict[str, Any]],
    ) -> QualityReport | None:
        n = len(ideas)
        if n < 1:
            print_warning(f"[quality_eval] topic {topic_id}: no ideas; skipping.")
            return None

        def _eval(idx: int, metric: str) -> QualityScore | None:
            result = self.evaluate_metric(ideas[idx], metric)
            if result is None:
                print_warning(
                    f"[quality_eval] topic {topic_id}: idea {idx} metric '{metric}' "
                    "parse failed; skipping."
                )
            return result

        tasks = [(_eval, idx, metric) for idx in range(n) for metric in _METRICS]
        workers = min(self.max_workers, len(tasks))
        desc = f"{topic_id} ({n} ideas x {len(_METRICS)} metrics)"
        scores = [s for s in self._run_pool(tasks, workers, desc) if s is not None]

        if not scores:
            return None
        scores.sort(key=lambda s: (s.idea_idx, _METRICS.index(s.metric)))
        return QualityReport(topic_id=topic_id, n_ideas=n, scores=scores)


class GeminiQualityEvaluator(QualityEvaluator):
    """Async variant for Gemini models -- uses asyncio.gather instead of ThreadPoolExecutor.
    All prompts, parsers, and retry logic are inherited from QualityEvaluator.
    max_workers controls the asyncio.Semaphore cap on concurrent in-flight requests."""

    async def _acall_with_retry(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        parser: Any,
    ) -> Any:
        gen_kwargs = {**self.gen_kwargs}
        if response_format is not None:
            gen_kwargs["response_format"] = response_format
        for _ in range(1 + self.max_parsing_retries):
            raw = await self.client.generate_async(messages, **gen_kwargs)
            result = parser(raw)
            if result is not None:
                return result
        return None

    async def _evaluate_async(
        self,
        *,
        topic_id: str,
        ideas: list[dict[str, Any]],
    ) -> QualityReport | None:
        n = len(ideas)
        if n < 1:
            print_warning(f"[quality_eval] topic {topic_id}: no ideas; skipping.")
            return None

        sem = asyncio.Semaphore(self.max_workers)

        async def _throttled(coro: Any) -> Any:
            async with sem:
                return await coro

        async def _aeval(idx: int, metric: str) -> QualityScore | None:
            messages = [
                {"role": "system", "content": _METRIC_SYSTEM_PROMPTS[metric]},
                {"role": "user", "content": self._build_idea_user_message(ideas[idx], metric)},
            ]
            result = await self._acall_with_retry(
                messages,
                _QUALITY_RESPONSE_FORMAT,
                lambda raw: _parse_quality_score(raw, idx=idx, metric=metric),
            )
            if result is None:
                print_warning(
                    f"[quality_eval] topic {topic_id}: idea {idx} metric '{metric}' "
                    "parse failed; skipping."
                )
            return result

        coros = [_aeval(idx, metric) for idx in range(n) for metric in _METRICS]
        results = await asyncio.gather(*[_throttled(c) for c in coros])
        scores = [s for s in results if s is not None]

        if not scores:
            return None
        scores.sort(key=lambda s: (s.idea_idx, _METRICS.index(s.metric)))
        return QualityReport(topic_id=topic_id, n_ideas=n, scores=scores)

    def evaluate(
        self,
        *,
        topic_id: str,
        ideas: list[dict[str, Any]],
    ) -> QualityReport | None:
        return asyncio.run(self._evaluate_async(topic_id=topic_id, ideas=ideas))
