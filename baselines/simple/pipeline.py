"""Stateless-independent and single-shot batch baselines, prompt-matched to sequential memory.

Two ladder rungs below the sequential-memory baseline, sharing its exact prompt lineage
(the verbatim ``IDEATOR_SYSTEM_PROMPT`` + ``op_free_baseline`` + the shared opening-message
builder):

``stateless``
    ``n_ideas`` fully independent generator calls per topic. Every call is byte-identical to
    the sequential-memory baseline's FIRST call: empty signature memory, so the free operator
    contains no avoid block and diversity pressure is background-relative only ("must not be a
    direct combination, extension, or ablation of the background papers"). No cross-idea
    signal of any kind — this isolates what sampling stochasticity alone buys.

``single_shot``
    ONE generator call per topic returns all ``n_ideas`` ideas, delimited by ``=== IDEA k ===``
    headers. The directive is the free operator's head pluralized, with the memory-avoid block
    replaced by a within-response mutual-distinctness clause phrased to mirror the sequential
    wording ("each idea must attack a DIFFERENT problem or use a DIFFERENT core mechanism than
    every other idea in this response"). The non-obviousness and soundness pressure blocks are
    imported verbatim. This isolates what batching with self-visible siblings buys.

Neither mode has a critic, evaluator, selection, repair, or refinement call. The steno runs
after generation exactly as in the other baselines, for output compatibility only — its
signatures are never fed back to the generator.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ideagent.agent_prompts import (
    IDEATOR_SYSTEM_PROMPT,
    build_ideator_opening_user_message,
)
from ideagent.data import (
    build_background_context,
    load_raw_topic_groups,
    parse_paper,
    truncate_by_tokens_rough,
)
from ideagent.experiment_logging import (
    ExperimentLogger,
    set_trace_context,
    text_sha256,
    write_run_manifest,
    write_trace_summary,
)
from ideagent.response_logging import AsyncResponseLogger
from ideagent.sequential_memory_baseline import _fallback_core
from ideagent.signature import extract_signature, signature_to_final_idea_core
from ideagent.single_shot_baseline import single_shot_response_format
from ideagent.utils import append_jsonl, create_jsonl, prepare_output_dir, read_jsonl
from ideagent.prompts import (
    NON_OBVIOUSNESS_PRESSURE,
    SOUND_BY_CONSTRUCTION,
    op_free_baseline,
)


MODES = ("stateless", "single_shot")


def stateless_directive() -> str:
    """Byte-identical to the sequential-memory baseline's first (empty-memory) directive."""

    return op_free_baseline([])


def single_shot_directive(n_ideas: int) -> str:
    """The free operator adapted to one batch response.

    Same composition order as ``op_free_baseline`` (head, distinctness, non-obviousness
    pressure, soundness pressure, restatement), with the memory-avoid block replaced by
    within-response mutual distinctness — the only diversity signal a single call can act on
    besides the background itself. Output is a structured JSON array enforced by
    ``single_shot_response_format`` (exactly n_ideas items) rather than a delimited-text
    format the model has to freely comply with, since the count itself is now a hard schema
    constraint; the pacing block still matters for keeping each item proportionate.
    """

    head = (
        f"Propose {n_ideas} NOVEL, non-obvious, and SOUND research ideas for this problem "
        "space. None may be a direct combination, extension, or ablation of the background "
        "papers. For each idea, you are free to choose any problem and any mechanism (it may "
        "combine several parts of the pipeline) -- just make each genuinely distinct from "
        "what already exists."
    )
    distinctness = (
        "MUTUAL DISTINCTNESS -- the ideas in this response are the whole search: each idea "
        "must attack a DIFFERENT problem or use a DIFFERENT core mechanism than every other "
        "idea in this response (no duplicates)."
    )
    restate = (
        "Respond with COMPLETE ideas -- full motivation, mechanism, and justification for "
        "each -- not deltas."
    )
    pacing = (
        f"PACE YOURSELF -- you must fully complete all {n_ideas} ideas in this single "
        f"response. Budget your writing so each of the {n_ideas} ideas receives a roughly "
        "equal, proportionate share of the response: complete but not exhaustive. Do not "
        f"spend the whole response on the first idea and stop early. Producing fewer than "
        f"{n_ideas} ideas is a failure, even if the ones you did produce are excellent."
    )
    output_format = (
        f"FORMAT: return exactly {n_ideas} items in the 'ideas' array, each with a single "
        "'idea_text' field containing that idea's complete prose. No scores, rankings, or "
        "commentary anywhere in the response."
    )
    return "\n\n".join(
        [head, distinctness, NON_OBVIOUSNESS_PRESSURE, SOUND_BY_CONSTRUCTION, restate,
         pacing, output_format]
    )


def split_single_shot_response(raw: str, *, n_ideas: int, min_idea_chars: int) -> list[str]:
    """Parse a structured-output batch response into exactly ``n_ideas`` ideas.

    The count is already enforced by single_shot_response_format's minItems/maxItems schema
    constraint, but this still fails closed on malformed JSON or a too-short idea, same as
    before.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"single-shot response is not valid JSON: {exc}") from exc
    ideas_field = data.get("ideas") if isinstance(data, dict) else None
    if not isinstance(ideas_field, list) or len(ideas_field) != n_ideas:
        found = len(ideas_field) if isinstance(ideas_field, list) else "no"
        raise ValueError(
            f"single-shot response has {found} ideas; expected {n_ideas}"
        )
    ideas = [str(item.get("idea_text", "")).strip() for item in ideas_field]
    for index, idea_text in enumerate(ideas, start=1):
        if len(idea_text) < min_idea_chars:
            raise ValueError(
                f"single-shot idea {index} is too short ({len(idea_text)} characters; "
                f"minimum {min_idea_chars})"
            )
    return ideas


@dataclass(frozen=True)
class SimpleBaselineConfig:
    mode: str = "stateless"
    n_ideas: int = 10
    resume: bool = True
    init_topic_idx: int = 0
    end_topic_idx: int | None = 1
    max_background_tokens: int | None = None
    min_idea_chars: int = 300
    signature_char_cap: int = 256

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"baseline.mode must be one of {MODES}")
        if self.n_ideas <= 0:
            raise ValueError("baseline.n_ideas must be a positive integer")
        if self.init_topic_idx < 0:
            raise ValueError("baseline.init_topic_idx must be >= 0")
        if self.end_topic_idx is not None and self.end_topic_idx <= self.init_topic_idx:
            raise ValueError("baseline.end_topic_idx must be > init_topic_idx")
        if self.max_background_tokens is not None and self.max_background_tokens <= 0:
            raise ValueError("baseline.max_background_tokens must be null or positive")
        if self.min_idea_chars < 1:
            raise ValueError("baseline.min_idea_chars must be positive")
        if self.signature_char_cap < 16:
            raise ValueError("baseline.signature_char_cap must be >= 16")


def _generation_mode(config: SimpleBaselineConfig) -> str:
    return "stateless_independent" if config.mode == "stateless" else "single_shot_batch"


def run_simple_baseline(
    *,
    input_jsonl: str | Path,
    output_dir: str | Path,
    generator: Any,
    gen_kwargs: dict[str, Any],
    config: SimpleBaselineConfig,
    steno_client: Any,
    steno_gen_kwargs: dict[str, Any] | None = None,
    response_logger: AsyncResponseLogger | None = None,
    experiment_logger: ExperimentLogger | None = None,
) -> list[dict[str, Any]]:
    """Generate exactly ``n_ideas`` ideas per topic in the configured mode."""

    config.validate()
    if steno_client is None:
        raise ValueError("A steno_client is required for output-compatible signatures")

    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, resume=config.resume, init_topic_idx=config.init_topic_idx)
    system_prompt = IDEATOR_SYSTEM_PROMPT
    generation_mode = _generation_mode(config)
    generator_calls = config.n_ideas if config.mode == "stateless" else 1
    if experiment_logger is not None:
        if config.mode == "stateless":
            diversity_note = (
                "background-relative only: every call is byte-identical to the "
                "sequential-memory baseline's first (empty-memory) call, so no cross-idea "
                "signal exists by design"
            )
        else:
            diversity_note = (
                "within-response mutual distinctness replaces the sequential memory-avoid "
                "block; non-obviousness and soundness pressure blocks are byte-identical"
            )
        fairness_contract = {
            "comparison_target": "yield_qd",
            "matched": {
                "candidate_output_budget": config.n_ideas,
                "generator_calls": generator_calls,
                "one_idea_per_generator_call": config.mode == "stateless",
                "background_builder": "build_background_context",
                "ideator_system_prompt_exact": True,
                "ideator_system_prompt_sha256": text_sha256(system_prompt),
                "diversity_instruction": diversity_note,
                "posthoc_evaluation_field": "items[].idea_text",
            },
            "intentional_treatment_difference": {
                "critic_calls": 0,
                "quality_or_novelty_feedback_during_generation": False,
                "repair_or_refinement": False,
                "selection_calls": 0,
                "memory": "none (steno signatures are extracted for output only)",
            },
            "not_claimed": [
                "equal total model calls because QD invokes evaluators and critics",
                "equal realized token count or monetary cost",
            ],
        }
        manifest = {
            "mode": generation_mode,
            "config": asdict(config),
            "models": {
                "generator": getattr(generator, "model_id", ""),
                "steno": getattr(steno_client, "model_id", ""),
            },
            "sampling": {"generator": gen_kwargs, "steno": steno_gen_kwargs or {}},
            "prompts": {
                "ideator_system_prompt": system_prompt,
                "ideator_system_prompt_sha256": text_sha256(system_prompt),
                "directive": (
                    stateless_directive()
                    if config.mode == "stateless"
                    else single_shot_directive(config.n_ideas)
                ),
            },
            "fairness_contract": fairness_contract,
            "storage_policy": {
                "model_context": "background only (no memory)",
                "disk": "full prompts, full outputs, full idea text, and signatures",
            },
        }
        manifest_path = write_run_manifest(output_dir, logger=experiment_logger, manifest=manifest)
        experiment_logger.log(
            "run_started", mode=generation_mode, config=asdict(config),
            models={
                "generator": getattr(generator, "model_id", ""),
                "steno": getattr(steno_client, "model_id", ""),
            },
            manifest_path=str(manifest_path), fairness_contract=fairness_contract,
        )
    tag = f"_from{config.init_topic_idx}" if config.init_topic_idx else ""
    output_path = output_dir / f"final_idea_cores{tag}.jsonl"

    records: list[dict[str, Any]] = []
    completed_topic_ids: set[str] = set()
    if config.resume and output_path.exists():
        for record in read_jsonl(output_path):
            topic_id = str(record.get("topic_id", ""))
            if topic_id:
                completed_topic_ids.add(topic_id)
                records.append(record)
        print(f"Resume: {len(completed_topic_ids)} topic(s) already done, skipping.", flush=True)
    else:
        create_jsonl(output_path)

    groups = load_raw_topic_groups(input_jsonl)[config.init_topic_idx : config.end_topic_idx]
    signature_kwargs = steno_gen_kwargs or {}

    for group in groups:
        topic_id = str(group["topic_id"])
        if topic_id in completed_topic_ids:
            print(f"Topic {topic_id}: already completed, skipping.", flush=True)
            continue

        papers = [parse_paper(paper) for paper in group["background_papers"]]
        background = build_background_context(papers)
        if config.max_background_tokens:
            background = truncate_by_tokens_rough(background, config.max_background_tokens)

        print(
            f"\n=== {generation_mode} baseline — topic {topic_id}: "
            f"{generator_calls} generator call(s) for {config.n_ideas} ideas ===",
            flush=True,
        )
        if experiment_logger is not None:
            experiment_logger.log(
                "topic_started", topic_id=topic_id, mode=generation_mode,
                background=background,
            )
        cache_key = hashlib.sha256(f"{topic_id}/ideator".encode()).hexdigest()

        if config.mode == "stateless":
            directive = stateless_directive()
            # Every one of the n_ideas calls sends byte-identical messages (no per-idea
            # variation, no memory) -- OpenAI's native `n` batching would be the natural fit,
            # but it caps at 8 and n_ideas=10 exceeds that. Fan out n real, independent
            # .generate() calls via a thread pool instead: same messages/kwargs as the
            # original sequential loop, just issued concurrently rather than one at a time.
            # One shared trace context for the whole fan-out, set once before submitting --
            # matching GeminiClient/AnthropicClient's own generate_many fan-out pattern
            # (clients.py). Setting distinct per-idea context from inside each worker thread
            # would race against TracedClient's single shared, mutable context dict and could
            # mislabel which idea a trace entry actually belongs to.
            set_trace_context(
                generator, topic_id=topic_id, candidate_id="batch",
                generation_index=1, stage="candidate_generation",
                memory_policy="none_stateless",
            )
            call_kwargs = dict(gen_kwargs)
            call_kwargs["prompt_cache_key"] = cache_key
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": build_ideator_opening_user_message(
                        background, prior_cores_block=None,
                        critic_opening_challenge=directive,
                    ),
                },
            ]

            def _one(_idea_idx: int) -> str:
                return str(generator.generate(messages, **call_kwargs)).strip()

            with ThreadPoolExecutor(max_workers=config.n_ideas) as executor:
                idea_texts = list(
                    executor.map(_one, range(1, config.n_ideas + 1))
                )
            for idea_idx, idea_text in enumerate(idea_texts, start=1):
                if len(idea_text) < config.min_idea_chars:
                    raise ValueError(
                        f"stateless idea {idea_idx} is too short ({len(idea_text)} "
                        f"characters; minimum {config.min_idea_chars})"
                    )
                print(f"  idea {idea_idx}/{config.n_ideas} generated", flush=True)
            raw_response = None
        else:
            directive = single_shot_directive(config.n_ideas)
            set_trace_context(
                generator, topic_id=topic_id, candidate_id="batch",
                generation_index=1, stage="batch_generation",
                memory_policy="none_single_shot",
            )
            call_kwargs = dict(gen_kwargs)
            call_kwargs["prompt_cache_key"] = cache_key
            call_kwargs["response_format"] = single_shot_response_format(config.n_ideas)
            raw_response = str(
                generator.generate(
                    [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": build_ideator_opening_user_message(
                                background, prior_cores_block=None,
                                critic_opening_challenge=directive,
                            ),
                        },
                    ],
                    **call_kwargs,
                )
            )
            idea_texts = split_single_shot_response(
                raw_response, n_ideas=config.n_ideas, min_idea_chars=config.min_idea_chars
            )
            print(f"  batch response split into {len(idea_texts)} ideas", flush=True)

        items: list[dict[str, Any]] = []
        generation_rows: list[dict[str, Any]] = []
        generations_path = output_dir / f"generations_{topic_id}.jsonl"
        with generations_path.open("w", encoding="utf-8") as generation_log:
            for idea_idx, idea_text in enumerate(idea_texts, start=1):
                idea_id = f"i{idea_idx:03d}"
                if response_logger is not None:
                    response_logger.log(
                        role="ideator", event="open", text=idea_text,
                        metadata={
                            "topic_id": topic_id,
                            "target_paper_id": str(idea_idx),
                            "generation_mode": generation_mode,
                        },
                    )
                set_trace_context(
                    steno_client, topic_id=topic_id, candidate_id=idea_id,
                    generation_index=idea_idx, stage="signature_extraction",
                )
                signature = extract_signature(
                    idea_text,
                    idea_id=idea_id,
                    client=steno_client,
                    gen_kwargs=signature_kwargs,
                    char_cap=config.signature_char_cap,
                )
                if signature is not None:
                    core = signature_to_final_idea_core(
                        signature, idea_text=idea_text, episode_id=str(idea_idx)
                    )
                else:
                    core = _fallback_core(idea_id=str(idea_idx), idea_text=idea_text)
                if response_logger is not None:
                    # Same "core" companion record the sequential baseline writes, so the
                    # responses/ layout stays byte-compatible for downstream tools.
                    response_logger.log(
                        role="ideator", event="core", text=str(core.get("core_claim", "")),
                        metadata={
                            "topic_id": topic_id,
                            "target_paper_id": str(idea_idx),
                            "generation_mode": generation_mode,
                            "core": core,
                        },
                    )
                items.append(
                    {
                        "idea_id": idea_id,
                        "idea_text": idea_text,
                        "signature_ok": signature is not None,
                        "final_idea_core": core,
                    }
                )
                row = {
                    "topic_id": topic_id,
                    "generation_index": idea_idx,
                    "idea_id": idea_id,
                    "generation_mode": generation_mode,
                    "generation_directive": directive,
                    "system_prompt": system_prompt,
                    "idea_text": idea_text,
                    "idea_char_count": len(idea_text),
                    "idea_word_count": len(idea_text.split()),
                    "signature_ok": signature is not None,
                    "signature": signature.to_json() if signature is not None else None,
                    "critic_calls": 0,
                    "quality_evaluator_calls": 0,
                }
                generation_rows.append(row)
                generation_log.write(json.dumps(row, ensure_ascii=False) + "\n")
                generation_log.flush()
                if experiment_logger is not None:
                    experiment_logger.log("baseline_generation_completed", **row)

        record = {
            "topic_id": topic_id,
            "generation_mode": generation_mode,
            "generator_calls": generator_calls,
            "critic_calls": 0,
            "quality_evaluator_calls": 0,
            "n_requested": config.n_ideas,
            "n_generated": len(items),
            "n_background_papers": len(papers),
            "memory_policy": "none",
            "items": items,
        }
        records.append(record)
        append_jsonl(output_path, record)
        analysis = {
            "topic_id": topic_id,
            "generation_mode": generation_mode,
            "n_generations": len(generation_rows),
            "generator_calls": generator_calls,
            "critic_calls": 0,
            "quality_evaluator_calls": 0,
            "signature_successes": sum(row["signature_ok"] for row in generation_rows),
            "idea_word_counts": [row["idea_word_count"] for row in generation_rows],
            "raw_response": raw_response,
            "ideas": generation_rows,
        }
        (output_dir / f"analysis_{topic_id}.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if experiment_logger is not None:
            experiment_logger.log(
                "topic_completed", topic_id=topic_id,
                mode=generation_mode, analysis_summary=analysis,
            )
        print(
            f"--- topic {topic_id}: generated {len(items)}/{config.n_ideas} ideas ---",
            flush=True,
        )

    if experiment_logger is not None:
        experiment_logger.log(
            "run_completed", mode=generation_mode,
            topic_records=len(records), records=records,
        )
        summary_path = write_trace_summary(output_dir, logger=experiment_logger)
        experiment_logger.log("trace_summary_written", path=str(summary_path))
    return records
