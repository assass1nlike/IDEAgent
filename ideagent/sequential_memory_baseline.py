"""Candidate-matched direct-generation baseline with compact semantic memory and no feedback.

This is the candidate-budget-matched control for Yield/QD.  Each call uses the exact QD ideator
system prompt and free-generation operator.  After a proposal, the same steno signature extractor
used by QD records a compact problem/mechanism fingerprint.  Later calls see those signatures so
they can avoid duplicates, but never see quality scores, gates, critic feedback, acceptance labels,
repairs, refinements, or prior full idea text.

It is intentionally implemented as reset-per-candidate sequential generation rather than a
growing chat conversation: this mirrors QD's compact-memory context and prevents token growth.
"""
from __future__ import annotations

import hashlib
import json
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
from ideagent.signature import (
    SemanticSignature,
    extract_signature,
    signature_to_final_idea_core,
)
from ideagent.utils import append_jsonl, create_jsonl, prepare_output_dir, read_jsonl
from ideagent.prompts import op_free_baseline


@dataclass(frozen=True)
class SequentialMemoryBaselineConfig:
    n_ideas: int = 10
    resume: bool = True
    init_topic_idx: int = 0
    end_topic_idx: int | None = 1
    max_background_tokens: int | None = None
    min_idea_chars: int = 300
    signature_char_cap: int = 256

    def validate(self) -> None:
        if self.n_ideas <= 0:
            raise ValueError("sequential_memory.n_ideas must be a positive integer")
        if self.init_topic_idx < 0:
            raise ValueError("sequential_memory.init_topic_idx must be >= 0")
        if self.end_topic_idx is not None and self.end_topic_idx <= self.init_topic_idx:
            raise ValueError("sequential_memory.end_topic_idx must be > init_topic_idx")
        if self.max_background_tokens is not None and self.max_background_tokens <= 0:
            raise ValueError("sequential_memory.max_background_tokens must be null or positive")
        if self.min_idea_chars < 1:
            raise ValueError("sequential_memory.min_idea_chars must be positive")
        if self.signature_char_cap < 16:
            raise ValueError("sequential_memory.signature_char_cap must be >= 16")


def _fallback_core(*, idea_id: str, idea_text: str) -> dict[str, Any]:
    return {
        "episode_id": idea_id,
        "core_claim": idea_text[:500],
        "idea_summary": idea_text,
        "mechanism": "",
        "problem_targeted": "",
        "source_gap": "",
        "expected_effect": "",
        "major_keywords": [],
        "generic_theme": "",
        "parent_theme": "",
    }


def _log_idea(
    response_logger: AsyncResponseLogger | None,
    *,
    topic_id: str,
    idea_idx: int,
    event: str,
    text: str,
    core: dict[str, Any] | None = None,
) -> None:
    if response_logger is None:
        return
    metadata: dict[str, Any] = {
        "topic_id": topic_id,
        "target_paper_id": str(idea_idx),
        "generation_mode": "sequential_compact_memory",
    }
    if core is not None:
        metadata["core"] = core
    response_logger.log(role="ideator", event=event, text=text, metadata=metadata)


def run_sequential_memory_baseline(
    *,
    input_jsonl: str | Path,
    output_dir: str | Path,
    generator: Any,
    gen_kwargs: dict[str, Any],
    config: SequentialMemoryBaselineConfig,
    steno_client: Any,
    steno_gen_kwargs: dict[str, Any] | None = None,
    response_logger: AsyncResponseLogger | None = None,
    experiment_logger: ExperimentLogger | None = None,
) -> list[dict[str, Any]]:
    """Generate exactly ``n_ideas`` candidates, one call each, with compact prior memory."""

    config.validate()
    if steno_client is None:
        raise ValueError("A steno_client is required for compact signature memory")

    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, resume=config.resume, init_topic_idx=config.init_topic_idx)
    if experiment_logger is not None:
        fairness_contract = {
            "comparison_target": "yield_qd",
            "matched": {
                "candidate_output_budget": config.n_ideas,
                "one_idea_per_generator_call": True,
                "generator_calls": config.n_ideas,
                "background_builder": "build_background_context",
                "ideator_system_prompt_exact": True,
                "ideator_system_prompt_sha256": text_sha256(IDEATOR_SYSTEM_PROMPT),
                "substantive_free_operator": "same head + sound-by-construction + restatement",
                "reset_context_each_candidate": True,
                "topic_wide_prompt_cache_key": True,
                "posthoc_evaluation_field": "items[].idea_text",
            },
            "intentional_treatment_difference": {
                "critic_calls": 0,
                "quality_or_novelty_feedback_during_generation": False,
                "repair_or_refinement": False,
                "memory": (
                    "all earlier compact signatures, neutrally labelled; no acceptance, rejection, "
                    "score, or critic signal"
                ),
            },
            "not_claimed": [
                "equal total model calls because QD invokes evaluators and critics",
                "equal realized token count or monetary cost",
            ],
        }
        manifest = {
            "mode": "sequential_compact_memory",
            "config": asdict(config),
            "models": {
                "generator": getattr(generator, "model_id", ""),
                "steno": getattr(steno_client, "model_id", ""),
            },
            "sampling": {"generator": gen_kwargs, "steno": steno_gen_kwargs or {}},
            "prompts": {
                "ideator_system_prompt": IDEATOR_SYSTEM_PROMPT,
                "ideator_system_prompt_sha256": text_sha256(IDEATOR_SYSTEM_PROMPT),
                "first_free_directive": op_free_baseline([]),
            },
            "fairness_contract": fairness_contract,
            "storage_policy": {
                "model_context": "compact signatures only",
                "disk": "full prompts, full outputs, full idea text, and signatures",
            },
        }
        manifest_path = write_run_manifest(
            output_dir, logger=experiment_logger, manifest=manifest
        )
        experiment_logger.log(
            "run_started", mode="sequential_compact_memory", config=asdict(config),
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
            f"\n=== Sequential compact-memory baseline — topic {topic_id}: "
            f"{config.n_ideas} generator calls ===",
            flush=True,
        )
        prior_signatures: list[SemanticSignature] = []
        items: list[dict[str, Any]] = []
        generation_rows: list[dict[str, Any]] = []
        generations_path = output_dir / f"generations_{topic_id}.jsonl"

        if experiment_logger is not None:
            experiment_logger.log(
                "topic_started", topic_id=topic_id, mode="sequential_compact_memory",
                background=background,
            )

        with generations_path.open("w", encoding="utf-8") as generation_log:
            for idea_idx in range(1, config.n_ideas + 1):
                idea_id = f"i{idea_idx:03d}"
                memory_before = [signature.to_json() for signature in prior_signatures]
                # Same substantive free operator as QD, with a neutral memory heading: without an
                # evaluator the baseline must not imply that earlier outputs were "accepted".
                directive = op_free_baseline(prior_signatures)
                messages = [
                    {"role": "system", "content": IDEATOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_ideator_opening_user_message(
                            background, prior_cores_block=None,
                            critic_opening_challenge=directive,
                        ),
                    },
                ]
                set_trace_context(
                    generator, topic_id=topic_id, candidate_id=idea_id,
                    generation_index=idea_idx, stage="candidate_generation",
                    memory_policy="all_prior_compact_signatures",
                )
                call_kwargs = dict(gen_kwargs)
                call_kwargs["prompt_cache_key"] = hashlib.sha256(
                    f"{topic_id}/ideator".encode()
                ).hexdigest()
                idea_text = str(generator.generate(messages, **call_kwargs)).strip()
                if len(idea_text) < config.min_idea_chars:
                    raise ValueError(
                        f"sequential-memory idea {idea_idx} is too short "
                        f"({len(idea_text)} characters; minimum {config.min_idea_chars})"
                    )
                _log_idea(
                    response_logger,
                    topic_id=topic_id,
                    idea_idx=idea_idx,
                    event="open",
                    text=idea_text,
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
                    prior_signatures.append(signature)
                    core = signature_to_final_idea_core(
                        signature, idea_text=idea_text, episode_id=str(idea_idx)
                    )
                else:
                    core = _fallback_core(idea_id=str(idea_idx), idea_text=idea_text)
                _log_idea(
                    response_logger,
                    topic_id=topic_id,
                    idea_idx=idea_idx,
                    event="core",
                    text=str(core.get("core_claim", "")),
                    core=core,
                )
                item = {
                    "idea_id": idea_id,
                    "idea_text": idea_text,
                    "signature_ok": signature is not None,
                    "final_idea_core": core,
                }
                items.append(item)
                row = {
                    "topic_id": topic_id,
                    "generation_index": idea_idx,
                    "idea_id": idea_id,
                    "memory_policy": "all_prior_compact_signatures",
                    "memory_before": memory_before,
                    "generation_directive": directive,
                    "system_prompt": IDEATOR_SYSTEM_PROMPT,
                    "idea_text": idea_text,
                    "idea_char_count": len(idea_text),
                    "idea_word_count": len(idea_text.split()),
                    "signature_ok": signature is not None,
                    "signature": signature.to_json() if signature is not None else None,
                    "memory_after": [s.to_json() for s in prior_signatures],
                    "critic_calls": 0,
                    "quality_evaluator_calls": 0,
                }
                generation_rows.append(row)
                generation_log.write(json.dumps(row, ensure_ascii=False) + "\n")
                generation_log.flush()
                if experiment_logger is not None:
                    experiment_logger.log("baseline_generation_completed", **row)
                print(
                    f"  idea {idea_idx}/{config.n_ideas}: "
                    f"compact-memory entries={len(prior_signatures)}",
                    flush=True,
                )

        record = {
            "topic_id": topic_id,
            "generation_mode": "sequential_compact_memory",
            "generator_calls": config.n_ideas,
            "critic_calls": 0,
            "quality_evaluator_calls": 0,
            "n_requested": config.n_ideas,
            "n_generated": len(items),
            "n_background_papers": len(papers),
            "memory_policy": "all_prior_compact_signatures",
            "items": items,
        }
        records.append(record)
        append_jsonl(output_path, record)
        analysis = {
            "topic_id": topic_id,
            "generation_mode": "sequential_compact_memory",
            "n_generations": len(generation_rows),
            "generator_calls": len(generation_rows),
            "critic_calls": 0,
            "quality_evaluator_calls": 0,
            "signature_successes": sum(row["signature_ok"] for row in generation_rows),
            "memory_entries_by_generation": [len(row["memory_before"]) for row in generation_rows],
            "idea_word_counts": [row["idea_word_count"] for row in generation_rows],
            "ideas": generation_rows,
        }
        (output_dir / f"analysis_{topic_id}.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if experiment_logger is not None:
            experiment_logger.log(
                "topic_completed", topic_id=topic_id,
                mode="sequential_compact_memory", analysis_summary=analysis,
            )
        print(f"--- topic {topic_id}: generated {len(items)}/{config.n_ideas} ideas ---", flush=True)

    if experiment_logger is not None:
        experiment_logger.log(
            "run_completed", mode="sequential_compact_memory",
            topic_records=len(records), records=records,
        )
        summary_path = write_trace_summary(output_dir, logger=experiment_logger)
        experiment_logger.log("trace_summary_written", path=str(summary_path))
    return records
