"""Closed-book NOVA-style baseline: iterative expand-and-select idea search.

Instantiates NOVA's iterative expansion loop (Hu et al., ACL Findings 2025) inside the closed
background-bank setting so it is directly comparable to Yield/QD and the sequential-memory
baseline. Retrieval of external knowledge is removed (closed book); everything else keeps
NOVA's shape:

  * rounds of candidate generation conditioned on a small pool of "seed directions",
  * a self-reflection call after every non-final round that SELECTS the next seed pool from
    that round's ideas and REPLACES the previous pool (NOVA: "old seed ideas are replaced
    with the newly generated seed ideas"),
  * a final diversity-first selection over the accumulated pool. An LLM selection over compact
    signatures stands in for NOVA's k-means cluster centers, keeping the pipeline
    embedding-free like the rest of this codebase.

Fairness: generation reuses the exact ideator system prompt, free-generation operator,
background builder, and steno signature memory of the sequential-memory baseline. The only
additions are the seed-direction block in expansion rounds and the selection calls. By default
the generator model performs selection (NOVA's self-reflection uses the ideation LLM itself);
passing ``selection_client`` swaps in an external judge with the IDENTICAL selection prompt,
so the two modes differ only in which model executes the call. Either way, selection never
critiques, repairs, or refines idea text — no content feedback ever reaches the generator, so
this remains a no-repair baseline.
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
from ideagent.sequential_memory_baseline import _fallback_core
from ideagent.signature import (
    SemanticSignature,
    _first_json,
    extract_signature,
    signature_to_final_idea_core,
)
from ideagent.utils import append_jsonl, create_jsonl, prepare_output_dir, read_jsonl
from ideagent.prompts import op_free_baseline


_REFLECT_SYSTEM = (
    "You are the selection stage of an iterative research-ideation search. You never "
    "rewrite, critique, or repair ideas; you only SELECT which of this round's ideas are the "
    "most promising directions to expand in the next round. Prefer ideas that are mutually "
    "distinct (different problems and different core mechanisms) and whose mechanism looks "
    "non-obvious yet internally consistent. Reply with JSON only."
)

_FINAL_SYSTEM = (
    "You are the final selection stage of an iterative research-ideation search. You never "
    "rewrite, critique, or repair ideas; you only SELECT the final portfolio from the "
    "accumulated pool. Mutual distinctness comes FIRST: no two selected ideas may share the "
    "same problem and core mechanism. Among distinct candidates, prefer the most promising "
    "(non-obvious yet internally consistent) mechanisms. Reply with JSON only."
)

_SELECTION_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "selection",
        "schema": {
            "type": "object",
            "properties": {
                "selected": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
            "required": ["selected", "rationale"],
            "additionalProperties": False,
        },
    },
}


def _seed_direction_block(seeds: list[SemanticSignature]) -> str:
    return (
        "SEED DIRECTIONS (self-selected from the previous round as the most promising — no "
        "external judgment involved). Treat them as live frontiers: push deeper into the "
        "underlying phenomenon, transfer the diagnosis to a different failure, or attack the "
        "same territory with a fundamentally different mechanism. Your new idea must still be "
        "distinct from EVERY previously generated idea, including these seeds:\n"
        + "\n".join(f"  - {signature.compact()}" for signature in seeds)
    )


def _selection_user(
    background: str,
    candidates: list[SemanticSignature],
    *,
    k: int,
    stage: str,
) -> str:
    listing = "\n".join(f"  - {signature.compact()}" for signature in candidates)
    if stage == "reflect":
        ask = (
            f"Select exactly {k} idea ids from THIS ROUND to become next round's seed "
            "directions."
        )
        header = "THIS ROUND'S IDEAS (compact signatures):"
    else:
        ask = (
            f"Select exactly {k} idea ids forming the final portfolio of mutually distinct, "
            "promising ideas."
        )
        header = "ACCUMULATED IDEA POOL (compact signatures):"
    return "\n\n".join(
        [
            background,
            f"{header}\n{listing}",
            ask
            + ' Reply with JSON: {"selected": ["<id>", ...], "rationale": "<one short '
            'paragraph>"}. Use only ids shown above; no duplicates.',
        ]
    )


def _select_ids(
    client: Any,
    *,
    gen_kwargs: dict[str, Any],
    system: str,
    user: str,
    candidate_ids: list[str],
    k: int,
    cache_key: str,
) -> tuple[list[str], str]:
    """One selection call with a single retry; fails closed on a second bad reply."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    call_kwargs = dict(gen_kwargs)
    call_kwargs["prompt_cache_key"] = cache_key
    call_kwargs["response_format"] = _SELECTION_RESPONSE_FORMAT
    valid = set(candidate_ids)
    last_error = ""
    for _ in range(2):
        obj = _first_json(str(client.generate(messages, **call_kwargs)))
        selected = obj.get("selected")
        if isinstance(selected, list):
            picked = [str(idea_id) for idea_id in selected]
            if len(picked) == k and len(set(picked)) == k and set(picked) <= valid:
                return picked, str(obj.get("rationale", ""))
            last_error = f"invalid selection {picked!r}"
        else:
            last_error = "no selected list in reply"
    raise ValueError(f"selection failed closed after retry: {last_error} (expected {k} of {candidate_ids})")


@dataclass(frozen=True)
class NovaClosedBookConfig:
    rounds: int = 3
    ideas_per_round: int = 10
    seeds_per_round: int = 3
    final_k: int = 10
    # seed_directions=False disables NOVA's seed steering AND the per-round reflections: every
    # generation call is then byte-identical to the sequential-memory baseline (same system
    # prompt, same op_free_baseline directive, same signature memory, built by the same
    # functions), and the only selection call left is the final portfolio pick.
    seed_directions: bool = True
    resume: bool = True
    init_topic_idx: int = 0
    end_topic_idx: int | None = 1
    max_background_tokens: int | None = None
    min_idea_chars: int = 300
    signature_char_cap: int = 256

    def validate(self) -> None:
        if self.rounds <= 0:
            raise ValueError("nova.rounds must be a positive integer")
        if self.ideas_per_round <= 0:
            raise ValueError("nova.ideas_per_round must be a positive integer")
        if not 0 < self.seeds_per_round <= self.ideas_per_round:
            raise ValueError("nova.seeds_per_round must be in [1, ideas_per_round]")
        if self.final_k <= 0:
            raise ValueError("nova.final_k must be a positive integer")
        if self.final_k > self.rounds * self.ideas_per_round:
            raise ValueError("nova.final_k cannot exceed rounds * ideas_per_round")
        if self.init_topic_idx < 0:
            raise ValueError("nova.init_topic_idx must be >= 0")
        if self.end_topic_idx is not None and self.end_topic_idx <= self.init_topic_idx:
            raise ValueError("nova.end_topic_idx must be > init_topic_idx")
        if self.max_background_tokens is not None and self.max_background_tokens <= 0:
            raise ValueError("nova.max_background_tokens must be null or positive")
        if self.min_idea_chars < 1:
            raise ValueError("nova.min_idea_chars must be positive")
        if self.signature_char_cap < 16:
            raise ValueError("nova.signature_char_cap must be >= 16")


def run_nova_closed_book(
    *,
    input_jsonl: str | Path,
    output_dir: str | Path,
    generator: Any,
    gen_kwargs: dict[str, Any],
    config: NovaClosedBookConfig,
    steno_client: Any,
    steno_gen_kwargs: dict[str, Any] | None = None,
    selection_client: Any | None = None,
    selection_gen_kwargs: dict[str, Any] | None = None,
    response_logger: AsyncResponseLogger | None = None,
    experiment_logger: ExperimentLogger | None = None,
) -> list[dict[str, Any]]:
    """Run rounds of expand-and-select and return final_k selected ideas per topic.

    ``selection_client=None`` (default) keeps NOVA-faithful self-reflection: the generator
    model performs every selection call. Passing an external judge client swaps only the model
    behind those calls; prompts and ids-only semantics are identical in both modes.
    """

    config.validate()
    if steno_client is None:
        raise ValueError("A steno_client is required for compact signature memory")
    selector = selection_client if selection_client is not None else generator
    selector_kwargs = (
        (selection_gen_kwargs or {}) if selection_client is not None else gen_kwargs
    )
    selection_mode = "external_judge" if selection_client is not None else "self_reflection"
    memory_policy = (
        "all_prior_compact_signatures+seed_pool"
        if config.seed_directions
        else "all_prior_compact_signatures"
    )

    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, resume=config.resume, init_topic_idx=config.init_topic_idx)
    system_prompt = IDEATOR_SYSTEM_PROMPT
    generator_calls = config.rounds * config.ideas_per_round
    selection_calls = ((config.rounds - 1) if config.seed_directions else 0) + 1
    if experiment_logger is not None:
        fairness_contract = {
            "comparison_target": "yield_qd",
            "matched": {
                "candidate_output_budget": config.final_k,
                "one_idea_per_generator_call": True,
                "generator_calls": generator_calls,
                "background_builder": "build_background_context",
                "ideator_system_prompt_exact": True,
                "ideator_prompts_byte_identical_to_sequential_memory": not config.seed_directions,
                "ideator_system_prompt_sha256": text_sha256(system_prompt),
                "closed_book": "no retrieval; background bank only (NOVA's retrieval removed)",
                "reset_context_each_candidate": True,
                "topic_wide_prompt_cache_key": True,
                "posthoc_evaluation_field": "items[].idea_text",
            },
            "intentional_treatment_difference": {
                "critic_calls": 0,
                "quality_or_novelty_feedback_during_generation": False,
                "repair_or_refinement": False,
                "selection_calls": selection_calls,
                "selection_mode": selection_mode,
                "selection_semantics": (
                    "NOVA-style selection: picks ids from compact signatures, never critiques "
                    "or edits idea text; executed by "
                    + (
                        "an external judge model with the identical prompt"
                        if selection_client is not None
                        else "the generator model (self-reflection)"
                    )
                ),
                "memory": (
                    "all earlier compact signatures, neutrally labelled"
                    + (
                        ", plus the current seed-direction pool"
                        if config.seed_directions
                        else " (identical to the sequential-memory baseline)"
                    )
                    + "; no score, gate, or critic signal"
                ),
            },
            "not_claimed": [
                "equal total model calls because QD invokes evaluators and critics",
                "equal realized token count or monetary cost",
                "faithfulness to NOVA's open-book retrieval, which the task forbids",
            ],
        }
        manifest = {
            "mode": "nova_closed_book",
            "config": asdict(config),
            "models": {
                "generator": getattr(generator, "model_id", ""),
                "steno": getattr(steno_client, "model_id", ""),
                "selector": getattr(selector, "model_id", ""),
            },
            "sampling": {
                "generator": gen_kwargs,
                "steno": steno_gen_kwargs or {},
                "selector": selector_kwargs,
            },
            "prompts": {
                "ideator_system_prompt": system_prompt,
                "ideator_system_prompt_sha256": text_sha256(system_prompt),
                "reflect_system_prompt": _REFLECT_SYSTEM,
                "final_selection_system_prompt": _FINAL_SYSTEM,
                "first_free_directive": op_free_baseline([]),
            },
            "fairness_contract": fairness_contract,
            "storage_policy": {
                "model_context": "compact signatures only",
                "disk": "full prompts, full outputs, full idea text, and signatures",
            },
        }
        manifest_path = write_run_manifest(output_dir, logger=experiment_logger, manifest=manifest)
        experiment_logger.log(
            "run_started", mode="nova_closed_book", config=asdict(config),
            models={
                "generator": getattr(generator, "model_id", ""),
                "steno": getattr(steno_client, "model_id", ""),
                "selector": getattr(selector, "model_id", ""),
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
            f"\n=== NOVA-style closed-book baseline — topic {topic_id}: "
            f"{config.rounds}x{config.ideas_per_round} generator calls, "
            f"select {config.final_k} ===",
            flush=True,
        )
        prior_signatures: list[SemanticSignature] = []
        seed_pool: list[SemanticSignature] = []
        pool_items: dict[str, dict[str, Any]] = {}
        generation_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        generations_path = output_dir / f"generations_{topic_id}.jsonl"
        ideator_cache_key = hashlib.sha256(f"{topic_id}/ideator".encode()).hexdigest()
        selection_cache_key = hashlib.sha256(f"{topic_id}/reflect".encode()).hexdigest()

        if experiment_logger is not None:
            experiment_logger.log(
                "topic_started", topic_id=topic_id, mode="nova_closed_book",
                background=background,
            )

        with generations_path.open("w", encoding="utf-8") as generation_log:
            idea_counter = 0
            for round_idx in range(1, config.rounds + 1):
                round_signatures: list[SemanticSignature] = []
                seed_ids = [signature.id for signature in seed_pool]
                for _ in range(config.ideas_per_round):
                    idea_counter += 1
                    idea_id = f"i{idea_counter:03d}"
                    memory_before = [signature.to_json() for signature in prior_signatures]
                    directive_parts = []
                    if seed_pool:
                        directive_parts.append(_seed_direction_block(seed_pool))
                    directive_parts.append(op_free_baseline(prior_signatures))
                    directive = "\n\n".join(directive_parts)
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
                    set_trace_context(
                        generator, topic_id=topic_id, candidate_id=idea_id,
                        generation_index=idea_counter, stage="candidate_generation",
                        memory_policy=memory_policy,
                    )
                    call_kwargs = dict(gen_kwargs)
                    call_kwargs["prompt_cache_key"] = ideator_cache_key
                    idea_text = str(generator.generate(messages, **call_kwargs)).strip()
                    if len(idea_text) < config.min_idea_chars:
                        raise ValueError(
                            f"nova-closed-book idea {idea_counter} is too short "
                            f"({len(idea_text)} characters; minimum {config.min_idea_chars})"
                        )
                    if response_logger is not None:
                        response_logger.log(
                            role="ideator", event="open", text=idea_text,
                            metadata={
                                "topic_id": topic_id,
                                "target_paper_id": str(idea_counter),
                                "generation_mode": "nova_closed_book",
                                "round": round_idx,
                            },
                        )

                    set_trace_context(
                        steno_client, topic_id=topic_id, candidate_id=idea_id,
                        generation_index=idea_counter, stage="signature_extraction",
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
                        round_signatures.append(signature)
                        core = signature_to_final_idea_core(
                            signature, idea_text=idea_text, episode_id=str(idea_counter)
                        )
                    else:
                        core = _fallback_core(idea_id=str(idea_counter), idea_text=idea_text)
                    if response_logger is not None:
                        # Same "core" companion record the sequential baseline writes, so the
                        # responses/ layout stays byte-compatible for downstream tools.
                        response_logger.log(
                            role="ideator", event="core", text=str(core.get("core_claim", "")),
                            metadata={
                                "topic_id": topic_id,
                                "target_paper_id": str(idea_counter),
                                "generation_mode": "nova_closed_book",
                                "round": round_idx,
                                "core": core,
                            },
                        )
                    pool_items[idea_id] = {
                        "idea_id": idea_id,
                        "idea_text": idea_text,
                        "signature_ok": signature is not None,
                        "final_idea_core": core,
                    }
                    row = {
                        "topic_id": topic_id,
                        "generation_index": idea_counter,
                        "idea_id": idea_id,
                        "round": round_idx,
                        "seed_pool_ids": seed_ids,
                        "memory_policy": memory_policy,
                        "memory_before": memory_before,
                        "generation_directive": directive,
                        "system_prompt": system_prompt,
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
                        f"  round {round_idx}/{config.rounds} idea "
                        f"{idea_counter}/{generator_calls}: "
                        f"compact-memory entries={len(prior_signatures)}",
                        flush=True,
                    )

                if config.seed_directions and round_idx < config.rounds:
                    # NOVA self-reflection: pick next seed pool from THIS round only; the
                    # previous pool is replaced, not extended.
                    k = min(config.seeds_per_round, len(round_signatures))
                    if k == 0:
                        raise ValueError(
                            f"round {round_idx} produced no parseable signatures; cannot "
                            "select a seed pool"
                        )
                    set_trace_context(
                        selector, topic_id=topic_id, candidate_id=f"reflect_r{round_idx}",
                        generation_index=idea_counter, stage="seed_selection",
                    )
                    picked, rationale = _select_ids(
                        selector,
                        gen_kwargs=selector_kwargs,
                        system=_REFLECT_SYSTEM,
                        user=_selection_user(
                            background, round_signatures, k=k, stage="reflect"
                        ),
                        candidate_ids=[s.id for s in round_signatures],
                        k=k,
                        cache_key=selection_cache_key,
                    )
                    by_id = {s.id: s for s in round_signatures}
                    seed_pool = [by_id[idea_id] for idea_id in picked]
                    selection_row = {
                        "topic_id": topic_id,
                        "stage": "seed_selection",
                        "selection_mode": selection_mode,
                        "round": round_idx,
                        "candidate_ids": [s.id for s in round_signatures],
                        "selected_ids": picked,
                        "rationale": rationale,
                    }
                    selection_rows.append(selection_row)
                    if experiment_logger is not None:
                        experiment_logger.log("baseline_selection_completed", **selection_row)
                    print(
                        f"  round {round_idx} reflection: seed pool -> {picked}", flush=True
                    )

            # Final diversity-first selection over every parseable idea in the pool.
            k_final = min(config.final_k, len(prior_signatures))
            if k_final == 0:
                raise ValueError("no parseable signatures in pool; cannot select a portfolio")
            set_trace_context(
                selector, topic_id=topic_id, candidate_id="final_selection",
                generation_index=idea_counter, stage="final_selection",
            )
            picked, rationale = _select_ids(
                selector,
                gen_kwargs=selector_kwargs,
                system=_FINAL_SYSTEM,
                user=_selection_user(background, prior_signatures, k=k_final, stage="final"),
                candidate_ids=[s.id for s in prior_signatures],
                k=k_final,
                cache_key=selection_cache_key,
            )
            selection_row = {
                "topic_id": topic_id,
                "stage": "final_selection",
                "selection_mode": selection_mode,
                "round": config.rounds,
                "candidate_ids": [s.id for s in prior_signatures],
                "selected_ids": picked,
                "rationale": rationale,
            }
            selection_rows.append(selection_row)
            if experiment_logger is not None:
                experiment_logger.log("baseline_selection_completed", **selection_row)
            print(f"  final selection: {picked}", flush=True)

        items = [pool_items[idea_id] for idea_id in picked]
        record = {
            "topic_id": topic_id,
            "generation_mode": "nova_closed_book",
            "generator_calls": generator_calls,
            "selection_calls": selection_calls,
            "selection_mode": selection_mode,
            "selector_model": getattr(selector, "model_id", ""),
            "critic_calls": 0,
            "quality_evaluator_calls": 0,
            "n_requested": config.final_k,
            "n_generated": len(pool_items),
            "n_selected": len(items),
            "n_background_papers": len(papers),
            "memory_policy": memory_policy,
            "seed_directions": config.seed_directions,
            "items": items,
        }
        records.append(record)
        append_jsonl(output_path, record)
        analysis = {
            "topic_id": topic_id,
            "generation_mode": "nova_closed_book",
            "n_generations": len(generation_rows),
            "generator_calls": len(generation_rows),
            "selection_calls": len(selection_rows),
            "selection_mode": selection_mode,
            "critic_calls": 0,
            "quality_evaluator_calls": 0,
            "signature_successes": sum(row["signature_ok"] for row in generation_rows),
            "memory_entries_by_generation": [len(row["memory_before"]) for row in generation_rows],
            "idea_word_counts": [row["idea_word_count"] for row in generation_rows],
            "selections": selection_rows,
            "final_selected_ids": picked,
            "ideas": generation_rows,
        }
        (output_dir / f"analysis_{topic_id}.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if experiment_logger is not None:
            experiment_logger.log(
                "topic_completed", topic_id=topic_id,
                mode="nova_closed_book", analysis_summary=analysis,
            )
        print(
            f"--- topic {topic_id}: generated {len(pool_items)}, "
            f"selected {len(items)}/{config.final_k} ideas ---",
            flush=True,
        )

    if experiment_logger is not None:
        experiment_logger.log(
            "run_completed", mode="nova_closed_book",
            topic_records=len(records), records=records,
        )
        summary_path = write_trace_summary(output_dir, logger=experiment_logger)
        experiment_logger.log("trace_summary_written", path=str(summary_path))
    return records
