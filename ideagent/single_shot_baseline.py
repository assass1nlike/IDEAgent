"""Strong, true single-call baseline for topic-level research ideation.

The generator receives the same background context and substantive ideation criteria used by
the agentic/QD arm, then returns all N ideas in one structured response.  There is deliberately
no critic, evaluator, archive, repair, refinement, or second generator turn.  A cheap steno may
extract ``final_idea_core`` records afterward for compatibility with the existing evaluation
pipeline; that extraction cannot change the generated ideas.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ideagent.agent_prompts import (
    IDEATOR_SOUNDNESS_RUBRIC,
    PLAIN_LANGUAGE_RULE,
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
from ideagent.memory import FinalIdeaCore, MemoryManager
from ideagent.response_logging import AsyncResponseLogger
from ideagent.utils import append_jsonl, create_jsonl, prepare_output_dir, read_jsonl
from ideagent.yield_prompts import SOUND_BY_CONSTRUCTION


SINGLE_SHOT_SYSTEM_PROMPT = (
    "You are a creative but rigorous research ideator. In one response, you must produce a "
    "portfolio of genuinely NOVEL, DIVERSE, SOUND, CLEAR, and meaningfully testable research "
    "ideas. You will receive background papers that define a problem space. Ground the ideas in "
    "that space, but do not limit yourself to the papers' specific methods. There is no critic "
    "and no later revision: each idea must be the strongest complete version you can produce now.\n\n"
    "TREAT THE BACKGROUND AS CONTEXT, NOT AS A PARTS LIST. Do not merely combine, extend, "
    "specialize, apply, or ablate methods from the papers. Look for an unresolved problem, a "
    "shared assumption that can be changed, a new intervention point, or a connection to a "
    "different line of reasoning. The mechanism should be motivated by the background while "
    "remaining intellectually independent of any single paper's contribution.\n\n"
    "OPTIMIZE THESE PROPERTIES TOGETHER:\n"
    "1. NON-OBVIOUSNESS. A careful reader of the background should need a genuine independent "
    "leap to reach the idea. A new name, routine combination, direct extension, or different "
    "application of a familiar mechanism is not enough. Surprise borrowed from an invalid step "
    "is not novelty; make the surprising move valid.\n"
    "2. PORTFOLIO DIVERSITY. Diversity is judged between every pair of ideas in this response. "
    "Ideas must differ substantively in the problem they target and/or in their core mechanism, "
    "causal lever, intervention point, or underlying assumption. Do not output a collection of "
    "parameter, "
    "dataset, domain, component, or presentation variants of a smaller set of mechanisms. An "
    "idea may freely compose multiple parts of a system; do not force ideas into predefined loci.\n"
    "3. SOUNDNESS. The causal chain from intervention to claimed effect must be internally "
    "consistent. Do not assume information unavailable in the proposed setting, hide an oracle, "
    "reason circularly, or depend on a signal that becomes constant, empty, or uninformative in "
    "a realistic case. State the central assumptions and address the most likely failure mode or "
    "alternative explanation.\n"
    "4. MECHANISM CLARITY. State the exact intervention, the important steps, what each step "
    "changes, and why those changes should produce the expected outcome. A knowledgeable "
    "researcher should be able to implement the core idea without guessing at missing causal "
    "steps. Distinguish the proposed mechanism from motivation and from hoped-for results.\n"
    "5. FEASIBILITY AND EVALUATION (NON-GATING). State what would be built or changed, what "
    "data/benchmark/environment would be needed, and what measurement would support or falsify "
    "the central claim. Be candid when the required capability or resource does not yet exist. "
    "Feasibility is a reporting dimension, not a veto: do not simplify, conventionalize, or "
    "discard an otherwise novel and sound direction merely to make it immediately buildable.\n\n"
    + SOUND_BY_CONSTRUCTION
    + "\n\n"
    + IDEATOR_SOUNDNESS_RUBRIC
    + "\n\n"
    + PLAIN_LANGUAGE_RULE
    + "\n\n"
    "Before returning the answer, silently self-audit the complete portfolio. Compare every pair "
    "for shared problem framing and mechanism; replace superficial variants. Stress-test each "
    "causal chain for unavailable information, circularity, degeneracy, and unsupported leaps. "
    "Check that each mechanism states an honest implementation and evaluation path, including any "
    "capability that still has to be created. Do not output the audit, scores, rankings, or "
    "comparisons.\n\n"
    "Return only the requested JSON object. Each idea_text must be natural, self-contained "
    "research prose, not bookkeeping fields or an outline for later work."
)


@dataclass(frozen=True)
class SingleShotBaselineConfig:
    """Controls data slicing and validation, never iterative generation."""

    n_ideas: int = 10
    resume: bool = True
    init_topic_idx: int = 0
    end_topic_idx: int | None = 1
    max_background_tokens: int | None = None
    min_idea_chars: int = 300

    def validate(self) -> None:
        if self.n_ideas <= 0:
            raise ValueError("single_shot.n_ideas must be a positive integer")
        if self.init_topic_idx < 0:
            raise ValueError("single_shot.init_topic_idx must be >= 0")
        if self.end_topic_idx is not None and self.end_topic_idx <= self.init_topic_idx:
            raise ValueError("single_shot.end_topic_idx must be > init_topic_idx")
        if self.max_background_tokens is not None and self.max_background_tokens <= 0:
            raise ValueError("single_shot.max_background_tokens must be null or positive")
        if self.min_idea_chars < 1:
            raise ValueError("single_shot.min_idea_chars must be positive")


def single_shot_response_format(n_ideas: int) -> dict[str, Any]:
    """A dynamic schema that makes the one response contain exactly ``n_ideas`` items."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"single_shot_research_ideas_{n_ideas}",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ideas": {
                        "type": "array",
                        "minItems": n_ideas,
                        "maxItems": n_ideas,
                        "items": {
                            "type": "object",
                            "properties": {"idea_text": {"type": "string"}},
                            "required": ["idea_text"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["ideas"],
                "additionalProperties": False,
            },
        },
    }


def build_single_shot_messages(*, background_context: str, n_ideas: int) -> list[dict[str, str]]:
    user_prompt = (
        f"{background_context}\n\n"
        f"Generate exactly {n_ideas} complete research ideas in this single response. The set as "
        "a whole must maximize the number of ideas that are simultaneously diverse, non-obvious, "
        "sound, and mechanistically clear. Also make feasibility explicit as instructed, without "
        "discarding a strong ambitious mechanism merely because it needs more work.\n\n"
        "Each idea_text must be concise, complete, self-contained research prose covering the "
        "motivation and problem, the non-obvious insight, the concrete mechanism and causal "
        "justification, central assumptions and likely failure mode, implementation/evaluation "
        "path, and expected effect.\n\n"
        f"The ideas array must contain exactly {n_ideas} items. Do not include scores, rankings, "
        "critic dialogue, or a recommendation about which ideas are best."
    )
    return [
        {"role": "system", "content": SINGLE_SHOT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"single-shot generator returned invalid JSON: {exc}") from exc


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_single_shot_ideas(
    raw: str, *, n_ideas: int, min_idea_chars: int
) -> list[dict[str, str]]:
    """Fail closed without asking the ideator for a repair (which would cease to be one-shot)."""

    obj = _extract_json(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("ideas"), list):
        raise ValueError("single-shot generator response must be an object with an ideas array")
    ideas = obj["ideas"]
    if len(ideas) != n_ideas:
        raise ValueError(
            f"single-shot generator returned {len(ideas)} ideas; exactly {n_ideas} required"
        )

    parsed: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    for index, item in enumerate(ideas, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"single-shot idea {index} must be an object")
        idea_text = str(item.get("idea_text", "")).strip()
        if len(idea_text) < min_idea_chars:
            raise ValueError(
                f"single-shot idea {index} is too short ({len(idea_text)} characters; "
                f"minimum {min_idea_chars})"
            )
        norm_text = _normalized(idea_text)
        if norm_text in seen_texts:
            raise ValueError(f"single-shot idea {index} exactly duplicates an earlier idea")
        seen_texts.add(norm_text)
        parsed.append({"idea_text": idea_text})
    return parsed


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
        "generation_mode": "single_call_batch",
    }
    if core is not None:
        metadata["core"] = core
    response_logger.log(role="ideator", event=event, text=text, metadata=metadata)


def run_single_shot_baseline(
    *,
    input_jsonl: str | Path,
    output_dir: str | Path,
    generator: Any,
    gen_kwargs: dict[str, Any],
    config: SingleShotBaselineConfig,
    steno_client: Any,
    steno_gen_kwargs: dict[str, Any] | None = None,
    response_logger: AsyncResponseLogger | None = None,
    experiment_logger: ExperimentLogger | None = None,
) -> list[dict[str, Any]]:
    """Generate one N-idea batch per topic and postprocess it without generation feedback."""

    config.validate()
    if steno_client is None:
        raise ValueError("A steno_client is required to produce evaluator-compatible idea cores")

    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, resume=config.resume, init_topic_idx=config.init_topic_idx)
    if experiment_logger is not None:
        fairness_contract = {
            "comparison_target": "yield_qd",
            "matched": {
                "candidate_output_budget": config.n_ideas,
                "background_builder": "build_background_context",
                "substantive_objectives": [
                    "portfolio_diversity", "non_obviousness", "soundness",
                    "mechanism_clarity", "feasibility_reporting", "plain_language",
                ],
                "posthoc_evaluation_field": "items[].idea_text",
            },
            "intentional_treatment_difference": {
                "generator_calls": 1,
                "ideas_per_call": config.n_ideas,
                "critic_calls": 0,
                "generation_memory": "none; all ideas are emitted in one batch",
                "repair_or_refinement": False,
            },
            "not_claimed": [
                "generator-call matched", "per-call output-limit matched",
                "realized-token matched", "monetary-cost matched",
            ],
            "recommended_role": (
                "strict one-call secondary baseline; use sequential_compact_memory as the primary "
                "candidate-call-matched baseline"
            ),
        }
        manifest = {
            "mode": "single_call_batch",
            "config": asdict(config),
            "models": {
                "generator": getattr(generator, "model_id", ""),
                "steno": getattr(steno_client, "model_id", ""),
            },
            "sampling": {"generator": gen_kwargs, "steno": steno_gen_kwargs or {}},
            "prompts": {
                "system_prompt": SINGLE_SHOT_SYSTEM_PROMPT,
                "system_prompt_sha256": text_sha256(SINGLE_SHOT_SYSTEM_PROMPT),
                "response_format": single_shot_response_format(config.n_ideas),
            },
            "fairness_contract": fairness_contract,
            "storage_policy": {
                "model_context": "one batch prompt; no prior generated ideas or feedback",
                "disk": "full batch prompt/output, parsed full idea text, and steno cores",
            },
        }
        manifest_path = write_run_manifest(
            output_dir, logger=experiment_logger, manifest=manifest
        )
        experiment_logger.log(
            "run_started", mode="single_call_batch", config=asdict(config),
            models=manifest["models"], manifest_path=str(manifest_path),
            fairness_contract=fairness_contract,
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

    steno = MemoryManager(steno_client=steno_client, gen_kwargs=steno_gen_kwargs or {})
    groups = load_raw_topic_groups(input_jsonl)[config.init_topic_idx : config.end_topic_idx]
    response_format = single_shot_response_format(config.n_ideas)

    for group in groups:
        topic_id = str(group["topic_id"])
        if topic_id in completed_topic_ids:
            print(f"Topic {topic_id}: already completed, skipping.", flush=True)
            continue

        papers = [parse_paper(paper) for paper in group["background_papers"]]
        # Match Yield/QD context construction exactly, including its optional truncation order.
        background_context = build_background_context(papers)
        if config.max_background_tokens:
            background_context = truncate_by_tokens_rough(
                background_context, config.max_background_tokens
            )
        messages = build_single_shot_messages(
            background_context=background_context, n_ideas=config.n_ideas
        )
        if experiment_logger is not None:
            experiment_logger.log(
                "topic_started", topic_id=topic_id, mode="single_call_batch",
                background=background_context, exact_generator_messages=messages,
            )

        print(
            f"\n=== Single-shot baseline — topic {topic_id}: "
            f"one generator call for {config.n_ideas} ideas ===",
            flush=True,
        )
        request_kwargs = dict(gen_kwargs)
        request_kwargs["response_format"] = response_format
        set_trace_context(
            generator, topic_id=topic_id, stage="batch_candidate_generation",
            generation_index=1, ideas_requested=config.n_ideas,
        )
        raw = generator.generate(messages, **request_kwargs)
        ideas = parse_single_shot_ideas(
            raw, n_ideas=config.n_ideas, min_idea_chars=config.min_idea_chars
        )

        items: list[dict[str, Any]] = []
        generation_rows: list[dict[str, Any]] = []
        generations_path = output_dir / f"generations_{topic_id}.jsonl"
        for idea_idx, idea in enumerate(ideas, start=1):
            idea_text = idea["idea_text"]
            _log_idea(
                response_logger,
                topic_id=topic_id,
                idea_idx=idea_idx,
                event="open",
                text=idea_text,
            )
            set_trace_context(
                steno_client, topic_id=topic_id, candidate_id=str(idea_idx),
                generation_index=idea_idx, stage="posthoc_core_extraction",
            )
            core: FinalIdeaCore = steno.extract_final_idea_core(
                topic_id=topic_id,
                target_paper_id=str(idea_idx),
                round_no=1,
                background_context=background_context,
                latest_ideator_turn=idea_text,
            )
            core_json = core.to_json()
            _log_idea(
                response_logger,
                topic_id=topic_id,
                idea_idx=idea_idx,
                event="core",
                text=core.core_claim,
                core=core_json,
            )
            item = {
                "idea_id": str(idea_idx),
                "idea_text": idea_text,
                "final_idea_core": core_json,
            }
            items.append(item)
            generation_rows.append({
                "topic_id": topic_id,
                "generation_mode": "single_call_batch",
                "batch_call_index": 1,
                "idea_index_in_batch": idea_idx,
                "idea_id": str(idea_idx),
                "idea_text": idea_text,
                "idea_char_count": len(idea_text),
                "idea_word_count": len(idea_text.split()),
                "final_idea_core": core_json,
                "critic_calls": 0,
                "quality_evaluator_calls": 0,
            })

        with generations_path.open("w", encoding="utf-8") as generation_log:
            for row in generation_rows:
                generation_log.write(json.dumps(row, ensure_ascii=False) + "\n")

        record = {
            "topic_id": topic_id,
            "generation_mode": "single_call_batch",
            "generator_calls": 1,
            "critic_calls": 0,
            "quality_evaluator_calls": 0,
            "n_requested": config.n_ideas,
            "n_generated": len(items),
            "n_background_papers": len(papers),
            "items": items,
        }
        records.append(record)
        append_jsonl(output_path, record)
        analysis = {
            "topic_id": topic_id,
            "generation_mode": "single_call_batch",
            "generator_calls": 1,
            "critic_calls": 0,
            "quality_evaluator_calls": 0,
            "n_generations": len(generation_rows),
            "idea_word_counts": [row["idea_word_count"] for row in generation_rows],
            "exact_generator_messages": messages,
            "raw_batch_output": raw,
            "ideas": generation_rows,
        }
        (output_dir / f"analysis_{topic_id}.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if experiment_logger is not None:
            experiment_logger.log(
                "topic_completed", topic_id=topic_id, mode="single_call_batch",
                raw_batch_output=raw, analysis_summary=analysis, final_record=record,
            )
        print(f"--- topic {topic_id}: generated {len(items)}/{config.n_ideas} ideas ---", flush=True)

    if experiment_logger is not None:
        experiment_logger.log(
            "run_completed", mode="single_call_batch",
            topic_records=len(records), records=records,
        )
        summary_path = write_trace_summary(output_dir, logger=experiment_logger)
        experiment_logger.log("trace_summary_written", path=str(summary_path))
    return records
