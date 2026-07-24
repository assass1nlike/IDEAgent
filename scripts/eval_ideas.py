"""Run comprehensive quality eval and topical diversity eval in one command.

Combines what were previously two separate scripts (eval_full_idea_quality.py and
eval_topical_diversity.py) into a single entry point. Each evaluator still runs
exactly as before -- same evaluator classes, same config sections, same output
schemas -- so this changes nothing about the underlying evaluation, only how many
commands you need to invoke.

    python scripts/eval_ideas.py --config configs/comp_quality_eval_opus47.yaml configs/diversity_eval_opus47.yaml \\
        --input results_organized/generations/<baseline>/final_idea_cores.jsonl \\
        --quality-output results_organized/<judge>/<baseline>/comp_quality_report.jsonl \\
        --diversity-output results_organized/<judge>/<baseline>/diversity_report_<baseline>.jsonl \\
        --resume

Pass --skip-quality or --skip-diversity to run only one half (equivalent to the old
single-purpose scripts). --start-idx/--end-idx slice the topic list (0-indexed,
end exclusive) for sharding a run across parallel processes/API keys.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ideagent.clients import GeminiClient, build_client
from ideagent.comp_quality_eval import (
    ComprehensiveQualityEvaluator,
    GeminiComprehensiveQualityEvaluator,
    discover_topics,
    extract_ideas_from_record,
    extract_ideas_from_responses,
)
from ideagent.console import print_warning
from ideagent.diversity_eval import (
    AXIS_CALL_MODES,
    DEFAULT_AXIS_CALL_MODE,
    GeminiTopicalDiversityEvaluator,
    TopicalDiversityEvaluator,
    detect_format,
    extract_ideas_from_agentic_record,
)
from ideagent.utils import append_jsonl, create_jsonl, load_config, read_jsonl


def _build_client_pair(evaluator_cfg: dict, client_cfg: dict):
    model_id = evaluator_cfg.get("model_id")
    if not model_id:
        raise ValueError("evaluator.model_id must be set")
    client = build_client(
        model_id=model_id,
        vllm_port=int(evaluator_cfg.get("port", 8000)),
        api_key_env=client_cfg.get("api_key_env", "VLLM_API_KEY"),
        request_timeout=float(client_cfg.get("request_timeout", 120.0)),
    )
    escalation_model_id = evaluator_cfg.get("escalation_model_id")
    escalation_client = None
    if escalation_model_id:
        escalation_client = build_client(
            model_id=escalation_model_id,
            vllm_port=int(evaluator_cfg.get("port", 8000)),
            api_key_env=client_cfg.get("api_key_env", "VLLM_API_KEY"),
            request_timeout=float(client_cfg.get("request_timeout", 120.0)),
        )
        print(f"Escalation: {escalation_model_id} after {evaluator_cfg.get('escalation_after_attempts', 5)} failed attempts")
    return client, escalation_client


def _quality_sampling(cfg: dict) -> dict:
    return {
        "temperature": float(cfg.get("temperature", 0.3)),
        "top_p": float(cfg.get("top_p", 0.95)),
        "top_k": cfg.get("top_k"),
        "enable_thinking": cfg.get("enable_thinking", False),
        "thinking_effort": cfg.get("thinking_effort"),
        "max_new_tokens": int(cfg.get("max_new_tokens", 2048)),
    }


def _diversity_sampling(cfg: dict) -> dict:
    return {
        "temperature": float(cfg.get("temperature", 0.3)),
        "top_p": float(cfg.get("top_p", 0.95)),
        "top_k": cfg.get("top_k"),
        "enable_thinking": cfg.get("enable_thinking", False),
        "thinking_effort": cfg.get("thinking_effort"),
        "max_new_tokens": int(cfg.get("max_new_tokens", 2048)),
        "signature_thinking_effort": cfg.get("signature_thinking_effort"),
        "axis_thinking_effort": cfg.get("axis_thinking_effort"),
        "pair_thinking_effort": cfg.get("pair_thinking_effort"),
        "signature_max_new_tokens": cfg.get("signature_max_new_tokens"),
        "axis_max_new_tokens": cfg.get("axis_max_new_tokens"),
        "joint_axis_thinking_effort": cfg.get("joint_axis_thinking_effort"),
        "joint_axis_max_new_tokens": cfg.get("joint_axis_max_new_tokens"),
        "pair_max_new_tokens": cfg.get("pair_max_new_tokens"),
    }


def run_quality_eval(
    cfg: dict, *, input_path: Path | None, responses_root_cfg: Path | None,
    output_path: Path, resume: bool, max_workers: int | None,
    start_idx: int | None, end_idx: int | None,
) -> None:
    data_cfg = cfg.get("data", {})
    evaluator_cfg = cfg.get("evaluator", {})
    client_cfg = cfg.get("client", {})
    eval_cfg = cfg.get("eval", {})

    if bool(input_path) == bool(responses_root_cfg):
        raise ValueError(
            "Exactly one of --input / data.responses_root must be set for quality eval "
            "(mutually exclusive): input_jsonl derives topics from agentic_ideas.jsonl's "
            "items and infers responses_root as its sibling 'responses/' dir; "
            "responses_root discovers topics directly from the responses/ directory tree."
        )
    if input_path:
        input_path = Path(input_path)
        responses_root = input_path.parent / "responses"
        print(f"[quality] Input:          {input_path}")
    else:
        responses_root = Path(responses_root_cfg)
    print(f"[quality] Responses root: {responses_root}")
    print(f"[quality] Output:         {output_path}")

    client, escalation_client = _build_client_pair(evaluator_cfg, client_cfg)
    EvaluatorClass = GeminiComprehensiveQualityEvaluator if isinstance(client, GeminiClient) else ComprehensiveQualityEvaluator
    evaluator = EvaluatorClass(
        client=client,
        gen_kwargs=_quality_sampling(evaluator_cfg),
        max_parsing_retries=int(eval_cfg.get("max_parsing_retries", 2)),
        max_workers=int(max_workers if max_workers is not None else eval_cfg.get("max_workers", 10)),
        escalation_client=escalation_client,
        escalation_after_attempts=int(evaluator_cfg.get("escalation_after_attempts", 5)),
    )

    max_topics = eval_cfg.get("max_topics")  # None means run all

    completed: set[str] = set()
    if resume and Path(output_path).exists():
        for r in read_jsonl(output_path):
            if r.get("topic_id"):
                completed.add(r["topic_id"])
        print(f"[quality] Resume: {len(completed)} topic(s) already done, skipping.")
    else:
        create_jsonl(output_path)

    input_records: dict[str, dict] = {}
    if input_path:
        topics: list[tuple[str, int]] = []
        for record in read_jsonl(input_path):
            topic_id = record.get("topic_id", "(unknown)")
            try:
                detect_format(record)
            except ValueError as exc:
                print_warning(f"[comp_quality_eval] {topic_id}: {exc}; skipping.")
                continue
            topics.append((topic_id, len(record.get("items", []))))
            input_records[str(topic_id)] = record
    else:
        topics = discover_topics(responses_root)
    topics = topics[start_idx:end_idx]
    print(f"[quality] Discovered {len(topics)} topic(s).")

    n_processed = 0
    for topic_id, n_ideas in topics:
        if max_topics is not None and len(completed) + n_processed >= max_topics:
            print(f"[quality] Reached max_topics={max_topics} (total across resumed runs); stopping.")
            break
        if topic_id in completed:
            continue
        # Prefer the common raw field in the JSONL. Fall back to legacy episode logs for
        # pre-change artifacts that do not contain idea_text.
        ideas = extract_ideas_from_record(input_records.get(str(topic_id), {}))
        if not ideas:
            ideas = extract_ideas_from_responses(responses_root, topic_id, n_ideas)
        if not ideas:
            print_warning(f"[comp_quality_eval] {topic_id}: no final ideator responses found; skipping.")
            continue
        print(f"  [{topic_id}] evaluating {len(ideas)} ideas (full final response) ...")
        report = evaluator.evaluate(topic_id=topic_id, ideas=ideas)
        if report is None:
            print_warning(f"[comp_quality_eval] {topic_id}: evaluation failed; skipping.")
            continue
        append_jsonl(output_path, asdict(report))
        n_processed += 1


def run_diversity_eval(
    cfg: dict, *, input_path: Path, output_path: Path, resume: bool,
    max_workers: int | None, axis_call_mode: str | None,
    start_idx: int | None, end_idx: int | None,
) -> None:
    data_cfg = cfg.get("data", {})
    evaluator_cfg = cfg.get("evaluator", {})
    client_cfg = cfg.get("client", {})
    eval_cfg = cfg.get("eval", {})

    print(f"[diversity] Input:  {input_path}")
    print(f"[diversity] Output: {output_path}")

    client, escalation_client = _build_client_pair(evaluator_cfg, client_cfg)
    EvaluatorClass = GeminiTopicalDiversityEvaluator if isinstance(client, GeminiClient) else TopicalDiversityEvaluator
    resolved_axis_call_mode = axis_call_mode or evaluator_cfg.get("axis_call_mode", DEFAULT_AXIS_CALL_MODE)
    evaluator = EvaluatorClass(
        client=client,
        gen_kwargs=_diversity_sampling(evaluator_cfg),
        max_parsing_retries=int(eval_cfg.get("max_parsing_retries", 2)),
        max_workers=int(max_workers if max_workers is not None else eval_cfg.get("max_workers", 10)),
        axis_call_mode=resolved_axis_call_mode,
        escalation_client=escalation_client,
        escalation_after_attempts=int(evaluator_cfg.get("escalation_after_attempts", 5)),
        # Sub-topic checkpoint (signatures/NB/axis judgments/pair classifications), so a
        # topic that fails partway through on one hard pair resumes from everything that
        # already succeeded instead of redoing it. No-op for GeminiTopicalDiversityEvaluator
        # (inherited but unused there -- its own evaluate() doesn't read it).
        cache_path=Path(str(output_path) + ".partial.jsonl"),
    )
    expected_version = evaluator.evaluation_version
    print(f"[diversity] Axis calls: {evaluator.axis_call_mode} ({expected_version})")

    max_topics = eval_cfg.get("max_topics")  # None means run all

    completed: set[str] = set()
    if resume and Path(output_path).exists():
        existing = read_jsonl(output_path)
        incompatible = [
            r.get("topic_id", "(unknown)") for r in existing
            if r.get("evaluation_version") != expected_version
        ]
        if incompatible:
            raise SystemExit(
                f"Refusing to append {expected_version} records to an incompatible diversity "
                f"report ({len(incompatible)} incompatible topic(s)). Choose a new "
                "--diversity-output path or archive the old report first."
            )
        for r in existing:
            if r.get("topic_id"):
                completed.add(r["topic_id"])
        print(f"[diversity] Resume: {len(completed)} topic(s) already done, skipping.")
    else:
        create_jsonl(output_path)

    raw_records = read_jsonl(input_path)[start_idx:end_idx]

    n_processed = 0
    for record in raw_records:
        if max_topics is not None and len(completed) + n_processed >= max_topics:
            print(f"[diversity] Reached max_topics={max_topics} (total across resumed runs); stopping.")
            break
        topic_id = record.get("topic_id", "(unknown)")
        if topic_id in completed:
            continue
        try:
            detect_format(record)
        except ValueError as exc:
            print_warning(f"[diversity_eval] {topic_id}: {exc}; skipping.")
            continue
        ideas = extract_ideas_from_agentic_record(record)
        print(f"  [{topic_id}] evaluating {len(ideas)} ideas ...")
        try:
            report = evaluator.evaluate(topic_id=topic_id, ideas=ideas)
        except RuntimeError as exc:
            # evaluate() raises rather than returning a partial report when any wave-1/
            # pairwise call exhausts its retries, since a topic with some ideas scored
            # and others missing would silently corrupt downstream lineage-selection and
            # aggregate stats. Skip this topic (nothing written for it) and move on to
            # the rest of the batch instead of aborting the whole run over one topic.
            print_warning(f"[diversity_eval] {topic_id}: {exc}; skipping.")
            continue
        if report is None:
            print_warning(f"[diversity_eval] {topic_id}: evaluation failed; skipping.")
            continue
        append_jsonl(output_path, asdict(report))
        n_processed += 1


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ModuleNotFoundError:
        pass

    parser = argparse.ArgumentParser(description="Run comprehensive quality eval and topical diversity eval in one command.")
    parser.add_argument(
        "--config", type=Path, nargs="+", required=True,
        help="One config (shared by both evals) or two configs (quality first, diversity second).",
    )
    parser.add_argument("--input", type=Path, default=None, help="Override data.input_jsonl for both evals.")
    parser.add_argument("--responses-root", type=Path, default=None, help="Override data.responses_root (quality only).")
    parser.add_argument("--quality-output", type=Path, default=None, help="Override quality data.output_jsonl.")
    parser.add_argument("--diversity-output", type=Path, default=None, help="Override diversity data.output_jsonl.")
    parser.add_argument("--resume", action="store_true", default=None, help="Skip already-completed topics, for both evals.")
    parser.add_argument("--max-workers", type=int, default=None, help="Override eval.max_workers for both evals.")
    parser.add_argument("--start-idx", type=int, default=None, help="0-indexed start of the topic slice to run (shared by both evals).")
    parser.add_argument("--end-idx", type=int, default=None, help="0-indexed, exclusive end of the topic slice to run (shared by both evals).")
    parser.add_argument("--axis-call-mode", choices=AXIS_CALL_MODES, default=None, help="Override diversity evaluator.axis_call_mode.")
    parser.add_argument("--skip-quality", action="store_true", help="Run diversity eval only.")
    parser.add_argument("--skip-diversity", action="store_true", help="Run quality eval only.")
    args = parser.parse_args()

    if len(args.config) not in (1, 2):
        parser.error("--config takes exactly one (shared) or two (quality, diversity) paths.")
    quality_cfg_path = args.config[0]
    diversity_cfg_path = args.config[-1]

    if not args.skip_quality:
        cfg = load_config(quality_cfg_path)
        data_cfg = cfg.get("data", {})
        resume = args.resume if args.resume is not None else data_cfg.get("resume", False)
        run_quality_eval(
            cfg,
            input_path=args.input or data_cfg.get("input_jsonl"),
            responses_root_cfg=args.responses_root or data_cfg.get("responses_root"),
            output_path=args.quality_output or data_cfg.get("output_jsonl", "data/comp_quality_report.jsonl"),
            resume=resume,
            max_workers=args.max_workers,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )

    if not args.skip_diversity:
        cfg = load_config(diversity_cfg_path)
        data_cfg = cfg.get("data", {})
        resume = args.resume if args.resume is not None else data_cfg.get("resume", False)
        run_diversity_eval(
            cfg,
            input_path=args.input or data_cfg.get("input_jsonl", "data/agentic_ideas.jsonl"),
            output_path=args.diversity_output or data_cfg.get("output_jsonl", "data/diversity_report.jsonl"),
            resume=resume,
            max_workers=args.max_workers,
            axis_call_mode=args.axis_call_mode,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )


if __name__ == "__main__":
    main()
