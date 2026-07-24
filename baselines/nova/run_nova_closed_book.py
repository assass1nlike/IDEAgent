"""Run the closed-book NOVA-style expand-and-select baseline.

    python baselines/nova/run_nova_closed_book.py \
        --config baselines/nova/nova_closed_book.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines.nova.pipeline import NovaClosedBookConfig, run_nova_closed_book
from ideagent.clients import build_client
from ideagent.experiment_logging import ExperimentLogger, TracedClient
from ideagent.response_logging import AsyncResponseLogger
from ideagent.utils import load_config


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _sampling(agent_cfg: dict) -> dict:
    kwargs: dict = {
        "enable_thinking": agent_cfg.get("enable_thinking", False),
        "thinking_effort": agent_cfg.get("thinking_effort"),
        "max_new_tokens": int(agent_cfg.get("max_new_tokens", 8192)),
    }
    for key in (
        "temperature", "top_p", "top_k", "min_p",
        "presence_penalty", "repetition_penalty",
    ):
        if agent_cfg.get(key) is not None:
            kwargs[key] = agent_cfg[key]
    return kwargs


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass

    parser = argparse.ArgumentParser(
        description="Closed-book NOVA-style iterative expand-and-select baseline."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT_DIR / "baselines" / "nova" / "nova_closed_book.yaml"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg.get("data", {})
    run_cfg = cfg.get("nova", {})
    client_cfg = cfg.get("client", {})
    generator_cfg = cfg.get("generator", {})
    steno_cfg = cfg.get("steno", {})
    judge_cfg = cfg.get("judge", {})

    generator_model = generator_cfg.get("model_id")
    steno_model = steno_cfg.get("model_id")
    if not generator_model:
        raise ValueError(f"generator.model_id must be set in {args.config}")
    if not steno_model:
        raise ValueError(f"steno.model_id must be set in {args.config}")

    timeout = float(client_cfg.get("request_timeout", 3600))
    default_api_key_env = client_cfg.get("api_key_env", "GEMINI_API_KEY")
    raw_generator = build_client(
        model_id=generator_model,
        vllm_port=int(generator_cfg.get("port", 8000)),
        api_key_env=generator_cfg.get("api_key_env", default_api_key_env),
        request_timeout=timeout,
    )
    raw_steno = build_client(
        model_id=steno_model,
        vllm_port=int(steno_cfg.get("port", 8000)),
        api_key_env=steno_cfg.get("api_key_env", default_api_key_env),
        request_timeout=timeout,
    )
    judge_model = judge_cfg.get("model_id")
    raw_judge = (
        build_client(
            model_id=judge_model,
            vllm_port=int(judge_cfg.get("port", 8000)),
            api_key_env=judge_cfg.get("api_key_env", default_api_key_env),
            request_timeout=timeout,
        )
        if judge_model
        else None
    )

    config = NovaClosedBookConfig(
        rounds=int(run_cfg.get("rounds", 3)),
        ideas_per_round=int(run_cfg.get("ideas_per_round", 10)),
        seeds_per_round=int(run_cfg.get("seeds_per_round", 3)),
        final_k=int(run_cfg.get("final_k", 10)),
        seed_directions=bool(run_cfg.get("seed_directions", True)),
        resume=bool(run_cfg.get("resume", True)),
        init_topic_idx=int(run_cfg.get("init_topic_idx", 0)),
        end_topic_idx=_optional_int(run_cfg.get("end_topic_idx")),
        max_background_tokens=_optional_int(run_cfg.get("max_background_tokens")),
        min_idea_chars=int(run_cfg.get("min_idea_chars", 300)),
        signature_char_cap=int(run_cfg.get("signature_char_cap", 256)),
    )

    output_dir = Path(data["output_dir"])
    trace_logger = ExperimentLogger(output_dir / "experiment_trace.jsonl")
    generator = TracedClient(raw_generator, logger=trace_logger, role="ideator")
    steno = TracedClient(raw_steno, logger=trace_logger, role="steno")
    judge = (
        TracedClient(raw_judge, logger=trace_logger, role="selector")
        if raw_judge is not None
        else None
    )
    response_logger = AsyncResponseLogger(root_path=output_dir / "responses")
    generator_calls = config.rounds * config.ideas_per_round
    selection_calls = ((config.rounds - 1) if config.seed_directions else 0) + 1
    print("NOVA-style closed-book baseline models:")
    if judge is None:
        print(
            f"  generator: {generator_model} ({generator_calls} generation + "
            f"{selection_calls} self-reflection selection calls per topic)"
        )
        print("  selector:  generator model (NOVA-faithful self-reflection)")
    else:
        print(f"  generator: {generator_model} ({generator_calls} generation calls per topic)")
        print(f"  selector:  {judge_model} (external judge, {selection_calls} selection calls)")
    print(f"  steno:     {steno_model} (compact memory only)")
    print("  critic:    none")
    print("  evaluator: none during generation (selection picks ids only)")
    if config.seed_directions:
        print(
            f"  loop:      {config.rounds} rounds x {config.ideas_per_round} ideas, "
            f"{config.seeds_per_round} seeds/round, final_k={config.final_k}"
        )
    else:
        print(
            f"  loop:      {generator_calls} generations with prompts byte-identical to the "
            f"sequential-memory baseline (seed steering off), final_k={config.final_k}"
        )

    with response_logger:
        records = run_nova_closed_book(
            input_jsonl=data["input_jsonl"],
            output_dir=output_dir,
            generator=generator,
            gen_kwargs=_sampling(generator_cfg),
            config=config,
            steno_client=steno,
            steno_gen_kwargs=_sampling(steno_cfg),
            selection_client=judge,
            selection_gen_kwargs=_sampling(judge_cfg) if judge is not None else None,
            response_logger=response_logger,
            experiment_logger=trace_logger,
        )
    print(
        f"\n[nova-closed-book baseline complete] topic records: {len(records)}"
        f"\noutput_dir: {output_dir}"
    )


if __name__ == "__main__":
    main()
