"""Run the candidate-matched compact-memory baseline with no critic or quality feedback.

    python baselines/simple/sequential_memory.py \
        --config baselines/simple/sequential_memory.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ideagent.clients import build_client
from ideagent.experiment_logging import ExperimentLogger, TracedClient
from ideagent.response_logging import AsyncResponseLogger
from ideagent.sequential_memory_baseline import (
    SequentialMemoryBaselineConfig,
    run_sequential_memory_baseline,
)
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
        description="Generate N ideas sequentially with compact prior-signature memory."
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT_DIR / "baselines" / "simple" / "sequential_memory.yaml",
    )
    parser.add_argument("--n", type=int, default=None, help="Override sequential_memory.n_ideas.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg.get("data", {})
    run_cfg = cfg.get("sequential_memory", {})
    client_cfg = cfg.get("client", {})
    generator_cfg = cfg.get("generator", {})
    steno_cfg = cfg.get("steno", {})

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

    config = SequentialMemoryBaselineConfig(
        n_ideas=int(args.n if args.n is not None else run_cfg.get("n_ideas", 10)),
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
    response_logger = AsyncResponseLogger(root_path=output_dir / "responses")
    print("Sequential compact-memory baseline models:")
    print(f"  generator: {generator_model} ({config.n_ideas} calls per topic)")
    print(f"  steno:     {steno_model} (compact memory only)")
    print("  critic:    none")
    print("  evaluator: none during generation")

    with response_logger:
        records = run_sequential_memory_baseline(
            input_jsonl=data["input_jsonl"],
            output_dir=output_dir,
            generator=generator,
            gen_kwargs=_sampling(generator_cfg),
            config=config,
            steno_client=steno,
            steno_gen_kwargs=_sampling(steno_cfg),
            response_logger=response_logger,
            experiment_logger=trace_logger,
        )
    print(
        f"\n[sequential-memory baseline complete] topic records: {len(records)}"
        f"\noutput_dir: {output_dir}"
    )


if __name__ == "__main__":
    main()
