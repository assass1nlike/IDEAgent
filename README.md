# IDEAgent

An agentic quality-diversity (QD) search system for generating research ideas,
plus baseline generation methods used for comparison.

## What Is Here

```text
.
├── ideagent/          # runtime package: IDEAgent's search/repair/refinement
│                      # loop, quality + diversity evaluators, shared infra
├── baselines/
│   ├── simple/        # stateless-independent, single-shot batch, and
│   │                  # sequential-memory baselines
│   └── nova/          # NOVA-style closed-book baseline
├── scripts/           # runners and evaluators (run_yield_archive.py,
│                      # eval_ideas.py, select_lineage_representatives.py)
├── configs/           # canonical YAML configs for each pipeline
└── data/              # topic input data
```

## Setup

```bash
pip install -e .
```

Set the relevant API keys as environment variables or in a `.env` file
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` — see `ideagent/clients.py`).

## Run IDEAgent (main system)

```bash
python scripts/run_yield_archive.py --config configs/ideagent.yaml
```

## Run a baseline

```bash
# Stateless-independent or single-shot batch
python baselines/simple/run_simple_baseline.py --config baselines/simple/stateless.yaml
python baselines/simple/run_simple_baseline.py --config baselines/simple/single_shot.yaml

# Sequential-memory
python baselines/simple/sequential_memory.py --config baselines/simple/sequential_memory.yaml

# NOVA closed-book
python baselines/nova/run_nova_closed_book.py --config baselines/nova/nova_closed_book.yaml
```

## Evaluate

Runs both quality and diversity eval in one command:

```bash
python scripts/eval_ideas.py \
    --config configs/comp_quality_eval_opus47.yaml configs/diversity_eval_opus47.yaml \
    --input <final_idea_cores.jsonl> \
    --quality-output <quality_report.jsonl> \
    --diversity-output <diversity_report.jsonl> \
    --resume
```

Pass `--skip-quality` or `--skip-diversity` to run only one half. `--start-idx`/
`--end-idx` slice the topic list for sharding a run. `--max-workers` overrides
both evaluators' concurrency.
