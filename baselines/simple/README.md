# Simple baselines: stateless independent, single-shot batch & sequential memory

The three lowest rungs of the baseline ladder. Stateless independent and single-shot batch are
prompt-matched to the sequential-memory baseline (the verbatim `IDEATOR_SYSTEM_PROMPT`, same
`op_free_baseline` lineage, same opening-message builder, same steno for output compatibility).
All three output exactly `n_ideas = 10` ideas per topic in the standard
`final_idea_cores*.jsonl` format.

## Stateless independent (`stateless.yaml`)

10 fully independent generator calls. Every call is **byte-identical to the sequential-memory
baseline's first (empty-memory) call**: the free operator's only diversity pressure is
background-relative ("must not be a direct combination, extension, or ablation of the
background papers") — no cross-idea signal exists by design. Isolates what sampling
stochasticity alone buys; the delta to sequential memory is exactly the signature memory.

## Single-shot batch (`single_shot.yaml`)

ONE generator call returns all 10 ideas, delimited by `=== IDEA k ===` headers (fail-closed
parsing: wrong count or a too-short idea aborts the topic). The directive keeps the free
operator's composition order and reuses `NON_OBVIOUSNESS_PRESSURE` and
`SOUND_BY_CONSTRUCTION` verbatim; the memory-avoid block is replaced by within-response
mutual distinctness ("each idea must attack a DIFFERENT problem or use a DIFFERENT core
mechanism than every other idea in this response") — the only diversity signal a single call
can act on besides the background. `max_new_tokens` is raised to 65536 since one response
carries all 10 complete ideas.

## Sequential memory (`sequential_memory.yaml`)

10 sequential generator calls, no critic or quality feedback. Each call sees a compact
semantic-signature memory of every idea generated so far in that topic (from the same steno
extractor QD uses), so the only diversity pressure beyond the background is "don't repeat a
prior idea's signature" — one step up the ladder from stateless independent, which has no
cross-idea signal at all.

## Run

```bash
python baselines/simple/run_simple_baseline.py --config baselines/simple/stateless.yaml
python baselines/simple/run_simple_baseline.py --config baselines/simple/single_shot.yaml
python baselines/simple/sequential_memory.py --config baselines/simple/sequential_memory.yaml
```

## Ladder position

| Rung | Cross-idea signal during generation |
| --- | --- |
| Single-shot batch | Self-visible siblings within one response |
| Stateless independent | None |
| Sequential memory | Compact signatures of all prior ideas |
| NOVA-style closed book | Signatures + judge-selected seed directions + final selection |
| IDEAgent | Full evaluator feedback, repair, refinement, archive |

Note: `ideagent/single_shot_baseline.py` is an older implementation with its own prompt
scaffold, kept only because `pipeline.py` imports its response-format constant to keep
this baseline's single-shot output schema in sync.
