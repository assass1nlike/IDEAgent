# NOVA-style closed-book baseline

A budget-matched instantiation of NOVA's iterative expand-and-select strategy
(Hu et al., ACL Findings 2025, arXiv:2410.14255) inside our closed background-bank task, so it
is directly comparable to IDEAgent and the sequential-memory baseline.

## Mapping to NOVA

| NOVA (as published)                                   | This baseline (closed book)                                  |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| Input: one seed paper + arXiv retrieval               | Input: the topic's background bank only (retrieval removed)  |
| Iterations of candidate generation from seed ideas    | `rounds` x `ideas_per_round` generator calls (default 3x10 = 30, matching IDEAgent's cap B(1+K_aux) = 30) |
| Self-reflection cuts 10 candidates to 3 seed ideas    | One selection call per non-final round picks `seeds_per_round` ids from that round's compact signatures |
| Old seed ideas replaced each iteration                | Seed pool replaced, never extended                           |
| Final k-means over pool, cluster centers kept         | One diversity-first LLM selection call picks `final_k` ids (embedding-free, like the rest of the codebase) |
| Same LLM ideates and self-reflects                    | Self-reflection mode; the shipped config instead routes selection to an external judge (gemini-3.1-pro) |

Selection calls only ever pick ids from compact signatures — they never critique, repair, or
refine idea text, so no content feedback reaches the generator (unlike IDEAgent's
repair/refinement, this remains a no-feedback baseline in the same sense as sequential memory).
There is deliberately NO improvement loop: ideas evolve only implicitly, by expansion from
selected seeds across rounds, never by editing an existing idea.

## Prompt structure (reused from sequential memory)

Generation prompts are built from the sequential-memory baseline's own pieces
(the verbatim `IDEATOR_SYSTEM_PROMPT`, `op_free_baseline`,
`build_ideator_opening_user_message`):

- **Round 1:** byte-identical to the sequential-memory baseline.
- **Rounds 2+:** identical except the seed-direction block prepended to the directive slot —
  nothing else changes.

Setting `seed_directions: false` is the ablation that drops the seed block (and the per-round
reflections), collapsing the run to flat sequential-memory generations plus one final selection.

## Selection modes

- **External judge (shipped config):** `judge.model_id: gemini-3.1-pro` executes the seed-pool
  and final selection calls; gpt-5.6-sol only ideates. Same model family as IDEAgent's internal
  judges.
- **Self-reflection (NOVA-faithful):** comment out the `judge:` block and the generator model
  executes the same calls with the IDENTICAL prompt — the two modes differ only in which model
  runs the selection, so their delta cleanly isolates "who selects".

`selection_mode` / `selector_model` are recorded in the manifest, per-topic records, and every
selection row.

## Shared with the sequential-memory baseline / QD

- Exact ideator system prompt (verbatim `IDEATOR_SYSTEM_PROMPT`) and free-generation
  operator (`op_free_baseline`), including the neutral no-duplicate signature memory.
- Background builder, steno signature extractor, resume logic, manifest/trace/response logging.
- Output format: `final_idea_cores*.jsonl` with `items[].idea_text` as the evaluation field,
  so all downstream evaluation scripts work unchanged.

## Run

```bash
python baselines/nova/run_nova_closed_book.py --config baselines/nova/nova_closed_book.yaml
```

## Per-topic call accounting (shipped config)

- 30 generation calls (gpt-5.6-sol, same sampling as QD's ideator)
- 3 selection calls (2 reflections + 1 final; gemini-3.1-pro external judge)
- 30 steno calls (gemini-2.5-flash, memory only)
