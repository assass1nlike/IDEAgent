"""Deterministic graph utilities for diversity and portfolio Yield.

The semantic judgments come from the evaluator, but all aggregation is performed in
code.  In particular, a set is distinct at threshold D only when *every* selected pair
has score >= D.  This is a maximum-clique problem in the compatibility graph.

The experiments use N=10.  The branch-and-bound implementation below is exact for that
regime and deterministic under ties; it deliberately avoids the formerly used greedy
approximation, which can undercount Yield.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


RELATIONSHIP_CLASS_TO_SCORE: dict[str, int] = {
    "semantic_duplicate": 0,
    "cosmetic_rewrite": 1,
    "minor_variant": 2,
    "meaningful_variant": 3,
    "sibling_direction": 4,
    "partially_distinct": 5,
    "distinct_direction": 6,
    "clearly_distinct": 7,
    "highly_distinct": 8,
    "orthogonal": 9,
}

AXIS_RELATION_TO_DISTANCE: dict[str, int] = {
    "same": 0,
    "variant": 1,
    "related": 2,
    "distinct": 3,
}


def pair_score_lookup(pairwise_scores: Iterable[Any]) -> dict[tuple[int, int], int]:
    """Return a symmetric ``(idea_i, idea_j) -> score`` lookup.

    Accepts either report dictionaries or PairwiseScore-like objects so the same code
    can be used during evaluation and during JSONL post-processing.
    """

    result: dict[tuple[int, int], int] = {}
    for pair in pairwise_scores:
        if isinstance(pair, Mapping):
            i = int(pair["idea_i"])
            j = int(pair["idea_j"])
            score = int(pair["score"])
        else:
            i = int(pair.idea_i)
            j = int(pair.idea_j)
            score = int(pair.score)
        result[(i, j)] = score
        result[(j, i)] = score
    return result


def _maximum_clique(
    vertices: Sequence[int],
    compatible: Callable[[int, int], bool],
    *,
    max_size: int | None = None,
    weights: Mapping[int, float] | None = None,
) -> list[int]:
    """Find the exact best mutually compatible subset.

    Objectives are lexicographic: maximum cardinality, then maximum total ``weights``
    when supplied, then the lexicographically smallest idea-index tuple.  ``max_size``
    caps cardinality (for example, the final top-five portfolio).
    """

    ordered = sorted({int(v) for v in vertices})
    if max_size is not None and max_size < 0:
        raise ValueError("max_size must be non-negative or None")
    cap = len(ordered) if max_size is None else min(int(max_size), len(ordered))
    weight = weights or {}
    best: tuple[int, ...] = ()
    best_weight = float("-inf")

    adjacency = {
        v: {u for u in ordered if u != v and compatible(v, u)}
        for v in ordered
    }

    def consider(chosen: tuple[int, ...]) -> None:
        nonlocal best, best_weight
        canonical = tuple(sorted(chosen))
        total = sum(float(weight.get(v, 0.0)) for v in canonical)
        if len(canonical) > len(best):
            best, best_weight = canonical, total
        elif len(canonical) == len(best):
            if total > best_weight + 1e-12:
                best, best_weight = canonical, total
            elif abs(total - best_weight) <= 1e-12 and canonical < best:
                best = canonical

    def expand(chosen: tuple[int, ...], candidates: tuple[int, ...]) -> None:
        consider(chosen)
        if len(chosen) >= cap:
            return
        if len(chosen) + len(candidates) < len(best):
            return

        remaining = list(candidates)
        while remaining:
            if len(chosen) + len(remaining) < len(best):
                break
            v = remaining.pop(0)
            next_candidates = tuple(u for u in remaining if u in adjacency[v])
            expand(chosen + (v,), next_candidates)

    expand((), tuple(ordered))
    return list(best)


def maximum_distinct_subset(
    idea_indices: Sequence[int],
    pair_scores: Mapping[tuple[int, int], int] | Iterable[Any],
    *,
    min_score: int = 6,
    max_size: int | None = None,
    weights: Mapping[int, float] | None = None,
) -> list[int]:
    """Exact largest subset whose every pair has diversity >= ``min_score``."""

    lookup = (
        dict(pair_scores)
        if isinstance(pair_scores, Mapping)
        else pair_score_lookup(pair_scores)
    )
    return _maximum_clique(
        idea_indices,
        lambda i, j: lookup.get((i, j), -1) >= min_score,
        max_size=max_size,
        weights=weights,
    )


def maximum_similarity_cluster(
    idea_indices: Sequence[int],
    pair_scores: Mapping[tuple[int, int], int] | Iterable[Any],
    *,
    max_score: int,
) -> list[int]:
    """Exact largest subset whose every pair has diversity <= ``max_score``.

    Requiring all pairwise relations to satisfy the cutoff avoids transitive chaining,
    where A resembles B and B resembles C even though A and C are unrelated.
    """

    lookup = (
        dict(pair_scores)
        if isinstance(pair_scores, Mapping)
        else pair_score_lookup(pair_scores)
    )
    return _maximum_clique(
        idea_indices,
        lambda i, j: lookup.get((i, j), 10) <= max_score,
    )
