"""Retrieval metrics.

hit_rate  : did any expected page appear anywhere in the retrieved chunks?
            Measures whether the right content reached the model at all.
mrr       : 1 / rank of the first correct chunk, averaged.
            Measures whether the right content reached the model EARLY.
            This matters because context windows are ordered and budgeted:
            a correct chunk at rank 8 of 10 is worth less than one at rank 1.
"""

from typing import Dict, List, Sequence


def first_relevant_rank(pages_in_order: Sequence[int], expected: Sequence[int]) -> int:
    """1-based rank of the first retrieved chunk on an expected page, else 0."""
    expected_set = set(expected)
    for rank, page in enumerate(pages_in_order, start=1):
        if page in expected_set:
            return rank
    return 0


def hit(pages_in_order: Sequence[int], expected: Sequence[int]) -> bool:
    return first_relevant_rank(pages_in_order, expected) > 0


def reciprocal_rank(pages_in_order: Sequence[int], expected: Sequence[int]) -> float:
    rank = first_relevant_rank(pages_in_order, expected)
    return 1.0 / rank if rank else 0.0


def aggregate(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {"hit_rate": 0.0, "mrr": 0.0, "n": 0}

    return {
        "n": len(rows),
        "hit_rate": round(sum(1 for r in rows if r["hit"]) / len(rows), 3),
        "mrr": round(sum(r["reciprocal_rank"] for r in rows) / len(rows), 3),
    }
