from eval.metrics import aggregate, first_relevant_rank, hit, reciprocal_rank


def test_rank_and_rr():
    assert first_relevant_rank([5, 20, 21], [20]) == 2
    assert reciprocal_rank([5, 20, 21], [20]) == 0.5
    assert first_relevant_rank([1, 2, 3], [20]) == 0
    assert reciprocal_rank([1, 2, 3], [20]) == 0.0
    assert hit([20], [20]) is True


def test_aggregate():
    rows = [
        {"hit": True, "reciprocal_rank": 1.0},
        {"hit": True, "reciprocal_rank": 0.5},
        {"hit": False, "reciprocal_rank": 0.0},
    ]
    summary = aggregate(rows)
    assert summary["n"] == 3
    assert summary["hit_rate"] == 0.667
    assert summary["mrr"] == 0.5
