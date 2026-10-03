import pytest

from tailtrace.certificate import certify, lower_bound
from tailtrace.cost import RankCost


def test_rejects_duplicate_missing_empty_and_overcapacity():
    models = [RankCost(max_samples=2)] * 2
    for groups in ([[0, 1], [1, 2]], [[0, 1], [2]], [[], [0, 1, 2, 3]], [[0, 1, 2], [3]]):
        with pytest.raises(ValueError):
            certify([0, 1, 2, 3], [2, 3, 4, 5], groups, models)


def test_bound_includes_unavoidable_rank_overheads():
    models = [RankCost(1, 0, 9, 3), RankCost(2, 0, 17, 3)]
    cert = certify([0, 1, 2], [2, 4, 3], [[1], [0, 2]], models)
    assert cert["global_tokens"] == 6
    assert cert["lower_bound"] <= cert["makespan"]
    assert cert["evidence"] == "analytic_proxy"
    assert lower_bound([0], [100], [RankCost(max_attention_cells=4)]) == float("inf")
