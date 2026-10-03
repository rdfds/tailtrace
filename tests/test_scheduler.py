import pytest

from tailtrace.certificate import certify
from tailtrace.cost import RankCost
from tailtrace.scheduler import schedule


def test_fixed_physical_rank_gets_more_work_when_faster():
    models = [RankCost(1, max_samples=7), RankCost(4, max_samples=7)]
    groups = schedule(list(range(8)), [16] * 8, models)
    assert len(groups[0]) > len(groups[1])
    assert groups == schedule(list(range(8)), [16] * 8, models)
    assert certify(list(range(8)), [16] * 8, groups, models)["global_tokens"] == 120


def test_respects_long_sample_placement_and_padded_caps():
    models = [RankCost(3, max_samples=7, max_padded_tokens=20), RankCost(1, max_samples=7)]
    lengths = [64, 4, 4, 4, 4, 4, 4, 4]
    groups = schedule(list(range(8)), lengths, models)
    assert 0 in groups[1]
    certify(list(range(8)), lengths, groups, models)


def test_baseline_candidate_prevents_proxy_regression():
    for seed in range(20):
        lengths = [2 + (i * 37 + seed * 13) % 79 for i in range(8)]
        models = [RankCost(1, 2, 3, 8), RankCost(2, 1, 9, 8)]
        baseline = certify(list(range(8)), lengths, [list(range(4)), list(range(4, 8))], models)
        candidate = certify(list(range(8)), lengths, schedule(list(range(8)), lengths, models), models)
        assert candidate["makespan"] <= baseline["makespan"]


def test_failure_is_not_an_infeasibility_claim():
    with pytest.raises(ValueError, match="infeasibility is not proven"):
        schedule([0, 1], [100, 100], [RankCost(max_padded_tokens=3)] * 2)
