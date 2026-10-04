import itertools
import random

import pytest

from tailtrace.certificate import certify, lower_bound
from tailtrace.cost import RankCost
from tailtrace.oracle import solve_exact


def brute_force(lengths, models):
    best = (float("inf"), float("inf"))
    for assignment in itertools.product(range(len(models)), repeat=len(lengths)):
        groups = [[i for i, r in enumerate(assignment) if r == rank] for rank in range(len(models))]
        try:
            cert = certify(list(range(len(lengths))), lengths, groups, models)
        except ValueError:
            continue
        best = min(best, (cert["makespan"], sum(cert["rank_costs"])))
    return best


@pytest.mark.parametrize("seed", range(10))
def test_matches_independent_enumeration_with_heterogeneity_and_caps(seed):
    rng = random.Random(seed)
    lengths = [rng.randint(2, 12) for _ in range(6)]
    models = [RankCost(1, 2, 3, 4, 32), RankCost(2, 1, 7, 5, 40)]
    brute = brute_force(lengths, models)
    result = solve_exact(list(range(6)), lengths, models)
    if brute[0] == float("inf"):
        assert result["status"] == "infeasible"
    else:
        assert result["status"] == "optimal"
        assert (result["upper_bound"], sum(result["certificate"]["rank_costs"])) == brute
        assert lower_bound(list(range(6)), lengths, models) <= brute[0]
        assert result["lower_bound"] == result["upper_bound"]


def test_truncated_search_does_not_claim_optimal_or_infeasible():
    models = [RankCost(max_samples=4)] * 2
    result = solve_exact(list(range(4)), [3, 4, 5, 6], models, node_budget=1)
    assert result["status"] == "budget_exhausted"
    assert result["groups"] is None
    warm = solve_exact(
        list(range(4)), [3, 4, 5, 6], models, node_budget=1, incumbent=[[0, 1], [2, 3]]
    )
    assert warm["groups"] == [[0, 1], [2, 3]]
    assert warm["lower_bound"] <= warm["upper_bound"]
