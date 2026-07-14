import pytest

from tailtrace.planner import padded_cost, plan_epoch


@pytest.mark.parametrize("world,batch", [(1, 3), (2, 2), (4, 4)])
def test_preserves_each_global_batch_and_token_denominator(world, batch):
    lengths = [2 + (i * 71) % 200 for i in range(131)]
    baseline = plan_epoch(lengths, world, batch, 7, "random")
    balanced = plan_epoch(lengths, world, batch, 7, "balanced")
    seen = []
    for a, b in zip(baseline, balanced, strict=True):

        def flatten(p):
            return sorted(i for rank in p.ranks for i in rank)

        assert flatten(a) == flatten(b)
        assert all(1 <= len(rank) <= 2 * batch for rank in b.ranks)
        assert a.tokens == b.tokens
        assert max(b.costs) <= max(a.costs)
        seen.extend(flatten(b))
    assert len(seen) == len(set(seen))
    assert balanced == plan_epoch(lengths, world, batch, 7, "balanced")


def test_padded_attention_not_sum_of_squared_lengths():
    assert padded_cost([0, 1], [3, 9]) == 128


def test_reject_bad_lengths():
    with pytest.raises(ValueError):
        plan_epoch([1, 3], 2, 1, 0)


