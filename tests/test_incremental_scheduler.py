import random

from tailtrace.certificate import certify
from tailtrace.cost import RankCost
from tailtrace.scheduler import improve, improve_reference, objective


def test_incremental_moves_match_full_rescoring_with_caps_and_float_ties():
    rng = random.Random(381)
    for case in range(120):
        world = 2 + case % 3
        batch = 2 + case % 6
        ids = list(range(world * batch))
        lengths = [rng.randrange(2, 65) for _ in ids]
        rng.shuffle(ids)
        groups = [ids[r * batch : (r + 1) * batch] for r in range(world)]
        models = [
            RankCost(
                quadratic=rng.choice([0.1, 0.333333333, 1.0, 3.7]),
                linear=rng.choice([0, 0.2, 11.125]),
                overhead=rng.random(),
                max_samples=batch + case % 3,
                max_attention_cells=(batch + 2) * 64**2 if case % 2 else None,
            )
            for _ in range(world)
        ]
        actual = improve(groups, lengths, models, rounds=4)
        reference = improve_reference(groups, lengths, models, rounds=4)
        assert actual == reference, case
        assert objective(actual, lengths, models) == objective(reference, lengths, models)
        certify(ids, lengths, actual, models)
        assert groups == [ids[r * batch : (r + 1) * batch] for r in range(world)]
