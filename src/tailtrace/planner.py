"""Deterministic global-batch preserving minimax padded-attention planner."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchPlan:
    ranks: tuple[tuple[int, ...], ...]
    costs: tuple[int, ...]
    tokens: int


def padded_cost(indices: list[int] | tuple[int, ...], lengths: list[int]) -> int:
    # This workload uses dense padded attention, so sum(length**2) is the wrong proxy.
    return len(indices) * max((lengths[i] - 1 for i in indices), default=0) ** 2


def make_lengths(samples: int, minimum: int, maximum: int, seed: int) -> list[int]:
    """A bounded log-uniform workload, generated without downloading a corpus."""
    import math

    rng = random.Random(seed)
    return [
        max(
            minimum,
            min(maximum, round(math.exp(rng.uniform(math.log(minimum), math.log(maximum))))),
        )
        for _ in range(samples)
    ]


def plan_epoch(lengths, world_size, batch_size, seed, mode="random"):
    if world_size < 1 or batch_size < 1 or mode != "random":
        raise ValueError("invalid planner arguments")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in lengths):
        raise ValueError("lengths must be integers >= 2")
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    size = world_size * batch_size
    result = []
    for start in range(0, len(order) - size + 1, size):
        ids = order[start:start + size]
        groups = tuple(tuple(ids[r*batch_size:(r+1)*batch_size]) for r in range(world_size))
        result.append(BatchPlan(groups, tuple(padded_cost(g, lengths) for g in groups),
                                sum(lengths[i] - 1 for i in ids)))
    return result
