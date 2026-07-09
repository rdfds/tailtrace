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
