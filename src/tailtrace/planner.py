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


