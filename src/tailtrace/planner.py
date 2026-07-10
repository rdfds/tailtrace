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


def plan_epoch(
    lengths: list[int], world_size: int, batch_size: int, seed: int, mode: str = "balanced"
) -> list[BatchPlan]:
    if world_size < 1 or batch_size < 1 or mode not in {"random", "balanced"}:
        raise ValueError("invalid planner arguments")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in lengths):
        raise ValueError("lengths must be integers >= 2")
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    global_size = world_size * batch_size
    plans = []
    for step, start in enumerate(range(0, len(order) - global_size + 1, global_size)):
        ids = order[start : start + global_size]
        if mode == "random":
            groups = [ids[r * batch_size : (r + 1) * batch_size] for r in range(world_size)]
        else:
            sorted_ids = sorted(ids, key=lambda i: (lengths[i], i), reverse=True)
            candidates = [
                [ids[r * batch_size : (r + 1) * batch_size] for r in range(world_size)],
                [sorted_ids[r * batch_size : (r + 1) * batch_size] for r in range(world_size)],
            ]
            # Variable local cardinality is essential: with fixed cardinality, the rank
            # containing the longest sample always has the same padded-attention cost.
            greedy = [[i] for i in sorted_ids[:world_size]]
            for i in sorted_ids[world_size:]:
                costs = [padded_cost(g, lengths) for g in greedy]
                eligible = [r for r, g in enumerate(greedy) if len(g) < 2 * batch_size]

                def score(r, greedy=greedy, i=i, costs=costs):
                    new = padded_cost(greedy[r] + [i], lengths)
                    other = [costs[s] for s in range(world_size) if s != r]
                    return max([new, *other]), new, r

                chosen = min(eligible, key=score)
                greedy[chosen].append(i)
            candidates.append(greedy)

            def objective(candidate):
                costs = [padded_cost(g, lengths) for g in candidate]
                return max(costs), sum(costs)

            # Including the random baseline guarantees no regression in the proxy tail.
            groups = min(candidates, key=objective)
            # Rotate ownership instead of permanently assigning long work to rank zero.
            offset = step % world_size
            groups = groups[offset:] + groups[:offset]
        costs = tuple(padded_cost(g, lengths) for g in groups)
        plans.append(
            BatchPlan(tuple(tuple(g) for g in groups), costs, sum(lengths[i] - 1 for i in ids))
        )
    return plans


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
