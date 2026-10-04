"""Independent assignment checks and admissible analytic lower bounds."""

from __future__ import annotations

import math
from collections import Counter

from tailtrace.cost import group_cost, group_shape


def validate_problem(ids, lengths, models):
    if not models or len(ids) < len(models) or len(set(ids)) != len(ids):
        raise ValueError("require unique samples and at least one sample per rank")
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(lengths) for i in ids):
        raise ValueError("sample index out of range")
    if any(
        isinstance(lengths[i], bool) or not isinstance(lengths[i], int) or lengths[i] < 2
        for i in ids
    ):
        raise ValueError("sequence lengths must be integers >= 2")


def lower_bound(ids, lengths, models):
    """Relax padding, ownership, and capacity coupling, never the objective."""
    singleton = []
    work = []
    for i in ids:
        width = lengths[i] - 1
        eligible = [m for m in models if m.permits(1, width)]
        if not eligible:
            return math.inf
        singleton.append(min(m.predict(1, width) for m in eligible))
        # Every sample's unpadded work is <= its share of padded group work.
        work.append(min(m.quadratic * width**2 + m.linear * width for m in eligible))
    return max(max(singleton), (sum(work) + sum(m.overhead for m in models)) / len(models))


def certify(ids, lengths, groups, models):
    validate_problem(ids, lengths, models)
    if len(groups) != len(models) or Counter(i for g in groups for i in g) != Counter(ids):
        raise ValueError("assignment must preserve the global batch exactly")
    costs = []
    for rank, (group, model) in enumerate(zip(groups, models, strict=True)):
        if not group or not model.permits(*group_shape(group, lengths)):
            raise ValueError(f"rank {rank} violates a capacity or nonempty constraint")
        cost = group_cost(group, lengths, model)
        if not math.isfinite(cost):
            raise ValueError("predicted cost overflow")
        costs.append(cost)
    bound = lower_bound(ids, lengths, models)
    maximum = max(costs)
    return {
        "schema_version": 1,
        "evidence": "analytic_proxy",
        "sample_count": len(ids),
        "global_tokens": sum(lengths[i] - 1 for i in ids),
        "rank_samples": list(map(len, groups)),
        "rank_costs": costs,
        "makespan": maximum,
        "lower_bound": bound,
        "bound_gap": maximum / bound - 1,
        "capacity_feasible": True,
    }
