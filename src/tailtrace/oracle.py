"""Budgeted exact labeled-rank search for small global batches."""

from __future__ import annotations

from tailtrace.certificate import certify, lower_bound, validate_problem
from tailtrace.cost import group_cost, group_shape


def solve_exact(ids, lengths, models, node_budget=100_000, incumbent=None):
    validate_problem(ids, lengths, models)
    if isinstance(node_budget, bool) or not isinstance(node_budget, int) or node_budget < 1:
        raise ValueError("node_budget must be a positive integer")
    if len(ids) > 18 or len(models) > 4:
        raise ValueError("oracle is restricted to <=18 samples and <=4 ranks")
    best = None
    objective = (float("inf"), float("inf"))
    if incumbent is not None:
        certificate = certify(ids, lengths, incumbent, models)
        best = [list(g) for g in incumbent]
        objective = (certificate["makespan"], sum(certificate["rank_costs"]))
    order = sorted(ids, key=lambda i: (lengths[i], i), reverse=True)
    groups = [[] for _ in models]
    nodes = 0
    exhausted = False

    def visit(position):
        nonlocal nodes, exhausted, best, objective
        if nodes >= node_budget:
            exhausted = True
            return
        nodes += 1
        if sum(not g for g in groups) > len(order) - position:
            return
        if sum(m.max_samples - len(g) for m, g in zip(models, groups, strict=True)) < len(order) - position:
            return
        costs = [group_cost(g, lengths, m) for g, m in zip(groups, models, strict=True)]
        if (max(costs), sum(costs)) >= objective:
            return
        if position == len(order):
            best = [g.copy() for g in groups]
            objective = (max(costs), sum(costs))
            return
        i = order[position]
        eligible = [r for r, m in enumerate(models) if m.permits(*group_shape([*groups[r], i], lengths))]
        eligible.sort(key=lambda r: (group_cost([*groups[r], i], lengths, models[r]), r))
        for rank in eligible:
            groups[rank].append(i)
            visit(position + 1)
            groups[rank].pop()
            if exhausted:
                return

    visit(0)
    bound = lower_bound(ids, lengths, models)
    return {
        "schema_version": 1,
        "evidence": "analytic_proxy",
        "status": "budget_exhausted" if exhausted else ("optimal" if best else "infeasible"),
        "nodes": nodes,
        "node_budget": node_budget,
        "groups": best,
        "upper_bound": objective[0] if best else None,
        "lower_bound": objective[0] if best and not exhausted else (bound if bound != float("inf") else None),
        "certificate": certify(ids, lengths, best, models) if best else None,
    }
