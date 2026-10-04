"""Fleet-aware greedy construction and deterministic local improvement."""

from __future__ import annotations

from tailtrace.certificate import certify, validate_problem
from tailtrace.cost import group_cost, group_shape


def objective(groups, lengths, models):
    costs = [group_cost(g, lengths, m) for g, m in zip(groups, models, strict=True)]
    return max(costs), sum(costs)


def feasible(groups, lengths, models):
    return all(
        g and m.permits(*group_shape(g, lengths)) for g, m in zip(groups, models, strict=True)
    )


def improve(groups, lengths, models, rounds=12):
    """Accept only strict lexicographic improvements; fixed ranks retain their models."""
    groups = [list(g) for g in groups]
    current = objective(groups, lengths, models)
    for _ in range(rounds):
        winner = None
        winner_score = current
        for source, group in enumerate(groups):
            for i in sorted(group):
                for target in range(len(models)):
                    if source == target:
                        continue
                    # Relocations alter cardinality; swaps can escape capacity-bound layouts.
                    for j in [None, *sorted(groups[target])]:
                        candidate = [g.copy() for g in groups]
                        candidate[source].remove(i)
                        candidate[target].append(i)
                        if j is not None:
                            candidate[target].remove(j)
                            candidate[source].append(j)
                        if not feasible(candidate, lengths, models):
                            continue
                        score = objective(candidate, lengths, models)
                        if score < winner_score:
                            winner, winner_score = candidate, score
        if winner is None:
            break
        groups, current = winner, winner_score
    return groups


def schedule(ids, lengths, models, local_rounds=12):
    validate_problem(ids, lengths, models)
    if len(models) > 64 or len(ids) > 512:
        raise ValueError("fleet heuristic supports <=64 ranks and <=512 samples per batch")
    candidates = []
    if len(ids) % len(models) == 0:
        batch = len(ids) // len(models)
        baseline = [list(ids[r * batch : (r + 1) * batch]) for r in range(len(models))]
        if feasible(baseline, lengths, models):
            candidates.append(baseline)
    # Different ownership priorities help when only some ranks can host long samples.
    priorities = [
        list(range(len(models))),
        sorted(
            range(len(models)), key=lambda r: models[r].predict(1, max(lengths[i] - 1 for i in ids))
        ),
    ]
    for priority in priorities:
        groups = [[] for _ in models]
        for position, i in enumerate(sorted(ids, key=lambda i: (lengths[i], i), reverse=True)):
            remaining = len(ids) - position - 1
            eligible = []
            for rank, model in enumerate(models):
                trial = [g.copy() for g in groups]
                trial[rank].append(i)
                if not model.permits(*group_shape(trial[rank], lengths)):
                    continue
                if sum(not g for g in trial) > remaining:
                    continue
                if (
                    sum(m.max_samples - len(g) for m, g in zip(models, trial, strict=True))
                    < remaining
                ):
                    continue
                eligible.append((objective(trial, lengths, models), priority.index(rank), rank))
            if not eligible:
                break
            groups[min(eligible)[2]].append(i)
        else:
            if feasible(groups, lengths, models):
                candidates.append(groups)
    if not candidates:
        raise ValueError("heuristic found no feasible schedule; infeasibility is not proven")
    # Retain the baseline even when a greedy construction becomes trapped by capacities.
    best = min(candidates, key=lambda g: objective(g, lengths, models))
    best = improve(best, lengths, models, local_rounds)
    certify(ids, lengths, best, models)
    return tuple(tuple(sorted(g)) for g in best)
