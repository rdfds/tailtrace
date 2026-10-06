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


def beam_candidates(ids, lengths, models, width):
    """Bounded constructive search; deduplicate cost-equivalent partial rank states."""
    states = [[[] for _ in models]]
    order = sorted(ids, key=lambda i: (lengths[i], i), reverse=True)
    for position, i in enumerate(order):
        remaining = len(order) - position - 1
        candidates = {}
        for groups in states:
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
                key = tuple(group_shape(g, lengths) for g in trial)
                candidates.setdefault(key, trial)
        states = sorted(candidates.values(), key=lambda g: objective(g, lengths, models))[:width]
        if not states:
            return []
    return [g for g in states if feasible(g, lengths, models)]


def schedule(ids, lengths, models, local_rounds=12, beam_width=64, retain_local_incumbent=True):
    validate_problem(ids, lengths, models)
    if len(models) > 64 or len(ids) > 512:
        raise ValueError("fleet heuristic supports <=64 ranks and <=512 samples per batch")
    if (
        isinstance(beam_width, bool)
        or not isinstance(beam_width, int)
        or not 0 <= beam_width <= 256
    ):
        raise ValueError("beam_width must be an integer in [0, 256]")
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
    local_candidates = candidates.copy()
    # Limit branching overhead to small batches; large workloads retain greedy + moves.
    if beam_width and len(ids) <= 32 and len(models) <= 8:
        candidates.extend(beam_candidates(ids, lengths, models, beam_width))
    if not candidates:
        raise ValueError("heuristic found no feasible schedule; infeasibility is not proven")
    # Retain the baseline even when a greedy construction becomes trapped by capacities.
    best = min(candidates, key=lambda g: objective(g, lengths, models))
    best = improve(best, lengths, models, local_rounds)
    # A lower-cost construction can enter a worse local-search basin. Retain the
    # independently refined greedy incumbent, rather than discarding its search path.
    if retain_local_incumbent and local_candidates:
        local_best = improve(
            min(local_candidates, key=lambda g: objective(g, lengths, models)),
            lengths,
            models,
            local_rounds,
        )
        best = min((best, local_best), key=lambda g: objective(g, lengths, models))
    certify(ids, lengths, best, models)
    return tuple(tuple(sorted(g)) for g in best)
