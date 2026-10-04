"""Small nonnegative least-squares fit with shape-separated held-out validation."""

from __future__ import annotations

import itertools
import math
import random
import statistics


def _solve(matrix, target):
    rows = [list(row) + [value] for row, value in zip(matrix, target, strict=True)]
    n = len(rows)
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-10:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [x / scale for x in rows[column]]
        for r in range(n):
            if r != column:
                scale = rows[r][column]
                rows[r] = [x - scale * y for x, y in zip(rows[r], rows[column], strict=True)]
    return [row[-1] for row in rows]


def fit_nonnegative(shapes):
    """Enumerate the seven active faces of this three-parameter NNLS problem."""
    features = [[batch * width**2, batch * width, 1.0] for batch, width, _ in shapes]
    target = [ms for _, _, ms in shapes]
    scales = [max(row[i] for row in features) for i in range(3)]
    x = [[v / s for v, s in zip(row, scales, strict=True)] for row in features]
    gram = [[sum(row[i] * row[j] for row in x) for j in range(3)] for i in range(3)]
    rhs = [sum(row[i] * y for row, y in zip(x, target, strict=True)) for i in range(3)]
    if _solve(gram, rhs) is None:
        raise ValueError("calibration shapes do not identify quadratic, linear, and overhead terms")
    candidates = []
    for count in (1, 2, 3):
        for active in itertools.combinations(range(3), count):
            solution = _solve([[gram[i][j] for j in active] for i in active], [rhs[i] for i in active])
            if solution is None or any(v < -1e-9 for v in solution):
                continue
            coefficients = [0.0] * 3
            for i, value in zip(active, solution, strict=True):
                coefficients[i] = max(0.0, value) / scales[i]
            error = sum((sum(a * b for a, b in zip(row, coefficients, strict=True)) - y) ** 2 for row, y in zip(features, target, strict=True))
            candidates.append((error, coefficients))
    if not candidates:
        raise ValueError("nonnegative fit failed")
    return min(candidates, key=lambda item: item[0])[1]


def fit_rank(rows, seed=17):
    replicates = {}
    for row in rows:
        batch, width, ms = row["batch"], row["width"], row["ms"]
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (batch, width)):
            raise ValueError("batch and width must be positive integers")
        if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
            raise ValueError("timings must be positive finite milliseconds")
        replicates.setdefault((batch, width), []).append(ms)
    shapes = [(b, w, statistics.median(values)) for (b, w), values in sorted(replicates.items())]
    if len(shapes) < 8:
        raise ValueError("require at least eight distinct calibration shapes")
    random.Random(seed).shuffle(shapes)
    heldout_count = max(2, len(shapes) // 4)
    train, heldout = shapes[heldout_count:], shapes[:heldout_count]
    coefficients = fit_nonnegative(train)
    if coefficients[0] + coefficients[1] <= 0:
        raise ValueError("fit has no positive workload-dependent term")
    predictions = [coefficients[0] * b * w**2 + coefficients[1] * b * w + coefficients[2] for b, w, _ in heldout]
    errors = [abs(p / actual - 1) for p, (_, _, actual) in zip(predictions, heldout, strict=True)]
    return {
        "model": {"quadratic": coefficients[0], "linear": coefficients[1], "overhead": coefficients[2], "max_samples": max(b for b, _, _ in shapes)},
        "domain": {"min_batch": min(b for b, _, _ in shapes), "max_batch": max(b for b, _, _ in shapes), "min_width": min(w for _, w, _ in shapes), "max_width": max(w for _, w, _ in shapes)},
        "fit": {"method": "median_replicates_nnls", "split_seed": seed, "train_shapes": [list(s[:2]) for s in train], "heldout_shapes": [list(s[:2]) for s in heldout], "heldout_median_relative_error": statistics.median(errors), "heldout_max_relative_error": max(errors), "heldout_rmse_ms": math.sqrt(statistics.mean((p - actual) ** 2 for p, (_, _, actual) in zip(predictions, heldout, strict=True)))},
    }
