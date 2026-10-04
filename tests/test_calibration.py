import pytest

from tailtrace.calibration import fit_nonnegative, fit_rank


def observations():
    return [{"batch": b, "width": w, "ms": (0.003 * b * w**2 + 0.02 * b * w + 0.7) * noise} for b in (1, 2, 4, 8) for w in (4, 8, 16, 32) for noise in (1, 1, 10)]


def test_recovers_coefficients_with_median_outlier_rejection_and_disjoint_holdout():
    result = fit_rank(observations())
    assert [result["model"][k] for k in ("quadratic", "linear", "overhead")] == pytest.approx([0.003, 0.02, 0.7])
    assert result["fit"]["heldout_max_relative_error"] < 1e-9
    assert not set(map(tuple, result["fit"]["train_shapes"])) & set(map(tuple, result["fit"]["heldout_shapes"]))


def test_nnls_can_put_a_negative_unconstrained_term_on_the_boundary():
    shapes = [(b, w, b * w**2 - 0.1 * b * w + 2) for b in (1, 2, 4) for w in (4, 8, 16)]
    coefficients = fit_nonnegative(shapes)
    assert all(c >= 0 for c in coefficients)
    assert coefficients[1] == 0


def test_rejects_nonidentifiable_and_invalid_observations():
    with pytest.raises(ValueError, match="identify"):
        fit_nonnegative([(b, 4, b + 1) for b in range(1, 10)])
    with pytest.raises(ValueError, match="timings"):
        fit_rank([{"batch": 1, "width": 8, "ms": float("nan")}])
