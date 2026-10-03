import pytest

from tailtrace.cost import RankCost


@pytest.mark.parametrize("field,value", [("quadratic", -1), ("linear", float("nan")), ("overhead", True), ("max_samples", None), ("max_padded_tokens", 0)])
def test_invalid_model(field, value):
    with pytest.raises(ValueError):
        RankCost(**{field: value})


def test_monotone_compute_and_independent_caps():
    model = RankCost(2, 3, 5, max_samples=4, max_padded_tokens=12, max_attention_cells=40)
    assert model.predict(2, 3) == 59
    assert model.predict(0, 0) == 0
    assert model.predict(3, 3) > model.predict(2, 3)
    assert model.permits(3, 3)
    assert not model.permits(4, 4)
    assert not model.permits(5, 1)
