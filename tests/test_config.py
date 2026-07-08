import pytest

from tailtrace.config import TrainConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 65},
        {"warmup": 20},
        {"strategy": "fsdp2"},
        {"kernel": "cuda"},
        {"batch_size": -1},
        {"heads": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
