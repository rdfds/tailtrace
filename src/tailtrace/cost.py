"""Monotone rank-local compute estimates; resource caps are declared assumptions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RankCost:
    quadratic: float = 1.0
    linear: float = 0.0
    overhead: float = 0.0
    max_samples: int = 8
    max_padded_tokens: int | None = None
    max_attention_cells: int | None = None

    def __post_init__(self):
        for name in ("quadratic", "linear", "overhead"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.quadratic + self.linear <= 0:
            raise ValueError("compute coefficients cannot both be zero")
        for name in ("max_samples", "max_padded_tokens", "max_attention_cells"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.max_samples is None:
            raise ValueError("max_samples is required")

    def predict(self, count: int, width: int) -> float:
        if count == 0:
            return 0.0
        return self.quadratic * count * width**2 + self.linear * count * width + self.overhead

    def permits(self, count: int, width: int) -> bool:
        return (
            count <= self.max_samples
            and (self.max_padded_tokens is None or count * width <= self.max_padded_tokens)
            and (self.max_attention_cells is None or count * width**2 <= self.max_attention_cells)
        )

    def to_dict(self):
        return asdict(self)


def group_shape(group, lengths):
    return len(group), max((lengths[i] - 1 for i in group), default=0)


def group_cost(group, lengths, model):
    return model.predict(*group_shape(group, lengths))
