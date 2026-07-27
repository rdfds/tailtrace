"""Half-open interval algebra; concurrent streams must not be double counted."""

from __future__ import annotations

Interval = tuple[float, float]


def union(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("interval ends before it starts")
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals: list[Interval]) -> float:
    return sum(end - start for start, end in union(intervals))


