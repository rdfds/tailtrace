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


def clip(intervals: list[Interval], window: Interval) -> list[Interval]:
    a, b = window
    return [(max(a, x), min(b, y)) for x, y in intervals if min(b, y) > max(a, x)]


def intersection(left: list[Interval], right: list[Interval]) -> list[Interval]:
    a, b = union(left), union(right)
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            result.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result
