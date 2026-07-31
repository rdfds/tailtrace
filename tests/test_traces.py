import json

import pytest

from tailtrace.intervals import duration, intersection, union
from tailtrace.traces import analyze_trace


def test_union_and_overlap():
    assert union([(0, 5), (2, 6), (6, 7), (9, 9)]) == [(0, 7)]
    assert duration(intersection([(0, 10)], [(2, 3), (5, 12)])) == 6
    with pytest.raises(ValueError):
        union([(5, 2)])


def event(name, start, dur, cat="kernel", device=0):
    return {
        "ph": "X",
        "name": name,
        "ts": start,
        "dur": dur,
        "cat": cat,
        "args": {"device": device},
    }


