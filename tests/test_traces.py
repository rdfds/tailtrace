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


def test_stream_overlap_not_double_counted(tmp_path):
    events = [
        event("tailtrace/step/0", 0, 10000, "user_annotation"),
        event("gemm", 0, 6000),
        event("gemm2", 2000, 3000),
        event("ncclAllReduce", 4000, 4000),
        event("cudaLaunchKernel", 0, 10000, "cuda_runtime"),
    ]
    p = tmp_path / "trace.json"
    p.write_text(json.dumps({"traceEvents": events}))
    row = analyze_trace(p)["steps"][0]
    assert row["compute_ms"] == 6
    assert row["collective_ms"] == 4
    assert row["overlap_ms"] == 2
    assert row["exposed_collective_ms"] == 2
    assert row["gpu_busy_ms"] == 8
    assert row["uncovered_ms"] == 2


def test_reject_multiple_devices(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(
        json.dumps(
            {
                "traceEvents": [
                    event("tailtrace/step/0", 0, 10),
                    event("gemm", 0, 5, device=0),
                    event("gemm", 0, 5, device=1),
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="multi-device"):
        analyze_trace(p)
