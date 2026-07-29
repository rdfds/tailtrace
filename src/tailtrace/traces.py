from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

from tailtrace.intervals import clip, duration, intersection

STEP = re.compile(r"^tailtrace/step/(\d+)$")


def analyze_trace(path: str | Path, rank: int = 0, device_id: int | None = None) -> dict:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        document = json.load(f)
    events = document if isinstance(document, list) else document.get("traceEvents", [])
    complete = []
    for e in events:
        if e.get("ph") != "X":
            continue
        start, dur = float(e.get("ts", 0)), float(e.get("dur", 0))
        if not math.isfinite(start) or not math.isfinite(dur) or dur < 0:
            raise ValueError("trace contains non-finite or negative durations")
        complete.append(e)
    # CUDA runtime API calls are CPU events; they must never count as GPU work.
    gpu = [
        e
        for e in complete
        if str(e.get("cat", "")).lower() in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    devices = {
        e.get("args", {}).get("device") for e in gpu if e.get("args", {}).get("device") is not None
    }
    if len(devices) > 1 and device_id is None:
        raise ValueError("multi-device trace: select --device-id explicitly")
    if device_id is not None:
        gpu = [e for e in gpu if e.get("args", {}).get("device") == device_id]
    compute, comm, copies = [], [], []
    for e in gpu:
        span = (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        if str(e.get("cat", "")).lower() != "kernel":
            copies.append(span)
        elif "nccl" in e.get("name", "").lower():
            comm.append(span)
        else:
            compute.append(span)
    windows = [
        (int(m[1]), (float(e["ts"]), float(e["ts"]) + float(e["dur"])))
        for e in complete
        if (m := STEP.match(e.get("name", "")))
    ]
    if not windows:
        raise ValueError(
            "no tailtrace/step/N ranges; use TailTrace profiling or annotate the trace"
        )
    if len({s for s, _ in windows}) != len(windows):
        raise ValueError("duplicate step ranges: supply one rank per trace")
    rows = []
    for step, window in sorted(windows):
        c, n, h = clip(compute, window), clip(comm, window), clip(copies, window)
        overlap = duration(intersection(c, n))
        rows.append(
            {
                "rank": rank,
                "step": step,
                "window_ms": (window[1] - window[0]) / 1000,
                "compute_ms": duration(c) / 1000,
                "collective_ms": duration(n) / 1000,
                "overlap_ms": overlap / 1000,
                "exposed_collective_ms": (duration(n) - overlap) / 1000,
                "transfer_ms": duration(h) / 1000,
                "gpu_busy_ms": duration(c + n + h) / 1000,
                "uncovered_ms": max(0, window[1] - window[0] - duration(c + n + h)) / 1000,
            }
        )
    return {
        "schema_version": 1,
        "source": str(path),
        "rank": rank,
        "evidence": document.get("tailtrace_evidence", "observed")
        if isinstance(document, dict)
        else "observed",
        "gpu_events": len(gpu),
        "steps": rows,
        "limitations": [
            "NCCL kernel occupancy includes waiting and does not isolate network time.",
            "All intervals use a single rank-local clock; no cross-host arrival comparison.",
            "Uncovered time is not a hardware utilization measurement.",
            "Name-based NCCL classification; non-NCCL collectives are not classified.",
            "Async GPU work outside host step ranges is excluded; use diagnostic_sync for attribution.",
        ],
    }
