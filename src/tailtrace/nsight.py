"""Read-only adapter for Nsight Systems CUPTI/NVTX SQLite exports.

Schema/serialized ID reference: NVIDIA Nsight Systems Post-Collection Analysis Guide.
https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tailtrace.evidence import atomic_json
from tailtrace.traces import STEP


def import_sqlite(source: str, out: str, global_pid: int | None = None):
    path = Path(source).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    events = []
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"CUPTI_ACTIVITY_KIND_KERNEL", "NVTX_EVENTS", "StringIds"}
        if not required <= tables:
            raise ValueError(f"unsupported Nsight export: missing {sorted(required - tables)}")
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        kernels = list(db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"))
        if not kernels or not {"start", "end", "deviceId", "globalPid"} <= set(kernels[0].keys()):
            raise ValueError(
                "kernel table empty or missing required timestamp/device/process columns"
            )
        pids = {k["globalPid"] for k in kernels}
        if global_pid is None:
            if len(pids) != 1:
                raise ValueError(f"multiple processes: select --global-pid from {sorted(pids)}")
            global_pid = next(iter(pids))
        if global_pid not in pids:
            raise ValueError("selected globalPid is absent")
        for table, category in (
            ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
            ("CUPTI_ACTIVITY_KIND_MEMCPY", "gpu_memcpy"),
            ("CUPTI_ACTIVITY_KIND_MEMSET", "gpu_memset"),
        ):
            if table not in tables:
                continue
            for row in db.execute(f'SELECT * FROM "{table}"'):
                if row["globalPid"] != global_pid:
                    continue
                keys = set(row.keys())
                name = category
                if category == "kernel":
                    name_key = "demangledName" if "demangledName" in keys else "shortName"
                    if name_key not in keys:
                        raise ValueError("kernel table lacks a supported name column")
                    name = strings.get(row[name_key], str(row[name_key]))
                events.append(
                    {
                        "ph": "X",
                        "cat": category,
                        "name": name,
                        "ts": row["start"] / 1000,
                        "dur": (row["end"] - row["start"]) / 1000,
                        "pid": global_pid,
                        "tid": row["streamId"] if "streamId" in keys else 0,
                        "args": {"device": row["deviceId"]},
                    }
                )
        for row in db.execute("SELECT * FROM NVTX_EVENTS"):
            keys = set(row.keys())
            if not {"start", "end", "globalTid"} <= keys:
                raise ValueError("NVTX table lacks required range/process columns")
            # Global IDs encode <hardware:8><VM:8><PID:24><TID:24>.
            if (
                row["globalTid"] is None
                or row["globalTid"] >> 24 != global_pid >> 24
                or row["end"] is None
            ):
                continue
            text = (
                row["text"]
                if "text" in keys and row["text"] is not None
                else strings.get(row["textId"] if "textId" in keys else None, "")
            )
            if STEP.match(text):
                events.append(
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": text,
                        "ts": row["start"] / 1000,
                        "dur": (row["end"] - row["start"]) / 1000,
                        "pid": global_pid,
                        "tid": row["globalTid"],
                    }
                )
    if not any(e["cat"] == "user_annotation" for e in events):
        raise ValueError("no TailTrace step NVTX ranges; run with nvtx=true")
    result = {
        "tailtrace_evidence": "observed",
        "displayTimeUnit": "ms",
        "traceEvents": events,
        "tailtrace_import": {
            "source": str(path),
            "global_pid": global_pid,
            "input_timestamps": "ns",
            "output_timestamps": "us",
        },
    }
    atomic_json(out, result)
    return {"events": len(events), "global_pid": global_pid, "out": out}
