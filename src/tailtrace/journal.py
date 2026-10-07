"""Durable rank-local metrics with an explicit committed byte frontier."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from tailtrace.evidence import atomic_json


class RankJournal:
    def __init__(self, root, rank, start_step):
        self.root, self.rank, self.start_step = Path(root), rank, start_step
        self.stream = (self.root / f"metrics-rank{rank}.jsonl").open("xb")
        self.digest = hashlib.sha256()
        self.rows = self.bytes = self.flushes = 0
        self.closed = False

    def write(self, rows, complete=False):
        if self.closed:
            raise ValueError("journal is closed")
        encoded = []
        for offset, row in enumerate(rows):
            if row["rank"] != self.rank or row["step"] != self.start_step + self.rows + offset:
                raise ValueError("journal rank/step sequence differs")
            if not math.isfinite(row["step_ms"]) or row["step_ms"] <= 0:
                raise ValueError("journal step timing must be finite and positive")
            data = (json.dumps(row, allow_nan=False) + "\n").encode()
            if len(data) > 65536:
                raise ValueError("metric row exceeds 64 KiB")
            encoded.append(data)
        for data in encoded:
            self.stream.write(data)
            self.digest.update(data)
            self.bytes += len(data)
        self.rows += len(rows)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.flushes += 1
        atomic_json(self.root / f"progress-rank{self.rank}.json", self.snapshot(complete))

    def snapshot(self, complete=False):
        return {
            "schema_version": 1,
            "rank": self.rank,
            "start_step": self.start_step,
            "last_step": self.start_step + self.rows - 1,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.digest.hexdigest(),
            "flushes": self.flushes,
            "complete": complete,
        }

    def close(self):
        self.stream.close()
        self.closed = True


def iter_committed_rows(root, rank):
    """Validate a sealed prefix without retaining its rows in memory."""
    root = Path(root)
    progress = json.loads((root / f"progress-rank{rank}.json").read_text())
    if progress["rank"] != rank or progress["schema_version"] != 1:
        raise ValueError("progress rank identity or schema differs")
    for key in ("bytes", "rows", "start_step"):
        if type(progress[key]) is not int or progress[key] < 0:
            raise ValueError("invalid progress accounting")
    path = root / f"metrics-rank{rank}.jsonl"
    digest, remaining = hashlib.sha256(), progress["bytes"]
    with path.open("rb") as stream:
        while remaining:
            data = stream.read(min(65536, remaining))
            if not data:
                raise ValueError("committed metric prefix changed or truncated")
            digest.update(data)
            remaining -= len(data)
    if digest.hexdigest() != progress["sha256"]:
        raise ValueError("committed metric prefix changed or truncated")
    if progress["last_step"] != progress["start_step"] + progress["rows"] - 1:
        raise ValueError("progress row/step accounting differs")

    def iterator():
        count, remaining = 0, progress["bytes"]
        with path.open("rb") as stream:
            while remaining:
                data = stream.readline(min(65537, remaining + 1))
                if not data.endswith(b"\n") or len(data) > min(65536, remaining):
                    raise ValueError("committed metric row is truncated or oversized")
                row = json.loads(data)
                if row["rank"] != rank or row["step"] != progress["start_step"] + count:
                    raise ValueError("committed metric rank/step sequence differs")
                count += 1
                remaining -= len(data)
                yield row
        if count != progress["rows"]:
            raise ValueError("progress row/step accounting differs")

    return iterator(), progress


def committed_rows(root, rank):
    rows, progress = iter_committed_rows(root, rank)
    return list(rows), progress


def inspect_run(root):
    """Report durable coverage; a partial job is never a throughput result."""
    root = Path(root)
    completed = (root / "manifest.json").is_file()
    metadata = json.loads((root / ("manifest.json" if completed else "attempt.json")).read_text())
    ranks = []
    for rank in range(metadata["world_size"]):
        progress = root / f"progress-rank{rank}.json"
        failure = root / f"failure-rank{rank}.json"
        entry = {"rank": rank, "status": "no_committed_metrics", "rows": 0, "last_step": None}
        if progress.exists():
            try:
                rows, marker = iter_committed_rows(root, rank)
                if completed and (
                    marker != metadata["telemetry"]["rank_progress"][rank]
                    or not marker["complete"]
                    or (root / f"metrics-rank{rank}.jsonl").stat().st_size != marker["bytes"]
                ):
                    raise ValueError("completed run journal differs from manifest")
                entry.update(
                    status="complete" if marker["complete"] else "partial",
                    rows=sum(1 for _ in rows),
                    last_step=marker["last_step"],
                    start_step=marker["start_step"],
                    bytes=marker["bytes"],
                )
            except (ValueError, KeyError, OSError) as exc:
                entry.update(status="invalid", error=str(exc))
        if failure.exists():
            entry["failure"] = json.loads(failure.read_text())
        ranks.append(entry)
    frontiers = [r["last_step"] for r in ranks]
    durable = min(frontiers) if all(x is not None for x in frontiers) else None
    return {
        "schema_version": 1,
        "evidence": "observed",
        "status": "invalid"
        if any(r["status"] == "invalid" for r in ranks)
        or (completed and any(r["status"] != "complete" for r in ranks))
        else "complete"
        if completed
        else "partial",
        "world_size": metadata["world_size"],
        "durable_through_step": durable,
        "ranks": ranks,
        "limitations": [
            "Durable coverage is not proof of a committed optimizer state.",
            "Partial metrics are diagnostics, not a complete throughput benchmark.",
            "Bytes beyond a progress marker are uncommitted and ignored.",
        ],
    }
