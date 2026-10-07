"""Exact latency quantiles with disk storage and a bounded SQLite page cache."""

import math
import sqlite3
import tempfile
from pathlib import Path


class DiskQuantiles:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tailtrace-quantiles-")
        self.connection = sqlite3.connect(Path(self.directory.name) / "latency.sqlite")
        self.connection.execute("PRAGMA cache_size=-1024")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("CREATE TABLE latency (value REAL NOT NULL)")
        self.count = 0
        self.indexed = False

    def add(self, value):
        if not math.isfinite(value) or value <= 0 or self.indexed:
            raise ValueError("invalid latency or distribution already indexed")
        self.connection.execute("INSERT INTO latency VALUES (?)", (value,))
        self.count += 1

    def percentile(self, q):
        if not self.count or not 0 <= q <= 1:
            raise ValueError("empty distribution or invalid quantile")
        if not self.indexed:
            self.connection.execute("CREATE INDEX sorted_latency ON latency(value)")
            self.indexed = True
        position = (self.count - 1) * q
        lo, hi = math.floor(position), math.ceil(position)

        def at(offset):
            return self.connection.execute(
                "SELECT value FROM latency ORDER BY value LIMIT 1 OFFSET ?", (offset,)
            ).fetchone()[0]

        a, b = at(lo), at(hi)
        return (a + b) / 2 if q == 0.5 else a + (b - a) * (position - lo)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()
        self.directory.cleanup()
