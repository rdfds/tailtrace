import sqlite3

import pytest

from tailtrace.nsight import import_sqlite
from tailtrace.traces import analyze_trace


def make_db(path, extra_process=False):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT)")
        db.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [(1, "gemm"), (2, "ncclAllReduce"), (3, "tailtrace/step/0")],
        )
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER, streamId INTEGER, demangledName INTEGER)"
        )
        pid = 7 << 24
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
            [(0, 6000000, 0, pid, 0, 1), (4000000, 8000000, 0, pid, 1, 2)],
        )
        if extra_process:
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 2, 0, ?, 0, 1)", (8 << 24,)
            )
        db.execute(
            "CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER)"
        )
        db.execute("INSERT INTO NVTX_EVENTS VALUES (0, 10000000, ?, NULL, 3)", (pid + 19,))


def test_nsight_adapter_units_process_and_registered_strings(tmp_path):
    source, target = tmp_path / "source.sqlite", tmp_path / "trace.json"
    make_db(source)
    before = source.read_bytes()
    import_sqlite(str(source), str(target))
    row = analyze_trace(target)["steps"][0]
    assert row["gpu_busy_ms"] == 8
    assert row["exposed_collective_ms"] == 2
    assert source.read_bytes() == before


def test_explicit_process_selection(tmp_path):
    source, target = tmp_path / "source.sqlite", tmp_path / "trace.json"
    make_db(source, extra_process=True)
    with pytest.raises(ValueError, match="multiple processes"):
        import_sqlite(str(source), str(target))
    import_sqlite(str(source), str(target), 7 << 24)
