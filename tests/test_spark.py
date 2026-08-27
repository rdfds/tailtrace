import json
import os

import pytest

pytest.importorskip("pyspark")

from tailtrace.spark import aggregate


@pytest.mark.spark
def test_separate_runs_and_rank_coverage(tmp_path):
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--master local[2] pyspark-shell")
    for name, ms in (("a", 10), ("b", 5)):
        root = tmp_path / name
        root.mkdir()
        for rank in range(2):
            row = {
                "rank": rank,
                "step": 0,
                "warmup": False,
                "local_tokens": 4,
                "global_tokens": 8,
                "padded_tokens": 4,
                "step_ms": float(ms),
                "loss_sum": 8.0,
                "peak_memory_bytes": 0,
            }
            (root / f"metrics-rank{rank}.jsonl").write_text(json.dumps(row) + "\n")
    aggregate(str(tmp_path / "*" / "metrics-rank*.jsonl"), str(tmp_path / "warehouse"))
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[2]").getOrCreate()
    try:
        rows = spark.read.parquet(str(tmp_path / "warehouse")).collect()
        assert len(rows) == 2
        assert {r["tokens_per_second"] for r in rows} == {800, 1600}
        assert {r["observed_ranks"] for r in rows} == {2}
    finally:
        spark.stop()
