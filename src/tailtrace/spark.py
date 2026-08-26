from __future__ import annotations


def step_aggregates(df):
    """Keep independent runs separate using their input parent directory."""
    from pyspark.sql import functions as F

    source = F.input_file_name()
    with_run = df.withColumn("run", F.regexp_replace(source, r"/metrics-rank[^/]*$", ""))
    duplicates = (
        with_run.groupBy("run", "step", "rank").count().filter("count != 1").limit(1).count()
    )
    if duplicates:
        raise ValueError("duplicate rank/step metrics")
    rows = with_run.filter(~F.col("warmup"))
    result = rows.groupBy("run", "step").agg(
        F.countDistinct("rank").alias("observed_ranks"),
        F.max("step_ms").alias("step_ms"),
        F.min("step_ms").alias("fastest_rank_ms"),
        F.sum("local_tokens").alias("tokens"),
        F.min("global_tokens").alias("min_global_tokens"),
        F.max("global_tokens").alias("max_global_tokens"),
        F.sum("padded_tokens").alias("padded_tokens"),
        F.sum("loss_sum").alias("loss_sum"),
        F.max("peak_memory_bytes").alias("peak_memory_bytes"),
    )
    # The planner's global token count provides a rank-coverage check even without a manifest.
    invalid = (
        result.filter(
            (F.col("tokens") != F.col("max_global_tokens"))
            | (F.col("min_global_tokens") != F.col("max_global_tokens"))
            | (F.col("step_ms") <= 0)
        )
        .limit(1)
        .count()
    )
    if invalid:
        raise ValueError("incomplete ranks, inconsistent global tokens, or invalid timings")
    return (
        result.withColumn("tokens_per_second", F.col("tokens") / F.col("step_ms") * 1000)
        .withColumn("padding_fraction", 1 - F.col("tokens") / F.col("padded_tokens"))
        .withColumn("loss", F.col("loss_sum") / F.col("tokens"))
    )


