from __future__ import annotations

import argparse
import json

from tailtrace.evidence import atomic_json
from tailtrace.report import write_html


def main():
    parser = argparse.ArgumentParser(
        prog="tailtrace", description="Distributed training evidence lab"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    corpus = commands.add_parser("corpus-build", help="build immutable UTF-8 byte-token shards")
    corpus.add_argument("inputs", nargs="+")
    corpus.add_argument("--out", required=True)
    corpus.add_argument("--max-length", type=int, default=128)
    corpus.add_argument("--min-length", type=int, default=2)
    planner_bench = commands.add_parser(
        "planner-bench", help="measure identical local-search implementations"
    )
    planner_bench.add_argument("--out", required=True)
    planner_bench.add_argument("--repeats", type=int, default=7)
    planner_bench.add_argument("--rounds", type=int, default=3)
    planner_bench.add_argument("--seed", type=int, default=17)
    recovery = commands.add_parser(
        "recovery-audit", help="inject a real two-rank CPU process crash and verify recovery"
    )
    recovery.add_argument("--config", required=True)
    recovery.add_argument("--out", required=True)
    recovery.add_argument("--fault-step", type=int, default=2)
    demo = commands.add_parser("demo", help="generate a labeled synthetic offline example")
    demo.add_argument("--out", default="runs/demo")
    train = commands.add_parser("train", help="train under Python or torchrun")
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--resume")
    calibration = commands.add_parser("calibrate", help="fit isolated per-rank compute models")
    calibration.add_argument("--config", required=True)
    calibration.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    calibration.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32, 64])
    calibration.add_argument("--repeats", type=int, default=7)
    calibration.add_argument("--warmup", type=int, default=2)
    calibration.add_argument("--out", required=True)
    campaign = commands.add_parser(
        "schedule-audit", help="audit heuristic schedules against an exact proxy oracle"
    )
    campaign.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 31, 43])
    campaign.add_argument("--node-budget", type=int, default=100_000)
    campaign.add_argument("--out", required=True)
    analyze = commands.add_parser("analyze", help="analyze one rank's Chrome/Kineto trace")
    analyze.add_argument("trace")
    analyze.add_argument("--rank", type=int, default=0)
    analyze.add_argument("--device-id", type=int)
    analyze.add_argument("--out", required=True)
    nsys = commands.add_parser("nsys-import", help="convert a read-only Nsight SQLite export")
    nsys.add_argument("source")
    nsys.add_argument("--global-pid", type=int)
    nsys.add_argument("--out", required=True)
    summary = commands.add_parser("summarize", help="validate and summarize rank metrics")
    summary.add_argument("run")
    summary.add_argument("--out", required=True)
    compare = commands.add_parser("compare", help="compare paired observed runs by seed")
    compare.add_argument("--baseline", nargs="+", required=True)
    compare.add_argument("--candidate", nargs="+", required=True)
    compare.add_argument("--out", required=True)
    experiment = commands.add_parser("experiment", help="run randomized paired local experiments")
    experiment.add_argument("--config", required=True)
    experiment.add_argument("--intervention", required=True)
    experiment.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 31])
    experiment.add_argument("--workers", type=int, default=2)
    experiment.add_argument("--out", required=True)
    ray = commands.add_parser("ray-train", help="launch DDP/FSDP2 with Ray Train")
    ray.add_argument("--config", required=True)
    ray.add_argument("--out", required=True)
    ray.add_argument("--workers", type=int, default=2)
    ray.add_argument("--address")
    ray.add_argument("--storage", required=True)
    spark = commands.add_parser("spark-aggregate", help="aggregate rank JSONL into Parquet")
    spark.add_argument("input", help="glob or URI of metrics-rank*.jsonl files")
    spark.add_argument("--out", required=True)
    benchmark = commands.add_parser("kernel-bench", help="verify and benchmark the CUDA operator")
    benchmark.add_argument("--rows", type=int, default=4096)
    benchmark.add_argument("--width", type=int, default=1024)
    benchmark.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    benchmark.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "recovery-audit":
        from tailtrace.config import TrainConfig
        from tailtrace.recovery import audit

        data = audit(TrainConfig.load(args.config), args.out, args.fault_step)
        print(json.dumps(data["comparison"], indent=2))
        return
    if args.command == "planner-bench":
        from tailtrace.planner_bench import benchmark

        data = benchmark(args.out, args.repeats, args.rounds, args.seed)
        print(
            json.dumps(
                [
                    {k: c[k] for k in ("world_size", "samples", "median_ms", "median_ratio")}
                    for c in data["cases"]
                ],
                indent=2,
            )
        )
        return
    if args.command == "corpus-build":
        from tailtrace.corpus import build_corpus

        print(
            json.dumps(
                build_corpus(args.inputs, args.out, args.max_length, args.min_length), indent=2
            )
        )
        return
    if args.command == "schedule-audit":
        from pathlib import Path

        from tailtrace.campaign import run_campaign
        from tailtrace.campaign_report import write_campaign_html

        data = run_campaign(args.out, args.seeds, args.node_budget)
        write_campaign_html(data, Path(args.out) / "report.html")
        print(json.dumps(data["summary"], indent=2))
        return
    if args.command == "calibrate":
        from tailtrace.collect import collect
        from tailtrace.config import TrainConfig

        print(
            json.dumps(
                collect(
                    TrainConfig.load(args.config),
                    args.batches,
                    args.lengths,
                    args.repeats,
                    args.warmup,
                    args.out,
                ),
                indent=2,
            )
        )
        return
    if args.command == "train":
        from tailtrace.config import TrainConfig
        from tailtrace.train import run

        run(TrainConfig.load(args.config), args.out, args.resume)
        return
    if args.command == "nsys-import":
        from tailtrace.nsight import import_sqlite

        print(json.dumps(import_sqlite(args.source, args.out, args.global_pid), indent=2))
        return
    if args.command == "experiment":
        from tailtrace.experiment import run_experiment

        print(
            json.dumps(
                run_experiment(args.config, args.intervention, args.seeds, args.workers, args.out),
                indent=2,
            )
        )
        return
    if args.command == "ray-train":
        from tailtrace.ray_runner import launch

        launch(args)
        return
    if args.command == "spark-aggregate":
        from tailtrace.spark import aggregate

        aggregate(args.input, args.out)
        return
    if args.command == "kernel-bench":
        from tailtrace.kernel_bench import benchmark

        data = benchmark(args.rows, args.width, args.dtype)
    elif args.command == "demo":
        from tailtrace.demo import generate

        data = generate(args.out)
        print(json.dumps(data, indent=2))
        return
    elif args.command == "analyze":
        from tailtrace.traces import analyze_trace

        data = analyze_trace(args.trace, args.rank, args.device_id)
    elif args.command == "summarize":
        from tailtrace.report import summarize_run

        data = summarize_run(args.run)
    elif args.command == "compare":
        from tailtrace.report import compare_runs

        data = compare_runs(args.baseline, args.candidate)
    atomic_json(args.out, data)
    write_html(data, str(args.out) + ".html")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
