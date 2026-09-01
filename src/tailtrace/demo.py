from __future__ import annotations

from pathlib import Path

from tailtrace.evidence import atomic_json
from tailtrace.planner import plan_epoch
from tailtrace.report import write_html
from tailtrace.traces import analyze_trace


def generate(out: str | Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    events = [
        {"ph": "X", "cat": "user_annotation", "name": "tailtrace/step/0", "ts": 0, "dur": 10000},
        {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 0, "dur": 6000, "args": {"device": 0}},
        {
            "ph": "X",
            "cat": "kernel",
            "name": "ncclAllReduce",
            "ts": 4000,
            "dur": 4000,
            "args": {"device": 0},
        },
    ]
    trace = out / "synthetic-trace.json"
    atomic_json(trace, {"tailtrace_evidence": "synthetic", "traceEvents": events})
    report = analyze_trace(trace)
    lengths = [128, 16, 16, 16, 16, 16, 16, 16]
    a, b = (plan_epoch(lengths, 2, 4, 1, mode)[0] for mode in ("random", "balanced"))
    report["planner_example"] = {
        "lengths": lengths,
        "random_rank_costs": a.costs,
        "balanced_rank_costs": b.costs,
        "random_rank_counts": list(map(len, a.ranks)),
        "balanced_rank_counts": list(map(len, b.ranks)),
        "global_tokens": a.tokens,
        "claim": "Analytic proxy example; no GPU execution or measured speedup.",
    }
    atomic_json(out / "analysis.json", report)
    write_html(report, out / "report.html")
    return report
