"""Paired local CPU timings of identical full-rescoring and incremental moves."""

from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

from tailtrace.certificate import certify
from tailtrace.cost import RankCost
from tailtrace.evidence import atomic_json, provenance
from tailtrace.scheduler import improve, improve_reference


def benchmark(out, repeats=7, rounds=3, seed=17):
    if repeats < 3 or rounds < 1:
        raise ValueError("require repeats >= 3 and rounds >= 1")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    cases = []
    methods = {"full_rescoring": improve_reference, "incremental": improve}
    for world, batch in ((2, 4), (4, 8), (8, 8), (8, 16)):
        lengths = [rng.randrange(2, 129) for _ in range(world * batch)]
        ids = list(range(len(lengths)))
        rng.shuffle(ids)
        groups = [ids[r * batch : (r + 1) * batch] for r in range(world)]
        models = [
            RankCost(quadratic=1 + r * 0.37, linear=0.2, max_samples=batch * 2)
            for r in range(world)
        ]
        # Untimed warmup exercises both paths; every timed result is checked too.
        reference = improve_reference(groups, lengths, models, rounds)
        assert improve(groups, lengths, models, rounds) == reference
        certify(ids, lengths, reference, models)
        observations = []
        for repeat in range(repeats):
            order = list(methods)
            rng.shuffle(order)
            timings = {}
            for name in order:
                start = time.perf_counter_ns()
                actual = methods[name](groups, lengths, models, rounds)
                timings[name] = (time.perf_counter_ns() - start) / 1e6
                if actual != reference:
                    raise AssertionError("timed scheduler output differs from reference")
            observations.append({"repeat": repeat, "order": order, "ms": timings})
        medians = {name: statistics.median(x["ms"][name] for x in observations) for name in methods}
        cases.append(
            {
                "world_size": world,
                "samples": len(lengths),
                "lengths": lengths,
                "initial_groups": groups,
                "models": [m.to_dict() for m in models],
                "assignment": reference,
                "observations": observations,
                "median_ms": medians,
                "median_ratio": medians["full_rescoring"] / medians["incremental"],
            }
        )
    data = {
        "schema_version": 1,
        "evidence": "observed_cpu",
        "scope": "local_search_only",
        "reference": "v0.2.0 full-rescoring move/swap traversal",
        "seed": seed,
        "repeats": repeats,
        "rounds": rounds,
        "hardware": provenance(),
        "cases": cases,
        "all_assignments_identical": True,
        "limitations": [
            "single host",
            "no GPU or training throughput measurement",
            "no claim about beam construction or oracle runtime",
        ],
    }
    atomic_json(out / "benchmark.json", data)
    return data
