"""Reproducible adversarial proxy campaign; no generated GPU measurements."""

from __future__ import annotations

import math
import random
import statistics
from pathlib import Path

from tailtrace.certificate import certify
from tailtrace.cost import RankCost
from tailtrace.evidence import atomic_json
from tailtrace.oracle import solve_exact
from tailtrace.planner import plan_epoch
from tailtrace.scheduler import schedule


def cases(seeds):
    for seed in seeds:
        for world in (2, 3):
            batch_size = 3 if world == 2 else 4
            n = world * batch_size
            rng = random.Random(seed)
            distributions = {
                "flat": [32] * n,
                "one_outlier": [128, *([8] * (n - 1))],
                "two_outliers": [128, 96, *([8] * (n - 2))],
                "bimodal": [8] * (n // 2) + [64] * (n - n // 2),
                "staircase": [8 + i * 11 for i in range(n)],
                "log_uniform": [
                    round(math.exp(rng.uniform(math.log(4), math.log(128)))) for _ in range(n)
                ],
            }
            fleets = {
                "homogeneous": [RankCost(1, 4, 20, n - 1)] * world,
                "speed_skew": [RankCost(1 + r * 2, 4 + r * 8, 20, n - 1) for r in range(world)],
                "compute_crossover": [
                    RankCost(1 + r, 40 / (r + 1), 30 * (r + 1), n - 1) for r in range(world)
                ],
                "tight_capacity": [
                    RankCost(1 + r, 8, 20, 3 if r else n - world + 1, 180 if r else 600)
                    for r in range(world)
                ],
            }
            for name, lengths in distributions.items():
                for fleet_name, models in fleets.items():
                    yield {
                        "id": f"{name}/{fleet_name}/w{world}/seed{seed}",
                        "distribution": name,
                        "fleet": fleet_name,
                        "seed": seed,
                        "batch_size": batch_size,
                        "lengths": lengths,
                        "models": [m.to_dict() for m in models],
                    }


def evaluate(case, node_budget):
    lengths = case["lengths"]
    models = [RankCost(**m) for m in case["models"]]
    baseline_plan = plan_epoch(lengths, len(models), case["batch_size"], case["seed"], "random")[0]
    ids = [i for g in baseline_plan.ranks for i in g]
    schedules = {}
    for name in (
        "random",
        "balanced",
        "fleet_greedy",
        "fleet_local",
        "fleet_beam_single",
        "fleet_beam",
    ):
        try:
            if name in {"random", "balanced"}:
                groups = plan_epoch(lengths, len(models), case["batch_size"], case["seed"], name)[
                    0
                ].ranks
            else:
                groups = schedule(
                    ids,
                    lengths,
                    models,
                    local_rounds=0 if name == "fleet_greedy" else 12,
                    beam_width=64 if name in {"fleet_beam", "fleet_beam_single"} else 0,
                    retain_local_incumbent=name != "fleet_beam_single",
                )
            schedules[name] = {
                "groups": [list(g) for g in groups],
                "certificate": certify(ids, lengths, groups, models),
                "status": "feasible",
            }
        except ValueError as error:
            schedules[name] = {"status": "unavailable", "reason": str(error)}
    incumbents = [s for s in schedules.values() if s["status"] == "feasible"]
    incumbent = (
        min(
            incumbents,
            key=lambda s: (s["certificate"]["makespan"], sum(s["certificate"]["rank_costs"])),
        )
        if incumbents
        else None
    )
    oracle = solve_exact(
        ids, lengths, models, node_budget, incumbent["groups"] if incumbent else None
    )
    if oracle["status"] == "optimal":
        for result in incumbents:
            result["proven_regret"] = result["certificate"]["makespan"] / oracle["upper_bound"] - 1
    return {**case, "shuffled_ids": ids, "schedules": schedules, "oracle": oracle}


def summarize(rows):
    summary = {
        "cases": len(rows),
        "oracle_status": {
            status: sum(r["oracle"]["status"] == status for r in rows)
            for status in ("optimal", "infeasible", "budget_exhausted")
        },
        "methods": {},
    }
    for method in (
        "random",
        "balanced",
        "fleet_greedy",
        "fleet_local",
        "fleet_beam_single",
        "fleet_beam",
    ):
        values = [
            r["schedules"][method]["proven_regret"]
            for r in rows
            if "proven_regret" in r["schedules"][method]
        ]
        summary["methods"][method] = {
            "feasible_cases": sum(r["schedules"][method]["status"] == "feasible" for r in rows),
            "proven_cases": len(values),
            "optimum_reached": sum(v < 1e-12 for v in values),
            "median_proven_regret": statistics.median(values) if values else None,
            "worst_proven_regret": max(values) if values else None,
        }
    return summary


def run_campaign(out, seeds=(17, 23, 31, 43), node_budget=100_000):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("require distinct campaign seeds")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"campaign output already exists: {out}")
    rows = [evaluate(case, node_budget) for case in cases(seeds)]
    data = {
        "schema_version": 1,
        "evidence": "analytic_proxy",
        "seeds": list(seeds),
        "node_budget": node_budget,
        "summary": summarize(rows),
        "cases": rows,
        "limitations": [
            "Costs and capacity limits are declared analytic assumptions, not GPU measurements.",
            "Optimality applies only to completed small-batch searches under this cost model.",
            "A proxy-optimal assignment can perform worse on hardware; validate isolated fits and paired training runs.",
            "Shape caps are not a GPU-memory safety guarantee.",
        ],
    }
    atomic_json(out / "campaign.json", data)
    return data
