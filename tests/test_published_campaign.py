import json
from pathlib import Path

from tailtrace.campaign import summarize
from tailtrace.certificate import certify
from tailtrace.cost import RankCost


def test_published_assignments_and_summary_are_internally_certifiable():
    data = json.loads(
        (Path(__file__).parents[1] / "results/scheduling-audit/campaign.json").read_text()
    )
    assert data["summary"] == summarize(data["cases"])
    for case in data["cases"]:
        models = [RankCost(**m) for m in case["models"]]
        for result in case["schedules"].values():
            if result["status"] == "feasible":
                certificate = certify(
                    case["shuffled_ids"], case["lengths"], result["groups"], models
                )
                assert certificate == result["certificate"]
                if case["oracle"]["status"] == "optimal":
                    assert certificate["makespan"] >= case["oracle"]["upper_bound"]
        oracle = case["oracle"]
        if oracle["groups"]:
            assert (
                certify(case["shuffled_ids"], case["lengths"], oracle["groups"], models)
                == oracle["certificate"]
            )
