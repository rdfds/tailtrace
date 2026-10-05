import json

import pytest

from tailtrace.campaign import cases, evaluate, run_campaign, summarize


def test_deterministic_cases_and_feasible_baseline_nonregression():
    assert list(cases([7])) == list(cases([7]))
    rows = [evaluate(c, 100_000) for c in cases([7])]
    assert len(rows) == 48
    assert any(row["oracle"]["status"] == "optimal" for row in rows)
    for row in rows:
        a, b = row["schedules"]["random"], row["schedules"]["fleet_local"]
        if a["status"] == b["status"] == "feasible":
            assert b["certificate"]["makespan"] <= a["certificate"]["makespan"]
    assert summarize(rows)["methods"]["fleet_local"]["proven_cases"] > 0


def test_budgeted_campaign_retains_unknown_results_and_refuses_overwrite(tmp_path):
    out = tmp_path / "campaign"
    data = run_campaign(out, [3], node_budget=1)
    assert data["summary"]["oracle_status"]["budget_exhausted"] == 48
    assert data["summary"]["methods"]["fleet_local"]["proven_cases"] == 0
    assert json.loads((out / "campaign.json").read_text()) == data
    with pytest.raises(FileExistsError):
        run_campaign(out, [3])
