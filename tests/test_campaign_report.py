from tailtrace.campaign import run_campaign
from tailtrace.campaign_report import write_campaign_html


def test_report_has_offline_interactions_and_escapes_script_payload(tmp_path):
    data = run_campaign(tmp_path / "data", [3], node_budget=1)
    data["cases"][0]["id"] = '</script><script src="https://evil.example/x">'
    out = tmp_path / "report.html"
    write_campaign_html(data, out)
    text = out.read_text()
    assert "</script><script src=" not in text
    assert "addEventListener" in text
    assert "createElementNS" in text
    assert "\\u003c/script>" in text
    assert "__PAYLOAD__" not in text
