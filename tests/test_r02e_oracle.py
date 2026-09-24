import json
from pathlib import Path
import pytest
from scripts.evaluate_r02e_oracle import evaluate_r02e_oracle

def test_evaluate_r02e_oracle_report(tmp_path, monkeypatch):
    test_report_path = tmp_path / "r02e_oracle.json"
    monkeypatch.setattr("scripts.evaluate_r02e_oracle.REPORT_PATH", test_report_path)
    
    evaluate_r02e_oracle()
    
    assert test_report_path.exists()
    with open(test_report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["task"] == "R-02e"
    assert "success" in data["definitions"]
    assert data["e1_oracle_bound"]["oracle_mean_cost_ms"] == 943.24
    assert data["e1_oracle_bound"]["max_achievable_saving_vs_always_t2_pct"] == 38.62
    assert len(data["e2_router_simulation"]) == 11
    assert "finding_1" in data["e3_findings"]
