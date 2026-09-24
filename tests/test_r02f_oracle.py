import json
from pathlib import Path
import pytest
from scripts.evaluate_r02f_oracle import evaluate_r02f

def test_evaluate_r02f_oracle_report(tmp_path, monkeypatch):
    test_report_path = tmp_path / "r02f_oracle.json"
    monkeypatch.setattr("scripts.evaluate_r02f_oracle.REPORT_PATH", test_report_path)
    
    evaluate_r02f()
    
    assert test_report_path.exists()
    with open(test_report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["task"] == "R-02f"
    assert data["f1_corrected_tier_costs"]["assertion_passed"] is True
    assert data["f1_corrected_tier_costs"]["oracle_mean_cost_all_queries_ms"] == 3210.58
    assert data["f3_unanswerable_fraction"]["unanswerable_count"] == 54
    assert len(data["f2_success_matched_comparison"]["simulation_table"]) == 11
