import json
from pathlib import Path
import pytest
from scripts.evaluate_r02d import evaluate_r02d

def test_evaluate_r02d_report_generation(tmp_path, monkeypatch):
    test_report_path = tmp_path / "r02d.json"
    monkeypatch.setattr("scripts.evaluate_r02d.REPORT_PATH", test_report_path)
    
    evaluate_r02d()
    
    assert test_report_path.exists()
    with open(test_report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["task"] == "R-02d"
    assert data["d1_model_name_check"]["models_differ"] is True
    assert data["d3_signal_equivalence"]["same_quantity"] is True
    assert data["d4_extractive_tier_0"]["tier_0_cost_p50_ms"] == 75.76
    assert data["d4_extractive_tier_0"]["cost_ratio_tier2_to_tier0"] == 20.29
    assert data["d4_extractive_tier_0"]["top1_chunk_contains_gold_accuracy"] == 0.3787
