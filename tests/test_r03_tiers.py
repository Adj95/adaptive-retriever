"""Tests for scripts/evaluate_r03_tiers.py (Task R-03)."""

import pytest
from adaptive_retriever.config import Config
from scripts.evaluate_r03_tiers import build_tier_config

def test_build_tier_config_tier_1():
    cfg = Config.load("config/default.yaml")
    t1_cfg = build_tier_config(cfg, 1)
    
    funnel_cfg = t1_cfg.raw["funnel"]
    assert funnel_cfg["hybrid"]["dense_limit"] == 5
    assert funnel_cfg["hybrid"]["sparse_limit"] == 0
    assert funnel_cfg["hybrid"]["weights"] == {"dense": 1.0, "sparse": 0.0}
    assert funnel_cfg["prefilter"]["kind_filter"] == "raw"
    assert funnel_cfg["rerank"]["mode"] == "never"
    assert funnel_cfg["rerank"]["output_size"] == 5

def test_build_tier_config_tier_2():
    cfg = Config.load("config/default.yaml")
    t2_cfg = build_tier_config(cfg, 2)
    
    funnel_cfg = t2_cfg.raw["funnel"]
    assert funnel_cfg["hybrid"]["dense_limit"] == 10
    assert funnel_cfg["hybrid"]["sparse_limit"] == 10
    assert funnel_cfg["hybrid"]["weights"] == {"dense": 0.5, "sparse": 0.5}
    assert funnel_cfg["prefilter"]["kind_filter"] == "raw"
    assert funnel_cfg["rerank"]["mode"] == "always"
    assert funnel_cfg["rerank"]["input_size"] == 10
    assert funnel_cfg["rerank"]["output_size"] == 10

def test_build_tier_config_tier_3():
    cfg = Config.load("config/default.yaml")
    t3_cfg = build_tier_config(cfg, 3)
    
    funnel_cfg = t3_cfg.raw["funnel"]
    assert funnel_cfg["hybrid"]["dense_limit"] == 20
    assert funnel_cfg["hybrid"]["sparse_limit"] == 20
    assert funnel_cfg["hybrid"]["weights"] == {"dense": 0.5, "sparse": 0.5}
    assert funnel_cfg["prefilter"]["kind_filter"] == "all"
    assert funnel_cfg["rerank"]["mode"] == "always"
    assert funnel_cfg["rerank"]["input_size"] == 20
    assert funnel_cfg["rerank"]["output_size"] == 20
