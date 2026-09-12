"""Tests configuration validation rules per spec §7.1 and Addenda."""

import pytest
import logging
from adaptive_retriever.config import Config


def test_judge_position_swap_required():
    invalid_judge = {
        "ingest": {"profiler": {"enabled": True}},
        "eval": {
            "judge": {
                "enabled": True,
                "position_swap": False,
            }
        },
    }
    with pytest.raises(ValueError, match="position_swap"):
        Config(invalid_judge)


def test_escalation_rerank_always_warning(caplog):
    config_data = {
        "ingest": {"profiler": {"enabled": True}},
        "eval": {"judge": {"enabled": False}},
        "escalation": {"enabled": True},
        "funnel": {"rerank": {"mode": "always"}},
    }
    with caplog.at_level(logging.WARNING):
        Config(config_data)
    assert any("escalation rung 1" in record.message for record in caplog.records)


def test_router_fixed_fallback_required():
    no_fixed_parser = {
        "ingest": {
            "profiler": {"enabled": True},
            "parser_router": {"enabled": False, "fixed_parser": None},
        },
        "eval": {"judge": {"enabled": False}},
    }
    with pytest.raises(ValueError, match="fixed_parser is not set"):
        Config(no_fixed_parser)
