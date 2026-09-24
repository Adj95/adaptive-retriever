"""Tests for scripts/verify_report.py (Task R-02a).

Uses the 6 real past failure fixtures to confirm each check fails on bad data and passes on valid data.
"""

import pytest
from pathlib import Path
from scripts.verify_report import (
    VerificationError,
    check_range,
    check_arithmetic,
    check_subgroup_sums,
    check_quantisation,
    check_count_deltas,
    check_rate_plausibility,
    check_cross_report_drift,
    verify_report,
)

def test_check_1_range_real_fixture():
    """Fixture 1: nDCG@10 CI [0.7188, 1.0114] must fail Check 1 (RANGE)."""
    bad_data = {
        "metrics": {
            "nDCG@10": "0.8500 [0.7188, 1.0114]"
        }
    }
    with pytest.raises(VerificationError) as exc_info:
        check_range(bad_data)
    assert "RANGE violation" in str(exc_info.value)

    good_data = {
        "metrics": {
            "nDCG@10": "0.8500 [0.7188, 0.9814]"
        }
    }
    check_range(good_data)


def test_check_2_arithmetic_real_fixture():
    """Fixture 2: MRR reported +0.1028, 0.4949 - 0.3922 = 0.1027 must fail Check 2 (ARITHMETIC)."""
    bad_data = {
        "table": [
            {
                "metric": "MRR",
                "mean_A": 0.3922,
                "mean_B": 0.4949,
                "diff_CB": 0.1028  # Actual difference is 0.1027
            }
        ]
    }
    with pytest.raises(VerificationError) as exc_info:
        check_arithmetic(bad_data)
    assert "ARITHMETIC violation" in str(exc_info.value)

    good_data = {
        "table": [
            {
                "metric": "MRR",
                "mean_A": 0.3922,
                "mean_B": 0.4949,
                "diff_CB": 0.1027
            }
        ]
    }
    check_arithmetic(good_data)


def test_check_3_subgroup_sums_real_fixture():
    """Fixture 3: (131 x 0.3092 + 38 x 0.6316)/169 = 0.381691 != reported 0.3917."""
    bad_data = {
        "subgroups": {
            "total_n": 169,
            "total_mean": 0.3917,  # Bad reported total
            "groups": [
                {"name": "group_a", "n": 131, "mean": 0.3092},
                {"name": "group_b", "n": 38, "mean": 0.6316},
            ]
        }
    }
    with pytest.raises(VerificationError) as exc_info:
        check_subgroup_sums(bad_data)
    assert "SUBGROUP SUMS violation" in str(exc_info.value)

    good_data = {
        "subgroups": {
            "total_n": 169,
            "total_mean": (131 * 0.3092 + 38 * 0.6316) / 169,
            "groups": [
                {"name": "group_a", "n": 131, "mean": 0.3092},
                {"name": "group_b", "n": 38, "mean": 0.6316},
            ]
        }
    }
    check_subgroup_sums(good_data)


def test_check_4_quantisation_real_fixture():
    """Fixture 4: Recall@10 diff -0.0059 x 169 = 1 query item delta."""
    data = {
        "full_queries": 169,
        "table": [
            {
                "metric": "Recall@10",
                "mean_B": 0.6568,
                "mean_C": 0.6509,
                "diff_CB": -0.0059
            }
        ]
    }
    notes = check_quantisation(data)
    assert any("QUANTISATION FLAG" in n for n in notes)
    assert any("1 query item delta" in n for n in notes)


def test_check_5_count_deltas_real_fixture():
    """Fixture 5: 7,270 materialized -> 889 indexed must fail Check 5 (COUNT DELTAS)."""
    bad_data = {
        "count_deltas": {
            "before": 39814,
            "after": 40703,
            "added": 7270  # Actual diff is 889
        }
    }
    with pytest.raises(VerificationError) as exc_info:
        check_count_deltas(bad_data)
    assert "COUNT DELTAS violation" in str(exc_info.value)

    good_data = {
        "count_deltas": {
            "before": 39814,
            "after": 47084,
            "added": 7270
        }
    }
    check_count_deltas(good_data)


def test_check_6_rate_plausibility_real_fixture():
    """Fixture 6: 447 generations in 4.22 s (0.0094 s/gen) vs 6.679 s/gen benchmark (>10x diff)."""
    bad_data = {
        "benchmark": {
            "count": 447,
            "wall_clock_sec": 4.22,
            "known_sec_per_item": 6.679
        }
    }
    with pytest.raises(VerificationError) as exc_info:
        check_rate_plausibility(bad_data)
    assert "RATE PLAUSIBILITY violation" in str(exc_info.value)

    good_data = {
        "benchmark": {
            "count": 447,
            "wall_clock_sec": 4.22,
            "known_sec_per_item": 0.0094
        }
    }
    check_rate_plausibility(good_data)


def test_check_7_cross_report_drift():
    """Test Check 7: CROSS-REPORT DRIFT detects metric value changes."""
    data_curr = {
        "table": [
            {"metric": "Recall@10", "mean_B": 0.6568, "Arm_B": "0.6568"}
        ]
    }
    data_prev = {
        "table": [
            {"metric": "Recall@10", "mean_B": 0.6000, "Arm_B": "0.6000"}
        ]
    }
    drift_lines = check_cross_report_drift(data_curr, data_prev)
    assert any("Recall@10.Arm_B" in line for line in drift_lines)
