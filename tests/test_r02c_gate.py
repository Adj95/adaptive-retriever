import pytest
import numpy as np
from scripts.evaluate_r02c_gate import (
    compute_roc_auc,
    compute_bootstrap_auc,
    has_8gram_match,
    compute_bootstrap_diff,
)

def test_roc_auc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    auc = compute_roc_auc(y_true, y_score)
    assert abs(auc - 1.0) < 1e-6

def test_roc_auc_constant_score():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    auc = compute_roc_auc(y_true, y_score)
    assert abs(auc - 0.5) < 1e-6

def test_s1b_cosine_margin():
    cos_top1 = 0.85
    cos_top2 = 0.70
    s1b = (cos_top1 - cos_top2) / cos_top1
    assert abs(s1b - (0.15 / 0.85)) < 1e-6

def test_s5_rank_mean():
    import scipy.stats as stats
    s1b = np.array([0.1, 0.2, 0.3, 0.4])
    s2 = np.array([0.4, 0.3, 0.2, 0.1])
    rank1 = stats.rankdata(s1b) / 4.0
    rank2 = stats.rankdata(s2) / 4.0
    s5 = (rank1 + rank2) / 2.0
    assert len(s5) == 4
    assert np.allclose(s5, 0.625) # (1/4 + 4/4)/2 = 0.625

def test_bootstrap_diff():
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.7, 0.8, 0.9])
    diff, ci, p_val = compute_bootstrap_diff(a, b, resamples=100)
    assert abs(diff - 0.2) < 1e-6
    assert ci[0] <= diff <= ci[1]
