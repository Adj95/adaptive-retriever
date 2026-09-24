"""Unit tests for scripts/evaluate_r02b_gate.py (Task R-02b)."""

import pytest
import numpy as np
from scripts.evaluate_r02b_gate import (
    extract_content_words,
    has_8gram_match,
    compute_roc_auc,
    compute_bootstrap_auc,
)

def test_extract_content_words():
    text = "The quick brown fox jumps over the lazy dog."
    words = extract_content_words(text)
    assert "quick" in words
    assert "brown" in words
    assert "the" not in words
    assert "over" not in words

def test_has_8gram_match():
    sentence = "This is a simple eight word test sentence for matching."
    context = "Here is context containing This is a simple eight word test sentence for matching in text."
    assert has_8gram_match(sentence, context) is True

    short_context = "This is unrelated text."
    assert has_8gram_match(sentence, short_context) is False

def test_compute_roc_auc():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    auc = compute_roc_auc(y_true, y_score)
    assert auc == 1.0

    y_score_inv = np.array([0.1, 0.2, 0.8, 0.9])
    auc_inv = compute_roc_auc(y_true, y_score_inv)
    assert auc_inv == 0.0

def test_compute_bootstrap_auc():
    y_true = np.array([1]*20 + [0]*20)
    y_score = np.array([0.9]*20 + [0.1]*20)
    auc_obs, ci = compute_bootstrap_auc(y_true, y_score, resamples=100, seed=42)
    assert auc_obs == 1.0
    assert ci[0] == 1.0
    assert ci[1] == 1.0
