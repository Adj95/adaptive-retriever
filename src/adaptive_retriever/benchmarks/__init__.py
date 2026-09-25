"""BEIR Benchmark module for Adaptive Retriever.

Contains relevance contract and utilities for BEIR datasets (NFCorpus & HotpotQA).
"""

from .beir import is_relevant_beir, get_qrel_score_beir

__all__ = ["is_relevant_beir", "get_qrel_score_beir"]
