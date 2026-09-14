"""Unit tests for Query Profiler module."""

import pytest
from adaptive_retriever.models import QueryClass, QueryFingerprint
from adaptive_retriever.query.profiler import profile_query


def test_profile_query_conceptual():
    fp = profile_query("what methods were used to evaluate results")
    assert isinstance(fp, QueryFingerprint)
    assert fp.raw_query == "what methods were used to evaluate results"
    assert fp.query_class == QueryClass.CONCEPTUAL
    assert fp.class_confidence >= 0.5


def test_profile_query_filterable_with_table():
    fp = profile_query("show performance table for dataset")
    assert fp.query_class == QueryClass.FILTERABLE
    assert fp.extracted_filters.get("contains_table") is True


def test_profile_query_exact_lookup():
    fp = profile_query('find document "DOC_CLASS_9982"')
    assert fp.query_class == QueryClass.EXACT_LOOKUP
    assert fp.has_exact_identifiers is True


def test_profile_query_multihop_comparison():
    fp = profile_query("compare accuracy between model A and model B")
    assert fp.query_class == QueryClass.MULTIHOP
    assert fp.has_comparative_language is True
