"""Unit tests for prefilter (stage 1 of retrieval funnel)."""

import pytest
from qdrant_client.http import models as rest

from adaptive_retriever.funnel.prefilter import INDEXED_FIELDS, build_prefilter
from adaptive_retriever.models import QueryClass, QueryFingerprint


def test_prefilter_built_from_entity_constraint():
    profile = {"entities": ["Google"]}
    filter_obj, trace = build_prefilter(profile)

    assert filter_obj is not None
    assert isinstance(filter_obj, rest.Filter)
    assert len(filter_obj.must) == 1
    cond = filter_obj.must[0]
    assert cond.key == "entities"
    assert cond.match.value == "Google"
    assert trace["skipped"] is False
    assert trace["applied_constraints"] == {"entities": "Google"}


def test_prefilter_multiple_entities_match_any():
    profile = {"entities": ["Google", "DeepMind"]}
    filter_obj, trace = build_prefilter(profile)

    assert filter_obj is not None
    assert isinstance(filter_obj, rest.Filter)
    assert len(filter_obj.must) == 1
    cond = filter_obj.must[0]
    assert cond.key == "entities"
    assert getattr(cond.match, "any") == ["Google", "DeepMind"]
    assert trace["applied_constraints"] == {"entities": ["Google", "DeepMind"]}


def test_prefilter_unindexed_field_never_reaches_filter():
    profile = {
        "entities": ["PyTorch"],
        "unindexed_author": "John Doe",
        "doc_id": "doc123",
        "chunk_id": "c456",
    }
    filter_obj, trace = build_prefilter(profile)

    assert filter_obj is not None
    must_keys = [cond.key for cond in filter_obj.must]
    assert "unindexed_author" not in must_keys
    assert "doc_id" not in must_keys
    assert "chunk_id" not in must_keys
    assert must_keys == ["entities"]

    for key in must_keys:
        assert key in INDEXED_FIELDS

    assert "unindexed_author" in trace["unindexed_skipped"]
    assert trace["unindexed_skipped"]["unindexed_author"] == "John Doe"


def test_prefilter_disabled_yields_none_and_skipped_trace():
    profile = {"entities": ["Google"]}
    cfg = {"funnel": {"prefilter": {"enabled": False}}}

    filter_obj, trace = build_prefilter(profile, cfg=cfg)

    assert filter_obj is None
    assert trace["skipped"] is True
    assert trace["reason"] == "prefilter_disabled"


def test_prefilter_no_indexed_constraints_yields_none():
    profile = {"unindexed_search_term": "foo"}
    filter_obj, trace = build_prefilter(profile)

    assert filter_obj is None
    assert trace["skipped"] is False
    assert trace["reason"] == "no_indexed_constraints"


def test_prefilter_query_fingerprint_extracted_filters():
    profile = QueryFingerprint(
        raw_query="find table data",
        token_count=3,
        has_exact_identifiers=False,
        has_numeric_intent=False,
        has_comparative_language=False,
        query_class=QueryClass.CONCEPTUAL,
        class_confidence=1.0,
        extracted_filters={"doc_class": "table_heavy"},
    )
    filter_obj, trace = build_prefilter(profile)

    assert filter_obj is not None
    assert len(filter_obj.must) == 1
    assert filter_obj.must[0].key == "doc_class"
    assert filter_obj.must[0].match.value == "table_heavy"
    assert trace["applied_constraints"]["doc_class"] == "table_heavy"


def test_prefilter_invalid_config_raises_value_error():
    profile = {"entities": ["Python"]}
    # Bare dict missing top-level "funnel"
    with pytest.raises(ValueError, match="must contain top-level 'funnel' key"):
        build_prefilter(profile, cfg={"prefilter": {"enabled": True}})

    # Non-dict non-Config object
    with pytest.raises(ValueError, match="Invalid config type"):
        build_prefilter(profile, cfg="not_a_config_object")
