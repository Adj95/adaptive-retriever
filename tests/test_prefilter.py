"""Unit tests for prefilter (stage 1 of retrieval funnel)."""

import pytest
from qdrant_client.http import models as rest

from adaptive_retriever.config import Config
from adaptive_retriever.funnel.prefilter import INDEXED_FIELDS, build_prefilter
from adaptive_retriever.models import DocClass, QueryProfile, RepresentationKind


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


def test_prefilter_query_profile_model_input():
    profile = QueryProfile(
        raw_query="Find Q4 reports",
        doc_class=DocClass.DIGITAL_PROSE,
        contains_table=True,
        kind=RepresentationKind.SUMMARY,
    )
    filter_obj, trace = build_prefilter(profile)

    assert filter_obj is not None
    must_keys = {cond.key for cond in filter_obj.must}
    assert must_keys == {"doc_class", "contains_table", "kind"}
    assert trace["applied_constraints"]["doc_class"] == "digital_prose"
    assert trace["applied_constraints"]["contains_table"] is True
    assert trace["applied_constraints"]["kind"] == "summary"
