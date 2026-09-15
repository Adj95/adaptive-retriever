"""Unit tests for query transformation interface and routing (Task A)."""

from pathlib import Path
import pytest

from adaptive_retriever.config import Config
from adaptive_retriever.models import QueryClass, QueryFingerprint
from adaptive_retriever.query.transform import TransformResult, transform_query


def test_strategy_from_real_config(tmp_path):
    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg.raw["query"]["transformation"]["strategy"] = "none"

    fp = QueryFingerprint(
        raw_query="test query",
        token_count=2,
        has_exact_identifiers=False,
        has_numeric_intent=False,
        has_comparative_language=False,
        query_class=QueryClass.CONCEPTUAL,
    )
    res = transform_query(fp, cfg=cfg)
    assert res.strategy == "none"


def test_enabled_false_short_circuits(tmp_path):
    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg.raw["query"]["transformation"]["enabled"] = False

    fp = QueryFingerprint(
        raw_query="what methods were used",
        token_count=4,
        has_exact_identifiers=False,
        has_numeric_intent=False,
        has_comparative_language=False,
        query_class=QueryClass.CONCEPTUAL,
    )

    res = transform_query(fp, cfg=cfg)
    assert res.strategy == "none"
    assert res.queries == ["what methods were used"]
    assert res.trace.get("skipped") is True
    assert res.trace.get("reason") == "transformation_disabled"


def test_fixed_overrides_strategy(tmp_path):
    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg.raw["query"]["transformation"]["fixed"] = "hyde"
    cfg.raw["query"]["transformation"]["strategy"] = "none"

    fp = QueryFingerprint(
        raw_query="find doc_123",
        token_count=2,
        has_exact_identifiers=True,
        has_numeric_intent=False,
        has_comparative_language=False,
        query_class=QueryClass.EXACT_LOOKUP,
    )

    res = transform_query(fp, cfg=cfg)
    assert res.strategy == "hyde"


def test_cache_hit_on_second_identical_call(tmp_path):
    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg.raw["query"]["transformation"]["fixed"] = "hyde"

    fp = QueryFingerprint(
        raw_query="cached query test",
        token_count=3,
        has_exact_identifiers=False,
        has_numeric_intent=False,
        has_comparative_language=False,
        query_class=QueryClass.CONCEPTUAL,
    )

    res1 = transform_query(fp, cfg=cfg)
    assert res1.trace.get("cache_hit") is False

    res2 = transform_query(fp, cfg=cfg)
    assert res2.trace.get("cache_hit") is True


def test_unindexed_field_rejected(tmp_path):
    """Test that an unindexed field never reaches extracted_filters and is moved to rejected_fields."""
    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg.raw["query"]["transformation"]["fixed"] = "self_query"

    fp = QueryFingerprint(
        raw_query="find results",
        token_count=2,
        has_exact_identifiers=False,
        has_numeric_intent=False,
        has_comparative_language=False,
        extracted_filters={"unindexed_field": "val123", "chunk_id": "chunk_999"},
        query_class=QueryClass.FILTERABLE,
    )

    res = transform_query(fp, cfg=cfg)
    assert "unindexed_field" not in res.extracted_filters
    assert "chunk_id" not in res.extracted_filters
    rejected = res.trace.get("details", {}).get("rejected_fields", {})
    assert rejected.get("unindexed_field") == "val123"
    assert rejected.get("chunk_id") == "chunk_999"


def test_zero_match_filter_dropped(tmp_path):
    """Test that a filter matching zero documents in VectorStore is dropped with a recorded reason."""
    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg.raw["query"]["transformation"]["fixed"] = "self_query"

    fp = QueryFingerprint(
        raw_query="find table_heavy docs",
        token_count=3,
        has_exact_identifiers=False,
        has_numeric_intent=False,
        has_comparative_language=False,
        query_class=QueryClass.FILTERABLE,
    )

    class MockVSZero:
        def count(self, count_filter=None):
            return 0

    res = transform_query(fp, cfg=cfg, vs=MockVSZero())
    assert res.extracted_filters == {}
    assert res.trace.get("details", {}).get("dropped_reason") == "matches_no_documents"


def test_malformed_llm_fallback_no_exception(tmp_path):
    """Test that malformed LLM output yields no filter and no exception."""
    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg.raw["query"]["transformation"]["fixed"] = "self_query"

    fp = QueryFingerprint(
        raw_query="complex question without rule keywords",
        token_count=5,
        has_exact_identifiers=False,
        has_numeric_intent=False,
        has_comparative_language=False,
        query_class=QueryClass.FILTERABLE,
    )

    def mock_bad_llm(prompt):
        return "This is NOT json: {invalid json content line 1..."

    res = transform_query(fp, cfg=cfg, llm=mock_bad_llm)
    assert res.extracted_filters == {}
    assert res.trace.get("details", {}).get("llm_attempted") is True
    assert res.trace.get("details", {}).get("llm_parse_success") is False


def test_valid_doc_class_extraction(tmp_path):
    """Test that a valid doc_class mention in query text produces the right filter condition."""
    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg.raw["query"]["transformation"]["fixed"] = "self_query"

    fp = QueryFingerprint(
        raw_query="what are the financial figures in the table_heavy documents",
        token_count=9,
        has_exact_identifiers=False,
        has_numeric_intent=False,
        has_comparative_language=False,
        query_class=QueryClass.FILTERABLE,
    )

    res = transform_query(fp, cfg=cfg)
    assert res.extracted_filters.get("doc_class") == "table_heavy"
    assert res.extracted_filters.get("contains_table") is True


def test_transform_filter_precedence_over_profile(tmp_path):
    """Test that transform_query's extracted_filters overwrite profile's extracted_filters on key collision."""
    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg.raw["query"]["transformation"]["fixed"] = "self_query"

    profile = QueryFingerprint(
        raw_query="table_heavy document search",
        token_count=3,
        has_exact_identifiers=False,
        has_numeric_intent=False,
        has_comparative_language=False,
        extracted_filters={"doc_class": "scanned", "tier": "high"},
        query_class=QueryClass.FILTERABLE,
    )

    transform_res = transform_query(profile, cfg=cfg)
    assert transform_res.extracted_filters["doc_class"] == "table_heavy"

    profile.extracted_filters.update(transform_res.extracted_filters)

    assert profile.extracted_filters["doc_class"] == "table_heavy"
    assert profile.extracted_filters["tier"] == "high"


def test_run_funnel_zero_match_filter_does_not_collapse_recall(tmp_path):
    """Integration test: run run_funnel end-to-end with a query whose filter matches 0 documents.
    Verify zero-match guard drops the filter so final candidate count is greater than zero.
    """
    from adaptive_retriever.funnel.pipeline import run_funnel
    from adaptive_retriever.models import RetrievalCandidate

    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg.raw["query"]["transformation"]["enabled"] = True
    cfg.raw["query"]["transformation"]["fixed"] = "self_query"

    class MockVSEndToEnd:
        def count(self, count_filter=None):
            if count_filter is not None:
                return 0
            return 100

        def query_hybrid(self, query_text=None, limit=10, query_filter=None, cfg=None, **kwargs):
            if query_filter is not None:
                return []
            return [
                RetrievalCandidate(
                    chunk_id=f"c_{i}",
                    text=f"sample text {i}",
                    fused_score=0.9 - i * 0.05,
                    metadata={"matched_kind": "raw"},
                )
                for i in range(limit)
            ]

    vs = MockVSEndToEnd()
    cands, trace = run_funnel("find python source code in script", vs=vs, cfg=cfg)
    assert len(cands) > 0, "Zero-match guard failed: returned 0 candidates!"
    assert trace["stages"]["transformation"]["details"]["dropped_reason"] == "matches_no_documents"
    assert trace["stages"]["prefilter"]["decision"] == "no_constraints"


