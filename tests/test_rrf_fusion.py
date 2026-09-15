"""Tests for RRF fusion, representation collapse, and hybrid query wiring.

Every assertion here is written so that it can actually FAIL. Where a test
checks that a config value reaches the fusion, the value used is distinct from
every other number in the test, so a function returning a constant - or reading
the wrong field - cannot pass by coincidence.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from adaptive_retriever.config import Config
from adaptive_retriever.funnel.fusion import collapse_to_best, fuse_rrf
from adaptive_retriever.models import QueryClass
from adaptive_retriever.vectorstore import VectorStore

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


# --------------------------------------------------------------- fuse_rrf ---

def test_fuse_rrf_hand_computed_fixture():
    ranked_lists = {"dense": ["docA", "docB"], "sparse": ["docB", "docC"]}
    weights = {"dense": 0.5, "sparse": 0.5}

    results = fuse_rrf(ranked_lists, rrf_k=60, weights=weights)
    assert len(results) == 3

    doc_b, doc_a, doc_c = results

    # docB is found by both sources: dense rank 2, sparse rank 1
    assert doc_b[0] == "docB"
    assert doc_b[1] == pytest.approx((0.5 / 62.0) + (0.5 / 61.0))
    assert doc_b[2] == {"dense": 2, "sparse": 1}

    assert doc_a[0] == "docA"
    assert doc_a[1] == pytest.approx(0.5 / 61.0)
    assert doc_a[2] == {"dense": 1, "sparse": None}

    assert doc_c[0] == "docC"
    assert doc_c[1] == pytest.approx(0.5 / 62.0)
    assert doc_c[2] == {"dense": None, "sparse": 2}


def test_fuse_rrf_rank_is_one_based():
    """Top hit must contribute w / (rrf_k + 1), not w / (rrf_k + 0)."""
    res = fuse_rrf({"dense": ["docA"]}, rrf_k=9, weights={"dense": 1.0})
    assert res[0][1] == pytest.approx(1.0 / 10.0)
    assert res[0][2]["dense"] == 1


def test_fuse_rrf_rrf_k_changes_scores():
    ranked_lists = {"dense": ["docA", "docB"], "sparse": ["docB", "docC"]}
    weights = {"dense": 0.5, "sparse": 0.5}

    res_60 = fuse_rrf(ranked_lists, rrf_k=60, weights=weights)
    res_1 = fuse_rrf(ranked_lists, rrf_k=1, weights=weights)

    score_60 = next(r[1] for r in res_60 if r[0] == "docA")
    score_1 = next(r[1] for r in res_1 if r[0] == "docA")

    assert score_60 != pytest.approx(score_1)
    assert score_1 == pytest.approx(0.5 / 2.0)


def test_fuse_rrf_weights_change_ordering():
    ranked_lists = {"dense": ["docA"], "sparse": ["docB"]}

    equal = fuse_rrf(ranked_lists, rrf_k=60, weights={"dense": 0.5, "sparse": 0.5})
    assert [r[0] for r in equal] == ["docA", "docB"]  # tie -> id ascending

    sparse_heavy = fuse_rrf(ranked_lists, rrf_k=60, weights={"dense": 0.1, "sparse": 0.9})
    assert [r[0] for r in sparse_heavy] == ["docB", "docA"]


def test_fuse_rrf_missing_weight_raises():
    """A source with no weight must not silently default to 1.0."""
    with pytest.raises(ValueError, match="No weight supplied"):
        fuse_rrf(
            {"dense": ["docA"], "sparse": ["docB"]},
            rrf_k=60,
            weights={"dense": 0.5},
        )


def test_fuse_rrf_duplicate_id_in_one_list_uses_best_rank():
    """If a source repeats an id, the better rank wins, not the later one."""
    res = fuse_rrf({"dense": ["docA", "docB", "docA"]}, rrf_k=0, weights={"dense": 1.0})
    by_id = {r[0]: r for r in res}
    assert by_id["docA"][2]["dense"] == 1
    assert by_id["docA"][1] == pytest.approx(1.0 / 1.0)


def test_fuse_rrf_negative_k_raises():
    with pytest.raises(ValueError, match="rrf_k must be >= 0"):
        fuse_rrf({"dense": ["docA"]}, rrf_k=-1, weights={"dense": 1.0})


def test_fuse_rrf_is_deterministic():
    ranked_lists = {"dense": ["b", "a", "c"], "sparse": ["c", "a", "b"]}
    weights = {"dense": 0.5, "sparse": 0.5}
    first = fuse_rrf(ranked_lists, rrf_k=60, weights=weights)
    for _ in range(5):
        assert fuse_rrf(ranked_lists, rrf_k=60, weights=weights) == first


# -------------------------------------------------------- collapse_to_best ---

def test_collapse_keeps_best_ranked_member_of_each_group():
    ranked = ["c1_summary", "c2_raw", "c1_raw", "c1_question_0"]
    group_of = {
        "c1_summary": "c1",
        "c1_raw": "c1",
        "c1_question_0": "c1",
        "c2_raw": "c2",
    }
    order, winner = collapse_to_best(ranked, group_of)

    assert order == ["c1", "c2"]
    assert winner["c1"] == "c1_summary"  # first occurrence wins
    assert winner["c2"] == "c2_raw"


def test_collapse_unknown_member_keeps_its_own_id():
    order, winner = collapse_to_best(["x", "y"], {})
    assert order == ["x", "y"]
    assert winner == {"x": "x", "y": "y"}


def test_representation_count_does_not_bias_ranking():
    """The reason collapse happens BEFORE fusion, not after.

    Chunk A is high-tier and has four representations, which land at ranks
    2, 3, 4 and 5. Chunk B is low-tier with a single representation at rank 1.

    Collapsing first: B is rank 1 and A is rank 2, so B wins - correct, B's best
    representation beat A's best representation.

    Fusing at representation level and merging afterwards by summing would give
    A four contributions against B's one, and A would win purely for being
    indexed more times. This test fails if that ever gets reintroduced.
    """
    group_of = {
        "B_raw": "B",
        "A_raw": "A",
        "A_summary": "A",
        "A_question_0": "A",
        "A_keywords": "A",
    }
    ranked = ["B_raw", "A_raw", "A_summary", "A_question_0", "A_keywords"]

    order, _ = collapse_to_best(ranked, group_of)
    fused = fuse_rrf({"dense": order}, rrf_k=60, weights={"dense": 1.0})

    assert [r[0] for r in fused] == ["B", "A"]

    # and the naive alternative really would have flipped it
    naive = fuse_rrf({"dense": ranked}, rrf_k=60, weights={"dense": 1.0})
    a_total = sum(r[1] for r in naive if r[0].startswith("A"))
    b_total = sum(r[1] for r in naive if r[0].startswith("B"))
    assert a_total > b_total


# ---------------------------------------------------- real config contract ---

def test_hybrid_weights_sum_to_one():
    """Fused scores must be comparable - hybrid weights must sum to 1.0."""
    cfg = Config.load(CONFIG_PATH)
    weights = cfg.raw["funnel"]["hybrid"]["weights"]

    total = sum(float(v) for v in weights.values())
    assert total == pytest.approx(1.0), f"weights sum to {total}, not 1.0"
    assert set(weights) == {"dense", "sparse"}, f"weights has sources {sorted(weights)}"


# ------------------------------------------------------------ query_hybrid ---

def _payload(chunk_id, kind, text, rep_id=None):
    return {
        "chunk_id": chunk_id,
        "rep_id": rep_id or f"{chunk_id}_{kind}",
        "kind": kind,
        "tier": "high",
        "text": text,
        "doc_id": "doc_1",
        "doc_class": "digital_prose",
    }


def _store(dense_hits, sparse_hits, cfg, raw_payloads=None):
    """A VectorStore with only the pieces query_hybrid touches."""
    vs = VectorStore.__new__(VectorStore)
    vs.cfg = cfg
    vs.embedding_client = MagicMock()
    vs.embedding_client.encode.return_value = [0.0] * 384
    vs.sparse_client = MagicMock()
    vs.sparse_client.encode.return_value = MagicMock()
    vs.search_dense = MagicMock(return_value=dense_hits)
    vs.search_sparse = MagicMock(return_value=sparse_hits)
    if raw_payloads is not None:
        vs._fetch_raw_payloads = MagicMock(return_value=raw_payloads)
    return vs


def test_query_hybrid_requires_a_config():
    """It must not quietly fall back to loading config/default.yaml itself."""
    vs = _store([], [], cfg=None, raw_payloads={})
    with pytest.raises(ValueError, match="needs a config"):
        vs.query_hybrid("anything")



def test_query_hybrid_unsupported_fusion_raises():
    cfg = {"funnel": {"hybrid": {"fusion": "linear_sum"}}}
    vs = _store([], [], cfg=cfg, raw_payloads={})
    with pytest.raises(ValueError, match="Unsupported fusion mode 'linear_sum'"):
        vs.query_hybrid("text")


def test_query_hybrid_rejects_weights_that_do_not_sum_to_one():
    cfg = {
        "funnel": {
            "hybrid": {
                "fusion": "rrf",
                "weights": {"dense": 0.8, "sparse": 0.8},
            }
        }
    }
    vs = _store([], [], cfg=cfg, raw_payloads={})
    with pytest.raises(ValueError, match="not 1.0"):
        vs.query_hybrid("text")


def test_query_hybrid_config_values_reach_the_search_and_the_fusion(monkeypatch):
    """rrf_k, dense_limit and sparse_limit must come from config, not defaults.

    Every number below is distinct, so reading the wrong field cannot pass.
    """
    seen = {}
    from adaptive_retriever import vectorstore as vs_mod

    real_fuse = vs_mod.fuse_rrf

    def spy(ranked_lists, rrf_k, weights, **kw):
        seen["rrf_k"] = rrf_k
        seen["weights"] = weights
        seen["ranked_lists"] = ranked_lists
        return real_fuse(ranked_lists, rrf_k=rrf_k, weights=weights, **kw)

    monkeypatch.setattr(vs_mod, "fuse_rrf", spy)

    cfg = {
        "funnel": {
            "hybrid": {
                "fusion": "rrf",
                "rrf_k": 7,
                "dense_limit": 23,
                "sparse_limit": 41,
                "weights": {"dense": 0.75, "sparse": 0.25},
            }
        }
    }

    # mock similarity scores are deliberately NOT equal to any weight, so a bug
    # that used the similarity score in place of the weight would be caught
    dense_hits = [("c1_raw", _payload("c1", "raw", "chunk one"), 0.61)]
    sparse_hits = [("c2_raw", _payload("c2", "raw", "chunk two"), 0.58)]
    raw = {
        "c1": _payload("c1", "raw", "chunk one"),
        "c2": _payload("c2", "raw", "chunk two"),
    }
    vs = _store(dense_hits, sparse_hits, cfg=cfg, raw_payloads=raw)

    candidates = vs.query_hybrid("query text")

    assert seen["rrf_k"] == 7
    assert seen["weights"] == {"dense": 0.75, "sparse": 0.25}

    vs.search_dense.assert_called_once_with([0.0] * 384, limit=23, query_filter=None)
    assert vs.search_sparse.call_args.kwargs["limit"] == 41

    assert [c.chunk_id for c in candidates] == ["c1", "c2"]
    assert candidates[0].fused_score == pytest.approx(0.75 / 8.0)
    assert candidates[1].fused_score == pytest.approx(0.25 / 8.0)


def test_query_hybrid_returns_each_chunk_once():
    """Three representations of one chunk must not occupy three result slots."""
    cfg = {
        "funnel": {
            "hybrid": {
                "fusion": "rrf",
                "rrf_k": 60,
                "weights": {"dense": 0.5, "sparse": 0.5},
            }
        }
    }
    dense_hits = [
        ("c1_summary", _payload("c1", "summary", "a summary of chunk one"), 0.9),
        ("c1_raw", _payload("c1", "raw", "chunk one raw text"), 0.8),
        ("c1_question_0", _payload("c1", "question", "what is chunk one?"), 0.7),
        ("c2_raw", _payload("c2", "raw", "chunk two raw text"), 0.6),
    ]
    raw = {
        "c1": _payload("c1", "raw", "chunk one raw text"),
        "c2": _payload("c2", "raw", "chunk two raw text"),
    }
    vs = _store(dense_hits, [], cfg=cfg, raw_payloads=raw)

    candidates = vs.query_hybrid("text", limit=10)

    assert [c.chunk_id for c in candidates] == ["c1", "c2"]
    assert len({c.chunk_id for c in candidates}) == len(candidates)


def test_query_hybrid_returns_raw_chunk_text_not_the_matched_summary():
    """The generator must never cite a model-written summary as source text."""
    cfg = {
        "funnel": {
            "hybrid": {
                "fusion": "rrf",
                "rrf_k": 60,
                "weights": {"dense": 0.5, "sparse": 0.5},
            }
        }
    }
    dense_hits = [("c1_summary", _payload("c1", "summary", "MODEL WRITTEN SUMMARY"), 0.9)]
    raw = {"c1": _payload("c1", "raw", "THE ACTUAL SOURCE SENTENCE.")}
    vs = _store(dense_hits, [], cfg=cfg, raw_payloads=raw)

    c = vs.query_hybrid("text")[0]

    assert c.text == "THE ACTUAL SOURCE SENTENCE."
    assert "SUMMARY" not in c.text
    assert c.metadata["matched_kind"] == "summary"
    assert c.metadata["matched_rep_id"] == "c1_summary"
    assert c.rep_id == "c1_summary"


def test_query_hybrid_reports_which_representation_won_per_source():
    cfg = {
        "funnel": {
            "hybrid": {
                "fusion": "rrf",
                "rrf_k": 60,
                "weights": {"dense": 0.5, "sparse": 0.5},
            }
        }
    }
    dense_hits = [("c1_summary", _payload("c1", "summary", "summary text"), 0.9)]
    sparse_hits = [("c1_keywords", _payload("c1", "keywords", "kw1 kw2"), 0.8)]
    raw = {"c1": _payload("c1", "raw", "raw text")}
    vs = _store(dense_hits, sparse_hits, cfg=cfg, raw_payloads=raw)

    c = vs.query_hybrid("text")[0]

    assert c.metadata["winning_rep_by_source"] == {
        "dense": "c1_summary",
        "sparse": "c1_keywords",
    }
    assert c.metadata["source_ranks"] == {"dense": 1, "sparse": 1}


def test_query_hybrid_raises_when_a_point_has_no_chunk_id():
    cfg = {
        "funnel": {
            "hybrid": {
                "fusion": "rrf",
                "rrf_k": 60,
                "weights": {"dense": 0.5, "sparse": 0.5},
            }
        }
    }
    broken = {"rep_id": "orphan", "kind": "raw", "text": "x"}
    vs = _store([("orphan", broken, 0.9)], [], cfg=cfg, raw_payloads={})

    with pytest.raises(RuntimeError, match="no chunk_id"):
        vs.query_hybrid("text")
