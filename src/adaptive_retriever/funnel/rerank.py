"""Gated reranking (stage 3 of retrieval funnel)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from adaptive_retriever.models import RetrievalCandidate

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker wrapping sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as err:
            raise RuntimeError(
                "sentence-transformers is required for cross-encoder reranker but is not importable."
            ) from err

        logger.info("Loading reranker '%s' on %s", self.model_name, self.device)
        try:
            self._model = CrossEncoder(self.model_name, device=self.device)
        except Exception as err:
            raise RuntimeError(
                f"Failed to load reranker model '{self.model_name}': {err}"
            ) from err

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        self._ensure_loaded()
        pairs = [(query, t) for t in texts]
        raw_scores = self._model.predict(pairs)
        scores = [float(s) for s in raw_scores]
        assert len(scores) == len(texts), (
            f"Reranker returned {len(scores)} scores for {len(texts)} texts"
        )
        return scores


def _get_rerank_cfg(cfg: Any) -> Dict[str, Any]:
    if hasattr(cfg, "raw"):
        raw_cfg = cfg.raw
    elif isinstance(cfg, dict):
        if "funnel" in cfg:
            raw_cfg = cfg
        elif "rerank" in cfg:
            raw_cfg = {"funnel": cfg}
        else:
            raw_cfg = {"funnel": {"rerank": cfg}}
    else:
        raw_cfg = {"funnel": {"rerank": {}}}
    return raw_cfg.get("funnel", {}).get("rerank", {})


def should_rerank(
    candidates: List[RetrievalCandidate],
    cfg: Any = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Pure gate logic deciding if reranking should fire based on margin."""
    rerank_cfg = _get_rerank_cfg(cfg)
    mode = rerank_cfg.get("mode", "gated")
    gate_score_margin = float(rerank_cfg.get("gate_score_margin", 0.15))
    output_size = int(rerank_cfg.get("output_size", 5))

    if mode == "never":
        return False, {"mode": "never", "should_rerank": False, "reason": "mode_never"}
    elif mode == "always":
        return True, {"mode": "always", "should_rerank": True, "reason": "mode_always"}
    elif mode == "gated":
        if not candidates:
            return False, {
                "mode": "gated",
                "should_rerank": False,
                "margin": 0.0,
                "threshold": gate_score_margin,
                "reason": "no_candidates",
            }

        top_score = candidates[0].fused_score if candidates[0].fused_score is not None else 0.0
        cutoff_idx = min(output_size, len(candidates)) - 1
        cutoff_score = (
            candidates[cutoff_idx].fused_score
            if candidates[cutoff_idx].fused_score is not None
            else 0.0
        )
        margin = float(top_score - cutoff_score)
        fire = margin < gate_score_margin

        trace = {
            "mode": "gated",
            "should_rerank": fire,
            "margin": margin,
            "threshold": gate_score_margin,
            "top_score": top_score,
            "cutoff_score": cutoff_score,
            "output_size": output_size,
        }
        return fire, trace
    else:
        raise ValueError(
            f"Invalid rerank mode '{mode}'. Expected 'gated', 'always', or 'never'."
        )


def apply_rerank(
    query: str,
    candidates: List[RetrievalCandidate],
    reranker: Reranker,
    cfg: Any = None,
) -> List[RetrievalCandidate]:
    """Score top input_size candidates with cross-encoder, sort, truncate to output_size, renumber final_rank."""
    if not candidates:
        return []

    rerank_cfg = _get_rerank_cfg(cfg)
    input_size = int(rerank_cfg.get("input_size", 100))
    output_size = int(rerank_cfg.get("output_size", 5))

    to_score = list(candidates[:input_size])
    texts = [c.text for c in to_score]
    scores = reranker.score(query, texts)
    assert len(scores) == len(to_score), (
        f"Reranker returned {len(scores)} scores for {len(to_score)} candidates"
    )

    scored_candidates = []
    for cand, score in zip(to_score, scores):
        updated = cand.model_copy(update={"rerank_score": score})
        scored_candidates.append(updated)

    scored_candidates.sort(
        key=lambda c: (
            -(c.rerank_score if c.rerank_score is not None else 0.0),
            -(c.fused_score if c.fused_score is not None else 0.0),
            c.rep_id or "",
        )
    )

    final_list = []
    for rank, cand in enumerate(scored_candidates[:output_size], 1):
        renumbered = cand.model_copy(update={"final_rank": rank})
        final_list.append(renumbered)

    return final_list
