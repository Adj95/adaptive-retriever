"""Gated reranking (stage 3 of retrieval funnel)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from adaptive_retriever.models import RetrievalCandidate

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Reranker:
    """Cross-encoder reranker wrapping sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = _resolve_device(device)
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

    def ensure_ready(self) -> None:
        """Eagerly load the model to fail at startup rather than mid-run."""
        self._ensure_loaded()

    def get_device_str(self) -> str:
        self._ensure_loaded()
        if hasattr(self._model, "model") and hasattr(self._model.model, "device"):
            return str(self._model.model.device)
        return str(self.device)

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            self.last_rerank_breakdown = {"tokenize_ms": 0.0, "forward_ms": 0.0, "sort_ms": 0.0, "device": self.get_device_str()}
            return []
        self._ensure_loaded()
        pairs = [(query, t) for t in texts]

        import time
        import torch

        # 1. Tokenize / Batch preparation
        t0 = time.perf_counter()
        features = self._model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        t1 = time.perf_counter()
        tokenize_ms = round((t1 - t0) * 1000, 2)

        # 2. Forward pass on CUDA / CPU
        t2 = time.perf_counter()
        target_device = getattr(self._model.model, "device", self.device)
        features = {k: v.to(target_device) for k, v in features.items()}
        with torch.no_grad():
            outputs = self._model.model(**features)
            logits = outputs.logits.squeeze(-1)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            scores = logits.cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
        t3 = time.perf_counter()
        forward_ms = round((t3 - t2) * 1000, 2)

        assert len(scores) == len(texts), (
            f"Reranker returned {len(scores)} scores for {len(texts)} texts"
        )

        self.last_rerank_breakdown = {
            "tokenize_ms": tokenize_ms,
            "forward_ms": forward_ms,
            "sort_ms": 0.0,
            "device": str(target_device),
        }
        return scores


def _get_rerank_cfg(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        raw_cfg = {"funnel": {"rerank": {}}}
    elif hasattr(cfg, "raw"):
        raw_cfg = cfg.raw
    elif isinstance(cfg, dict):
        if "funnel" in cfg:
            raw_cfg = cfg
        else:
            raise ValueError(
                f"Invalid config dict: must contain top-level 'funnel' key. Got keys: {list(cfg.keys())}"
            )
    else:
        raise ValueError(
            f"Invalid config type '{type(cfg).__name__}'. Expected Config instance or dict containing 'funnel'."
        )
    return raw_cfg.get("funnel", {}).get("rerank", {})


def should_rerank(
    candidates: List[RetrievalCandidate],
    cfg: Any = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Pure gate logic deciding if reranking should fire based on relative score margin."""
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
                "raw_margin": 0.0,
                "threshold": gate_score_margin,
                "reason": "no_candidates",
            }

        top_score = (
            candidates[0].fused_score
            if candidates[0].fused_score is not None
            else 0.0
        )
        if top_score <= 0.0:
            return True, {
                "mode": "gated",
                "should_rerank": True,
                "reason": "zero_top_score",
                "margin": 0.0,
                "raw_margin": 0.0,
                "threshold": gate_score_margin,
                "top_score": top_score,
                "cutoff_score": 0.0,
                "output_size": output_size,
            }

        cutoff_idx = min(output_size, len(candidates)) - 1
        cutoff_score = (
            candidates[cutoff_idx].fused_score
            if candidates[cutoff_idx].fused_score is not None
            else 0.0
        )
        raw_margin = float(top_score - cutoff_score)
        margin = float(raw_margin / top_score)
        fire = margin < gate_score_margin

        trace = {
            "mode": "gated",
            "should_rerank": fire,
            "margin": margin,
            "raw_margin": raw_margin,
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

    import time
    t_sort = time.perf_counter()
    scored_candidates.sort(
        key=lambda c: (
            -(c.rerank_score if c.rerank_score is not None else 0.0),
            -(c.fused_score if c.fused_score is not None else 0.0),
            c.rep_id or "",
        )
    )
    sort_ms = round((time.perf_counter() - t_sort) * 1000, 2)

    if hasattr(reranker, "last_rerank_breakdown") and isinstance(reranker.last_rerank_breakdown, dict):
        reranker.last_rerank_breakdown["sort_ms"] = sort_ms

    final_list = []
    for rank, cand in enumerate(scored_candidates[:output_size], 1):
        renumbered = cand.model_copy(update={"final_rank": rank})
        final_list.append(renumbered)

    return final_list
