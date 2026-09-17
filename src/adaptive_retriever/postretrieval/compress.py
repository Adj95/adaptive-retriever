"""Post-retrieval candidate text compression (Task A)."""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from adaptive_retriever.models import RetrievalCandidate


def split_sentences(text: str) -> List[str]:
    """Split text into sentence spans using punctuation and newlines while keeping spans verbatim."""
    if not text or not text.strip():
        return []

    raw_spans = re.split(r"(?<=[.!?])\s+|\n+", text)
    spans = []
    for s in raw_spans:
        st = s.strip()
        if st and st in text:
            spans.append(st)
    return spans


def cosine_sim(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def compress(
    query: str,
    candidates: List[RetrievalCandidate],
    cfg: Optional[Any] = None,
    llm: Optional[Any] = None,
    embedding_client: Optional[Any] = None,
) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
    """Drop or shorten candidate text that does not bear on the query.

    Cascade strategy:
      1. Sentence-level embedding similarity first.
      2. Fall to LLM extractor when best sentence similarity is below llm_extractor_if_below.

    Guarantees:
      - Verbatim substring enforcement: every surviving span in compressed candidate text
        must be a verbatim substring of the original candidate text.
      - Tracks and reports compression ratio per candidate.
      - mode: "none" is a pass-through.
    """
    comp_cfg = {}
    if cfg is not None:
        if hasattr(cfg, "raw"):
            comp_cfg = cfg.raw.get("postretrieval", {}).get("compression", {})
        elif isinstance(cfg, dict):
            comp_cfg = cfg.get("postretrieval", {}).get("compression", {})

    mode = comp_cfg.get("mode", "cascade")
    emb_threshold = float(comp_cfg.get("embeddings_threshold", 0.35))
    llm_trigger_below = float(comp_cfg.get("llm_extractor_if_below", 0.50))

    if mode == "none" or not candidates:
        trace = {
            "mode": mode,
            "input_count": len(candidates),
            "output_count": len(candidates),
            "overall_compression_ratio": 1.0,
            "details": [
                {
                    "chunk_id": c.chunk_id,
                    "char_in": len(c.text),
                    "char_out": len(c.text),
                    "compression_ratio": 1.0,
                    "tier_fired": "none",
                }
                for c in candidates
            ],
        }
        return candidates, trace

    # Ensure embedding client if embeddings or cascade mode
    if embedding_client is None and mode in ("cascade", "embeddings"):
        from adaptive_retriever.vectorstore import DenseEmbeddingModel
        embedding_client = DenseEmbeddingModel()

    q_emb = embedding_client.encode(query) if embedding_client else []

    # 1. Collect all sentence spans across all candidates into a single batch list
    all_spans = []
    cand_spans_map = []
    for cand in candidates:
        original_text = cand.text
        if not original_text:
            cand_spans_map.append([])
            continue
        spans = split_sentences(original_text)
        if not spans:
            spans = [original_text]
        cand_spans_map.append(spans)
        all_spans.extend(spans)

    print(f"[COMPRESS_CALL_SITE] len(candidates)={len(candidates)}, len(all_spans)={len(all_spans)}")

    # 2. Batch encode all sentence spans in a single vectorized call
    span_sim_map: Dict[str, float] = {}
    if len(q_emb) > 0 and embedding_client is not None and len(all_spans) > 0:
        if hasattr(embedding_client, "encode_batch"):
            all_embs = embedding_client.encode_batch(all_spans)
        else:
            all_embs = [embedding_client.encode(s) for s in all_spans]

        q_vec = np.array(q_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        all_arr = np.array(all_embs, dtype=np.float32)
        norms = np.linalg.norm(all_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        if q_norm > 0:
            sims = np.dot(all_arr / norms, (q_vec / q_norm).reshape(-1, 1)).flatten()
        else:
            sims = np.zeros(len(all_spans))

        for s, sim in zip(all_spans, sims):
            span_sim_map[s] = float(sim)

    compressed_candidates: List[RetrievalCandidate] = []
    candidate_details: List[Dict[str, Any]] = []
    tier_counts: Dict[str, int] = {
        "embeddings": 0,
        "llm_extractor": 0,
        "pass_through": 0,
    }

    for idx, cand in enumerate(candidates):
        original_text = cand.text
        char_in = len(original_text)
        if char_in == 0:
            candidate_details.append({
                "chunk_id": cand.chunk_id,
                "char_in": 0,
                "char_out": 0,
                "compression_ratio": 1.0,
                "tier_fired": "none",
            })
            compressed_candidates.append(cand)
            continue

        spans = cand_spans_map[idx]

        # Evaluate sentence similarities from batch map
        span_sims: List[Tuple[str, float]] = [(s, span_sim_map.get(s, 0.0)) for s in spans]
        best_sim = max((sim for _, sim in span_sims), default=0.0)

        tier_fired = "embeddings"
        selected_spans: List[str] = []

        if mode == "cascade" and best_sim < llm_trigger_below and llm is not None:
            # Tier 2: LLM Extractor
            tier_fired = "llm_extractor"
            tier_counts["llm_extractor"] += 1

            from adaptive_retriever.prompts import load_prompt
            prompt_template, prompt_ver = load_prompt("postretrieval/compress")
            prompt = prompt_template.format(context=original_text, query=query)
            resp = llm.complete(
                prompt,
                prompt_version=prompt_ver,
                max_tokens=256,
                stage="postretrieval_compress",
            )

            raw_extracted = [line.strip() for line in resp.splitlines() if line.strip()]
            for line in raw_extracted:
                if line in original_text:
                    selected_spans.append(line)
        else:
            # Tier 1: Embeddings threshold filtering
            tier_fired = "embeddings"
            tier_counts["embeddings"] += 1
            for s, sim in span_sims:
                if sim >= emb_threshold:
                    selected_spans.append(s)

        # Reconstruct compressed text preserving verbatim order in original_text
        if selected_spans:
            selected_spans.sort(key=lambda s: original_text.find(s))
            seen = set()
            ordered_spans = []
            for s in selected_spans:
                if s not in seen:
                    seen.add(s)
                    ordered_spans.append(s)
            compressed_text = "\n".join(ordered_spans)
        else:
            compressed_text = ""

        # MANDATORY VERBATIM SUBSTRING ASSERTION
        # Every surviving span in compressed_text must be a verbatim substring of original_text
        for span in selected_spans:
            assert span in original_text, f"Compression failed verbatim check! Span '{span}' is not a verbatim substring of original text."

        char_out = len(compressed_text)
        ratio = round(char_out / char_in, 4) if char_in > 0 else 1.0

        meta = dict(cand.metadata) if cand.metadata else {}
        meta["compression"] = {
            "char_in": char_in,
            "char_out": char_out,
            "compression_ratio": ratio,
            "tier_fired": tier_fired,
        }

        updated_cand = cand.model_copy(update={"text": compressed_text, "metadata": meta})
        compressed_candidates.append(updated_cand)

        candidate_details.append({
            "chunk_id": cand.chunk_id,
            "char_in": char_in,
            "char_out": char_out,
            "compression_ratio": ratio,
            "tier_fired": tier_fired,
        })

    total_in = sum(d["char_in"] for d in candidate_details)
    total_out = sum(d["char_out"] for d in candidate_details)
    overall_ratio = round(total_out / total_in, 4) if total_in > 0 else 1.0

    trace = {
        "mode": mode,
        "input_count": len(candidates),
        "output_count": len(compressed_candidates),
        "overall_compression_ratio": overall_ratio,
        "tier_counts": tier_counts,
        "details": candidate_details,
    }

    return compressed_candidates, trace
