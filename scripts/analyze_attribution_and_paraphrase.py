import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from adaptive_retriever.config import Config
from adaptive_retriever.models import RetrievalCandidate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_sentences(text: str) -> List[str]:
    """Split text into normalized non-empty sentences."""
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def get_words(text: str) -> List[str]:
    """Extract lowercased word tokens from text."""
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(words: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Extract word-level n-grams from token list."""
    if len(words) < n:
        return set()
    return set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def fast_attribute_sentence(
    sentence: str,
    candidates: List[RetrievalCandidate],
    n: int = 8,
) -> Optional[str]:
    """Attribute sentence to candidate chunk with maximum n-gram overlap."""
    sent_words = get_words(sentence)
    if len(sent_words) < n:
        return None
    sent_ngrams = get_ngrams(sent_words, n)
    if not sent_ngrams:
        return None

    best_chunk_id = None
    max_count = 0

    for cand in candidates:
        cand_words = get_words(cand.text)
        if len(cand_words) < n:
            continue
        cand_ngrams = get_ngrams(cand_words, n)
        overlap_count = len(sent_ngrams & cand_ngrams)
        if overlap_count > max_count:
            max_count = overlap_count
            best_chunk_id = cand.chunk_id

    return best_chunk_id if max_count > 0 else None


def main():
    cfg = Config.load("config/default.yaml")
    raw_path = Path("reports/citation_comparison_raw.jsonl")
    if not raw_path.exists():
        logger.error(f"{raw_path} not found!")
        return

    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} records from {raw_path}")

    # 1. Load candidate pools from cache
    cand_cache_path = Path("reports/candidate_pools_cache.json")
    if not cand_cache_path.exists():
        logger.error(f"{cand_cache_path} does not exist!")
        return

    logger.info(f"Loading candidate pools from {cand_cache_path}...")
    with open(cand_cache_path, "r", encoding="utf-8") as f:
        raw_pools = json.load(f)

    candidate_pools = {}
    for q, c_list in raw_pools.items():
        candidate_pools[q] = [
            RetrievalCandidate(chunk_id=c["chunk_id"], text=c["text"], score=c.get("score", 0.0))
            for c in c_list
        ]

    # 2. Load Sentence Transformer on GPU / CPU
    dense_model_name = cfg.raw.get("models", {}).get("dense_encoder", "BAAI/bge-small-en-v1.5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SentenceTransformer '{dense_model_name}' on {device}...")
    embedder = SentenceTransformer(dense_model_name, device=device)

    supported_sims = []
    unsupported_sims = []

    ngram_sweeps = {
        8: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
        6: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
        5: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
        4: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
    }

    logger.info("Computing per-query sentence similarity distribution and n-gram sweeps...")
    # Pre-collect all unique sentences to encode in batch
    all_ans_sents = []
    ans_sent_indices = [] # (rec_idx, local_sent_idx)
    
    query_cand_sents = {} # query -> list of candidate sentences
    
    # 1. Shuffled mapping: query_i receives candidate pool from query_(i + 17) % N
    N = len(records)
    shuffled_queries = [records[(i + 17) % N]["query"] for i in range(N)]

    # Collect candidate sentences ONLY for active real and shuffled queries
    active_queries = set([rec["query"] for rec in records] + shuffled_queries)
    query_cand_sents = {} # query -> list of candidate sentences

    for rec_idx, rec in enumerate(records):
        query = rec["query"]
        text_ans = rec["text_ans"]
        ans_sents = split_sentences(text_ans)
        for s_idx, s in enumerate(ans_sents):
            all_ans_sents.append(s)
            ans_sent_indices.append((rec_idx, s_idx))
            
    query_cand_texts = {}
    for q in active_queries:
        cands = candidate_pools.get(q, [])
        query_cand_texts[q] = [c.text for c in cands if c.text]

    logger.info(f"Encoding {len(all_ans_sents)} total answer sentences...")
    ans_embeddings_flat = embedder.encode(all_ans_sents, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    
    unique_cand_texts = list(set([t for texts in query_cand_texts.values() for t in texts if t]))
    text_to_idx = {t: i for i, t in enumerate(unique_cand_texts)}
    
    logger.info(f"Encoding {len(unique_cand_texts)} unique candidate chunk texts...")
    unique_cand_embeds = embedder.encode(unique_cand_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    cand_embeddings_by_query = {}
    for q, texts in query_cand_texts.items():
        if texts:
            indices = [text_to_idx[t] for t in texts if t in text_to_idx]
            if indices:
                cand_embeddings_by_query[q] = unique_cand_embeds[indices]

    # Reconstruct ans_embeds by query (Real vs Shuffled)
    # Shuffled mapping: query_i receives candidate pool from query_(i + 17) % N
    N = len(records)
    shuffled_queries = [records[(i + 17) % N]["query"] for i in range(N)]

    query_ans_max_sims_real = {}
    query_ans_max_sims_shuffled = {}
    
    curr_idx = 0
    for rec_idx, rec in enumerate(records):
        query_real = rec["query"]
        query_shuffled = shuffled_queries[rec_idx]
        ans_sents = split_sentences(rec["text_ans"])
        n_ans = len(ans_sents)
        if n_ans == 0:
            query_ans_max_sims_real[id(rec)] = []
            query_ans_max_sims_shuffled[id(rec)] = []
            continue
        
        rec_ans_embeds = ans_embeddings_flat[curr_idx : curr_idx + n_ans]
        curr_idx += n_ans
        
        # Real
        cand_embeds_r = cand_embeddings_by_query.get(query_real)
        if cand_embeds_r is not None and len(cand_embeds_r) > 0:
            sim_matrix_r = np.dot(rec_ans_embeds, cand_embeds_r.T)
            query_ans_max_sims_real[id(rec)] = np.max(sim_matrix_r, axis=1).tolist()
        else:
            query_ans_max_sims_real[id(rec)] = [0.0] * n_ans

        # Shuffled
        cand_embeds_s = cand_embeddings_by_query.get(query_shuffled)
        if cand_embeds_s is not None and len(cand_embeds_s) > 0:
            sim_matrix_s = np.dot(rec_ans_embeds, cand_embeds_s.T)
            query_ans_max_sims_shuffled[id(rec)] = np.max(sim_matrix_s, axis=1).tolist()
        else:
            query_ans_max_sims_shuffled[id(rec)] = [0.0] * n_ans

    supported_sims_real = []
    unsupported_sims_real = []
    unsupported_sims_shuffled = []

    ngram_sweeps_real = {
        8: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
        6: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
        5: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
        4: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
    }
    ngram_sweeps_shuffled = {
        8: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
        6: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
        5: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
        4: {"answers_with_cit": 0, "total_cits": 0, "unsupported_sents": 0, "total_sents": 0},
    }

    for rec_idx, rec in enumerate(records):
        query_real = rec["query"]
        query_shuffled = shuffled_queries[rec_idx]
        text_ans = rec["text_ans"]
        
        cands_r = candidate_pools.get(query_real, [])
        cands_s = candidate_pools.get(query_shuffled, [])
        ans_sents = split_sentences(text_ans)
        
        max_sims_r = query_ans_max_sims_real[id(rec)]
        max_sims_s = query_ans_max_sims_shuffled[id(rec)]

        # Precompute candidate n-grams
        def build_ngrams_by_n(cands_list):
            cw_list = [get_words(c.text) for c in cands_list]
            return {
                n_val: [
                    (cands_list[i].chunk_id, get_ngrams(cw, n_val))
                    for i, cw in enumerate(cw_list)
                ]
                for n_val in [8, 6, 5, 4]
            }

        cand_ngrams_r = build_ngrams_by_n(cands_r)
        cand_ngrams_s = build_ngrams_by_n(cands_s)

        def get_attr(sentence: str, n_val: int, cand_ngrams_dict) -> Optional[str]:
            sent_words = get_words(sentence)
            if len(sent_words) < n_val:
                return None
            s_ngrams = get_ngrams(sent_words, n_val)
            if not s_ngrams:
                return None
            best_cid = None
            max_c = 0
            for cid, c_ngrams in cand_ngrams_dict[n_val]:
                if not c_ngrams:
                    continue
                cnt = len(s_ngrams & c_ngrams)
                if cnt > max_c:
                    max_c = cnt
                    best_cid = cid
            return best_cid if max_c > 0 else None

        # Real Sentence similarity split by n=8 attribution
        for s_idx, s in enumerate(ans_sents):
            sim_val_r = float(max_sims_r[s_idx])
            sim_val_s = float(max_sims_s[s_idx])
            attr_r = get_attr(s, 8, cand_ngrams_r)
            if attr_r:
                supported_sims_real.append(sim_val_r)
            else:
                unsupported_sims_real.append(sim_val_r)
            
            unsupported_sims_shuffled.append(sim_val_s)

        # n-gram sweeps
        for n_val in [8, 6, 5, 4]:
            # Real
            seen_cits_r = set()
            unsupp_cnt_r = 0
            for s in ans_sents:
                attr = get_attr(s, n_val, cand_ngrams_r)
                if attr:
                    seen_cits_r.add(attr)
                else:
                    unsupp_cnt_r += 1
            if seen_cits_r:
                ngram_sweeps_real[n_val]["answers_with_cit"] += 1
            ngram_sweeps_real[n_val]["total_cits"] += len(seen_cits_r)
            ngram_sweeps_real[n_val]["unsupported_sents"] += unsupp_cnt_r
            ngram_sweeps_real[n_val]["total_sents"] += len(ans_sents)

            # Shuffled Control
            seen_cits_s = set()
            unsupp_cnt_s = 0
            for s in ans_sents:
                attr = get_attr(s, n_val, cand_ngrams_s)
                if attr:
                    seen_cits_s.add(attr)
                else:
                    unsupp_cnt_s += 1
            if seen_cits_s:
                ngram_sweeps_shuffled[n_val]["answers_with_cit"] += 1
            ngram_sweeps_shuffled[n_val]["total_cits"] += len(seen_cits_s)
            ngram_sweeps_shuffled[n_val]["unsupported_sents"] += unsupp_cnt_s
            ngram_sweeps_shuffled[n_val]["total_sents"] += len(ans_sents)

    def calc_percentiles(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0, "mean": 0.0, "count": 0}
        a = np.array(arr)
        return {
            "min": round(float(np.min(a)), 4),
            "p25": round(float(np.percentile(a, 25)), 4),
            "p50": round(float(np.percentile(a, 50)), 4),
            "p75": round(float(np.percentile(a, 75)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "mean": round(float(np.mean(a)), 4),
            "count": len(arr),
        }

    def format_sweep(sweep_dict):
        res = {}
        for n, data in sweep_dict.items():
            tot_s = data["total_sents"]
            res[f"n={n}"] = {
                "pct_answers_with_at_least_one_citation": round((data["answers_with_cit"] / N) * 100.0, 2),
                "mean_citations_per_answer": round(data["total_cits"] / N, 4),
                "pct_unsupported_sentences": round((data["unsupported_sents"] / tot_s * 100.0) if tot_s else 0.0, 2),
                "total_sentences": tot_s,
                "unsupported_sentences_count": data["unsupported_sents"],
            }
        return res

    output_report = {
        "sentence_similarity_distribution": {
            "supported_sentences_real_n8": calc_percentiles(supported_sims_real),
            "unsupported_sentences_real_n8": calc_percentiles(unsupported_sims_real),
            "unsupported_sentences_shuffled_control": calc_percentiles(unsupported_sims_shuffled),
        },
        "ngram_threshold_sweep": {
            "real_candidate_pool": format_sweep(ngram_sweeps_real),
            "shuffled_candidate_pool_control": format_sweep(ngram_sweeps_shuffled),
        },
    }

    out_path = Path("reports/paraphrase_attribution_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)

    logger.info(f"Analysis saved to {out_path}")
    print(json.dumps(output_report, indent=2))


if __name__ == "__main__":
    main()
