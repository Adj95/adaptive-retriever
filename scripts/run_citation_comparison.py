"""Script to evaluate citation mechanisms (Task 4, Phase 5) with batched LLM generation,
zero-shot saved answer reuse, and smoke-testing capability.
"""

import argparse
import gc
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Any

import torch

from adaptive_retriever.config import Config
from adaptive_retriever.funnel.pipeline import run_funnel
from adaptive_retriever.funnel.rerank import Reranker
from adaptive_retriever.generation.generate import (
    attribute_sentence_to_chunks,
    parse_citations_defensive,
)
from adaptive_retriever.llm import get_llm_client
from adaptive_retriever.prompts import load_prompt
from adaptive_retriever.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eval_items() -> List[Dict[str, Any]]:
    eval_path = Path("data/eval/eval_set.jsonl")
    items = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get("provenance") != "synthetic_rejected" and "rejected_by" not in data:
                    items.append(data)
    return items


def load_saved_zero_shot_answers() -> Dict[str, str]:
    report_path = Path("reports/eval_citation_rate_report.json")
    if not report_path.exists():
        raise FileNotFoundError(f"{report_path} not found.")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    query_to_ans = {}
    for detail in data.get("per_query_details", []):
        query_to_ans[detail["query"].strip()] = detail["text_ans"]
    return query_to_ans


def format_context_prompt(prompt_template: str, query: str, candidates: List[Any]) -> str:
    context_blocks = []
    current_chars = 0
    max_context_chars = 6000
    for cand in candidates:
        if current_chars >= max_context_chars and len(context_blocks) >= 3:
            break
        cand_text = cand.text[:1500] if len(cand.text) > 1500 else cand.text
        context_blocks.append(f"[{cand.chunk_id}]: {cand_text}")
        current_chars += len(cand_text)
    context_str = "\n\n".join(context_blocks)
    return prompt_template.format(context=context_str, query=query)


def evaluate_text_citations(text_ans: str, candidates: List[Any]) -> Dict[str, Any]:
    valid_chunk_ids = {c.chunk_id for c in candidates}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_ans) if s.strip()]
    
    # 1. Model-emitted citation parsing
    model_citations = parse_citations_defensive(text_ans, valid_chunk_ids, strict=False)
    
    # 2. Post-hoc attribution citation parsing
    attr_citations = []
    seen_attr = set()
    unsupported_sents = 0
    for s in sentences:
        attr_chunk = attribute_sentence_to_chunks(s, candidates, min_n_gram=8)
        if attr_chunk:
            if attr_chunk not in seen_attr:
                seen_attr.add(attr_chunk)
                attr_citations.append(attr_chunk)
        else:
            unsupported_sents += 1

    return {
        "sentences_count": len(sentences),
        "unsupported_sentences_count": unsupported_sents,
        "model_citations": model_citations,
        "attribution_citations": attr_citations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-item smoke test and exit")
    parser.add_argument("--full-run", action="store_true", help="Execute full 169 query run")
    args = parser.parse_args()

    # Default to smoke-test if neither specified
    if not args.smoke_test and not args.full_run:
        args.smoke_test = True

    cfg = Config.load("config/default.yaml")
    items = load_eval_items()
    logger.info(f"Loaded {len(items)} non-rejected evaluation queries.")

    # 1. Candidate Retrieval
    vs = VectorStore.from_config(cfg)
    reranker = Reranker(
        model_name=cfg.raw.get("models", {}).get("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        device=cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"),
    )

    logger.info("Stage 1: Retrieving candidates for evaluation queries...")
    retrieved_pool = []
    eval_items = items[:2] if args.smoke_test else items
    for idx, item in enumerate(eval_items):
        query = item["query"]
        candidates, trace = run_funnel(query, vs, cfg, reranker=reranker)
        confidence = trace.get("confidence", 1.0)
        retrieved_pool.append({
            "query": query,
            "candidates": candidates,
            "confidence": confidence,
        })

    # Free reranker per requirement 4
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Reranker freed and CUDA memory cleared.")

    # 2. Zero-Shot Evaluation (using saved answers from reports/eval_citation_rate_report.json)
    saved_answers = load_saved_zero_shot_answers()
    
    zs_model_with_cit = 0
    zs_model_total_cits = 0
    zs_attr_with_cit = 0
    zs_attr_total_cits = 0
    zs_total_sents = 0
    zs_unsupported_sents = 0

    for entry in retrieved_pool:
        q = entry["query"]
        text_ans = saved_answers.get(q.strip(), "")
        eval_res = evaluate_text_citations(text_ans, entry["candidates"])
        
        cits_m = eval_res["model_citations"]
        if cits_m:
            zs_model_with_cit += 1
        zs_model_total_cits += len(cits_m)

        cits_a = eval_res["attribution_citations"]
        if cits_a:
            zs_attr_with_cit += 1
        zs_attr_total_cits += len(cits_a)

        zs_total_sents += eval_res["sentences_count"]
        zs_unsupported_sents += eval_res["unsupported_sentences_count"]

    # 3. Few-Shot Generation Pass
    llm = get_llm_client(cfg)
    llm.max_new_tokens = 192
    llm.truncation_retry = False
    llm.batch_size = 8
    prompt_template, prompt_version = load_prompt("generation/answer_with_citations_fewshot")
    
    batch_size = getattr(llm, "batch_size", 16)
    max_new_tokens = getattr(llm, "max_new_tokens", 192)

    fewshot_prompts = [
        format_context_prompt(prompt_template, entry["query"], entry["candidates"])
        for entry in retrieved_pool
    ]

    logger.info(f"Running LLM generation for {len(fewshot_prompts)} items (batch_size={batch_size}, max_new_tokens={max_new_tokens})...")
    
    t_gen_start = time.time()
    responses = llm.complete_batch(fewshot_prompts, prompt_version=prompt_version, stage="generation")
    t_gen_end = time.time()
    
    gen_elapsed = t_gen_end - t_gen_start
    per_item_sec = gen_elapsed / len(fewshot_prompts)
    
    # 169 queries processed in batches of 16 -> 11 batches total
    num_batches_169 = (169 + batch_size - 1) // batch_size
    per_batch_sec = gen_elapsed / ((len(fewshot_prompts) + batch_size - 1) // batch_size)
    projected_total_sec = num_batches_169 * per_batch_sec
    projected_total_min = projected_total_sec / 60.0

    if args.smoke_test:
        print("\n========================================================================")
        print("SMOKE-TEST PROJECTION REPORT (2 items tested)")
        print("========================================================================")
        print(f"Batch size actually used  : {batch_size}")
        print(f"max_new_tokens used      : {max_new_tokens}")
        print(f"Precision used           : fp32 (torch.float32)")
        print(f"Smoke-test wall-clock    : {gen_elapsed:.2f} s for {len(fewshot_prompts)} items")
        print(f"Per-item cost            : {per_item_sec:.2f} s/item")
        print(f"Projected 169 wall-clock : {projected_total_sec:.2f} s ({projected_total_min:.2f} minutes)")
        if projected_total_min > 45.0:
            print("WARNING: Projected total runtime EXCEEDS 45 minutes threshold!")
        else:
            print("Projection is WITHIN 45 minutes threshold.")
        print("========================================================================\n")
        return

    # If full run: stream raw items to reports/citation_comparison_raw.jsonl
    raw_path = Path("reports/citation_comparison_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f_out:
        fs_model_with_cit = 0
        fs_model_total_cits = 0
        fs_attr_with_cit = 0
        fs_attr_total_cits = 0
        fs_total_sents = 0
        fs_unsupported_sents = 0

        t_loop_start = time.time()
        for idx, (entry, resp) in enumerate(zip(retrieved_pool, responses)):
            text_ans = resp.text.strip()
            eval_res = evaluate_text_citations(text_ans, entry["candidates"])
            
            cits_m = eval_res["model_citations"]
            if cits_m:
                fs_model_with_cit += 1
            fs_model_total_cits += len(cits_m)

            cits_a = eval_res["attribution_citations"]
            if cits_a:
                fs_attr_with_cit += 1
            fs_attr_total_cits += len(cits_a)

            fs_total_sents += eval_res["sentences_count"]
            fs_unsupported_sents += eval_res["unsupported_sentences_count"]

            record = {
                "idx": idx + 1,
                "query": entry["query"],
                "text_ans": text_ans,
                "model_citations": cits_m,
                "attribution_citations": cits_a,
                "sentences_count": eval_res["sentences_count"],
                "unsupported_sentences_count": eval_res["unsupported_sentences_count"],
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

            if (idx + 1) % 20 == 0 or (idx + 1) == len(retrieved_pool):
                elapsed_so_far = time.time() - t_loop_start
                avg_sec = elapsed_so_far / (idx + 1)
                rem_sec = avg_sec * (len(retrieved_pool) - (idx + 1))
                logger.info(f"Progress: {idx + 1}/{len(retrieved_pool)} completed. ETA: {rem_sec:.1f} s")

    N = len(retrieved_pool)
    summary = {
        "model (zero-shot)": {
            "pct_answers_with_at_least_one_citation": round((zs_model_with_cit / N) * 100.0, 2),
            "mean_citations_per_answer": round(zs_model_total_cits / N, 4),
            "pct_unsupported_sentences": round((zs_unsupported_sents / zs_total_sents * 100.0) if zs_total_sents else 0.0, 2),
        },
        "model (few-shot)": {
            "pct_answers_with_at_least_one_citation": round((fs_model_with_cit / N) * 100.0, 2),
            "mean_citations_per_answer": round(fs_model_total_cits / N, 4),
            "pct_unsupported_sentences": round((fs_unsupported_sents / fs_total_sents * 100.0) if fs_total_sents else 0.0, 2),
        },
        "attribution": {
            "pct_answers_with_at_least_one_citation": round((zs_attr_with_cit / N) * 100.0, 2),
            "mean_citations_per_answer": round(zs_attr_total_cits / N, 4),
            "pct_unsupported_sentences": round((zs_unsupported_sents / zs_total_sents * 100.0) if zs_total_sents else 0.0, 2),
        }
    }

    with open("reports/citation_mechanisms_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Completed citation comparison evaluation.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
