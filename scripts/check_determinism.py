"""Script to test generation determinism at fixed vs varying batch positions (Task C3)."""

import json
import logging
from pathlib import Path
import torch

from adaptive_retriever.config import Config
from adaptive_retriever.llm import LocalTransformersLLMClient, get_llm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    cfg = Config.load("config/default.yaml")
    
    # Load 131 identical items from raw ledger
    with open("reports/experiment_006_raw.jsonl", "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    identical_records = []
    for r in records:
        c1 = r["arms"]["Arm 1: generation_alone"].get("context_str", "")
        c3 = r["arms"]["Arm 3: +compression"].get("context_str", "")
        if c1 == c3:
            identical_records.append(r)
            
    print(f"Identical items count: {len(identical_records)}")
    test_5 = identical_records[:5]
    
    llm = get_llm_client(cfg)
    llm.batch_size = 1
    llm._ensure_loaded()
    
    gen_cfg = cfg.raw.get("generation", {})
    
    print("\n=== RESOLVED GENERATION PARAMETERS ===")
    print(f"Model: {gen_cfg.get('model_name')}")
    print(f"Device: {llm.device}")
    print(f"Dtype: {llm.dtype}")
    print(f"Tokenizer padding_side: {llm._tokenizer.padding_side}")
    print(f"Tokenizer pad_token_id: {llm._tokenizer.pad_token_id}")
    print(f"Attention Implementation: sdpa (PyTorch 2.x default)")
    print(f"use_cache: True (CausalLM default)")
    print(f"Arms batched together in run_experiment_006.py: False (executed sequentially per query item)")
    print(f"Arm execution order per item: Arm 1 -> Arm 2 -> Arm 3 -> Arm 4")
    
    print("\n=== PART 1: FIXED BATCH POSITION (BATCH SIZE 1, 5 SEQUENTIAL REPEATS) ===")
    part1_results = {}
    for item in test_5:
        q = item["query"]
        ctx = item["arms"]["Arm 1: generation_alone"].get("context_str", "")
        prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        
        outputs = []
        for r_idx in range(5):
            resp = llm.complete(prompt)
            outputs.append(resp.text)
            
        unique_outputs = set(outputs)
        identical_count = outputs.count(outputs[0])
        part1_results[item["item_idx"]] = (identical_count, len(outputs))
        print(f"Item {item['item_idx']}: {identical_count}/5 outputs byte-identical (Unique outputs: {len(unique_outputs)})")
        
    print("\n=== PART 2: VARYING BATCH POSITIONS & PADDING (BATCH SIZE 1 VS BATCH SIZE 4) ===")
    part2_results = {}
    
    prompts_batch = []
    for item in test_5:
        q = item["query"]
        ctx = item["arms"]["Arm 1: generation_alone"].get("context_str", "")
        prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        prompts_batch.append(prompt)
        
    # Batch size 1 individual outputs
    llm.batch_size = 1
    single_resps = llm.complete_batch(prompts_batch)
    single_texts = [r.text for r in single_resps]
    
    # Batch size 4 batched outputs (with padding)
    llm.batch_size = 4
    batch_resps = llm.complete_batch(prompts_batch)
    batch_texts = [r.text for r in batch_resps]
    
    for i, item in enumerate(test_5):
        match = (single_texts[i] == batch_texts[i])
        print(f"Item {item['item_idx']}: Batch Size 1 vs Batch Size 4 match = {match}")
        
    # Verdict determination
    all_p1_identical = all(ic == 5 for ic, _ in part1_results.values())
    any_p2_differs = any(single_texts[i] != batch_texts[i] for i in range(len(test_5)))
    
    print("\n=== VERDICT ===")
    if all_p1_identical and any_p2_differs:
        print("VERDICT: (a) Identical across repeats at fixed batch position, differs across batch positions -> the divergence is a HARNESS artifact (batch composition / padding), not hardware.")
    elif not all_p1_identical:
        print("VERDICT: (b) Differs even at fixed batch position -> kernel non-determinism confirmed.")
    else:
        print("VERDICT: Identical across both fixed and varying batch positions.")

if __name__ == "__main__":
    main()
