#!/usr/bin/env python3
"""GPU-side offline LLM generation script for Colab / Kaggle / GPU environments (Change Order §6).

Self-contained script. Does NOT import from adaptive_retriever.
Dependencies: torch, transformers, tqdm.

Usage:
  python generate_offline.py --requests requests.jsonl --results results.jsonl \
         --model Qwen/Qwen2.5-0.5B-Instruct --batch-size 16 --max-new-tokens 96
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="GPU-side offline LLM generation script.")
    parser.add_argument("--requests", type=str, required=True, help="Path to input requests.jsonl")
    parser.add_argument("--results", type=str, required=True, help="Path to output results.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for text generation")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max new tokens per generation")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto | cuda | cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of requests to process")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    t0 = time.time()

    requests_path = args.requests
    results_path = args.results

    if not os.path.exists(requests_path):
        print(f"Error: Requests file not found at {requests_path}")
        sys.exit(1)

    # 1. Resume check: read existing results.jsonl
    completed_ids = set()
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    if not data.get("_meta") and "request_id" in data:
                        completed_ids.add(data["request_id"])
                except Exception:
                    pass

    # Read requests.jsonl (skip header line if _meta: true)
    all_requests = []
    meta_header = None
    with open(requests_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            if data.get("_meta"):
                meta_header = data
            else:
                all_requests.append(data)

    pending_requests = [r for r in all_requests if r.get("request_id") not in completed_ids]

    if args.limit > 0:
        pending_requests = pending_requests[: args.limit]

    print(f"Resumed: {len(completed_ids)} done, {len(pending_requests)} remaining out of {len(all_requests)} total.")

    if not pending_requests:
        print("All requests already completed. Nothing to do!")
        sys.exit(0)

    # 2. Device and dtype resolution (§3.2)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model '{args.model}' on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Batched generation traps (§3.3)
    # Trap 1: Left padding is MANDATORY for decoder-only batched generation
    tokenizer.padding_side = "left"

    # Trap 2: Pad token fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    # 3. Sort pending requests by tokenized prompt length to reduce padding waste
    indexed_requests = list(enumerate(pending_requests))
    indexed_requests.sort(key=lambda x: len(tokenizer.encode(x[1]["prompt"])))

    # Ensure results directory exists
    os.makedirs(os.path.dirname(os.path.abspath(results_path)) or ".", exist_ok=True)
    results_file = open(results_path, "a", encoding="utf-8")

    total_generated = 0
    start_gen_time = time.time()

    pbar = tqdm(total=len(pending_requests), desc="Generating representations")

    try:
        for i in range(0, len(indexed_requests), args.batch_size):
            batch = indexed_requests[i : i + args.batch_size]

            formatted_prompts = []
            for _, req in batch:
                p = req["prompt"]
                sys_msg = req.get("system")
                msgs = []
                if sys_msg:
                    msgs.append({"role": "system", "content": sys_msg})
                msgs.append({"role": "user", "content": p})
                try:
                    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    fmt = f"{sys_msg}\n{p}" if sys_msg else p
                formatted_prompts.append(fmt)

            # Trap 3: Tokenize with return_tensors and pass attention_mask
            inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            input_len = input_ids.shape[1]

            # Trap 4: Wrap generation in torch.inference_mode()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode ONLY completion slice: outputs[:, input_len:]
            generated_ids = outputs[:, input_len:]
            decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Re-assemble results and write to file immediately with flush + fsync
            for j, (orig_idx, req) in enumerate(batch):
                gen_text = decoded_texts[j].strip()
                t_in = int(attention_mask[j].sum().item())
                t_out = int((generated_ids[j] != tokenizer.pad_token_id).sum().item())

                result_rec = {
                    "request_id": req["request_id"],
                    "chunk_id": req["chunk_id"],
                    "kind": req["kind"],
                    "index": req.get("index", 0),
                    "text": gen_text,
                    "model": args.model,
                    "tokens_in": max(1, t_in),
                    "tokens_out": max(1, t_out),
                    "generated_at": datetime.now().isoformat(),
                }

                results_file.write(json.dumps(result_rec) + "\n")
                total_generated += 1

            results_file.flush()
            os.fsync(results_file.fileno())

            pbar.update(len(batch))

    except Exception as e:
        results_file.flush()
        os.fsync(results_file.fileno())
        results_file.close()
        print(f"\nError during batch generation: {e}")
        sys.exit(1)

    results_file.close()
    elapsed = time.time() - start_gen_time

    mean_s_per_completion = elapsed / max(1, total_generated)
    print("\nOffline Generation Completed Successfully!")
    print(f"Total Generated: {total_generated}")
    print(f"Elapsed Time: {elapsed:.2f}s")
    print(f"Mean Time per Completion: {mean_s_per_completion:.3f}s")


if __name__ == "__main__":
    main()
