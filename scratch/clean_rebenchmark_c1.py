import time
import torch
import shutil
from pathlib import Path
from adaptive_retriever.llm import LocalTransformersLLMClient

print("=== C1: CLEAN GENERATION RE-BENCHMARK (CUDA GPU) ===", flush=True)

base_texts = [
    f"Paragraph {i}: In financial accounting, derivative contracts such as futures, options, and swaps "
    "are utilized by corporations to manage market risk, currency volatility, and interest rate exposure. "
    "Under ASC 815, derivatives are recognized on the balance sheet at fair value."
    for i in range(256)
]

cache_path = Path("data/cache_clean_benchmark")
if cache_path.exists():
    try:
        shutil.rmtree(cache_path)
    except Exception:
        pass

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

client = LocalTransformersLLMClient(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir=cache_path,
    device="cuda",
    dtype="float32",
    batch_size=16,
    max_new_tokens=128,
)

client._ensure_loaded()
print(f"Loaded LLM on device: {client.device}, dtype: {client.dtype}", flush=True)
assert client.device == "cuda", f"ERROR: LLM loaded on {client.device} instead of cuda!"

# 1. Warm-up pass (one discarded batch of 16 prompts)
warmup_prompts = [f"Summarize text {j}:\n{base_texts[j]}" for j in range(16)]
_ = client.complete_batch(warmup_prompts, max_tokens=128)
print("Warm-up batch of 16 completed and discarded.", flush=True)

# 2. Time 200 generations at batch 16, max_new_tokens=128
N_BENCH = 200
bench_prompts = [f"Summarize text {j + 1000}:\n{base_texts[j % len(base_texts)]}" for j in range(N_BENCH)]

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

t0 = time.perf_counter()
responses = client.complete_batch(bench_prompts, max_tokens=128)
t_elapsed = time.perf_counter() - t0

gen_count = len(responses)
sec_per_gen = t_elapsed / gen_count
total_tokens_out = sum(r.tokens_out for r in responses)
tokens_per_sec = total_tokens_out / t_elapsed if t_elapsed > 0 else 0.0
peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

full_pass_gens = 10449
proj_seconds = full_pass_gens * sec_per_gen
proj_minutes = proj_seconds / 60.0
proj_hours = proj_seconds / 3600.0

print("\n=== CLEAN RE-BENCHMARK RESULTS (200 Generations, BS 16, max_new_tokens 128) ===")
print(f"Total Wall-Clock Time   : {t_elapsed:.2f} seconds")
print(f"Seconds per Generation  : {sec_per_gen:.4f} s/gen ({sec_per_gen * 1000:.1f} ms/gen)")
print(f"Aggregate Tokens / Sec  : {tokens_per_sec:.2f} tok/s")
print(f"Peak VRAM Allocated     : {peak_vram_mb:.1f} MB")
print(f"Projected Full Pass ({full_pass_gens} gens): {proj_seconds:.1f} s = {proj_minutes:.2f} minutes ({proj_hours:.2f} hours)")

try:
    client.cache.close()
    shutil.rmtree(cache_path)
except Exception:
    pass
