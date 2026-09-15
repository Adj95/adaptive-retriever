"""Is the cross-encoder slow because of CPU execution, or VRAM contention?

On 2026-09-14 the reranker scored 100 pairs in ~1,012 ms on cuda:0.
On 2026-09-15 the same call took ~6,844 ms, and the agent concluded "CPU
execution". But torch.cuda.is_available() is True in this venv.

The difference between those two dates is that Phase 3 added Qwen2.5-0.5B to the
pipeline. In fp32 that is ~2 GB of weights on a 4 GB card that already gives
~0.5 GB to Windows. This measures the reranker alone, then again with the
generator resident, so the two explanations can be told apart.

Reads nothing, writes nothing.

    python check_gpu_now.py
"""
import time

import torch

SEP = "=" * 72


def mb(x):
    return f"{x / 1e6:,.0f} MB"


print(SEP)
print("1. ENVIRONMENT")
print(SEP)
print(f"  torch            : {torch.__version__}")
print(f"  torch.version.cuda: {torch.version.cuda}")
print(f"  cuda available   : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("  CUDA not available in THIS process. That is the answer.")

props = torch.cuda.get_device_properties(0)
print(f"  device           : {props.name}")
print(f"  total VRAM       : {mb(props.total_memory)}")
free, total = torch.cuda.mem_get_info()
print(f"  free VRAM now    : {mb(free)} of {mb(total)}")

PAIRS = [("what methods were used to evaluate results", "lorem ipsum dolor sit amet " * 60)] * 100

# ------------------------------------------------- 2. reranker on its own ---
print()
print(SEP)
print("2. CROSS-ENCODER ALONE (100 pairs)")
print(SEP)
from sentence_transformers import CrossEncoder  # noqa: E402

torch.cuda.reset_peak_memory_stats()
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
ce.predict(PAIRS[:4])  # warm the kernels
torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
ce.predict(PAIRS)
torch.cuda.synchronize()
alone_ms = (time.perf_counter() - t0) * 1000

print(f"  model device     : {ce.model.device}")
print(f"  100 pairs        : {alone_ms:,.0f} ms")
print(f"  peak allocated   : {mb(torch.cuda.max_memory_allocated())}")
free, _ = torch.cuda.mem_get_info()
print(f"  free VRAM after  : {mb(free)}")

# -------------------------------------- 3. same, with the generator loaded ---
print()
print(SEP)
print("3. SAME CALL, WITH Qwen2.5-0.5B RESIDENT (fp32)")
print(SEP)
try:
    from transformers import AutoModelForCausalLM  # noqa: E402

    t0 = time.perf_counter()
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32
    ).to("cuda")
    llm.eval()
    torch.cuda.synchronize()
    print(f"  generator loaded : {(time.perf_counter() - t0):.1f} s")
    print(f"  peak allocated   : {mb(torch.cuda.max_memory_allocated())}")
    free, _ = torch.cuda.mem_get_info()
    print(f"  free VRAM left   : {mb(free)}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ce.predict(PAIRS)
    torch.cuda.synchronize()
    together_ms = (time.perf_counter() - t0) * 1000
    print(f"  100 pairs now    : {together_ms:,.0f} ms")
    print(f"  peak allocated   : {mb(torch.cuda.max_memory_allocated())}")

except torch.cuda.OutOfMemoryError as exc:
    print(f"  CUDA OUT OF MEMORY loading the generator: {str(exc)[:160]}")
    print("  That alone explains the slowdown - the pipeline cannot hold both.")
    raise SystemExit(1)
except Exception as exc:  # noqa: BLE001
    print(f"  Failed to load generator: {type(exc).__name__}: {str(exc)[:200]}")
    raise SystemExit(1)

# ------------------------------------------------------------- verdict ---
print()
print(SEP)
print("VERDICT")
print(SEP)
print(f"  reranker alone        : {alone_ms:,.0f} ms")
print(f"  reranker + generator  : {together_ms:,.0f} ms")
ratio = together_ms / alone_ms if alone_ms else 0
print(f"  slowdown factor       : {ratio:.2f}x")
print()
if alone_ms > 4000:
    print("  The reranker is slow even ALONE on cuda. Not contention -")
    print("  investigate the GPU itself (thermal throttling, WDDM overhead,")
    print("  another process holding the card).")
elif ratio > 2.0:
    print("  CONFIRMED: holding the fp32 generator resident starves the")
    print("  cross-encoder. Fix by loading the generator lazily and freeing it")
    print("  (del + torch.cuda.empty_cache()) between stages, or by running")
    print("  generation as a separate pass rather than inline.")
else:
    print("  Both are fast. The 6,844 ms figure did not come from this cause -")
    print("  re-measure inside the experiment script itself and print")
    print("  reranker.model.device at that point.")
