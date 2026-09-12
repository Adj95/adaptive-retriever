"""Benchmark raw generation speed, outside the pipeline.

    python bench_gpu.py

Answers one question: is the slowness the GPU, or the code around it?
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH = 32
NEW_TOKENS = 64

print("=" * 62)
print("GPU")
print("=" * 62)
print("torch          :", torch.__version__, "| cuda", torch.version.cuda)
print("cuda available :", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA. Stop here and report this.")

props = torch.cuda.get_device_properties(0)
print("device         :", props.name)
print("total VRAM     :", round(props.total_memory / 1024**3, 2), "GB")
print("compute cap    :", f"{props.major}.{props.minor}")
print("SM count       :", props.multi_processor_count)
free, total = torch.cuda.mem_get_info()
print("free VRAM now  :", round(free / 1024**3, 2), "GB  <- low means another app is using the GPU")

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# A prompt roughly the size the pipeline sends (800 chars of chunk text).
body = ("The Federal Reserve maintained its target range for the federal funds rate "
        "while noting that inflation remains elevated relative to the committee's "
        "longer-run objective. Participants discussed labour market conditions. ") * 4
prompt = f"Summarise the following text in one or two sentences.\n\nText:\n{body[:800]}\n\nSummary:"
msgs = [{"role": "user", "content": prompt}]
formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
texts = [formatted] * BATCH

enc = tok(texts, return_tensors="pt", padding=True)
print("\nprompt tokens per item:", enc["input_ids"].shape[1])


def run(dtype, label):
    print("\n" + "=" * 62)
    print(f"{label}   batch={BATCH}  max_new_tokens={NEW_TOKENS}")
    print("=" * 62)
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to("cuda").eval()
    print(f"  load           : {time.perf_counter() - t0:.1f}s")

    inputs = tok(texts, return_tensors="pt", padding=True).to("cuda")
    ilen = inputs["input_ids"].shape[1]

    # Warm-up (kernel autotuning, allocator) - not timed.
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=8, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    el = time.perf_counter() - t0

    steps = out.shape[1] - ilen
    print(f"  generated      : {steps} tokens x {BATCH} sequences")
    print(f"  wall time      : {el:.2f}s")
    print(f"  ---> {BATCH / el:6.2f} completions/sec")
    print(f"  ---> {steps / el:6.2f} decode steps/sec")
    print(f"  peak VRAM      : {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return BATCH / el


fp16 = run(torch.float16, "FLOAT16 (what the pipeline uses)")
fp32 = run(torch.float32, "FLOAT32")

print("\n" + "=" * 62)
print("VERDICT")
print("=" * 62)
print(f"  fp16: {fp16:.2f} comp/s      fp32: {fp32:.2f} comp/s")
print(f"  pipeline measured: 0.31 comp/s")
print()
best = max(fp16, fp32)
if best < 1.0:
    print("  The GPU itself is the limit. Local generation is not viable;")
    print("  use the Kaggle offload path.")
elif best > 3 * 0.31:
    print(f"  Raw generation is {best / 0.31:.0f}x faster than the pipeline.")
    print("  The GPU is fine - the pipeline is losing the time elsewhere.")
else:
    print("  Pipeline is close to raw speed. The GPU is the ceiling.")
if fp32 > fp16 * 1.3:
    print("  NOTE: fp32 beats fp16 on this card. Set models.llm.dtype: float32")
