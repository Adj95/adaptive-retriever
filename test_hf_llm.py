import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Starting LLM test script...", flush=True)
t0 = time.time()

from adaptive_retriever.llm import LocalTransformersLLMClient

client = LocalTransformersLLMClient()
print(f"Loaded client in {time.time() - t0:.2f}s", flush=True)

t1 = time.time()
res = client.complete("Summarize this text in 1 sentence: Artificial intelligence is the simulation of human intelligence processes by machines.", stage="test")
print(f"Generated text in {time.time() - t1:.2f}s:", flush=True)
print(f"Model: {res.model}", flush=True)
print(f"Text: {res.text}", flush=True)
print(f"Tokens in: {res.tokens_in}, Tokens out: {res.tokens_out}", flush=True)
