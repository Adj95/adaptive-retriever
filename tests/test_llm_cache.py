"""LLM cache test suite verifying diskcache hits, zero-cost cached responses, and key invalidation on prompt version change."""

import pytest
from adaptive_retriever.llm import CachedLLMClient, LLMResponse


class MockCachedLLMClient(CachedLLMClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_call_count = 0

    def _call_api(self, prompt, system, max_tokens, temperature):
        self.api_call_count += 1
        return {
            "choices": [{"message": {"content": f"Response to: {prompt}"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
        }


def test_llm_cache_hit_and_version_invalidation(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    client = MockCachedLLMClient(
        cache_dir=cache_dir,
        price_per_1k_in=0.002,
        price_per_1k_out=0.004,
    )

    prompt_text = "What is the capital of France?"

    # First call: cache miss
    res1 = client.complete(prompt_text, prompt_version="1")
    assert res1.cached is False
    assert client.api_call_count == 1
    assert res1.usd_cost > 0.0

    # Second call (same prompt + same version): cache hit!
    res2 = client.complete(prompt_text, prompt_version="1")
    assert res2.cached is True
    assert res2.usd_cost == 0.0
    assert res2.text == res1.text
    assert client.api_call_count == 1  # No additional API call made

    # Third call (same prompt + DIFFERENT prompt_version): cache miss!
    res3 = client.complete(prompt_text, prompt_version="2")
    assert res3.cached is False
    assert client.api_call_count == 2


def test_complete_batch_partial_hits(tmp_path):
    cache_dir = tmp_path / "llm_cache_batch"

    class BatchMockClient(CachedLLMClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_api_call_count = 0

        def _call_api_batch(self, prompts, system, max_tokens, temperature):
            self.batch_api_call_count += 1
            return [
                {
                    "choices": [{"message": {"content": f"BatchResponse: {p}"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
                for p in prompts
            ]

    client = BatchMockClient(cache_dir=cache_dir)

    # 1. Warm up cache for P1 and P3
    res_p1 = client.complete("P1")
    res_p3 = client.complete("P3")

    initial_api_calls = client.batch_api_call_count
    assert initial_api_calls == 2

    # 2. Batch request with P1 (cached), P2 (miss), P3 (cached), P4 (miss)
    batch_prompts = ["P1", "P2", "P3", "P4"]
    batch_res = client.complete_batch(batch_prompts)

    assert len(batch_res) == 4
    assert [r.text for r in batch_res] == [
        "BatchResponse: P1",
        "BatchResponse: P2",
        "BatchResponse: P3",
        "BatchResponse: P4",
    ]

    # Verify cached flags and order preservation
    assert batch_res[0].cached is True
    assert batch_res[1].cached is False
    assert batch_res[2].cached is True
    assert batch_res[3].cached is False

    # Under complete_batch with partial hits, only miss_prompts (P2, P4) triggered 1 underlying batch call
    assert client.batch_api_call_count == initial_api_calls + 1


def test_llm_cache_incremental_flush(tmp_path):
    cache_dir = tmp_path / "llm_cache_incremental"

    class IncrementalMockClient(CachedLLMClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_size = 2

        def _call_api_batch(self, prompts, system, max_tokens, temperature):
            return [
                {
                    "choices": [{"message": {"content": f"Output: {p}"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
                for p in prompts
            ]

    client = IncrementalMockClient(cache_dir=cache_dir)
    prompts = [f"Prompt_{i}" for i in range(5)]
    res = client.complete_batch(prompts)
    assert len(res) == 5
    assert len(client.cache) == 5

