"""LLM Client protocol, cached base client, and local HuggingFace Transformers client.

CORRECTIONS (2026-09-11)
------------------------
1. `models.llm.max_new_tokens` and `temperature` were read from config by
   nobody. `get_llm_client()` never passed them, `generate_representations()`
   never passed them, and `_call_api_batch` fell back to a hardcoded
   `max_tokens or 96`. Setting 192 in default.yaml did nothing.
2. Truncation was FATAL. A 0.5B model handed dense mathematics or a
   bibliography will sometimes exceed any token budget; that is normal model
   behaviour, not a corrupt result. One such output killed a 428-document run
   at document zero. Truncated output is now retried once at double the budget,
   then trimmed at the last sentence boundary, counted, and kept. A run fails
   only if the truncation RATE exceeds a configured budget.
3. The retry block caught the exception it had just raised and re-raised it by
   string-matching its own message, silently swallowing every other exception.
4. `complete_batch` ended with `[r for r in responses if r is not None]`, which
   silently changes list length. Callers `zip()` that list against the request
   list, so a single dropped entry would misalign every subsequent
   representation onto the wrong chunk. It now asserts instead.
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from diskcache import Cache
from pydantic import BaseModel

logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_MAX_NEW_TOKENS = 192

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _is_rule_fallback(text: str) -> bool:
    if not text:
        return True
    return (
        text.startswith("Question #")
        or text.startswith("Summary of chunk ")
        or "regarding:" in text
    )


def trim_to_sentence_boundary(text: str) -> str:
    """Trim a truncated generation back to its last complete sentence.

    Falls back to the last whitespace, then to the raw text. Never returns
    something longer than the input.
    """
    text = text.strip()
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    matches = list(_SENTENCE_END.finditer(text))
    if matches:
        return text[: matches[-1].start() + 1].strip()
    last_space = text.rfind(" ")
    if last_space > 0:
        return text[:last_space].strip()
    return text


def _kind_from_prompt(prompt: str) -> str:
    """Best-effort kind label for truncation accounting."""
    low = prompt.lower()
    if "keyword" in low:
        return "keywords"
    if "question" in low:
        return "question"
    return "summary"


class LLMResponse(BaseModel):
    text: str
    tokens_in: int
    tokens_out: int
    usd_cost: float
    cached: bool
    model: str
    truncated: bool = False


class LLMClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        prompt_version: str = "1",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stage: str = "general",
    ) -> LLMResponse:
        ...

    def complete_batch(
        self,
        prompts: List[str],
        *,
        system: Optional[str] = None,
        prompt_version: str = "1",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stage: str = "general",
    ) -> List[LLMResponse]:
        ...


class CachedLLMClient:
    """Base LLM client with diskcache support and batched completion handling."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        cache_dir: Path | str = "data/cache",
        price_per_1k_in: float = 0.0,
        price_per_1k_out: float = 0.0,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.cache = Cache(directory=str(cache_dir))
        self.price_per_1k_in = price_per_1k_in
        self.price_per_1k_out = price_per_1k_out
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.stats: Dict[str, Any] = {
            "llm_calls_generated": 0,
            "llm_calls_cached": 0,
            "generation_time": 0.0,
            "truncated_count": {"summary": 0, "question": 0, "keywords": 0},
            "max_new_tokens": self.max_new_tokens,
        }

    # ------------------------------------------------------------------ stats

    def throughput_summary(self) -> Dict[str, Any]:
        gen = self.stats["llm_calls_generated"]
        t = self.stats["generation_time"]
        total = gen + self.stats["llm_calls_cached"]
        trunc = sum(self.stats["truncated_count"].values())
        return {
            "llm_calls_generated": gen,
            "llm_calls_cached": self.stats["llm_calls_cached"],
            "generation_seconds": round(t, 2),
            "generation_rate_per_s": round(gen / t, 2) if gen and t > 0 else None,
            "generation_rate_note": None if gen else "n/a (all cached)",
            "truncated_count": dict(self.stats["truncated_count"]),
            "truncation_rate": round(trunc / total, 4) if total else None,
            "max_new_tokens": self.max_new_tokens,
        }

    def assert_truncation_within_budget(self, max_rate: float = 0.20) -> None:
        """Fail the RUN, not the item, when truncation is systemic.

        A few truncations mean the model rambled. A fifth of them means the
        token budget is wrong, and that is a finding to report, not to tune
        away quietly.
        """
        total = self.stats["llm_calls_generated"] + self.stats["llm_calls_cached"]
        if not total:
            return
        trunc = sum(self.stats["truncated_count"].values())
        rate = trunc / total
        if rate > max_rate:
            raise RuntimeError(
                f"Truncation rate {rate:.1%} ({trunc}/{total}) exceeds the "
                f"{max_rate:.0%} budget at max_new_tokens={self.max_new_tokens}. "
                f"Raise models.llm.max_new_tokens or shorten the prompts. "
                f"Per kind: {self.stats['truncated_count']}"
            )

    # ------------------------------------------------------------------ cache

    def _compute_key(
        self,
        prompt: str,
        system: Optional[str],
        prompt_version: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> str:
        key_data = {
            "model": self.model_name,
            "system": system or "",
            "prompt_version": prompt_version,
            "prompt": prompt,
            "max_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        raw_json = json.dumps(key_data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_json).hexdigest()

    def _call_api(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> dict:
        return self._call_api_batch([prompt], system, max_tokens, temperature)[0]

    def _call_api_batch(
        self,
        prompts: List[str],
        system: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> List[dict]:
        return [self._call_api(p, system, max_tokens, temperature) for p in prompts]

    # --------------------------------------------------------------- complete

    def complete_batch(
        self,
        prompts: List[str],
        *,
        system: Optional[str] = None,
        prompt_version: str = "1",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stage: str = "general",
    ) -> List[LLMResponse]:
        if not prompts:
            return []

        keys = [
            self._compute_key(p, system, prompt_version, max_tokens, temperature)
            for p in prompts
        ]

        responses: List[Optional[LLMResponse]] = [None] * len(prompts)
        miss_indices: List[int] = []
        miss_prompts: List[str] = []

        for idx, key in enumerate(keys):
            cached_val = self.cache.get(key)
            if cached_val is not None and not _is_rule_fallback(cached_val.get("text", "")):
                self.stats["llm_calls_cached"] += 1
                responses[idx] = LLMResponse(
                    text=cached_val["text"],
                    tokens_in=cached_val["tokens_in"],
                    tokens_out=cached_val["tokens_out"],
                    usd_cost=0.0,
                    cached=True,
                    model=self.model_name,
                    truncated=bool(cached_val.get("truncated", False)),
                )
                if cached_val.get("truncated"):
                    kind = _kind_from_prompt(prompts[idx])
                    self.stats["truncated_count"][kind] += 1
            else:
                miss_indices.append(idx)
                miss_prompts.append(prompts[idx])

        if miss_prompts:
            from tqdm import tqdm

            chunk_size = getattr(self, "batch_size", 16)
            pbar = tqdm(
                total=len(miss_prompts),
                desc="Generating LLM representations",
                unit="req",
                leave=False,
            )
            for sub_start in range(0, len(miss_prompts), chunk_size):
                sub_indices = miss_indices[sub_start : sub_start + chunk_size]
                sub_prompts = miss_prompts[sub_start : sub_start + chunk_size]

                t_sub0 = time.perf_counter()
                api_results = self._call_api_batch(sub_prompts, system, max_tokens, temperature)
                self.stats["generation_time"] += time.perf_counter() - t_sub0
                self.stats["llm_calls_generated"] += len(sub_prompts)

                if len(api_results) != len(sub_prompts):
                    raise RuntimeError(
                        f"Backend returned {len(api_results)} results for "
                        f"{len(sub_prompts)} prompts. Refusing to continue with a "
                        f"misaligned batch."
                    )

                for i, api_res in enumerate(api_results):
                    orig_idx = sub_indices[i]
                    cache_key = keys[orig_idx]

                    generated_text = api_res["choices"][0]["message"]["content"].strip()
                    was_truncated = bool(api_res.get("truncated", False))
                    usage = api_res.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", max(1, len(sub_prompts[i]) // 4))
                    tokens_out = usage.get("completion_tokens", max(1, len(generated_text) // 4))
                    cost = (tokens_in / 1000.0 * self.price_per_1k_in) + (
                        tokens_out / 1000.0 * self.price_per_1k_out
                    )

                    if was_truncated:
                        self.stats["truncated_count"][_kind_from_prompt(sub_prompts[i])] += 1

                    if not _is_rule_fallback(generated_text):
                        self.cache.set(
                            cache_key,
                            {
                                "text": generated_text,
                                "tokens_in": tokens_in,
                                "tokens_out": tokens_out,
                                "truncated": was_truncated,
                            },
                        )

                    responses[orig_idx] = LLMResponse(
                        text=generated_text,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        usd_cost=cost,
                        cached=False,
                        model=self.model_name,
                        truncated=was_truncated,
                    )
                pbar.update(len(sub_prompts))
            pbar.close()

        missing = [i for i, r in enumerate(responses) if r is None]
        if missing:
            raise RuntimeError(
                f"complete_batch produced no response for prompt indices {missing[:10]}. "
                f"Dropping them would misalign every downstream representation."
            )
        return [r for r in responses]  # type: ignore[misc]

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        prompt_version: str = "1",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stage: str = "general",
    ) -> LLMResponse:
        return self.complete_batch(
            [prompt],
            system=system,
            prompt_version=prompt_version,
            max_tokens=max_tokens,
            temperature=temperature,
            stage=stage,
        )[0]


class LocalTransformersLLMClient(CachedLLMClient):
    """Local PyTorch/Transformers CausalLM batched text-generation client."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        cache_dir: Path | str = "data/cache",
        price_per_1k_in: float = 0.0,
        price_per_1k_out: float = 0.0,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 16,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        truncation_retry: bool = True,
    ):
        self.truncation_retry = truncation_retry
        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            price_per_1k_in=price_per_1k_in,
            price_per_1k_out=price_per_1k_out,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        import torch

        if device == "auto":
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
                self.device = "cuda" if free_bytes >= 2.0 * 1024 * 1024 * 1024 else "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            if dtype == "auto" or dtype == "float16":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = torch.float32

        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes < 2.0 * 1024 * 1024 * 1024:
                logger.info("Free VRAM (%d MB) < 2000 MB; placing LLM on CPU to avoid OOM", free_bytes // (1024 * 1024))
                self.device = "cpu"
                self.dtype = torch.float32

        logger.info(
            "Loading local HuggingFace model '%s' on %s (%s), max_new_tokens=%d",
            self.model_name,
            self.device,
            self.dtype,
            self.max_new_tokens,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)
        self._model.eval()

    def _format(self, prompt: str, system: Optional[str]) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            return self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception as err:  # noqa: BLE001 - tokenizer may lack a template
            logger.warning("Chat template unavailable for %s: %s", self.model_name, err)
            return f"{system}\n{prompt}" if system else prompt

    def _generate(self, texts: List[str], max_t: int, temperature: Optional[float]):
        import torch

        inputs = self._tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        input_len = inputs["input_ids"].shape[1]
        do_sample = temperature is not None and temperature > 0.0
        with torch.inference_mode():
            outputs = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_t,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        return outputs[:, input_len:], inputs["attention_mask"]

    def _hit_limit(self, row, max_t: int) -> bool:
        """True when generation stopped on the token budget rather than EOS."""
        eos = self._tokenizer.eos_token_id
        return int(row.shape[0]) >= max_t and eos not in row

    def _call_api_batch(
        self,
        prompts: List[str],
        system: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> List[dict]:
        if not prompts:
            return []

        self._ensure_loaded()
        import torch

        max_t = max_tokens if max_tokens is not None else self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature

        # Sort by length to minimise padding waste; restore order at the end.
        indexed = sorted(enumerate(prompts), key=lambda x: len(x[1]))
        results_by_idx: Dict[int, dict] = {}

        cursor = 0
        while cursor < len(indexed):
            batch = indexed[cursor : cursor + max(1, self.batch_size)]
            texts = [self._format(p, system) for _, p in batch]

            try:
                gen_ids, attn = self._generate(texts, max_t, temp)
            except Exception as gen_err:
                err_s = str(gen_err).lower()
                oom_type = getattr(torch, "OutOfMemoryError", ())
                is_oom = "out of memory" in err_s or isinstance(gen_err, oom_type)
                if is_oom and self.batch_size > 1:
                    old = self.batch_size
                    self.batch_size = max(1, self.batch_size // 2)
                    logger.warning("OOM: batch_size %d -> %d, retrying", old, self.batch_size)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

            decoded = self._tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for i, (orig_idx, orig_prompt) in enumerate(batch):
                text = decoded[i].strip()
                t_in = int(attn[i].sum().item())
                t_out = int((gen_ids[i] != self._tokenizer.pad_token_id).sum().item())
                truncated = False

                if self._hit_limit(gen_ids[i], max_t):
                    # A retry regenerates ONE sequence, and a batch of 1 costs
                    # nearly the same per decode step as a batch of 32 - measured
                    # at ~45s for a single item. Off by default; trimming is
                    # nearly as good and ~100x cheaper.
                    if self.truncation_retry:
                        retry_ids, retry_attn = self._generate([texts[i]], max_t * 2, temp)
                        text = self._tokenizer.batch_decode(
                            retry_ids, skip_special_tokens=True
                        )[0].strip()
                        t_in = int(retry_attn[0].sum().item())
                        t_out = int((retry_ids[0] != self._tokenizer.pad_token_id).sum().item())
                        still_over = self._hit_limit(retry_ids[0], max_t * 2)
                    else:
                        still_over = True

                    if still_over:
                        # Still over budget. Trim cleanly and KEEP it. A summary
                        # cut at a sentence boundary is degraded, not corrupt,
                        # and must not kill a 428-document run.
                        trimmed = trim_to_sentence_boundary(text)
                        if trimmed:
                            text = trimmed
                        truncated = True
                        logger.debug(
                            "Truncated twice at max_new_tokens=%d; trimmed to sentence "
                            "boundary (%d chars)",
                            max_t * 2,
                            len(text),
                        )

                results_by_idx[orig_idx] = {
                    "choices": [{"message": {"content": text}}],
                    "usage": {"prompt_tokens": max(1, t_in), "completion_tokens": max(1, t_out)},
                    "truncated": truncated,
                }
            cursor += len(batch)

        if len(results_by_idx) != len(prompts):
            raise RuntimeError(
                f"Generated {len(results_by_idx)} results for {len(prompts)} prompts."
            )
        return [results_by_idx[i] for i in range(len(prompts))]


def get_llm_client(config: Optional[Any] = None) -> LLMClient:
    """Factory. Reads EVERY llm setting from config - including max_new_tokens
    and temperature, which the previous version silently ignored."""
    if config is None:
        return LocalTransformersLLMClient()

    if hasattr(config, "raw"):
        raw_root = config.raw
    else:
        raw_root = config
    raw_llm = raw_root.get("models", {}).get("llm", {})
    cache_dir = raw_root.get("paths", {}).get("cache_dir", "data/cache")

    return LocalTransformersLLMClient(
        model_name=raw_llm.get("model", "Qwen/Qwen2.5-0.5B-Instruct"),
        cache_dir=cache_dir,
        device=raw_llm.get("device", "auto"),
        dtype=raw_llm.get("dtype", "auto"),
        batch_size=raw_llm.get("batch_size", 16),
        max_new_tokens=raw_llm.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
        temperature=raw_llm.get("temperature", 0.0),
        truncation_retry=raw_llm.get("truncation_retry", True),
        price_per_1k_in=raw_llm.get("price_per_1k_in", 0.0),
        price_per_1k_out=raw_llm.get("price_per_1k_out", 0.0),
    )
