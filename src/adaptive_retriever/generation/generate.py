"""Grounded answer generation with verified citations (Task C, spec §6)."""

from dataclasses import dataclass, field
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

from adaptive_retriever.cache import cache_get, cache_set
from adaptive_retriever.models import RetrievalCandidate
from adaptive_retriever.prompts import load_prompt

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    text: str
    citations: List[str] = field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None
    trace: Dict[str, Any] = field(default_factory=dict)


def parse_citations_defensive(raw_text: str, valid_chunk_ids: set[str], strict: bool = True) -> List[str]:
    """Parse citations from generated text and verify resolvability.
    
    If strict=True, raises ValueError if a cited chunk_id is not in valid_chunk_ids (hallucinated citation).
    If strict=False, skips unresolvable/hallucinated chunk IDs.
    """
    if not raw_text:
        return []

    # Find all bracketed IDs like [chunk_id]
    matches = re.findall(r"\[([a-zA-Z0-9_\-\.:]+)\]", raw_text)
    if not matches:
        return []

    citations = []
    seen = set()
    for m in matches:
        cited_id = m.strip()
        if cited_id not in valid_chunk_ids:
            if strict:
                raise ValueError(
                    f"Hallucinated citation '{cited_id}' not found in candidate context chunk IDs."
                )
            continue
        if cited_id not in seen:
            seen.add(cited_id)
            citations.append(cited_id)

    return citations


def attribute_sentence_to_chunks(
    sentence: str,
    candidates: List[RetrievalCandidate],
    min_n_gram: int = 8,
) -> Optional[str]:
    """Find candidate chunk with highest verbatim n-gram overlap (n >= min_n_gram words, normalized whitespace & case).
    
    Returns attributed chunk_id if qualifying overlap found, else None.
    """
    clean_sent = re.sub(r"\s+", " ", sentence.strip().lower())
    words = re.findall(r"\b\w+\b", clean_sent)
    if not words:
        return None

    required_n = min(min_n_gram, len(words))
    if len(words) < required_n or required_n < 1:
        return None

    best_chunk_id = None
    max_matching_len = 0

    for cand in candidates:
        cand_norm = re.sub(r"\s+", " ", cand.text.strip().lower())
        has_any = False
        for start in range(len(words) - required_n + 1):
            if " ".join(words[start : start + required_n]) in cand_norm:
                has_any = True
                break
        if not has_any:
            continue

        cand_max_n = 0
        for window_len in range(len(words), required_n - 1, -1):
            if cand_max_n >= window_len:
                break
            found_match = False
            for start in range(len(words) - window_len + 1):
                ngram_str = " ".join(words[start : start + window_len])
                if ngram_str in cand_norm:
                    cand_max_n = window_len
                    found_match = True
                    break
            if found_match:
                break

        if cand_max_n >= required_n and cand_max_n > max_matching_len:
            max_matching_len = cand_max_n
            best_chunk_id = cand.chunk_id

    return best_chunk_id


def _build_cache_key(
    query: str,
    candidate_ids: List[str],
    prompt_version: str,
    model_name: str,
    citation_mode: str = "attribution",
    context_str: str = "",
) -> str:
    context_hash = hashlib.sha256(context_str.encode("utf-8")).hexdigest()[:16]
    composite = f"{query}|{','.join(candidate_ids)}|{context_hash}|{prompt_version}|{model_name}|{citation_mode}"
    key_hash = hashlib.sha256(composite.encode("utf-8")).hexdigest()
    return f"gen_ans:{key_hash}"


def generate_answer(
    query: str,
    candidates: List[RetrievalCandidate],
    cfg: Optional[Any] = None,
    llm: Optional[Any] = None,
    confidence: Optional[float] = None,
    ignore_cache: bool = False,
) -> GeneratedAnswer:
    """Generate a grounded answer with verified citations based strictly on candidate context."""
    trace: Dict[str, Any] = {
        "query": query,
        "n_candidates": len(candidates),
        "confidence": confidence,
    }

    # 1. Config validation
    gen_cfg = {}
    if cfg is not None:
        gen_cfg = cfg.raw.get("generation", {})
    
    enabled = gen_cfg.get("enabled", True)
    if not enabled:
        return GeneratedAnswer(
            text="",
            refused=True,
            refusal_reason="Generation disabled in configuration.",
            trace=trace,
        )

    refuse_thresh = gen_cfg.get("refuse_if_confidence_below", 0.15)
    if confidence is not None and confidence < refuse_thresh:
        return GeneratedAnswer(
            text="",
            refused=True,
            refusal_reason=f"Retrieval confidence {confidence:.4f} is below refusal threshold {refuse_thresh:.4f}.",
            trace=trace,
        )

    if not candidates:
        return GeneratedAnswer(
            text="",
            refused=True,
            refusal_reason="No candidate passages provided in context.",
            trace=trace,
        )

    citation_mode = gen_cfg.get("citation_mode", "attribution")
    trace["citation_mode"] = citation_mode

    # 2. Prompt loading
    prompt_name = gen_cfg.get("prompt_name")
    if prompt_name:
        prompt_template, prompt_version = load_prompt(prompt_name)
    elif gen_cfg.get("few_shot", False):
        prompt_template, prompt_version = load_prompt("generation/answer_with_citations_fewshot")
    else:
        prompt_template, prompt_version = load_prompt("generation/answer_with_citations")
    trace["prompt_version"] = prompt_version

    model_name = getattr(llm, "model_name", "Qwen/Qwen2.5-0.5B-Instruct")

    # 3. Format context
    context_blocks = []
    valid_chunk_ids = set()
    ordered_chunk_ids = []
    max_context_chars = 6000  # ~1500 tokens context cap
    current_chars = 0

    for cand in candidates:
        if current_chars >= max_context_chars and len(context_blocks) >= 3:
            break
        cand_text = cand.text[:1500] if len(cand.text) > 1500 else cand.text
        valid_chunk_ids.add(cand.chunk_id)
        ordered_chunk_ids.append(cand.chunk_id)
        context_blocks.append(f"[{cand.chunk_id}]: {cand_text}")
        current_chars += len(cand_text)

    context_str = "\n\n".join(context_blocks)
    formatted_prompt = prompt_template.format(context=context_str, query=query)

    # 4. Cache check
    cache_key = _build_cache_key(query, ordered_chunk_ids, prompt_version, model_name, citation_mode, context_str=context_str)
    if not ignore_cache:
        cached_val = cache_get(cache_key)
        if cached_val is not None and isinstance(cached_val, dict):
            trace["cache_hit"] = True
            return GeneratedAnswer(
                text=cached_val["text"],
                citations=cached_val["citations"],
                refused=cached_val["refused"],
                refusal_reason=cached_val.get("refusal_reason"),
                trace={**trace, "cached": True},
            )

    trace["cache_hit"] = False

    # 5. LLM execution
    if llm is None:
        from adaptive_retriever.llm import get_llm_client
        if cfg is not None:
            llm = get_llm_client(cfg)

    if llm is None:
        return GeneratedAnswer(
            text="",
            refused=True,
            refusal_reason="No LLM client available for answer generation.",
            trace=trace,
        )

    try:
        old_llm_cache_get = None
        if ignore_cache and hasattr(llm, "cache") and llm.cache is not None and hasattr(llm.cache, "get"):
            old_llm_cache_get = llm.cache.get
            llm.cache.get = lambda k: None

        try:
            if hasattr(llm, "complete"):
                res = llm.complete(formatted_prompt, prompt_version=prompt_version, stage="generation")
                raw_response = res.text if hasattr(res, "text") else str(res)
            elif hasattr(llm, "generate"):
                raw_response = llm.generate(formatted_prompt)
            elif callable(llm):
                raw_response = llm(formatted_prompt)
            else:
                raise AttributeError("LLM client does not support complete() or generate()")
        finally:
            if old_llm_cache_get is not None:
                llm.cache.get = old_llm_cache_get
        text_ans = raw_response.strip()
    except Exception as err:
        logger.warning("LLM generation failed: %s", err)
        return GeneratedAnswer(
            text="",
            refused=True,
            refusal_reason=f"LLM generation failed: {err}",
            trace=trace,
        )

    if not text_ans:
        return GeneratedAnswer(
            text="",
            refused=True,
            refusal_reason="LLM produced an empty response string.",
            trace=trace,
        )

    # Clean leaked bracketed chunk IDs if citation_mode is attribution
    if citation_mode == "attribution":
        text_ans = re.sub(r"\[[a-zA-Z0-9_\-\.:]+\]:?", "", text_ans)
        text_ans = re.sub(r"\s+", " ", text_ans).strip()

    # Check for explicit refusal string produced by LLM
    if "context is insufficient" in text_ans.lower():
        ans = GeneratedAnswer(
            text=text_ans,
            citations=[],
            refused=True,
            refusal_reason="LLM stated context is insufficient.",
            trace=trace,
        )
        cache_set(cache_key, {
            "text": ans.text,
            "citations": ans.citations,
            "refused": ans.refused,
            "refusal_reason": ans.refusal_reason,
        })
        return ans

    # 6. Parse / Derive citations based on citation_mode
    cite_chunks_enabled = gen_cfg.get("cite_chunks", True)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_ans) if s.strip()]
    unsupported_sentences = 0

    if citation_mode == "model":
        strict = gen_cfg.get("strict_citations", False)
        try:
            citations = parse_citations_defensive(text_ans, valid_chunk_ids, strict=strict)
            if citations:
                trace["citation_parse_status"] = "success"
            else:
                trace["citation_parse_status"] = "no_citations"
        except ValueError as err:
            raise err
        except Exception as parse_err:
            trace["citation_parse_status"] = f"error: {parse_err}"
            citations = []

        # Count unsupported sentences for reporting parity
        for s in sentences:
            attr = attribute_sentence_to_chunks(s, candidates, min_n_gram=8)
            if not attr:
                unsupported_sentences += 1
    else:
        # Post-hoc attribution mode
        citations = []
        seen = set()
        for s in sentences:
            attr = attribute_sentence_to_chunks(s, candidates, min_n_gram=8)
            if attr:
                if attr not in seen:
                    seen.add(attr)
                    citations.append(attr)
            else:
                unsupported_sentences += 1
        trace["citation_parse_status"] = "derived_attribution"

    trace["sentence_count"] = len(sentences)
    trace["unsupported_sentence_count"] = unsupported_sentences
    trace["unsupported_sentence_pct"] = (
        (unsupported_sentences / len(sentences) * 100.0) if sentences else 0.0
    )

    refused = False
    refusal_reason = None
    if cite_chunks_enabled and not citations:
        refused = True
        refusal_reason = (
            "Answer contained no resolvable citations."
            if citation_mode == "model"
            else "Answer contained no resolvable citations via post-hoc attribution."
        )

    ans = GeneratedAnswer(
        text=text_ans,
        citations=citations,
        refused=refused,
        refusal_reason=refusal_reason,
        trace=trace,
    )

    # 7. Store in cache
    cache_set(cache_key, {
        "text": ans.text,
        "citations": ans.citations,
        "refused": ans.refused,
        "refusal_reason": ans.refusal_reason,
    })

    return ans
