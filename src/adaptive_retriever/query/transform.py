"""Query transformation interface and routing.

Routes query fingerprints to strategy implementations ("none", "hyde", "decompose", "self_query")
based on query.transformation.strategy config or fixed override, with caching and trace logging.
"""

import copy
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from diskcache import Cache

from adaptive_retriever.models import QueryClass

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    strategy: str                 # "none" | "hyde" | "decompose" | "self_query"
    queries: List[str]            # >= 1; ["original"] for "none"
    extracted_filters: Dict[str, Any]
    trace: Dict[str, Any]         # what was done, why, elapsed_ms, cache hit/miss


def _get_transformation_cfg(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        raw_cfg = {"query": {"transformation": {}}}
    elif hasattr(cfg, "raw"):
        raw_cfg = cfg.raw
    elif isinstance(cfg, dict):
        if "query" in cfg:
            raw_cfg = cfg
        else:
            raise ValueError(
                f"Invalid config dict: must contain top-level 'query' key. Got keys: {list(cfg.keys())}"
            )
    else:
        raise ValueError(
            f"Invalid config type '{type(cfg).__name__}'. Expected Config instance or dict containing 'query'."
        )
    return raw_cfg.get("query", {}).get("transformation", {})


def _get_cache_dir(cfg: Any) -> str:
    if cfg is None:
        return "data/cache"
    if hasattr(cfg, "raw"):
        return str(cfg.raw.get("paths", {}).get("cache_dir", "data/cache"))
    if isinstance(cfg, dict):
        return str(cfg.get("paths", {}).get("cache_dir", "data/cache"))
    return "data/cache"


VALID_DOC_CLASSES = {
    "digital_prose",
    "table_heavy",
    "scanned",
    "html",
    "code",
    "transcript",
}

INDEXED_FIELDS = {
    "doc_class",
    "contains_table",
    "entities",
    "date_refs",
    "access_tags",
    "kind",
    "tier",
}


def _execute_self_query(
    raw_query: str,
    initial_filters: Dict[str, Any],
    cfg: Any,
    llm: Optional[Any] = None,
    vs: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute self_query strategy: extract structured filters from query text."""
    candidate_filters = copy.deepcopy(initial_filters)
    rule_extracted: Dict[str, Any] = {}
    rejected_fields: Dict[str, Any] = {}
    dropped_reason = None
    llm_attempted = False
    llm_parse_success = False

    # 1. Rule-based extraction
    q_lower = raw_query.lower()

    if "table_heavy" in q_lower or "table heavy" in q_lower:
        rule_extracted["doc_class"] = "table_heavy"
    elif "scanned" in q_lower:
        rule_extracted["doc_class"] = "scanned"
    elif "digital_prose" in q_lower or "digital prose" in q_lower:
        rule_extracted["doc_class"] = "digital_prose"
    elif "html" in q_lower:
        rule_extracted["doc_class"] = "html"
    elif "source code" in q_lower or "python code" in q_lower or "script" in q_lower:
        rule_extracted["doc_class"] = "code"
    elif "transcript" in q_lower or "interview" in q_lower:
        rule_extracted["doc_class"] = "transcript"

    if any(term in q_lower for term in ["table", "tabular", "chart"]):
        rule_extracted["contains_table"] = True

    years = re.findall(r"\b(19\d{2}|20\d{2})\b", raw_query)
    if years:
        rule_extracted["date_refs"] = years if len(years) > 1 else years[0]

    candidate_filters.update(rule_extracted)

    # 2. LLM fallback if no filters extracted yet and llm is available
    if not candidate_filters and llm is not None:
        llm_attempted = True
        try:
            from adaptive_retriever.prompts import load_prompt
            prompt_template, _ = load_prompt("transform/self_query")
            prompt = prompt_template.format(query=raw_query)

            if hasattr(llm, "generate"):
                response_text = llm.generate(prompt)
            elif hasattr(llm, "complete"):
                response_text = llm.complete(prompt)
            elif callable(llm):
                response_text = llm(prompt)
            else:
                response_text = ""

            if isinstance(response_text, str) and response_text.strip():
                text_str = response_text.strip()
                if "{" in text_str and "}" in text_str:
                    json_str = text_str[text_str.find("{"):text_str.rfind("}") + 1]
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        llm_parse_success = True
                        llm_filters = data.get("filters", data)
                        if isinstance(llm_filters, dict):
                            candidate_filters.update(llm_filters)
        except Exception as e:
            logger.warning("Self-query LLM fallback failed silently: %s", e)
            llm_parse_success = False

    # 3. Filter validation & enforcement against INDEXED_FIELDS & VALID_DOC_CLASSES
    valid_filters: Dict[str, Any] = {}
    for k, v in candidate_filters.items():
        if k not in INDEXED_FIELDS:
            rejected_fields[k] = v
            continue

        if k == "doc_class":
            v_str = str(v.value) if hasattr(v, "value") else str(v)
            if v_str in VALID_DOC_CLASSES:
                valid_filters[k] = v_str
            else:
                rejected_fields[k] = v
        elif k == "contains_table":
            valid_filters[k] = bool(v)
        else:
            valid_filters[k] = v

    # 4. Check 0-match filter drop against VectorStore
    match_count = None
    if vs is not None and valid_filters:
        try:
            from adaptive_retriever.funnel.prefilter import build_prefilter
            filter_obj, _ = build_prefilter({"extracted_filters": valid_filters}, cfg=cfg)
            if filter_obj is not None:
                res_cnt = vs.count(count_filter=filter_obj)
                if isinstance(res_cnt, int):
                    match_count = res_cnt
                    if match_count == 0:
                        dropped_reason = "matches_no_documents"
                        valid_filters = {}
        except Exception as e:
            logger.warning("Error checking filter document count: %s", e)

    trace_details = {
        "candidate_filters": candidate_filters,
        "rule_extracted": rule_extracted,
        "rejected_fields": rejected_fields,
        "llm_attempted": llm_attempted,
        "llm_parse_success": llm_parse_success,
        "match_count": match_count,
    }
    if dropped_reason:
        trace_details["dropped_reason"] = dropped_reason

    return valid_filters, trace_details


def _execute_hyde(
    raw_query: str,
    extracted_filters: Dict[str, Any],
    cfg: Any,
    llm: Optional[Any] = None,
    vs: Optional[Any] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Execute HyDE strategy: generate a hypothetical answer passage."""
    from adaptive_retriever.prompts import load_prompt

    t0 = time.perf_counter()
    trans_cfg = _get_transformation_cfg(cfg)
    threshold = trans_cfg.get("vocab_overlap_threshold", 0.35)

    if llm is None:
        return [raw_query], {"fallback": "llm_unavailable"}

    try:
        prompt_template, version = load_prompt("transform/hyde")
        prompt = prompt_template.format(query=raw_query)

        if hasattr(llm, "complete"):
            resp = llm.complete(prompt, prompt_version=version, max_tokens=128)
            gen_text = resp.text.strip() if hasattr(resp, "text") else str(resp).strip()
            tokens_in = getattr(resp, "tokens_in", 0)
            tokens_out = getattr(resp, "tokens_out", 0)
        elif callable(llm):
            gen_text = str(llm(prompt)).strip()
            tokens_in, tokens_out = len(prompt) // 4, len(gen_text) // 4
        else:
            gen_text = ""
            tokens_in, tokens_out = 0, 0

        if not gen_text:
            return [raw_query], {"fallback": "generation_failed"}

        t1 = time.perf_counter()
        return [gen_text], {
            "status": "success",
            "prompt_version": version,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "generation_time_ms": round((t1 - t0) * 1000, 2),
            "hypothetical_passage": gen_text[:100] + "..." if len(gen_text) > 100 else gen_text,
        }
    except Exception as e:
        logger.warning("HyDE generation failed: %s", e)
        return [raw_query], {"fallback": "generation_failed", "error": str(e)}


def _execute_decompose(
    raw_query: str,
    extracted_filters: Dict[str, Any],
    cfg: Any,
    llm: Optional[Any] = None,
    vs: Optional[Any] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Execute Decompose strategy: split compound query into 2-3 sub-queries."""
    from adaptive_retriever.prompts import load_prompt

    t0 = time.perf_counter()

    if llm is None:
        return [raw_query], {"fallback": "llm_unavailable"}

    try:
        prompt_template, version = load_prompt("transform/decompose")
        prompt = prompt_template.format(query=raw_query)

        if hasattr(llm, "complete"):
            resp = llm.complete(prompt, prompt_version=version, max_tokens=128)
            gen_text = resp.text.strip() if hasattr(resp, "text") else str(resp).strip()
            tokens_in = getattr(resp, "tokens_in", 0)
            tokens_out = getattr(resp, "tokens_out", 0)
        elif callable(llm):
            gen_text = str(llm(prompt)).strip()
            tokens_in, tokens_out = len(prompt) // 4, len(gen_text) // 4
        else:
            gen_text = ""
            tokens_in, tokens_out = 0, 0

        if not gen_text:
            return [raw_query], {"fallback": "generation_failed"}

        lines = [line.strip() for line in gen_text.splitlines() if line.strip()]
        candidates: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*|^[-*]\s*", "", line).strip()
            if cleaned:
                candidates.append(cleaned)

        raw_norm = set(re.findall(r"\w+", raw_query.lower()))
        valid_subqueries: List[str] = []
        rejected_subqueries: List[Tuple[str, str]] = []

        for sub_q in candidates:
            if len(valid_subqueries) >= 3:
                break
            sub_norm = set(re.findall(r"\w+", sub_q.lower()))

            inter = len(raw_norm & sub_norm)
            union = len(raw_norm | sub_norm)
            jaccard_orig = inter / union if union > 0 else 0.0

            if jaccard_orig >= 0.85 or sub_q.lower() == raw_query.lower():
                rejected_subqueries.append((sub_q, "duplicate_of_original"))
                continue

            is_dup = False
            for prev_q in valid_subqueries:
                prev_norm = set(re.findall(r"\w+", prev_q.lower()))
                p_inter = len(prev_norm & sub_norm)
                p_union = len(prev_norm | sub_norm)
                p_jaccard = p_inter / p_union if p_union > 0 else 0.0
                if p_jaccard >= 0.85:
                    is_dup = True
                    break

            if is_dup:
                rejected_subqueries.append((sub_q, "duplicate_subquery"))
                continue

            valid_subqueries.append(sub_q)

        if not valid_subqueries:
            return [raw_query], {
                "fallback": "all_subqueries_rejected",
                "rejected_subqueries": rejected_subqueries,
                "raw_generation": gen_text,
            }

        t1 = time.perf_counter()
        return valid_subqueries, {
            "status": "success",
            "prompt_version": version,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "subquery_count": len(valid_subqueries),
            "rejected_subqueries": rejected_subqueries,
            "generation_time_ms": round((t1 - t0) * 1000, 2),
        }
    except Exception as e:
        logger.warning("Decomposition generation failed: %s", e)
        return [raw_query], {"fallback": "generation_failed", "error": str(e)}


def transform_query(
    fingerprint: Any,
    cfg: Any,
    llm: Optional[Any] = None,
    vs: Optional[Any] = None,
) -> TransformResult:
    """Route query fingerprint to transformation strategy based on config strategy or fixed override."""
    t0 = time.perf_counter()
    trans_cfg = _get_transformation_cfg(cfg)

    enabled = trans_cfg.get("enabled", True)
    fixed = trans_cfg.get("fixed", "none")
    configured_strategy = trans_cfg.get("strategy", "none")

    # Determine strategy (fixed override vs global configured strategy)
    if fixed and fixed != "none":
        strategy = fixed
    else:
        strategy = configured_strategy

    # Extract raw_query, query_class, and extracted_filters from fingerprint
    raw_query = ""
    query_cls = None
    extracted_filters: Dict[str, Any] = {}

    if hasattr(fingerprint, "raw_query"):
        raw_query = fingerprint.raw_query
    elif isinstance(fingerprint, dict) and "raw_query" in fingerprint:
        raw_query = str(fingerprint["raw_query"])
    elif isinstance(fingerprint, str):
        raw_query = fingerprint

    if hasattr(fingerprint, "query_class"):
        qc = fingerprint.query_class
        query_cls = qc.value if hasattr(qc, "value") else str(qc)
    elif isinstance(fingerprint, dict) and "query_class" in fingerprint:
        qc = fingerprint["query_class"]
        query_cls = qc.value if hasattr(qc, "value") else str(qc)

    if hasattr(fingerprint, "extracted_filters") and isinstance(fingerprint.extracted_filters, dict):
        extracted_filters = copy.deepcopy(fingerprint.extracted_filters)
    elif isinstance(fingerprint, dict) and "extracted_filters" in fingerprint and isinstance(fingerprint["extracted_filters"], dict):
        extracted_filters = copy.deepcopy(fingerprint["extracted_filters"])

    # 1. Enabled short-circuit
    if not enabled:
        t1 = time.perf_counter()
        elapsed_ms = round((t1 - t0) * 1000, 2)
        return TransformResult(
            strategy="none",
            queries=[raw_query] if raw_query else [],
            extracted_filters={},
            trace={
                "skipped": True,
                "reason": "transformation_disabled",
                "elapsed_ms": elapsed_ms,
                "cache_hit": False,
            },
        )

    # 3. Strategy 'none'
    if strategy == "none":
        t1 = time.perf_counter()
        elapsed_ms = round((t1 - t0) * 1000, 2)
        return TransformResult(
            strategy="none",
            queries=[raw_query] if raw_query else [],
            extracted_filters={},
            trace={
                "skipped": False,
                "reason": "strategy_none",
                "elapsed_ms": elapsed_ms,
                "cache_hit": False,
            },
        )

    # 4. Strategy 'hyde', 'decompose', 'self_query' caching & dispatch
    cache_dir = _get_cache_dir(cfg)
    cache = Cache(directory=cache_dir)
    model_name = getattr(llm, "model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    prompt_version = "1.0"

    cache_key_data = {
        "raw_query": raw_query,
        "strategy": strategy,
        "model": model_name,
        "prompt_version": prompt_version,
    }
    key_bytes = json.dumps(cache_key_data, sort_keys=True).encode("utf-8")
    cache_key = hashlib.sha256(key_bytes).hexdigest()

    cached_val = cache.get(cache_key)
    if cached_val is not None and isinstance(cached_val, dict):
        t1 = time.perf_counter()
        elapsed_ms = round((t1 - t0) * 1000, 2)
        det = cached_val.get("details", {})
        return TransformResult(
            strategy=strategy,
            queries=cached_val.get("queries", [raw_query]),
            extracted_filters=cached_val.get("extracted_filters", {}),
            trace={
                "skipped": False,
                "elapsed_ms": elapsed_ms,
                "cache_hit": True,
                "details": det,
                "dropped_reason": det.get("dropped_reason"),
                "rejected_fields": det.get("rejected_fields"),
                "candidate_filters": det.get("candidate_filters"),
                "match_count": det.get("match_count"),
            },
        )

    # Dispatch to specific strategy
    res_queries = [raw_query]
    if strategy == "self_query":
        res_filters, res_details = _execute_self_query(
            raw_query, extracted_filters, cfg=cfg, llm=llm, vs=vs
        )
    elif strategy == "hyde":
        res_queries, res_details = _execute_hyde(
            raw_query, extracted_filters, cfg=cfg, llm=llm, vs=vs
        )
        res_filters = copy.deepcopy(extracted_filters)
    elif strategy == "decompose":
        res_queries, res_details = _execute_decompose(
            raw_query, extracted_filters, cfg=cfg, llm=llm, vs=vs
        )
        res_filters = copy.deepcopy(extracted_filters)
    else:
        res_filters = copy.deepcopy(extracted_filters)
        res_details = {"status": f"stub_{strategy}"}

    t1 = time.perf_counter()
    elapsed_ms = round((t1 - t0) * 1000, 2)

    # Write result to cache
    cache_payload = {
        "queries": res_queries,
        "extracted_filters": res_filters,
        "details": res_details,
    }
    cache.set(cache_key, cache_payload)

    return TransformResult(
        strategy=strategy,
        queries=res_queries,
        extracted_filters=res_filters,
        trace={
            "skipped": False,
            "elapsed_ms": elapsed_ms,
            "cache_hit": False,
            "details": res_details,
            "dropped_reason": res_details.get("dropped_reason"),
            "rejected_fields": res_details.get("rejected_fields"),
            "candidate_filters": res_details.get("candidate_filters"),
            "match_count": res_details.get("match_count"),
        },
    )
