"""Query Profiler module (spec §9.3).

Analyzes input raw query strings to extract structural features, identifier spans,
and intent flags, producing a QueryFingerprint used by downstream pre-filtering
and weighted hybrid fusion.
"""

import re
from typing import Any, Dict, List
from adaptive_retriever.models import QueryClass, QueryFingerprint


def profile_query(raw_query: str) -> QueryFingerprint:
    """Extract features and classify query into a QueryClass with evidence-based confidence."""
    query = raw_query.strip()
    words = query.split()
    token_count = len(words)

    # 1. Exact identifier spans (quotes, alphanumeric codes containing digits/underscores)
    identifier_spans = re.findall(
        r'"[^"]+"|\b[A-Za-z0-9]*[0-9_][A-Za-z0-9_-]{3,}\b|\b[A-Z0-9_]{5,}\b', query
    )
    has_exact_identifiers = len(identifier_spans) > 0 or bool(
        re.search(r"\b(doc_id|chunk_id|rep_id|uuid)\b", query, re.I)
    )

    # 2. Numeric intent
    has_numeric_intent = bool(
        re.search(
            r"\b\d+(\.\d+)?\b|\b(how many|count|number|percentage|ratio|total|average|sum|score|margin|threshold)\b",
            query,
            re.I,
        )
    )

    # 3. Comparative language
    has_comparative_language = bool(
        re.search(
            r"\b(compare|versus|vs\.?|difference|differ|better|worse|higher|lower|than|compared to)\b",
            query,
            re.I,
        )
    )

    # 4. Extracted filters heuristics
    extracted_filters: Dict[str, Any] = {}
    lower = query.lower()
    if "table" in lower or "tabular" in lower:
        extracted_filters["contains_table"] = True
    if "scanned" in lower:
        extracted_filters["doc_class"] = "scanned"
    elif "code" in lower:
        extracted_filters["doc_class"] = "code"
    elif "prose" in lower or "article" in lower:
        extracted_filters["doc_class"] = "digital_prose"

    # Evidence-backed classification and confidence calculation
    if extracted_filters:
        query_class = QueryClass.FILTERABLE
        confidence = round(min(0.98, max(0.60, 0.65 + 0.15 * len(extracted_filters) + (0.10 if "doc_class" in extracted_filters else 0.0))), 4)
    elif has_comparative_language or (" and " in lower and ("between" in lower or "both" in lower)):
        query_class = QueryClass.MULTIHOP
        confidence = round(min(0.95, max(0.60, 0.60 + (0.25 if has_comparative_language else 0.10) + (0.10 if " and " in lower else 0.0))), 4)
    elif has_numeric_intent and ("total" in lower or "how many" in lower or "count" in lower or "average" in lower):
        query_class = QueryClass.AGGREGATION
        confidence = round(min(0.95, max(0.60, 0.65 + (0.15 if "how many" in lower or "count" in lower else 0.05))), 4)
    elif has_exact_identifiers or (token_count <= 3 and bool(re.search(r"\b(find|lookup)\b", query, re.I))):
        query_class = QueryClass.EXACT_LOOKUP
        confidence = round(min(0.98, max(0.65, 0.65 + 0.20 * len(identifier_spans) + (0.10 if token_count <= 3 else 0.0))), 4)
    elif " and " in lower and ("how" in lower or "what" in lower):
        query_class = QueryClass.COMPOUND
        confidence = round(min(0.90, max(0.55, 0.55 + 0.15 * lower.count(" and ") + (0.10 if token_count > 6 else 0.0))), 4)
    else:
        query_class = QueryClass.CONCEPTUAL
        prose_signal = 1.0 if re.search(r"\b(what|how|why|describe|explain|methods)\b", query, re.I) else 0.0
        confidence = round(min(0.96, max(0.55, 0.55 + 0.25 * prose_signal + 0.15 * min(1.0, token_count / 10.0))), 4)

    return QueryFingerprint(
        raw_query=query,
        token_count=token_count,
        has_exact_identifiers=has_exact_identifiers,
        identifier_spans=identifier_spans,
        has_numeric_intent=has_numeric_intent,
        has_comparative_language=has_comparative_language,
        multihop_signal_score=0.8 if query_class == QueryClass.MULTIHOP else 0.1,
        corpus_vocab_overlap=0.5,
        extracted_filters=extracted_filters,
        query_class=query_class,
        class_confidence=confidence,
    )
