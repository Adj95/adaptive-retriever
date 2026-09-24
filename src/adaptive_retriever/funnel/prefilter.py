"""Structured pre-filter (stage 1 of retrieval funnel)."""

from typing import Any, Dict, Optional, Tuple
from qdrant_client.http import models as rest

INDEXED_FIELDS = {
    "doc_class",
    "contains_table",
    "entities",
    "date_refs",
    "access_tags",
    "kind",
    "tier",
}


def build_prefilter(
    profile: Any,
    cfg: Optional[Any] = None,
) -> Tuple[Optional[rest.Filter], Dict[str, Any]]:
    """Return a Qdrant filter built ONLY from indexed payload fields, plus a trace dict."""
    if cfg is None:
        raw_cfg = {"funnel": {"prefilter": {"enabled": True}}}
    elif hasattr(cfg, "raw"):
        raw_cfg = cfg.raw
    elif isinstance(cfg, dict):
        if "funnel" in cfg:
            raw_cfg = cfg
        else:
            raise ValueError(
                f"Invalid config dict: must contain top-level 'funnel' key. Got keys: {list(cfg.keys())}"
            )
    else:
        raise ValueError(
            f"Invalid config type '{type(cfg).__name__}'. Expected Config instance or dict containing 'funnel'."
        )

    prefilter_cfg = raw_cfg.get("funnel", {}).get("prefilter", {})
    enabled = prefilter_cfg.get("enabled", True)

    if not enabled:
        return None, {
            "skipped": True,
            "reason": "prefilter_disabled",
            "applied_constraints": {},
        }

    # Extract constraints strictly from extracted_filters (or plain dict)
    if hasattr(profile, "extracted_filters") and isinstance(profile.extracted_filters, dict):
        raw_filters = profile.extracted_filters
    elif isinstance(profile, dict):
        if "extracted_filters" in profile and isinstance(profile["extracted_filters"], dict):
            raw_filters = profile["extracted_filters"]
        else:
            raw_filters = profile
    else:
        raw_filters = {}

    must_conditions = []
    applied_constraints: Dict[str, Any] = {}
    unindexed_skipped: Dict[str, Any] = {}

    for key, val in raw_filters.items():
        if val is None:
            continue

        if key not in INDEXED_FIELDS:
            unindexed_skipped[key] = val
            continue

        if key == "contains_table":
            b_val = bool(val)
            match = rest.MatchValue(value=b_val)
            must_conditions.append(rest.FieldCondition(key=key, match=match))
            applied_constraints[key] = b_val
        elif isinstance(val, list):
            if len(val) == 0:
                continue
            elif len(val) == 1:
                elem = val[0].value if hasattr(val[0], "value") else str(val[0])
                match = rest.MatchValue(value=elem)
                applied_constraints[key] = elem
            else:
                elems = [v.value if hasattr(v, "value") else str(v) for v in val]
                match = rest.MatchAny(any=elems)
                applied_constraints[key] = elems
            must_conditions.append(rest.FieldCondition(key=key, match=match))
        else:
            v_str = val.value if hasattr(val, "value") else str(val)
            match = rest.MatchValue(value=v_str)
            must_conditions.append(rest.FieldCondition(key=key, match=match))
            applied_constraints[key] = v_str

    kind_filter = prefilter_cfg.get("kind_filter", None)
    if kind_filter and kind_filter != "all" and "kind" not in applied_constraints:
        match = rest.MatchValue(value=str(kind_filter))
        must_conditions.append(rest.FieldCondition(key="kind", match=match))
        applied_constraints["kind"] = str(kind_filter)

    if not must_conditions:
        return None, {
            "skipped": False,
            "reason": "no_indexed_constraints",
            "applied_constraints": {},
            "unindexed_skipped": unindexed_skipped,
        }

    filter_obj = rest.Filter(must=must_conditions)
    trace = {
        "skipped": False,
        "applied_constraints": applied_constraints,
        "unindexed_skipped": unindexed_skipped,
    }
    return filter_obj, trace
