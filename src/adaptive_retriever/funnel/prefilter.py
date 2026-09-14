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


def _extract_profile_dict(profile: Any) -> Dict[str, Any]:
    raw_dict: Dict[str, Any] = {}

    if isinstance(profile, dict):
        if "extracted_filters" in profile and isinstance(profile["extracted_filters"], dict):
            raw_dict.update(profile["extracted_filters"])
        for k, v in profile.items():
            if k != "extracted_filters":
                raw_dict[k] = v
    else:
        if hasattr(profile, "extracted_filters"):
            ef = getattr(profile, "extracted_filters")
            if isinstance(ef, dict):
                raw_dict.update(ef)
        if hasattr(profile, "model_dump"):
            dump = profile.model_dump()
            if "extracted_filters" in dump and isinstance(dump["extracted_filters"], dict):
                raw_dict.update(dump["extracted_filters"])
            raw_dict.update(dump)
        elif hasattr(profile, "__dict__"):
            raw_dict.update(profile.__dict__)

        for field in INDEXED_FIELDS:
            if hasattr(profile, field):
                val = getattr(profile, field)
                if val is not None:
                    raw_dict[field] = val

    return raw_dict


def build_prefilter(
    profile: Any,
    cfg: Optional[Any] = None,
) -> Tuple[Optional[rest.Filter], Dict[str, Any]]:
    """Return a Qdrant filter built ONLY from indexed payload fields, plus a trace dict."""
    if cfg is not None:
        if hasattr(cfg, "raw"):
            raw_cfg = cfg.raw
        elif isinstance(cfg, dict):
            if "funnel" in cfg:
                raw_cfg = cfg
            elif "prefilter" in cfg:
                raw_cfg = {"funnel": cfg}
            else:
                raw_cfg = {"funnel": {"prefilter": cfg}}
        else:
            raw_cfg = {"funnel": {"prefilter": {"enabled": True}}}
    else:
        raw_cfg = {"funnel": {"prefilter": {"enabled": True}}}

    enabled = raw_cfg.get("funnel", {}).get("prefilter", {}).get("enabled", True)
    if not enabled:
        return None, {"skipped": True, "reason": "prefilter_disabled", "applied_constraints": {}}

    candidates = _extract_profile_dict(profile)
    must_conditions = []
    applied_constraints: Dict[str, Any] = {}
    unindexed_skipped: Dict[str, Any] = {}

    for key, val in candidates.items():
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
