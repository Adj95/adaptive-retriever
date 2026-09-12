"""Configuration loading, validation, and hash calculation."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict
import yaml

logger = logging.getLogger(__name__)

def canonicalize_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: canonicalize_dict(v) for k, v in sorted(d.items())}
    elif isinstance(d, list):
        return [canonicalize_dict(x) for x in d]
    return d

def compute_hash(data: Dict[str, Any]) -> str:
    canonical = canonicalize_dict(data)
    json_bytes = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()

class Config:
    def __init__(self, config_dict: Dict[str, Any], path: Path | None = None):
        self.raw = config_dict
        self.path = path
        self.validate()

    @classmethod
    def load(cls, path: Path | str) -> "Config":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found at {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data, path=p)

    def validate(self) -> None:
        # Rule 1: UNCONDITIONAL PROFILER RULE (Addendum A1/A2)
        profiler_enabled = (
            self.raw.get("ingest", {})
            .get("profiler", {})
            .get("enabled", True)
        )
        if not profiler_enabled:
            raise ValueError(
                "Unconditional profiler rule violation: 'ingest.profiler.enabled' "
                "must be true across ALL configuration files."
            )

        # Rule 2: Judge position swap rule (§7.1)
        judge_cfg = self.raw.get("eval", {}).get("judge", {})
        if judge_cfg.get("enabled", False) and not judge_cfg.get("position_swap", False):
            raise ValueError(
                "Uncorrected pairwise judging is forbidden: 'eval.judge.position_swap' "
                "must be true when judge is enabled."
            )

        # Rule 3: Escalation vs rerank warning
        escalation_enabled = self.raw.get("escalation", {}).get("enabled", False)
        rerank_mode = self.raw.get("funnel", {}).get("rerank", {}).get("mode", "gated")
        if escalation_enabled and rerank_mode == "always":
            logger.warning(
                "Escalation enabled while funnel.rerank.mode is 'always': escalation rung 1 (enable_rerank) will be a no-op."
            )

        # Rule 4: Router fixed fallbacks
        ingest_cfg = self.raw.get("ingest", {})
        if not ingest_cfg.get("parser_router", {}).get("enabled", True):
            if not ingest_cfg.get("parser_router", {}).get("fixed_parser"):
                raise ValueError("parser_router.enabled is false but fixed_parser is not set.")
        if not ingest_cfg.get("chunk_router", {}).get("enabled", True):
            if not ingest_cfg.get("chunk_router", {}).get("fixed_chunker"):
                raise ValueError("chunk_router.enabled is false but fixed_chunker is not set.")

    @property
    def config_hash(self) -> str:
        return compute_hash(self.raw)

    @property
    def ingest_config_hash(self) -> str:
        llm_cfg = dict(self.raw.get("models", {}).get("llm", {}))
        llm_cfg.pop("device", None)
        llm_cfg.pop("batch_size", None)
        llm_cfg.pop("dtype", None)

        ingest_subtree = {
            "ingest": self.raw.get("ingest", {}),
            "dense_embedding": self.raw.get("models", {}).get("dense_embedding"),
            "llm": llm_cfg,
        }
        return compute_hash(ingest_subtree)

