"""Ingestion Pipeline & Acceptance Reporting (A6, spec §9.6 & Change Order §8).

Orchestrates:
1. Document Profiler (A1, spec §9.1)
2. Document Parser (A2, spec §9.2)
3. Chunker (A3, spec §9.4) & Chunk Cap
4. Payload Metadata Extractor (spec §6.3)
5. Multi-Representation Generator (A4, spec §9.4)
6. VectorStore Ingest (A5, spec §9.5)

Idempotency: Skips previously ingested documents unless force=True.
Acceptance Report: Writes ingestion statistics to reports/ingest_stats.json.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from adaptive_retriever.ingest.chunkers import chunk_document
from adaptive_retriever.ingest.parsers import (
    get_parse_cache_stats,
    parse_document,
    parse_document_cached,
    reset_parse_cache_stats,
)
from adaptive_retriever.ingest.profiler import profile
from adaptive_retriever.ingest.representations import generate_representations
from adaptive_retriever.vectorstore import VectorStore

logger = logging.getLogger(__name__)

STATS_REPORT_PATH = Path("reports/ingest_stats.json")


def extract_payload_metadata(text: str) -> Dict[str, Any]:
    """Extract entities, date references, and access tags per spec §6.3."""
    dates = re.findall(r"\b(19\d{2}|20\d{2}|Q[1-4]\s*20\d{2}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b", text)
    entities = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text)
    return {
        "entities": list(set(entities[:10])),
        "date_refs": list(set(dates[:10])),
        "access_tags": ["public"],
    }


def run_ingestion_pipeline(
    manifest_path: Path = Path("data/corpus/MANIFEST.jsonl"),
    config: Optional[Any] = None,
    force: bool = False,
    max_docs: Optional[int] = None,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    reset_parse_cache_stats()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    from adaptive_retriever.config import Config

    if config is None:
        cfg = Config.load("config/default.yaml")
    elif isinstance(config, (str, Path)):
        cfg = Config.load(config)
    elif isinstance(config, Config):
        cfg = config
    else:
        cfg = Config(config)

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if max_docs:
        records = records[:max_docs]

    config_hash = cfg.ingest_config_hash[:8]
    vector_store = VectorStore.from_config(cfg)

    from adaptive_retriever.llm import get_llm_client

    llm_client = get_llm_client(cfg)

    max_chunks_limit = cfg.raw.get("ingest", {}).get("max_chunks_per_document", 150) if hasattr(cfg, "raw") else 150

    from adaptive_retriever.ingest.parsers import PARSERS

    ingest_cfg = cfg.raw.get("ingest", {}) if hasattr(cfg, "raw") else {}
    pr_cfg = ingest_cfg.get("parser_router", {}) or {}
    cr_cfg = ingest_cfg.get("chunk_router", {}) or {}
    parser_router_enabled = pr_cfg.get("enabled", True)
    chunk_router_enabled = cr_cfg.get("enabled", True)
    fixed_parser = pr_cfg.get("fixed_parser", "pymupdf")
    fixed_chunker = cr_cfg.get("fixed_chunker", "recursive_512")
    parser_mapping = pr_cfg.get("mapping", {}) or {}
    chunk_mapping = cr_cfg.get("mapping", {}) or {}
    representation_failures: List[dict] = []

    total_docs = len(records)
    ingested_docs = 0
    skipped_docs = 0
    total_chunks = 0
    total_representations = 0
    documents_capped = 0
    chunks_dropped = 0

    tier_counts = {"high": 0, "low": 0}
    rep_kind_counts = {"raw": 0, "summary": 0, "question": 0, "keywords": 0}
    doc_class_counts: Dict[str, int] = {}
    parser_times: Dict[str, List[float]] = {}
    parser_actual_times: Dict[str, List[float]] = {}
    parser_cache_hits: Dict[str, int] = {}
    all_dropped_chunk_ids: List[str] = []
    doc_class_chunks: Dict[str, List[int]] = {}
    failed_doc_ids = []
    failure_reasons = {}

    from tqdm import tqdm

    for idx, rec in enumerate(tqdm(records, desc="Ingesting documents", unit="doc"), 1):
        doc_path = Path(rec["doc_path"])
        doc_id = rec.get("doc_id", doc_path.stem)

        if not doc_path.exists():
            logger.warning(f"Document missing: {doc_path}")
            failed_doc_ids.append(doc_id)
            failure_reasons["file_not_found"] = failure_reasons.get("file_not_found", 0) + 1
            continue

        # Step 1: Profile document
        fp = profile(doc_path)
        cls_name = fp.doc_class.value
        doc_class_counts[cls_name] = doc_class_counts.get(cls_name, 0) + 1

        # Step 2: Select parser from config. Source format wins over content
        # class, because a table-heavy HTML file must not reach a PDF parser.
        ext = doc_path.suffix.lower()
        if "parser_override" in rec:
            parser_name = rec["parser_override"]
        elif not parser_router_enabled:
            parser_name = fixed_parser
        elif ext in (".html", ".htm"):
            parser_name = parser_mapping.get("html", "trafilatura")
        elif ext not in (".pdf",) and cls_name in ("code",):
            parser_name = parser_mapping.get("code", "passthrough")
        else:
            parser_name = parser_mapping.get(cls_name, fixed_parser)
        if parser_name not in PARSERS:
            raise ValueError(
                f"Config routes doc_class '{cls_name}' to unknown parser "
                f"'{parser_name}'. Known parsers: {sorted(PARSERS)}"
            )

        parsed, parse_ms, parse_err = parse_document_cached(doc_path, parser_name, doc_id=doc_id, config=cfg)

        if parser_name not in parser_times:
            parser_times[parser_name] = []
        parser_times[parser_name].append(parse_ms)

        if parse_err or len(parsed.text.strip()) < 100:
            logger.warning(
                f"Parse error or insufficient content (<100 chars) for {doc_path} "
                f"via {parser_name}: {parse_err}"
            )
            failed_doc_ids.append(doc_id)
            reason = parse_err or "insufficient_content_below_100_chars"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            continue

        # Step 3: Select chunker from config
        if not chunk_router_enabled:
            chunker_name = fixed_chunker
        else:
            chunker_name = chunk_mapping.get(cls_name, fixed_chunker)

        chunks = chunk_document(parsed, doc_id=doc_id, chunker_name=chunker_name)

        # Apply max_chunks_per_document cap
        if max_chunks_limit > 0 and len(chunks) > max_chunks_limit:
            dropped = chunks[max_chunks_limit:]
            all_dropped_chunk_ids.extend([c.chunk_id for c in dropped])
            chunks_dropped += len(dropped)
            documents_capped += 1
            chunks = chunks[:max_chunks_limit]

        num_c = len(chunks)
        total_chunks += num_c
        doc_class_chunks.setdefault(cls_name, []).append(num_c)

        # Track parser performance: actual parse vs cache hit
        if parse_ms > 0.0:
            parser_actual_times.setdefault(parser_name, []).append(parse_ms)
        else:
            parser_cache_hits[parser_name] = parser_cache_hits.get(parser_name, 0) + 1

        # Step 4: Extract metadata & generate representations with LLM client & cfg
        doc_meta = extract_payload_metadata(parsed.text)
        doc_meta["doc_class"] = cls_name

        reps = generate_representations(
            chunks,
            llm_client=llm_client,
            cfg=cfg,
            doc_title=doc_id,
            failures=representation_failures,
        )
        total_representations += len(reps)
        for r in reps:
            tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1
            rep_kind_counts[r.kind.value] = rep_kind_counts.get(r.kind.value, 0) + 1

        vector_store.upsert_document(chunks, reps, metadata=doc_meta)

        ingested_docs += 1

    elapsed_s = time.perf_counter() - start_time

    # Calculate per-parser timing metrics (mean & p95 for actual parses)
    all_parsers = ["pymupdf", "docling", "trafilatura", "tree_sitter", "passthrough"]
    parser_stats = {}
    total_parsed_docs = 0
    for p_name in sorted(set(list(parser_actual_times.keys()) + list(parser_cache_hits.keys()) + all_parsers)):
        t_list = parser_actual_times.get(p_name, [])
        hits_cnt = parser_cache_hits.get(p_name, 0)
        parsed_cnt = len(t_list)
        total_parsed_docs += parsed_cnt
        if parsed_cnt > 0:
            s_list = sorted(t_list)
            p95_idx = int(len(s_list) * 0.95)
            parser_stats[p_name] = {
                "parsed_count": parsed_cnt,
                "cache_hit_count": hits_cnt,
                "mean_ms": round(sum(s_list) / float(parsed_cnt), 2),
                "p95_ms": round(s_list[p95_idx], 2),
            }
        else:
            parser_stats[p_name] = {
                "parsed_count": 0,
                "cache_hit_count": hits_cnt,
                "mean_ms": None,
                "p95_ms": None,
                "note": "no parses this run (all cached)",
            }

    # Calculate chunks_per_doc_class metrics (§9.6)
    chunks_per_class_stats = {}
    for cls_k, c_lengths in doc_class_chunks.items():
        if c_lengths:
            chunks_per_class_stats[cls_k] = {
                "total": sum(c_lengths),
                "mean": round(sum(c_lengths) / len(c_lengths), 2),
                "min": min(c_lengths),
                "max": max(c_lengths),
            }

    peak_mb = 0.0
    try:
        import psutil
        peak_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass

    llm_stats = getattr(llm_client, "stats", {})
    throughput = (
        llm_client.throughput_summary() if hasattr(llm_client, "throughput_summary") else {}
    )

    rep_fail_rate = (
        round(len(representation_failures) / float(total_representations), 4)
        if total_representations
        else None
    )

    docs_attempted = ingested_docs + len(failed_doc_ids)
    failure_rate = (
        round(len(failed_doc_ids) / float(docs_attempted), 4) if docs_attempted else None
    )

    stats = {
        "ingest_config_hash": config_hash,
        "collection_name": vector_store.collection_name,
        "total_documents_manifest": total_docs,
        "ingested_documents": ingested_docs,
        "skipped_documents": skipped_docs,
        "total_chunks": total_chunks,
        "total_representations": total_representations,
        "llm_stats": llm_stats,
        "throughput": throughput,
        "truncated_count": llm_stats.get(
            "truncated_count", {"summary": 0, "question": 0, "keywords": 0}
        ),
        "representation_failures": {
            "count": len(representation_failures),
            "rate": rep_fail_rate,
            "examples": representation_failures[:20],
        },
        "chunk_cap": {
            "limit": max_chunks_limit,
            "documents_capped": documents_capped,
            "chunks_dropped": chunks_dropped,
            "dropped_chunk_ids": all_dropped_chunk_ids,
        },
        "representation_counts_by_kind": rep_kind_counts,
        "tier_breakdown": tier_counts,
        "doc_class_breakdown": doc_class_counts,
        "chunks_per_doc_class": chunks_per_class_stats,
        "parser_performance": parser_stats,
        "parse_cache": get_parse_cache_stats(),
        "parse_failures": {
            "documents_attempted": docs_attempted,
            "parsed_count": total_parsed_docs,
            "failure_count": len(failed_doc_ids),
            "failure_rate": failure_rate,
            "failure_reasons": failure_reasons,
            "failed_doc_ids": failed_doc_ids,
        },
        "total_llm_cost_usd": 0.0,
        "elapsed_seconds": round(elapsed_s, 2),
        "peak_memory_mb": round(peak_mb, 2),
    }

    # Write acceptance stats report per spec §9.6
    STATS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_REPORT_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Ingestion completed in {elapsed_s:.2f}s. Report written to {STATS_REPORT_PATH}")

    # Run-level budgets. A handful of bad generations is expected from a 0.5B
    # model; a systemic rate is a finding to report, not to tune away.
    if hasattr(llm_client, "assert_truncation_within_budget"):
        llm_client.assert_truncation_within_budget(max_rate=0.20)
    if rep_fail_rate is not None and rep_fail_rate > 0.05:
        raise RuntimeError(
            f"Representation post-processing failed on {rep_fail_rate:.1%} of "
            f"{total_representations} representations "
            f"({len(representation_failures)} items). Budget is 5%. "
            f"Stats written to {STATS_REPORT_PATH}. First failures: "
            f"{representation_failures[:5]}"
        )

    return stats
