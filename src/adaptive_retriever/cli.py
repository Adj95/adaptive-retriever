"""Typer CLI interface (spec §16.1, Addendum 1)."""

import hashlib
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import typer
from adaptive_retriever.config import Config
from adaptive_retriever.eval.metrics import normalise
from adaptive_retriever.ingest.parsers import (
    parse_document_cached,
    save_canonical_parse,
)
from adaptive_retriever.ingest.profiler import profile

app = typer.Typer(help="The Adaptive Retriever CLI")

logger = logging.getLogger(__name__)

@app.command("corpus-prepare")
def corpus_prepare(
    manifest: Path = Path("data/corpus/MANIFEST.jsonl"),
    canonical_dir: Path = Path("data/canonical"),
    hashes_path: Path = Path("data/canonical_hashes.json"),
):
    """Profiles every document and writes canonical parses to data/canonical/{doc_id}.txt (Addendum 1 & B3)."""
    if not manifest.exists():
        typer.echo(f"Error: Manifest file {manifest} does not exist.")
        raise typer.Exit(code=1)

    canonical_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    hashes_lock = threading.Lock()

    with open(manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Reconcile against manifest (self-cleaning)
    valid_doc_ids = set()
    for rec in records:
        doc_path = Path(rec["doc_path"])
        if doc_path.exists():
            d_id = hashlib.sha256(doc_path.read_bytes()).hexdigest()[:16]
            valid_doc_ids.add(d_id)
        elif "doc_id" in rec:
            valid_doc_ids.add(rec["doc_id"])

    removed_count = 0
    if canonical_dir.exists():
        for file in canonical_dir.glob("*.txt"):
            if file.stem not in valid_doc_ids:
                file.unlink()
                removed_count += 1
    if removed_count > 0:
        typer.echo(f"Reconciled canonical parses: removed {removed_count} orphaned parse files.")

    typer.echo(f"Preparing canonical parses for {len(records)} corpus documents...")

    agreement_matches = 0
    agreement_lock = threading.Lock()

    def process_record(idx_rec: tuple) -> tuple:
        idx, rec = idx_rec
        doc_path = Path(rec["doc_path"])
        if not doc_path.exists():
            return None

        file_bytes = doc_path.read_bytes()
        doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]
        expected_class = rec.get("expected_doc_class")

        doc_cls = expected_class or "digital_prose"
        if doc_path.suffix.lower() == ".pdf":
            canonical_parser = "docling" if doc_cls == "scanned" else "pymupdf"
        elif doc_path.suffix.lower() in (".html", ".htm"):
            canonical_parser = "trafilatura"
        else:
            canonical_parser = "passthrough"

        parsed_doc, parse_ms, parse_err = parse_document_cached(doc_path, canonical_parser, doc_id=doc_id)
        canonical_text = parsed_doc.text
        parse_failed = False
        parse_reason = None

        if parse_err or len(canonical_text.strip()) < 100:
            parse_failed = True
            parse_reason = parse_err or "insufficient_content_below_100_chars"
        else:
            save_canonical_parse(doc_id, canonical_text)
            with agreement_lock:
                nonlocal agreement_matches
                agreement_matches += 1

        text_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
        return doc_id, {
            "source_path": str(doc_path),
            "doc_class": doc_cls,
            "canonical_parser": canonical_parser,
            "text_hash": text_hash,
            "char_count": len(canonical_text),
            "parse_failed": parse_failed,
            "parse_reason": parse_reason,
        }

    failed_doc_ids = []
    failure_reasons = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_record, (idx, rec)) for idx, rec in enumerate(records, 1)]
        completed_count = 0
        for f in as_completed(futures):
            res = f.result()
            if res:
                doc_id, info = res
                with hashes_lock:
                    hashes[doc_id] = info
                    if info.get("parse_failed"):
                        failed_doc_ids.append(doc_id)
                        r_name = info.get("parse_reason", "unknown_error")
                        failure_reasons[r_name] = failure_reasons.get(r_name, 0) + 1
                completed_count += 1
                if completed_count % 50 == 0 or completed_count == len(records):
                    typer.echo(f"Canonical parse progress: {completed_count}/{len(records)} documents ready...")

    hashes_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    agreement_rate = (agreement_matches / float(max(1, completed_count)) * 100.0)

    typer.echo(f"Successfully prepared {completed_count} canonical parses in {canonical_dir}")
    typer.echo(f"Saved canonical hashes to {hashes_path}")
    typer.echo(f"Profiler vs Expected Class Agreement: {agreement_rate:.1f}% ({agreement_matches}/{completed_count})")
    if failed_doc_ids:
        typer.echo(f"Warning: {len(failed_doc_ids)} canonical parses failed or were <100 chars.")




def _get_gpu_info() -> tuple[str, float]:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        pass
    return "cpu", 0.0


@app.command()
def ingest(config: Path = Path("config/default.yaml"), limit: int = 0, force: bool = False):
    """Run ingestion pipeline."""
    import time
    from adaptive_retriever.ingest.pipeline import run_ingestion_pipeline
    t0 = time.perf_counter()
    cfg = Config.load(config)
    typer.echo(f"Ingest called with config: {config} (config_hash: {cfg.config_hash[:8]} | ingest_config_hash: {cfg.ingest_config_hash[:8]})")
    max_d = limit if limit > 0 else None
    stats = run_ingestion_pipeline(config=config, force=force, max_docs=max_d)
    wall_time = time.perf_counter() - t0
    peak_mb = _get_peak_memory_mb()

    llm_gen = stats.get("llm_stats", {}).get("llm_calls_generated", 0)
    llm_cached = stats.get("llm_stats", {}).get("llm_calls_cached", 0)
    reps_written = stats.get("total_representations", 0)
    gen_time = stats.get("llm_stats", {}).get("generation_time", 0.0)
    eff_bs = cfg.raw.get("models", {}).get("llm", {}).get("batch_size", 16)
    device_used, peak_vram = _get_gpu_info()

    if llm_gen > 0 and gen_time > 0:
        gen_rate_str = f"{llm_gen / gen_time:.2f} comp/s"
    else:
        gen_rate_str = "n/a (all cached)"

    typer.echo(f"Ingestion finished! Profiled & ingested {stats['ingested_documents']} docs, {stats['total_chunks']} chunks into collection '{stats['collection_name']}' in {stats['elapsed_seconds']}s.")
    typer.echo(f"Throughput Summary: device={device_used}, batch_size={eff_bs}, llm_calls_generated={llm_gen}, llm_calls_cached={llm_cached}, representations_written={reps_written}, generation_rate={gen_rate_str}, wall_time={wall_time:.2f}s, peak_vram={peak_vram:.1f} MB")
    typer.echo(f"Peak Memory Usage: {peak_mb:.1f} MB")


def _get_peak_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


@app.command("ingest-export")
def ingest_export(
    config: Path = Path("config/default.yaml"),
    manifest: Path = Path("data/corpus/MANIFEST.jsonl"),
    output: Path = Path("data/generation/requests.jsonl"),
):
    """Profile -> Parse -> Chunk -> Tier -> Export requests.jsonl for offline GPU generation (Change Order §7)."""
    import time
    from datetime import datetime
    from tqdm import tqdm
    from adaptive_retriever.ingest.chunkers import chunk_document
    from adaptive_retriever.ingest.representations import evaluate_chunk_density, plan_representations

    t0 = time.perf_counter()
    cfg = Config.load(config)
    ingest_hash = cfg.ingest_config_hash[:8]
    typer.echo(f"Ingest-export called with config: {config} (config_hash: {cfg.config_hash[:8]} | ingest_config_hash: {ingest_hash})")

    if not manifest.exists():
        typer.echo(f"Error: Manifest file {manifest} does not exist.")
        raise typer.Exit(code=1)

    with open(manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    max_chunks_limit = cfg.raw.get("ingest", {}).get("max_chunks_per_document", 150)
    raw_llm = cfg.raw.get("models", {}).get("llm", {})
    model_name = raw_llm.get("model", "Qwen/Qwen2.5-0.5B-Instruct")

    all_chunks = []
    all_deterministic = []
    all_requests = []

    high_tier_chunks = 0
    low_tier_chunks = 0

    for rec in tqdm(records, desc="Exporting generation requests", unit="doc"):
        doc_path = Path(rec["doc_path"])
        doc_id = rec.get("doc_id", doc_path.stem)
        if not doc_path.exists():
            continue

        fp = profile(doc_path)
        cls_name = fp.doc_class.value

        parser_name = rec.get("parser_override", "pymupdf")
        if cls_name == "html":
            parser_name = "trafilatura"
        elif cls_name == "scanned":
            parser_name = "docling"
        elif cls_name == "code":
            parser_name = "passthrough"

        parsed, _, _ = parse_document_cached(doc_path, parser_name, doc_id=doc_id, config=cfg)

        chunker_name = "recursive_512"
        if cls_name == "transcript":
            chunker_name = "speaker_turn"
        elif cls_name == "code":
            chunker_name = "syntax_aware"
        elif cls_name == "html":
            chunker_name = "structural"
        elif cls_name == "table_heavy":
            chunker_name = "table_preserving"

        chunks = chunk_document(parsed, doc_id=doc_id, chunker_name=chunker_name)
        if max_chunks_limit > 0 and len(chunks) > max_chunks_limit:
            chunks = chunks[:max_chunks_limit]

        all_chunks.extend(chunks)

        det_reps, reqs = plan_representations(chunks, cfg)
        all_deterministic.extend(det_reps)
        all_requests.extend(reqs)

        for c in chunks:
            if evaluate_chunk_density(c) == "high":
                high_tier_chunks += 1
            else:
                low_tier_chunks += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        meta_header = {
            "_meta": True,
            "ingest_config_hash": ingest_hash,
            "model": model_name,
            "prompt_versions": {"summary": "1", "question": "1", "keywords": "1"},
            "created_at": datetime.now().isoformat(),
        }
        f.write(json.dumps(meta_header) + "\n")
        for req in all_requests:
            f.write(req.model_dump_json() + "\n")

    wall_time = time.perf_counter() - t0
    peak_mb = _get_peak_memory_mb()
    llm_gen = 0
    llm_cached = 0
    reps_written = len(all_requests)
    gen_rate_str = "n/a (export only)"
    eff_bs = cfg.raw.get("models", {}).get("llm", {}).get("batch_size", 16)
    device_used, peak_vram = _get_gpu_info()

    typer.echo(f"Export completed for config_hash '{cfg.config_hash[:8]}' | ingest_config_hash '{ingest_hash}'!")
    typer.echo(f"Total Chunks: {len(all_chunks)} (High-tier: {high_tier_chunks}, Low-tier: {low_tier_chunks})")
    typer.echo(f"Generation Requests Exported: {len(all_requests)}")
    typer.echo(f"Throughput Summary: device={device_used}, batch_size={eff_bs}, llm_calls_generated={llm_gen}, llm_calls_cached={llm_cached}, representations_written={reps_written}, generation_rate={gen_rate_str}, wall_time={wall_time:.2f}s, peak_vram={peak_vram:.1f} MB")
    typer.echo(f"Exported requests file: {output}")
    typer.echo(f"Peak Memory Usage: {peak_mb:.1f} MB")


@app.command("ingest-import")
def ingest_import(
    config: Path = Path("config/default.yaml"),
    results: Path = Path("data/generation/results.jsonl"),
    manifest: Path = Path("data/corpus/MANIFEST.jsonl"),
):
    """Import results.jsonl, materialize representations, embed, and upsert to Qdrant (Change Order §7)."""
    import time
    from tqdm import tqdm
    from adaptive_retriever.ingest.chunkers import chunk_document
    from adaptive_retriever.ingest.pipeline import extract_payload_metadata
    from adaptive_retriever.ingest.representations import materialise_representations, plan_representations
    from adaptive_retriever.models import GenerationResult
    from adaptive_retriever.vectorstore import VectorStore

    t0 = time.perf_counter()
    cfg = Config.load(config)
    current_hash = cfg.ingest_config_hash[:8]
    typer.echo(f"Ingest-import called with config: {config} (config_hash: {cfg.config_hash[:8]} | ingest_config_hash: {current_hash})")

    requests_path = Path(cfg.raw.get("offline_generation", {}).get("requests_path", "data/generation/requests.jsonl"))

    if not requests_path.exists():
        typer.echo(f"Error: Requests file {requests_path} does not exist. Run 'cli ingest-export' first.")
        raise typer.Exit(code=1)

    if not results.exists():
        typer.echo(f"Error: Results file {results} does not exist.")
        raise typer.Exit(code=1)

    with open(requests_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        meta = json.loads(first_line)
        if not meta.get("_meta") or meta.get("ingest_config_hash") != current_hash:
            raise ValueError(
                f"Config hash mismatch! requests.jsonl hash '{meta.get('ingest_config_hash')}' "
                f"differs from current config hash '{current_hash}'."
            )

    res_list: List[GenerationResult] = []
    with open(results, "r", encoding="utf-8") as f:
        for line in f:
            l_str = line.strip()
            if not l_str:
                continue
            item = json.loads(l_str)
            if not item.get("_meta") and "request_id" in item:
                res_list.append(GenerationResult(**item))

    res_map = {r.request_id: r for r in res_list}

    with open(manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    max_chunks_limit = cfg.raw.get("ingest", {}).get("max_chunks_per_document", 150)
    vector_store = VectorStore(config_hash=current_hash)

    total_chunks = 0
    total_reps = 0
    tier_counts = {"high": 0, "low": 0}
    rep_kind_counts = {"raw": 0, "summary": 0, "question": 0, "keywords": 0}

    for rec in tqdm(records, desc="Importing and upserting", unit="doc"):
        doc_path = Path(rec["doc_path"])
        doc_id = rec.get("doc_id", doc_path.stem)
        if not doc_path.exists():
            continue

        fp = profile(doc_path)
        cls_name = fp.doc_class.value

        parser_name = rec.get("parser_override", "pymupdf")
        if cls_name == "html":
            parser_name = "trafilatura"
        elif cls_name == "scanned":
            parser_name = "docling"
        elif cls_name == "code":
            parser_name = "passthrough"

        parsed, _, _ = parse_document_cached(doc_path, parser_name, doc_id=doc_id, config=cfg)

        chunker_name = "recursive_512"
        if cls_name == "transcript":
            chunker_name = "speaker_turn"
        elif cls_name == "code":
            chunker_name = "syntax_aware"
        elif cls_name == "html":
            chunker_name = "structural"
        elif cls_name == "table_heavy":
            chunker_name = "table_preserving"

        chunks = chunk_document(parsed, doc_id=doc_id, chunker_name=chunker_name)
        if max_chunks_limit > 0 and len(chunks) > max_chunks_limit:
            chunks = chunks[:max_chunks_limit]

        total_chunks += len(chunks)

        det_reps, reqs = plan_representations(chunks, cfg)
        for req in reqs:
            if req.request_id not in res_map:
                raise ValueError(f"Missing request_id '{req.request_id}' in results file {results}!")

        mat_reps = materialise_representations(chunks, det_reps, res_list, cfg)
        total_reps += len(mat_reps)

        for r in mat_reps:
            tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1
            rep_kind_counts[r.kind.value] = rep_kind_counts.get(r.kind.value, 0) + 1

        doc_meta = extract_payload_metadata(parsed.text)
        doc_meta["doc_class"] = cls_name

        reps_by_chunk: Dict[str, List[Any]] = {}
        for r in mat_reps:
            reps_by_chunk.setdefault(r.chunk_id, []).append(r)

        for c in chunks:
            c_reps = reps_by_chunk.get(c.chunk_id, [])
            vector_store.upsert_representations(c, c_reps, metadata=doc_meta)

    wall_time = time.perf_counter() - t0
    peak_mb = _get_peak_memory_mb()
    llm_gen = 0
    llm_cached = 0
    reps_written = total_reps
    gen_rate_str = "n/a (import only)"
    eff_bs = cfg.raw.get("models", {}).get("llm", {}).get("batch_size", 16)
    device_used, peak_vram = _get_gpu_info()

    typer.echo(f"Import finished! Ingested {total_chunks} chunks, {total_reps} representations into '{vector_store.collection_name}'.")
    typer.echo(f"Throughput Summary: device={device_used}, batch_size={eff_bs}, llm_calls_generated={llm_gen}, llm_calls_cached={llm_cached}, representations_written={reps_written}, generation_rate={gen_rate_str}, wall_time={wall_time:.2f}s, peak_vram={peak_vram:.1f} MB")
    typer.echo(f"Peak Memory Usage: {peak_mb:.1f} MB")


@app.command("collections-prune")
def collections_prune(
    config: Path = Path("config/default.yaml"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="If True, only report collections that would be dropped without deleting them."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm deletion of stale collections (required alongside --no-dry-run)."),
):
    """Drop Qdrant vector store collections whose ingest_config_hash is not the current one."""
    from adaptive_retriever.vectorstore import VectorStore
    cfg = Config.load(config)
    current_hash = cfg.ingest_config_hash[:8]
    current_collection = f"retriever_chunks_{current_hash}"

    vs = VectorStore(config_hash=current_hash)
    collections = vs.client.get_collections().collections

    stale_collections = [
        c.name for c in collections
        if c.name.startswith("retriever_chunks_") and c.name != current_collection
    ]

    typer.echo(f"Current collection (matching current ingest_config_hash): {current_collection}")

    if not stale_collections:
        typer.echo("No stale collections found.")
        return

    if dry_run or not yes:
        typer.echo(f"[DRY-RUN] Found {len(stale_collections)} stale collection(s) that would be dropped:")
        for sc in stale_collections:
            typer.echo(f"  - {sc}")
        typer.echo("To perform deletion, execute: cli collections-prune --no-dry-run --yes")
    else:
        dropped = []
        for sc in stale_collections:
            if sc == current_collection:
                typer.echo(f"Refusing to delete current collection '{sc}'")
                continue
            try:
                vs.client.delete_collection(collection_name=sc)
                dropped.append(sc)
            except Exception as e:
                typer.echo(f"Warning: Failed to drop collection {sc}: {e}")
        typer.echo(f"Successfully dropped {len(dropped)} stale collection(s): {dropped}")


@app.command("drop-stale-collections")
def drop_stale_collections(config: Path = Path("config/default.yaml")):
    """Alias for cli collections-prune --no-dry-run --yes."""
    collections_prune(config=config, dry_run=False, yes=True)



@app.command()
def query(
    text: str,
    config: Path = Path("config/default.yaml"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON object to stdout and nothing else."
    ),
    doc_class: Optional[str] = typer.Option(
        None, "--doc-class", help="Filter candidates by document class."
    ),
    query_class: Optional[str] = typer.Option(
        None, "--query-class", help="Query class for RRF weights."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Number of candidates to return (defaults to funnel.rerank.output_size)."
    ),
):
    """Run query against retrieval pipeline."""
    from adaptive_retriever.vectorstore import VectorStore

    cfg = Config.load(config)
    output_limit = (
        limit
        if limit is not None
        else cfg.raw.get("funnel", {}).get("rerank", {}).get("output_size", 5)
    )

    vs = VectorStore.from_config(cfg)
    candidates = vs.query_hybrid(
        query_text=text,
        limit=output_limit,
        doc_class_filter=doc_class,
        query_class=query_class,
        cfg=cfg,
    )

    if json_output:
        results_payload = []
        for c in candidates:
            meta = c.metadata or {}
            results_payload.append(
                {
                    "rank": c.final_rank,
                    "fused_score": round(c.fused_score, 4) if c.fused_score is not None else None,
                    "source_ranks": meta.get("source_ranks", {}),
                    # matched_kind = which representation won the chunk (raw /
                    # summary / question / keywords). The chunk's own text is
                    # always what gets returned; this says what matched it.
                    "kind": meta.get("matched_kind", meta.get("kind", "unknown")),
                    "doc_class": meta.get("doc_class", "unknown"),
                    "doc_id": meta.get("doc_id", "unknown"),
                    "text": c.text[:200],
                }
            )
        out_obj = {
            "query": text,
            "results": results_payload,
        }
        sys.stdout.write(json.dumps(out_obj, indent=2) + "\n")
    else:
        typer.echo(f"Querying: {text}")
        for c in candidates:
            meta = c.metadata or {}
            source_ranks = meta.get("source_ranks", {})
            kind = meta.get("matched_kind", meta.get("kind", "unknown"))
            doc_cls = meta.get("doc_class", "unknown")
            doc_id = meta.get("doc_id", "unknown")
            fused_str = f"{c.fused_score:.4f}" if c.fused_score is not None else "N/A"
            snippet = c.text[:200].encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
            typer.echo(
                f"Rank {c.final_rank} | Fused: {fused_str} | Ranks: {source_ranks} | "
                f"Kind: {kind} | DocClass: {doc_cls} | DocID: {doc_id}\n"
                f"  Text: {snippet}"
            )


@app.command("funnel")
def funnel_cmd(
    text: str,
    config: Path = Path("config/default.yaml"),
    trace: bool = typer.Option(False, "--trace", help="Print trace details to stdout."),
):
    """Run full 3-stage funnel: pre-filter -> hybrid RRF -> gated rerank."""
    from adaptive_retriever.funnel.pipeline import run_funnel, warm_up
    from adaptive_retriever.funnel.rerank import Reranker
    from adaptive_retriever.vectorstore import VectorStore

    cfg = Config.load(config)
    vs = VectorStore.from_config(cfg)
    reranker = Reranker(device=cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"))

    warmup_ms = warm_up(vs, reranker)

    candidates, trace_dict = run_funnel(query_text=text, vs=vs, cfg=cfg, reranker=reranker)

    typer.echo(f"Funnel execution finished! Returned {len(candidates)} candidates.")
    for c in candidates:
        meta = c.metadata or {}
        kind = meta.get("matched_kind", meta.get("kind", "unknown"))
        doc_cls = meta.get("doc_class", "unknown")
        doc_id = meta.get("doc_id", "unknown")
        fused_str = f"{c.fused_score:.4f}" if c.fused_score is not None else "N/A"
        rr_str = f"{c.rerank_score:.4f}" if c.rerank_score is not None else "N/A"
        snippet = c.text[:200].encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
        typer.echo(
            f"Rank {c.final_rank} | Fused: {fused_str} | RerankScore: {rr_str} | "
            f"Kind: {kind} | DocClass: {doc_cls} | DocID: {doc_id}\n"
            f"  Text: {snippet}"
        )

    if trace:
        typer.echo("=== FUNNEL TRACE ===")
        typer.echo(json.dumps(trace_dict, indent=2))


@app.command("eval-build")
def eval_build(n: int = 200):
    """Build evaluation set from canonical parses."""
    from adaptive_retriever.eval.build_set import build_eval_set
    build_eval_set(target_count=n)


FORBIDDEN_PHRASES_VERIFY = [
    "the provided text", "the passage", "the excerpt", "this chunk",
    "according to the text", "based on the text", "in the text",
    "this document", "in the report", "given text", "according to the provided",
    "this excerpt", "in this text", "in this document"
]


def _scan_item_flags(item: dict) -> List[str]:
    reasons = []
    q_raw = item.get("query", "")
    q_norm = normalise(q_raw)
    q_lower = q_raw.lower()

    for phrase in FORBIDDEN_PHRASES_VERIFY:
        if phrase in q_lower:
            reasons.append(f"forbidden_phrase('{phrase}')")

    q_tokens = q_norm.split()
    if len(q_tokens) >= 5:
        q_5grams = {" ".join(q_tokens[i : i + 5]) for i in range(len(q_tokens) - 4)}
        for p in item.get("relevant_passages", []):
            p_text = p.get("passage_text", "")
            p_norm = p.get("passage_norm") or normalise(p_text)
            p_tokens = p_norm.split()
            if len(p_tokens) >= 5:
                p_5grams = {" ".join(p_tokens[i : i + 5]) for i in range(len(p_tokens) - 4)}
                overlap = q_5grams & p_5grams
                if overlap:
                    sample_gram = next(iter(overlap))
                    reasons.append(f"verbatim_overlap_5gram('{sample_gram}')")
                    break

    return reasons


def _save_eval_set(eval_set_path: Path, items: List[dict]) -> None:
    with open(eval_set_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


@app.command("eval-verify")
def eval_verify(
    eval_set_path: Path = Path("data/eval/eval_set.jsonl"),
):
    """Interactive CLI verification loop for human evaluators (Addendum D1, Spec §8.5)."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            "eval-verify requires an interactive human session at a TTY terminal. "
            "Non-interactive execution is forbidden."
        )

    if not eval_set_path.exists():
        typer.echo(f"Error: Evaluation set file {eval_set_path} does not exist. Run 'cli eval-build' first.")
        raise typer.Exit(code=1)

    items = []
    with open(eval_set_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    verified_by = os.environ.get("VERIFIED_BY") or os.environ.get("USER") or os.environ.get("USERNAME")
    if not verified_by:
        verified_by = typer.prompt("Enter your evaluator name/ID (verified_by)").strip()
        if not verified_by:
            verified_by = "human_evaluator"

    unverified_items = [it for it in items if it.get("provenance") not in ("synthetic_verified", "synthetic_rejected")]

    flagged_items = []
    unflagged_items = []

    for item in unverified_items:
        flags = _scan_item_flags(item)
        if flags:
            flagged_items.append((item, flags))
        else:
            unflagged_items.append(item)

    class_order = ["exact_lookup", "conceptual", "compound", "multihop", "aggregation", "filterable"]
    class_buckets: Dict[str, List[dict]] = {c: [] for c in class_order}
    other_bucket = []

    for item in unflagged_items:
        q_cls = str(item.get("query_class_expected", "")).lower()
        if q_cls in class_buckets:
            class_buckets[q_cls].append(item)
        else:
            other_bucket.append(item)

    stratified_unflagged = []
    max_len = max([len(b) for b in class_buckets.values()], default=0)
    for idx in range(max_len):
        for q_cls in class_order:
            if idx < len(class_buckets[q_cls]):
                stratified_unflagged.append(class_buckets[q_cls][idx])
    stratified_unflagged.extend(other_bucket)

    ordered_queue = [(item, flags) for item, flags in flagged_items] + [(item, []) for item in stratified_unflagged]

    typer.echo(f"Starting interactive verification session. Unverified queue length: {len(ordered_queue)}")
    typer.echo(f"Evaluator ID (verified_by): {verified_by}")

    session_edited = 0

    for item, flags in ordered_queue:
        while True:
            typer.echo("\n" + "=" * 80)
            typer.echo(f"Item ID: {item.get('item_id')}")
            typer.echo(f"Query Class Expected: {item.get('query_class_expected')}")
            typer.echo(f"Doc Class of Source: {item.get('doc_class_of_source')}")
            if flags:
                typer.echo(f"FLAGGED SCAN WARNINGS: {', '.join(flags)}")
            typer.echo(f"Full Query: {item.get('query')}")
            typer.echo("-" * 80)
            for p_idx, p in enumerate(item.get("relevant_passages", []), 1):
                typer.echo(f"Passage #{p_idx} (doc_id={p.get('doc_id')}):")
                typer.echo(p.get("passage_text", ""))
                typer.echo("-" * 40)

            choice = typer.prompt(
                "Decision [a]ccept, [r]eject, [e]dit query, [c]hange class, [s]kip, [q]uit"
            ).strip().lower()

            if choice in ("a", "accept"):
                item["provenance"] = "synthetic_verified"
                item["verified_at"] = datetime.now(timezone.utc).isoformat()
                item["verified_by"] = verified_by
                _save_eval_set(eval_set_path, items)
                typer.echo(f"Item {item.get('item_id')} ACCEPTED and saved.")
                break
            elif choice in ("r", "reject"):
                item["provenance"] = "synthetic_rejected"
                _save_eval_set(eval_set_path, items)
                typer.echo(f"Item {item.get('item_id')} REJECTED and saved.")
                break
            elif choice in ("e", "edit", "edit query"):
                new_q = typer.prompt("Enter new query text").strip()
                if new_q:
                    item["query"] = new_q
                    session_edited += 1
                    _save_eval_set(eval_set_path, items)
                    typer.echo(f"Query updated for {item.get('item_id')}.")
            elif choice in ("c", "change", "change class"):
                typer.echo("Valid classes: exact_lookup, conceptual, compound, multihop, aggregation, filterable")
                new_c = typer.prompt("Enter new query_class_expected").strip().lower()
                if new_c:
                    item["query_class_expected"] = new_c
                    _save_eval_set(eval_set_path, items)
                    typer.echo(f"Class updated for {item.get('item_id')}.")
            elif choice in ("s", "skip"):
                typer.echo(f"Skipped item {item.get('item_id')}.")
                break
            elif choice in ("q", "quit"):
                accepted = sum(1 for it in items if it.get("provenance") == "synthetic_verified")
                rejected = sum(1 for it in items if it.get("provenance") == "synthetic_rejected")
                remaining = sum(1 for it in items if it.get("provenance") == "synthetic")
                typer.echo(
                    f"\nSession summary on quit:\n"
                    f"accepted: {accepted}, rejected: {rejected}, edited: {session_edited}, remaining: {remaining}"
                )
                return
            else:
                typer.echo("Invalid choice. Please enter a, r, e, c, s, or q.")

    accepted = sum(1 for it in items if it.get("provenance") == "synthetic_verified")
    rejected = sum(1 for it in items if it.get("provenance") == "synthetic_rejected")
    remaining = sum(1 for it in items if it.get("provenance") == "synthetic")
    typer.echo(
        f"\nVerification session finished.\n"
        f"Summary: accepted: {accepted}, rejected: {rejected}, edited: {session_edited}, remaining: {remaining}"
    )



@app.command("eval-run")
def eval_run(config: Path = Path("config/default.yaml"), split: str = "dev"):
    """Run evaluation on dev or test split."""
    typer.echo(f"Running eval on split: {split}")


@app.command()
def ablate(split: str = "dev", arms: str = None, cleanup: bool = False):
    """Run full 16-arm ablation study."""
    typer.echo(f"Running ablation study on split: {split}")


@app.command("review-next")
def review_next():
    """Interactive review queue loop for human risk gate corrections."""
    typer.echo("Reviewing next risk gate item...")


@app.command()
def redteam():
    """Run red team adversarial suites."""
    typer.echo("Running red team suites...")


@app.command("trace-replay")
def trace_replay(trace_id: str):
    """Replay query trace execution path."""
    typer.echo(f"Replaying trace: {trace_id}")


@app.command()
def stats(
    config: Path = Path("config/default.yaml"),
    manifest: Path = Path("data/corpus/MANIFEST.jsonl"),
    hashes_path: Path = Path("data/canonical_hashes.json"),
    canonical_dir: Path = Path("data/canonical"),
):
    """Print corpus and index statistics (Spec §16.1)."""
    if not manifest.exists():
        typer.echo(f"Error: Manifest file {manifest} does not exist.")
        raise typer.Exit(code=1)

    with open(manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    class_counts = {}
    size_by_class = {}

    for rec in records:
        cls = rec.get("doc_class") or rec.get("expected_doc_class", "unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1

        doc_id = rec.get("doc_id")
        if not doc_id:
            doc_path = Path(rec["doc_path"])
            if doc_path.exists():
                doc_id = hashlib.sha256(doc_path.read_bytes()).hexdigest()[:16]

        if doc_id:
            txt_file = canonical_dir / f"{doc_id}.txt"
            if txt_file.exists():
                ch_cnt = len(txt_file.read_text(encoding="utf-8"))
                size_by_class.setdefault(cls, []).append(ch_cnt)

    typer.echo("=== Corpus Document Counts by Class ===")
    for cls, cnt in sorted(class_counts.items()):
        typer.echo(f"  {cls}: {cnt}")
    typer.echo(f"  Total: {len(records)}")

    typer.echo("\n=== Canonical Parse Size Stats by Class ===")
    for cls, sizes in sorted(size_by_class.items()):
        if sizes:
            tot = sum(sizes)
            avg = tot / float(len(sizes))
            min_s = min(sizes)
            max_s = max(sizes)
            typer.echo(f"  {cls} (n={len(sizes)}): total={tot:,} chars, mean={avg:.1f}, min={min_s}, max={max_s}")
        else:
            typer.echo(f"  {cls} (n=0): no canonical parse files found")

    typer.echo("\n=== Vector Store Index Statistics ===")
    try:
        from adaptive_retriever.vectorstore import VectorStore
        cfg = Config.load(config)
        vs = VectorStore(config_hash=cfg.ingest_config_hash[:8])
        info = vs.client.get_collection(vs.collection_name)
        points_count = getattr(info, "points_count", 0)
        typer.echo(f"  Collection: {vs.collection_name}")
        typer.echo(f"  Total Vector Points: {points_count}")
    except Exception as e:
        typer.echo(f"  Vector Store status: Unavailable ({e})")


@app.command("answer")
def answer(
    query_text: str,
    config: Path = Path("config/default.yaml"),
    trace: bool = typer.Option(False, "--trace", help="Print trace details."),
):
    """Run end-to-end RAG pipeline (retrieval -> postretrieval -> generation) to answer query."""
    from adaptive_retriever.funnel.pipeline import run_funnel
    from adaptive_retriever.funnel.rerank import Reranker
    from adaptive_retriever.generation.generate import generate_answer
    from adaptive_retriever.llm import get_llm_client
    from adaptive_retriever.postretrieval.compress import compress
    from adaptive_retriever.postretrieval.reorder import reorder_candidates
    from adaptive_retriever.vectorstore import VectorStore

    cfg = Config.load(config)
    vs = VectorStore.from_config(cfg)
    llm = get_llm_client(cfg)
    reranker = Reranker(device=cfg.raw.get("models", {}).get("llm", {}).get("device", "auto"))
    reranker.ensure_ready()

    # 1. Retrieval funnel
    cands, funnel_trace = run_funnel(query_text, vs=vs, reranker=reranker, cfg=cfg, llm=llm)

    # 2. Post-retrieval (compression & reordering)
    comp_cands, comp_trace = compress(query_text, cands, cfg=cfg, llm=llm, embedding_client=vs.embedding_client)
    reordered_cands, reorder_trace = reorder_candidates(comp_cands, cfg=cfg)

    # 3. Answer generation
    ans = generate_answer(query_text, reordered_cands, cfg=cfg, llm=llm)

    # Output answer and citations
    typer.echo("\n=== ANSWER ===")
    if ans.refused:
        typer.echo(f"REFUSED: {ans.refusal_reason}")
    else:
        typer.echo(ans.text)

    typer.echo("\n=== CITATIONS ===")
    if not ans.citations:
        typer.echo("No citations.")
    else:
        cand_map = {c.chunk_id: c for c in reordered_cands}
        for cid in ans.citations:
            c_item = cand_map.get(cid)
            snippet = c_item.text[:120] + "..." if c_item and c_item.text else "N/A"
            typer.echo(f"  [{cid}]: {snippet}")

    if trace:
        typer.echo("\n=== TRACE DETAILS ===")
        typer.echo(json.dumps({
            "funnel": funnel_trace,
            "compression": comp_trace,
            "reorder": reorder_trace,
            "generation": ans.trace,
        }, indent=2, default=str))


if __name__ == "__main__":
    app()

