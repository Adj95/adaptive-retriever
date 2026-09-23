"""Embedded Qdrant VectorStore Handler (A5, spec §9.5 & Addendum v1.2 §D).

Runs against a Qdrant server (`vectorstore.mode: server`) or a local embedded
store. Payload indexes are a no-op in local mode, so server mode is the
supported configuration for the funnel's structured pre-filter.
Collection naming: {collection}_{ingest_config_hash[:8]}.
Named vectors: `dense` (384-dim, Cosine) and `sparse` (BM25).
Payload indexing: `doc_class`, `contains_table`, `entities`, `date_refs`,
`access_tags`, `kind`, `tier`.
Hybrid search: two independent searches fused by our own weighted RRF in
`funnel.fusion` - NOT Qdrant's server-side rest.Fusion.RRF, which cannot apply
per-query-class source weights.

CRITICAL CORRECTION (2026-09-11)
-------------------------------
The previous version wrote SHA-256 hash bytes as "dense vectors" and
process-randomised `hash(word) % 10000` buckets as "sparse vectors". Measured
on that scheme, a paraphrase pair scored cosine 0.0044 while an unrelated pair
scored 0.029 - the vectors carried no semantic signal whatsoever, and the
sparse indices were not even stable between processes.

Rules now enforced here:
  * Dense vectors come from the configured sentence-transformers model. If it
    cannot be loaded, this module RAISES. There is no hash fallback.
  * Sparse vectors come from a real BM25 model. If it cannot be loaded, this
    module RAISES.
  * The reranker RAISES on load failure instead of returning constant scores.
  * Point IDs use 64 bits, not 32. At 22k points a 32-bit ID space carries a
    ~5.7% chance of at least one silent collision.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from adaptive_retriever.funnel.fusion import collapse_to_best, fuse_rrf
from adaptive_retriever.models import Chunk, Representation, RetrievalCandidate

logger = logging.getLogger(__name__)

STORAGE_PATH = Path("data/qdrant_storage")
DENSE_DIM = 384


def _resolve_device(device: str) -> str:
    target = device if device != "auto" else "cuda"
    if target == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                _ = torch.zeros(1).cuda()
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return target


class DenseEmbeddingModel:
    """Dense embedding model (BAAI/bge-small-en-v1.5) with CUDA when available.

    Loads eagerly on first use and RAISES on failure. A retrieval system whose
    embedder silently degrades produces numbers that look like results and are
    not.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise RuntimeError(
                "sentence-transformers is required for dense embeddings but is not "
                "importable. Install it: pip install sentence-transformers. "
                "No hash-vector fallback exists - see module docstring."
            ) from err

        logger.info(
            "Loading dense embedding model '%s' on %s", self.model_name, self.device
        )
        self._model = SentenceTransformer(self.model_name, device=self.device)

        dim = (
            self._model.get_embedding_dimension()
            if hasattr(self._model, "get_embedding_dimension")
            else self._model.get_sentence_embedding_dimension()
        )
        if dim != DENSE_DIM:
            raise ValueError(
                f"Embedding model '{self.model_name}' produces {dim}-dim vectors but the "
                f"Qdrant collection is configured for {DENSE_DIM}. Change "
                f"models.dense_embedding or DENSE_DIM - do not silently truncate."
            )

    def encode(self, text: str) -> List[float]:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


class SparseEmbeddingModel:
    """BM25 sparse vectors via fastembed.

    The previous implementation used `abs(hash(word)) % 10000`. Python
    randomises str hashing per process (PYTHONHASHSEED), so index-time and
    query-time vectors were computed in different coordinate systems and could
    never match. RAISES on load failure.
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import SparseTextEmbedding
        except ImportError as err:
            raise RuntimeError(
                "fastembed is required for BM25 sparse vectors but is not importable. "
                "Install it: pip install fastembed. The previous hash-bucket scheme "
                "was not stable across processes and has been removed."
            ) from err

        logger.info("Loading sparse (BM25) model '%s'", self.model_name)
        self._model = SparseTextEmbedding(model_name=self.model_name)

    def encode(self, text: str) -> rest.SparseVector:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> List[rest.SparseVector]:
        if not texts:
            return []
        self._ensure_loaded()
        out: List[rest.SparseVector] = []
        for emb in self._model.embed(list(texts)):
            out.append(
                rest.SparseVector(
                    indices=[int(i) for i in emb.indices],
                    values=[float(v) for v in emb.values],
                )
            )
        if len(out) != len(texts):
            raise RuntimeError(
                f"Sparse model returned {len(out)} vectors for {len(texts)} texts. "
                "Refusing to continue with a misaligned batch."
            )
        return out


class RerankerModel:
    """Cross-encoder reranker. RAISES on load failure.

    The previous version returned [1.0] * len(pairs) when the model could not
    load - a reranker that silently does nothing while reporting success.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = _resolve_device(device)
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as err:
            raise RuntimeError(
                "sentence-transformers is required for the cross-encoder reranker "
                "but is not importable. Install it: pip install sentence-transformers."
            ) from err
        logger.info("Loading reranker '%s' on %s", self.model_name, self.device)
        self._model = CrossEncoder(self.model_name, device=self.device)

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        self._ensure_loaded()
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


def stable_point_id(rep_id: str) -> int:
    """64-bit deterministic point ID.

    Was 32-bit (md5[:8]). With ~22,000 points that carries roughly a 5.7%
    chance of at least one collision, and a collision silently overwrites a
    point.
    """
    return int(hashlib.sha256(rep_id.encode("utf-8")).hexdigest()[:16], 16)


class VectorStore:
    def __init__(
        self,
        collection_base_name: str = "retriever_chunks",
        config_hash: Optional[str] = None,
        storage_path: Path = STORAGE_PATH,
        embedding_client: Optional[Any] = None,
        sparse_client: Optional[Any] = None,
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        sparse_model_name: str = "Qdrant/bm25",
        device: str = "auto",
        mode: str = "server",
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        cfg: Optional[Any] = None,
        timeout: float = 60.0,
    ):
        if config_hash is None:
            raise ValueError(
                "VectorStore requires an explicit config_hash. The old "
                "compute_ingest_config_hash() default hashed a hardcoded "
                "'all-MiniLM-L6-v2' and shadowed Config.ingest_config_hash."
            )
        self.mode = mode
        self.url = url
        self.api_key = api_key if api_key is not None else os.environ.get("QDRANT_API_KEY")
        self.cfg = cfg
        self.timeout = timeout

        if self.mode == "server":
            try:
                self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)
            except (NameError, ImportError, AttributeError):
                raise
            except Exception as err:
                raise RuntimeError(
                    f"Failed to connect to Qdrant server at '{self.url}': {err}"
                ) from err
        elif self.mode == "local":
            storage_path = Path(storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_path))
        else:
            raise ValueError(
                f"Invalid vectorstore mode '{self.mode}'. Expected 'local' or 'server'."
            )

        self.config_hash = config_hash
        self.collection_name = f"{collection_base_name}_{self.config_hash[:8]}"
        self.embedding_client = embedding_client or DenseEmbeddingModel(
            model_name=dense_model_name, device=device
        )
        self.sparse_client = sparse_client or SparseEmbeddingModel(
            model_name=sparse_model_name
        )
        self._ensure_collection()

    @classmethod
    def from_config(cls, cfg: Any, **kwargs) -> "VectorStore":
        raw = cfg.raw if hasattr(cfg, "raw") else cfg
        models = raw.get("models", {})
        vs = raw.get("vectorstore", {})
        return cls(
            collection_base_name=kwargs.pop("collection_base_name", "retriever_chunks"),
            config_hash=cfg.ingest_config_hash[:8],
            storage_path=Path(vs.get("storage_path", STORAGE_PATH)),
            dense_model_name=models.get("dense_embedding", "BAAI/bge-small-en-v1.5"),
            sparse_model_name=models.get("sparse_model", "Qdrant/bm25"),
            device=models.get("llm", {}).get("device", "auto"),
            mode=kwargs.pop("mode", vs.get("mode", "server")),
            url=kwargs.pop("url", vs.get("url", "http://localhost:6333")),
            api_key=kwargs.pop("api_key", None),
            cfg=cfg,
            **kwargs,
        )

    def _ensure_collection(self) -> None:
        try:
            exists = self.client.collection_exists(self.collection_name)
        except (NameError, ImportError, AttributeError):
            raise
        except Exception as err:
            if self.mode == "server":
                raise RuntimeError(
                    f"Failed to connect to Qdrant server at '{self.url}': {err}"
                ) from err
            raise

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": rest.VectorParams(size=DENSE_DIM, distance=rest.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": rest.SparseVectorParams(index=rest.SparseIndexParams())
                },
            )
            for field, schema in (
                ("doc_class", rest.PayloadSchemaType.KEYWORD),
                ("contains_table", rest.PayloadSchemaType.BOOL),
                ("entities", rest.PayloadSchemaType.KEYWORD),
                ("date_refs", rest.PayloadSchemaType.KEYWORD),
                ("access_tags", rest.PayloadSchemaType.KEYWORD),
                ("kind", rest.PayloadSchemaType.KEYWORD),
                ("tier", rest.PayloadSchemaType.KEYWORD),
            ):
                self.create_payload_index(field, schema)

    def create_payload_index(self, field_name: str, field_schema: Any) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        except (NameError, ImportError, AttributeError):
            raise
        except Exception as e:
            if self.mode == "local":
                # Qdrant local mode does not support payload indexes. That is a
                # documented no-op, not a failure - but any OTHER error must surface.
                msg = str(e).lower()
                if "local" in msg or "not supported" in msg or "no effect" in msg:
                    logger.debug("Payload index '%s' skipped in local mode", field_name)
                    return
            logger.error(
                "Failed to create payload index for '%s' in '%s': %s",
                field_name,
                self.collection_name,
                e,
            )
            raise

    def scroll(
        self,
        scroll_filter: Optional[Any] = None,
        limit: int = 100,
        offset: Optional[Any] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Tuple[List[Any], Optional[Any]]:
        try:
            return self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
        except Exception as e:
            logger.error("Failed to scroll '%s': %s", self.collection_name, e)
            raise

    def count(self, count_filter: Optional[Any] = None, exact: bool = True) -> int:
        try:
            res = self.client.count(
                collection_name=self.collection_name,
                count_filter=count_filter,
                exact=exact,
            )
            return res.count
        except Exception as e:
            logger.error("Failed to count '%s': %s", self.collection_name, e)
            raise

    # ------------------------------------------------------------------ upsert

    def _build_points(
        self,
        items: Sequence[Tuple[Chunk, Representation]],
        metadata: Dict[str, Any],
    ) -> List[rest.PointStruct]:
        if not items:
            return []

        texts = [rep.text for _, rep in items]
        dense_vecs = self.embedding_client.encode_batch(texts)
        sparse_vecs = self.sparse_client.encode_batch(texts)

        if not (len(dense_vecs) == len(sparse_vecs) == len(items)):
            raise RuntimeError(
                f"Vector/item count mismatch: {len(items)} items, "
                f"{len(dense_vecs)} dense, {len(sparse_vecs)} sparse."
            )

        points: List[rest.PointStruct] = []
        for (chunk, rep), dv, sv in zip(items, dense_vecs, sparse_vecs):
            points.append(
                rest.PointStruct(
                    id=stable_point_id(rep.rep_id),
                    vector={"dense": dv, "sparse": sv},
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "rep_id": rep.rep_id,
                        "kind": rep.kind.value,
                        "tier": rep.tier,
                        "text": rep.text,
                        "doc_class": metadata.get("doc_class", "unknown"),
                        "contains_table": chunk.contains_table,
                        "entities": metadata.get("entities", []),
                        "date_refs": metadata.get("date_refs", []),
                        "access_tags": metadata.get("access_tags", []),
                    },
                )
            )
        return points

    def upsert_representations(
        self,
        chunk: Chunk,
        representations: List[Representation],
        metadata: Dict[str, Any],
    ) -> None:
        points = self._build_points([(chunk, r) for r in representations], metadata)
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def upsert_document(
        self,
        chunks: Sequence[Chunk],
        representations: Sequence[Representation],
        metadata: Dict[str, Any],
        upsert_batch: int = 256,
    ) -> int:
        """Encode and upsert a whole document in one pass.

        Far faster than per-chunk upserts on GPU: one encoder call per document
        instead of one per chunk.
        """
        by_id = {c.chunk_id: c for c in chunks}
        items: List[Tuple[Chunk, Representation]] = []
        for rep in representations:
            chunk = by_id.get(rep.chunk_id)
            if chunk is None:
                raise RuntimeError(
                    f"Representation '{rep.rep_id}' references unknown chunk "
                    f"'{rep.chunk_id}'. Refusing to index an orphaned vector."
                )
            items.append((chunk, rep))

        written = 0
        for start in range(0, len(items), upsert_batch):
            batch = items[start : start + upsert_batch]
            points = self._build_points(batch, metadata)
            if points:
                self.client.upsert(
                    collection_name=self.collection_name, points=points
                )
                written += len(points)
        return written

    def search_dense(
        self,
        query_vector: List[float],
        limit: int,
        query_filter: Optional[Any] = None,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        points = getattr(res, "points", res)
        out: List[Tuple[str, Dict[str, Any], float]] = []
        for p in points:
            payload = getattr(p, "payload", {}) or {}
            rep_id = payload.get("rep_id", "")
            score = float(getattr(p, "score", 0.0))
            out.append((rep_id, payload, score))
        return out

    def search_sparse(
        self,
        query_vector: Any,
        limit: int,
        query_filter: Optional[Any] = None,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="sparse",
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        points = getattr(res, "points", res)
        out: List[Tuple[str, Dict[str, Any], float]] = []
        for p in points:
            payload = getattr(p, "payload", {}) or {}
            rep_id = payload.get("rep_id", "")
            score = float(getattr(p, "score", 0.0))
            out.append((rep_id, payload, score))
        return out

    # ------------------------------------------------------------------- query

    def _fetch_raw_payloads(self, chunk_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Return each chunk's RAW representation payload, keyed by chunk_id.

        Every point in the index stores the text of ONE representation. A chunk
        retrieved because its summary or a generated question matched carries
        the summary/question text in its payload, not the chunk's own text. If
        that text were handed downstream, the generator would quote and cite a
        model-written summary as if it were source evidence.

        The raw representation's id is `f"{chunk_id}_raw"` (see
        ingest/representations.py, which enforces RAW.text == chunk.text), and
        point ids are a deterministic hash of the rep id. So the real text is a
        direct id lookup - no filter, no scan, and no payload index needed.
        """
        if not chunk_ids:
            return {}
        wanted = list(dict.fromkeys(chunk_ids))
        ids = [stable_point_id(f"{cid}_raw") for cid in wanted]
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        by_chunk: Dict[str, Dict[str, Any]] = {}
        for p in points:
            payload = getattr(p, "payload", {}) or {}
            cid = payload.get("chunk_id")
            if cid:
                by_chunk[cid] = payload

        missing = [c for c in wanted if c not in by_chunk]
        if missing:
            raise RuntimeError(
                f"No RAW representation found for chunk(s) {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}. Every chunk must have one "
                f"(ingest/representations.py builds it for both tiers), so the "
                f"index is incomplete or was written by an older ingest. "
                f"Re-run `cli ingest`."
            )
        return by_chunk

    def query_hybrid(
        self,
        query_text: str,
        limit: int = 10,
        doc_class_filter: Optional[str] = None,
        query_class: Optional[str] = None,
        cfg: Optional[Any] = None,
        query_filter: Optional[Any] = None,
        dense_query_text: Optional[str] = None,
        sparse_query_text: Optional[str] = None,
    ) -> List[RetrievalCandidate]:
        """Dense + sparse search fused with weighted RRF, returned as chunks.

        Two things here are easy to get wrong and are handled explicitly:

        1. The index holds one point per REPRESENTATION, so one chunk can take
           several positions in a single source's ranking. Representations are
           collapsed to their chunk BEFORE fusion, so each chunk appears at most
           once and a chunk is not rewarded for having more representations.
        2. The text returned is always the chunk's own RAW text, never the
           summary or question that happened to match.
        """
        active_cfg = cfg if cfg is not None else self.cfg
        if active_cfg is None:
            raise ValueError(
                "query_hybrid needs a config. Pass cfg=..., or build the store "
                "with VectorStore.from_config so self.cfg is set. Loading a "
                "default file here would use settings the caller never chose."
            )
        raw_cfg = active_cfg.raw if hasattr(active_cfg, "raw") else active_cfg

        hybrid_cfg = raw_cfg.get("funnel", {}).get("hybrid", {})
        fusion = hybrid_cfg.get("fusion", "rrf")
        if fusion != "rrf":
            raise ValueError(f"Unsupported fusion mode '{fusion}'. Expected 'rrf'.")

        weights = hybrid_cfg.get("weights", {"dense": 0.5, "sparse": 0.5})
        total = sum(float(v) for v in weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"funnel.hybrid.weights sums to {total:.4f}, not 1.0."
            )

        dense_limit = int(hybrid_cfg.get("dense_limit", 100))
        sparse_limit = int(hybrid_cfg.get("sparse_limit", 100))
        rrf_k = int(hybrid_cfg.get("rrf_k", 60))

        dense_q = dense_query_text if dense_query_text is not None else query_text
        sparse_q = sparse_query_text if sparse_query_text is not None else query_text

        import time
        t_de = time.perf_counter()
        dense_vec = self.embedding_client.encode(dense_q)
        dense_encode_ms = round((time.perf_counter() - t_de) * 1000, 2)

        t_se = time.perf_counter()
        sparse_vec = self.sparse_client.encode(sparse_q)
        sparse_encode_ms = round((time.perf_counter() - t_se) * 1000, 2)

        effective_filter = query_filter
        if effective_filter is None and doc_class_filter:
            effective_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_class", match=rest.MatchValue(value=doc_class_filter)
                    )
                ]
            )

        t_ds = time.perf_counter()
        dense_hits = self.search_dense(dense_vec, limit=dense_limit, query_filter=effective_filter)
        dense_search_ms = round((time.perf_counter() - t_ds) * 1000, 2)

        t_ss = time.perf_counter()
        sparse_hits = self.search_sparse(sparse_vec, limit=sparse_limit, query_filter=effective_filter)
        sparse_search_ms = round((time.perf_counter() - t_ss) * 1000, 2)

        t_fu = time.perf_counter()
        rep_payload: Dict[str, Dict[str, Any]] = {}
        chunk_of_rep: Dict[str, str] = {}
        for rep_id, payload, _score in list(dense_hits) + list(sparse_hits):
            if not rep_id:
                continue
            payload = payload or {}
            rep_payload.setdefault(rep_id, payload)
            chunk_id = payload.get("chunk_id")
            if not chunk_id:
                raise RuntimeError(
                    f"Point '{rep_id}' has no chunk_id in its payload. Its chunk "
                    f"cannot be identified, so the index is malformed. "
                    f"Re-run `cli ingest`."
                )
            chunk_of_rep[rep_id] = chunk_id

        ranked_lists: Dict[str, List[str]] = {}
        rep_winner: Dict[str, Dict[str, str]] = {}
        for source, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
            chunk_order, winner = collapse_to_best(
                [h[0] for h in hits if h[0]], chunk_of_rep
            )
            ranked_lists[source] = chunk_order
            rep_winner[source] = winner

        fused = fuse_rrf(ranked_lists, rrf_k=rrf_k, weights=weights)[:limit]
        fusion_ms = round((time.perf_counter() - t_fu) * 1000, 2)

        t_hy = time.perf_counter()
        raw_payloads = self._fetch_raw_payloads([row[0] for row in fused])
        raw_hydration_ms = round((time.perf_counter() - t_hy) * 1000, 2)

        self.last_query_breakdown = {
            "dense_encode_ms": dense_encode_ms,
            "sparse_encode_ms": sparse_encode_ms,
            "dense_search_ms": dense_search_ms,
            "sparse_search_ms": sparse_search_ms,
            "fusion_ms": fusion_ms,
            "raw_hydration_ms": raw_hydration_ms,
        }

        candidates: List[RetrievalCandidate] = []
        for rank, (chunk_id, fused_score, source_ranks) in enumerate(fused, 1):
            best_rep, best_rank = "", None
            for source, winner in rep_winner.items():
                r = source_ranks.get(source)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank, best_rep = r, winner.get(chunk_id, "")

            raw = raw_payloads.get(chunk_id, {})
            meta = dict(raw)
            meta["source_ranks"] = source_ranks
            meta["matched_kind"] = (rep_payload.get(best_rep, {}) or {}).get("kind", "raw")
            meta["matched_rep_id"] = best_rep
            meta["winning_rep_by_source"] = {
                s: w[chunk_id] for s, w in rep_winner.items() if chunk_id in w
            }
            candidates.append(
                RetrievalCandidate(
                    chunk_id=chunk_id,
                    rep_id=best_rep,
                    fused_score=fused_score,
                    final_rank=rank,
                    text=raw.get("text", ""),
                    metadata=meta,
                )
            )
        return candidates

    def close(self) -> None:
        try:
            self.client.close()
        except Exception as e:  # noqa: BLE001 - close during shutdown is best effort
            logger.debug("VectorStore close failed: %s", e)
