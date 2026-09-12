"""Embedded Qdrant VectorStore Handler (A5, spec §9.5 & Addendum v1.2 §D).

Uses qdrant-client in local embedded persistent mode at `data/qdrant_storage/`.
Collection naming: {collection}_{ingest_config_hash[:8]}.
Named vectors: `dense` (384-dim, Cosine) and `sparse` (BM25).
Payload indexing: `doc_class`, `contains_table`, `entities`, `date_refs`, `access_tags`.
Hybrid search: Prefetch + RRF Fusion (rest.FusionQuery(fusion=rest.Fusion.RRF)).

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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from adaptive_retriever.models import Chunk, Representation, RetrievalCandidate

logger = logging.getLogger(__name__)

STORAGE_PATH = Path("data/qdrant_storage")
DENSE_DIM = 384


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


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
    ):
        if config_hash is None:
            raise ValueError(
                "VectorStore requires an explicit config_hash. The old "
                "compute_ingest_config_hash() default hashed a hardcoded "
                "'all-MiniLM-L6-v2' and shadowed Config.ingest_config_hash."
            )
        storage_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(storage_path))
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
            **kwargs,
        )

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
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
        except Exception as e:
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

    # ------------------------------------------------------------------- query

    def query_hybrid(
        self,
        query_text: str,
        limit: int = 10,
        doc_class_filter: Optional[str] = None,
    ) -> List[RetrievalCandidate]:
        dense_vec = self.embedding_client.encode(query_text)
        sparse_vec = self.sparse_client.encode(query_text)

        query_filter = None
        if doc_class_filter:
            query_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_class", match=rest.MatchValue(value=doc_class_filter)
                    )
                ]
            )

        prefetches = [
            rest.Prefetch(query=dense_vec, using="dense", limit=limit * 2, filter=query_filter),
            rest.Prefetch(query=sparse_vec, using="sparse", limit=limit * 2, filter=query_filter),
        ]

        # No silent fallback to dense-only search: a hybrid query that quietly
        # became a dense query would make every fusion ablation meaningless.
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetches,
            query=rest.FusionQuery(fusion=rest.Fusion.RRF),
            limit=limit,
        )

        candidates: List[RetrievalCandidate] = []
        points = getattr(results, "points", results)
        for rank, res in enumerate(points, 1):
            p = res.payload or {}
            candidates.append(
                RetrievalCandidate(
                    chunk_id=p.get("chunk_id", ""),
                    rep_id=p.get("rep_id", ""),
                    fused_score=float(getattr(res, "score", 0.0)),
                    final_rank=rank,
                    text=p.get("text", ""),
                    metadata=p,
                )
            )
        return candidates

    def close(self) -> None:
        try:
            self.client.close()
        except Exception as e:  # noqa: BLE001 - close during shutdown is best effort
            logger.debug("VectorStore close failed: %s", e)
