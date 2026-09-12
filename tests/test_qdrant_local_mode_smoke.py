"""Smoke test for Qdrant client in local embedded persistent mode (Addendum B1).
Verifies dense+sparse named vectors, payload filtering, RRF fusion, and persistence across process restarts.
"""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def test_qdrant_local_mode_dense_sparse_rrf_persistence(tmp_path):
    storage_dir = tmp_path / "qdrant_storage"
    collection_name = "test_collection"

    # Step 1: Initial client connection
    client = QdrantClient(path=str(storage_dir))

    # Step 2: Create collection with named vectors (dense + sparse)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": rest.VectorParams(size=384, distance=rest.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": rest.SparseVectorParams(index=rest.SparseIndexParams(on_disk=False))
        },
    )

    # Step 3: Create payload indexes
    client.create_payload_index(collection_name, "doc_class", rest.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, "contains_table", rest.PayloadSchemaType.BOOL)
    client.create_payload_index(collection_name, "entities", rest.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, "date_refs", rest.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, "access_tags", rest.PayloadSchemaType.KEYWORD)

    # Step 4: Upsert points with full §6.3 payload schema
    payload_1 = {
        "rep_id": "doc1:00000:raw:0",
        "chunk_id": "doc1:00000",
        "doc_id": "doc1",
        "kind": "raw",
        "tier": "high",
        "doc_class": "digital_prose",
        "section_path": ["1. Introduction"],
        "contains_table": False,
        "page_numbers": [1],
        "ingested_at": "2026-09-08T00:00:00Z",
        "doc_version": 1,
        "entities": ["Alpha", "Beta"],
        "date_refs": ["2026-01-01"],
        "access_tags": ["public"],
    }

    payload_2 = {
        "rep_id": "doc2:00000:raw:0",
        "chunk_id": "doc2:00000",
        "doc_id": "doc2",
        "kind": "raw",
        "tier": "high",
        "doc_class": "table_heavy",
        "section_path": ["Financial Results"],
        "contains_table": True,
        "page_numbers": [3],
        "ingested_at": "2026-09-08T00:00:00Z",
        "doc_version": 1,
        "entities": ["Gamma"],
        "date_refs": ["2025-12-31"],
        "access_tags": ["public"],
    }

    dummy_dense_1 = [0.1] * 384
    dummy_dense_2 = [0.2] * 384

    points = [
        rest.PointStruct(
            id=1,
            vector={
                "dense": dummy_dense_1,
                "sparse": rest.SparseVector(indices=[0, 5, 10], values=[1.0, 0.8, 0.5]),
            },
            payload=payload_1,
        ),
        rest.PointStruct(
            id=2,
            vector={
                "dense": dummy_dense_2,
                "sparse": rest.SparseVector(indices=[2, 5, 12], values=[0.9, 0.7, 0.4]),
            },
            payload=payload_2,
        ),
    ]

    client.upsert(collection_name=collection_name, points=points)

    # Step 5: Test payload filtering
    filter_doc_class = rest.Filter(
        must=[
            rest.FieldCondition(
                key="doc_class", match=rest.MatchValue(value="digital_prose")
            )
        ]
    )
    search_res = client.query_points(
        collection_name=collection_name,
        query=[0.1] * 384,
        using="dense",
        query_filter=filter_doc_class,
        limit=10,
    )
    assert len(search_res.points) == 1
    assert search_res.points[0].id == 1

    # Step 6: Test Prefetch + RRF Fusion
    prefetch_dense = rest.Prefetch(
        query=[0.1] * 384, using="dense", limit=10
    )
    prefetch_sparse = rest.Prefetch(
        query=rest.SparseVector(indices=[5], values=[1.0]), using="sparse", limit=10
    )

    fusion_res = client.query_points(
        collection_name=collection_name,
        prefetch=[prefetch_dense, prefetch_sparse],
        query=rest.FusionQuery(fusion=rest.Fusion.RRF),
        limit=10,
    )
    assert len(fusion_res.points) == 2

    client.close()

    # Step 7: Test persistence across process restart
    client_reopen = QdrantClient(path=str(storage_dir))
    count_res = client_reopen.count(collection_name=collection_name)
    assert count_res.count == 2

    retrieved = client_reopen.retrieve(collection_name=collection_name, ids=[2])
    assert len(retrieved) == 1
    assert retrieved[0].payload["doc_class"] == "table_heavy"

    client_reopen.close()
