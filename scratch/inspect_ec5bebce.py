from qdrant_client import QdrantClient

client = QdrantClient(path="data/qdrant_storage")
res, _ = client.scroll(collection_name="retriever_chunks_ec5bebce", limit=200)

summaries = [p.payload for p in res if p.payload.get("kind") == "summary"]
questions = [p.payload for p in res if p.payload.get("kind") == "question"]
keywords = [p.payload for p in res if p.payload.get("kind") == "keywords"]

out_lines = []
out_lines.append("=== QWEN2.5-0.5B GENERATED SUMMARIES (sample 3) ===")
for s in summaries[:3]:
    out_lines.append(f"ID: {s.get('rep_id')}")
    out_lines.append(f"TEXT: {s.get('text')}")
    out_lines.append("-" * 60)

out_lines.append("\n=== QWEN2.5-0.5B GENERATED QUESTIONS (sample 3) ===")
for q in questions[:3]:
    out_lines.append(f"ID: {q.get('rep_id')}")
    out_lines.append(f"TEXT: {q.get('text')}")
    out_lines.append("-" * 60)

out_lines.append("\n=== QWEN2.5-0.5B GENERATED KEYWORDS (sample 3) ===")
for k in keywords[:3]:
    out_lines.append(f"ID: {k.get('rep_id')}")
    out_lines.append(f"TEXT: {k.get('text')}")
    out_lines.append("-" * 60)

with open("scratch/sample_ec5bebce.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Successfully written sample representations to scratch/sample_ec5bebce.txt")
