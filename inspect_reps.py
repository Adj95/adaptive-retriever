from qdrant_client import QdrantClient

client = QdrantClient(path="data/qdrant_storage")
res, _ = client.scroll(collection_name="retriever_chunks_aa06ab0c", limit=500)

summaries = [p.payload for p in res if p.payload.get("kind") == "summary"]
questions = [p.payload for p in res if p.payload.get("kind") == "question"]

out_lines = []
out_lines.append("=== SUMMARIES (sample 3) ===")
for s in summaries[:3]:
    out_lines.append(f"ID: {s.get('rep_id')}")
    out_lines.append(f"TEXT: {s.get('text')}")
    out_lines.append("-" * 60)

out_lines.append("\n=== QUESTIONS (sample 3) ===")
for q in questions[:3]:
    out_lines.append(f"ID: {q.get('rep_id')}")
    out_lines.append(f"TEXT: {q.get('text')}")
    out_lines.append("-" * 60)

with open("sample_representations.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Successfully written sample representations to sample_representations.txt")
