"""Verify the Qdrant server switch did what it was supposed to.

Two questions:
  1. Do payload indexes actually exist on the server? (the whole point)
  2. Does vectorstore mode/url leak into ingest_config_hash? (it must not)

    python check_server_mode.py
"""
import copy
import json
import sys
import urllib.request

from adaptive_retriever.config import Config

cfg = Config.load("config/default.yaml")
raw = cfg.raw
vs = raw.get("vectorstore", {})
url = vs.get("url", "http://localhost:6333")
collection = f"retriever_chunks_{cfg.ingest_config_hash[:8]}"

print("=" * 70)
print("1. CONFIG")
print("=" * 70)
print(f"  vectorstore.mode   : {vs.get('mode')}")
print(f"  vectorstore.url    : {url}")
print(f"  ingest_config_hash : {cfg.ingest_config_hash[:8]}")
print(f"  collection         : {collection}")

print()
print("=" * 70)
print("2. HASH ISOLATION - mode/url/storage_path must NOT change the hash")
print("=" * 70)
base = cfg.ingest_config_hash
failures = []
for key, alt in (("mode", "local"), ("url", "http://example:9999"), ("storage_path", "data/elsewhere")):
    mutated = copy.deepcopy(raw)
    mutated.setdefault("vectorstore", {})[key] = alt
    h = Config(mutated).ingest_config_hash
    ok = h == base
    print(f"  vectorstore.{key:<13} -> {h[:8]}   {'OK' if ok else 'LEAKED INTO HASH'}")
    if not ok:
        failures.append(key)

for key, alt in (("device", "cpu"), ("batch_size", 999), ("dtype", "float16")):
    mutated = copy.deepcopy(raw)
    mutated.setdefault("models", {}).setdefault("llm", {})[key] = alt
    h = Config(mutated).ingest_config_hash
    ok = h == base
    print(f"  models.llm.{key:<14} -> {h[:8]}   {'OK' if ok else 'LEAKED INTO HASH'}")
    if not ok:
        failures.append(key)

print()
print("=" * 70)
print("3. SERVER STATE - payload indexes and point count")
print("=" * 70)
try:
    with urllib.request.urlopen(f"{url}/collections/{collection}", timeout=10) as r:
        info = json.loads(r.read())["result"]
except Exception as e:  # noqa: BLE001
    print(f"  Could not reach {url}: {e}")
    print("  Is the container running?  docker ps")
    sys.exit(1)

print(f"  status        : {info.get('status')}")
print(f"  points_count  : {info.get('points_count')}")
print(f"  indexed_vectors: {info.get('indexed_vectors_count')}")

schema = info.get("payload_schema") or {}
print(f"\n  payload indexes: {len(schema)}")
for field, meta in sorted(schema.items()):
    dt = meta.get("data_type") if isinstance(meta, dict) else meta
    pts = meta.get("points") if isinstance(meta, dict) else "?"
    print(f"    {field:<18} {str(dt):<10} points={pts}")

expected = {"doc_class", "contains_table", "entities", "date_refs", "access_tags", "kind", "tier"}
missing = expected - set(schema)

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
ok = True
if failures:
    print(f"  FAIL  these leaked into ingest_config_hash: {failures}")
    print("        switching Qdrant mode would force a needless full re-index")
    ok = False
else:
    print("  PASS  hash correctly ignores storage location and device settings")

if missing:
    print(f"  FAIL  payload indexes missing: {sorted(missing)}")
    print("        the Phase 2 structured pre-filter would fall back to a linear scan")
    ok = False
elif not schema:
    print("  FAIL  NO payload indexes at all - still effectively local mode")
    ok = False
else:
    print(f"  PASS  all {len(expected)} payload indexes present on the server")

if info.get("points_count") != 22422:
    print(f"  WARN  points_count is {info.get('points_count')}, expected 22422")
    print("        the re-index may still be running, or did not complete")
    ok = False
else:
    print("  PASS  22,422 points")

sys.exit(0 if ok else 1)
