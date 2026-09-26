"""Run this BEFORE any experiment, and paste the output.

Every wasted hour this week traced to something this script checks in 3 seconds:
the wrong Python interpreter (CPU-only torch, 7x slowdown reported as a hardware
finding), a Qdrant server that was not running (30 minutes against a local
store), a config that had drifted, a hash that had moved.

Reads only. Writes nothing. Exits non-zero on any FAIL.

    python check_preflight.py
"""
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import subprocess
import sys
import urllib.request
from pathlib import Path

EXPECTED_HASH = "2e6f1c8d"
EXPECTED_POINTS = 47034

fails, warns = [], []
SEP = "=" * 72


def line(label, value, ok=None):
    mark = "" if ok is None else ("  OK" if ok else "  FAIL")
    print(f"  {label:<26}: {value}{mark}")


print(SEP)
print("PREFLIGHT")
print(SEP)

# ------------------------------------------------------- 1. interpreter ----
exe = Path(sys.executable).resolve()
venv = (Path.cwd() / ".venv" / "Scripts" / "python.exe").resolve()
venv_ok = exe == venv or ".venv" in str(exe)
line("python", str(exe), venv_ok)
if not venv_ok:
    fails.append("wrong interpreter - not the project .venv")
    print("       -> experiments have silently run under Anaconda's python before,")
    print("          with a CPU-only torch, and the 7x slowdown was reported as a")
    print("          hardware finding. Activate .venv first.")

# --------------------------------------------------------------- 2. torch --
try:
    import torch

    cuda_ok = torch.cuda.is_available()
    line("torch", torch.__version__)
    line("torch.version.cuda", str(torch.version.cuda), torch.version.cuda is not None)
    line("cuda available", str(cuda_ok), cuda_ok)
    if cuda_ok:
        line("device", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info()
        line("free VRAM", f"{free / 1e6:,.0f} MB of {total / 1e6:,.0f} MB")
        if free < 1.5e9:
            warns.append(f"only {free / 1e6:,.0f} MB VRAM free - close other GPU users")
    else:
        fails.append("CUDA not available in this interpreter")
    if torch.version.cuda is None:
        fails.append("CPU-only torch wheel installed")
except Exception as exc:  # noqa: BLE001
    line("torch", f"IMPORT FAILED: {exc}", False)
    fails.append("torch not importable")

# ------------------------------------------------------------- 3. config ---
print()
print(SEP)
print("CONFIG")
print(SEP)
try:
    from adaptive_retriever.config import Config

    cfg = Config.load("config/default.yaml")
    h = cfg.ingest_config_hash[:8]
    line("ingest_config_hash", h, h == EXPECTED_HASH)
    if h != EXPECTED_HASH:
        fails.append(f"ingest_config_hash moved to {h} (expected {EXPECTED_HASH})")
        print(f"       -> the collection name has changed; a re-index would be forced.")

    vs = cfg.raw.get("vectorstore", {})
    hy = cfg.raw.get("funnel", {}).get("hybrid", {})
    rr = cfg.raw.get("funnel", {}).get("rerank", {})
    tf = cfg.raw.get("query", {}).get("transformation", {})
    pr_comp = cfg.raw.get("postretrieval", {}).get("compression", {})
    pr_reorder = cfg.raw.get("postretrieval", {}).get("reorder", {})
    max_tokens = cfg.raw.get("models", {}).get("llm", {}).get("max_new_tokens", 128)

    line("vectorstore.mode", vs.get("mode"), vs.get("mode") == "server")
    if vs.get("mode") != "server":
        fails.append("vectorstore.mode is not 'server' - payload indexes are a no-op locally")
    line("hybrid.weights", json.dumps(hy.get("weights")))
    line("hybrid.rrf_k", hy.get("rrf_k"))
    line("rerank.mode / margin", f"{rr.get('mode')} / {rr.get('gate_score_margin')}")
    line("transformation.strategy", tf.get("strategy"))

    emb_thresh = float(pr_comp.get("embeddings_threshold", 0.40))
    line("postretrieval.compression.mode", pr_comp.get("mode"))
    line("postretrieval.compression.embeddings_threshold", emb_thresh, abs(emb_thresh - 0.40) < 1e-6)
    if abs(emb_thresh - 0.40) >= 1e-6:
        fails.append(f"embeddings_threshold is {emb_thresh} (expected 0.40 - the shipped results value)")
    line("postretrieval.reorder.enabled", pr_reorder.get("enabled"))
    line("generation.max_new_tokens", max_tokens)

    try:
        from adaptive_retriever.llm import get_llm_client
        client = get_llm_client(cfg)
        client._ensure_loaded()
        resolved_llm_dev = str(client.device)
        line("llm.resolved_device", resolved_llm_dev, resolved_llm_dev == "cuda")
        if resolved_llm_dev != "cuda":
            fails.append(f"LLM client resolved device is '{resolved_llm_dev}' (expected 'cuda')")
    except Exception as exc:
        line("llm.resolved_device", f"FAILED: {exc}", False)
        fails.append(f"LLM client resolution failed: {exc}")

    w = hy.get("weights") or {}
    if w and abs(sum(float(v) for v in w.values()) - 1.0) > 1e-6:
        fails.append("funnel.hybrid.weights do not sum to 1.0")
except Exception as exc:  # noqa: BLE001
    line("config", f"FAILED: {type(exc).__name__}: {exc}", False)
    fails.append("config would not load")

# ------------------------------------------------------------- 4. qdrant ---
print()
print(SEP)
print("QDRANT")
print(SEP)
url = (cfg.raw.get("vectorstore", {}).get("url") if "cfg" in dir() else None) or "http://127.0.0.1:6333"
url_req = url.replace("localhost", "127.0.0.1")
collection = f"retriever_chunks_{EXPECTED_HASH}"
try:
    with urllib.request.urlopen(f"{url_req}/collections/{collection}", timeout=8) as r:
        info = json.loads(r.read())["result"]
    pts = info.get("points_count")
    line("url", url, True)
    line("collection", collection, True)
    line("points", f"{pts:,}", pts == EXPECTED_POINTS)
    if pts != EXPECTED_POINTS:
        fails.append(f"points_count is {pts}, expected {EXPECTED_POINTS}")
    schema = info.get("payload_schema") or {}
    need = {"doc_class", "contains_table", "entities", "date_refs", "access_tags", "kind", "tier"}
    missing = need - set(schema)
    line("payload indexes", f"{len(schema)} present", not missing)
    if missing:
        fails.append(f"payload indexes missing: {sorted(missing)}")

    # Check for stale collections not matching current ingest_config_hash and empty collections
    with urllib.request.urlopen(f"{url_req}/collections", timeout=8) as r:
        all_cols = json.loads(r.read())["result"]["collections"]
    stale_cols = [c["name"] for c in all_cols if c["name"] != collection and not c["name"].startswith("beir_")]
    empty_cols = []
    for c in all_cols:
        c_name = c["name"]
        try:
            with urllib.request.urlopen(f"{url_req}/collections/{c_name}", timeout=8) as cr:
                c_pts = json.loads(cr.read())["result"].get("points_count", 0)
                if c_pts == 0:
                    empty_cols.append(c_name)
        except Exception:
            pass

    line("stale collections", "None" if not stale_cols else f"FOUND: {stale_cols}", not stale_cols)
    if stale_cols:
        fails.append(f"Qdrant contains collections that do not match ingest_config_hash '{EXPECTED_HASH}': {stale_cols}")

    line("empty collections", "None" if not empty_cols else f"FOUND: {empty_cols}", not empty_cols)
    if empty_cols:
        fails.append(f"Qdrant contains empty collections with 0 points: {empty_cols}")
except Exception as exc:  # noqa: BLE001
    line("url", f"{url}  UNREACHABLE: {type(exc).__name__}", False)
    fails.append("Qdrant server unreachable")
    print("       -> docker ps ; docker start qdrant")

# ---------------------------------------------------------------- 5. git ---
print()
print(SEP)
print("GIT")
print(SEP)
try:
    out = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, timeout=15)
    dirty = [l for l in out.stdout.splitlines() if l.strip()]
    line("uncommitted files", str(len(dirty)))
    if len(dirty) > 30:
        warns.append(f"{len(dirty)} uncommitted files - commit before a long run")
    head = subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True, timeout=15)
    line("HEAD", head.stdout.strip()[:60])
except Exception as exc:  # noqa: BLE001
    warns.append(f"git unavailable: {exc}")

# ------------------------------------------------------------- verdict ----
print()
print(SEP)
print("VERDICT")
print(SEP)
for w in warns:
    print(f"  WARN  {w}")
for f in fails:
    print(f"  FAIL  {f}")
if fails:
    print("\n  DO NOT START A RUN. Fix the failures above first.")
    sys.exit(1)
print("  PASS  environment, config, index and git are in the expected state.")
print("\n  Paste this block at the top of your report.")
sys.exit(0)
