"""Full Corpus acquisition script for The Adaptive Retriever (Addenda A1, C, C1 & Change Order §9).
Fetches 400+ heterogeneous real public documents across 6 slices:
1. arXiv PDFs -> expected_doc_class: digital_prose (>=70)
2. SEC EDGAR 10-K filings -> expected_doc_class: table_heavy (>=70)
3. Rendered scanned PDFs from arXiv -> expected_doc_class: scanned (>=45, synthetic_scan: True)
4. Wikipedia HTML pages -> expected_doc_class: html (>=70)
5. GitHub source code files -> expected_doc_class: code (>=70)
6. Federal Reserve FOMC transcripts & press conferences -> expected_doc_class: transcript (>=70)

Outputs data/corpus/MANIFEST.jsonl with full metadata and relative paths.
All synthetic document generators have been completely deleted per Change Order §9.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
MANIFEST_PATH = CORPUS_DIR / "MANIFEST.jsonl"
FETCH_LOG_PATH = CORPUS_DIR / "fetch_log.json"

MANIFEST_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
SEEN_DOC_IDS = set()

PER_CLASS_FLOORS = {
    "table_heavy": {"min_count": 70, "min_mean_bytes": 204800},
    "transcript": {"min_count": 70, "min_mean_bytes": 81920},
    "scanned": {"min_count": 45, "min_mean_bytes": 307200},
    "digital_prose": {"min_count": 70, "min_mean_bytes": 204800},
    "html": {"min_count": 70, "min_mean_bytes": 30720},
    "code": {"min_count": 70, "min_mean_bytes": 2048},
}


def get_sec_user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise ValueError(
            "SEC_USER_AGENT environment variable is required for fetching SEC EDGAR documents. "
            "Please set SEC_USER_AGENT='Name user@domain.com' before running fetch_corpus.py."
        )
    parts = ua.split()
    has_name = len(parts) >= 2
    has_email = any("@" in p and "." in p for p in parts)
    if not (has_name and has_email):
        raise ValueError(
            f"Invalid SEC_USER_AGENT '{ua}'. Must contain both a name and an email address "
            "(e.g., 'AdaptiveRetrieverResearch research@adaptive-retriever.org') per SEC EDGAR requirements."
        )
    return ua


def load_fetch_log() -> dict:
    if FETCH_LOG_PATH.exists():
        try:
            return json.loads(FETCH_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_fetch_log(log_data: dict) -> None:
    with LOG_LOCK:
        FETCH_LOG_PATH.write_text(json.dumps(log_data, indent=2), encoding="utf-8")


def append_manifest_record(record: dict) -> bool:
    with MANIFEST_LOCK:
        file_path = PROJECT_ROOT / record["doc_path"]
        if file_path.exists():
            file_bytes = file_path.read_bytes()
            doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]
            record["doc_id"] = doc_id
            if doc_id in SEEN_DOC_IDS:
                print(f"DUPLICATE DETECTED: doc_id={doc_id} for path={record['doc_path']}. Skipping record.")
                return False
            SEEN_DOC_IDS.add(doc_id)

        with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return True


# 1. DIGITAL PROSE (arXiv PDFs)
def fetch_arxiv_item(arxiv_id: str, title: str, log_data: dict) -> bool:
    filename = f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
    doc_key = f"arxiv:{filename}"
    file_path = CORPUS_DIR / filename
    rel_path = str(file_path.relative_to(PROJECT_ROOT).as_posix())

    if doc_key in log_data and file_path.exists() and file_path.stat().st_size > 10000:
        record = {
            "doc_path": rel_path,
            "source": "arxiv",
            "source_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "licence": "arxiv_open_access",
            "retrieved_at": datetime.now().isoformat(),
            "expected_doc_class": "digital_prose",
            "synthetic_scan": False,
        }
        append_manifest_record(record)
        return True

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    time.sleep(0.5)
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            res = client.get(url, headers={"User-Agent": "AdaptiveRetrieverResearch/1.0"})
            if res.status_code == 200 and len(res.content) > 10000:
                file_path.write_bytes(res.content)
                record = {
                    "doc_path": rel_path,
                    "source": "arxiv",
                    "source_url": url,
                    "licence": "arxiv_open_access",
                    "retrieved_at": datetime.now().isoformat(),
                    "expected_doc_class": "digital_prose",
                    "synthetic_scan": False,
                }
                append_manifest_record(record)
                with LOG_LOCK:
                    log_data[doc_key] = {"status_code": 200, "bytes": len(res.content)}
                return True
            else:
                with LOG_LOCK:
                    log_data[doc_key] = {"status_code": res.status_code, "error": f"HTTP {res.status_code}"}
                print(f"Failed to fetch arXiv doc {arxiv_id}: HTTP {res.status_code}")
                return False
    except Exception as e:
        with LOG_LOCK:
            log_data[doc_key] = {"status_code": 0, "error": str(e)}
        print(f"Error fetching arXiv doc {arxiv_id}: {e}")
        return False


# 2. TABLE HEAVY (SEC EDGAR 10-K Filings)
def fetch_sec_10k_item(cik: str, accession_no: str, primary_doc: str, log_data: dict) -> bool:
    acc_clean = accession_no.replace("-", "")
    ext = Path(primary_doc).suffix or ".htm"
    filename = f"sec_10k_{cik}_{acc_clean}{ext}"
    doc_key = f"sec:{filename}"
    file_path = CORPUS_DIR / filename
    rel_path = str(file_path.relative_to(PROJECT_ROOT).as_posix())

    if doc_key in log_data and file_path.exists() and file_path.stat().st_size > 100000:
        record = {
            "doc_path": rel_path,
            "source": "sec_edgar_10k",
            "source_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary_doc}",
            "licence": "public_domain_us_gov",
            "retrieved_at": datetime.now().isoformat(),
            "expected_doc_class": "table_heavy",
            "synthetic_scan": False,
        }
        append_manifest_record(record)
        return True

    sec_ua = get_sec_user_agent()
    time.sleep(0.12)

    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary_doc}"
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            res = client.get(url, headers={"User-Agent": sec_ua})
            if res.status_code == 200 and len(res.content) > 100000:
                file_path.write_bytes(res.content)
                record = {
                    "doc_path": rel_path,
                    "source": "sec_edgar_10k",
                    "source_url": url,
                    "licence": "public_domain_us_gov",
                    "retrieved_at": datetime.now().isoformat(),
                    "expected_doc_class": "table_heavy",
                    "synthetic_scan": False,
                }
                append_manifest_record(record)
                with LOG_LOCK:
                    log_data[doc_key] = {"status_code": 200, "bytes": len(res.content)}
                return True
            else:
                with LOG_LOCK:
                    log_data[doc_key] = {"status_code": res.status_code, "error": f"HTTP {res.status_code}"}
                print(f"Failed SEC fetch for CIK {cik} accession {accession_no}: HTTP {res.status_code}")
                return False
    except Exception as e:
        with LOG_LOCK:
            log_data[doc_key] = {"status_code": 0, "error": str(e)}
        print(f"Error fetching SEC document CIK {cik}: {e}")
        return False


# 3. WIKIPEDIA HTML
def fetch_wiki_item(topic: str, log_data: dict) -> bool:
    filename = f"wikipedia_{topic.lower()}.html"
    doc_key = f"wikipedia:{topic}"
    file_path = CORPUS_DIR / filename
    rel_path = str(file_path.relative_to(PROJECT_ROOT).as_posix())

    if doc_key in log_data and file_path.exists() and file_path.stat().st_size > 15000:
        record = {
            "doc_path": rel_path,
            "source": "wikipedia",
            "source_url": f"https://en.wikipedia.org/api/rest_v1/page/html/{topic}",
            "licence": "CC-BY-SA-4.0",
            "retrieved_at": datetime.now().isoformat(),
            "expected_doc_class": "html",
            "synthetic_scan": False,
        }
        append_manifest_record(record)
        return True

    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{topic}"
    headers = {"User-Agent": "AdaptiveRetrieverResearch/1.0 (academic_research@domain.com)"}

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            res = client.get(url, headers=headers)
            if res.status_code == 200 and len(res.content) > 10000:
                file_path.write_bytes(res.content)
                record = {
                    "doc_path": rel_path,
                    "source": "wikipedia",
                    "source_url": url,
                    "licence": "CC-BY-SA-4.0",
                    "retrieved_at": datetime.now().isoformat(),
                    "expected_doc_class": "html",
                    "synthetic_scan": False,
                }
                append_manifest_record(record)
                with LOG_LOCK:
                    log_data[doc_key] = {"status_code": 200, "bytes": len(res.content)}
                return True
            else:
                with LOG_LOCK:
                    log_data[doc_key] = {"status_code": res.status_code, "error": f"HTTP {res.status_code}"}
                print(f"Failed Wikipedia fetch for {topic}: HTTP {res.status_code}")
                return False
    except Exception as e:
        with LOG_LOCK:
            log_data[doc_key] = {"status_code": 0, "error": str(e)}
        print(f"Error fetching Wikipedia page {topic}: {e}")
        return False


# 4. GITHUB CODE
def fetch_github_item(repo: str, file_subpath: str, log_data: dict) -> bool:
    filename = f"github_{repo.replace('/', '_')}_{file_subpath.replace('/', '_')}"
    doc_key = f"github:{filename}"
    file_path = CORPUS_DIR / filename
    rel_path = str(file_path.relative_to(PROJECT_ROOT).as_posix())

    if doc_key in log_data and file_path.exists() and file_path.stat().st_size > 1000:
        record = {
            "doc_path": rel_path,
            "source": "github_open_source_repo",
            "source_url": f"https://raw.githubusercontent.com/{repo}/main/{file_subpath}",
            "licence": "MIT_or_permissive",
            "retrieved_at": datetime.now().isoformat(),
            "expected_doc_class": "code",
            "synthetic_scan": False,
        }
        append_manifest_record(record)
        return True

    url = f"https://raw.githubusercontent.com/{repo}/main/{file_subpath}"
    headers = {"User-Agent": "AdaptiveRetrieverResearch/1.0"}

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            res = client.get(url, headers=headers)
            if res.status_code != 200:
                url = f"https://raw.githubusercontent.com/{repo}/master/{file_subpath}"
                res = client.get(url, headers=headers)

            if res.status_code == 200 and len(res.content) > 1000:
                file_path.write_bytes(res.content)
                record = {
                    "doc_path": rel_path,
                    "source": "github_open_source_repo",
                    "source_url": url,
                    "licence": "MIT_or_permissive",
                    "retrieved_at": datetime.now().isoformat(),
                    "expected_doc_class": "code",
                    "synthetic_scan": False,
                }
                append_manifest_record(record)
                with LOG_LOCK:
                    log_data[doc_key] = {"status_code": 200, "bytes": len(res.content)}
                return True
            else:
                with LOG_LOCK:
                    log_data[doc_key] = {"status_code": res.status_code, "error": f"HTTP {res.status_code}"}
                print(f"Failed GitHub fetch for {repo}/{file_subpath}: HTTP {res.status_code}")
                return False
    except Exception as e:
        with LOG_LOCK:
            log_data[doc_key] = {"status_code": 0, "error": str(e)}
        print(f"Error fetching GitHub file {repo}/{file_subpath}: {e}")
        return False


# 5. TRANSCRIPTS (Federal Reserve FOMC press conferences & meeting transcripts)
def fetch_transcript_item(date_str: str, kind: str, log_data: dict) -> bool:
    if kind == "pressconf":
        filename = f"transcript_fomc_pressconf_{date_str}.pdf"
        url = f"https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{date_str}.pdf"
    else:
        filename = f"transcript_fomc_meeting_{date_str}.pdf"
        url = f"https://www.federalreserve.gov/monetarypolicy/files/FOMC{date_str}meeting.pdf"

    doc_key = f"transcript:{filename}"
    file_path = CORPUS_DIR / filename
    rel_path = str(file_path.relative_to(PROJECT_ROOT).as_posix())

    if doc_key in log_data and file_path.exists() and file_path.stat().st_size > 50000:
        record = {
            "doc_path": rel_path,
            "source": "gov_transcripts_fomc",
            "source_url": url,
            "licence": "public_domain_us_gov",
            "retrieved_at": datetime.now().isoformat(),
            "expected_doc_class": "transcript",
            "synthetic_scan": False,
        }
        append_manifest_record(record)
        return True

    headers = {"User-Agent": "AdaptiveRetrieverResearch/1.0"}

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            res = client.get(url, headers=headers)
            if res.status_code == 200 and len(res.content) > 50000:
                file_path.write_bytes(res.content)
                record = {
                    "doc_path": rel_path,
                    "source": "gov_transcripts_fomc",
                    "source_url": url,
                    "licence": "public_domain_us_gov",
                    "retrieved_at": datetime.now().isoformat(),
                    "expected_doc_class": "transcript",
                    "synthetic_scan": False,
                }
                append_manifest_record(record)
                with LOG_LOCK:
                    log_data[doc_key] = {"status_code": 200, "bytes": len(res.content)}
                return True
            else:
                with LOG_LOCK:
                    log_data[doc_key] = {"status_code": res.status_code, "error": f"HTTP {res.status_code}"}
                print(f"Failed FOMC transcript fetch for {date_str} ({kind}): HTTP {res.status_code}")
                return False
    except Exception as e:
        with LOG_LOCK:
            log_data[doc_key] = {"status_code": 0, "error": str(e)}
        print(f"Error fetching transcript {date_str}: {e}")
        return False


# 6. SCANNED (Render 45 arXiv PDFs to 200 DPI page images & rebuild as image-only PDFs)
def render_scanned_pdf_from_arxiv(src_pdf_path: Path, idx: int, log_data: dict) -> bool:
    arxiv_id = src_pdf_path.stem.replace("arxiv_", "")
    filename = f"scanned_arxiv_{idx:03d}_{arxiv_id}.pdf"
    doc_key = f"scanned:{filename}"
    file_path = CORPUS_DIR / filename
    rel_path = str(file_path.relative_to(PROJECT_ROOT).as_posix())

    if doc_key in log_data and file_path.exists() and file_path.stat().st_size > 200000:
        record = {
            "doc_path": rel_path,
            "source": "arxiv_rendered_scan",
            "source_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "licence": "arxiv_open_access",
            "retrieved_at": datetime.now().isoformat(),
            "expected_doc_class": "scanned",
            "synthetic_scan": True,
        }
        append_manifest_record(record)
        return True

    try:
        doc = pymupdf.open(src_pdf_path)
        new_doc = pymupdf.open()
        mat = pymupdf.Matrix(200 / 72.0, 200 / 72.0)

        pages_to_process = min(len(doc), 15)
        for i in range(pages_to_process):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("jpeg")
            img_doc = pymupdf.open("jpeg", img_bytes)
            pdf_bytes = img_doc.convert_to_pdf()
            img_pdf = pymupdf.open("pdf", pdf_bytes)
            new_doc.insert_pdf(img_pdf)
            img_doc.close()
            img_pdf.close()

        doc.close()
        new_doc.save(file_path, deflate=True)
        new_doc.close()

        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        record = {
            "doc_path": rel_path,
            "source": "arxiv_rendered_scan",
            "source_url": url,
            "licence": "arxiv_open_access",
            "retrieved_at": datetime.now().isoformat(),
            "expected_doc_class": "scanned",
            "synthetic_scan": True,
        }
        append_manifest_record(record)
        with LOG_LOCK:
            log_data[doc_key] = {"status_code": 200, "bytes": file_path.stat().st_size}
        return True
    except Exception as e:
        with LOG_LOCK:
            log_data[doc_key] = {"status_code": 0, "error": str(e)}
        print(f"Error rendering scanned PDF for {src_pdf_path}: {e}")
        return False


def validate_corpus_post_fetch() -> dict:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"Post-fetch validation failed: MANIFEST.jsonl not found at {MANIFEST_PATH}")

    records = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    class_stats = {}
    for r in records:
        doc_class = r.get("expected_doc_class")
        doc_path = PROJECT_ROOT / r.get("doc_path")
        if not doc_path.exists():
            continue
        size_bytes = doc_path.stat().st_size
        if doc_class not in class_stats:
            class_stats[doc_class] = {"count": 0, "total_bytes": 0}
        class_stats[doc_class]["count"] += 1
        class_stats[doc_class]["total_bytes"] += size_bytes

    print("\n" + "=" * 70)
    print("=== POST-FETCH VALIDATION GATE REPORT ===")
    print("=" * 70)

    has_failures = False
    results = {}
    for doc_class, rules in PER_CLASS_FLOORS.items():
        stats = class_stats.get(doc_class, {"count": 0, "total_bytes": 0})
        cnt = stats["count"]
        mean_bytes = (stats["total_bytes"] / float(cnt)) if cnt > 0 else 0.0
        results[doc_class] = {"count": cnt, "mean_bytes": mean_bytes}

        min_cnt = rules["min_count"]
        min_mean = rules["min_mean_bytes"]

        status_cnt = "PASS" if cnt >= min_cnt else "FAIL"
        status_mean = "PASS" if mean_bytes >= min_mean else "FAIL"

        if cnt < min_cnt or mean_bytes < min_mean:
            has_failures = True

        print(
            f"Class: {doc_class:<15} | Count: {cnt:>3}/{min_cnt:>2} [{status_cnt}] | "
            f"Mean Size: {mean_bytes:>10.1f} bytes (Floor: {min_mean:>6} B) [{status_mean}]"
        )

    print("=" * 70)

    if has_failures:
        raise RuntimeError("POST-FETCH VALIDATION FAILED! Corpus size or count below floor.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Real Corpus acquisition script for The Adaptive Retriever.")
    args = parser.parse_args()

    # Validate SEC_USER_AGENT per §9
    sec_ua = get_sec_user_agent()
    print(f"SEC EDGAR User-Agent verified: {sec_ua}", flush=True)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    log_data = load_fetch_log()

    # Clear manifest for clean validation build
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    print("Starting REAL DOCUMENT CORPUS acquisition (Zero synthetic generators)...", flush=True)

    # 1. Digital Prose (arXiv PDFs) - Target 75
    arxiv_ids = [
        "1706.03762", "2203.15556", "2005.14165", "2302.13971", "2106.09685",
        "2001.08361", "1810.04805", "1910.10683", "2004.12832", "2201.11903",
        "2204.02311", "2205.14135", "2303.08774", "2305.10601", "2307.09288"
    ]
    for i in range(1, 61):
        arxiv_ids.append(f"2401.{i:05d}")

    # 2. SEC EDGAR CIKs - Target 85 filings
    ciks = [
        "0000320193", "0000789019", "0001652044", "0001018724", "0001045810",
        "0000019617", "0000034088", "0000200406", "0000050863", "0000093410",
        "0000080424", "0000310158", "0000070840", "0000040545", "0000018001",
        "0000804700", "0000732712", "0000051143", "0000066740", "0000014031",
        "0000731766", "0000088698", "0000092222", "0000072971", "0000002488",
        "0000067716", "0001065280", "0001326801", "0001318605", "0001090872",
        "0000004962", "0000006120", "0000012659", "0000093410", "0000070840"
    ]

    # Collect 10-K items dynamically from SEC submissions API
    sec_items = []
    headers = {"User-Agent": sec_ua}
    for cik in ciks:
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            res = httpx.get(url, headers=headers, timeout=15.0)
            if res.status_code == 200:
                recent = res.json().get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                accs = recent.get("accessionNumber", [])
                docs = recent.get("primaryDocument", [])
                cnt = 0
                for f, a, d in zip(forms, accs, docs):
                    if f == "10-K" and cnt < 4:
                        sec_items.append((cik, a, d))
                        cnt += 1
            time.sleep(0.12)
        except Exception as e:
            print(f"Error querying SEC submissions for CIK {cik}: {e}")

    # 3. Wikipedia HTML topics - Target 75
    wiki_topics = [
        "Artificial_intelligence", "Quantum_computing", "Photosynthesis", "Plate_tectonics",
        "Machine_learning", "Renaissance", "Thermodynamics", "Gene_editing",
        "Industrial_Revolution", "Neuroscience", "Solar_System", "Economics",
        "Microbiology", "Cryptography", "Philosophy_of_mind", "Astrophysics",
        "Deep_learning", "Natural_language_processing", "Computer_vision", "Robotics",
        "Algorithm", "Data_structure", "Operating_system", "Computer_network",
        "Database", "Cloud_computing", "Cybersecurity", "Software_engineering",
        "Bioinformatics", "Genomics", "Evolution", "Ecology", "Biochemistry",
        "Organic_chemistry", "Quantum_mechanics", "General_relativity", "Particle_physics",
        "Condensed_matter_physics", "Fluid_dynamics", "Optics", "Electromagnetism",
        "Nanotechnology", "Materials_science", "Renewable_energy", "Nuclear_power",
        "Space_exploration", "Cosmology", "Exoplanet", "Climate_change", "Oceanography",
        "Geology", "Seismology", "Volcanology", "Meteorology", "Hydrology",
        "Microeconomics", "Macroeconomics", "Financial_economics", "Game_theory",
        "Behavioral_economics", "Econometrics", "Monetary_policy", "Fiscal_policy",
        "International_trade", "Economic_growth", "Inflation", "Unemployment",
        "Central_bank", "Stock_market", "Corporate_finance", "Algorithmic_trading",
        "Information_theory", "Probability_theory", "Graph_theory", "Linear_algebra", "Calculus"
    ]

    # 4. GitHub files - Target 75
    github_files = [
        ("psf/requests", "src/requests/api.py"),
        ("psf/requests", "src/requests/models.py"),
        ("psf/requests", "src/requests/sessions.py"),
        ("psf/requests", "src/requests/adapters.py"),
        ("psf/requests", "src/requests/auth.py"),
        ("psf/requests", "src/requests/cookies.py"),
        ("psf/requests", "src/requests/exceptions.py"),
        ("psf/requests", "src/requests/hooks.py"),
        ("psf/requests", "src/requests/structures.py"),
        ("psf/requests", "src/requests/utils.py"),
        ("pallets/flask", "src/flask/app.py"),
        ("pallets/flask", "src/flask/blueprints.py"),
        ("pallets/flask", "src/flask/cli.py"),
        ("pallets/flask", "src/flask/config.py"),
        ("pallets/flask", "src/flask/ctx.py"),
        ("pallets/flask", "src/flask/helpers.py"),
        ("pallets/flask", "src/flask/logging.py"),
        ("pallets/flask", "src/flask/signals.py"),
        ("pallets/flask", "src/flask/views.py"),
        ("django/django", "django/db/models/base.py"),
        ("django/django", "django/views/generic/base.py"),
        ("django/django", "django/forms/forms.py"),
        ("django/django", "django/http/request.py"),
        ("django/django", "django/http/response.py"),
        ("django/django", "django/urls/base.py"),
        ("django/django", "django/core/management/base.py"),
        ("django/django", "django/utils/text.py"),
        ("django/django", "django/middleware/common.py"),
        ("django/django", "django/db/backends/base/base.py"),
        ("django/django", "django/template/base.py"),
        ("django/django", "django/test/client.py"),
        ("django/django", "django/contrib/auth/models.py"),
        ("django/django", "django/core/cache/base.py"),
        ("django/django", "django/core/files/base.py"),
        ("scikit-learn/scikit-learn", "sklearn/base.py"),
        ("scikit-learn/scikit-learn", "sklearn/pipeline.py"),
        ("scikit-learn/scikit-learn", "sklearn/utils/validation.py"),
        ("scikit-learn/scikit-learn", "sklearn/linear_model/_base.py"),
        ("scikit-learn/scikit-learn", "sklearn/cluster/_kmeans.py"),
        ("scikit-learn/scikit-learn", "sklearn/metrics/_classification.py"),
        ("scikit-learn/scikit-learn", "sklearn/model_selection/_split.py"),
        ("scikit-learn/scikit-learn", "sklearn/tree/_classes.py"),
        ("scikit-learn/scikit-learn", "sklearn/ensemble/_forest.py"),
        ("scikit-learn/scikit-learn", "sklearn/preprocessing/_data.py"),
        ("fastapi/fastapi", "fastapi/applications.py"),
        ("fastapi/fastapi", "fastapi/routing.py"),
        ("fastapi/fastapi", "fastapi/params.py"),
        ("fastapi/fastapi", "fastapi/datastructures.py"),
        ("fastapi/fastapi", "fastapi/exceptions.py"),
        ("fastapi/fastapi", "fastapi/requests.py"),
        ("fastapi/fastapi", "fastapi/responses.py"),
        ("fastapi/fastapi", "fastapi/encoders.py"),
        ("huggingface/transformers", "src/transformers/modeling_utils.py"),
        ("huggingface/transformers", "src/transformers/tokenization_utils.py"),
        ("huggingface/transformers", "src/transformers/trainer.py"),
        ("huggingface/transformers", "src/transformers/configuration_utils.py"),
        ("huggingface/transformers", "src/transformers/generation/utils.py"),
        ("huggingface/transformers", "src/transformers/models/bert/modeling_bert.py"),
        ("huggingface/transformers", "src/transformers/models/gpt2/modeling_gpt2.py"),
        ("huggingface/transformers", "src/transformers/models/llama/modeling_llama.py"),
        ("huggingface/transformers", "src/transformers/models/auto/modeling_auto.py"),
        ("huggingface/transformers", "src/transformers/optimization.py"),
        ("huggingface/transformers", "src/transformers/feature_extraction_utils.py"),
        ("pandas-dev/pandas", "pandas/core/frame.py"),
        ("pandas-dev/pandas", "pandas/core/series.py"),
        ("pandas-dev/pandas", "pandas/core/indexing.py"),
        ("pandas-dev/pandas", "pandas/core/groupby/generic.py"),
        ("pandas-dev/pandas", "pandas/core/arrays/categorical.py"),
        ("pandas-dev/pandas", "pandas/io/parsers/readers.py"),
        ("numpy/numpy", "numpy/core/numeric.py"),
        ("numpy/numpy", "numpy/linalg/linalg.py"),
        ("numpy/numpy", "numpy/fft/_pocketfft.py"),
        ("scipy/scipy", "scipy/optimize/_minimize.py"),
        ("scipy/scipy", "scipy/stats/_continuous_distns.py"),
        ("textualize/rich", "rich/console.py"),
        ("textualize/rich", "rich/table.py"),
        ("textualize/rich", "rich/progress.py"),
        ("textualize/rich", "rich/syntax.py"),
        ("textualize/rich", "rich/panel.py"),
        ("sympy/sympy", "sympy/core/basic.py"),
        ("sympy/sympy", "sympy/core/expr.py"),
        ("sympy/sympy", "sympy/matrices/dense.py"),
        ("sympy/sympy", "sympy/solvers/solvers.py"),
        ("sympy/sympy", "sympy/polys/polytools.py"),
        ("python/cpython", "Lib/os.py"),
        ("python/cpython", "Lib/pathlib.py"),
        ("python/cpython", "Lib/json/__init__.py"),
        ("python/cpython", "Lib/asyncio/base_events.py"),
        ("python/cpython", "Lib/urllib/request.py"),
    ]

    # 5. FOMC Transcripts & Press Conferences - Target 75
    fomc_pressconf_dates = [
        "20240131", "20240320", "20240501", "20240612", "20240731", "20240918", "20241107", "20241218",
        "20230201", "20230322", "20230503", "20230614", "20230726", "20230920", "20231101", "20231213",
        "20220126", "20220316", "20220504", "20220615", "20220727", "20220921", "20221102", "20221214",
        "20210127", "20210317", "20210428", "20210616", "20210728", "20210922", "20211103", "20211215",
        "20200129", "20200303", "20200315", "20200429", "20200610", "20200729", "20200916", "20201105", "20201216",
        "20190130", "20190320", "20190501", "20190619", "20190731", "20190918", "20191030", "20191211",
        "20180321", "20180613", "20180926", "20181219", "20170315", "20170614", "20170920", "20171213",
        "20160316", "20160615", "20160921", "20161214", "20150318", "20150617", "20150917", "20151216"
    ]
    fomc_meeting_dates = [
        "20180321", "20180502", "20180613", "20180801", "20180926", "20181108", "20181219",
        "20170131", "20170315", "20170503", "20170614", "20170726", "20170920", "20171031", "20171213",
        "20160127", "20160316", "20160427", "20160615", "20160727", "20160921", "20161102", "20161214"
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        # 1. Digital Prose
        for aid in arxiv_ids[:75]:
            futures.append(executor.submit(fetch_arxiv_item, aid, f"ArXiv {aid}", log_data))

        # 2. SEC 10-Ks
        for cik, acc, doc in sec_items[:85]:
            futures.append(executor.submit(fetch_sec_10k_item, cik, acc, doc, log_data))

        # 3. Wikipedia HTML
        for topic in wiki_topics[:75]:
            futures.append(executor.submit(fetch_wiki_item, topic, log_data))

        # 4. GitHub Code
        for repo, fpath in github_files[:85]:
            futures.append(executor.submit(fetch_github_item, repo, fpath, log_data))

        # 5. Transcripts
        for d in fomc_pressconf_dates[:60]:
            futures.append(executor.submit(fetch_transcript_item, d, "pressconf", log_data))
        for d in fomc_meeting_dates[:20]:
            futures.append(executor.submit(fetch_transcript_item, d, "meeting", log_data))

        completed = 0
        for f in as_completed(futures):
            if f.result():
                completed += 1
            if completed % 50 == 0:
                print(f"Fetch progress: {completed}/{len(futures)} downloaded...")
                save_fetch_log(log_data)

    save_fetch_log(log_data)

    # 6. Scanned: Render 45 downloaded arXiv PDFs to 200 DPI image-only PDFs
    print("Rendering 45 scanned image-only PDFs from arXiv originals at 200 DPI...")
    arxiv_files = list(CORPUS_DIR.glob("arxiv_*.pdf"))
    scanned_count = 0
    for idx, pdf_file in enumerate(arxiv_files[:45], 1):
        if render_scanned_pdf_from_arxiv(pdf_file, idx, log_data):
            scanned_count += 1
    print(f"Scanned PDF rendering complete: {scanned_count}/45 rendered.")

    save_fetch_log(log_data)

    # Automated Post-Fetch Validation Gate
    validate_corpus_post_fetch()


if __name__ == "__main__":
    main()
