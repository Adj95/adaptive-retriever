"""Document Profiler (A1, spec §9.1).
Deterministic document fingerprinting without an LLM.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF
from adaptive_retriever.models import DocClass, DocumentFingerprint

LOW_CONFIDENCE_LOG = Path("reports/low_confidence_fingerprints.jsonl")

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs", ".php",
    ".rb", ".cs", ".sh", ".sql", ".css", ".json", ".yaml", ".yml"
}


def log_low_confidence_fingerprint(fp: DocumentFingerprint) -> None:
    LOW_CONFIDENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOW_CONFIDENCE_LOG, "a", encoding="utf-8") as f:
        f.write(fp.model_dump_json() + "\n")


def profile(path: Path | str) -> DocumentFingerprint:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found for profiling: {p}")

    ext = p.suffix.lower()
    text_content = ""
    page_count = 1
    table_region_count = 0
    heading_depth = 0
    column_count_estimate = 1
    has_extractable_text = False
    text_to_image_ratio = 1.0

    if ext in (".pdf",):
        try:
            doc = fitz.open(str(p))
            page_count = max(len(doc), 1)
            total_chars = 0
            page_texts = []
            for i, page in enumerate(doc):
                txt = page.get_text()
                page_texts.append(txt)
                total_chars += len(txt)
                if i < 2:
                    try:
                        tables = page.find_tables()
                        table_region_count += len(tables)
                    except Exception:
                        pass
                table_region_count += len(re.findall(r"\|.*\|.*\|", txt))

            text_content = "\n".join(page_texts)
            chars_per_page = total_chars / float(page_count)
            has_extractable_text = chars_per_page >= 100.0
            text_to_image_ratio = min(2.0, max(0.0, chars_per_page / 1800.0))
            doc.close()

        except Exception:
            has_extractable_text = False
            text_to_image_ratio = 0.0
    else:
        text_content = p.read_text(encoding="utf-8", errors="ignore")
        has_extractable_text = len(text_content.strip()) > 50
        page_count = max(1, len(text_content) // 3000)

        if ext in (".html", ".htm"):
            table_region_count = len(re.findall(r"<table", text_content, re.IGNORECASE))
            headings = re.findall(r"<h([1-6])", text_content, re.IGNORECASE)
            if headings:
                heading_depth = max(int(h) for h in headings)

    lines = [line for line in text_content.splitlines() if line.strip()]
    total_lines = max(len(lines), 1)

    # Heuristics
    code_lines = 0
    speaker_lines = 0
    speaker_regex = re.compile(r"^[A-Z][A-Za-z .]{1,30}:\s")

    for line in lines:
        stripped = line.strip()
        if speaker_regex.match(stripped):
            speaker_lines += 1
        if ext in CODE_EXTENSIONS or stripped.startswith(("def ", "class ", "import ", "function ", "return ", "if ", "for ")):
            code_lines += 1

    code_block_ratio = code_lines / float(total_lines)
    speaker_turn_ratio = speaker_lines / float(total_lines)
    avg_line_length = len(text_content) / float(total_lines) if total_lines > 0 else 0.0

    # Language detection helper (simple stop-word heuristic)
    language = "en"
    lower_text = text_content[:5000].lower()
    french_words = {"le", "la", "les", "des", "est", "une", "pour", "dans", "est", "pas", "sur"}
    german_words = {"der", "die", "das", "und", "ist", "für", "mit", "den", "von"}
    words = set(re.findall(r"\b[a-z]{2,}\b", lower_text))
    if len(words & french_words) >= 4:
        language = "fr"
    elif len(words & german_words) >= 4:
        language = "de"

    # Decision protocol (first match wins per spec §9.1)
    doc_class = DocClass.UNKNOWN
    confidence = 0.20

    table_density = table_region_count / float(page_count)

    if not has_extractable_text or text_to_image_ratio < 0.15:
        doc_class = DocClass.SCANNED
        if not has_extractable_text:
            confidence = 0.98
        else:
            confidence = round(0.60 + 0.35 * (1.0 - min(1.0, text_to_image_ratio / 0.15)), 4)

    elif ext in CODE_EXTENSIONS or (code_block_ratio > 0.5 and ext not in (".pdf", ".html", ".htm", ".txt", ".md")):
        doc_class = DocClass.CODE
        if ext in CODE_EXTENSIONS:
            confidence = round(0.85 + 0.15 * min(1.0, code_block_ratio), 4)
        else:
            dist = (code_block_ratio - 0.5) / 0.5
            confidence = round(0.50 + 0.50 * min(1.0, max(0.0, dist)), 4)

    elif speaker_turn_ratio > 0.25:
        doc_class = DocClass.TRANSCRIPT
        dist = (speaker_turn_ratio - 0.25) / 0.75
        confidence = round(0.50 + 0.48 * min(1.0, max(0.05, dist)), 4)

    elif table_density > 0.5:
        doc_class = DocClass.TABLE_HEAVY
        dist = (table_density - 0.5) / 1.5
        confidence = round(0.50 + 0.50 * min(1.0, max(0.0, dist)), 4)

    elif ext in (".html", ".htm") or re.search(r"<!DOCTYPE\s+html|<html[\s>]", text_content[:1000], re.IGNORECASE):
        doc_class = DocClass.HTML
        html_tag_count = len(re.findall(r"<[a-z1-6]+[\s/>]", text_content, re.IGNORECASE))
        tag_density = min(1.0, html_tag_count / float(total_lines))
        confidence = round(0.70 + 0.30 * tag_density, 4)

    elif has_extractable_text:
        doc_class = DocClass.DIGITAL_PROSE
        chars_per_page = len(text_content) / float(page_count)
        prose_score = min(1.0, max(0.0, (chars_per_page - 500) / 2500.0))
        confidence = round(0.50 + 0.45 * prose_score, 4)

    fp = DocumentFingerprint(
        source_path=str(p),
        mime_type="text/html" if ext in (".html", ".htm") else "application/pdf" if ext == ".pdf" else "text/plain",
        page_count=page_count,
        char_count=len(text_content),
        text_to_image_ratio=text_to_image_ratio,
        has_extractable_text=has_extractable_text,
        table_region_count=table_region_count,
        heading_depth=heading_depth,
        avg_line_length=avg_line_length,
        column_count_estimate=column_count_estimate,
        code_block_ratio=code_block_ratio,
        speaker_turn_ratio=speaker_turn_ratio,
        language=language,
        doc_class=doc_class,
        class_confidence=confidence,
    )

    if fp.class_confidence < 0.4:
        log_low_confidence_fingerprint(fp)

    return fp
