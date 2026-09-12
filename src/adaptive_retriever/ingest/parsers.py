"""Parser router and parser implementations (A2, spec §9.2)."""

import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TableRegion(BaseModel):
    char_start: int
    char_end: int
    header_text: str
    markdown: str
    page_no: Optional[int] = None


class ParsedDocument(BaseModel):
    text: str
    page_map: List[Tuple[int, int, int]] = []
    sections: List[Tuple[str, int, int]] = []
    tables: List[TableRegion] = []


class Parser(Protocol):
    name: str

    def parse(self, path: Path) -> ParsedDocument:
        ...


class PyMuPDFParser:
    name = "pymupdf"

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import fitz
            doc = fitz.open(str(path))
            pages = []
            for page in doc:
                txt = page.get_text("text")
                if txt.strip():
                    pages.append(txt.strip())
            doc.close()
            text = "\n\n".join(pages)
            return ParsedDocument(text=text)
        except Exception as e:
            logger.error(f"PyMuPDFParser failed on {path}: {e}")
            return ParsedDocument(text="")



_RAPID_OCR_ENGINE = None


def _get_rapid_ocr():
    global _RAPID_OCR_ENGINE
    if _RAPID_OCR_ENGINE is None:
        import rapidocr
        _RAPID_OCR_ENGINE = rapidocr.RapidOCR()
    return _RAPID_OCR_ENGINE


class DoclingParser:
    name = "docling"

    def parse(self, path: Path, ocr_max_pages: int = 15) -> ParsedDocument:
        try:
            import fitz
            doc = fitz.open(str(path))
            page_count = len(doc)
            extracted_pages = []

            max_pages = min(page_count, ocr_max_pages)
            ocr = None

            for i in range(max_pages):
                try:
                    p_text = doc[i].get_text("text").strip()
                    if len(p_text) > 30:
                        extracted_pages.append(p_text)
                    else:
                        if ocr is None:
                            ocr = _get_rapid_ocr()
                        pix = doc[i].get_pixmap(dpi=150)
                        img_bytes = pix.tobytes("png")
                        res = ocr(img_bytes)
                        if hasattr(res, "txts") and res.txts:
                            page_text = "\n".join(res.txts)
                        elif isinstance(res, (list, tuple)) and res and res[0]:
                            page_text = "\n".join(line[1] for line in res[0] if len(line) > 1)
                        else:
                            page_text = ""
                        if page_text:
                            extracted_pages.append(page_text)
                except Exception as page_err:
                    logger.warning(f"DoclingParser page {i} failed on {path}: {page_err}")

            doc.close()
            full_text = "\n\n".join(extracted_pages)
            return ParsedDocument(text=full_text)
        except Exception as e:
            logger.error(f"DoclingParser failed on {path}: {e}")
            return ParsedDocument(text="")


class TrafilaturaParser:
    name = "trafilatura"

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import trafilatura
            html_content = path.read_text(encoding="utf-8", errors="ignore")
            text = trafilatura.extract(html_content, include_tables=True, include_links=False)
            if not text:
                import re
                text = re.sub(r'<[^>]+>', ' ', html_content)
                text = re.sub(r'\s+', ' ', text).strip()
            return ParsedDocument(text=text)
        except Exception as e:
            logger.error(f"TrafilaturaParser failed on {path}: {e}")
            try:
                import re
                html_content = path.read_text(encoding="utf-8", errors="ignore")
                text = re.sub(r'<[^>]+>', ' ', html_content)
                text = re.sub(r'\s+', ' ', text).strip()
                return ParsedDocument(text=text)
            except Exception:
                return ParsedDocument(text="")


class TreeSitterParser:
    name = "tree_sitter"

    def parse(self, path: Path) -> ParsedDocument:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(text=text)
        except Exception as e:
            logger.error(f"TreeSitterParser failed on {path}: {e}")
            return ParsedDocument(text="")


class PassthroughParser:
    name = "passthrough"

    def parse(self, path: Path) -> ParsedDocument:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(text=text)
        except Exception as e:
            logger.error(f"PassthroughParser failed on {path}: {e}")
            return ParsedDocument(text="")


PARSERS = {
    "pymupdf": PyMuPDFParser(),
    "docling": DoclingParser(),
    "trafilatura": TrafilaturaParser(),
    "tree_sitter": TreeSitterParser(),
    "passthrough": PassthroughParser(),
}

PARSE_CACHE_DIR = Path("data/parse_cache")
CANONICAL_DIR = Path("data/canonical")

_PARSE_CACHE_STATS = {"hits": 0, "misses": 0}


def get_parse_cache_stats() -> dict:
    return dict(_PARSE_CACHE_STATS)


def reset_parse_cache_stats() -> None:
    _PARSE_CACHE_STATS["hits"] = 0
    _PARSE_CACHE_STATS["misses"] = 0


import hashlib


def get_parse_cache_key(doc_id: str, parser_name: str, parser_version: str = "1.0") -> str:
    raw_key = f"{doc_id}:{parser_name}:{parser_version}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


def get_cached_parse(
    doc_id: str,
    parser_name: str = "pymupdf",
    parser_version: str = "1.0",
) -> Optional[ParsedDocument]:
    # Check parse_cache dir first
    cache_key = get_parse_cache_key(doc_id, parser_name, parser_version)
    cache_file = PARSE_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return ParsedDocument(**data)
        except (json.JSONDecodeError, OSError, ValueError) as err:
            logger.warning(f"Corrupt or unreadable cache file at {cache_file}: {err}")

    # Check canonical dir fallback (only valid 16-char sha256 hex doc_ids)
    if len(doc_id) == 16 and all(c in "0123456789abcdef" for c in doc_id):
        canonical_file = CANONICAL_DIR / f"{doc_id}.txt"
        if canonical_file.exists():
            text = canonical_file.read_text(encoding="utf-8")
            if text.strip():
                return ParsedDocument(text=text)

    return None


def save_cached_parse(
    doc_id: str,
    parser_name: str,
    parsed_doc: ParsedDocument,
    parser_version: str = "1.0",
) -> None:
    PARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_parse_cache_key(doc_id, parser_name, parser_version)
    cache_file = PARSE_CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(parsed_doc.model_dump_json(), encoding="utf-8")


def save_canonical_parse(doc_id: str, text: str) -> None:
    """Canonical parses in data/canonical/ are written ONLY by corpus-prepare (Item 3)."""
    assert len(doc_id) == 16 and all(c in "0123456789abcdef" for c in doc_id), (
        f"Invalid doc_id '{doc_id}'! Canonical parse files must be named <16-char sha256 hex>.txt"
    )
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    canonical_file = CANONICAL_DIR / f"{doc_id}.txt"
    canonical_file.write_text(text, encoding="utf-8")


def parse_document(
    path: Path,
    parser_name: str,
    config: Any = None,
    ocr_max_pages: Optional[int] = None,
) -> Tuple[ParsedDocument, float, Optional[str]]:
    start_time = time.perf_counter()
    parser = PARSERS.get(parser_name, PARSERS["pymupdf"])

    if ocr_max_pages is None and config is not None:
        if hasattr(config, "raw"):
            ocr_max_pages = config.raw.get("ingest", {}).get("ocr_max_pages", 15)
        elif isinstance(config, dict):
            ocr_max_pages = config.get("ingest", {}).get("ocr_max_pages", 15)

    if ocr_max_pages is None:
        ocr_max_pages = 15

    try:
        if parser_name == "docling" and hasattr(parser, "parse"):
            parsed = parser.parse(path, ocr_max_pages=ocr_max_pages)
        else:
            parsed = parser.parse(path)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        error = None if parsed.text else "empty_parse_output"
        return parsed, elapsed_ms, error
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return ParsedDocument(text=""), elapsed_ms, str(e)


def parse_document_cached(
    path: Path,
    parser_name: str,
    doc_id: Optional[str] = None,
    config: Any = None,
    parser_version: str = "1.0",
) -> Tuple[ParsedDocument, float, Optional[str]]:
    if doc_id is None or len(doc_id) != 16 or not all(c in "0123456789abcdef" for c in doc_id):
        if path.exists():
            doc_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        else:
            doc_id = hashlib.sha256(path.stem.encode("utf-8")).hexdigest()[:16]

    cached = get_cached_parse(doc_id, parser_name, parser_version)
    if cached is not None and len(cached.text.strip()) >= 100:
        _PARSE_CACHE_STATS["hits"] += 1
        return cached, 0.0, None

    _PARSE_CACHE_STATS["misses"] += 1
    parsed, elapsed_ms, err = parse_document(path, parser_name, config=config)
    if not err and len(parsed.text.strip()) >= 100:
        save_cached_parse(doc_id, parser_name, parsed, parser_version=parser_version)

    return parsed, elapsed_ms, err
