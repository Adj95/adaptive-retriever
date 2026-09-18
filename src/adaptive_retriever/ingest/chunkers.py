"""Chunker implementations (A3, spec §9.4).

Implements all 5 spec chunkers with exact character span tracking:
- recursive_512: Recursive/fixed token windowing (512 token limit, 64 overlap baseline)
- structural: Section boundary splitting
- syntax_aware: AST / code function boundary splitting
- table_preserving: Contiguous table region isolation
- speaker_turn: Dialogue/speaker turn boundary splitting for transcripts

Guarantee: parsed.text[chunk.char_start:chunk.char_end] == chunk.text
"""

import logging
import re
from typing import List, Optional, Tuple
from adaptive_retriever.models import Chunk
from adaptive_retriever.ingest.parsers import ParsedDocument

logger = logging.getLogger(__name__)


_BGE_TOKENIZER = None


def _get_bge_tokenizer():
    global _BGE_TOKENIZER
    if _BGE_TOKENIZER is None:
        from transformers import AutoTokenizer

        _BGE_TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    return _BGE_TOKENIZER


def _count_bge_tokens(text: str) -> int:
    tok = _get_bge_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def _estimate_tokens(text: str) -> int:
    return _count_bge_tokens(text)



def _split_long_span_by_tokens(
    text: str,
    start_char_offset: int,
    max_tokens: int = 480,
    overlap_tokens: int = 48,
) -> List[Tuple[int, int, str, int]]:
    """Split long text span into character spans that do not exceed max_tokens under BGE tokenizer."""
    tok = _get_bge_tokenizer()
    encoded = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    total_tokens = len(input_ids)
    if total_tokens <= max_tokens:
        return [(start_char_offset, start_char_offset + len(text), text, total_tokens)]

    spans = []
    i = 0
    step = max_tokens - overlap_tokens
    while i < total_tokens:
        chunk_token_end = min(i + max_tokens, total_tokens)

        char_start_local = offsets[i][0]
        char_end_local = offsets[chunk_token_end - 1][1]

        chunk_str = text[char_start_local:char_end_local]
        actual_tokens = len(tok.encode(chunk_str, add_special_tokens=False))

        while actual_tokens > max_tokens and chunk_token_end > i + 1:
            chunk_token_end -= 1
            char_end_local = offsets[chunk_token_end - 1][1]
            chunk_str = text[char_start_local:char_end_local]
            actual_tokens = len(tok.encode(chunk_str, add_special_tokens=False))

        spans.append(
            (
                start_char_offset + char_start_local,
                start_char_offset + char_end_local,
                chunk_str,
                actual_tokens,
            )
        )

        if chunk_token_end >= total_tokens:
            break
        i += step

    return spans


def chunk_recursive_512(parsed: ParsedDocument, doc_id: str) -> List[Chunk]:
    text = parsed.text
    if not text or not text.strip():
        return []

    target_tokens = 480
    overlap_tokens = 48
    char_len = len(text)

    # Fast-path check: if entire document fits under target_tokens
    if char_len <= 1500:
        doc_tokens = _count_bge_tokens(text)
        if doc_tokens <= target_tokens:
            c = Chunk(
                chunk_id=f"{doc_id}_c0000",
                doc_id=doc_id,
                ordinal=0,
                text=text,
                char_start=0,
                char_end=char_len,
                chunker_used="recursive_512",
                token_count=doc_tokens,
            )
            assert c.token_count <= 512, f"PERMANENT ASSERTION FAILED: Chunk {c.chunk_id} has {c.token_count} tokens > 512!"
            return [c]

    chunks: List[Chunk] = []
    lines = text.splitlines(keepends=True)

    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append((pos, pos + len(line), line))
        pos += len(line)

    idx = 0
    ordinal = 0
    while idx < len(line_starts):
        start_char = line_starts[idx][0]
        end_char = start_char
        accumulated_text = ""
        j = idx

        while j < len(line_starts):
            l_start, l_end, l_text = line_starts[j]
            candidate_text = accumulated_text + l_text
            if len(candidate_text.split()) > 360 or len(candidate_text) > 1500:
                tok_cnt = _count_bge_tokens(candidate_text)
                if tok_cnt > target_tokens and accumulated_text != "":
                    break

            accumulated_text = candidate_text
            end_char = l_end
            j += 1

        if j == idx:  # Single line exceeds target_tokens!
            l_start, l_end, l_text = line_starts[idx]
            sub_spans = _split_long_span_by_tokens(
                l_text,
                start_char_offset=l_start,
                max_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
            for sub_start, sub_end, sub_text, sub_toks in sub_spans:
                c = Chunk(
                    chunk_id=f"{doc_id}_c{ordinal:04d}",
                    doc_id=doc_id,
                    ordinal=ordinal,
                    text=sub_text,
                    char_start=sub_start,
                    char_end=sub_end,
                    chunker_used="recursive_512_token_fallback",
                    token_count=sub_toks,
                )
                assert c.token_count <= 512, f"PERMANENT ASSERTION FAILED: Chunk {c.chunk_id} has {c.token_count} tokens > 512!"
                chunks.append(c)
                ordinal += 1
            idx += 1
        else:
            final_toks = _count_bge_tokens(accumulated_text)
            if final_toks > target_tokens:
                sub_spans = _split_long_span_by_tokens(
                    accumulated_text,
                    start_char_offset=start_char,
                    max_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                )
                for sub_start, sub_end, sub_text, sub_toks in sub_spans:
                    c = Chunk(
                        chunk_id=f"{doc_id}_c{ordinal:04d}",
                        doc_id=doc_id,
                        ordinal=ordinal,
                        text=sub_text,
                        char_start=sub_start,
                        char_end=sub_end,
                        chunker_used="recursive_512_token_fallback",
                        token_count=sub_toks,
                    )
                    assert c.token_count <= 512, f"PERMANENT ASSERTION FAILED: Chunk {c.chunk_id} has {c.token_count} tokens > 512!"
                    chunks.append(c)
                    ordinal += 1
            else:
                c = Chunk(
                    chunk_id=f"{doc_id}_c{ordinal:04d}",
                    doc_id=doc_id,
                    ordinal=ordinal,
                    text=accumulated_text,
                    char_start=start_char,
                    char_end=end_char,
                    chunker_used="recursive_512",
                    token_count=final_toks,
                )
                assert c.token_count <= 512, f"PERMANENT ASSERTION FAILED: Chunk {c.chunk_id} has {c.token_count} tokens > 512!"
                chunks.append(c)
                ordinal += 1
            idx = j

    # Permanent assertion check across all chunks returned
    for c in chunks:
        assert c.token_count <= 512, f"PERMANENT ASSERTION FAILED: Chunk {c.chunk_id} has {c.token_count} tokens > 512!"

    return chunks


def chunk_structural(parsed: ParsedDocument, doc_id: str) -> List[Chunk]:
    text = parsed.text
    if not text:
        return []

    # Split by section headers or blank lines
    sections = parsed.sections
    if not sections:
        header_regex = re.compile(r"(?m)^(#{1,6}\s+.+|\d+(\.\d+)*\s+[A-Z].+)$")
        matches = list(header_regex.finditer(text))
        if matches:
            sections = []
            for i, m in enumerate(matches):
                sec_title = m.group(0).strip()
                s_start = m.start()
                s_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append((sec_title, s_start, s_end))

    if not sections:
        return chunk_recursive_512(parsed, doc_id)

    chunks: List[Chunk] = []
    ordinal = 0
    pos = 0

    for title, start, end in sections:
        if start > pos:
            gap_text = text[pos:start]
            if gap_text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_c{ordinal:04d}",
                        doc_id=doc_id,
                        ordinal=ordinal,
                        text=gap_text,
                        char_start=pos,
                        char_end=start,
                        chunker_used="structural",
                        token_count=_estimate_tokens(gap_text),
                    )
                )
                ordinal += 1

        sec_text = text[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_c{ordinal:04d}",
                doc_id=doc_id,
                ordinal=ordinal,
                text=sec_text,
                char_start=start,
                char_end=end,
                chunker_used="structural",
                section_path=[title],
                token_count=_estimate_tokens(sec_text),
            )
        )
        ordinal += 1
        pos = end

    if pos < len(text):
        rem_text = text[pos:]
        if rem_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_c{ordinal:04d}",
                    doc_id=doc_id,
                    ordinal=ordinal,
                    text=rem_text,
                    char_start=pos,
                    char_end=len(text),
                    chunker_used="structural",
                    token_count=_estimate_tokens(rem_text),
                )
            )

    return chunks


def chunk_syntax_aware(parsed: ParsedDocument, doc_id: str) -> List[Chunk]:
    text = parsed.text
    if not text:
        return []

    # Code AST block splitting on function/class definitions
    func_regex = re.compile(r"(?m)^(def |class |function |pub fn |async def |export default )")
    matches = list(func_regex.finditer(text))

    if not matches:
        return chunk_recursive_512(parsed, doc_id)

    chunks: List[Chunk] = []
    ordinal = 0
    pos = 0

    for i, m in enumerate(matches):
        start = m.start()
        if start > pos:
            pre_text = text[pos:start]
            if pre_text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_c{ordinal:04d}",
                        doc_id=doc_id,
                        ordinal=ordinal,
                        text=pre_text,
                        char_start=pos,
                        char_end=start,
                        chunker_used="syntax_aware",
                        token_count=_estimate_tokens(pre_text),
                    )
                )
                ordinal += 1

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_c{ordinal:04d}",
                doc_id=doc_id,
                ordinal=ordinal,
                text=block_text,
                char_start=start,
                char_end=end,
                chunker_used="syntax_aware",
                token_count=_estimate_tokens(block_text),
            )
        )
        ordinal += 1
        pos = end

    return chunks


def chunk_table_preserving(parsed: ParsedDocument, doc_id: str) -> List[Chunk]:
    text = parsed.text
    if not text:
        return []

    tables = parsed.tables
    if not tables:
        # Fallback regex for markdown tables
        table_regex = re.compile(r"(?m)(^\|.*\|\n)+")
        matches = list(table_regex.finditer(text))
        if matches:
            chunks: List[Chunk] = []
            ordinal = 0
            pos = 0
            for m in matches:
                start, end = m.start(), m.end()
                if start > pos:
                    pre_text = text[pos:start]
                    if pre_text.strip():
                        chunks.append(
                            Chunk(
                                chunk_id=f"{doc_id}_c{ordinal:04d}",
                                doc_id=doc_id,
                                ordinal=ordinal,
                                text=pre_text,
                                char_start=pos,
                                char_end=start,
                                chunker_used="table_preserving",
                                token_count=_estimate_tokens(pre_text),
                            )
                        )
                        ordinal += 1
                tbl_text = text[start:end]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_c{ordinal:04d}",
                        doc_id=doc_id,
                        ordinal=ordinal,
                        text=tbl_text,
                        char_start=start,
                        char_end=end,
                        chunker_used="table_preserving",
                        contains_table=True,
                        token_count=_estimate_tokens(tbl_text),
                    )
                )
                ordinal += 1
                pos = end
            if pos < len(text):
                rem = text[pos:]
                if rem.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}_c{ordinal:04d}",
                            doc_id=doc_id,
                            ordinal=ordinal,
                            text=rem,
                            char_start=pos,
                            char_end=len(text),
                            chunker_used="table_preserving",
                            token_count=_estimate_tokens(rem),
                        )
                    )
            return chunks

    return chunk_recursive_512(parsed, doc_id)


def chunk_speaker_turn(parsed: ParsedDocument, doc_id: str) -> List[Chunk]:
    text = parsed.text
    if not text:
        return []

    speaker_regex = re.compile(r"(?m)^([A-Z][A-Za-z .]{1,30}):\s")
    matches = list(speaker_regex.finditer(text))

    if not matches:
        return chunk_recursive_512(parsed, doc_id)

    chunks: List[Chunk] = []
    ordinal = 0
    pos = 0

    for i, m in enumerate(matches):
        start = m.start()
        if start > pos:
            pre_text = text[pos:start]
            if pre_text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_c{ordinal:04d}",
                        doc_id=doc_id,
                        ordinal=ordinal,
                        text=pre_text,
                        char_start=pos,
                        char_end=start,
                        chunker_used="speaker_turn",
                        token_count=_estimate_tokens(pre_text),
                    )
                )
                ordinal += 1

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        turn_text = text[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_c{ordinal:04d}",
                doc_id=doc_id,
                ordinal=ordinal,
                text=turn_text,
                char_start=start,
                char_end=end,
                chunker_used="speaker_turn",
                token_count=_estimate_tokens(turn_text),
            )
        )
        ordinal += 1
        pos = end

    return chunks


CHUNKERS = {
    "recursive_512": chunk_recursive_512,
    "structural": chunk_structural,
    "syntax_aware": chunk_syntax_aware,
    "table_preserving": chunk_table_preserving,
    "speaker_turn": chunk_speaker_turn,
}


def chunk_document(parsed: ParsedDocument, doc_id: str, chunker_name: str = "recursive_512") -> List[Chunk]:
    fn = CHUNKERS.get(chunker_name, chunk_recursive_512)
    chunks = fn(parsed, doc_id)
    # Verification assertion
    for c in chunks:
        actual_span = parsed.text[c.char_start:c.char_end]
        assert actual_span == c.text, f"Span mismatch for chunk {c.chunk_id}: expected {len(c.text)} chars, got {len(actual_span)}"
    return chunks
