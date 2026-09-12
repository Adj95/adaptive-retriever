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
from typing import List, Optional
from adaptive_retriever.models import Chunk
from adaptive_retriever.ingest.parsers import ParsedDocument

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def chunk_recursive_512(parsed: ParsedDocument, doc_id: str) -> List[Chunk]:
    text = parsed.text
    if not text:
        return []

    target_tokens = 512
    overlap_tokens = 64
    char_len = len(text)
    words = text.split()

    if not words:
        return []

    # If document is within 512 tokens, return single chunk
    if len(words) <= target_tokens:
        return [
            Chunk(
                chunk_id=f"{doc_id}_c0000",
                doc_id=doc_id,
                ordinal=0,
                text=text,
                char_start=0,
                char_end=char_len,
                chunker_used="recursive_512",
                token_count=len(words),
            )
        ]

    chunks: List[Chunk] = []
    # Sliding window on paragraphs/sentences/line boundaries to preserve exact spans
    lines = text.splitlines(keepends=True)
    curr_start = 0
    curr_end = 0
    curr_words = 0
    ordinal = 0

    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append((pos, pos + len(line), line))
        pos += len(line)

    idx = 0
    while idx < len(line_starts):
        start_char = line_starts[idx][0]
        end_char = start_char
        chunk_lines = []
        words_in_chunk = 0

        j = idx
        while j < len(line_starts):
            l_start, l_end, l_text = line_starts[j]
            l_words = len(l_text.split())
            if words_in_chunk > 0 and (words_in_chunk + l_words) > target_tokens:
                break
            end_char = l_end
            words_in_chunk += l_words
            chunk_lines.append(l_text)
            j += 1

        if j == idx:  # Single line exceeds target_tokens
            l_start, l_end, l_text = line_starts[idx]
            end_char = l_end
            words_in_chunk = len(l_text.split())
            j = idx + 1

        chunk_text = text[start_char:end_char]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_c{ordinal:04d}",
                doc_id=doc_id,
                ordinal=ordinal,
                text=chunk_text,
                char_start=start_char,
                char_end=end_char,
                chunker_used="recursive_512",
                token_count=_estimate_tokens(chunk_text),
            )
        )
        ordinal += 1

        # Advance window with overlap
        if j >= len(line_starts):
            break

        # Step back for overlap
        overlap_w = 0
        back_idx = j - 1
        while back_idx > idx and overlap_w < overlap_tokens:
            overlap_w += len(line_starts[back_idx][2].split())
            back_idx -= 1
        idx = max(idx + 1, back_idx + 1)

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
