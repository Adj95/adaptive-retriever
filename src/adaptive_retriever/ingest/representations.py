"""Multi-representation generator & tiering module (A4, spec §9.4, §6.2, & Change Order §5).

Per-chunk cost decision tiering:
- High tier: [raw, summary, question, keywords] (information-dense chunks)
- Low tier: [raw] (boilerplate/low-density chunks)

Invariant (§6.2): RepresentationKind.RAW text MUST equal chunk.text exactly.
Contextual Prefixing (§9.4): Prepend section/doc context headers for dense embeddings.
"""

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from adaptive_retriever.models import (
    Chunk,
    GenerationRequest,
    GenerationResult,
    Representation,
    RepresentationKind,
)

logger = logging.getLogger(__name__)


def evaluate_chunk_density(chunk: Chunk, cfg: Any = None) -> str:
    """Assign chunk tier based on information density metrics.
    Cost-decision tiering: only information-dense chunks (uniqueness >= 0.52 and word_count >= 55)
    receive high-tier multi-representation treatment (~25-30% target density).
    """
    vc = {}
    if cfg is not None:
        raw = cfg.raw if hasattr(cfg, "raw") else (cfg if isinstance(cfg, dict) else {})
        vc = raw.get("ingest", {}).get("representations", {}).get("value_classifier", {}) or {}

    min_words_floor = int(vc.get("min_word_floor", 40))
    min_token_count = int(vc.get("min_token_count", 55))
    min_uniqueness = float(vc.get("min_uniqueness_ratio", 0.52))

    words = chunk.text.split()
    word_count = len(words)

    if word_count < min_words_floor:
        return "low"

    unique_words = len(set(w.lower() for w in words if len(w) > 2))
    uniqueness_ratio = unique_words / float(word_count) if word_count > 0 else 0.0

    if uniqueness_ratio >= min_uniqueness and word_count >= min_token_count:
        return "high"
    return "low"


def format_contextual_prefix(chunk: Chunk, doc_title: str = "") -> str:
    """Prepend contextual metadata headers per spec §9.4."""
    prefix_parts = []
    if doc_title:
        prefix_parts.append(f"Document: {doc_title}")
    if chunk.section_path:
        prefix_parts.append(f"Section: {' > '.join(chunk.section_path)}")

    if prefix_parts:
        return f"[{' | '.join(prefix_parts)}]\n{chunk.text}"
    return chunk.text


def _is_rule_fallback(text: str) -> bool:
    if not text:
        return True
    return text.startswith("Question #") or text.startswith("Summary of chunk ") or "regarding:" in text


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = PROJECT_ROOT / "prompts" / "representations"

_PREAMBLE_STRS = (
    r"^(here are some potential keywords related to the given text|here are potential keywords)[\:\s,]*\n*",
    r"^(here are key keywords|here are the keywords|here are key|here is a summary of the provided text)[\:\s,]*\n*",
    r"^(here is a summary|here is|here are|based on the provided text|based on the text)[\:\s,]*\n*",
    r"^(based on|according to the text|according to|the following is|this chunk)[\:\s,]*\n*",
    r"^(this passage|this excerpt|summary of|keywords for|potential keywords)[\:\s,]*\n*",
    r"^(key terms|question[s]?|in summary|in conclusion|to summarize)[\:\s,]*\n*",
)

PREAMBLE_PATTERNS = [re.compile(s, re.IGNORECASE) for s in _PREAMBLE_STRS]

FORBIDDEN_PHRASES = [
    "the provided text", "the passage", "the excerpt", "this chunk",
    "according to the text", "based on the text", "in the text"
]


def load_prompt_template(filename: str) -> Tuple[str, str]:
    path = (PROMPTS_DIR / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Required prompt template file missing at absolute path '{path}'! "
            f"Silent fallbacks are strictly forbidden."
        )
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt template file at '{path}' is empty!")
    lines = content.splitlines()
    ver = "2"
    if lines and lines[0].startswith("# version:"):
        ver = lines[0].split(":")[1].strip()
        content = "\n".join(lines[1:]).strip()
    return content, ver


def post_process_representation(text: str, kind: RepresentationKind, strict_mode: bool = False) -> str:
    cleaned = text.strip()

    changed = True
    while changed and cleaned:
        changed = False
        for pat in PREAMBLE_PATTERNS:
            new_cleaned = pat.sub("", cleaned).strip().lstrip(",:; ")
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                changed = True

    if kind == RepresentationKind.KEYWORDS:
        lines = cleaned.splitlines()
        terms = []
        for line in lines:
            line = re.sub(r"^\s*(\d+[\.\)]|[\*\-\•])\s*", "", line).strip()
            line = re.sub(r"\*\*|__|`", "", line).strip()
            if ":" in line and not line.startswith("http"):
                line = line.split(":", 1)[0].strip()
            for part in line.split(","):
                part_clean = part.strip().strip('"').strip("'")
                if part_clean and len(part_clean) < 60 and not part_clean.lower().startswith("here") and not part_clean.lower().startswith("related to"):
                    terms.append(part_clean)
        cleaned = ", ".join(dict.fromkeys(terms))

    elif kind in (RepresentationKind.QUESTION, RepresentationKind.SUMMARY):
        # Strip forbidden phrases ONLY where they open the output. Removing them
        # mid-sentence corrupts legitimate content: "the passage of the Act"
        # must not become "of the Act".
        lead = re.compile(
            r"^(?:" + "|".join(re.escape(p) for p in FORBIDDEN_PHRASES) + r")[\s,:;]*",
            re.IGNORECASE,
        )
        prev = None
        while prev != cleaned:
            prev = cleaned
            cleaned = lead.sub("", cleaned).strip().lstrip(",:; ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]

    is_invalid = (
        not cleaned
        or any(pat.match(cleaned) for pat in PREAMBLE_PATTERNS)
        or (kind == RepresentationKind.KEYWORDS and ("\n" in cleaned or bool(re.search(r"\d+\.\s", cleaned))))
    )

    if is_invalid:
        raise ValueError(
            f"Post-processing failed for representation '{kind.value}': output is empty or contains forbidden preamble/format! "
            f"Raw text: '{text[:60]}', Cleaned: '{cleaned[:60]}'"
        )

    return cleaned


def plan_representations(
    chunks: List[Chunk],
    cfg: Any = None,
) -> Tuple[List[Representation], List[GenerationRequest]]:
    """Deterministic work only — no model calls.
    Returns (deterministic_representations, requests_needing_generation).
    RAW representations and contextual prefixes are built here.
    Low-tier chunks produce RAW only and zero requests.
    """
    if cfg is None:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        questions_per_chunk = 1
    elif hasattr(cfg, "raw"):
        model_name = cfg.raw.get("models", {}).get("llm", {}).get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        questions_per_chunk = cfg.raw.get("ingest", {}).get("representations", {}).get("questions_per_chunk", 1)
    elif isinstance(cfg, dict):
        model_name = cfg.get("models", {}).get("llm", {}).get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        questions_per_chunk = cfg.get("ingest", {}).get("representations", {}).get("questions_per_chunk", 1)
    else:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        questions_per_chunk = 1

    sum_tpl, sum_ver = load_prompt_template("summary.txt")
    kw_tpl, kw_ver = load_prompt_template("keywords.txt")
    q_tpl, q_ver = load_prompt_template("question_gen.txt")

    deterministic: List[Representation] = []
    requests: List[GenerationRequest] = []

    for chunk in chunks:
        tier = evaluate_chunk_density(chunk, cfg)

        # Invariant (§6.2): RAW.text == chunk.text exactly
        raw_rep = Representation(
            rep_id=f"{chunk.chunk_id}_raw",
            chunk_id=chunk.chunk_id,
            kind=RepresentationKind.RAW,
            text=chunk.text,
            generated_by=None,
            tier=tier,
        )
        deterministic.append(raw_rep)

        if tier == "high":
            # 1. SUMMARY
            try:
                prompt_sum = sum_tpl.format(text=chunk.text[:800])
            except KeyError as err:
                raise KeyError(f"Failed to format 'summary.txt' prompt template! Missing placeholder key: {err}") from err
            except Exception as err:
                raise RuntimeError(f"Failed to format 'summary.txt' prompt template: {err}") from err

            sum_hash = hashlib.sha256(prompt_sum.encode("utf-8")).hexdigest()[:8]
            req_id_sum = hashlib.sha256(f"{chunk.chunk_id}|summary|0|{sum_ver}|{sum_hash}".encode("utf-8")).hexdigest()[:16]
            requests.append(
                GenerationRequest(
                    request_id=req_id_sum,
                    chunk_id=chunk.chunk_id,
                    kind=RepresentationKind.SUMMARY,
                    index=0,
                    prompt=prompt_sum,
                    prompt_version=sum_ver,
                    model=model_name,
                )
            )

            # 2. QUESTIONS
            for q_idx in range(questions_per_chunk):
                try:
                    prompt_q = q_tpl.format(count=questions_per_chunk, text=chunk.text[:800])
                except KeyError as err:
                    raise KeyError(f"Failed to format 'question_gen.txt' prompt template! Missing placeholder key: {err}") from err
                except Exception as err:
                    raise RuntimeError(f"Failed to format 'question_gen.txt' prompt template: {err}") from err

                q_hash = hashlib.sha256(prompt_q.encode("utf-8")).hexdigest()[:8]
                req_id_q = hashlib.sha256(f"{chunk.chunk_id}|question|{q_idx}|{q_ver}|{q_hash}".encode("utf-8")).hexdigest()[:16]
                requests.append(
                    GenerationRequest(
                        request_id=req_id_q,
                        chunk_id=chunk.chunk_id,
                        kind=RepresentationKind.QUESTION,
                        index=q_idx,
                        prompt=prompt_q,
                        prompt_version=q_ver,
                        model=model_name,
                    )
                )

            # 3. KEYWORDS
            try:
                prompt_kw = kw_tpl.format(text=chunk.text[:800])
            except KeyError as err:
                raise KeyError(f"Failed to format 'keywords.txt' prompt template! Missing placeholder key: {err}") from err
            except Exception as err:
                raise RuntimeError(f"Failed to format 'keywords.txt' prompt template: {err}") from err

            kw_hash = hashlib.sha256(prompt_kw.encode("utf-8")).hexdigest()[:8]
            req_id_kw = hashlib.sha256(f"{chunk.chunk_id}|keywords|0|{kw_ver}|{kw_hash}".encode("utf-8")).hexdigest()[:16]
            requests.append(
                GenerationRequest(
                    request_id=req_id_kw,
                    chunk_id=chunk.chunk_id,
                    kind=RepresentationKind.KEYWORDS,
                    index=0,
                    prompt=prompt_kw,
                    prompt_version=kw_ver,
                    model=model_name,
                )
            )

    return deterministic, requests


def materialise_representations(
    chunks: List[Chunk],
    deterministic: List[Representation],
    results: List[GenerationResult],
    cfg: Any = None,
    failures: Optional[List[dict]] = None,
) -> List[Representation]:
    """Join results back onto chunks by request_id. Raise on any missing
    request_id — never substitute a placeholder.
    """
    results_map: dict[str, GenerationResult] = {r.request_id: r for r in results}
    det_map: dict[str, List[Representation]] = {}
    for r in deterministic:
        det_map.setdefault(r.chunk_id, []).append(r)

    all_representations: List[Representation] = []

    strict_m = getattr(cfg, "strict_mode", True) if cfg is not None else False

    # Re-plan to know exact request_ids expected per chunk
    _, expected_requests = plan_representations(chunks, cfg)
    reqs_by_chunk: dict[str, List[GenerationRequest]] = {}
    for req in expected_requests:
        reqs_by_chunk.setdefault(req.chunk_id, []).append(req)

    for chunk in chunks:
        chunk_reps = list(det_map.get(chunk.chunk_id, []))
        chunk_reqs = reqs_by_chunk.get(chunk.chunk_id, [])
        tier = evaluate_chunk_density(chunk, cfg)

        for req in chunk_reqs:
            if req.request_id not in results_map:
                raise RuntimeError(
                    f"Missing generation result for request_id '{req.request_id}' on chunk '{chunk.chunk_id}'. "
                    f"Silent fallbacks are forbidden under Change Order §0.2."
                )

            res = results_map[req.request_id]
            if _is_rule_fallback(res.text):
                raise RuntimeError(
                    f"Rule fallback text detected in result for request_id '{req.request_id}' on chunk '{chunk.chunk_id}'. "
                    f"Silent fallbacks are forbidden under Change Order §0.2."
                )

            try:
                processed_text = post_process_representation(res.text, req.kind, strict_mode=strict_m)
            except ValueError as err:
                # Unusable single output. Record it loudly and skip THIS
                # representation. The run-level rate budget in the pipeline
                # decides whether the run as a whole has failed. This is not a
                # silent fallback: nothing is substituted, and the failure is
                # reported in ingest_stats.json.
                if failures is None:
                    raise
                failures.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "kind": req.kind.value,
                        "reason": str(err)[:200],
                    }
                )
                logger.warning(
                    "Dropping unusable %s representation for chunk %s: %s",
                    req.kind.value,
                    chunk.chunk_id,
                    str(err)[:120],
                )
                continue

            rep_id = (
                f"{chunk.chunk_id}_question_{req.index}"
                if req.kind == RepresentationKind.QUESTION
                else f"{chunk.chunk_id}_{req.kind.value}"
            )

            chunk_reps.append(
                Representation(
                    rep_id=rep_id,
                    chunk_id=chunk.chunk_id,
                    kind=req.kind,
                    text=processed_text,
                    generated_by=res.model,
                    tier=tier,
                )
            )

        # RAW invariance assertion (§6.2)
        raw_reps = [r for r in chunk_reps if r.kind == RepresentationKind.RAW]
        if raw_reps:
            assert (
                raw_reps[0].text == chunk.text
            ), f"RAW representation text mutated! Expected {len(chunk.text)} chars, got {len(raw_reps[0].text)}"

        all_representations.extend(chunk_reps)

    return all_representations


def generate_representations(
    chunks: List[Chunk] | Chunk,
    llm_client: Any = None,
    cfg: Any = None,
    doc_title: str = "",
    questions_per_chunk: int = 1,
    high_tier_kinds: List[RepresentationKind] = None,
    low_tier_kinds: List[RepresentationKind] = None,
    strict_mode: bool = False,
    failures: Optional[List[dict]] = None,
) -> List[Representation]:
    """Inline path, unchanged behaviour: plan -> batched complete -> materialise."""
    if isinstance(chunks, Chunk):
        chunks_list = [chunks]
    else:
        chunks_list = chunks

    if cfg is None:
        cfg = {
            "models": {"llm": {"model": "Qwen/Qwen2.5-0.5B-Instruct"}},
            "ingest": {"representations": {"questions_per_chunk": questions_per_chunk}},
        }

    deterministic, requests = plan_representations(chunks_list, cfg)

    if not requests:
        return deterministic

    if llm_client is None:
        raise RuntimeError(
            f"Strict mode active: LLM client is required for {len(requests)} non-RAW representations on chunks."
        )

        # Per-kind token budgets. A batch runs until its LONGEST sequence finishes,
    # so batching a 25-token keyword list together with a 120-token summary
    # makes the keyword pay the summary's cost. Measured: mixed batches at
    # max_new_tokens=192 ran 53s each for 16 items (0.28 comp/s).
    kind_budgets = {
        RepresentationKind.SUMMARY: 128,
        RepresentationKind.QUESTION: 64,
        RepresentationKind.KEYWORDS: 48,
    }
    if cfg is not None:
        _raw = cfg.raw if hasattr(cfg, "raw") else (cfg if isinstance(cfg, dict) else {})
        _by_kind_cfg = _raw.get("models", {}).get("llm", {}).get("max_new_tokens_by_kind", {}) or {}
        for _k, _v in _by_kind_cfg.items():
            try:
                kind_budgets[RepresentationKind(_k)] = int(_v)
            except ValueError as _err:
                raise ValueError(
                    f"Unknown representation kind '{_k}' in models.llm.max_new_tokens_by_kind"
                ) from _err

    slots_by_kind: dict = {}
    for _i, _req in enumerate(requests):
        slots_by_kind.setdefault(_req.kind, []).append(_i)

    llm_responses: List[Any] = [None] * len(requests)
    for _kind, _slots in slots_by_kind.items():
        _sub_prompts = [requests[i].prompt for i in _slots]
        _sub_resps = llm_client.complete_batch(
            _sub_prompts,
            max_tokens=kind_budgets.get(_kind),
            stage="ingest_representation",
        )
        if len(_sub_resps) != len(_slots):
            raise RuntimeError(
                f"complete_batch returned {len(_sub_resps)} responses for "
                f"{len(_slots)} '{_kind.value}' prompts."
            )
        for _slot, _resp in zip(_slots, _sub_resps):
            llm_responses[_slot] = _resp

    if any(r is None for r in llm_responses):
        raise RuntimeError("Missing LLM response after per-kind batching.")

    results: List[GenerationResult] = []
    for req, resp in zip(requests, llm_responses):
        results.append(
            GenerationResult(
                request_id=req.request_id,
                chunk_id=req.chunk_id,
                kind=req.kind,
                index=req.index,
                text=resp.text,
                model=resp.model,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                generated_at=datetime.now().isoformat(),
            )
        )

    return materialise_representations(
        chunks_list, deterministic, results, cfg, failures=failures
    )
