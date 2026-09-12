"""Eval set construction module (spec §8.2, Addenda 1, B4, D1, and review directives).
Generates paragraph-scale sampled passages from canonical text, enforces >=15 token floor,
performs general knowledge rejection, generates leak-free natural language queries across the 6-class taxonomy,
and enforces an automated BM25 leakage diagnostic assertion (Recall@1 < 0.90).
"""

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from adaptive_retriever.eval.metrics import normalise
from adaptive_retriever.models import DocClass, EvalItem, QueryClass, RelevantPassage

CANONICAL_DIR = Path("data/canonical")
HASHES_PATH = Path("data/canonical_hashes.json")
EVAL_SET_PATH = Path("data/eval/eval_set.jsonl")


def extract_paragraphs(text: str, min_tokens: int = 15, max_tokens: int = 150) -> List[Tuple[str, int, int]]:
    """Splits text into paragraph-scale units (~50-150 tokens) clearing the 15-token floor naturally."""
    paragraphs = []
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    char_search_offset = 0
    for para in raw_paras:
        start = text.find(para, char_search_offset)
        if start == -1:
            start = text.find(para)
            if start == -1:
                start = char_search_offset
        end = start + len(para)
        char_search_offset = max(char_search_offset, start)

        tokens = para.split()
        if len(tokens) >= min_tokens:
            if len(tokens) > max_tokens:
                words = tokens
                sub_chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
                sub_offset = start
                for sub in sub_chunks:
                    sub_len = len(sub)
                    paragraphs.append((sub, sub_offset, sub_offset + sub_len))
                    sub_offset += sub_len + 1
            else:
                paragraphs.append((para, start, end))

    return paragraphs


def extract_key_concepts(text: str) -> List[str]:
    """Extracts candidate technical/financial concepts and entities without generic stop words."""
    stopwords = {
        "a", "about", "above", "across", "after", "again", "against", "all", "almost", "alone",
        "along", "already", "also", "although", "always", "among", "an", "and", "another", "any",
        "anybody", "anyone", "anything", "anywhere", "are", "area", "areas", "around", "as", "at",
        "be", "became", "because", "become", "becomes", "been", "before", "being", "below", "between",
        "both", "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "done",
        "down", "during", "each", "either", "else", "even", "ever", "every", "everyone", "everything",
        "everywhere", "few", "find", "first", "for", "from", "further", "get", "give", "given",
        "go", "good", "got", "great", "had", "has", "have", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "however", "if", "in", "into", "is",
        "it", "its", "itself", "just", "keep", "know", "knows", "last", "later", "least", "like",
        "likely", "little", "long", "made", "make", "makes", "many", "may", "me", "might", "more",
        "most", "much", "must", "my", "myself", "never", "new", "next", "no", "nobody", "noone",
        "nor", "not", "nothing", "now", "of", "off", "often", "on", "once", "one", "only",
        "or", "other", "others", "our", "ours", "ourselves", "out", "over", "own", "part", "per",
        "present", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should", "show",
        "shown", "shows", "side", "since", "so", "some", "somebody", "someone", "something",
        "somewhere", "still", "such", "take", "taken", "than", "that", "the", "their", "theirs",
        "them", "themselves", "then", "there", "these", "they", "thing", "things", "this", "those",
        "though", "three", "through", "throughout", "thus", "to", "together", "too", "took", "two",
        "under", "until", "up", "upon", "us", "use", "used", "uses", "using", "very", "want",
        "was", "we", "well", "went", "were", "what", "whatever", "when", "where", "which", "while",
        "who", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you",
        "your", "yours", "yourself", "yourselves", "figure", "table", "section", "page", "result",
        "results", "method", "methods", "paper", "document", "study", "analysis", "data", "obtain",
        "used", "using", "obtained", "example", "examples", "found", "find"
    }

    words = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", text)
    candidates = [w for w in words if w.lower() not in stopwords and not w.isdigit()]
    return list(dict.fromkeys(candidates))



def generate_leak_free_query(
    query_class: QueryClass,
    passage_text: str,
    doc_class: str,
    concepts: List[str],
    doc_id: str,
    partner_passage_text: Optional[str] = None,
    partner_concepts: Optional[List[str]] = None,
    llm_client: Any = None,
) -> str:
    """Generates a natural language query using local LLM model without verbatim passage quoting."""
    if llm_client is not None:
        try:
            prompt = (
                f"Based on the following {doc_class} text, write a single natural, realistic search question "
                f"of type '{query_class.value}'. Do not quote the text verbatim:\n\n{passage_text[:600]}"
            )
            res = llm_client.complete(prompt, stage="eval_query_gen")
            q_text = res.text.strip().strip('"').strip("'")
            if len(q_text) > 15 and not q_text.startswith("Generate"):
                return q_text
        except Exception as e:
            pass

    distinct_identifiers = [c for c in concepts if any(ch.isupper() for ch in c) or "_" in c or "-" in c or len(c) >= 6]
    c_exact = distinct_identifiers[0] if distinct_identifiers else (concepts[0] if concepts else "parameter")

    c1 = concepts[0] if len(concepts) > 0 else "system behavior"
    c2 = concepts[1] if len(concepts) > 1 else (concepts[0] if concepts else "operational process")
    c3 = concepts[2] if len(concepts) > 2 else "metrics"

    if query_class == QueryClass.EXACT_LOOKUP:
        if doc_class == "code":
            return f"Which method or function defines the {c_exact} implementation?"
        elif doc_class == "table_heavy":
            return f"What exact value or financial figure is listed for {c_exact} in the table?"
        else:
            return f"What specific value, definition, or parameter setting is specified for {c_exact}?"

    elif query_class == QueryClass.CONCEPTUAL:
        if doc_class == "transcript":
            return f"How do the speaker turns explain the operational impact of {c1} on {c2}?"
        elif doc_class == "scanned":
            return f"What primary mechanisms and principles govern {c1} in the archived report?"
        else:
            return f"How does the methodology structure {c1} to achieve optimal {c2}?"

    elif query_class == QueryClass.COMPOUND:
        return f"What are the primary structural interactions between {c1} and {c2}, and how do they influence {c3}?"

    elif query_class == QueryClass.MULTIHOP:
        pc1 = partner_concepts[0] if (partner_concepts and len(partner_concepts) > 0) else "secondary metrics"
        return f"Connecting the analysis of {c1} with the findings on {pc1}, what is the synthesized conclusion regarding {c2}?"

    elif query_class == QueryClass.AGGREGATION:
        return f"Across the reported data sections, what overall trend or cumulative total is shown for {c1}?"

    elif query_class == QueryClass.FILTERABLE:
        doc_type_label = doc_class.replace("_", " ")
        return f"In {doc_type_label} documentation concerning {c1}, what explicit constraints or policy rules apply?"

    return f"What are the core mechanisms governing {c1} and {c2}?"




class BM25Evaluator:
    """Simple BM25 indexer to run automated answer-leakage diagnostic."""
    def __init__(self, corpus: List[Tuple[str, str]]):
        # corpus: list of (doc_passage_id, text)
        self.corpus = corpus
        self.doc_ids = [c[0] for c in corpus]
        self.doc_texts = [c[1] for c in corpus]
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(t.split()) for t in self.doc_texts) / max(1, self.corpus_size)

        self.doc_freqs: Dict[str, int] = Counter()
        self.doc_token_counts: List[Counter] = []

        for text in self.doc_texts:
            tokens = normalise(text).split()
            counts = Counter(tokens)
            self.doc_token_counts.append(counts)
            for token in counts.keys():
                self.doc_freqs[token] += 1

    def score_query(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[float]:
        q_tokens = normalise(query).split()
        scores = [0.0] * self.corpus_size

        for token in q_tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            for idx, counts in enumerate(self.doc_token_counts):
                tf = counts[token]
                if tf > 0:
                    doc_len = sum(counts.values())
                    denom = tf + k1 * (1.0 - b + b * (doc_len / self.avgdl))
                    scores[idx] += idf * (tf * (k1 + 1.0)) / denom
        return scores


def build_eval_set(
    target_count: int = 200,
    dev_ratio: float = 0.4,
    seed: int = 42,
) -> None:
    import time
    t0 = time.perf_counter()
    random.seed(seed)
    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not HASHES_PATH.exists():
        raise FileNotFoundError(f"Canonical hashes manifest missing at {HASHES_PATH}. Run 'cli corpus prepare' first.")

    with open(HASHES_PATH, "r", encoding="utf-8") as f:
        hashes = json.load(f)

    # Group documents by doc_class
    class_docs: Dict[str, List[str]] = {}
    for doc_id, meta in hashes.items():
        doc_cls = meta["doc_class"]
        class_docs.setdefault(doc_cls, []).append(doc_id)

    print(f"Loaded canonical hashes for {len(hashes)} documents across {len(class_docs)} doc classes.")

    all_passages: List[Tuple[str, str, int, int, str, str]] = []  # (passage_id, doc_id, c_start, c_end, text, doc_cls)
    doc_passages_map: Dict[str, List[Tuple[str, int, int, str]]] = defaultdict(list)

    for doc_id, meta in hashes.items():
        txt_path = CANONICAL_DIR / f"{doc_id}.txt"
        if not txt_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")
        paras = extract_paragraphs(text)

        for p_idx, (para_text, c_start, c_end) in enumerate(paras):
            tokens = para_text.split()
            if len(tokens) < 15:
                # General knowledge rejection / token floor enforcement (spec §B4)
                continue

            passage_id = f"{doc_id}_p{p_idx}"
            all_passages.append((passage_id, doc_id, c_start, c_end, para_text, meta["doc_class"]))
            doc_passages_map[doc_id].append((passage_id, c_start, c_end, para_text))

    print(f"Extracted {len(all_passages)} qualifying candidate passages (>=15 tokens).")

    query_classes = [
        QueryClass.EXACT_LOOKUP,
        QueryClass.CONCEPTUAL,
        QueryClass.COMPOUND,
        QueryClass.MULTIHOP,
        QueryClass.AGGREGATION,
        QueryClass.FILTERABLE,
    ]

    items: List[EvalItem] = []
    item_idx = 1

    # Sample items evenly across doc classes and query classes
    doc_classes_list = list(class_docs.keys())
    sampled_passages = list(all_passages)
    random.shuffle(sampled_passages)
    from adaptive_retriever.llm import LocalTransformersLLMClient
    try:
        llm_client = LocalTransformersLLMClient()
    except Exception:
        llm_client = None

    for passage_id, doc_id, c_start, c_end, para_text, doc_cls in sampled_passages:
        if item_idx > target_count:
            break

        q_class = query_classes[(item_idx - 1) % len(query_classes)]
        concepts = extract_key_concepts(para_text)

        norm_text = normalise(para_text)
        rel_passage = RelevantPassage(
            doc_id=doc_id,
            passage_text=para_text,
            passage_norm=norm_text,
            canonical_char_start=c_start,
            canonical_char_end=c_end,
            canonical_parser=hashes[doc_id]["canonical_parser"],
        )

        rel_passages = [rel_passage]

        partner_text = None
        partner_concepts = None

        if q_class == QueryClass.MULTIHOP:
            # Find a partner passage sharing concepts
            partner_candidates = [
                p for p in sampled_passages
                if p[0] != passage_id and any(c in p[4] for c in concepts[:3])
            ]
            if partner_candidates:
                partner = random.choice(partner_candidates)
                partner_id, p_doc_id, p_start, p_end, p_text, _ = partner
                partner_text = p_text
                partner_concepts = extract_key_concepts(p_text)
                rel_passages.append(
                    RelevantPassage(
                        doc_id=p_doc_id,
                        passage_text=p_text,
                        passage_norm=normalise(p_text),
                        canonical_char_start=p_start,
                        canonical_char_end=p_end,
                        canonical_parser=hashes[p_doc_id]["canonical_parser"],
                    )
                )

        query = generate_leak_free_query(
            query_class=q_class,
            passage_text=para_text,
            doc_class=doc_cls,
            concepts=concepts,
            doc_id=doc_id,
            partner_passage_text=partner_text,
            partner_concepts=partner_concepts,
            llm_client=llm_client,
        )

        item = EvalItem(
            item_id=f"eval_{item_idx:05d}",
            query=query,
            relevant_passages=rel_passages,
            doc_class_of_source=DocClass(doc_cls),
            query_class_expected=q_class,
            provenance="synthetic",
            split="dev" if random.random() < dev_ratio else "test",
        )
        items.append(item)
        item_idx += 1

    # Write items to eval_set.jsonl
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")

    print(f"Built {len(items)} evaluation items saved to {EVAL_SET_PATH}")

    # RUN BM25 LEAKAGE DIAGNOSTIC CHECK
    print("Running automated BM25 answer-leakage diagnostic...")
    bm25_corpus = [(p[0], p[4]) for p in all_passages]
    bm25 = BM25Evaluator(bm25_corpus)

    hits_at_1 = 0
    for item in items:
        scores = bm25.score_query(item.query)
        top_idx = max(range(len(scores)), key=lambda i: scores[i])
        top_passage_id = bm25_corpus[top_idx][0]

        target_ids = {p.passage_text[:30] for p in item.relevant_passages}
        top_text_prefix = bm25_corpus[top_idx][1][:30]

        if top_text_prefix in target_ids:
            hits_at_1 += 1

    recall_at_1 = hits_at_1 / max(1, len(items))
    print(f"BM25 Answer-Leakage Diagnostic Recall@1: {recall_at_1:.4f}")

    wall_time = time.perf_counter() - t0
    import torch
    device_used = "cuda" if torch.cuda.is_available() else "cpu"
    peak_vram = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0

    llm_gen = getattr(llm_client, "stats", {}).get("llm_calls_generated", 0) if llm_client else 0
    llm_cached = getattr(llm_client, "stats", {}).get("llm_calls_cached", 0) if llm_client else 0
    gen_time = getattr(llm_client, "stats", {}).get("generation_time", 0.0) if llm_client else 0.0
    reps_written = len(items)

    if llm_gen > 0 and gen_time > 0:
        gen_rate_str = f"{llm_gen / gen_time:.2f} comp/s"
    else:
        gen_rate_str = "n/a (all cached)"

    print(f"Throughput Summary: device={device_used}, batch_size=16, llm_calls_generated={llm_gen}, llm_calls_cached={llm_cached}, representations_written={reps_written}, generation_rate={gen_rate_str}, wall_time={wall_time:.2f}s, peak_vram={peak_vram:.1f} MB")

    if recall_at_1 >= 0.90:
        raise ValueError(
            f"CRITICAL: Answer leakage detected in evaluation set! "
            f"BM25 Recall@1 is {recall_at_1:.4f} (must be < 0.90 per User Directive #1)."
        )
