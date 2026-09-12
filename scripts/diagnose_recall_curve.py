"""Diagnostic script to compute QueryClass-sliced recall curve (@1, @10, @100) and audit 10 items from @100-miss tail."""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_retriever.eval.metrics import normalise

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANONICAL_DIR = Path("data/canonical")
HASHES_PATH = Path("data/canonical_hashes.json")
EVAL_SET_PATH = Path("data/eval/eval_set.jsonl")


def extract_paragraphs(text: str, min_tokens: int = 15, max_tokens: int = 150):
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


class FastBM25Ranker:
    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.inverted_index = defaultdict(list)

        total_tokens = 0
        for doc_idx, (p_id, text) in enumerate(corpus):
            tokens = normalise(text).split()
            self.doc_lengths.append(len(tokens))
            total_tokens += len(tokens)
            counts = Counter(tokens)
            for token, count in counts.items():
                self.inverted_index[token].append((doc_idx, count))

        self.avgdl = total_tokens / max(1, self.corpus_size)

    def rank(self, query: str, k1: float = 1.5, b: float = 0.75):
        q_tokens = normalise(query).split()
        scores = defaultdict(float)

        for token in set(q_tokens):
            postings = self.inverted_index.get(token, [])
            if not postings:
                continue
            df = len(postings)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + k1 * (1.0 - b + b * (doc_len / self.avgdl))
                scores[doc_idx] += idf * (tf * (k1 + 1.0)) / denom

        if not scores:
            return list(range(min(100, self.corpus_size)))

        ranked_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return ranked_indices


def main():
    with open(HASHES_PATH, "r", encoding="utf-8") as f:
        hashes = json.load(f)

    all_passages = []
    for doc_id in hashes.keys():
        txt_path = CANONICAL_DIR / f"{doc_id}.txt"
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8")
        paras = extract_paragraphs(text)
        for p_idx, (para_text, c_start, c_end) in enumerate(paras):
            if len(para_text.split()) >= 15:
                all_passages.append((f"{doc_id}_p{p_idx}", para_text))

    print(f"Fast BM25 Corpus size: {len(all_passages)} candidate passages.")
    bm25 = FastBM25Ranker(all_passages)

    eval_items = []
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_items.append(json.loads(line))

    # Metric accumulators overall and per query class
    class_totals = Counter()
    class_r1 = Counter()
    class_r10 = Counter()
    class_r100 = Counter()

    overall_r1 = 0
    overall_r10 = 0
    overall_r100 = 0

    miss_100_items = []

    for item in eval_items:
        q_class = item.get("query_class_expected", "unknown")
        query = item["query"]
        target_prefixes = {p["passage_text"][:40] for p in item["relevant_passages"]}

        ranked_indices = bm25.rank(query)

        target_rank = None
        for r, idx in enumerate(ranked_indices, 1):
            cand_prefix = all_passages[idx][1][:40]
            if cand_prefix in target_prefixes:
                target_rank = r
                break

        class_totals[q_class] += 1

        if target_rank == 1:
            overall_r1 += 1
            class_r1[q_class] += 1
        if target_rank is not None and target_rank <= 10:
            overall_r10 += 1
            class_r10[q_class] += 1
        if target_rank is not None and target_rank <= 100:
            overall_r100 += 1
            class_r100[q_class] += 1
        else:
            miss_100_items.append({
                "item_id": item["item_id"],
                "query_class": q_class,
                "doc_class": item.get("doc_class_of_source"),
                "query": query,
                "target_rank": target_rank,
                "target_text": item["relevant_passages"][0]["passage_text"],
                "top1_text": all_passages[ranked_indices[0]][1] if ranked_indices else "N/A",
            })

    total = len(eval_items)
    print("\n================ OVERALL BM25 RECALL CURVE ================")
    print(f"Recall@1:   {overall_r1 / total:.4f} ({overall_r1}/{total})")
    print(f"Recall@10:  {overall_r10 / total:.4f} ({overall_r10}/{total})")
    print(f"Recall@100: {overall_r100 / total:.4f} ({overall_r100}/{total})")
    print("===========================================================\n")

    print("================ RECALL CURVE SLICED BY QUERY_CLASS ================")
    print(f"{'QueryClass':<15}{'Count':>10}{'Recall@1':>15}{'Recall@10':>15}{'Recall@100':>15}")
    print("-" * 70)
    for qc in ["exact_lookup", "conceptual", "compound", "multihop", "aggregation", "filterable"]:
        c_cnt = class_totals[qc]
        r1 = class_r1[qc] / max(1, c_cnt)
        r10 = class_r10[qc] / max(1, c_cnt)
        r100 = class_r100[qc] / max(1, c_cnt)
        print(f"{qc:<15}{c_cnt:>10}{r1:>15.4f}{r10:>15.4f}{r100:>15.4f}")
    print("-" * 70)

    print(f"\n================ 10 TAIL AUDIT ITEMS (@100 MISSES: {len(miss_100_items)} Total) ================")
    for idx, m_item in enumerate(miss_100_items[:10], 1):
        clean_target = m_item['target_text'].replace('\n', ' ')
        clean_top1 = m_item['top1_text'].replace('\n', ' ')
        print(f"\nTail Miss #{idx} [{m_item['item_id']}] (QueryClass: {m_item['query_class']}, DocClass: {m_item['doc_class']}, Target Rank: {m_item['target_rank']})")
        print(f"Query:        {m_item['query']}")
        print(f"Target Text:  {clean_target[:140]}...")
        print(f"BM25 Rank 1:  {clean_top1[:140]}...")


if __name__ == "__main__":
    main()
