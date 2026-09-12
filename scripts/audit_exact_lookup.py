"""Audit script for exact_lookup, aggregation, and filterable items."""

import json
import sys
from collections import Counter
from pathlib import Path
from adaptive_retriever.eval.metrics import normalise

sys.stdout.reconfigure(encoding='utf-8')

EVAL_SET_PATH = Path("data/eval/eval_set.jsonl")


def main():
    items = []
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    exact_items = [i for i in items if i.get("query_class_expected") == "exact_lookup"]
    agg_items = [i for i in items if i.get("query_class_expected") == "aggregation"]
    filt_items = [i for i in items if i.get("query_class_expected") == "filterable"]

    print(f"Loaded {len(items)} items. exact_lookup count: {len(exact_items)}")
    print("\n--- 5 SAMPLE EXACT_LOOKUP ITEMS ---")
    for idx, item in enumerate(exact_items[:5], 1):
        q = item["query"]
        p_text = item["relevant_passages"][0]["passage_text"]
        norm_q = normalise(q)
        print(f"\nItem #{idx} [{item['item_id']}]")
        print(f"Query:          {q}")
        print(f"Normalized Q:   {norm_q}")
        p_snippet = p_text[:140].replace('\n', ' ')
        print(f"Passage snippet:{p_snippet}...")

    print("\n--- RELEVANT_PASSAGES COUNTS ---")
    agg_counts = Counter([len(i['relevant_passages']) for i in agg_items])
    filt_counts = Counter([len(i['relevant_passages']) for i in filt_items])
    print(f"Aggregation multi-passage counts distribution: {dict(agg_counts)}")
    print(f"Filterable multi-passage counts distribution:  {dict(filt_counts)}")


if __name__ == "__main__":
    main()

