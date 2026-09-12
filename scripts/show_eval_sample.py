"""Displays a sample batch of eval items for human verification inspection (UTF-8 safe)."""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_SET_PATH = Path("data/eval/eval_set.jsonl")


def main():
    if not EVAL_SET_PATH.exists():
        print("eval_set.jsonl not found.")
        return

    items = []
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(f"Loaded {len(items)} items from eval_set.jsonl.\n")
    print("================ SAMPLE BATCH FOR HUMAN VERIFICATION AUDIT ================")
    for idx, item in enumerate(items[:10], 1):
        clean_p1 = item['relevant_passages'][0]['passage_text'][:150].replace('\n', ' ')
        print(f"\nItem #{idx} [{item['item_id']}]")
        print(f"  DocClass:    {item['doc_class_of_source']}")
        print(f"  QueryClass:  {item['query_class_expected']}")
        print(f"  Query:       {item['query']}")
        print(f"  Passage:     {clean_p1}...")
        if len(item['relevant_passages']) > 1:
            clean_p2 = item['relevant_passages'][1]['passage_text'][:150].replace('\n', ' ')
            print(f"  Passage #2:  {clean_p2}...")
    print("============================================================================\n")


if __name__ == "__main__":
    main()
