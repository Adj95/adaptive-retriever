"""Guards Addendum D1 & Build Spec §0.3, §8.5: synthetic_verified provenance may only be assigned by interactive cli eval-verify."""

import ast
import json
from pathlib import Path

import pytest

from adaptive_retriever.cli import eval_verify


def test_eval_verify_raises_on_non_tty_stdin(tmp_path):
    """Calling eval-verify with non-TTY stdin raises rather than marking anything."""
    eval_file = tmp_path / "eval_set.jsonl"
    item = {
        "item_id": "eval_test001",
        "query": "What is the test parameter?",
        "relevant_passages": [
            {
                "doc_id": "doc1",
                "passage_text": "Sample passage text for test parameter.",
                "passage_norm": "sample passage text for test parameter",
                "canonical_char_start": 0,
                "canonical_char_end": 40,
                "canonical_parser": "test",
            }
        ],
        "doc_class_of_source": "code",
        "query_class_expected": "exact_lookup",
        "provenance": "synthetic",
        "split": "dev",
    }
    eval_file.write_text(json.dumps(item) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="eval-verify requires an interactive human session at a TTY terminal"):
        eval_verify(eval_set_path=eval_file)

    content = eval_file.read_text(encoding="utf-8")
    assert "synthetic_verified" not in content
    loaded = json.loads(content.strip())
    assert loaded["provenance"] == "synthetic"


def test_every_synthetic_verified_item_has_verified_at():
    """Every item with provenance == 'synthetic_verified' also has a non-empty verified_at ISO timestamp."""
    eval_file = Path("data/eval/eval_set.jsonl")
    if not eval_file.exists():
        pytest.skip("data/eval/eval_set.jsonl does not exist")

    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("provenance") == "synthetic_verified":
                verified_at = item.get("verified_at")
                assert verified_at and isinstance(verified_at, str) and len(verified_at.strip()) > 0, (
                    f"Item {item.get('item_id')} has provenance 'synthetic_verified' but lacks verified_at ISO timestamp"
                )


def test_synthetic_verified_assignment_appears_in_exactly_one_place():
    """Grep source and assert writing/assigning 'synthetic_verified' string literal appears in exactly one place."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    py_files = list(src_dir.rglob("*.py"))

    assignments = []
    for py_file in py_files:
        text = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value == "synthetic_verified":
                rel_path = py_file.relative_to(src_dir.parent)
                assignments.append(f"{rel_path}:{node.lineno}")

    assert len(assignments) == 1, (
        f"Expected direct assignment of 'synthetic_verified' in exactly 1 place in src/, found {len(assignments)}: {assignments}"
    )
    assert "cli.py" in assignments[0], (
        f"The single function assigning 'synthetic_verified' must be in cli.py, found {assignments[0]}"
    )
