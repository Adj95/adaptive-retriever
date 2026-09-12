"""Test that fetch_corpus.py contains no synthetic document generator functions per Change Order §9 & §10."""

from pathlib import Path

FETCH_CORPUS_PATH = Path(__file__).resolve().parent.parent / "scripts" / "fetch_corpus.py"


def test_no_synthetic_corpus_generators():
    assert FETCH_CORPUS_PATH.exists(), f"fetch_corpus.py does not exist at {FETCH_CORPUS_PATH}"

    code_text = FETCH_CORPUS_PATH.read_text(encoding="utf-8")

    # List of synthetic document generator patterns / terms that were deleted per Change Order §9
    removed_patterns = [
        "paper_text +=",
        "tbl_md +=",
        "html_content +=",
        "py_code +=",
        "CHAIRMAN POWELL",
        "pix.clear_with(245)",
    ]

    for pattern in removed_patterns:
        assert pattern not in code_text, (
            f"fetch_corpus.py still contains removed synthetic generator pattern: '{pattern}'. "
            "All synthetic document generators must be deleted per Change Order §9."
        )
