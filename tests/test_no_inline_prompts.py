"""Guards Addendum A2: No string literal > 200 characters allowed in src/ outside prompts.py (excluding docstrings)."""

import ast
from pathlib import Path


def get_non_docstring_literals(file_path: Path) -> list[tuple[int, str]]:
    text = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(file_path))
    except SyntaxError:
        return []

    docstrings = set()

    # Collect raw docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            raw_doc = ast.get_docstring(node, clean=False)
            if raw_doc:
                docstrings.add(raw_doc)
            clean_doc = ast.get_docstring(node, clean=True)
            if clean_doc:
                docstrings.add(clean_doc)

    literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue
            if len(node.value) > 200:
                literals.append((node.lineno, node.value))

    return literals


def test_no_inline_prompts_in_src():
    src_dir = Path(__file__).resolve().parent.parent / "src"
    py_files = list(src_dir.rglob("*.py"))

    violations = []
    for py_file in py_files:
        if py_file.name == "prompts.py":
            continue

        literals = get_non_docstring_literals(py_file)
        for lineno, val in literals:
            violations.append(f"{py_file.name}:{lineno} (length {len(val)}): {val[:40]}...")

    assert not violations, f"Found inline prompt string literals > 200 chars outside prompts.py:\n" + "\n".join(violations)
