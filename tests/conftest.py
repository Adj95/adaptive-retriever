import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import builtins
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (PROJECT_ROOT / "data").resolve()


@pytest.fixture(autouse=True)
def guard_no_data_writes(monkeypatch):
    """Fails any test that attempts to open a path under data/ for writing or appending.
    All tests must use pytest's tmp_path or tempfile.TemporaryDirectory() for temporary data.
    """
    orig_open = builtins.open
    orig_path_open = Path.open
    orig_write_text = Path.write_text
    orig_write_bytes = Path.write_bytes

    def is_under_data(target_path) -> bool:
        try:
            resolved = Path(target_path).resolve()
            return DATA_DIR in resolved.parents or resolved == DATA_DIR
        except Exception:
            return False

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "+", "x")):
            if is_under_data(file):
                pytest.fail(
                    f"Test policy violation: Attempted write open ('{mode}') on path under data/: {file}. "
                    "Tests must use pytest's tmp_path fixture or tempfile for test isolation."
                )
        return orig_open(file, mode, *args, **kwargs)

    def guarded_path_open(self, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "+", "x")):
            if is_under_data(self):
                pytest.fail(
                    f"Test policy violation: Attempted write open ('{mode}') on Path under data/: {self}. "
                    "Tests must use pytest's tmp_path fixture or tempfile for test isolation."
                )
        return orig_path_open(self, mode, *args, **kwargs)

    def guarded_write_text(self, data, encoding=None, errors=None, newline=None):
        if is_under_data(self):
            pytest.fail(
                f"Test policy violation: Attempted write_text on Path under data/: {self}. "
                "Tests must use pytest's tmp_path fixture or tempfile for test isolation."
            )
        return orig_write_text(self, data, encoding=encoding, errors=errors, newline=newline)

    def guarded_write_bytes(self, data):
        if is_under_data(self):
            pytest.fail(
                f"Test policy violation: Attempted write_bytes on Path under data/: {self}. "
                "Tests must use pytest's tmp_path fixture or tempfile for test isolation."
            )
        return orig_write_bytes(self, data)

    monkeypatch.setattr("builtins.open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(Path, "write_text", guarded_write_text)
    monkeypatch.setattr(Path, "write_bytes", guarded_write_bytes)
