"""Test that imports every module under adaptive_retriever using pkgutil.walk_packages and verifies Ruff cleanliness."""

import importlib
import pkgutil
import subprocess
import sys
import adaptive_retriever


def test_all_modules_import():
    package = adaptive_retriever
    prefix = package.__name__ + "."
    for _, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix):
        importlib.import_module(modname)


def test_ruff_f821_e9_clean():
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        "F821,F811,E9",
        "src",
        "tests",
        "scripts",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Ruff check failed:\n{res.stdout}\n{res.stderr}"
