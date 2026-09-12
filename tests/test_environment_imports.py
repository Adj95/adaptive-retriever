"""Regression test for Windows Smart App Control DLL blocking on environment packages."""


def test_import_numpy():
    try:
        import numpy
    except Exception as e:
        assert False, f"Failed to import 'numpy': {e}"


def test_import_numpy_random():
    try:
        import numpy.random
    except Exception as e:
        assert False, f"Failed to import 'numpy.random': {e}"


def test_import_numpy_linalg():
    try:
        import numpy.linalg
    except Exception as e:
        assert False, f"Failed to import 'numpy.linalg': {e}"


def test_import_numpy_fft():
    try:
        import numpy.fft
    except Exception as e:
        assert False, f"Failed to import 'numpy.fft': {e}"


def test_import_sklearn():
    try:
        import sklearn
    except Exception as e:
        assert False, f"Failed to import 'sklearn': {e}"


def test_import_scipy():
    try:
        import scipy
    except Exception as e:
        assert False, f"Failed to import 'scipy': {e}"


def test_import_torch():
    try:
        import torch
    except Exception as e:
        assert False, f"Failed to import 'torch': {e}"


def test_import_torch_dynamo_trace_rules():
    try:
        import torch._dynamo.trace_rules
    except Exception as e:
        assert False, f"Failed to import 'torch._dynamo.trace_rules': {e}"


def test_import_transformers():
    try:
        import transformers
    except Exception as e:
        assert False, f"Failed to import 'transformers': {e}"


def test_import_sentence_transformers():
    try:
        import sentence_transformers
    except Exception as e:
        assert False, f"Failed to import 'sentence_transformers': {e}"


def test_import_qdrant_client():
    try:
        import qdrant_client
    except Exception as e:
        assert False, f"Failed to import 'qdrant_client': {e}"
