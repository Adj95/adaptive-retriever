"""Test ingest_config_hash sensitivity and invariance per Change Order §2 & §10."""

import copy
from adaptive_retriever.config import Config


def test_ingest_hash_ignores_device_and_batch_size():
    base_raw = {
        "models": {
            "llm": {
                "provider": "local_transformers",
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "device": "cpu",
                "dtype": "auto",
                "batch_size": 16,
                "max_new_tokens": 96,
                "temperature": 0.0,
            }
        },
        "ingest": {
            "max_chunks_per_document": 150,
            "representations": {
                "questions_per_chunk": 1,
            },
        },
    }

    cfg_base = Config(base_raw)
    hash_base = cfg_base.ingest_config_hash

    # Changing device must NOT change hash
    raw_cuda = copy.deepcopy(base_raw)
    raw_cuda["models"]["llm"]["device"] = "cuda"
    cfg_cuda = Config(raw_cuda)
    assert cfg_cuda.ingest_config_hash == hash_base, "Changing device from cpu to cuda must NOT change ingest_config_hash"

    # Changing batch_size must NOT change hash
    raw_bs = copy.deepcopy(base_raw)
    raw_bs["models"]["llm"]["batch_size"] = 32
    cfg_bs = Config(raw_bs)
    assert cfg_bs.ingest_config_hash == hash_base, "Changing batch_size from 16 to 32 must NOT change ingest_config_hash"

    # Changing dtype must NOT change hash
    raw_dtype = copy.deepcopy(base_raw)
    raw_dtype["models"]["llm"]["dtype"] = "float16"
    cfg_dtype = Config(raw_dtype)
    assert cfg_dtype.ingest_config_hash == hash_base, "Changing dtype from auto to float16 must NOT change ingest_config_hash"

    # Changing questions_per_chunk MUST change hash
    raw_q = copy.deepcopy(base_raw)
    raw_q["ingest"]["representations"]["questions_per_chunk"] = 3
    cfg_q = Config(raw_q)
    assert cfg_q.ingest_config_hash != hash_base, "Changing questions_per_chunk MUST change ingest_config_hash"

    # Changing max_new_tokens MUST change hash
    raw_tokens = copy.deepcopy(base_raw)
    raw_tokens["models"]["llm"]["max_new_tokens"] = 128
    cfg_tokens = Config(raw_tokens)
    assert cfg_tokens.ingest_config_hash != hash_base, "Changing max_new_tokens MUST change ingest_config_hash"

    # Changing model MUST change hash
    raw_model = copy.deepcopy(base_raw)
    raw_model["models"]["llm"]["model"] = "Qwen/Qwen2.5-1.5B-Instruct"
    cfg_model = Config(raw_model)
    assert cfg_model.ingest_config_hash != hash_base, "Changing model MUST change ingest_config_hash"

    # Changing max_chunks_per_document MUST change hash
    raw_cap = copy.deepcopy(base_raw)
    raw_cap["ingest"]["max_chunks_per_document"] = 200
    cfg_cap = Config(raw_cap)
    assert cfg_cap.ingest_config_hash != hash_base, "Changing max_chunks_per_document MUST change ingest_config_hash"
