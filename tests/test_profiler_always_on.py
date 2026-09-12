"""Guards Addendum A2: profiler is a measurement instrument and must be unconditionally enabled in ALL config files."""

from pathlib import Path
import pytest
import yaml
from adaptive_retriever.config import Config


def test_profiler_enabled_in_all_config_files():
    root_dir = Path(__file__).resolve().parent.parent
    config_dir = root_dir / "config"

    yaml_files = list(config_dir.glob("*.yaml")) + list((config_dir / "ablations").glob("*.yaml"))
    assert len(yaml_files) >= 2, f"Expected at least default.yaml and baseline_fixed.yaml, found {len(yaml_files)}"

    for yaml_file in yaml_files:
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        profiler_enabled = data.get("ingest", {}).get("profiler", {}).get("enabled")
        assert profiler_enabled is True, f"Profiler is not enabled in {yaml_file}"

        # Test loading through Config class validates unconditionally
        cfg = Config(data, path=yaml_file)
        assert cfg.raw["ingest"]["profiler"]["enabled"] is True


def test_profiler_disabled_raises_validation_error():
    invalid_data = {
        "ingest": {"profiler": {"enabled": False}},
        "eval": {"judge": {"enabled": False}},
    }
    with pytest.raises(ValueError, match="Unconditional profiler rule violation"):
        Config(invalid_data)
