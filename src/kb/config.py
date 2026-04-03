"""Configuration loader for the knowledge base."""

from pathlib import Path
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _ROOT / "config" / "kb.yml"


def load_config(config_path: Path | None = None) -> dict:
    """Load and return KB config, merging defaults."""
    path = config_path or _DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against repo root
    for key in ("raw", "wiki", "outputs"):
        cfg["paths"][key] = str(_ROOT / cfg["paths"][key])

    return cfg


def get_root() -> Path:
    return _ROOT
