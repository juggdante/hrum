"""Load YAML configs with ${VAR} / ${VAR:-default} environment expansion."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")

load_dotenv(ROOT / ".env")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@lru_cache
def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml with env expansion."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return _expand(yaml.safe_load(f))


def db_path() -> Path:
    """Absolute path of the DuckDB warehouse file."""
    p = Path(load_config("platform")["db_path"])
    return p if p.is_absolute() else ROOT / p
