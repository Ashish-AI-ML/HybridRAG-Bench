"""
config.py — Centralized configuration management for HybridRAG Bench.

Loads configs/default.yaml, then deep-merges an optional profile override
(e.g., configs/dev.yaml), then applies environment-variable overrides for
secrets.  All pipeline code imports from this module — nothing is hardcoded.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env into environment (no-op if file missing)
load_dotenv(override=False)

ROOT_DIR = Path(__file__).parent.parent.resolve()
CONFIGS_DIR = ROOT_DIR / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(profile: str | None = None) -> dict[str, Any]:
    """
    Load and return the merged configuration dict.

    Parameters
    ----------
    profile : str | None
        Optional profile name (e.g., ``"dev"``).  Loads
        ``configs/<profile>.yaml`` and deep-merges over the defaults.
        Defaults to the ``APP_ENV`` environment variable (or skips
        merge if neither is set).
    """
    cfg = _load_yaml(CONFIGS_DIR / "default.yaml")

    # Resolve profile from argument or env
    effective_profile = profile or os.getenv("APP_ENV")
    if effective_profile and effective_profile != "production":
        profile_path = CONFIGS_DIR / f"{effective_profile}.yaml"
        if profile_path.exists():
            override = _load_yaml(profile_path)
            cfg = _deep_merge(cfg, override)

    # ── Env-var overrides for secrets ──────────────────────────────────────
    llm_provider = os.getenv("LLM_PROVIDER")
    if llm_provider:
        cfg["generation"]["provider"] = llm_provider

    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        cfg.setdefault("secrets", {})["google_api_key"] = google_key

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        cfg.setdefault("secrets", {})["openai_api_key"] = openai_key

    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        cfg.setdefault("qdrant", {})["url"] = qdrant_url

    qdrant_key = os.getenv("QDRANT_API_KEY")
    if qdrant_key:
        cfg.setdefault("qdrant", {})["api_key"] = qdrant_key

    qdrant_mode = os.getenv("QDRANT_MODE", "local")
    cfg.setdefault("qdrant", {})["mode"] = qdrant_mode

    qdrant_local_path = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_storage")
    cfg.setdefault("qdrant", {})["local_path"] = qdrant_local_path

    eval_judge = os.getenv("EVAL_JUDGE")
    if eval_judge:
        cfg["evaluation"]["judge_model"] = eval_judge

    return cfg


# Module-level singleton — lazy init on first import of config.py
_CONFIG: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Return the cached config singleton."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def reload_config(profile: str | None = None) -> dict[str, Any]:
    """Force reload config (useful in tests)."""
    global _CONFIG
    _CONFIG = load_config(profile)
    return _CONFIG
