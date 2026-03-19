from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    ark_base_url: str | None = None
    ark_api_key: str | None = None
    ark_model: str | None = None
    intent_mode: str | None = None
    enable_hyde: bool | None = None
    hf_offline: bool | None = None
    hf_endpoint: str | None = None
    force_english: bool | None = None


def _default_config_path() -> Path:
    service_dir = Path(__file__).resolve().parents[3]
    return service_dir / "config.local.json"


def load_runtime_config() -> RuntimeConfig:
    cfg = _load_from_file()
    cfg = _apply_env_overrides(cfg)
    return cfg


def _load_from_file() -> RuntimeConfig:
    path_str = os.environ.get("RAG_CONFIG_PATH", "").strip()
    path = Path(path_str) if path_str else _default_config_path()
    if not path.exists():
        return RuntimeConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return RuntimeConfig()
    if not isinstance(raw, dict):
        return RuntimeConfig()

    def get_str(key: str) -> str | None:
        v = raw.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None

    def get_bool(key: str) -> bool | None:
        v = raw.get(key)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes"}:
                return True
            if s in {"0", "false", "no"}:
                return False
        if isinstance(v, int):
            return bool(v)
        return None

    return RuntimeConfig(
        ark_base_url=get_str("ark_base_url"),
        ark_api_key=get_str("ark_api_key"),
        ark_model=get_str("ark_model"),
        intent_mode=get_str("intent_mode"),
        enable_hyde=get_bool("enable_hyde"),
        hf_offline=get_bool("hf_offline"),
        hf_endpoint=get_str("hf_endpoint"),
        force_english=get_bool("force_english"),
    )


def _apply_env_overrides(cfg: RuntimeConfig) -> RuntimeConfig:
    base_url = os.environ.get("ARK_BASE_URL", "").strip() or cfg.ark_base_url
    api_key = os.environ.get("ARK_API_KEY", "").strip() or cfg.ark_api_key
    model = os.environ.get("ARK_MODEL", "").strip() or cfg.ark_model

    intent_mode = os.environ.get("INTENT_MODE", "").strip() or cfg.intent_mode

    enable_hyde_env = os.environ.get("ENABLE_HYDE", "").strip().lower()
    enable_hyde = cfg.enable_hyde
    if enable_hyde_env in {"1", "true", "yes"}:
        enable_hyde = True
    elif enable_hyde_env in {"0", "false", "no"}:
        enable_hyde = False

    hf_endpoint = os.environ.get("HF_ENDPOINT", "").strip() or cfg.hf_endpoint

    hf_offline_env = os.environ.get("HF_HUB_OFFLINE", "").strip().lower()
    hf_offline = cfg.hf_offline
    if hf_offline_env in {"1", "true", "yes"}:
        hf_offline = True
    elif hf_offline_env in {"0", "false", "no"}:
        hf_offline = False

    force_english_env = os.environ.get("FORCE_ENGLISH", "").strip().lower()
    force_english = cfg.force_english
    if force_english_env in {"1", "true", "yes"}:
        force_english = True
    elif force_english_env in {"0", "false", "no"}:
        force_english = False

    return RuntimeConfig(
        ark_base_url=base_url,
        ark_api_key=api_key,
        ark_model=model,
        intent_mode=intent_mode,
        enable_hyde=enable_hyde,
        hf_offline=hf_offline,
        hf_endpoint=hf_endpoint,
        force_english=force_english,
    )
