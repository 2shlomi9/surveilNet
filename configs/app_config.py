# configs/app_config.py
# Loads app settings from configs/app_settings.json (no .env).
# Safe defaults if file is missing or keys are absent.

from __future__ import annotations
import json, os
from dataclasses import dataclass

_CFG_PATH = os.path.join(os.path.dirname(__file__), "app_settings.json")

def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

_raw = _load_json(_CFG_PATH)

def _get(name, default):
    return _raw.get(name, default)

def _as_bool(v, default=False):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default

def _as_int(v, default=0):
    try: return int(v)
    except Exception: return default

def _as_float(v, default=0.0):
    try: return float(v)
    except Exception: return default

@dataclass(frozen=True)
class Settings:
    app_host: str
    app_port: int
    app_debug: bool

    process_max_concurrency: int
    upload_chunk_size_bytes: int
    snippet_window_sec: float

def _build_settings() -> Settings:
    host = str(_get("APP_HOST", "127.0.0.1"))
    port = _as_int(_get("APP_PORT", 5000), 5000)
    debug = _as_bool(_get("APP_DEBUG", True), True)

    max_conc = _as_int(_get("PROCESS_MAX_CONCURRENCY", 2), 2)
    chunk_mb = _as_float(_get("UPLOAD_CHUNK_SIZE_MB", 2), 2.0)
    win_sec  = _as_float(_get("SNIPPET_WINDOW_SEC", 6), 6.0)

    # guard rails
    if max_conc < 1: max_conc = 1
    if chunk_mb <= 0: chunk_mb = 1.0
    if win_sec <= 0: win_sec = 6.0

    return Settings(
        app_host=host,
        app_port=port,
        app_debug=debug,
        process_max_concurrency=max_conc,
        upload_chunk_size_bytes=int(chunk_mb * 1024 * 1024),
        snippet_window_sec=float(win_sec),
    )

settings = _build_settings()
