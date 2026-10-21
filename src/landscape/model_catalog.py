"""OpenRouter's model catalogue (any slug a user may pick), cached for an hour, and per-model request settings."""

from __future__ import annotations

import json
import logging
import re
import time

import httpx

from .config import APP_DIR

log = logging.getLogger(__name__)
CATALOG_URL = "https://openrouter.ai/api/v1/models"
CACHE = APP_DIR / "openrouter_models.json"
TTL_S = 3600
SLUG_RE = re.compile(r"^~?[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._:-]*$", re.I)
_MEM: dict = {}


def _per_m(x) -> float | None:
    try:
        return round(float(x) * 1e6, 4)
    except (TypeError, ValueError):
        return None


def _compact(m: dict) -> dict:
    pr = m.get("pricing") or {}
    rs = m.get("reasoning") or {}
    params = m.get("supported_parameters") or []
    return {"id": m["id"], "name": m.get("name") or m["id"], "created": m.get("created"),
            "context": m.get("context_length"), "max_output": (m.get("top_provider") or {}).get("max_completion_tokens"),
            "in": _per_m(pr.get("prompt")), "out": _per_m(pr.get("completion")),
            "cache_read": _per_m(pr.get("input_cache_read")), "cache_write": _per_m(pr.get("input_cache_write")),
            "reasoning": "reasoning" in params, "reasoning_mandatory": bool(rs.get("mandatory")),
            "efforts": rs.get("supported_efforts") or [],
            "intelligence": ((m.get("benchmarks") or {}).get("artificial_analysis") or {}).get("intelligence_index")}


def catalog(force: bool = False) -> list[dict]:
    """Text-out models, newest first. Memory, then a file cache, then OpenRouter; a stale file beats nothing."""
    now = time.time()
    if not force and _MEM.get("at", 0) > now - TTL_S:
        return _MEM["models"]
    if not force and CACHE.exists() and CACHE.stat().st_mtime > now - TTL_S:
        models = json.loads(CACHE.read_text())
    else:
        try:
            raw = httpx.get(CATALOG_URL, timeout=30).json()["data"]
            models = [_compact(m) for m in raw
                      if "text" in ((m.get("architecture") or {}).get("output_modalities") or ["text"])
                      and not m["id"].endswith(":batch")]
            models.sort(key=lambda m: -(m["created"] or 0))
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(models))
        except Exception as e:  # offline: the last copy, however old
            log.warning("OpenRouter catalogue unavailable: %s", e)
            models = json.loads(CACHE.read_text()) if CACHE.exists() else []
    _MEM.update(at=now, models=models)
    return models


def info(slug: str) -> dict | None:
    return next((m for m in catalog() if m["id"] == slug), None)


def label(slug: str) -> str:
    m = info(slug)
    name = m["name"] if m else slug
    return name.split(": ", 1)[-1]  # "Anthropic: Claude Opus 5.5" -> "Claude Opus 5.5"


def reasoning_for(slug: str, want: str) -> dict | None:
    """The reasoning setting closest to `want` that the model accepts (None: leave it to the model)."""
    m = info(slug)
    if m is None or not m["reasoning"]:
        return None
    efforts = m["efforts"]
    if not efforts:
        return {"effort": want}
    order = ["minimal", "low", "medium", "high", "xhigh", "max"]
    if want in efforts:
        return {"effort": want}
    i = order.index(want) if want in order else 1
    return {"effort": min(efforts, key=lambda e: abs(order.index(e) - i) if e in order else 99)}
