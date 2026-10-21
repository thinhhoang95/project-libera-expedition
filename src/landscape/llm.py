"""Minimal async OpenRouter client: JSON-schema output, provider routing, spend tracking."""

from __future__ import annotations

import asyncio
import json
import logging
import random

import httpx

from .config import OPENROUTER_API_KEY

log = logging.getLogger(__name__)
URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMBudgetExceeded(RuntimeError):
    pass


def quantizations_for(model: str) -> list[str] | None:
    """Precision filter for open-weight models; closed models' endpoints report no quantization."""
    return None if model.startswith(("anthropic/", "openai/", "google/gemini")) else ["fp8", "bf16", "fp16", "fp32"]


class OpenRouter:
    def __init__(self, model: str, quantizations: list[str] | None, concurrency: int,
                 spend_cap_usd: float):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OpenRouter key missing (OPENAI_API_KEY / OPENROUTER_API_KEY)")
        self.model = model
        self.provider = {"sort": "price", "require_parameters": True, "allow_fallbacks": True}
        if quantizations:
            self.provider["quantizations"] = quantizations
        self.sem = asyncio.Semaphore(concurrency)
        self.cap = spend_cap_usd
        self.spent = 0.0
        self.providers_used: dict[str, int] = {}
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=20.0),
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                     "X-Title": "landscape-engine"},
            limits=httpx.Limits(max_connections=concurrency + 4),
        )

    async def close(self):
        await self.http.aclose()

    async def chat_json(self, prompt: str, schema: dict, name: str, max_tokens: int = 1500,
                        system: str | None = None) -> tuple[dict | None, float, str | None]:
        """Returns (parsed_json or None, cost_usd, error)."""
        if self.spent >= self.cap:
            raise LLMBudgetExceeded(f"LLM spend ${self.spent:.3f} reached cap ${self.cap:.2f}")
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body = {
            "model": self.model, "messages": msgs, "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": name, "strict": True, "schema": schema}},
            "provider": self.provider,
            "reasoning": {"enabled": False},
            "usage": {"include": True},
        }
        if self.model.startswith("anthropic/"):
            # Claude 5.x endpoints take no temperature and cannot disable reasoning
            body.pop("temperature")
            body["reasoning"] = {"effort": "low"}
        cost_total, last_err = 0.0, None
        for parse_try in range(2):  # retry malformed JSON once
            data = None
            for attempt in range(6):
                try:
                    async with self.sem:
                        r = await self.http.post(URL, json=body)
                    if r.status_code in (429, 500, 502, 503, 504, 408):
                        raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                    r.raise_for_status()
                    data = r.json()
                    if "error" in data:
                        raise RuntimeError(str(data["error"])[:300])
                    break
                except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as e:
                    last_err = f"{type(e).__name__}: {str(e)[:200]}"
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (400, 401, 402, 403):
                        return None, cost_total, f"http {e.response.status_code}: {e.response.text[:300]}"
                    await asyncio.sleep(min(30, 1.5 * 2**attempt) * (0.5 + random.random()))
            if data is None:
                return None, cost_total, last_err
            usage = data.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            cost_total += cost
            self.spent += cost
            prov = data.get("provider") or "?"
            self.providers_used[prov] = self.providers_used.get(prov, 0) + 1
            try:
                content = data["choices"][0]["message"]["content"]
                return json.loads(content), cost_total, None
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
                last_err = f"malformed JSON: {e}"
        return None, cost_total, last_err
