"""Chat with a field report: a (usually small) model answers over the report and the evidence pack it was written from.

The prompt is laid out for caching. Its prefix is byte-identical across the turns of a conversation:
instructions, then the evidence pack (which never changes), then the report (which changes only when the reader
applies an edit). Three explicit `cache_control` breakpoints mark the evidence pack, the report and the latest
turn. OpenRouter passes them to Anthropic, Alibaba and Google, and turns them into `prompt_cache_breakpoint`s for
OpenAI. Providers that cache implicitly (DeepSeek, OpenAI, Gemini, Grok) reuse the same stable prefix. A
`session_id` per conversation keeps the turns on one provider, so its cache stays warm.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import AsyncIterator

import httpx

from . import field_report as fr
from . import model_catalog
from .config import OPENROUTER_API_KEY

log = logging.getLogger(__name__)
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
MAX_TURNS = 40  # older turns are dropped (the prefix, and so the cache, is unaffected)

INSTRUCTIONS = """You are the reader's research assistant, sitting next to a field intelligence report in an app. The reader is a researcher; they ask about the report, the field, and what to do. You have two sources, both below: the evidence pack (every product of a research-landscape engine for this field, plus an evidence ledger saying how far each product is backtested) and the report written from it. Nothing else counts as evidence.

How to answer:
- Answer the question first, in as few words as it needs. Short paragraphs or bullets; tables only to compare. No preamble, no restating the question, no closing offers.
- Ground every substantive claim in the pack: give the numbers, and link the papers and categories behind them. If the report says something the pack does not support, or the ledger only rates as descriptive, say so plainly.
- Distinguish what the report says from what the evidence shows when they differ. Use the ledger's words for confidence: backtested, a lead to check, descriptive.
- If the pack cannot answer, say what is missing and which step or view of the app would produce it. Never invent papers, numbers or ids.
- When the reader quotes a passage of the report, answer about that passage.

Links (the app makes them clickable): a paper or outside work [Short title](paper:W1234567890); a community or facet category [Label](cat:problem:0.5:medium:003). Only ids that appear in the pack.

Editing the report: when the reader asks you to change, fix, shorten, extend or rewrite part of the report, reply with a one-line explanation and then one block per change, in exactly this form:

```report-edit
<<<<<<< FIND
the exact current text of the report to replace, copied character for character (keep it short but unique)
=======
the new text
>>>>>>> REPLACE
```

The reader applies or rejects each block. Only propose edits when asked to change the report, keep the report's link format and confidence words, and never put claims in the report that the pack does not support."""


def _blocks(p: str, rid: str) -> list[dict]:
    rec = fr.read_report(p, rid)
    if rec is None:
        raise KeyError(rid)
    ev = fr.report_path(p, rid, ".evidence")
    pack = json.loads(ev.read_text()) if ev and ev.exists() else {}
    pack_txt = json.dumps(pack, ensure_ascii=False, separators=(",", ":"), default=str)  # deterministic
    return [
        {"type": "text", "text": INSTRUCTIONS},
        {"type": "text", "text": f"<evidence_pack>\n{pack_txt}\n</evidence_pack>",
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": f"<report id=\"{rid}\" model=\"{rec.get('model_label')}\" as_of=\"{rec.get('as_of')}\">\n"
                                 f"{rec['markdown']}\n</report>", "cache_control": {"type": "ephemeral"}},
    ]


def _user_text(m: dict) -> str:
    q = (m.get("quote") or "").strip()
    if not q:
        return m["content"]
    quoted = "\n".join("> " + ln for ln in q.splitlines())
    return f"About this passage of the report:\n{quoted}\n\n{m['content']}"


def build_request(p: str, rid: str, messages: list[dict], model: str, session: str) -> dict:
    hist = [m for m in messages if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()][-MAX_TURNS:]
    while hist and hist[0]["role"] != "user":
        hist.pop(0)
    msgs: list[dict] = [{"role": "system", "content": _blocks(p, rid)}]
    for i, m in enumerate(hist):
        text = _user_text(m) if m["role"] == "user" else m["content"]
        if i == len(hist) - 1:  # the conversation so far: the next turn reads it from the cache
            msgs.append({"role": m["role"], "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]})
        else:
            msgs.append({"role": m["role"], "content": text})
    body = {"model": model, "messages": msgs, "stream": True, "max_tokens": 8000, "usage": {"include": True},
            "session_id": session[:256], "prompt_cache_key": session[:64]}
    rs = model_catalog.reasoning_for(model, "low")
    if rs:
        body["reasoning"] = rs
    return body


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


async def stream(p: str, rid: str, messages: list[dict], model: str, session: str) -> AsyncIterator[bytes]:
    """Server-sent events for the app: reasoning and text deltas, then usage (with cache counts), then done."""
    if not OPENROUTER_API_KEY:
        yield _sse({"type": "error", "message": "No OpenRouter key in .env.local"})
        return
    try:
        body = build_request(p, rid, messages, model, session)
    except KeyError:
        yield _sse({"type": "error", "message": f"No report {rid}"})
        return
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "X-Title": "libera-expedition"}
    usage = {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=20.0)) as client:
            async with client.stream("POST", fr.URL, json=body, headers=headers) as r:
                if r.status_code >= 400:
                    raw = (await r.aread()).decode("utf-8", "replace")
                    try:
                        msg = json.loads(raw)["error"]["message"]
                    except (ValueError, KeyError, TypeError):
                        msg = raw[:400]
                    yield _sse({"type": "error", "message": f"OpenRouter {r.status_code}: {msg}"})
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    if "error" in ev:
                        yield _sse({"type": "error", "message": str(ev["error"].get("message", ev["error"]))[:400]})
                        return
                    usage = ev.get("usage") or usage
                    if ev.get("provider"):
                        usage.setdefault("_provider", ev["provider"])
                    for ch in ev.get("choices") or []:
                        d = ch.get("delta") or {}
                        if d.get("reasoning"):
                            yield _sse({"type": "reasoning", "text": d["reasoning"]})
                        if d.get("content"):
                            yield _sse({"type": "delta", "text": d["content"]})
    except httpx.HTTPError as e:
        yield _sse({"type": "error", "message": f"Connection to OpenRouter failed: {e}"})
        return
    det = usage.get("prompt_tokens_details") or {}
    yield _sse({"type": "usage", "model": model, "provider": usage.get("_provider"),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": det.get("cached_tokens") or 0, "cache_write_tokens": det.get("cache_write_tokens") or 0,
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost": usage.get("cost")})
    yield _sse({"type": "done"})


# ------------------------------------------------------------------ conversations, kept beside the report

def load_chats(p: str, rid: str) -> list[dict]:
    f = fr.report_path(p, rid, ".chats")
    if f is None or not f.exists():
        return []
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return []


def save_chat(p: str, rid: str, chat: dict) -> list[dict]:
    f = fr.report_path(p, rid, ".chats")
    if f is None or fr.read_report(p, rid) is None:
        raise KeyError(rid)
    if not re.match(r"^[a-z0-9-]{4,40}$", str(chat.get("id", ""))):
        raise ValueError("bad chat id")
    chats = [c for c in load_chats(p, rid) if c["id"] != chat["id"]]
    if chat.get("messages"):
        chat["updated_at"] = datetime.now().isoformat(timespec="seconds")
        chats.insert(0, chat)
    f.write_text(json.dumps(chats[:50], ensure_ascii=False))
    return chats
