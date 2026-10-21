"""Field report: list, read, queue and edit the AI field intelligence reports (field_report.py), chat with one
(report_chat.py), and the OpenRouter model catalogue every model picker on the page reads."""

from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import field_report as fr
from .. import jobs, model_catalog, report_chat
from ..config import OPENROUTER_API_KEY
from .data import Router

router = Router()


@router.get("/report-models")
def report_models() -> dict:
    """Presets and defaults; the full list is /openrouter/models."""
    return {"key_present": bool(OPENROUTER_API_KEY),
            "models": [{"key": k, "label": m["label"], "slug": fr.model_slug(k)} for k, m in fr.MODELS.items()],
            "default_report": fr.model_slug("claude"), "default_chat": report_chat.DEFAULT_MODEL}


@router.get("/openrouter/models")
def openrouter_models(refresh: bool = False) -> list[dict]:
    return model_catalog.catalog(force=refresh)


@router.get("/p/{p}/reports")
def reports(p: str) -> list[dict]:
    return fr.list_reports(p)


@router.get("/p/{p}/reports/{rid}")
def report(p: str, rid: str) -> dict:
    r = fr.read_report(p, rid)
    if r is None:
        raise HTTPException(404, f"No report {rid}")
    return r


class NewReport(BaseModel):
    model: str = "claude"
    focus: str | None = Field(None, max_length=2000)


@router.post("/p/{p}/reports")
def new_report(p: str, body: NewReport) -> dict:
    try:
        slug = fr.model_slug(body.model)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if not OPENROUTER_API_KEY:
        raise HTTPException(409, "No OpenRouter key: add OPENROUTER_API_KEY (or OPENAI_API_KEY = sk-or-…) to .env.local")
    argv = ["cli", "field-report", "--run-id", p, "--model", body.model]
    if body.focus and body.focus.strip():
        argv += ["--focus", body.focus.strip()]
    label = fr.MODELS[body.model]["label"] if body.model in fr.MODELS else model_catalog.label(slug)
    con = jobs.connect()
    jid = jobs.enqueue(con, p, [("field-report", f"Field report with {label}", argv)],
                       title=f"Field report · {label}", kind="field-report", lane="quick",
                       params={"model": body.model, "focus": body.focus})
    return jobs.job_dict(con, jid)


class Edit(BaseModel):
    find: str = Field(..., min_length=1, max_length=20000)
    replace: str = Field("", max_length=40000)


@router.post("/p/{p}/reports/{rid}/edit")
def edit(p: str, rid: str, body: Edit) -> dict:
    try:
        return fr.edit_report(p, rid, body.find, body.replace)
    except KeyError:
        raise HTTPException(404, f"No report {rid}") from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from None


# ------------------------------------------------------------------ chat

class ChatTurn(BaseModel):
    model: str = report_chat.DEFAULT_MODEL
    session: str = Field(..., min_length=4, max_length=200)
    messages: list[dict] = Field(..., min_length=1, max_length=200)


@router.post("/p/{p}/reports/{rid}/chat")
def chat(p: str, rid: str, body: ChatTurn) -> StreamingResponse:
    if not model_catalog.SLUG_RE.match(body.model):
        raise HTTPException(422, f"Not an OpenRouter slug: {body.model}")
    if fr.read_report(p, rid) is None:
        raise HTTPException(404, f"No report {rid}")
    return StreamingResponse(report_chat.stream(p, rid, body.messages, body.model, f"{p}:{rid}:{body.session}"),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/p/{p}/reports/{rid}/chats")
def chats(p: str, rid: str) -> list[dict]:
    return report_chat.load_chats(p, rid)


@router.put("/p/{p}/reports/{rid}/chats")
def put_chat(p: str, rid: str, body: dict) -> list[dict]:
    try:
        return report_chat.save_chat(p, rid, body)
    except KeyError:
        raise HTTPException(404, f"No report {rid}") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
