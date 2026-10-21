"""Data access for the web app: the database read replica, project artefacts and cached vectors."""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from functools import wraps
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import embed, external, portfolio, replica
from ..config import RunParams, project_dir


class ReadDB:
    """Read-only connection to the current replica; switches when a newer one is published."""

    def __init__(self):
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._con = None
        self._retired: list[tuple[float, duckdb.DuckDBPyConnection]] = []

    def version(self) -> str:
        p = replica.current()
        return p.name if p else ""

    def cursor(self):
        p = replica.current()
        if p is None:
            raise HTTPException(503, "The database has no published copy yet. Run a pipeline step, or wait for the "
                                     "first copy after start-up.")
        with self._lock:
            if p != self._path:
                con = duckdb.connect(str(p), read_only=True)
                if self._con is not None:
                    self._retired.append((time.time(), self._con))
                self._con, self._path = con, p
                _CACHE.clear()
            # close connections retired more than a minute ago (requests on them have finished)
            keep = []
            for t, c in self._retired:
                if time.time() - t > 60:
                    try:
                        c.close()
                    except Exception:
                        pass
                else:
                    keep.append((t, c))
            self._retired = keep
            return self._con.cursor()

    def q(self, sql: str, params=()) -> pd.DataFrame:
        cur = self.cursor()
        try:
            return cur.execute(sql, list(params)).df()
        finally:
            cur.close()

    def one(self, sql: str, params=()):
        cur = self.cursor()
        try:
            return cur.execute(sql, list(params)).fetchone()
        finally:
            cur.close()


DB = ReadDB()
q = DB.q

# ------------------------------------------------------------------ memo keyed by the replica and file versions

_CACHE: OrderedDict = OrderedDict()
_CACHE_LOCK = threading.Lock()


def memo(maxsize: int = 256):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__module__, fn.__qualname__, DB.version(), args, tuple(sorted(kwargs.items())))
            with _CACHE_LOCK:
                if key in _CACHE:
                    _CACHE.move_to_end(key)
                    return _CACHE[key]
            out = fn(*args, **kwargs)
            with _CACHE_LOCK:
                _CACHE[key] = out
                while len(_CACHE) > maxsize:
                    _CACHE.popitem(last=False)
            return out
        return wrapper
    return deco


_FILES: dict = {}


def read_file(path: Path) -> pd.DataFrame | None:
    """A parquet or CSV artefact, cached until the file changes."""
    if not path.exists():
        return None
    m = path.stat().st_mtime_ns
    hit = _FILES.get(path)
    if hit and hit[0] == m:
        return hit[1]
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    _FILES[path] = (m, df)
    return df


def artefact(project: str, kind: str, name: str) -> pd.DataFrame | None:
    return read_file(project_dir(project, kind, mkdir=False) / name)


def portfolio_file(project: str, name: str) -> pd.DataFrame | None:
    return artefact(project, "portfolio", name)


# ------------------------------------------------------------------ vectors

_VEC: dict = {}
_VEC_LOCK = threading.Lock()


def corpus_vectors(facet: str, version: str, dim: int = 768):
    """(position by paper id, matrix) from the shared vector store."""
    key = ("corpus", facet, version, dim)
    with _VEC_LOCK:
        if key not in _VEC:
            ids, v = embed.load_vectors(facet, version=version)
            _VEC[key] = ({x: k for k, x in enumerate(ids)}, embed.truncate(v, dim) if dim < v.shape[1] else v)
        return _VEC[key]


def ext_vectors(project: str, facet: str):
    key = ("ext", project, facet)
    with _VEC_LOCK:
        if key not in _VEC:
            _VEC[key] = external.vectors(project, facet, portfolio.DIM)
        return _VEC[key]


@memo()
def run_params(project: str) -> RunParams:
    row = DB.one("SELECT params FROM runs WHERE run_id = ?", (project,))
    if not row:
        raise HTTPException(404, f"No run {project} in the database yet")
    return RunParams.from_dict(json.loads(row[0]))


def embedding_version(project: str) -> str:
    return run_params(project).embedding.version


# ------------------------------------------------------------------ JSON

def records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def columns(df: pd.DataFrame) -> dict:
    """Columnar JSON for large tables (map points)."""
    return {c: json.loads(df[c].to_json(orient="values", date_format="iso", default_handler=str)) for c in df.columns}


def jl(x):
    """json.loads that tolerates NULL and already-decoded values."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    return json.loads(x) if isinstance(x, str) else x


def titles(ids) -> dict:
    ids = [i for i in dict.fromkeys(ids) if isinstance(i, str)]
    if not ids:
        return {}
    t = q(f"SELECT paper_id, year, title FROM papers WHERE paper_id IN ({','.join('?' * len(ids))})", ids)
    return {r.paper_id: {"paper_id": r.paper_id, "year": None if pd.isna(r.year) else int(r.year), "title": r.title}
            for r in t.itertuples()}


# ------------------------------------------------------------------ responses

def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if o is pd.NA or o is pd.NaT:
        return None
    return str(o)


class JSON(Response):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return json.dumps(content, default=_default, allow_nan=False, separators=(",", ":")).encode()


def _nan_free(o):
    """allow_nan=False rejects NaN floats inside plain Python values: map them to null."""
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: _nan_free(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_free(v) for v in o]
    return o


class Router(APIRouter):
    """APIRouter whose endpoints may return numpy and pandas values (rendered by `JSON`)."""

    def add_api_route(self, path, endpoint, **kw):

        @wraps(endpoint)
        def wrapped(*a, **k):
            out = endpoint(*a, **k)
            return out if isinstance(out, Response) else JSON(_nan_free(out))

        kw["response_model"] = None
        super().add_api_route(path, wrapped, **kw)
