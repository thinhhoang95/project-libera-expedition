"""Phase 4: local EmbeddingGemma 2 vectors per facet, a Parquet vector store and FAISS indexes.

Layout: data/shared/vectors/{facet}.parquet (paper_id, vector[768], embedding_model_version) and
data/shared/vectors/{facet}.faiss (HNSW, inner product on unit vectors). One version per file:
if the model version changes, the file is rebuilt from scratch.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import VEC_DIR, EmbeddingParams

log = logging.getLogger(__name__)

TEXT_FACETS = ["problem", "method", "innovation", "result", "limitations", "domain_data",
               "problem_gloss", "method_gloss"]
GLOSS_OF = {"problem": "problem_gloss", "method": "method_gloss"}


class Embedder:
    def __init__(self, p: EmbeddingParams):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.p = p
        self.model = SentenceTransformer(
            p.model, device=dev, model_kwargs={"torch_dtype": dtype},
            config_kwargs={"vision_config": None, "audio_config": None})
        self.model.max_seq_length = 384  # abstracts rarely exceed this; CPU cost is per token
        log.info("embedder %s on %s (%s)", p.model, dev, dtype)

    @property
    def version(self) -> str:
        return self.p.version

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.p.store_dim), dtype=np.float32)
        t0 = time.time()
        v = self.model.encode([self.p.prefix + t for t in texts], batch_size=self.p.batch_size,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=show_progress)
        log.info("encoded %d texts in %.0fs", len(texts), time.time() - t0)
        return v[:, : self.p.store_dim].astype(np.float32)


# ------------------------------------------------------------------ vector store

def _path(facet: str):
    return VEC_DIR / f"{facet}.parquet"


def load_vectors(facet: str, ids: list[str] | None = None, version: str | None = None):
    p = _path(facet)
    if not p.exists():
        return [], np.zeros((0, 768), dtype=np.float32)
    t = pq.read_table(p)
    pid = t.column("paper_id").to_pylist()
    vec = np.asarray(t.column("vector").combine_chunks().flatten().to_numpy(zero_copy_only=False),
                     dtype=np.float32).reshape(len(pid), -1)
    if version is not None:
        vers = t.column("embedding_model_version").to_pylist()
        keep = [i for i, v in enumerate(vers) if v == version]
        pid, vec = [pid[i] for i in keep], vec[keep]
    if ids is not None:
        pos = {k: i for i, k in enumerate(pid)}
        sel = [pos[i] for i in ids if i in pos]
        pid, vec = [pid[i] for i in sel], vec[sel]
    return pid, vec


def stored_ids(facet: str, version: str) -> set[str]:
    p = _path(facet)
    if not p.exists():
        return set()
    t = pq.read_table(p, columns=["paper_id", "embedding_model_version"])
    vers = set(t.column("embedding_model_version").to_pylist())
    if vers - {version}:
        log.warning("%s has vectors from another model version; rebuilding", facet)
        p.unlink()
        return set()
    return set(t.column("paper_id").to_pylist())


def append_vectors(facet: str, ids: list[str], vecs: np.ndarray, version: str) -> None:
    append_vectors_to(_path(facet), ids, vecs, version)


def append_vectors_to(p, ids: list[str], vecs: np.ndarray, version: str) -> None:
    """Upsert vectors into a Parquet file keyed by paper_id (or any string key)."""
    if not ids:
        return
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    dim = vecs.shape[1]
    new = pa.table({
        "paper_id": pa.array(ids),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1), pa.float32()), dim),
        "embedding_model_version": pa.array([version] * len(ids)),
    })
    if p.exists():
        old = pq.read_table(p)
        old = old.filter(pc.invert(pc.is_in(old["paper_id"], value_set=new["paper_id"])))
        new = pa.concat_tables([old, new])
    pq.write_table(new, p)


def build_index(facet: str) -> int:
    import faiss

    ids, vecs = load_vectors(facet)
    if not ids:
        return 0
    index = faiss.IndexHNSWFlat(vecs.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.add(vecs)
    faiss.write_index(index, str(VEC_DIR / f"{facet}.faiss"))
    pd.DataFrame({"paper_id": ids}).to_parquet(VEC_DIR / f"{facet}.faiss.ids.parquet")
    return len(ids)


# ------------------------------------------------------------------ helpers

def truncate(vecs: np.ndarray, dim: int) -> np.ndarray:
    v = vecs[:, :dim]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def blended(facet: str, alpha: float, ids: list[str] | None = None, dim: int = 768,
            version: str | None = None):
    """e_f^(alpha) = normalize((1-a) e_f + a e_gloss). Papers lacking a gloss keep e_f."""
    fid, fv = load_vectors(facet, ids, version)
    if alpha and facet in GLOSS_OF:
        gid, gv = load_vectors(GLOSS_OF[facet], fid, version)
        gpos = {k: i for i, k in enumerate(gid)}
        g = np.array([gv[gpos[k]] if k in gpos else fv[i] for i, k in enumerate(fid)], dtype=np.float32)
        fv = (1 - alpha) * fv + alpha * g
        fv /= np.linalg.norm(fv, axis=1, keepdims=True)
    return fid, truncate(fv, dim) if dim < fv.shape[1] else fv


# ------------------------------------------------------------------ jobs

def embed_abstracts(con, embedder: Embedder, paper_ids: list[str]) -> int:
    have = stored_ids("abstract", embedder.version)
    todo = [i for i in dict.fromkeys(paper_ids) if i not in have]
    if not todo:
        return 0
    con.register("_ids", pd.DataFrame({"paper_id": todo}))
    df = con.execute("""SELECT paper_id, coalesce(title, '') AS title, abstract_raw FROM papers
                        WHERE paper_id IN (SELECT paper_id FROM _ids)""").df()
    con.unregister("_ids")
    texts = [f"{t}. {a}" if isinstance(a, str) and a else t for t, a in zip(df.title, df.abstract_raw)]
    vecs = embedder.encode(texts)
    append_vectors("abstract", df.paper_id.tolist(), vecs, embedder.version)
    return len(df)


def embed_facets(con, run_id: str, embedder: Embedder, model_id: str, prompt_version: str) -> dict:
    """Embed every faithful facet and gloss of the run's corpus that is not yet stored."""
    out = {}
    for facet in TEXT_FACETS:
        have = stored_ids(facet, embedder.version)
        df = con.execute("""
            SELECT f.paper_id, f.text FROM facets f JOIN provenance v USING (paper_id)
            WHERE v.run_id = ? AND f.facet = ? AND f.model_id = ? AND f.prompt_version = ?
              AND f.text IS NOT NULL AND length(trim(f.text)) > 0""",
                         [run_id, facet, model_id, prompt_version]).df()
        df = df[~df.paper_id.isin(have)]
        if not df.empty:
            log.info("embedding %d %s texts", len(df), facet)
            append_vectors(facet, df.paper_id.tolist(), embedder.encode(df.text.tolist()),
                           embedder.version)
        out[facet] = len(df)
    corpus = con.execute("SELECT paper_id FROM provenance WHERE run_id = ?", [run_id]).df().paper_id
    out["abstract"] = embed_abstracts(con, embedder, corpus.tolist())
    for facet in TEXT_FACETS + ["abstract"]:
        build_index(facet)
    return out


_TOK = re.compile(r"[a-z0-9]+")


def _tokens(tag: str) -> frozenset:
    return frozenset(_TOK.findall(tag.lower()))


def normalize_tags(con, run_id: str, embedder: Embedder, model_id: str, prompt_version: str,
                   threshold: float = 0.93) -> dict:
    """Merge near-synonym tags into the most frequent member (greedy, non-transitive).

    Tags are visited by frequency; each joins the most frequent existing canonical with
    cosine > threshold, else becomes a canonical itself. Connected components are not used:
    they chain (A~B~C) and on the pilot merged 'systematic literature review' into
    'distributionally robust optimization'. A tag whose word set is a strict subset or
    superset of the canonical's is never merged: an added modifier makes a specialization
    ('robust' vs 'distributionally robust optimization'), not a synonym, yet scores ~0.96."""
    rows, out = [], {}
    for facet in ("problem_class", "method_family"):
        df = con.execute("""
            SELECT lower(trim(f.text)) AS tag, count(*) AS n FROM facets f
            JOIN provenance v USING (paper_id)
            WHERE v.run_id = ? AND f.facet = ? AND f.model_id = ? AND f.prompt_version = ?
              AND f.text IS NOT NULL GROUP BY 1 ORDER BY n DESC""",
                         [run_id, facet, model_id, prompt_version]).df()
        if df.empty:
            continue
        df = df.sort_values(["n", "tag"], ascending=[False, True]).reset_index(drop=True)
        v = embedder.encode(df.tag.tolist(), show_progress=False)
        toks = [_tokens(t) for t in df.tag]
        canon_idx: list[int] = []  # indices of canonical tags, in frequency order
        for i, tag in enumerate(df.tag):
            best, best_sim = i, 1.0
            if canon_idx:
                sims = v[canon_idx] @ v[i]
                for j in np.argsort(-sims):
                    if sims[j] <= threshold:
                        break
                    c = canon_idx[j]
                    if toks[i] < toks[c] or toks[c] < toks[i]:
                        continue  # specialization, not synonym
                    best, best_sim = c, float(sims[j])
                    break
            if best == i:
                canon_idx.append(i)
            rows.append((run_id, facet, tag, df.tag[best], int(df.n[i]), best_sim))
        df["canonical"] = [r[3] for r in rows[-len(df):]]
        out[facet] = {"distinct": len(df), "canonical": int(df.canonical.nunique())}
    con.execute("DELETE FROM tag_vocab WHERE run_id = ?", [run_id])
    if rows:
        con.executemany("INSERT INTO tag_vocab VALUES (?, ?, ?, ?, ?, ?)", rows)
    return out
