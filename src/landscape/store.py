"""DuckDB tables (see Data model in the build recipe) and the raw-response cache.

Raw JSON lives on disk under data/shared/raw/{source}/{kind}/{key}.json.gz, a local stand-in for
object storage. A cached entry is reused while it is younger than the run's freshness window.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import duckdb
import pandas as pd

from .config import DB_PATH, RAW_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY, seed_ids VARCHAR[], params JSON, cutoff_year INTEGER,
    model_versions JSON, spend_usd JSON, started_at TIMESTAMP, notes VARCHAR
);
CREATE TABLE IF NOT EXISTS papers (
    paper_id VARCHAR PRIMARY KEY, doi VARCHAR, s2_id VARCHAR, arxiv_id VARCHAR, title VARCHAR,
    abstract_raw VARCHAR, abstract_source VARCHAR, year INTEGER, pub_date DATE, venue_id VARCHAR,
    venue_name VARCHAR, type VARCHAR, language VARCHAR, is_retracted BOOLEAN, cited_by_count INTEGER,
    topics JSON, referenced_works VARCHAR[], fetched_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS authors (
    author_id VARCHAR PRIMARY KEY, s2_author_id VARCHAR, orcid VARCHAR, display_name VARCHAR
);
CREATE TABLE IF NOT EXISTS authorships (
    paper_id VARCHAR, author_id VARCHAR, position INTEGER, institution_ids VARCHAR[],
    PRIMARY KEY (paper_id, author_id)
);
CREATE TABLE IF NOT EXISTS institutions (
    institution_id VARCHAR PRIMARY KEY, name VARCHAR, country VARCHAR, type VARCHAR
);
-- "references" is an SQL keyword, so the table is called refs.
CREATE TABLE IF NOT EXISTS refs (
    citing_id VARCHAR, cited_id VARCHAR, source VARCHAR, contexts VARCHAR[], intents VARCHAR[],
    is_influential BOOLEAN, PRIMARY KEY (citing_id, cited_id)
);
CREATE TABLE IF NOT EXISTS provenance (
    run_id VARCHAR, paper_id VARCHAR, route VARCHAR, hop INTEGER, via_id VARCHAR, score DOUBLE,
    PRIMARY KEY (run_id, paper_id)
);
CREATE TABLE IF NOT EXISTS api_calls (
    run_id VARCHAR, source VARCHAR, endpoint VARCHAR, cost_usd DOUBLE, status INTEGER, ts TIMESTAMP
);
CREATE TABLE IF NOT EXISTS facets (
    paper_id VARCHAR, facet VARCHAR, text VARCHAR, model_id VARCHAR, prompt_version VARCHAR,
    created_at TIMESTAMP, PRIMARY KEY (paper_id, facet, model_id, prompt_version)
);
CREATE TABLE IF NOT EXISTS facet_status (
    paper_id VARCHAR, model_id VARCHAR, prompt_version VARCHAR, status VARCHAR, qa_flags JSON,
    error VARCHAR, cost_usd DOUBLE, created_at TIMESTAMP,
    PRIMARY KEY (paper_id, model_id, prompt_version)
);
CREATE TABLE IF NOT EXISTS tag_vocab (
    run_id VARCHAR, facet VARCHAR, tag VARCHAR, canonical VARCHAR, n INTEGER, sim_to_canonical DOUBLE
);
CREATE TABLE IF NOT EXISTS clusters (
    run_id VARCHAR, cluster_id VARCHAR, facet VARCHAR, resolution VARCHAR, alpha DOUBLE,
    label VARCHAR, description VARCHAR, top_terms VARCHAR[], stability DOUBLE, is_stable BOOLEAN,
    size INTEGER, parent_cluster_id VARCHAR, color VARCHAR, time_bucket_years INTEGER,
    label_model VARCHAR
);
CREATE TABLE IF NOT EXISTS paper_cluster (
    run_id VARCHAR, paper_id VARCHAR, facet VARCHAR, resolution VARCHAR, alpha DOUBLE,
    cluster_id VARCHAR, sim_to_centroid DOUBLE
);
-- raw_cluster_id keeps membership of unstable clusters for review (cluster_id is NULL there)
ALTER TABLE paper_cluster ADD COLUMN IF NOT EXISTS raw_cluster_id VARCHAR;
-- kind: stable | hub | unstable | merged | generic; hubs keep their papers and are reported separately
ALTER TABLE clusters ADD COLUMN IF NOT EXISTS kind VARCHAR;
ALTER TABLE clusters ADD COLUMN IF NOT EXISTS coherence DOUBLE;
ALTER TABLE clusters ADD COLUMN IF NOT EXISTS centrality DOUBLE;
-- consolidation: kind 'merged' (papers moved to merged_into); is_generic flags a too-vague category (kind
-- 'generic', papers unassigned, only with ClusterParams.drop_generic); note = the reviewer's reason
ALTER TABLE clusters ADD COLUMN IF NOT EXISTS merged_into VARCHAR;
ALTER TABLE clusters ADD COLUMN IF NOT EXISTS consolidation_note VARCHAR;
ALTER TABLE clusters ADD COLUMN IF NOT EXISTS is_generic BOOLEAN;
-- merged_by: 'embedding' (normalize_tags) or 'llm' (canonicalize_tags_llm); NULL when the tag is canonical
ALTER TABLE tag_vocab ADD COLUMN IF NOT EXISTS merged_by VARCHAR;
CREATE TABLE IF NOT EXISTS year_totals (
    run_id VARCHAR, facet VARCHAR, year INTEGER, n INTEGER
);
CREATE TABLE IF NOT EXISTS cluster_year (
    run_id VARCHAR, facet VARCHAR, resolution VARCHAR, alpha DOUBLE, cluster_id VARCHAR,
    year INTEGER, n INTEGER, n_total INTEGER, share DOUBLE
);
CREATE TABLE IF NOT EXISTS pm_matrix (
    run_id VARCHAR, alpha DOUBLE, problem_cluster VARCHAR, method_cluster VARCHAR, year INTEGER,
    count INTEGER, first_year INTEGER
);
CREATE TABLE IF NOT EXISTS pm_tag_matrix (
    run_id VARCHAR, problem_class VARCHAR, method_family VARCHAR, year INTEGER, count INTEGER,
    first_year INTEGER
);
CREATE TABLE IF NOT EXISTS citation_flows (
    run_id VARCHAR, facet VARCHAR, alpha DOUBLE, year INTEGER, from_cluster VARCHAR,
    to_cluster VARCHAR, count INTEGER, row_share DOUBLE
);
CREATE TABLE IF NOT EXISTS author_year (
    run_id VARCHAR, author_id VARCHAR, year INTEGER, facet VARCHAR, alpha DOUBLE,
    n_papers INTEGER, cluster_dist JSON, dominant_cluster VARCHAR, institution_ids VARCHAR[],
    centroid_ref VARCHAR
);
CREATE TABLE IF NOT EXISTS viz_layout (
    run_id VARCHAR, facet VARCHAR, alpha DOUBLE, paper_id VARCHAR, x DOUBLE, y DOUBLE
);
CREATE TABLE IF NOT EXISTS viz_cluster_layout (
    run_id VARCHAR, cluster_id VARCHAR, x DOUBLE, y DOUBLE, color VARCHAR
);
CREATE TABLE IF NOT EXISTS viz_cluster_year (
    run_id VARCHAR, cluster_id VARCHAR, year INTEGER, n INTEGER, share DOUBLE, status VARCHAR,
    scores JSON
);
CREATE TABLE IF NOT EXISTS signals (
    run_id VARCHAR, signal_type VARCHAR, subject JSON, score DOUBLE, rank INTEGER, features JSON,
    evidence_paper_ids VARCHAR[], judge_score DOUBLE, judge_rationale VARCHAR, cutoff_year INTEGER,
    variant VARCHAR, group_kind VARCHAR
);
CREATE TABLE IF NOT EXISTS paper_scores (
    run_id VARCHAR, paper_id VARCHAR, metric VARCHAR, facet VARCHAR, value DOUBLE
);
CREATE TABLE IF NOT EXISTS viz_flow_year (
    run_id VARCHAR, facet VARCHAR, alpha DOUBLE, kind VARCHAR, year INTEGER, from_cluster VARCHAR,
    to_cluster VARCHAR, count INTEGER
);
CREATE TABLE IF NOT EXISTS viz_author_year (
    run_id VARCHAR, facet VARCHAR, alpha DOUBLE, author_id VARCHAR, year INTEGER, x DOUBLE, y DOUBLE
);
CREATE TABLE IF NOT EXISTS review_actions (
    run_id VARCHAR, reviewer VARCHAR, view VARCHAR, subject JSON, action VARCHAR, note VARCHAR,
    alpha JSON, ts TIMESTAMP
);
"""


def connect(path: Path = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def upsert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    """INSERT OR REPLACE a DataFrame into a table with a primary key (columns matched by name)."""
    if df.empty:
        return
    con.register("_upsert", df)
    cols = ", ".join(df.columns)
    con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM _upsert")
    con.unregister("_upsert")


def replace_run_rows(con, table: str, run_id: str, df: pd.DataFrame, where: str = "") -> None:
    """Delete this run's rows (optionally narrowed by `where`) then append df."""
    con.execute(f"DELETE FROM {table} WHERE run_id = ? {where}", [run_id])
    if df.empty:
        return
    con.register("_rows", df)
    cols = ", ".join(df.columns)
    con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _rows")
    con.unregister("_rows")


# ---------------------------------------------------------------- raw cache

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _raw_path(source: str, kind: str, key: str) -> Path:
    safe = _SAFE.sub("_", key)
    if len(safe) > 120:
        safe = hashlib.sha1(key.encode()).hexdigest()
    return RAW_DIR / source / kind / f"{safe}.json.gz"


def raw_get(source: str, kind: str, key: str, freshness_days: int = 30):
    p = _raw_path(source, kind, key)
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) > freshness_days * 86400:
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def raw_put(source: str, kind: str, key: str, payload) -> None:
    p = _raw_path(source, kind, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(p)


def query_key(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
