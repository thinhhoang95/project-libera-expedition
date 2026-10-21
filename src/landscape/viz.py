"""Phase 6b precompute: fixed 2D layouts and per-year cluster rows for the explorer.

One UMAP fit per (facet, alpha) on the run's 256-d vectors, stored once so the map never moves
between sessions or years. The explorer only reads these tables.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import embed, store
from .config import RunParams

log = logging.getLogger(__name__)
LAYOUT_FACETS = ["abstract", "problem", "method", "innovation", "result", "limitations"]


def build_viz(con, run_id: str, p: RunParams) -> dict:
    import umap

    version = p.embedding.version
    ids = con.execute("SELECT paper_id FROM provenance WHERE run_id = ?", [run_id]).df().paper_id.tolist()
    layouts = []
    for facet in LAYOUT_FACETS:
        alphas = p.clusters.gloss_alphas if facet in embed.GLOSS_OF else [0.0]
        for alpha in alphas:
            pid, v = embed.blended(facet, alpha, ids, dim=p.embedding.graph_dim, version=version)
            if len(pid) < 50:
                continue
            xy = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42,
                           n_jobs=1).fit_transform(v)
            layouts.append(pd.DataFrame({"facet": facet, "alpha": alpha, "paper_id": pid,
                                         "x": xy[:, 0], "y": xy[:, 1]}))
            log.info("layout %s a=%.1f: %d papers", facet, alpha, len(pid))
    lay = pd.concat(layouts, ignore_index=True)
    store.replace_run_rows(con, "viz_layout", run_id, lay.assign(run_id=run_id))

    # cluster position = median of its papers on the layout of the same facet and alpha
    pc = con.execute("""SELECT paper_id, facet, alpha, cluster_id FROM paper_cluster
                        WHERE run_id = ? AND cluster_id IS NOT NULL""", [run_id]).df()
    cl = pc.merge(lay, on=["paper_id", "facet", "alpha"])
    cpos = cl.groupby("cluster_id")[["x", "y"]].median().reset_index()
    colors = con.execute("SELECT cluster_id, color FROM clusters WHERE run_id = ?", [run_id]).df()
    cpos = cpos.merge(colors, on="cluster_id", how="left")
    store.replace_run_rows(con, "viz_cluster_layout", run_id, cpos.assign(run_id=run_id))

    # yearly cluster rows; status and scores are filled by Phase 6 signals
    cy = con.execute("""SELECT cluster_id, year, n, share FROM cluster_year WHERE run_id = ?""",
                     [run_id]).df()
    cy["status"], cy["scores"] = None, None
    store.replace_run_rows(con, "viz_cluster_year", run_id, cy.assign(run_id=run_id))
    return {"layouts": int(lay.groupby(["facet", "alpha"]).ngroups), "points": int(len(lay)),
            "clusters_placed": int(len(cpos))}
