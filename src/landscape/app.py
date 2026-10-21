"""Phase 6b explorer prototype (Streamlit + Plotly), reading DuckDB directly.

    uv run landscape explorer        # http://localhost:8501

Views: Map & grid (one map, one grid, one time axis), Corpus view (Phase 2), Facet review
(Phase 3, gold set), Cluster review (Phase 5). Every judgment is written to `review_actions`.
DuckDB allows one process per file: close the explorer before running pipeline steps.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from landscape import store

ACCENT = "#e8590c"  # reserved for what a reviewer should look at: selection, playhead, new cells
MUTED = "#c9cdd2"
ALPHAS = [0.0, 0.3, 0.5, 0.8]
GLOSS_FACETS = {"problem", "method"}
RESOLUTIONS = ["coarse", "medium", "fine"]

st.set_page_config(page_title="Landscape explorer", layout="wide")


# ------------------------------------------------------------------ data access

@st.cache_resource
def _con():
    return store.connect()


@st.cache_data(ttl=600, show_spinner=False)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return _con().cursor().execute(sql, list(params)).df()


def record(view: str, subject: dict, action: str, note: str = "", alpha=None) -> None:
    reviewer = st.session_state.get("reviewer") or "anonymous"
    _con().cursor().execute(
        "INSERT INTO review_actions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [st.session_state.run_id, reviewer, view, json.dumps(subject), action, note,
         json.dumps(alpha) if alpha is not None else None, datetime.now()])


def my_actions(view: str) -> pd.DataFrame:
    return _con().cursor().execute(
        """SELECT * FROM review_actions WHERE run_id = ? AND view = ? AND reviewer = ?
           ORDER BY ts""", [st.session_state.run_id, view, st.session_state.get("reviewer") or "anonymous"]).df()


def paper_card(pid: str) -> None:
    run = st.session_state.run_id
    p = q("""SELECT p.*, v.route, v.hop, v.via_id FROM papers p
             LEFT JOIN provenance v ON v.paper_id = p.paper_id AND v.run_id = ?
             WHERE p.paper_id = ?""", (run, pid))
    if p.empty:
        st.info(f"{pid} not found")
        return
    p = p.iloc[0]
    authors = q("""SELECT a.display_name FROM authorships s JOIN authors a USING (author_id)
                   WHERE s.paper_id = ? ORDER BY s.position""", (pid,)).display_name.tolist()
    link = f"https://doi.org/{p.doi}" if isinstance(p.doi, str) else f"https://openalex.org/{pid}"
    st.markdown(f"**[{p.title}]({link})**  \n{int(p.year) if pd.notna(p.year) else '?'} · "
                f"{p.venue_name or 'unknown venue'} · route `{p.route}` hop {p.hop}")
    st.caption(", ".join(authors[:8]) + (" …" if len(authors) > 8 else ""))
    f = q("""SELECT facet, text FROM facets WHERE paper_id = ?""", (pid,))
    if not f.empty:
        f = f.drop_duplicates("facet").set_index("facet").text
        order = ["problem", "problem_class", "problem_gloss", "method", "method_family", "method_gloss",
                 "innovation", "result", "limitations", "domain_data", "contribution_type"]
        st.dataframe(pd.DataFrame({"facet": [k for k in order if k in f.index],
                                   "text": [f[k] for k in order if k in f.index]}),
                     hide_index=True, width="stretch")
    else:
        st.caption("title only: no facets")
    with st.expander(f"Abstract ({p.abstract_source or 'none'})"):
        st.write(p.abstract_raw or "—")


# ------------------------------------------------------------------ sidebar

# Phase 8 cutoff runs (<run>@T<year>) are internal to the backtest and have no layouts
runs = q("SELECT run_id FROM runs WHERE run_id NOT LIKE '%@T%' ORDER BY started_at DESC").run_id.tolist()
if not runs:
    st.error("No runs in the database yet.")
    st.stop()
with st.sidebar:
    st.session_state.run_id = st.selectbox("Run", runs)
    views = ["Map & grid", "Research lab", "Experiments", "Corpus view", "Facet review", "Cluster review"]
    requested_view = st.query_params.get("view", "Map & grid")
    view = st.radio("View", views, index=views.index(requested_view) if requested_view in views else 0)
    st.session_state.reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer", ""),
                                              placeholder="your name (stored with every judgment)")
RUN = st.session_state.run_id


def clustering_controls(key: str, facets=("problem", "method", "innovation", "result", "limitations")):
    facet = st.selectbox("Facet", list(facets), key=f"{key}_facet")
    alpha = st.select_slider("Gloss weight α", ALPHAS, value=0.3, key=f"{key}_alpha",
                             disabled=facet not in GLOSS_FACETS,
                             help="0 = the abstract's wording; higher groups by concept. "
                                  "Precomputed at 0, 0.3, 0.5, 0.8.")
    alpha = alpha if facet in GLOSS_FACETS else 0.0
    res = st.select_slider("Resolution", RESOLUTIONS, value="medium", key=f"{key}_res")
    return facet, alpha, res


# ================================================================== Map & grid

BADGE = {"emerging": "▲", "niche": "◆", "dormant": "○"}
FLOW_UP, FLOW_DOWN = "#2f9e44", "#c92a2a"  # citation-flow change vs previous window
BRIDGE_TYPES = {"bridge_lim_inn": "Limitation → innovation", "bridge_method_transfer": "Method transfer",
                "bridge_unresolved_limitation": "Unresolved limitation",
                "bridge_semantic_not_cited": "Semantic but not cited"}


def _arc(x0, y0, x1, y1, bend=0.15, n=20):
    """Quadratic Bezier between two points, bent to the left of the direction of travel."""
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - bend * dy, my + bend * dx
    t = np.linspace(0, 1, n)
    return ((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1,
            (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1)


def signals_df(types: tuple) -> pd.DataFrame:
    return q(f"""SELECT * FROM signals WHERE run_id = ? AND signal_type IN ({",".join("?" * len(types))})
                 ORDER BY signal_type, rank""", (RUN, *types))


def view_map():
    years = q("""SELECT year, n FROM year_totals WHERE run_id = ? AND facet = 'corpus'
                 AND year IS NOT NULL ORDER BY year""", (RUN,))
    if years.empty:
        st.warning("Run `landscape core` first.")
        return
    y0 = int(max(years.year.min(), years.loc[years.n >= 20, "year"].min()))
    y1 = int(years.year.max())
    have_signals = not q("SELECT 1 FROM signals WHERE run_id = ? LIMIT 1", (RUN,)).empty

    with st.sidebar:
        st.divider()
        facet, alpha, res = clustering_controls("map")
        st.markdown("**Layers**")
        show_papers = st.checkbox("Papers", True)
        show_clusters = st.checkbox("Clusters", True)
        show_density = st.checkbox("Activity shift", False)
        compare = st.checkbox("Compare with previous window", False, disabled=not show_density)
        nosig = dict(disabled=not have_signals, help=None if have_signals else "run `landscape signals`")
        # off by default: direction labels depend on the sampling mix (08-backtest, 08c: 6 of 15
        # topics reverse direction once route-mix changes are separated out)
        show_badges = st.checkbox("Status badges", False, **{**nosig, "help": nosig["help"] or
                                  "Direction labels (emerging / niche / dormant). Weak in the backtest and "
                                  "sensitive to how the corpus was sampled: read as prompts, not trends."})
        show_flows = st.checkbox("Citation flows", False, **nosig)
        show_mig = st.checkbox("Author migration", False, **nosig)
        show_novel = st.checkbox("Novel papers", False, **nosig)
        show_pockets = st.checkbox("Rising pockets", False, **{**nosig, "help": nosig["help"] or
                                   "Recent papers whose neighbourhood is filling up fastest (top 5% local momentum)"})
        show_bridges = st.checkbox("Bridges", False, **nosig)
        show_people = st.checkbox("People", False, **nosig)
        active = sum([show_density, show_flows, show_mig, show_novel, show_pockets, show_bridges, show_people])
        if active > 3:
            st.caption("Tip: more than three active layers get hard to read.")
        top_n = st.slider("Arcs / arrows shown", 5, 60, 30, disabled=not (show_flows or show_mig or show_bridges))

    if "window" not in st.session_state:
        st.session_state.window = (y0, y1)
    layout = q("""SELECT l.paper_id, l.x, l.y, p.year, p.title, c.cluster_id FROM viz_layout l
                  JOIN papers p USING (paper_id)
                  LEFT JOIN paper_cluster c ON c.run_id = l.run_id AND c.paper_id = l.paper_id
                       AND c.facet = l.facet AND c.alpha = l.alpha AND c.resolution = ?
                  WHERE l.run_id = ? AND l.facet = ? AND l.alpha = ?""", (res, RUN, facet, alpha))
    if layout.empty:
        st.warning("No layout for this facet. Run `landscape viz`.")
        return
    clusters = q("""SELECT c.cluster_id, c.label, c.description, c.size, c.stability, c.kind, c.centrality,
                           c.color, c.top_terms, c.parent_cluster_id, l.x, l.y
                    FROM clusters c LEFT JOIN viz_cluster_layout l USING (run_id, cluster_id)
                    WHERE c.run_id = ? AND c.facet = ? AND c.alpha = ? AND c.resolution = ?
                      AND c.kind IN ('stable', 'hub')""", (RUN, facet, alpha, res))
    cy = q("""SELECT cluster_id, year, n, share FROM cluster_year WHERE run_id = ? AND facet = ?
              AND alpha = ? AND resolution = ?""", (RUN, facet, alpha, res))
    sel = st.session_state.get("sel_cluster")
    if sel not in set(clusters.cluster_id):
        sel = None

    left, right = st.columns([3, 2], gap="medium")
    map_slot = left.container()
    axis_slot = left.container()

    # ---- time axis (defined first so every layer reads the same window)
    with axis_slot:
        c1, c2 = st.columns([6, 1])
        t1, t2 = c1.slider("Window", y0, y1, st.session_state.window, key="window_slider")
        st.session_state.window = (t1, t2)
        if c2.button("▶ Play" if not st.session_state.get("playing") else "■ Stop"):
            st.session_state.playing = not st.session_state.get("playing", False)
            if st.session_state.playing:
                st.session_state.window = (y0, y0)
                st.session_state.window_slider = (y0, y0)
            st.rerun()
        yr = years[(years.year >= y0)]
        fig = go.Figure(go.Bar(x=yr.year, y=yr.n, marker_color=[
            "#7d8590" if t1 <= y <= t2 else MUTED for y in yr.year], name="papers", hovertemplate="%{x}: %{y} papers"))
        if sel:
            s = cy[(cy.cluster_id == sel) & (cy.year >= y0)].sort_values("year")
            fig.add_trace(go.Scatter(x=s.year, y=s.share, yaxis="y2", mode="lines+markers",
                                     line=dict(color=ACCENT), name="selected cluster share",
                                     hovertemplate="%{x}: %{y:.1%}"))
        fig.add_vrect(x0=t1 - 0.5, x1=t2 + 0.5, line=dict(color=ACCENT, width=1.5), fillcolor="rgba(0,0,0,0)")
        fig.update_layout(height=150, margin=dict(l=10, r=10, t=5, b=5), showlegend=False,
                          yaxis=dict(title=None, showgrid=False),
                          yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False),
                          xaxis=dict(dtick=1), bargap=0.15)
        st.plotly_chart(fig, width="stretch", key="timeaxis")

    win = layout[(layout.year >= t1) & (layout.year <= t2)]
    before = layout[layout.year < t1]
    n_win = cy[(cy.year >= t1) & (cy.year <= t2)].groupby("cluster_id").n.sum()
    tot_win = len(win)
    prev_len = t2 - t1 + 1
    n_prev = cy[(cy.year >= t1 - prev_len) & (cy.year < t1)].groupby("cluster_id").n.sum()
    tot_prev = len(layout[(layout.year >= t1 - prev_len) & (layout.year < t1)])
    colors = dict(zip(clusters.cluster_id, clusters.color))
    labels = dict(zip(clusters.cluster_id, clusters.label))
    kinds = dict(zip(clusters.cluster_id, clusters.kind))
    pos = clusters.dropna(subset=["x"]).set_index("cluster_id")[["x", "y"]]

    # as-of status and scores at the playhead (end of window)
    vcy = q("""SELECT cluster_id, status, scores FROM viz_cluster_year WHERE run_id = ? AND year = ?
               AND scores IS NOT NULL""", (RUN, t2))
    status = dict(zip(vcy.cluster_id, vcy.status))
    scores = {c: json.loads(s) for c, s in zip(vcy.cluster_id, vcy.scores)}

    def disp(c):
        b = BADGE.get(status.get(c), "") if show_badges else ""
        h = " ◇hub" if kinds.get(c) == "hub" else ""
        return f"{b} {labels.get(c, c)}{h}".strip()

    # ---- map
    with map_slot:
        fig = go.Figure()
        if show_density and len(win) > 20:
            xr = (layout.x.min(), layout.x.max())
            yr_ = (layout.y.min(), layout.y.max())
            bins = 40
            h_now, xe, ye = np.histogram2d(win.x, win.y, bins=bins, range=[xr, yr_], density=True)
            if compare and tot_prev > 20:
                prev = layout[(layout.year >= t1 - prev_len) & (layout.year < t1)]
                h_prev, _, _ = np.histogram2d(prev.x, prev.y, bins=bins, range=[xr, yr_], density=True)
                z, cs, mid = (h_now - h_prev).T, "RdBu_r", 0
            else:
                z, cs, mid = h_now.T, "Greys", None
            fig.add_trace(go.Contour(z=z, x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2,
                                     colorscale=cs, zmid=mid, opacity=0.45, showscale=compare,
                                     contours=dict(coloring="heatmap"), line=dict(width=0),
                                     hoverinfo="skip", name="density"))
        if show_papers:
            fig.add_trace(go.Scattergl(x=before.x, y=before.y, mode="markers", hoverinfo="skip",
                                       marker=dict(size=3, color=MUTED, opacity=0.35), name="earlier"))
            dim = sel is not None
            col = [colors.get(c, "#9aa0a6") if (not dim or c == sel) else MUTED for c in win.cluster_id]
            fig.add_trace(go.Scattergl(
                x=win.x, y=win.y, mode="markers", customdata=win.paper_id,
                text=[f"{t[:90]}<br>{int(y)} · {labels.get(c, 'unclustered')}" for t, y, c in
                      zip(win.title.fillna(""), win.year, win.cluster_id)],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(size=5, color=col, opacity=0.85, line=dict(width=0)), name="papers"))

        if show_novel:
            nf = facet if facet in ("problem", "method", "innovation") else "abstract"
            nv = q("""SELECT paper_id, value FROM paper_scores WHERE run_id = ? AND metric = 'novelty'
                      AND facet = ?""", (RUN, nf))
            w = win.merge(nv, on="paper_id")
            if len(w):
                w = w[w.value >= w.value.quantile(0.95)]
                fig.add_trace(go.Scatter(
                    x=w.x, y=w.y, mode="markers", customdata=w.paper_id,
                    marker=dict(size=11, symbol="circle-open", color="#1f2328", line=dict(width=1.5)),
                    text=[f"novel ({nf}): {v:.3f} (descriptive; novelty does not predict 3-year citations)"
                          f"<br>{t[:80]}" for v, t in zip(w.value, w.title.fillna(""))],
                    hovertemplate="%{text}<extra></extra>", name="novel"))

        if show_pockets:
            lm = q("""SELECT paper_id, value FROM paper_scores WHERE run_id = ? AND metric = 'local_momentum'""", (RUN,))
            if len(lm):
                lm = lm[lm.value >= lm.value.quantile(0.95)]
            w = win.merge(lm, on="paper_id")
            if len(w):
                fig.add_trace(go.Scatter(
                    x=w.x, y=w.y, mode="markers", customdata=w.paper_id,
                    marker=dict(size=13, symbol="circle-open", color=FLOW_UP, line=dict(width=2)),
                    text=[f"rising pocket: local momentum {v:.2f}<br>{t[:80]}" for v, t in zip(w.value, w.title.fillna(""))],
                    hovertemplate="%{text}<extra></extra>", name="pockets"))
                st.caption("Rising pockets (green rings): recent papers whose closest earlier neighbours are "
                           "filling up with this year's papers faster than last year's (top 5%). Read them as "
                           "\"this small neighbourhood is getting busier\", a place to look rather than a "
                           "forecast: in the backtest they are about twice as good as random. A paper "
                           "recorded twice in OpenAlex can make a false pocket.")
            else:
                st.caption("Rising pockets mark papers from the last two years: move the window to the end.")

        def window_flows(kind):
            f = q("""SELECT year, from_cluster, to_cluster, count FROM viz_flow_year WHERE run_id = ?
                     AND kind = ? AND facet = ? AND alpha = ?""", (RUN, kind, facet, alpha))
            f = f[f.from_cluster != f.to_cluster]
            now = f[(f.year >= t1) & (f.year <= t2)].groupby(["from_cluster", "to_cluster"])["count"].sum()
            prev = f[(f.year >= t1 - prev_len) & (f.year < t1)].groupby(["from_cluster", "to_cluster"])["count"].sum()
            now = now[[a in pos.index and b in pos.index for a, b in now.index]]
            return now.sort_values(ascending=False).head(top_n), prev

        if (show_flows or show_mig) and res != "medium":
            st.caption("Citation flows and migration are computed for medium clusters.")
        if show_flows and res == "medium":
            now, prev = window_flows("citation")
            mx = now.max() if len(now) else 1
            for (a, b), n in now.items():
                xs, ys = _arc(pos.x[a], pos.y[a], pos.x[b], pos.y[b])
                d = n / max(prev_len, 1) - prev.get((a, b), 0) / max(prev_len, 1)
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", hoverinfo="text", opacity=0.55,
                    line=dict(width=1 + 7 * n / mx, color=FLOW_UP if d >= 0 else FLOW_DOWN),
                    hovertext=f"{labels[a]} → {labels[b]}<br>{n} citations in {t1}–{t2} "
                              f"({'+' if d >= 0 else ''}{d:.1f}/yr vs previous window)", name="flow"))
        if show_mig and res == "medium":
            now, _ = window_flows("migration")
            mx = now.max() if len(now) else 1
            for (a, b), n in now.items():
                fig.add_annotation(x=pos.x[b], y=pos.y[b], ax=pos.x[a], ay=pos.y[a], xref="x", yref="y",
                                   axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1,
                                   arrowwidth=1 + 4 * n / mx, arrowcolor="#495057", opacity=0.7,
                                   hovertext=f"{n} authors moved {labels[a]} → {labels[b]}", text="")

        if show_bridges:
            on_bridge_map = facet == "problem" and alpha == 0.5 and res == "medium"
            if not on_bridge_map:
                st.caption("Bridges link problem clusters at α = 0.5, medium resolution: switch the map "
                           "there to draw them. The Bridges tab lists them from any view.")
            else:
                b = signals_df(("bridge_lim_inn", "bridge_semantic_not_cited"))
                for r in b[b["rank"] <= max(5, top_n // 2)].itertuples():
                    s = json.loads(r.subject)
                    a_, b_ = s.get("from_cluster", s.get("cluster_a")), s.get("to_cluster", s.get("cluster_b"))
                    if a_ not in pos.index or b_ not in pos.index:
                        continue
                    xs, ys = _arc(pos.x[a_], pos.y[a_], pos.x[b_], pos.y[b_], bend=-0.2)
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines", line=dict(color=ACCENT, width=2, dash="dash"),
                        hoverinfo="text", hovertext=f"{BRIDGE_TYPES[r.signal_type]} #{r.rank}<br>"
                                                    f"{labels.get(a_)} → {labels.get(b_)}<br>judge: {r.judge_score}",
                        name="bridge"))

        if show_people:
            ppl = signals_df(("seeder_author", "broker_author"))
            opts = {}
            for r in ppl.itertuples():
                s = json.loads(r.subject)
                opts.setdefault(s["author_id"], f"{s.get('name')} ({r.signal_type.split('_')[0]} #{r.rank})")
            with st.sidebar:
                who = st.selectbox("Author path", [None] + list(opts), format_func=lambda a: "—" if a is None else opts[a])
            if who:
                path = q("""SELECT year, x, y FROM viz_author_year WHERE run_id = ? AND author_id = ?
                            AND facet = ? AND alpha = ? ORDER BY year""", (RUN, who, facet, alpha))
                path = path[path.year <= t2]
                if len(path):
                    fig.add_trace(go.Scatter(x=path.x, y=path.y, mode="lines+markers+text", text=path.year.astype(str),
                                             textposition="top right", textfont=dict(size=9),
                                             line=dict(color="#1f2328", width=2), marker=dict(size=6, color="#1f2328"),
                                             hoverinfo="skip", name="path"))
                    fig.add_trace(go.Scatter(x=path.x.tail(1), y=path.y.tail(1), mode="markers",
                                             marker=dict(size=14, color=ACCENT, symbol="star"), hoverinfo="skip"))

        if show_clusters and not clusters.empty:
            c = clusters.dropna(subset=["x"]).copy()
            c["share"] = c.cluster_id.map(n_win).fillna(0) / max(tot_win, 1)
            c = c.sort_values("share", ascending=False)
            # text only for the largest clusters in the window (and the selection and any cluster
            # with a status badge); the rest get a marker with the label on hover
            shown = c.head(12).cluster_id.tolist() + ([sel] if sel else [])
            if show_badges:
                shown += [k for k in c.cluster_id if status.get(k) in BADGE]
            txt = c[c.cluster_id.isin(shown)]
            fig.add_trace(go.Scatter(
                x=c.x, y=c.y, mode="markers", hoverinfo="text",
                marker=dict(size=[11 if k == "hub" else 7 for k in c.kind], color=c.color,
                            symbol=["diamond" if k == "hub" else "circle" for k in c.kind],
                            line=dict(color="white", width=1)),
                hovertext=[f"{disp(k)}<br>{s:.1%} of window" + (f"<br>status {t2}: {status[k]}" if status.get(k) else "")
                           for k, s in zip(c.cluster_id, c.share)], name="centroids"))
            fig.add_trace(go.Scatter(
                x=txt.x, y=txt.y, mode="text", text=[disp(k) for k in txt.cluster_id], textposition="top center",
                textfont=dict(size=(9 + 25 * np.sqrt(txt.share)).clip(10, 15).tolist(),
                              color=[ACCENT if cid == sel else "#1f2328" for cid in txt.cluster_id]),
                hoverinfo="skip", name="labels"))
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                          xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                          plot_bgcolor="white", dragmode="pan")
        ev = st.plotly_chart(fig, width="stretch", key="map", on_select="rerun",
                             selection_mode="points")
        pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
        if pts and pts[0].get("customdata"):
            pid = pts[0]["customdata"]
            st.session_state.sel_paper = pid
            hit = layout.loc[layout.paper_id == pid, "cluster_id"]
            if len(hit) and hit.iloc[0] in colors:
                st.session_state.sel_cluster = hit.iloc[0]
        if show_badges:
            st.caption(f"Status at {t2}: ▲ emerging · ◆ niche · ○ dormant · ◇ hub (unstable boundary, "
                       f"reported separately). Statuses exist for cutoff years with enough data. **Read with "
                       f"care:** in the backtest, emerging was barely better than chance, and several topics "
                       f"reverse direction once changes in the sampling mix are separated out. Treat a badge as "
                       f"a prompt to look, not as a trend.")

    # ---- grid + detail
    with right:
        tab_grid, tab_detail, tab_rank, tab_bridges, tab_people = st.tabs(
            ["Grid", "Detail", "Ranking", "Bridges", "People"])
        with tab_grid:
            grid_alpha = alpha if facet in GLOSS_FACETS else 0.3
            pm = q("""SELECT problem_cluster, method_cluster, year, count, first_year FROM pm_matrix
                      WHERE run_id = ? AND alpha = ?""", (RUN, grid_alpha))
            if pm.empty:
                st.info("No matrix for this α.")
            else:
                cl_all = q("""SELECT cluster_id, label, size, kind FROM clusters WHERE run_id = ? AND alpha = ?
                              AND resolution = 'medium' AND kind IN ('stable', 'hub')""", (RUN, grid_alpha))
                lab_all = dict(zip(cl_all.cluster_id, cl_all.label + np.where(cl_all.kind == "hub", " ◇", "")))
                w = pm[(pm.year >= t1) & (pm.year <= t2)]
                cells = w.groupby(["problem_cluster", "method_cluster"]).agg(
                    n=("count", "sum"), first=("first_year", "min")).reset_index()
                rows = cl_all[cl_all.cluster_id.str.startswith("problem:")].sort_values("size", ascending=False).cluster_id.tolist()
                cols = cl_all[cl_all.cluster_id.str.startswith("method:")].sort_values("size", ascending=False).cluster_id.tolist()
                mat = cells.pivot(index="problem_cluster", columns="method_cluster", values="n").reindex(index=rows, columns=cols).fillna(0)
                short = lambda c: (lab_all.get(c, c)[:28])  # noqa: E731
                fig = go.Figure(go.Heatmap(
                    z=np.log10(1 + mat.values), x=[short(c) for c in cols], y=[short(r) for r in rows],
                    customdata=mat.values, colorscale="Blues", showscale=False,
                    hovertemplate="%{y} × %{x}<br>%{customdata:.0f} papers in window<extra></extra>"))
                # 6.2 "new combination": first nonzero year within the last 2 years of the window
                new = cells[(cells["first"] >= max(t1, t2 - 1)) & (cells["first"] <= t2)
                            & cells.problem_cluster.isin(rows) & cells.method_cluster.isin(cols)]
                if len(new):
                    fig.add_trace(go.Scatter(
                        x=[short(c) for c in new.method_cluster], y=[short(r) for r in new.problem_cluster],
                        mode="markers", marker=dict(symbol="square-open", size=13, color=ACCENT, line=dict(width=2)),
                        hovertemplate="new combination (first year %{text})<extra></extra>",
                        text=new["first"].astype(int).astype(str)))
                n_tr = 0
                if have_signals and grid_alpha == 0.5:
                    tr = signals_df(("bridge_method_transfer",))
                    tr = tr[tr["rank"] <= 25]
                    sub = [json.loads(s) for s in tr.subject]
                    pts_ = [(s["method_cluster"], s["problem_cluster"], r) for s, r in zip(sub, tr["rank"])
                            if s["method_cluster"] in cols and s["problem_cluster"] in rows]
                    n_tr = len(pts_)
                    if pts_:
                        fig.add_trace(go.Scatter(
                            x=[short(m) for m, _, _ in pts_], y=[short(p_) for _, p_, _ in pts_], mode="markers",
                            marker=dict(symbol="diamond-open", size=14, color=ACCENT, line=dict(width=2)),
                            text=[f"method transfer candidate #{r}" for _, _, r in pts_],
                            hovertemplate="%{text}<extra></extra>"))
                if sel in rows or sel in cols:
                    if sel in rows:
                        fig.add_hrect(y0=rows.index(sel) - 0.5, y1=rows.index(sel) + 0.5, line=dict(color=ACCENT, width=2))
                    else:
                        fig.add_vrect(x0=cols.index(sel) - 0.5, x1=cols.index(sel) + 0.5, line=dict(color=ACCENT, width=2))
                fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0),
                                  xaxis=dict(tickangle=-50, tickfont=dict(size=9)),
                                  yaxis=dict(autorange="reversed", tickfont=dict(size=9)), showlegend=False)
                st.plotly_chart(fig, width="stretch", key="grid")
                st.caption(f"Medium clusters at α={grid_alpha} (◇ = hub). Colour = papers in {t1}–{t2} (log). "
                           f"□ new combinations first seen in {max(t1, t2 - 1)}–{t2} ({len(new)}). "
                           + (f"◇ top method-transfer candidates ({n_tr})." if grid_alpha == 0.5
                              else "Method-transfer candidates are outlined at α = 0.5."))
        with tab_rank:
            variant = st.radio("Corpus", ["all", "no_semantic"], horizontal=True,
                               format_func=lambda v: "whole corpus" if v == "all" else "without semantic route",
                               help="Signals are reported with and without the semantic-expansion route, "
                                    "so seed bias stays visible.")
            r = clusters[["cluster_id", "label", "kind", "size", "stability"]].copy()
            r["in_window"] = r.cluster_id.map(n_win).fillna(0).astype(int)
            r["share"] = r.in_window / max(tot_win, 1)
            r["share_prev"] = r.cluster_id.map(n_prev).fillna(0) / max(tot_prev, 1)
            r["Δ share"] = r.share - r.share_prev
            if variant == "all":
                sc = scores
            else:  # the no-semantic variant is stored as ranked signal rows at the latest cutoff
                sv = q("""SELECT subject, features FROM signals WHERE run_id = ? AND variant = 'no_semantic'
                          AND signal_type IN ('emerging_cluster', 'direction_growing', 'direction_shrinking', 'niche', 'dormant')""", (RUN,))
                sc = {json.loads(s)["cluster_id"]: json.loads(f) for s, f in zip(sv.subject, sv.features)}
                if t2 != y1:
                    st.caption(f"The no-semantic variant is computed at {y1} only.")
            for k in ("status", "g", "E", "N", "mig_net"):
                r[k] = r.cluster_id.map(lambda c: (sc.get(c) or {}).get(k) if k != "status" else
                                        (status.get(c) if variant == "all" else (sc.get(c) or {}).get("status")))
            r = r.sort_values("share", ascending=False)
            cfg = {"share": st.column_config.NumberColumn(format="percent"),
                   "Δ share": st.column_config.NumberColumn(format="percent"),
                   "stability": st.column_config.NumberColumn(format="%.2f"),
                   "g": st.column_config.NumberColumn("growth g", format="%.3f", help="slope of log share, last 3 years"),
                   "E": st.column_config.NumberColumn("emergence E′", format="%.2f",
                                                      help="z(momentum) + z(acceleration) − z(new-author share); "
                                                           "backtested replacement for the recipe's E"),
                   "N": st.column_config.NumberColumn("niche N", format="%.2f"),
                   "mig_net": st.column_config.NumberColumn("net migration"),
                   "cluster_id": None, "kind": None, "share_prev": None}
            for kind, title in (("stable", "Stable clusters"), ("hub", "Hub clusters (unstable boundary)")):
                part = r[r.kind == kind]
                if part.empty:
                    continue
                st.markdown(f"**{title}**")
                ev = st.dataframe(part, hide_index=True, width="stretch", on_select="rerun",
                                  selection_mode="single-row", key=f"rank_{kind}", column_config=cfg)
                if ev and ev.selection.rows:
                    st.session_state.sel_cluster = part.iloc[ev.selection.rows[0]].cluster_id
                    st.rerun()
            st.caption(f"Δ share compares {t1}–{t2} with the {prev_len} years before it. Scores are as of {t2}. "
                       "Shares and statuses describe this corpus as sampled; direction claims need a stable "
                       "yearly sample (see the Experiments view).")
        with tab_detail:
            options = [None] + clusters.sort_values("size", ascending=False).cluster_id.tolist()
            pick = st.selectbox("Cluster", options, index=options.index(sel) if sel in options else 0,
                                format_func=lambda c: "—" if c is None else f"{disp(c)} ({c.split(':')[-1]})")
            if pick != sel:
                st.session_state.sel_cluster = pick
                st.rerun()
            if sel:
                c = clusters.set_index("cluster_id").loc[sel]
                st.markdown(f"#### <span style='color:{c.color}'>●</span> {c.label}", unsafe_allow_html=True)
                if c.kind == "hub":
                    st.info(f"Hub cluster: a coherent body of work whose boundary shifts between clustering "
                            f"seeds (stability {c.stability:.2f}, centrality z {c.centrality:.2f}). Its signals "
                            f"are ranked separately from stable clusters.")
                st.caption(c.description if isinstance(c.description, str) else "")
                st.markdown(f"{c['size']} papers · stability {c.stability:.2f} · "
                            f"{int(n_win.get(sel, 0))} in {t1}–{t2}  \nterms: *{', '.join(list(c.top_terms)[:10])}*")
                if sel in scores:
                    s = scores[sel]
                    st.markdown(f"**As of {t2}:** status **{status.get(sel) or '—'}** · growth g {s.get('g')} · "
                                f"Δg {s.get('dg')} · emergence E {s.get('E')} · niche N {s.get('N')} · "
                                f"net author migration {s.get('mig_net')}")
                ev_p = q("""SELECT p.paper_id, p.year, p.title, round(pc.sim_to_centroid, 3) AS sim
                            FROM paper_cluster pc JOIN papers p USING (paper_id)
                            WHERE pc.run_id = ? AND pc.cluster_id = ? AND pc.resolution = ? AND pc.alpha = ?
                            ORDER BY pc.sim_to_centroid DESC LIMIT 10""", (RUN, sel, res, alpha))
                st.markdown("**Central papers** (evidence)")
                e = st.dataframe(ev_p[["year", "title", "sim"]], hide_index=True, width="stretch",
                                 on_select="rerun", selection_mode="single-row", key="evidence")
                if e and e.selection.rows:
                    st.session_state.sel_paper = ev_p.iloc[e.selection.rows[0]].paper_id
            if st.session_state.get("sel_paper"):
                st.divider()
                paper_card(st.session_state.sel_paper)
        with tab_bridges:
            mode = st.radio("Show", ["Cluster pairs", "Papers to bring in", "From outside the field"],
                            horizontal=True, key="br_mode")
            if mode == "Cluster pairs":
                bridges_panel()
            elif mode == "Papers to bring in":
                paper_bridges_panel(sel)
            else:
                external_imports_panel(sel)
        with tab_people:
            people_panel()

    if st.session_state.get("playing"):
        time.sleep(0.9)
        a, b = st.session_state.window
        nxt = b + 1 if b < y1 else y0
        st.session_state.window = (nxt, nxt)
        st.session_state.window_slider = (nxt, nxt)
        st.rerun()


def bridges_panel():
    b = signals_df(tuple(BRIDGE_TYPES))
    if b.empty:
        st.info("No bridge candidates. Run `landscape signals`.")
        return
    counts = b.signal_type.value_counts()
    typ = st.radio("Type", list(BRIDGE_TYPES), horizontal=False,
                   format_func=lambda t: f"{BRIDGE_TYPES[t]} ({counts.get(t, 0)})")
    d = b[b.signal_type == typ].sort_values("rank")
    if d.empty:
        st.caption("No candidates of this type for this run (see the signals doc for why this can be empty).")
        return

    def title(r):
        s = json.loads(r.subject)
        if "from_label" in s:
            return f"{s['from_label']} → {s['to_label']}"
        if "problem_label" in s:
            return f"{s['method_label']} → {s['problem_label']}"
        if "label_a" in s:
            return f"{s['label_a']} ~ {s['label_b']}"
        return s.get("label", "?")

    tab = pd.DataFrame({"rank": d["rank"], "candidate": [title(r) for r in d.itertuples()],
                        "judge": d.judge_score, "score": d.score.round(4)})
    ev = st.dataframe(tab.head(50), hide_index=True, width="stretch", height=260, on_select="rerun",
                      selection_mode="single-row", key=f"br_{typ}",
                      column_config={"judge": st.column_config.NumberColumn("judge 1–5", format="%.0f")})
    if not (ev and ev.selection.rows):
        st.caption("Ranked by the LLM judge (top 50 judged), then by score. Select a candidate for its evidence.")
        return
    r = d.iloc[ev.selection.rows[0]]
    f = json.loads(r.features)
    st.markdown(f"**#{r['rank']} {title(r)}**")
    if r.judge_rationale:
        st.markdown(f"Judge {int(r.judge_score)}/5: {r.judge_rationale}")
        if f.get("judge_obvious_reason"):
            st.caption(f"Obvious reason it hasn't been done: {f['judge_obvious_reason']}")
    titles = q(f"""SELECT paper_id, year, title FROM papers WHERE paper_id IN
                   ({",".join("?" * len(r.evidence_paper_ids))})""", tuple(r.evidence_paper_ids)).set_index("paper_id")

    def lim_text(pid, facet):
        t = q("SELECT text FROM facets WHERE paper_id = ? AND facet = ?", (pid, facet))
        return t.text.iloc[0] if len(t) else ""

    if typ == "bridge_lim_inn":
        for pr in f.get("pairs", []):
            l, i = pr["limitation"], pr["innovation"]
            st.markdown(f"- **Limitation** ({titles.year.get(l)}, *{titles.title.get(l, '')[:80]}*): {lim_text(l, 'limitations')}  \n"
                        f"  **Innovation** ({titles.year.get(i)}, *{titles.title.get(i, '')[:80]}*): {lim_text(i, 'innovation')}  \n"
                        f"  cosine {pr['sim']}")
        st.caption(f"Citations A→B: {f.get('flow_xy')} (max from A to any cluster: {f.get('flow_max')}).")
    else:
        for pid in r.evidence_paper_ids:
            st.markdown(f"- {titles.year.get(pid)} · {titles.title.get(pid, pid)}")
        st.json({k: v for k, v in f.items() if k not in ("judge_obvious_reason", "pairs")}, expanded=False)
    note = st.text_input("Note", key=f"note_{typ}_{r['rank']}")
    c = st.columns(3)
    for col, act in zip(c, ["worth pursuing", "obvious", "wrong"]):
        if col.button(act, key=f"{act}_{typ}_{r['rank']}"):
            record("bridges", {"signal_type": typ, "subject": json.loads(r.subject), "rank": int(r["rank"])},
                   act, note, alpha={"problem": 0.5, "method": 0.5})
            st.success("Saved to review_actions.")


def how_to_read(text: str) -> None:
    with st.expander("How to read this"):
        st.markdown(text)


def paper_bridges_panel(sel):
    b = signals_df(("bridge_paper", "bridge_paper_gem"))
    if b.empty:
        st.info("No paper bridges. Run `landscape signals`.")
        return
    how_to_read(
        "Pick a problem cluster. Each row is an **innovation paper from another cluster that no paper here "
        "cites yet**, ranked by how likely this cluster is to start citing it in the next three years.\n\n"
        "- **Likely to be brought in**: the top 10. About 1 in 4 is picked up within three years in the "
        "backtest (popularity alone: about 1 in 5). Many will already be familiar.\n"
        "- **Hidden gems**: papers cited at most once so far. Fewer are picked up, but popularity can't find "
        "them at all; this is where the list earns its keep.\n"
        "- **from** is the paper's own cluster, **cites** its corpus citations so far, and **score** the model's "
        "log-odds. Compare scores within one list only.\n"
        "- **Select a row** to see the innovation next to this cluster's closest stated limitation. Mark "
        "*wrong* if they don't connect, and *obvious* if everyone in the field already knows it.\n\n"
        "The ranking mostly reflects shared references with the cluster. The limitation match explains a "
        "pick; it is not what drives the ranking.")
    b["subj"] = b.subject.map(json.loads)
    b["cluster_id"] = b.subj.map(lambda s: s["cluster_id"])
    labs = dict(zip(b.cluster_id, b.subj.map(lambda s: s.get("label") or s["cluster_id"])))
    kinds = dict(zip(b.cluster_id, b.group_kind))
    size = dict(q("SELECT cluster_id, size FROM clusters WHERE run_id = ?", (RUN,)).itertuples(index=False))
    opts = sorted(labs, key=lambda c: -size.get(c, 0))  # largest first
    pick = st.selectbox("Problem cluster (α = 0.5, medium)", opts, index=opts.index(sel) if sel in opts else 0,
                        format_func=lambda c: f"{labs[c]}{' ◇hub' if kinds.get(c) == 'hub' else ''} ({size.get(c, '?')})",
                        key="pb_cluster")
    d = b[b.cluster_id == pick]
    all_labs = dict(q("SELECT cluster_id, label FROM clusters WHERE run_id = ?", (RUN,)).itertuples(index=False))
    chosen = None
    for typ, title, help_ in (
            ("bridge_paper", "Likely to be brought in",
             "Innovation papers no paper in this cluster cites yet, ranked by the adoption model "
             "(backtest p@10 0.23 vs popularity 0.19)"),
            ("bridge_paper_gem", "Hidden gems (cited ≤ 1×)",
             "Same model, restricted to papers cited at most once (backtest long-tail AUC 0.84 vs popularity 0.55)")):
        part = d[d.signal_type == typ].sort_values("rank")
        if part.empty:
            continue
        st.markdown(f"**{title}**", help=help_)
        f = part.features.map(json.loads)
        tab = pd.DataFrame({"rank": part["rank"], "paper": part.subj.map(lambda s: s.get("title") or s["paper_id"]),
                            "from": part.subj.map(lambda s: all_labs.get(s.get("from_cluster"), "")),
                            "cites": f.map(lambda x: x.get("citations_so_far")), "score": part.score.round(2)})
        ev = st.dataframe(tab, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                          key=f"pb_{typ}_{pick}")
        if ev and ev.selection.rows:
            chosen = part.iloc[ev.selection.rows[0]]
    if chosen is None:
        st.caption("Select a paper for the limitation it matches.")
        return
    s, f = chosen.subj, json.loads(chosen.features)

    def facet_text(pid, facet):
        t = q("SELECT text FROM facets WHERE paper_id = ? AND facet = ?", (pid, facet))
        return t.text.iloc[0] if len(t) else "—"

    st.markdown(f"**Innovation** ({s.get('title', s['paper_id'])}): {facet_text(s['paper_id'], 'innovation')}")
    lim = f.get("best_matching_limitation")
    if lim:
        lt = q("SELECT year, title FROM papers WHERE paper_id = ?", (lim,))
        st.markdown(f"**Closest limitation in {labs[pick]}** ({lt.year.iloc[0] if len(lt) else ''}, "
                    f"*{lt.title.iloc[0][:90] if len(lt) else lim}*): {facet_text(lim, 'limitations')}")
    st.caption(f"coupling {f.get('coupling')} · problem similarity {f.get('problem_sim')} · "
               f"limitation similarity {f.get('limitation_sim')} · {f.get('citations_so_far')} citations so far")
    if st.session_state.get("sel_paper") != s["paper_id"]:  # the Detail tab shows its card
        st.session_state.sel_paper = s["paper_id"]
        st.rerun()
    note = st.text_input("Note", key=f"pbnote_{pick}_{s['paper_id']}")
    c = st.columns(3)
    for col, act in zip(c, ["worth pursuing", "obvious", "wrong"]):
        if col.button(act, key=f"pb_{act}_{pick}_{s['paper_id']}"):
            record("bridges", {"signal_type": chosen.signal_type, "subject": s, "rank": int(chosen["rank"])},
                   act, note, alpha={"problem": 0.5})
            st.success("Saved to review_actions.")


@st.cache_data(show_spinner=False)
def _external(run: str, name: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(store.DATA / "backtest" / run / "external" / name)


def external_imports_panel(sel):
    path = store.DATA / "backtest" / RUN / "external" / "live_imports.parquet"
    if not path.exists():
        st.info("No external-import list for this run. Run `scripts/phase8b/external_mine.py` and "
                "`scripts/phase8b/exp6_imports.py` (see the 08b doc).")
        return
    L = _external(RUN, "live_imports.parquet", path.stat().st_mtime)
    how_to_read(
        f"Works **outside the corpus** that corpus papers cite, as of {int(L.as_of.iloc[0])}, ranked by how likely "
        "the field is to cite them more and more over the next three years. Only works with at least 3 recent "
        "citing papers in the corpus are listed.\n\n"
        "- **What drives the ranking** (experiment 6): the work's fit with the problems the field is moving "
        "*towards*, and its citation momentum across all of science (**global trend**, the ratio of this year's "
        "citations to the average of the two before). Both add to the in-field citation rate, modestly: about "
        "+0.025 AUC, with no sharper top of the list.\n"
        "- **Outside the field** keeps the third of the list whose abstracts are farthest from the corpus: ideas "
        "genuinely borrowed from elsewhere (subfield labels are too noisy for this). **In-field, missing from the "
        "corpus** shows the rest: often core papers of this field that the corpus never fetched, a coverage gap "
        "rather than a borrowing.\n"
        "- **Select a row** for the work's innovation, the closest paper in the corpus, and the corpus papers now "
        "citing it. Papers that ride an incoming wave out-draw their peers (exp5); being the first to import does "
        "not pay as well in-field.")
    c1, c2 = st.columns([3, 2])
    scope = c1.radio("Scope", ["Outside the field", "In-field, missing from the corpus", "All"], horizontal=True,
                     key="ext_scope")
    recent = c1.checkbox("Recent works only (last 10 years)", True, key="ext_recent",
                         help="Off also lists the classics the field keeps citing: foundations rather than trends.")
    d = L[L.year >= int(L.as_of.iloc[0]) - 10] if recent else L
    if scope == "Outside the field":
        d = d[d.outside_field]
    elif scope.startswith("In-field"):
        d = d[~d.outside_field]
    clusters = d.dropna(subset=["cluster_id"]).drop_duplicates("cluster_id").set_index("cluster_id").cluster_label
    opts = [None] + clusters.index.tolist()
    pick = c2.selectbox("Closest problem cluster", opts, index=opts.index(sel) if sel in opts else 0,
                        format_func=lambda c: "all" if c is None else clusters.get(c, c), key="ext_cluster")
    if pick:
        d = d[d.cluster_id == pick]
    d = d.head(40)
    tab = pd.DataFrame({"work": d.title, "year": d.year, "subfield": d.subfield,
                        "global trend": np.exp(d.g_trend).round(2), "citers here, last 3 yrs": d.recent_citers,
                        "closest problem cluster": d.cluster_label, "score": d.score.round(2)})
    ev = st.dataframe(tab, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                      key=f"ext_{scope}_{pick}")
    st.caption(f"{len(d)} shown. Global trend 1.5 means 50% more citations this year than the average of the two "
               "before, across all of science.")
    if not (ev and ev.selection.rows):
        return
    r = d.iloc[ev.selection.rows[0]]
    fpath = store.DATA / "backtest" / RUN / "external" / "facets.parquet"
    if fpath.exists():
        f = _external(RUN, "facets.parquet", fpath.stat().st_mtime)
        f = f[f.paper_id == r.paper_id].set_index("facet").text
        for k in ("problem", "innovation", "limitations"):
            if isinstance(f.get(k), str):
                st.markdown(f"**{k.capitalize()}** ({r.title}): {f[k]}")
    st.caption(f"{r.venue or ''} · {int(r.year) if pd.notna(r.year) else ''} · {int(r.cited_by_count or 0)} citations "
               f"overall · {int(r.corpus_citers_total)} citing papers in the corpus · "
               f"[OpenAlex](https://openalex.org/{r.paper_id})")
    epath = store.DATA / "backtest" / RUN / "external" / "edges.parquet"
    e = _external(RUN, "edges.parquet", epath.stat().st_mtime)
    riders = e[(e.cited_id == r.paper_id) & (e.citing_year >= int(r.as_of) - 2)].citing_id.unique().tolist()
    if riders:
        t = q(f"""SELECT paper_id, year, title FROM papers WHERE paper_id IN ({",".join("?" * len(riders))})
                  ORDER BY year DESC""", tuple(riders))
        st.markdown(f"**Corpus papers citing it since {int(r.as_of) - 2}** ({len(t)})")
        st.dataframe(t[["year", "title"]], hide_index=True, width="stretch", height=180)
    if isinstance(r.closest_corpus_paper, str):
        st.markdown("**Closest paper in the corpus** (its problem statement is nearest to this work)")
        paper_card(r.closest_corpus_paper)
    note = st.text_input("Note", key=f"extnote_{r.paper_id}")
    c = st.columns(3)
    for col, act in zip(c, ["worth pursuing", "obvious", "wrong"]):
        if col.button(act, key=f"ext_{act}_{r.paper_id}"):
            record("bridges", {"signal_type": "external_import", "paper_id": r.paper_id, "title": r.title,
                               "cluster_id": r.cluster_id}, act, note)
            st.success("Saved to review_actions.")


def people_panel():
    ppl = signals_df(("seeder_author", "broker_author", "institution_seeder", "institution_broker"))
    if ppl.empty:
        st.info("Run `landscape signals`.")
        return
    st.caption("Internal decision support only: these lists are not a public ranking of people. "
               "Descriptive roles, not predictions: in the Phase 8 backtest they did not beat "
               "plain popularity at predicting future citations.")
    for typ, title, col in (("seeder_author", "Seeders", "early papers in growing clusters"),
                            ("broker_author", "Brokers", "participation across clusters"),
                            ("institution_seeder", "Institutions (seeders)", "weighted score"),
                            ("institution_broker", "Institutions (brokers)", "weighted score")):
        d = ppl[ppl.signal_type == typ].sort_values("rank").head(15)
        if d.empty:
            continue
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame({"rank": d["rank"], "name": [json.loads(s).get("name") for s in d.subject],
                                   col: d.score.round(3)}), hide_index=True, width="stretch")
    st.caption("Turn on the People layer and pick an author to see their yearly path on the map.")
    ew = signals_df(("early_work",))
    if len(ew):
        st.markdown("**Early work**", help="Papers from the last two years most likely to become top-cited for "
                                           "their year (backtest AUC 0.82, p@50 0.67 vs citations-so-far 0.62). "
                                           "Author fame is deliberately not a feature.")
        how_to_read(
            "Papers from the last two years that the model expects to be in the **top quarter of their year "
            "for citations from this field** over the next three years.\n\n"
            "- **What drives it:** shared references with the field, early citations, and closeness to its "
            "cluster's core. Novelty counts *against* a paper: novel work is cited less in the short run.\n"
            "- **It predicts in-field uptake, not global impact.** In this corpus the two are unrelated. It "
            "also partly reflects audience size: papers on the field's mainstream questions simply have more "
            "potential readers. Read it as *what this community will read next*, not *the best papers*.\n"
            "- **cites** is corpus citations so far, often 0 for this year's papers. **score** is log-odds, so "
            "compare ranks rather than values.\n\n"
            "Select a row to open the paper's card in the Detail tab.")
        f = ew.features.map(json.loads)
        tab = pd.DataFrame({"rank": ew["rank"], "year": f.map(lambda x: x.get("year")),
                            "paper": ew.subject.map(lambda x: json.loads(x).get("title")),
                            "cites": f.map(lambda x: x.get("citations_so_far")), "score": ew.score.round(2)})
        ev = st.dataframe(tab.head(30), hide_index=True, width="stretch", on_select="rerun",
                          selection_mode="single-row", key="early_work")
        if ev and ev.selection.rows:
            pid = json.loads(ew.iloc[ev.selection.rows[0]].subject)["paper_id"]
            if st.session_state.get("sel_paper") != pid:
                st.session_state.sel_paper = pid
                st.rerun()
            st.caption("The Detail tab shows the selected paper's card.")


# ================================================================== Corpus view

def view_corpus():
    st.subheader("Corpus view")
    st.caption("Composition by entry route, hop and year. Use it to spot seed bias, e.g. a "
               "sub-field visible only through the semantic route.")
    k = q("""SELECT count(*) n, avg(CASE WHEN abstract_raw IS NULL OR length(abstract_raw) < 100 THEN 1 ELSE 0 END) miss
             FROM provenance JOIN papers USING (paper_id) WHERE run_id = ?""", (RUN,)).iloc[0]
    spend = json.loads(q("SELECT spend_usd FROM runs WHERE run_id = ?", (RUN,)).spend_usd[0] or "{}")
    c = st.columns(4)
    c[0].metric("Papers", f"{int(k.n):,}")
    c[1].metric("Missing abstract", f"{k.miss:.1%}", help="title-only after S2 and arXiv fill")
    c[2].metric("Metadata spend", f"${spend.get('openalex', 0):.3f}")
    c[3].metric("LLM spend", f"${spend.get('llm_extraction', 0) + spend.get('llm_labels', 0):.2f}")

    by = q("""SELECT p.year, v.route, count(*) n FROM provenance v JOIN papers p USING (paper_id)
              WHERE v.run_id = ? GROUP BY ALL ORDER BY 1""", (RUN,))
    # older backward-hop classics are few per year: lump them into one bar
    by["yl"] = np.where(by.year < 2010, "≤2009", by.year.astype("Int64").astype(str))
    by = by.groupby(["yl", "route"], as_index=False).n.sum().rename(columns={"yl": "year"})
    fig = go.Figure()
    for route, color in [("seed", ACCENT), ("ai_report", "#ae3ec9"), ("back", "#1c7ed6"),
                         ("forward", "#37b24d"), ("author", "#f59f00"), ("semantic", "#868e96")]:
        s = by[by.route == route]
        if len(s):
            fig.add_trace(go.Bar(x=s.year, y=s.n, name=route, marker_color=color))
    fig.update_layout(barmode="stack", height=320, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h"), xaxis=dict(type="category", categoryorder="array",
                                 categoryarray=["≤2009"] + sorted(y for y in by.year.unique() if y != "≤2009")))
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("**Routes and hops**")
        st.dataframe(q("""SELECT route, hop, count(*) papers,
                round(avg(CASE WHEN abstract_raw IS NULL OR length(abstract_raw) < 100 THEN 1 ELSE 0 END), 3) missing_abstract,
                sum((abstract_source = 's2')::INT) s2_filled, sum((abstract_source = 'arxiv')::INT) arxiv_filled,
                min(year) first_year, max(year) last_year
            FROM provenance JOIN papers USING (paper_id) WHERE run_id = ? GROUP BY ALL ORDER BY ALL""", (RUN,)),
            hide_index=True, width="stretch",
            column_config={"missing_abstract": st.column_config.NumberColumn(format="percent")})
    with b:
        st.markdown("**Primary topics: citation routes vs semantic route**")
        t = q("""SELECT json_extract_string(p.topics, '$[0].name') topic,
                        sum((v.route != 'semantic')::INT) citation_routes, sum((v.route = 'semantic')::INT) semantic
                 FROM provenance v JOIN papers p USING (paper_id) WHERE v.run_id = ?
                 GROUP BY 1 ORDER BY citation_routes + semantic DESC LIMIT 15""", (RUN,))
        st.dataframe(t, hide_index=True, width="stretch")

    st.markdown("**Browse papers by route**")
    r = st.selectbox("Route", by.route.unique().tolist())
    papers = q("""SELECT p.paper_id, p.year, p.title, v.hop, v.via_id, round(v.score, 3) score,
                         p.abstract_source FROM provenance v JOIN papers p USING (paper_id)
                  WHERE v.run_id = ? AND v.route = ? ORDER BY v.score DESC NULLS LAST, p.year DESC
                  LIMIT 500""", (RUN, r))
    ev = st.dataframe(papers.drop(columns="paper_id"), hide_index=True, width="stretch", height=300,
                      on_select="rerun", selection_mode="single-row", key="corpus_papers")
    if ev and ev.selection.rows:
        paper_card(papers.iloc[ev.selection.rows[0]].paper_id)

    st.divider()
    note = st.text_area("Composition notes", placeholder="e.g. 'Geometric analysis papers via the semantic route look off-field'")
    if st.button("Mark composition checked", type="primary"):
        record("corpus", {"run_id": RUN}, "composition_checked", note)
        st.success("Saved to review_actions.")
    prev = my_actions("corpus")
    if len(prev):
        st.caption(f"You marked this run {len(prev)} time(s); last {prev.ts.iloc[-1]:%Y-%m-%d %H:%M}.")


# ================================================================== Facet review

FAITHFUL = ["problem", "domain_data", "method", "innovation", "result", "limitations"]
GLOSSES = ["problem_gloss", "method_gloss"]


def view_facets():
    st.subheader("Facet review")
    run_p = json.loads(q("SELECT params FROM runs WHERE run_id = ?", (RUN,)).params[0])["extraction"]
    model, pv = run_p["model"], run_p["prompt_version"]
    ok = q("""SELECT s.paper_id, s.qa_flags FROM facet_status s JOIN provenance v USING (paper_id)
              WHERE v.run_id = ? AND s.model_id = ? AND s.prompt_version = ? AND s.status = 'ok'""",
           (RUN, model, pv))
    ok["h"] = [hashlib.md5(f"{RUN}|{p}".encode()).hexdigest() for p in ok.paper_id]
    gold = ok.sort_values("h").head(200).paper_id.tolist()
    mode = st.radio("Papers", ["Gold set (200, fixed sample)", "QA-flagged", "With limitations"],
                    horizontal=True)
    if mode.startswith("Gold"):
        pool = gold
    elif mode == "QA-flagged":
        pool = ok[ok.qa_flags.fillna("{}").str.contains("too_long|tag_too_long")].sort_values("h").paper_id.tolist()
    else:
        pool = q("""SELECT f.paper_id FROM facets f JOIN provenance v USING (paper_id)
                    WHERE v.run_id = ? AND f.facet = 'limitations' AND f.text IS NOT NULL
                      AND f.model_id = ? AND f.prompt_version = ? ORDER BY f.paper_id""", (RUN, model, pv)).paper_id.tolist()
    done = my_actions("facet_review")
    done_ids = {json.loads(s)["paper_id"] for s in done.subject} if len(done) else set()
    st.progress(len(done_ids & set(pool)) / max(len(pool), 1),
                text=f"{len(done_ids & set(pool))} of {len(pool)} reviewed by you")
    if not pool:
        st.info("Nothing in this pool.")
        return
    key = f"fi_{mode}"
    if key not in st.session_state:
        st.session_state[key] = next((i for i, p in enumerate(pool) if p not in done_ids), 0)
    c = st.columns([1, 1, 4])
    if c[0].button("◀ Previous"):
        st.session_state[key] = max(0, st.session_state[key] - 1)
    if c[1].button("Next ▶"):
        st.session_state[key] = min(len(pool) - 1, st.session_state[key] + 1)
    i = st.session_state[key]
    pid = pool[i]
    c[2].caption(f"Paper {i + 1} of {len(pool)} · `{pid}`" + (" · already reviewed" if pid in done_ids else ""))

    p = q("SELECT title, year, abstract_raw, abstract_source FROM papers WHERE paper_id = ?", (pid,)).iloc[0]
    f = q("""SELECT facet, text FROM facets WHERE paper_id = ? AND model_id = ? AND prompt_version = ?""",
          (pid, model, pv)).set_index("facet").text
    flags = json.loads(ok.set_index("paper_id").qa_flags.get(pid) or "{}")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"**{p.title}** ({int(p.year)})")
        st.write(p.abstract_raw)
        st.caption(f"abstract source: {p.abstract_source} · tags: `{f.get('problem_class')}` / "
                   f"`{f.get('method_family')}` · {f.get('contribution_type')} · QA flags: {flags or 'none'}")
    ratings = {}
    with right, st.form(f"facets_{pid}"):
        for k in FAITHFUL:
            st.markdown(f"**{k}**  \n{f.get(k) if f.get(k) else '*null*'}")
            ratings[k] = st.radio(k, ["supported", "partly supported", "unsupported"], horizontal=True,
                                  key=f"r_{pid}_{k}", label_visibility="collapsed", index=None)
        for k in GLOSSES:
            st.markdown(f"**{k}** *(model knowledge)*  \n{f.get(k)}")
            ratings[k] = st.radio(k, ["correct variant", "variant unclear (ok)", "wrong variant"],
                                  horizontal=True, key=f"r_{pid}_{k}", label_visibility="collapsed",
                                  index=None)
        note = st.text_input("Note")
        # no default rating: a preselected "supported" would bias the gold set upwards
        if st.form_submit_button("Save ratings", type="primary"):
            missing = [k for k, v in ratings.items() if v is None]
            if missing:
                st.error("Rate every field before saving: " + ", ".join(missing))
                st.stop()
            for k, v in ratings.items():
                record("facet_review", {"paper_id": pid, "facet": k, "model_id": model,
                                        "prompt_version": pv, "gold_set": pid in gold}, v, note)
            st.session_state[key] = min(len(pool) - 1, i + 1)
            st.rerun()

    st.divider()
    # read uncached: ratings change as reviewers save
    all_r = _con().cursor().execute(
        """SELECT json_extract_string(subject, '$.facet') facet, action, count(*) n FROM review_actions
           WHERE run_id = ? AND view = 'facet_review' AND json_extract_string(subject, '$.gold_set') = 'true'
           GROUP BY ALL""", [RUN]).df()
    if len(all_r):
        st.markdown("**Gold-set acceptance so far (all reviewers)**")
        faith = all_r[all_r.facet.isin(FAITHFUL)]
        sup = faith[faith.action == "supported"].n.sum() / max(faith.n.sum(), 1)
        lim = all_r[all_r.facet == "limitations"]
        unsup_lim = lim[lim.action == "unsupported"].n.sum() / max(lim.n.sum(), 1)
        gl = all_r[all_r.facet.isin(GLOSSES)]
        gl_ok = gl[gl.action != "wrong variant"].n.sum() / max(gl.n.sum(), 1)
        c = st.columns(3)
        c[0].metric("Faithful facets supported", f"{sup:.0%}", help="target ≥ 90%")
        c[1].metric("Limitations unsupported", f"{unsup_lim:.0%}", help="target < 5% (of non-null)")
        c[2].metric("Gloss correct variant", f"{gl_ok:.0%}", help="target ≥ 85%")


# ================================================================== Cluster review

def view_clusters():
    st.subheader("Cluster review")
    c = st.columns(3)
    with c[0]:
        facet = st.selectbox("Facet", ["problem", "method", "innovation", "result", "limitations"])
    with c[1]:
        alpha = st.select_slider("Gloss weight α", ALPHAS, value=0.3, disabled=facet not in GLOSS_FACETS)
        alpha = alpha if facet in GLOSS_FACETS else 0.0
    with c[2]:
        res = st.select_slider("Resolution", RESOLUTIONS, value="medium")
    cl = q("""SELECT cluster_id, label, size, stability, kind, parent_cluster_id, label_model,
                     description, top_terms, color FROM clusters
              WHERE run_id = ? AND facet = ? AND alpha = ? AND resolution = ? AND size >= 5
              ORDER BY size DESC""", (RUN, facet, alpha, res))
    left, right = st.columns([1, 1], gap="large")
    with left:
        ev = st.dataframe(cl[["label", "size", "stability", "kind"]], hide_index=True, width="stretch",
                          height=560, on_select="rerun", selection_mode="single-row", key="cl_table",
                          column_config={"label": st.column_config.TextColumn(width="medium"),
                                         "size": st.column_config.NumberColumn(width="small"),
                                         "stability": st.column_config.NumberColumn(format="%.2f", width="small"),
                                         "kind": st.column_config.TextColumn(width="small")})
        st.caption("stable: Jaccard ≥ 0.6 across 5 Leiden seeds. hub: unstable boundary but large and "
                   "coherent; kept and reported separately. unstable: papers unassigned downstream.")
    if not (ev and ev.selection.rows):
        right.info("Select a cluster.")
        return
    r = cl.iloc[ev.selection.rows[0]]
    with right:
        st.markdown(f"#### <span style='color:{r.color}'>●</span> {r.label}", unsafe_allow_html=True)
        st.caption(f"{r.description if isinstance(r.description, str) else ''}\n`{r.cluster_id}` · labelled by {r.label_model} · parent "
                   f"`{r.parent_cluster_id or '—'}`")
        st.markdown(f"**{r['size']} papers · stability {r.stability:.2f}**  \nterms: *{', '.join(list(r.top_terms))}*")
        s = q("""SELECT year, n, share FROM cluster_year WHERE run_id = ? AND cluster_id = ? ORDER BY year""",
              (RUN, r.cluster_id))
        if len(s):
            fig = go.Figure(go.Scatter(x=s.year, y=s.share, mode="lines+markers", line=dict(color=r.color)))
            fig.update_layout(height=140, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(tickformat=".0%"),
                              xaxis=dict(dtick=1))
            st.plotly_chart(fig, width="stretch")
        central = q("""SELECT p.paper_id, p.year, p.title, f.text, round(pc.sim_to_centroid, 3) sim
                       FROM paper_cluster pc JOIN papers p USING (paper_id)
                       LEFT JOIN facets f ON f.paper_id = pc.paper_id AND f.facet = pc.facet
                       WHERE pc.run_id = ? AND pc.facet = ? AND pc.alpha = ? AND pc.resolution = ?
                         AND pc.raw_cluster_id = ?
                       ORDER BY pc.sim_to_centroid DESC LIMIT 12""",
                    (RUN, facet, alpha, res, r.cluster_id))
        st.markdown("**Central papers** (facet text)")
        st.dataframe(central[["year", "title", "text", "sim"]], hide_index=True, width="stretch",
                     column_config={"text": st.column_config.TextColumn(width="large")})

        st.markdown("**Your judgment**")
        others = cl[cl.cluster_id != r.cluster_id]
        with st.form(f"cr_{r.cluster_id}"):
            action = st.radio("Action", ["approve", "rename", "merge", "split", "flag"], horizontal=True)
            new_label = st.text_input("New label (rename)")
            merge_into = st.selectbox("Merge into", [None] + others.cluster_id.tolist(),
                                      format_func=lambda c: "—" if c is None else others.set_index("cluster_id").label[c])
            note = st.text_input("Note (why; for split, how)")
            if st.form_submit_button("Save", type="primary"):
                subj = {"cluster_id": r.cluster_id, "facet": facet, "resolution": res}
                if action == "rename":
                    subj["new_label"] = new_label
                if action == "merge":
                    subj["merge_into"] = merge_into
                record("cluster_review", subj, action, note, alpha={facet: alpha})
                st.success("Saved to review_actions.")
        prev = _con().cursor().execute(
            """SELECT reviewer, action, note, ts FROM review_actions WHERE run_id = ? AND view = 'cluster_review'
               AND json_extract_string(subject, '$.cluster_id') = ? ORDER BY ts DESC""", [RUN, r.cluster_id]).df()
        if len(prev):
            st.dataframe(prev, hide_index=True, width="stretch")


def view_research_lab():
    from landscape.research_lab_view import render_lab

    render_lab(RUN, q, paper_card, record)


def view_experiments():
    from landscape.experiments_view import render_experiments

    render_experiments(RUN)


{"Map & grid": view_map, "Research lab": view_research_lab, "Experiments": view_experiments,
 "Corpus view": view_corpus, "Facet review": view_facets, "Cluster review": view_clusters}[view]()
