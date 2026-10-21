"""Explorer views built on the product portfolio (src/landscape/portfolio.py):

- Start here: orientation in a field (what the map covers, communities, where ideas come from, what to read,
  opportunities, coverage gaps, how far to trust each view).
- My paper: for one paper (in the corpus, or analysed by DOI with `landscape my-paper`), the communities most
  likely to be warm to it and the opportunities around it.

Every list says what it is backed by; numbers come from the experiments in docs/features/08b-signal-experiments.md.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import embed, external, portfolio

REASONS = {"lcoup": "cites the same literature", "sim_recent": "its recent work is close to yours",
           "prob_sim": "asks related problems", "lim_sim3_f": "states limitations your work addresses",
           "l_size": "large community"}


@st.cache_data(show_spinner=False)
def _parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load(run: str, name: str) -> pd.DataFrame | None:
    p = portfolio.out_dir(run) / name
    return _parquet(str(p), p.stat().st_mtime) if p.exists() else None


@st.cache_data(show_spinner=False)
def _corpus_vectors(facet: str, version: str, dim: int):
    ids, v = embed.load_vectors(facet, version=version)
    return {x: k for k, x in enumerate(ids)}, (embed.truncate(v, dim) if dim < v.shape[1] else v)


@st.cache_data(show_spinner=False)
def _ext_vectors(run: str, facet: str):
    return external.vectors(run, facet, portfolio.DIM)


def _how(text: str) -> None:
    with st.expander("How to read this"):
        st.markdown(text)


def _embedding_version(q, run) -> str:
    from .config import RunParams

    return RunParams.from_dict(json.loads(q("SELECT params FROM runs WHERE run_id = ?", (run,)).params.iloc[0])).embedding.version


# ================================================================== second waves

def _second_waves_table(S: pd.DataFrame, q, key: str, n: int = 12) -> None:
    """One kind of second wave: a table, and the evidence of the selected idea."""
    S = S.head(n).reset_index(drop=True)
    short = lambda xs: ", ".join(list(xs)[:2]) + (f" +{len(xs) - 2}" if len(xs) > 2 else "")  # noqa: E731
    tab = pd.DataFrame({"idea": S.idea, "from": S.origin, "year": S.year.astype("Int64"),
                        "new audiences": [f"{n_}: {short(a)}" for n_, a in zip(S.n_new, S.new_audiences)],
                        "citing papers (3 yrs / 3 before)": [f"{r} / {e}" for r, e in zip(S.recent, S.earlier)],
                        "quiet uptake": S.get("n_quiet", pd.Series(0, index=S.index)),
                        "likely next": [short(x) for x in S.likely_next]})
    ev = st.dataframe(tab, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key=key)
    if ev and ev.selection.rows:
        r = S.iloc[ev.selection.rows[0]]
        st.markdown(f"**{r.idea}**  \n*New audiences:* {', '.join(r.new_audiences)}  \n"
                    + (f"*Moving towards it without citing it:* {', '.join(r.quiet_uptake)}  \n" if len(r.get('quiet_uptake', [])) else "")
                    + (f"*Likely next:* {', '.join(r.likely_next)}  \n" if len(r.likely_next) else "")
                    + f"*Backed by:* {r.backed_by}")
        ev_ids = list(r.evidence)
        if ev_ids:
            t = q(f"SELECT year, title FROM papers WHERE paper_id IN ({','.join('?' * len(ev_ids))}) ORDER BY year DESC",
                  tuple(ev_ids))
            st.dataframe(t, hide_index=True, width="stretch")


SECOND_WAVES_HOW = (
    "Established ideas reaching **new audiences**: communities that began citing (or, for methods, using) them in the "
    "last three complete years. **Citing papers** compares the last three years with the three before; only ideas "
    "gaining use are listed (research works, and outside works from the last 15 years). **Quiet uptake** counts "
    "communities whose recent work moved towards the idea without citing it (experiment 9: most uptake is uncited). "
    "**Likely next** comes from the backtested models: community warmth for papers (experiment 10), method transfer "
    "for methods (experiment 4), the closest community for outside works. This is a watchlist of diffusion already "
    "under way, not advance warning: on the pilot, the import model improved ordering modestly (+0.025 AUC) without "
    "a sharper top of the list. Select an idea for its evidence.")


# ================================================================== Start here

def render_start_here(run: str, q, paper_card, record) -> None:
    summ_p = portfolio.out_dir(run) / "summary.json"
    st.title("Start here")
    if not summ_p.exists():
        st.info(f"Build the portfolio first: `uv run landscape external --run-id {run}` (once, about $1 of LLM "
                f"extraction) and `uv run landscape portfolio --run-id {run}`.")
        return
    summ = json.loads(summ_p.read_text())
    T = int(summ["as_of"])
    papers = q("""SELECT p.year, v.route FROM papers p JOIN provenance v USING (paper_id) WHERE v.run_id = ?""", (run,))
    cl = q("""SELECT cluster_id, label, kind, size FROM clusters WHERE run_id = ? AND facet = 'problem' AND alpha = 0.5
              AND resolution = 'medium' AND kind IN ('stable', 'hub')""", (run,))
    st.caption(f"Run **{run}** · as of {T} (the latest complete publication year) · built {summ['built_at'][:16]}")
    c = st.columns(5)
    c[0].metric("Papers in the map", f"{len(papers):,}")
    c[1].metric("Publication years", f"{int(papers.year.quantile(0.05))}–{int(papers.year.max())}",
                help=f"95% of papers; the oldest reference in the map is from {int(papers.year.min())}. "
                     f"{int(papers.year.max())} is a partial year.")
    c[2].metric("Problem communities", f"{(cl.kind == 'stable').sum()} + {(cl.kind == 'hub').sum()} hubs")
    if "external_reference_share" in summ:
        c[3].metric("References outside the map", f"{summ['external_reference_share']:.0%}")
        c[4].metric("External works tracked", f"{summ['external_works']:,}")

    with st.expander("What this map covers, and what it cannot tell you", expanded=False):
        routes = papers.route.value_counts()
        st.markdown(
            "The map grows from one or a few seed papers through their citation graph: references, papers citing "
            "them, their authors' newest work, and semantically close papers in the same topics "
            f"({', '.join(f'{k} {v:,}' for k, v in routes.items())} papers by route). It is a **map of one field "
            "as reached from its seed**, not a sample of all of science, and parts of it were fetched in "
            "citation order.\n\n"
            "- **Reliable here:** what to read, which papers and ideas a community is likely to take up next, "
            "and which communities are likely to be warm to a paper (all backtested; see *How far to trust each "
            "view* below).\n"
            "- **Read with care:** whether a topic is growing or shrinking. Several topics reverse direction once "
            "changes in how the map was sampled are separated out.\n"
            f"- **Not covered:** most references point outside the map; the {summ.get('external_works', 0):,} "
            "external works cited at least 5 times are tracked separately (below). Nothing here predicts "
            "citations from all of science, or where a brand-new community will form.")

    # ---- communities
    st.subheader("The communities")
    lab = q("""SELECT pc.cluster_id, p.year FROM paper_cluster pc JOIN papers p USING (paper_id)
               WHERE pc.run_id = ? AND pc.facet = 'problem' AND pc.alpha = 0.5 AND pc.resolution = 'medium'""", (run,))
    rec = lab[lab.year.between(T - 2, T)].groupby("cluster_id").size()
    prev = lab[lab.year.between(T - 5, T - 3)].groupby("cluster_id").size()
    n_rec, n_prev = max(lab.year.between(T - 2, T).sum(), 1), max(lab.year.between(T - 5, T - 3).sum(), 1)
    t = cl.assign(recent=cl.cluster_id.map(rec).fillna(0).astype(int),
                  share_now=cl.cluster_id.map(rec).fillna(0) / n_rec,
                  share_before=cl.cluster_id.map(prev).fillna(0) / n_prev)
    t["change"] = t.share_now - t.share_before
    t["community"] = t.label + np.where(t.kind == "hub", " ◇", "")
    t = t.sort_values("size", ascending=False)
    st.dataframe(t[["community", "size", "recent", "share_now", "change"]], hide_index=True, width="stretch", height=280,
                 column_config={"size": "papers", "recent": st.column_config.NumberColumn(f"papers {T - 2}–{T}"),
                                "share_now": st.column_config.NumberColumn(f"share {T - 2}–{T}", format="percent"),
                                "change": st.column_config.NumberColumn("vs 3 years before", format="percent")})
    st.caption("Problem communities (◇ = hub: a large, coherent body of work with a shifting boundary). The share "
               "change describes this map as sampled; it is not a forecast. Open **Map & grid** to explore them.")

    # ---- where ideas come from
    im = _load(run, "import_map.parquet")
    if im is not None and len(im):
        st.subheader("Where this field borrows from")
        top = im.head(10).iloc[::-1]
        fig = go.Figure([go.Bar(y=top.subfield, x=top.share_early, orientation="h", name=top.early_years.iloc[0],
                                marker_color="#c9cdd2"),
                         go.Bar(y=top.subfield, x=top.share_recent, orientation="h", name=top.recent_years.iloc[0],
                                marker_color="#e8590c")])
        fig.update_layout(barmode="group", height=360, margin=dict(l=0, r=0, t=10, b=0), xaxis_tickformat=".0%",
                          legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig, width="stretch", key="import_map")
        st.caption("Share of the citations to works outside the map, by the cited work's OpenAlex subfield (works "
                   "cited at least 5 times). Experiment 6: the field imports more of what fits where its questions "
                   "are moving and what trends across science.")

    # ---- now
    L = _load(run, "live_imports.parquet")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Being imported now")
        if L is not None:
            o = L[L.outside_field & (L.year >= T - 10)].head(8)
            st.dataframe(pd.DataFrame({"work": o.title, "subfield": o.subfield, "citing papers here": o.recent_citers,
                                       "closest community": o.cluster_label}), hide_index=True, width="stretch")
            st.caption(f"Recent works from outside the field that this field cites more and more ({T - 2}–{T}), "
                       "ranked by the experiment-6 model. More in **Map & grid › Bridges › From outside the field**.")
        else:
            st.caption("Not built: run `landscape external` and `landscape portfolio`.")
    with c2:
        st.subheader("Read now")
        ew = q("""SELECT subject, rank FROM signals WHERE run_id = ? AND signal_type = 'early_work' AND variant = 'all'
                  ORDER BY rank LIMIT 8""", (run,))
        if len(ew):
            st.dataframe(pd.DataFrame({"paper": [json.loads(s).get("title") for s in ew.subject]}), hide_index=True,
                         width="stretch")
            st.caption("Recent papers this field is most likely to read and cite next (experiment 2: AUC 0.82). A "
                       "reading list, not a quality ranking. More in **People › Early work**.")

    # ---- opportunities
    st.subheader("Opportunities to inspect")
    tabs = st.tabs(["Papers to bring in (hidden gems)", "Limitation → innovation", "Method transfer",
                    "Underconnected communities"])
    with tabs[0]:
        g = q("""SELECT subject, score FROM signals WHERE run_id = ? AND signal_type = 'bridge_paper_gem' ORDER BY score DESC
                 LIMIT 10""", (run,))
        if len(g):
            s = [json.loads(x) for x in g.subject]
            st.dataframe(pd.DataFrame({"paper": [x.get("title") for x in s], "for community": [x.get("label") for x in s]}),
                         hide_index=True, width="stretch")
            st.caption("Papers cited at most once that a community is likely to take up (experiment 1: long-tail AUC "
                       "0.84, where popularity is at chance). Full lists in **Bridges › Papers to bring in**.")
    with tabs[1]:
        li = q("""SELECT subject, judge_score, judge_rationale FROM signals WHERE run_id = ? AND signal_type = 'bridge_lim_inn'
                  AND judge_score IS NOT NULL ORDER BY judge_score DESC, rank LIMIT 6""", (run,))
        if len(li):
            s = [json.loads(x) for x in li.subject]
            st.dataframe(pd.DataFrame({"limitations in": [x.get("from_label") for x in s],
                                       "innovations in": [x.get("to_label") for x in s], "judge": li.judge_score,
                                       "why": li.judge_rationale}), hide_index=True, width="stretch")
            st.caption("Communities whose stated limitations match innovations elsewhere, with few citations between "
                       "them (Phase 8: about 2.4x random). Only a minority of abstracts state limitations: check the "
                       "evidence in **Bridges › Cluster pairs**.")
    with tabs[2]:
        mt = q("""SELECT subject, judge_score FROM signals WHERE run_id = ? AND signal_type = 'bridge_method_transfer'
                  ORDER BY rank LIMIT 6""", (run,))
        if len(mt):
            s = [json.loads(x) for x in mt.subject]
            st.dataframe(pd.DataFrame({"method": [x.get("method_label") for x in s],
                                       "problem": [x.get("problem_label") for x in s], "judge": mt.judge_score}),
                         hide_index=True, width="stretch")
            st.caption("Methods not yet used for a problem, though they are used for similar problems (experiment 4: "
                       "AUC 0.70-0.80). Shown as diamonds in the **Map & grid** grid.")
    with tabs[3]:
        from .research_lab import lab_dir
        op = lab_dir(run) / "opportunities.parquet"
        if op.exists():
            o = pd.read_parquet(op).sort_values("score", ascending=False).head(6)
            st.dataframe(pd.DataFrame({"from": o.source_label, "to": o.target_label, "citations seen": o.observed,
                                       "expected": o.expected.round(1)}), hide_index=True, width="stretch")
            st.caption("Semantically close communities that cite each other less than their size predicts (Phase 8c). "
                       "Hypotheses to review, not confirmed gaps. Details in **Research lab**.")
        else:
            st.caption("Run `landscape research-lab` to add these.")

    # ---- second waves
    SW = _load(run, "second_waves.parquet")
    if SW is not None and len(SW):
        st.subheader("Second waves: ideas reaching new audiences")
        _how(SECOND_WAVES_HOW)
        tabs = st.tabs(["Papers in the map", "From outside the field", "Methods"])
        for tab, kind in zip(tabs, ("paper", "external", "method")):
            with tab:
                part = SW[SW.kind == kind]
                if part.empty:
                    st.caption("None this year.")
                else:
                    _second_waves_table(part, q, f"sw_{kind}")

    # ---- coverage gaps
    if L is not None:
        st.subheader("Coverage gaps")
        gap = L[~L.outside_field].sort_values("recent_citers", ascending=False).head(8)
        st.dataframe(pd.DataFrame({"work": gap.title, "year": gap.year, "citing papers here": gap.recent_citers,
                                   "community": gap.cluster_label}), hide_index=True, width="stretch")
        st.caption("Works this field cites heavily that are not in the map: often core papers the seed's citation "
                   "graph never reached. Adding them (a new seed or a corpus expansion) makes every view more complete.")

    # ---- trust
    st.subheader("How far to trust each view")
    st.markdown(
        "| View | Verdict on the pilot | Backed by |\n| --- | --- | --- |\n"
        "| My paper › warm communities | **Use** | experiment 10: a top-3 community cites the paper in 69% of cases (25% at random) |\n"
        "| Bridges › Papers to bring in, hidden gems | **Use** | experiment 1 and the 8c chronological test (19.5% vs 13.6% for popularity in the top 10) |\n"
        "| People › Early work | **Use** as a reading list | experiment 2: AUC 0.82 |\n"
        "| Bridges › From outside the field | **Use** as a review list | experiment 6: modest gains (+0.025 AUC) |\n"
        "| Second waves (here and on My paper) | **Use** as a watchlist | experiments 4, 6, 9, 10: diffusion already under way, not advance warning |\n"
        "| Bridges › Cluster pairs, grid method transfer | **Use with care** | Phase 8, experiment 4 |\n"
        "| Status badges, Ranking, Rising pockets | **Describe only** | direction is sensitive to sampling |\n"
        "| People, institutions | **Describe only** | not better than popularity at predicting citations |\n\n"
        "Every result table, including the experiments that did not work, is in **Experiments**.")


# ================================================================== My paper

def _paper_record(run: str, q, pid: str) -> dict | None:
    """Corpus paper: warmth rows from the portfolio, vectors from the store."""
    W = _load(run, "warmth.parquet")
    if W is None:
        return None
    rows = W[W.paper_id == pid]
    if rows.empty:
        return None
    ver = _embedding_version(q, run)
    vec = {}
    for f in ("problem", "method", "innovation", "limitations", "abstract"):
        pos, v = _corpus_vectors(f, ver, 768)
        if pid in pos:
            vec[f] = v[pos[pid]]
    meta = q("SELECT title, year, doi FROM papers WHERE paper_id = ?", (pid,)).iloc[0]
    facets = dict(q("SELECT facet, text FROM facets WHERE paper_id = ?", (pid,)).itertuples(index=False))
    refs = q("SELECT cited_id FROM refs WHERE citing_id = ?", (pid,)).cited_id.tolist()
    return {"paper_id": pid, "title": meta.title, "year": meta.year, "doi": meta.doi, "in_corpus": True,
            "facets": facets, "vectors": vec, "warmth": rows.to_dict("records"), "refs": refs}


def _why(row: dict, spec: dict) -> str:
    z = {f: (row[f] - m) / s * c for f, m, s, c in zip(spec["features"], spec["mean"], spec["scale"], spec["coef"])}
    top = [REASONS[f] for f, v in sorted(z.items(), key=lambda kv: -kv[1]) if v > 0][:2]
    return "; ".join(top) or "no strong signal"


def render_my_paper(run: str, q, paper_card, record) -> None:
    st.title("My paper")
    if not (portfolio.out_dir(run) / "warmth.parquet").exists():
        st.info(f"Build the portfolio first: `uv run landscape portfolio --run-id {run}`.")
        return
    analysed = sorted((portfolio.out_dir(run) / "papers").glob("*.json")) if (portfolio.out_dir(run) / "papers").exists() else []
    src = st.radio("Paper", ["Find a paper in the map", "A paper analysed by DOI"], horizontal=True, key="mp_src")
    rec = None
    if src.startswith("Find"):
        term = st.text_input("Title contains", key="mp_term", placeholder="e.g. Wasserstein")
        if term:
            hits = q("""SELECT p.paper_id, p.year, p.title FROM papers p JOIN provenance v USING (paper_id)
                        WHERE v.run_id = ? AND p.title ILIKE ? ORDER BY p.year DESC LIMIT 50""", (run, f"%{term}%"))
            if hits.empty:
                st.caption("No title matches.")
            else:
                pid = st.selectbox("Match", hits.paper_id.tolist(), key="mp_pick",
                                   format_func=lambda x: f"{int(hits.set_index('paper_id').year[x])} · "
                                                         f"{hits.set_index('paper_id').title[x][:110]}")
                rec = _paper_record(run, q, pid)
                if rec is None:
                    st.info("This paper has no extracted innovation statement, so warmth cannot be scored.")
    else:
        st.caption(f"Analyse any paper: `uv run landscape my-paper --run-id {run} --doi <DOI>` (one LLM call, less than "
                   "a cent, unless the paper is already in the map). Close the explorer first.")
        if analysed:
            pick = st.selectbox("Analysed papers", analysed, format_func=lambda p: json.loads(p.read_text())["title"][:120],
                                key="mp_doi")
            rec = json.loads(pick.read_text())
            rec["vectors"] = {k: np.array(v, dtype=np.float32) for k, v in rec["vectors"].items()}
    if rec is None:
        return
    _render_paper(run, q, paper_card, record, rec)


def _render_paper(run, q, paper_card, record, rec) -> None:
    W = pd.DataFrame(rec["warmth"])
    labels = q("""SELECT cluster_id, label, kind FROM clusters WHERE run_id = ? AND facet = 'problem' AND alpha = 0.5
                  AND resolution = 'medium'""", (run,)).set_index("cluster_id")
    W["community"] = W.cluster_id.map(labels.label) + np.where(W.cluster_id.map(labels.kind) == "hub", " ◇", "")
    home = W[W.home].community.tolist()
    st.markdown(f"### {rec['title']}")
    f = rec.get("facets") or {}
    st.caption(f"{rec.get('year') or ''} · {'in the map' if rec.get('in_corpus') else 'analysed by DOI'}"
               + (f" · {'home community' if rec.get('in_corpus') else 'anchor community (cites it most)'}: **{home[0]}**"
                  if home else ""))
    with st.expander("What the paper says (extracted)", expanded=False):
        for k in ("problem", "method", "innovation", "limitations"):
            if f.get(k):
                st.markdown(f"**{k.capitalize()}:** {f[k]}")

    # ---- 1. warm communities
    st.subheader("Communities most likely to be warm to it")
    cand = W[W["rank"].notna()].sort_values("rank").copy()
    lo, hi = cand.warmth.min(), cand.warmth.max()
    cand["warmth_rel"] = (cand.warmth - lo) / (hi - lo) if hi > lo else 1.0
    spec = portfolio.model_spec(run, "community_warmth")
    cand["why"] = [_why(r, spec) for r in cand.to_dict("records")]
    ev = st.dataframe(pd.DataFrame({"rank": cand["rank"].astype(int), "community": cand.community,
                                    "warmth": cand.warmth_rel, "content fit": cand.content_fit.round(3), "why": cand.why}),
                      hide_index=True, width="stretch", height=320, on_select="rerun", selection_mode="single-row",
                      key=f"mp_warm_{rec['paper_id']}",
                      column_config={"warmth": st.column_config.ProgressColumn("warmth", min_value=0, max_value=1)})
    already = W[W.already_citing & ~W.home].community.tolist()
    st.caption((f"Already citing it: {', '.join(already)}. " if already else "") +
               "Ranked by the community-warmth model (experiment 10): among communities that do not cite a paper yet, "
               "a top-3 community went on to cite it within three years for 69% of papers (25% at random). "
               "**Content fit** is the similarity of your innovation to the community's problems: the best single "
               "guide to which communities' later work moves towards an idea, cited or not.")
    _how("**warmth** is relative within this paper (1 = the warmest). **why** names the strongest contributions: "
         "shared references with the community's papers, how close its recent methods and innovations are to your "
         "innovation, how related its problems are, and whether its stated limitations match. Select a community to "
         "see its recent papers closest to yours: who to read, cite or contact.")
    if ev and ev.selection.rows:
        c = cand.iloc[ev.selection.rows[0]].cluster_id
        inn = rec["vectors"].get("innovation")
        lab = q("""SELECT pc.paper_id, p.year, p.title FROM paper_cluster pc JOIN papers p USING (paper_id)
                   WHERE pc.run_id = ? AND pc.cluster_id = ? ORDER BY p.year DESC""", (run, c))
        ver = _embedding_version(q, run)
        pos, v = _corpus_vectors("innovation", ver, 768)
        lab = lab[lab.paper_id.isin(pos) & (lab.year >= lab.year.max() - 3)].copy()
        if inn is not None and len(lab):
            lab["closeness"] = v[[pos[x] for x in lab.paper_id]] @ np.asarray(inn)
            st.markdown("**Its recent papers closest to yours**")
            st.dataframe(lab.sort_values("closeness", ascending=False).head(6)[["year", "title"]], hide_index=True,
                         width="stretch")

    inn = rec["vectors"].get("innovation")
    prob = rec["vectors"].get("problem")
    ver = _embedding_version(q, run)
    c1, c2 = st.columns(2)
    # ---- 2. whose problems it may answer
    with c1:
        st.subheader("Whose stated problems it may answer")
        if inn is not None:
            pos, Lv = _corpus_vectors("limitations", ver, 768)
            ids = list(pos)
            s = Lv @ np.asarray(inn)
            top = [ids[k] for k in np.argsort(-s)[:8] if ids[k] != rec["paper_id"]][:6]
            if top:
                t = q(f"""SELECT p.paper_id, p.year, p.title, f.text FROM papers p JOIN facets f USING (paper_id)
                          WHERE f.facet = 'limitations' AND p.paper_id IN ({",".join("?" * len(top))})""", tuple(top))
                t = t.set_index("paper_id").loc[[x for x in top if x in set(t.paper_id)]].reset_index()
                st.dataframe(pd.DataFrame({"limitation": t.text, "paper": t.title, "year": t.year}), hide_index=True,
                             width="stretch")
                st.caption("Limitations stated in other papers' abstracts that are closest to your innovation. Few "
                           "abstracts state limitations, and closeness of wording does not establish that the "
                           "assumptions are compatible: candidates to read.")
    # ---- 3. ideas from outside the field
    with c2:
        st.subheader("Ideas from outside the field that fit")
        L = _load(run, "live_imports.parquet")
        if L is not None and (prob is not None or inn is not None):
            q256 = embed.truncate(np.asarray(prob if prob is not None else inn)[None, :], portfolio.DIM)[0]
            best = {}
            for facet in ("innovation", "method", "abstract"):
                pos, v = _ext_vectors(run, facet)
                for x in L.paper_id:
                    if x in pos and x not in best:
                        best[x] = float(v[pos[x]] @ q256)
            o = L[L.outside_field].assign(fit=L.paper_id.map(best)).dropna(subset=["fit"])
            o = o[~o.paper_id.isin(set(rec.get("refs") or []))]
            o = o.sort_values("fit", ascending=False).head(6)
            st.dataframe(pd.DataFrame({"work": o.title, "subfield": o.subfield, "fit": o.fit.round(3),
                                       "citing papers here": o.recent_citers}), hide_index=True, width="stretch")
            st.caption("Works from outside the field that this field is importing (experiment 6), ranked by closeness "
                       "to your problem; works you already cite are left out. Papers that borrow what is trending, "
                       "from nearby, draw more attention in-field (experiment 6, +0.016 AUC).")

    # ---- 4. papers to bring in (home community's list, closest to this paper first)
    st.subheader("Papers to bring in")
    hc = W[W.home].cluster_id.tolist()
    if hc:
        b = q("""SELECT signal_type, subject, score FROM signals WHERE run_id = ?
                 AND signal_type IN ('bridge_paper', 'bridge_paper_gem')""", (run,))
        b["s"] = b.subject.map(json.loads)
        b = b[b.s.map(lambda x: x["cluster_id"]) == hc[0]]
        b["paper_id"] = b.s.map(lambda x: x["paper_id"])
        b = b[~b.paper_id.isin(set(rec.get("refs") or [])) & (b.paper_id != rec["paper_id"])]
        pos, v = _corpus_vectors("innovation", ver, 768)
        if prob is not None:
            b["fit"] = [float(v[pos[x]] @ np.asarray(prob)) if x in pos else np.nan for x in b.paper_id]
        b = b.drop_duplicates("paper_id").sort_values("fit" if "fit" in b else "score", ascending=False).head(8)
        st.dataframe(pd.DataFrame({"paper": b.s.map(lambda x: x.get("title")),
                                   "list": b.signal_type.map({"bridge_paper": "likely", "bridge_paper_gem": "hidden gem"}),
                                   "fit": b.get("fit", pd.Series(dtype=float)).round(3)}), hide_index=True, width="stretch")
        st.caption("Innovations your home community is likely to take up but does not cite yet (experiment 1), closest "
                   "to your problem first; papers you already cite are left out.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Methods new to your problem")
        if hc:
            mt = q("""SELECT subject, judge_score FROM signals WHERE run_id = ? AND signal_type = 'bridge_method_transfer'
                      ORDER BY rank""", (run,))
            s = [json.loads(x) for x in mt.subject]
            m = [(x.get("method_label"), j) for x, j in zip(s, mt.judge_score) if x.get("problem_cluster") == hc[0]][:6]
            st.dataframe(pd.DataFrame(m, columns=["method", "judge"]), hide_index=True, width="stretch")
            st.caption("Method communities not yet used for your home community's problems, though similar problems use "
                       "them (experiment 4).")
    with c4:
        st.subheader("Early work nearby")
        ew = q("""SELECT subject FROM signals WHERE run_id = ? AND signal_type = 'early_work' AND variant = 'all'""", (run,))
        if len(ew) and rec["vectors"].get("abstract") is not None:
            s = [json.loads(x) for x in ew.subject]
            pos, v = _corpus_vectors("abstract", ver, 768)
            a = np.asarray(rec["vectors"]["abstract"])
            sc = [(x.get("title"), float(v[pos[x["paper_id"]]] @ a)) for x in s if x["paper_id"] in pos
                  and x["paper_id"] != rec["paper_id"]]
            st.dataframe(pd.DataFrame(sorted(sc, key=lambda t: -t[1])[:6], columns=["paper", "closeness"]).round(3),
                         hide_index=True, width="stretch")
            st.caption("Recent papers this field is likely to read next (experiment 2), closest to yours.")

    # ---- second waves near this paper
    SW = _load(run, "second_waves.parquet")
    near = set(W[W.home].community.str.replace(" ◇", "", regex=False)) | set(
        cand.head(3).community.str.replace(" ◇", "", regex=False))
    if SW is not None and len(SW) and near:
        hit = SW[[bool(near & (set(a) | set(b) | set(c))) for a, b, c in
                  zip(SW.new_audiences, SW.likely_next, SW.get("quiet_uptake", pd.Series([[]] * len(SW))))]]
        hit = hit[hit.idea_id != rec["paper_id"]]
        if len(hit):
            st.subheader("Second waves near you")
            st.caption("Ideas reaching your home or warmest communities as new audiences, quiet uptake or likely next. "
                       + SECOND_WAVES_HOW.split(" Select")[0])
            _second_waves_table(hit.sort_values(["n_new", "momentum"], ascending=False), q, f"mp_sw_{rec['paper_id']}", n=8)

    # ---- people in the warm communities (descriptive)
    st.subheader("Active in your warmest communities")
    top3 = cand.head(3).cluster_id.tolist()
    if top3:
        y1 = int(q("SELECT max(p.year) y FROM papers p JOIN provenance v USING (paper_id) WHERE v.run_id = ?", (run,)).y.iloc[0])
        ppl = q(f"""SELECT a.display_name AS name, pc.cluster_id, count(DISTINCT s.paper_id) AS papers
                    FROM paper_cluster pc JOIN papers p USING (paper_id) JOIN authorships s ON s.paper_id = pc.paper_id
                    JOIN authors a ON a.author_id = s.author_id
                    WHERE pc.run_id = ? AND pc.facet = 'problem' AND pc.alpha = 0.5 AND pc.resolution = 'medium'
                      AND pc.cluster_id IN ({",".join("?" * len(top3))}) AND p.year >= ?
                    GROUP BY ALL ORDER BY papers DESC LIMIT 12""", (run, *top3, y1 - 3))
        ppl["community"] = ppl.cluster_id.map(labels.label)
        # insider marker: a track record in the map before the recent window (experiments 5 and 11: insiders'
        # papers are cited more by others; big names stepping in do not start waves)
        names = tuple(ppl.name)
        if names:
            hist = q(f"""SELECT a.display_name AS name, count(DISTINCT s.paper_id) AS earlier
                         FROM authors a JOIN authorships s USING (author_id) JOIN papers p ON p.paper_id = s.paper_id
                         JOIN provenance v ON v.paper_id = p.paper_id AND v.run_id = ?
                         WHERE a.display_name IN ({",".join("?" * len(names))}) AND p.year < ? GROUP BY 1""",
                     (run, *names, y1 - 3)).set_index("name").earlier
            ppl["earlier papers in the map"] = ppl.name.map(hist).fillna(0).astype(int)
            ppl["track record"] = np.where(ppl["earlier papers in the map"] >= 2, "insider", "newer")
        st.dataframe(ppl[["name", "community", "papers", "earlier papers in the map", "track record"]], hide_index=True,
                     width="stretch", column_config={"papers": st.column_config.NumberColumn(f"papers since {y1 - 3}")})
        st.caption(f"Authors with the most papers since {y1 - 3} in your three warmest communities. **Insider** = at "
                   "least 2 earlier papers in the map: insiders' papers draw more citations from others (experiments 5 "
                   "and 11), while big names stepping somewhere new do not start a follow-up wave (experiment 11). "
                   "Descriptive decision support for whom to read or contact, not a ranking of people.")
    note = st.text_input("Note on these suggestions", key=f"mp_note_{rec['paper_id']}")
    cc = st.columns(3)
    for col, act in zip(cc, ["useful", "obvious", "wrong"]):
        if col.button(act, key=f"mp_{act}_{rec['paper_id']}"):
            record("my_paper", {"paper_id": rec["paper_id"], "warm_top3": top3}, act, note, alpha={"problem": 0.5})
            st.success("Saved to review_actions.")
