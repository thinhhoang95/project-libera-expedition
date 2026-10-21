"""Explorer views for the offline research lab; every candidate opens its paper evidence."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .research_lab import lab_dir


@st.cache_data(show_spinner=False)
def _load(path, modified):
    return pd.read_csv(path) if str(path).endswith(".csv") else pd.read_parquet(path)


def render_lab(run, q, paper_card, record):
    path = lab_dir(run)
    summary_path = path / "summary.json"
    if not summary_path.exists():
        st.info(f"Build the offline review artifacts first: landscape research-lab --run-id {run}")
        return
    audit = json.loads(summary_path.read_text())

    def read(name):
        file = path / name
        return _load(str(file), file.stat().st_mtime_ns)

    def evidence(papers):
        for pid in papers:
            paper_card(pid)

    st.title("Research lab")
    st.caption(f"Offline experiments · snapshot {audit['as_of']} · completed publication years through "
               f"{audit['complete_through']} · corpus selected retrospectively")
    st.write("Inspect how sampling changes the story, compare bridge lists, and explore underconnected "
             "communities. These are candidates for expert review; similarity alone does not show that a problem is solved.")
    columns = st.columns(4)
    for col, label, value in zip(columns,
            ["Unique works", "Explicit limitations", "Trend sign reversals", "Underconnected pairs"],
            [audit["unique_works"], audit["explicit_limitations"],
             f"{audit['supported_direction_sign_flips']} / {audit['supported_direction_clusters']}", audit["flow_opportunities"]]):
        col.metric(label, value)
    tabs = st.tabs(["Dynamics", "Paper bridges", "Underconnected communities", "People by topic", "Experiments"])

    with tabs[0]:
        routes = read("routes.parquet")
        year = audit["complete_through"]
        recent = routes[routes.year.between(year - 7, year + 1)].copy()
        recent["share"] = recent.n / recent.groupby("year").n.transform("sum")
        fig = px.bar(recent, x="year", y="share", color="route", custom_data=["n"],
                     labels={"share": "Sampling route share", "year": "Publication year"})
        fig.update_layout(barmode="stack", height=280, yaxis_tickformat=".0%")
        fig.update_traces(hovertemplate="%{x}: %{y:.1%} · %{customdata[0]} records")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{year + 1} is partial and is excluded from the comparison below. Provenance records "
                   "the first route that found each paper; route shares are not inclusion probabilities.")
        trends = read("trends.parquet").sort_values("composition_change", key=abs, ascending=False)
        if not st.checkbox("Include communities with fewer than 20 works in either period", key="lab_sparse_trends"):
            trends = trends[(trends.before_n >= 20) & (trends.after_n >= 20)]
        st.write(f"Compare {year - 5}–{year - 3} with {year - 2}–{year}. The observed share change splits exactly "
                 "into change within routes and change from the route mix. A sign reversal means the direction is sensitive to sampling.")
        display = trends[["label", "kind", "raw_change", "within_route_change", "composition_change", "sign_flip",
                          "before_n", "after_n", "route_sign_agreement"]]
        event = st.dataframe(display, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key="lab_trends",
                             column_config={k: st.column_config.NumberColumn(format="%.4f") for k in
                                            ["raw_change", "within_route_change", "composition_change"]})
        if event.selection.rows:
            row = trends.iloc[event.selection.rows[0]]
            series = read("timeseries.parquet")
            series = series[series.cluster_id == row.cluster_id]
            fig = px.line(series, x="year", y="share", color="route", markers=True,
                          custom_data=["n", "denominator"], title=row.label)
            fig.update_layout(height=320, yaxis_tickformat=".1%")
            fig.update_traces(hovertemplate="%{x}: %{y:.1%} · %{customdata[0]} / %{customdata[1]} works")
            st.plotly_chart(fig, width="stretch")
            st.caption("Very small route denominators make individual trajectories noisy. "
                       "This sensitivity analysis cannot recover an unbiased field trend.")
            with st.expander("Recent evidence papers"):
                evidence(row.evidence)

    with tabs[1]:
        bridges = read("live_bridges.parquet")
        labels = q("SELECT cluster_id, label, kind FROM clusters WHERE run_id = ?", (run,)).set_index("cluster_id")
        clusters = sorted(bridges.X.unique())
        cluster = st.selectbox("Problem community", clusters, key="lab_bridge_cluster",
                               format_func=lambda c, labels=labels: f"{labels.loc[c, 'label']} · {labels.loc[c, 'kind']} · {c.rsplit(':', 1)[-1]}")
        mode = st.radio("Review list", ["Broader methods", "Likely adoption", "Contrastive experiment"],
                        horizontal=True, key="lab_bridge_mode")
        group = bridges[bridges.X == cluster]
        chosen = group[group.diverse_pick] if mode == "Broader methods" else (
            group[group.baseline_pick] if mode == "Likely adoption" else group.nlargest(10, "contrastive_model"))
        chosen = chosen.sort_values("contrastive_model" if mode == "Contrastive experiment" else "baseline_model", ascending=False)
        st.caption("Scores rank likely citation adoption, not demonstrated usefulness or scientific opportunity. "
                   "Broader methods limits each list to three papers per method community. Aliases already cited by this community are excluded.")
        shown = chosen[["title", "pop", "age", "method", "n_lim", "baseline_model", "contrastive_model"]].copy()
        shown["method"] = shown.method.map(labels.label).fillna("Unassigned")
        ev = st.dataframe(shown, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="lab_bridge_rows")
        row = chosen.iloc[ev.selection.rows[0] if ev.selection.rows else 0] if len(chosen) else None
        if row is not None:
            if isinstance(row.limitation_text, str):
                st.markdown("**Stated limitation in the target community**")
                st.write(row.limitation_text)
                st.markdown("**Candidate innovation**")
                st.write(row.innovation_text)
                st.caption("This match is a hypothesis. Read the papers to check assumptions, chronology and compatibility.")
            else:
                st.info("This community has no retained explicit limitation for this candidate. The score uses problem similarity as fallback.")
            evidence([row.i] + ([row.best_limitation] if isinstance(row.best_limitation, str) else []))
            with st.form("lab_bridge_judgment"):
                judgment = st.radio("Expert judgment", ["useful and non-obvious", "obvious", "wrong", "uncertain"], index=3, horizontal=True)
                note = st.text_input("Why? Check assumptions and whether the innovation addresses this limitation.")
                if st.form_submit_button("Save judgment"):
                    record("research_lab_bridge", {"cluster_id": cluster, "paper_id": row.i,
                           "limitation_id": row.best_limitation, "list": mode, "snapshot": audit["as_of"]},
                           judgment, note, alpha={"problem": 0.5, "method": 0.5})
                    st.success("Saved.")

    with tabs[2]:
        opportunities = read("opportunities.parquet")
        st.write("These communities are semantically near but cite each other less than a degree-based null model expects. "
                 "The null accounts for each community's outgoing and incoming citation volume. Counts use the last three completed publication years.")
        st.caption("A deficit is descriptive. Topic boundaries, homophily and incompatible assumptions can explain it; "
                   "these pairs have not been validated as research opportunities.")
        if len(opportunities):
            opportunities = opportunities.copy()
            hub_pairs = (opportunities.source_kind == "hub") | (opportunities.target_kind == "hub")
            group_kind = st.radio("Community pair group", ["Stable communities", "Includes a hub"], horizontal=True)
            opportunities = opportunities[hub_pairs if group_kind == "Includes a hub" else ~hub_pairs]
        if len(opportunities):
            opportunities["observed / expected"] = opportunities.observed / opportunities.expected
            opportunities["pair"] = opportunities.source_label + " → " + opportunities.target_label
            fig = px.scatter(opportunities, x="semantic_similarity", y="observed / expected", size="expected",
                             color="score", hover_name="pair", custom_data=["observed", "expected"])
            fig.update_layout(height=340)
            st.plotly_chart(fig, width="stretch")
            ev = st.dataframe(opportunities[["source_label", "target_label", "source_kind", "target_kind", "observed",
                                            "expected", "semantic_similarity", "score"]], hide_index=True, width="stretch",
                              on_select="rerun", selection_mode="single-row", key="lab_opportunities")
            row = opportunities.iloc[ev.selection.rows[0] if ev.selection.rows else 0]
            st.markdown(f"**{row.source_label} → {row.target_label}**")
            st.caption(f"Observed {row.observed} citations; null expectation {row.expected:.1f}. "
                       "Evidence below is the nearest pair of problem facets, not a confirmed solution transfer.")
            evidence(row.evidence)
            with st.form("lab_opportunity_judgment"):
                judgment = st.radio("Opportunity judgment", ["useful and non-obvious", "obvious", "wrong", "uncertain"], index=3, horizontal=True)
                note = st.text_input("What makes this bridge useful or inappropriate?")
                if st.form_submit_button("Save opportunity judgment"):
                    record("research_lab_opportunity", {"source": row.source, "target": row.target,
                           "snapshot": audit["as_of"]}, judgment, note, alpha={"problem": 0.3})
                    st.success("Saved.")
        else:
            st.info("No pairs satisfy the minimum exposure and semantic-nearness gates.")

    with tabs[3]:
        people = read("people.parquet")
        labels = people.drop_duplicates("cluster_id").set_index("cluster_id").label
        topic = st.selectbox("Topic", labels.index.tolist(), key="lab_people_topic",
                             format_func=lambda c, labels=labels: f"{labels[c]} · {c.rsplit(':', 1)[-1]}")
        role = st.selectbox("Inspect role", ["Recent external adoption", "Recent contribution", "Cross-community participation"])
        feature = {"Recent external adoption": "external_adoption", "Recent contribution": "recent_fractional_works",
                   "Cross-community participation": "participation"}[role]
        group = people[people.cluster_id == topic].sort_values([feature, "recent_fractional_works"], ascending=False).head(20)
        st.caption("Internal topic-specific roles. Works are deduplicated, author credit is fractional, and external adoption "
                   "excludes citations from teams sharing any author. Counts reflect this sampled corpus. Participation describes breadth, not influence.")
        ev = st.dataframe(group[["name", "recent_fractional_works", "external_adoption", "participation", "topics", "first_topic_year"]],
                          hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="lab_people")
        if len(group):
            row = group.iloc[ev.selection.rows[0] if ev.selection.rows else 0]
            with st.expander(f"Evidence for {row['name']}"):
                evidence(row.evidence)

    with tabs[4]:
        metrics = read("bridge_metrics.csv")
        protocol = st.selectbox("Evaluation protocol", metrics.protocol.unique().tolist())
        scores = metrics[metrics.protocol == protocol]
        summary = scores.groupby("ranking")[["p@10", "methods@10", "auc", "tail_auc"]].mean().reset_index()
        chosen = summary[summary.ranking.isin(["lpop", "baseline_model", "contrastive_model", "diverse_baseline"])]
        fig = go.Figure()
        fig.add_bar(x=chosen.ranking, y=chosen["p@10"], name="Precision@10")
        fig.update_layout(height=300, yaxis_title="Future adoption precision@10", yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(summary.round(4), hide_index=True, width="stretch")
        st.caption(audit["protocols"][protocol])
        st.write("All rankings share the same alias-aware candidate pool. Centering and CSLS remove common semantic "
                 "directions and generic matches. The contrastive model adds these features to the existing five-feature model. "
                 "No experimental model replaced production weights.")
        with st.expander("Per-cutoff results and geometry audit"):
            st.dataframe(scores.round(4), hide_index=True, width="stretch")
            st.dataframe(read("geometry.parquet").round(3), hide_index=True, width="stretch")
        for limitation in audit["limitations"]:
            st.caption(limitation)
