"""Explorer view: every signal experiment's results for the selected run, adopted or not.

Results come from one corpus. Another field can give different answers, so nothing here is
hidden because it lost on the pilot: each table is shown with how to read it and how to
re-run it on another run (see src/landscape/experiments.py).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import experiments

# what was decided on the pilot; another run should be judged on its own tables
PILOT_DECISIONS = {
    "exp1_paper_bridges": "adopted: signals bridge_paper / bridge_paper_gem (Bridges tab, Papers to bring in)",
    "exp2_early_work": "adopted: signal early_work (People tab)",
    "exp3_micro_topics": "burstiness dropped; local momentum adopted lightly (map layer Rising pockets)",
    "exp4_method_transfer": "adopted: item CF for bridge_method_transfer (grid diamonds)",
    "exp5_beyond_rich_get_richer": "findings only; not used by any signal yet",
}
ORDER = list(PILOT_DECISIONS)
SCRIPTS = {
    "exp1_paper_bridges": ["exp1_paper_bridges.py", "exp1_evaluate.py"],
    "exp2_early_work": ["exp2_early_work.py"],
    "exp3_micro_topics": ["exp3_micro_topics.py"],
    "exp4_method_transfer": ["exp4_method_transfer.py"],
    "exp5_beyond_rich_get_richer": ["fetch_counts_by_year.py", "exp5_features.py", "exp5_evaluate.py [logit] [hgb]"],
}


@st.cache_data(show_spinner=False)
def _table(run: str, name: str, saved_at: str) -> pd.DataFrame:
    return experiments.load(run, name)


def _show(df: pd.DataFrame, key: str) -> None:
    num = df.select_dtypes("number").columns
    cfg = {c: st.column_config.NumberColumn(format="%.3f") for c in num
           if c not in ("pairs", "hits", "papers", "rows", "n_hits", "n_empty", "fold left out", "T")}
    st.dataframe(df, hide_index=True, width="stretch", column_config=cfg, key=key)


def _filters(df: pd.DataFrame, cols: list[str], key: str) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns and df[c].nunique() > 1]
    if not cols:
        return df
    picks = st.columns(len(cols))
    for col, c in zip(picks, cols):
        opts = df[c].drop_duplicates().tolist()
        v = col.selectbox(c.replace("_", " "), opts, key=f"{key}_{c}")
        df = df[df[c] == v]
    return df.drop(columns=cols)


def render_experiments(run: str) -> None:
    st.title("Experiments")
    st.caption(f"Run **{run}** · every result table the signal experiments saved for this corpus, "
               "adopted or not")
    with st.expander("How to read this page", expanded=False):
        st.markdown(
            "Each experiment asks whether a signal predicts what happened next, using only data available at a "
            "cutoff year T (2020–2023) and checking against T+1…T+3. Models are fitted on three cutoffs and scored "
            "on the fourth (*leave-one-cutoff-out*), so the numbers are honest out-of-sample estimates **for this "
            "corpus**.\n\n"
            "- **AUC**: how well a ranking separates hits from misses (0.5 = chance, 1 = perfect; below 0.5 means "
            "the score points the wrong way).\n"
            "- **p@k**: the share of hits among the top k; compare it with the base rate or *random* row.\n"
            "- **Baselines** (popularity, citations so far, random) are always shown. A signal is only interesting "
            "where it beats them.\n\n"
            "Results depend on the corpus: how it was sampled, the field and its size. A signal that lost here may "
            "win elsewhere, so all tables stay visible. The pilot's decisions are listed for reference only.")
    idx = experiments.load_index(run)
    backtest = experiments.BACKTEST / f"{run}_summary.csv"
    if not idx and not backtest.exists():
        st.info(f"No experiment results for {run} yet. Build the cutoff runs with `landscape backtest --run-id "
                f"{run}`, then run the scripts in `scripts/phase8b/` with `--run-id {run}` (see the 08b doc).")
        return

    by_exp: dict[str, list[tuple[str, dict]]] = {}
    for name, meta in idx.items():
        by_exp.setdefault(meta["experiment"], []).append((name, meta))
    exps = [e for e in ORDER if e in by_exp] + sorted(e for e in by_exp if e not in ORDER)
    labels = (["Phase 8 backtest"] if backtest.exists() else []) + [
        by_exp[e][0][1]["title"].split(":")[0] for e in exps]
    overview = pd.DataFrame({
        "experiment": labels,
        "tables": ([1] if backtest.exists() else []) + [len(by_exp[e]) for e in exps],
        "pilot decision (dro-pilot)": (["recipe signals reworked (see 08-backtest.md)"] if backtest.exists() else [])
        + [PILOT_DECISIONS.get(e, "—") for e in exps],
        "last saved": ([""] if backtest.exists() else []) + [max(m["saved_at"] for _, m in by_exp[e]) for e in exps]})
    st.dataframe(overview, hide_index=True, width="stretch")

    tabs = st.tabs(labels)
    k = 0
    if backtest.exists():
        with tabs[0]:
            st.markdown(
                "The Phase 8 backtest of the recipe's signals (docs/features/08-backtest.md). Each **signal** is scored against "
                "its baselines in each **scope** (clustering or pooled), averaged over cutoffs. Cluster lists are "
                "short (about 17–20 clusters), so p@5–p@20 differences of ±0.05 are noise; **n_hits** says how many "
                "hits there were to find.")
            bt = pd.read_csv(backtest)
            _show(_filters(bt, ["signal"], "bt"), "bt_table")
            st.caption(f"Re-run: `landscape backtest --run-id {run}`")
        k = 1
    for e, tab in zip(exps, tabs[k:]):
        with tab:
            st.caption(f"Pilot decision: {PILOT_DECISIONS.get(e, '—')}")
            for name, meta in sorted(by_exp[e], key=lambda nm: nm[0]):
                st.markdown(f"#### {meta['title']}")
                if meta.get("how_to_read"):
                    st.markdown(meta["how_to_read"])
                df = _table(run, name, meta["saved_at"])
                by = ["outcome", "population", "level", "hit"] + (["model"] if "coefficient" in df.columns else [])
                df = _filters(df, by, name)
                _show(df, f"tbl_{name}")
                st.caption(f"{meta['rows']} rows · saved {meta['saved_at']}")
            cmds = " · ".join(f"`uv run python scripts/phase8b/{s} --run-id {run}`" for s in SCRIPTS.get(e, []))
            if cmds:
                st.caption(f"Re-run: {cmds}")
