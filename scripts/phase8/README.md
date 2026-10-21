# Phase 8 analysis scripts

One-off analyses behind the choices in `docs/features/08-backtest.md`. They read the cutoff runs
built by `uv run landscape backtest --run-id dro-pilot` and only use data already in the database.

| Script | Question |
| --- | --- |
| `route_robustness.py` | Which cluster features predict future share change, and do they keep their sign when outcomes are measured on papers from different expansion routes? |
| `loco_feature_selection.py` | Leave-one-cutoff-out: pick sign-robust features on 3 cutoffs, test on the 4th (recipe hits). Also writes `data/projects/dro-pilot/backtest/cluster_features_*.parquet`. |
| `loco_smoothed_hits.py` | The same, scored with smoothed (3-year) hits. Run after the previous script. |
| `people_and_novelty.py` | Do broker scores add information beyond popularity? Does novelty predict citations? |
| `lim_flow_penalty.py` | Limitation → innovation: effect of the flow-penalty exponent on precision and on how "missing" the top bridges are. |

Run with `PYTHONIOENCODING=utf-8 uv run python scripts/phase8/<script>.py --run-id <run>` from the repo
root (the default run is `dro-pilot`). They open the database read-only. Each saves its result table to
`data/projects/<run>/backtest/experiments/`, which the Experiments view shows. The web app runs all five, in
the order above, as the optional pipeline step `experiments-8`
(`scripts/run_experiments.py --step experiments-8`).
