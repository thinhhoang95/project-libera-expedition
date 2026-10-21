import json
from dataclasses import replace
import numpy as np, pandas as pd
from landscape import store, signals
from landscape.backtest import Assigner
from landscape.cli import _params
con = store.connect(); base = "dro-pilot"; p0 = _params(con, base)
papers = con.execute("SELECT v.paper_id, p.year FROM provenance v JOIN papers p USING (paper_id) WHERE v.run_id=?", [base]).df()
year = papers.set_index("paper_id").year
edges = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance a ON a.paper_id=r.citing_id AND a.run_id=?
                       JOIN provenance b ON b.paper_id=r.cited_id AND b.run_id=?""", [base, base]).df()
edges["cy"] = edges.citing_id.map(year)
variants = {"as built": {}}
rows = []
for T in (2020, 2021, 2022, 2023):
    run = f"{base}@T{T}"
    A = Assigner(con, run, p0.embedding.version)
    pa = A.assign("problem", 0.5, papers.paper_id.tolist())
    e = edges.assign(f=edges.cited_id.map(lambda x: None), t=None)
    e = edges.assign(f=edges.citing_id.map(pa), t=edges.cited_id.map(pa))
    before = e[(e.cy > T - 3) & (e.cy <= T)].groupby(["f", "t"]).size() / 3
    after = e[(e.cy > T) & (e.cy <= T + 3)].groupby(["f", "t"]).size() / 3
    fut = e[(e.cy > T) & (e.cy <= T + 3)]
    Ps = sorted(pa.unique())
    for name, ov in variants.items():
        p = replace(p0); p.signals = replace(p0.signals, **ov)
        ctx = signals.Ctx(con, run, p)
        rows_b = [r for r in signals.bridges(ctx, T) if r["signal_type"] == "bridge_lim_inn"]
        for no_flow in (0.0, 0.25, 0.5, 1.0):
            def sc(r):
                f = json.loads(r["features"]); return f["mean_topk_sim"] * (1 - f["flow_xy"] / (f["flow_max"] + 1)) ** no_flow
            ranked = sorted(rows_b, key=sc, reverse=True)
            subj = [json.loads(r["subject"]) for r in ranked]
            pairs = [{pr["innovation"] for pr in json.loads(r["features"])["pairs"]} for r in ranked]
            cands = {(X, Y) for X in {s["from_cluster"] for s in subj} for Y in Ps if X != Y}
            def hit(X, Y, inn):
                if after.get((X, Y), 0) >= 3 * max(before.get((X, Y), 0), 1 / 3): return True
                return bool(inn) and fut[(fut.f == X) & fut.cited_id.isin(inn)].shape[0] > 0
            hs = [hit(s["from_cluster"], s["to_cluster"], inn) for s, inn in zip(subj, pairs)]
            label = f"flow penalty power {no_flow}"
            top = ranked[:50]
            rows.append({"T": T, "variant": label + " [median flow_xy of top 50]", "p@10": np.median([json.loads(r["features"])["flow_xy"] for r in sorted(rows_b, key=sc, reverse=True)[:50]]), "p@20": np.nan, "p@50": np.nan})
            rows.append({"T": T, "variant": label, "p@10": np.mean(hs[:10]), "p@20": np.mean(hs[:20]), "p@50": np.mean(hs[:50])})
r = pd.DataFrame(rows)
print(r.groupby("variant")[["p@10", "p@20", "p@50"]].mean().round(3).sort_values("p@50", ascending=False).to_string())
