"""Freeze today's predictions, with a baseline list for each, for a prospective test.

Every retrospective test so far can leak hindsight: the corpus was sampled in 2026, the language
models were trained on later papers, and leave-one-cutoff-out trains on the test period's future.
A frozen snapshot scored against papers that appear *after* it has none of these problems.

Writes data/snapshots/<run>/<date>/*.parquet plus manifest.json (SHA-256 of every file, the
corpus as of today, and the scoring protocol, fixed before any outcome exists). A copy of the
manifest goes to docs/snapshots/ so it is versioned in git: any later change to a frozen list
shows up as a hash mismatch.

Usage: uv run python scripts/phase8b/freeze_snapshot.py [--run-id dro-pilot] [--date YYYY-MM-DD]
       (LANDSCAPE_DB may point at a copy of the database)
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from landscape import embed, experiments, store
from landscape.cli import _params

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", default="dro-pilot")
ap.add_argument("--date", default=date.today().isoformat())
a = ap.parse_args()
RUN = a.run_id
OUT = store.DATA / "snapshots" / RUN / a.date
OUT.mkdir(parents=True, exist_ok=True)
con = experiments.connect_ro()
P = _params(con, RUN)
rng = np.random.default_rng(20261021)
files, lists = {}, []


def save(name, df, what, hit, baseline_of=None):
    df = df.reset_index(drop=True)
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    files[name] = df
    lists.append({"list": name, "rows": len(df), "what": what, "hit": hit, "baseline_of": baseline_of})


def sig(t):
    s = con.execute("SELECT * FROM signals WHERE run_id = ? AND signal_type = ? AND variant = 'all' ORDER BY rank",
                    [RUN, t]).df()
    subj = pd.json_normalize(s.subject.map(json.loads))
    return pd.concat([s.drop(columns=["subject"]).reset_index(drop=True), subj], axis=1)


papers = con.execute("""SELECT p.paper_id, p.year, p.type, p.title FROM papers p JOIN provenance v USING (paper_id)
                        WHERE v.run_id = ?""", [RUN]).df()
year = papers.set_index("paper_id").year
refs = con.execute("""SELECT r.citing_id, r.cited_id FROM refs r JOIN provenance v ON v.paper_id = r.citing_id
                      AND v.run_id = ?""", [RUN]).df()
corpus = set(papers.paper_id)
rc = refs[refs.cited_id.isin(corpus)]
cites = rc.groupby("cited_id").size()
pc = con.execute("""SELECT paper_id, cluster_id FROM paper_cluster WHERE run_id = ? AND facet = 'problem' AND alpha = 0.5
                    AND resolution = 'medium' AND cluster_id IS NOT NULL""", [RUN]).df().set_index("paper_id").cluster_id

# ---- 1. paper bridges (likely adoptions) vs popularity among the same kind of candidates
bp = sig("bridge_paper")
save("bridge_paper", bp[["cluster_id", "label", "paper_id", "title", "rank", "score"]],
     "Top 10 innovation papers per problem community (alpha 0.5, medium) not yet cited by it",
     "a paper published after the snapshot, assigned to the community (nearest frozen centroid), cites the work "
     "(or a known alias)")
inn_ids = set(embed.load_vectors("innovation", ids=list(corpus), version=P.embedding.version)[0])
cited_by = rc.assign(X=rc.citing_id.map(pc)).dropna(subset=["X"]).groupby("X").cited_id.apply(set)
rows = []
for X in bp.cluster_id.unique():
    already = cited_by.get(X, set())
    cand = [x for x in inn_ids if pc.get(x) != X and x not in already]
    top = sorted(cand, key=lambda x: (-cites.get(x, 0), x))[:10]
    rows += [{"cluster_id": X, "paper_id": x, "rank": k + 1, "corpus_citations": int(cites.get(x, 0))}
             for k, x in enumerate(top)]
save("bridge_paper_baseline_popularity", pd.DataFrame(rows), "Popularity baseline for bridge_paper: the 10 most "
     "cited innovation papers outside each community and not yet cited by it", "as bridge_paper", "bridge_paper")

# ---- 2. hidden gems vs a seeded random draw from the same pool (popularity is flat among papers cited <= 1x)
gm = sig("bridge_paper_gem")
save("bridge_paper_gem", gm[["cluster_id", "label", "paper_id", "title", "rank", "score"]],
     "Top 5 per community among papers cited at most once", "as bridge_paper", None)
rows = []
for X in gm.cluster_id.unique():
    already = cited_by.get(X, set())
    pool = sorted(x for x in inn_ids if pc.get(x) != X and x not in already and cites.get(x, 0) <= 1)
    pick = rng.choice(pool, size=min(5, len(pool)), replace=False) if pool else []
    rows += [{"cluster_id": X, "paper_id": x, "rank": k + 1} for k, x in enumerate(pick)]
save("bridge_paper_gem_baseline_random", pd.DataFrame(rows), "Random baseline for gems (seed 20261021)",
     "as bridge_paper", "bridge_paper_gem")

# ---- 3. early work vs citations so far among the same recent research papers
ew = sig("early_work")
save("early_work", ew[["paper_id", "title", "rank", "score"]], "Top 50 recent papers expected to be top-quartile "
     "for their year", "top quarter of its publication-year cohort (frozen corpus papers of that year) for "
     "citations received after the snapshot: (a) global, OpenAlex counts_by_year; (b) from new papers in the field",
     None)
y1 = int(papers.year.max())
pool = papers[papers.year.between(y1 - 1, y1) & papers.type.isin(["article", "preprint", "conference-paper",
                                                                  "proceedings-article"])].copy()
pool["c"] = pool.paper_id.map(cites).fillna(0)
save("early_work_baseline_citations", pool.sort_values(["c", "paper_id"], ascending=[False, True]).head(50)
     [["paper_id", "title", "c"]].rename(columns={"c": "corpus_citations"}),
     "Baseline: the 50 most cited recent research papers so far", "as early_work", "early_work")
save("early_work_cohort", pool[["paper_id", "year"]], "The cohort early_work is ranked within", "—", None)

# ---- 4. method transfer and limitation -> innovation (cluster level)
mt = sig("bridge_method_transfer").head(50)
save("method_transfer", mt[["problem_cluster", "method_cluster", "problem_label", "method_label", "rank", "score",
                            "judge_score"]], "Top 50 empty problem x method cells (alpha 0.5, medium)",
     "at least 2 papers published after the snapshot fall in the cell (nearest frozen centroids)", None)
pm = con.execute("""SELECT problem_cluster, method_cluster, sum(count) n FROM pm_matrix WHERE run_id = ? AND alpha = 0.5
                    GROUP BY ALL""", [RUN]).df()
M = pm.pivot(index="problem_cluster", columns="method_cluster", values="n").fillna(0)
cells = M.stack().rename("n").reset_index()
cells = cells[cells.n == 0]
cells["pop"] = cells.problem_cluster.map(M.sum(1)) * cells.method_cluster.map(M.sum(0))
save("method_transfer_baseline_popularity", cells.sort_values("pop", ascending=False).head(50),
     "Baseline: empty cells with the largest row x column totals", "as method_transfer", "method_transfer")
li = sig("bridge_lim_inn").head(50)
save("bridge_lim_inn", li[["from_cluster", "to_cluster", "from_label", "to_label", "rank", "score", "judge_score"]],
     "Top 50 limitation -> innovation community pairs",
     "(a) citations from the first community to the second at least triple per year after the snapshot; (b) a "
     "realized combination: a new paper whose problem is nearest the first community and whose method or "
     "innovation is nearest the second, whether or not it cites it", None)
pairs = li[["from_cluster", "to_cluster"]].drop_duplicates()
allc = sorted(set(pc))
rand = pd.DataFrame([(x, y) for x, y in zip(rng.choice(allc, 200), rng.choice(allc, 200)) if x != y][:50],
                    columns=["from_cluster", "to_cluster"])
save("bridge_lim_inn_baseline_random", rand, "Random community-pair baseline (seed 20261021)", "as bridge_lim_inn",
     "bridge_lim_inn")

# ---- 5. external imports (exp6 live list), vs the current in-field rate
lp = experiments.work_dir(RUN) / "external" / "live_imports.parquet"
if lp.exists():
    L = pd.read_parquet(lp)
    save("external_imports", L.head(100)[["paper_id", "title", "subfield", "score", "outside_field", "recent_citers",
                                          "cluster_id"]],
         "Top 100 external works by the exp6 model (as of the latest complete year)",
         "the share of new field papers citing the work in the 12 / 24 months after the snapshot is at least 1.5x "
         "its share in the three years before, with at least 3 citing papers", None)
    save("external_imports_outside", L[L.outside_field].head(50)[["paper_id", "title", "subfield", "score"]],
         "Top 50 of the outside-the-field scope", "as external_imports", None)
    save("external_imports_baseline_rate", L.sort_values("recent_citers", ascending=False).head(100)
         [["paper_id", "title", "recent_citers"]], "Baseline: the 100 most cited by the field in the last 3 years",
         "as external_imports", "external_imports")

# ---- 6. rising pockets and emerging clusters vs random / size
lm = con.execute("""SELECT paper_id, value FROM paper_scores WHERE run_id = ? AND metric = 'local_momentum'""",
                 [RUN]).df()
top = lm[lm.value >= lm.value.quantile(0.95)]
save("rising_pockets", top, "Recent papers in the top 5% of local momentum",
     "the share of papers published after the snapshot inside the paper's 15-nearest-neighbour ball is at least "
     "1.5x the share in the year before the snapshot", None)
save("rising_pockets_baseline_random", lm.sample(len(top), random_state=20261021), "Random baseline from the same "
     "papers", "as rising_pockets", "rising_pockets")
em = sig("emerging_cluster")
save("emerging_cluster", em[["cluster_id", "label", "rank", "score", "group_kind"]] if "label" in em else em,
     "Clusters ranked by E' (direction: read with care)", "the cluster's share of new papers after the snapshot is "
     "above its share in the three years before", None)

# ---- 7. the frame: today's corpus and the frozen centroids used to assign future papers
save("corpus_ids", papers[["paper_id", "year"]], "Every paper in the corpus at the snapshot: later papers are new",
     "—", None)
cent = Path(store.DATA) / "vectors" / f"cluster_centroids_{RUN}.parquet"
if cent.exists():
    shutil.copy(cent, OUT / "cluster_centroids.parquet")

regions = experiments.work_dir(RUN) / "exp8_regions_live.parquet"  # experiment B, when it has run
if regions.exists():
    shutil.copy(regions, OUT / "birth_regions.parquet")
    lists.append({"list": "birth_regions", "rows": len(pd.read_parquet(regions)),
                  "what": "Regions where experiment B expects new communities to form",
                  "hit": "see exp8_birth.py: the region becomes dense with papers published after the snapshot",
                  "baseline_of": None})

sha = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.glob("*.parquet"))
       if not f.name.startswith("._")}  # skip macOS AppleDouble files on exFAT
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
manifest = {
    "run": RUN, "snapshot_date": a.date, "created_at": datetime.now().isoformat(timespec="seconds"),
    "git_commit": commit, "corpus_papers": len(papers), "latest_publication_year": y1,
    "embedding_model": P.embedding.version, "extraction_model": P.extraction.model,
    "lists": lists, "sha256": sha,
    "protocol": {
        "horizons_months": [6, 12, 24],
        "new_papers": "Papers published after snapshot_date that are not in corpus_ids, collected by re-running the "
                      "run's expansion routes with the same parameters (record the date and the route of each).",
        "assignment": "New papers get facets and embeddings with the frozen models above and are assigned to the "
                      "nearest frozen centroid (cluster_centroids.parquet).",
        "comparison": "Each list is compared with its baseline_of list on the same hit definition: precision@k "
                      "(per community where lists are per community) with a bootstrap over communities. Report every "
                      "list, including the ones that lose.",
        "no_peeking": "The lists are not edited after this date; a change shows as a SHA-256 mismatch with the copy of "
                      "this manifest in docs/snapshots/.",
    },
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
docs = Path("docs/snapshots")
docs.mkdir(parents=True, exist_ok=True)
(docs / f"{RUN}-{a.date}.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: v for k, v in manifest.items() if k not in ("sha256", "protocol")}, indent=1)[:3000])
