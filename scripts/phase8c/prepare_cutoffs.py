"""Build unlabeled historical core objects, including years too sparse for all Phase 6 signals.

No downloads or LLM calls. Existing cutoff objects are reused. A 2019 partition permits a
purged 2019 -> 2022 test whose entire 2023-2025 outcome window has elapsed.
"""

import argparse
import json
from dataclasses import replace

from landscape import store
from landscape.cli import _params
from landscape.core import build_core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="dro-pilot")
    ap.add_argument("--cutoffs", default="2019")
    args = ap.parse_args()
    con = store.connect()
    try:
        params = _params(con, args.run_id)
        params = replace(params, clusters=replace(params.clusters, gloss_alphas=[0.3, 0.5]))
        for cutoff in map(int, args.cutoffs.split(",")):
            run = f"{args.run_id}@T{cutoff}"
            if con.execute("SELECT count(*) FROM clusters WHERE run_id = ?", [run]).fetchone()[0]:
                print(f"Reusing {run}")
                continue
            con.execute("""INSERT OR REPLACE INTO runs
                SELECT ?, seed_ids, ?, ?, model_versions, '{}', now(), 'offline research lab cutoff'
                FROM runs WHERE run_id = ?""", [run, json.dumps(params.to_dict()), cutoff, args.run_id])
            con.execute("DELETE FROM provenance WHERE run_id = ?", [run])
            con.execute("""INSERT INTO provenance SELECT ?, v.paper_id, v.route, v.hop, v.via_id, v.score
                FROM provenance v JOIN papers p USING(paper_id) WHERE v.run_id = ? AND p.year <= ?""",
                [run, args.run_id, cutoff])
            print(f"Building {run}", flush=True)
            stats = build_core(con, run, params, skip_labels=True, author_centroids=False)
            print(f"{stats['clusters']} clusters, zero API calls")
    finally:
        con.close()


if __name__ == "__main__":
    main()
