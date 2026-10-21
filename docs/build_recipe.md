# Research Landscape Engine — Build Recipe

Oct 20, 2026 · @Thinh

## Purpose and goals

The engine takes a seed paper (or a small seed set) and returns a ranked, evidence-backed map of its field: what the field contains, where it is moving, and where the gaps are. Every output is a ranked candidate with the papers that support it, meant for an expert to review. It is decision support, not an oracle.

Outputs, in priority order:

- **Landscape:** labeled clusters per facet (problem, method, innovation, result, limitation) and a problem × method matrix.
- **Direction:** clusters growing or shrinking as a share of output, author migration between clusters, shifting citation flows.
- **Emerging topics:** young, fast-growing clusters and first-ever problem–method combinations.
- **Niches:** coherent, active clusters that the rest of the field rarely cites.
- **Missing bridges and opportunities:** limitations in one cluster matched to innovations in another, method transfers, unresolved limitations.
- **Influence:** seeders of growing clusters, brokers across communities, disruptive papers, and the institutions behind them.
- **Integrity anomalies:** an internal screening module only (Phase 7). It flags patterns with evidence, never verdicts.

The central idea: each layer (citations, semantic similarity, authorship, time) is a weak signal alone. The useful outputs come from where layers **disagree**, for example two clusters that are close in meaning but never cite each other.

**Visual-first principle.** Human judgment is the final step, so no analytic ships without its view. Every score and ranked list appears in the explorer (Phase 6b) beside the visualization that lets a person check it, and every mark opens its evidence papers. Scores decide what people look at first; people decide what it means.

## Scope and constraints

Version 1 works from metadata only: authors, titles, abstracts, references, years, venues and institutions. No full-text parsing.

| Item | v1 decision |
| --- | --- |
| Inputs | Title, abstract, authorships (with institutions), reference list, year, venue, retraction flag |
| Optional extra | Semantic Scholar citation contexts and intent labels (free, same references call) |
| Corpus per seed | 5,000–20,000 papers |
| Metadata cost | About $0.30–0.50 per run of 2,000 papers + 2,000 authors on OpenAlex, inside the free $1/day key |
| LLM extraction volume | About 20,000 abstracts × (≈500 input + ≈250 output tokens) ≈ 15M tokens per run; price per chosen model |
| Embedding | Local GPU, open-weights model, no per-call cost |

Out of scope for v1:

- Full-text parsing (GROBID, S2ORC). Planned for v2 if citation function needs more than S2's labels.
- Image, statistics or data-fabrication checks.
- Any public ranking of named individuals.
- Graph neural networks. Added only if the backtest (Phase 8) shows simple features fall short.

Discussed and deferred (not to be built in v1):

- Venue prestige as a scoring weight or visual encoding.
- Review papers as seeds and as a prior taxonomy to compare against the clusters.

## Architecture and stack

The pipeline is a chain of batch jobs, one per phase, each reading and writing versioned tables under one `run_id`. Nothing is real-time; a full run on a 20,000-paper corpus should complete in hours, mostly LLM extraction time.

&#91;embedded content: pipeline · 8 phases, 3 stores, 2 side modules\]

The loop on the left is the semantic expansion route: after a first embedding pass, Phase 2 pulls in similar papers the citation graph missed. The backtest re-runs Phases 5–6 at past cutoffs and feeds tuned weights back into the signals. The explorer is the main output: it reads every table, and reviewers' judgments made there flow back into prompts, labels and signal weights.

| Layer | Choice |
| --- | --- |
| Language | Python 3.12 |
| HTTP clients | httpx (async) with retry and backoff |
| Storage | DuckDB over Parquet; object storage for raw JSON |
| LLM extraction | Cheap, fast model with JSON-schema output (Mercury-class; team to choose) |
| LLM judge and labels | A stronger model, low volume (hundreds of calls per run) |
| Embeddings | EmbeddingGemma 2 via sentence-transformers on a local GPU |
| Vector index | LanceDB or FAISS (HNSW) |
| Graphs and clustering | python-igraph + leidenalg |
| Numerics | numpy, scipy, scikit-learn |
| Orchestration | Dagster or Prefect; one job per phase, `run_id` passed through |

## Phase 1 — Data ingestion

OpenAlex is the primary source (authors, institutions, abstracts, references, retraction flags). Semantic Scholar (S2) fills missing abstracts and supplies citation contexts and intents. DOI is the join key between them; arXiv ID is the fallback.

### OpenAlex calls

| Need | Call | Cost |
| --- | --- | --- |
| Papers by DOI, 100 per call | `/works?filter=doi:a\|b\|…&per_page=100&select=…` | $0.10 per 1,000 calls |
| One paper by ID or DOI | `/works/W…` or `/works/doi:…` | Free |
| An author's works | `/works?filter=author.id:A…&per_page=100&cursor=*` | $0.10 per 1,000 calls |
| Up to 100 authors at once | `/works?filter=author.id:A1\|A2\|…&cursor=*` | Same |
| Forward citations of a paper | `/works?filter=cites:W…&cursor=*` | Same |
| Resolve reference IDs | `/works?filter=openalex:W1\|W2\|…` (100 per call) | Same |

Required `select` fields: `id, doi, title, publication_year, publication_date, authorships, referenced_works, abstract_inverted_index, primary_location, cited_by_count, is_retracted, topics, type, language`.

Rules:

- Always pass the API key, `per_page=100`, and `select=`.
- Never use `/authors?search=` in the pipeline: it is billed as search ($1 per 1,000). Take author IDs from `authorships`.
- Use cursor paging for anything over 10,000 results.
- Stay under 100 requests per second; exponential backoff on 429.
- Log `meta.cost_usd` per call and stop a run when it reaches its budget.
- Abstracts arrive as an inverted index. Rebuild them:

```python
def rebuild_abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))
```

### Semantic Scholar calls

| Need | Call |
| --- | --- |
| Missing abstracts, batch | `POST /graph/v1/paper/batch?fields=abstract,externalIds,year` with `{"ids": ["DOI:…", "ARXIV:…"]}` |
| Citation contexts and intents | `GET /graph/v1/paper/{id}/references?fields=contexts,intents,isInfluential,externalIds` |
| Cross-check author papers | `GET /graph/v1/author/{id}/papers` (limit up to 1,000 per page) |

Notes:

- Send the key in the `x-api-key` header. Confirm the current per-key rate limit and batch size when the key is issued.
- S2 states abstracts may be missing for legal reasons even when its website shows one, so expect gaps in both sources.
- Responses are capped at 10 MB; shrink batches if a call fails on size.

### Storage and idempotency

- Store every raw JSON response in object storage, keyed by source + ID + fetch date. Never re-fetch an ID already stored within the run's freshness window (default 30 days).
- Parse raw responses into the tables in the Data model section.
- For corpora above roughly 200,000 papers, load the free OpenAlex snapshot from S3 into DuckDB instead of calling the API.

## Phase 2 — Corpus expansion

Expansion grows the seed into a 5,000–20,000 paper corpus through four routes, each capped, and records which route brought each paper in. The semantic route exists to counter seed bias: a citation-only sample can only describe the seed's own neighborhood.

1. **Seed set:** one to five papers given by the user, optionally plus the references of a deep research report (see below).
2. **Backward hops:** references of the seed (hop 1), then references of hop-1 papers (hop 2). At hop 2, follow only substantive edges: S2 `isInfluential`, or a method/extension intent when available. Otherwise keep the top-k references by how often they are co-cited inside the corpus.
3. **Forward hops:** papers citing the seed and hop-1 papers, limited to the last N years.
4. **Author fan-out:** authors of the seed and of substantively cited hop-1 papers. Pull each author's works from the last M years, newest first, up to a per-author cap.
5. **Semantic expansion:** after a first embedding pass (Phase 4), take the dominant OpenAlex `topics` in the corpus, pull recent works under those topics, embed them, and keep only those above a similarity threshold to an existing cluster centroid.
6. **Stop** when the paper budget is reached; fill routes in the order above.

**Optional seed: deep research report.** The user may attach a report from a commercial AI research tool to widen the starting set beyond one citation neighborhood. Rules:

- Extract the report's references and resolve each one to an OpenAlex record by DOI, or by exact title match plus year. Drop anything that does not resolve; AI reports can cite papers that don't exist or misattribute claims.
- Resolved papers enter as seeds with route `ai_report` in provenance, and expansion then treats them like any other seed.
- Store the report file with the run, since the same prompt will not give the same report again.
- The report's text and framing never reach the LLM judge or the cluster labeler, so the system does not just repeat the report's view back.

Default parameters (tune per field):

| Parameter | Default |
| --- | --- |
| Max hops (backward) | 2 |
| Hop-2 references kept per paper | 15 |
| Forward-citation window N | 5 years |
| Author window M | 5 years |
| Works per author cap | 50 (most recent ones) |
| Semantic expansion similarity threshold | 0.75 cosine to nearest centroid (calibrate on a sample) |
| Paper budget | 20,000 |
| Metadata spend cap per run | $2.00 |

Every paper gets a provenance row: entry route, hop number, and the paper or author that brought it in. Signals in Phase 6 are reported both on the whole corpus and with the semantic route excluded, so seed bias stays visible.

## Phase 3 — LLM facet extraction

A cheap, fast model rewrites each abstract into six structured facets, plus two short glosses and two canonical tags that add the model's own background knowledge for the problem and method. The raw abstract is kept unchanged alongside them; the integrity module needs the original wording. We will use DeepSeek v4.1 flash because it is cheap. Make sure that the OpenRouter API calls the cheapest provider (without quantization), and we can parallelize the calls to save time.

### Output schema

```json
{
  "problem": "string — the task or question addressed, stated generically (faithful)",
  "problem_class": "string — 1–4 word canonical tag, e.g. 'stochastic optimization'",
  "problem_gloss": "string — 1–2 sentences of background on this problem class as it appears here (model knowledge)",
  "domain_data": "string — application domain and data used, or null (faithful)",
  "method": "string — the core technique or approach (faithful)",
  "method_family": "string — 1–4 word canonical tag, e.g. 'chance-constrained programming'",
  "method_gloss": "string — 1–2 sentences on how this technique works, in the variant this abstract describes (model knowledge)",
  "innovation": "string — what is new relative to prior work (faithful)",
  "result": "string — main finding, with numbers if stated (faithful)",
  "limitations": "string or null — ONLY if explicitly stated in the abstract (faithful)",
  "contribution_type": "method | dataset | benchmark | theory | survey | application | position | other",
  "has_quantitative_result": "boolean"
}
```

### Prompt (v1)

```text
You extract structured facets from a scientific abstract.
Faithful fields (problem, domain_data, method, innovation, result, limitations):
- Use only information in the abstract. Never infer or add.
- One or two standalone sentences each, at most 40 words, no pronouns referring outside the field.
- Use generic technical vocabulary rather than the paper's coined names, but keep the coined name in parentheses.
- "limitations": null unless the abstract explicitly states a limitation, weakness, or open problem.
- "domain_data": null if not stated.
Gloss fields (problem_gloss, method_gloss):
- Explain the general concept in at most 50 words, as a textbook would, matched to the specific variant this abstract describes.
- If the variant is unclear, describe the general family and say "variant unclear".
- Never copy gloss content into faithful fields.
Tags (problem_class, method_family):
- 1-4 words, lowercase, the most common textbook term for the class or family.
Write in English even if the abstract is not.
Return JSON matching the schema exactly.

Title: {title}
Abstract: {abstract}
```

Tags are normalized after extraction: embed all distinct tags, merge near-synonyms (cosine above 0.9, reviewed in the facet review view), and keep a tag vocabulary per run. The explorer can use `problem_class` × `method_family` as an alternative axis for the problem × method grid.

### Run settings

- Temperature 0, JSON-schema-constrained output.
- Store `model_id` and `prompt_version` on every facet row; re-run only when either changes.
- Skip papers with no abstract; mark them `title_only` so later phases can down-weight them.
- Batch requests and run async; retry malformed JSON once, then log and skip.

### Validation and QA

- Automatic checks: schema validity; no field longer than 60 words; flag a field if more than 80% of its words are copied in order from the abstract.
- Track the null rate of `limitations` per field of study. A rate near zero means the model is inventing limitations.
- Gold set: 200 papers labeled by two people from the target field. Score each facet as supported, partly supported, or unsupported by the abstract.
- Acceptance for v1: at least 90% of facets supported; fewer than 5% of non-null limitations unsupported.
- Repeat the gold check whenever the model or prompt changes.

* Gold-set reviewers also check each gloss: does it describe the variant the abstract uses? Target: at least 85% correct variant, with "variant unclear" counted as acceptable.

## Phase 4 — Embeddings and vector store

Each paper gets one vector per facet plus one for the full abstract, from a local open-weights model. Facet vectors drive clustering and matching; the abstract vector drives novelty and general similarity.

- **Model:** [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (Apache 2.0, 768 dimensions, Matryoshka truncation to 512, 256 or 128). It expects short task-instruction prefixes; use the documented retrieval or similarity prefix consistently and record which one.
- **Do not** use citation-trained embeddings (SPECTER and similar). They bake the citation graph into the vectors and erase the semantic-vs-citation disagreement the signals rely on.
- **Vectors per paper:** `problem`, `method`, `innovation`, `result`, `limitations` (when present), `domain_data` (when present), `abstract`.
- **Dimensions:** store 768; use 256 for kNN graphs and clustering; use 768 for limitation → innovation matching where precision matters.
- **Normalize** to unit length; cosine similarity everywhere.
- **Derived vectors:** author-year centroid per facet (mean of that author's papers that year); cluster centroid per facet.
- **Store:** one table or collection per facet in LanceDB or FAISS (HNSW index), keyed by paper ID, with `embedding_model_version`. Re-embed everything when the model version changes; never mix versions in one index.
- **Throughput check:** a 20,000-paper corpus is about 140,000 facet vectors, a small job on one consumer GPU.

### Gloss vectors and the blending weight

`problem_gloss` and `method_gloss` get their own vectors. For the problem and method facets, the vector used downstream blends the faithful facet with its gloss under a weight α between 0 and 1:

```latex
e_f^{(\alpha)} = \frac{(1-\alpha_f)\, e_f + \alpha_f\, e_{f,\mathrm{gloss}}}{\left\lVert (1-\alpha_f)\, e_f + \alpha_f\, e_{f,\mathrm{gloss}} \right\rVert}
```

α = 0 is the abstract's own wording; higher α pulls papers that use different words for the same idea closer together, at the cost of blurring what is distinctive about each one.

| Use | Default α | User can change? |
| --- | --- | --- |
| Method transfer, bridges, cross-domain similarity, cluster labels | 0.5 | Yes |
| Landscape map, coarse and medium clusters | 0.3 | Yes |
| Fine clusters, novelty | 0 | Yes |
| Duplicate detection, integrity module | 0 | No (fixed) |

How the user adjusts it:

- A per-facet α slider in the explorer's global controls, next to the year slider.
- Clusters and map layouts are precomputed, so the pipeline builds them at α = 0, 0.3, 0.5 and 0.8, and the slider snaps to those. Any other value starts a recompute job for Phases 5–6 on that run.
- Every run and every review action records the α values in effect, so ratings made at different settings stay comparable.
- Gloss vectors are stored at full 768 dimensions like the other facets; the blend is computed on the fly, not stored.

## Phase 5 — Core analytical objects

Five objects carry every signal in Phase 6. Build them once per run and version them with the run ID.

### 5.1 Facet clusters

1. Per facet, build a kNN graph on 256-d vectors (k = 20, cosine; edge weight = similarity; drop edges below the 5th percentile).
2. Run Leiden (`leidenalg`, CPM or modularity) at three resolutions: coarse, medium, fine. This gives a hierarchy.
3. Repeat with 5 random seeds. Keep a cluster only if its best-matching cluster in the other runs has Jaccard overlap of 0.6 or more.
4. Fit clusters on **all years at once**, then assign papers to them. Do not re-cluster per year; clusters stop lining up between years.\
Of course the user can manually give the seed parameters manually during exploration as well.
5. Label each cluster with the LLM: give it the 10 papers closest to the centroid (facet text only) plus the top 15 c-TF-IDF terms. Ask for a label of at most 6 words and a one-line description.

### 5.2 Problem × method matrix

Rows are problem clusters, columns are method clusters (medium resolution). Each cell holds a paper count, overall and per year.

```latex
M_{p,m,t} = \left|\{\, x : \mathrm{prob}(x)=p,\ \mathrm{meth}(x)=m,\ \mathrm{year}(x)=t \,\}\right|
```

Also store the first year each cell became nonzero.

### 5.3 Time slices

Yearly buckets (two-year buckets if a cluster has fewer than 20 papers a year). Every count is also stored as a share of that year's total, because the whole field grows:

```latex
\mathrm{share}_{c,t} = \frac{n_{c,t}}{N_t}
```

### 5.4 Citation flows

For each facet's medium clustering, F(t)\[i, j\] = number of citations from papers in cluster i published in year t to papers in cluster j. Keep raw counts and a row-normalized version.

### 5.5 Author profiles

For each author and year: distribution of their papers across clusters per facet, and the author-year centroid from Phase 4. Institutions inherit from authorships.

## Phase 6 — Signals and scoring

Each signal writes ranked rows to the `signals` table with a score and an evidence list of paper IDs. Scores combine z-scored features computed within the corpus; weights below are starting values that the backtest (Phase 8) tunes. Every signal is computed with a cutoff year so the same code runs in backtests.

**Definition of done:** a signal is finished only when its Phase 6b view shows it with its evidence. Each subsection below names that view.

**As-of computation:** cluster-level signals (6.1–6.3) are computed at every yearly cutoff in the corpus, not just the latest, so the map can show each cluster's status at the playhead year. Bridges (6.4) and influence (6.5) are computed for the latest year in v1.

### 6.1 Direction (moving into / away from)

- **Share growth:** slope of log share over the last 3 years, and the change in that slope (acceleration).

```latex
g_c = \mathrm{slope}_{t \in [T-2,\,T]} \; \log \mathrm{share}_{c,t}
```

- **Author migration:** A(t)\[i → j\] = number of authors whose dominant cluster was i in year t−1 and j in year t. Report net flow per cluster and the top flows. This is the earliest direction signal.
- **Reference age:** median age (in years) of references cited by a cluster's papers in year t. Falling = active research front; rising = consolidating.
- **Baseline shift:** where S2 intents exist, the share of a cluster's incoming citations labeled as result comparison rather than method use. A rising share means the cluster is becoming a baseline.

**View:** map layers for clusters, activity shift, citation flows and author migration, read through the time axis.

### 6.2 Emerging topics

- **Paper novelty**, per facet f:

```latex
\mathrm{nov}_f(x) = 1 - \max_{y:\ \mathrm{year}(y) < \mathrm{year}(x)} \cos\!\big(e_f(x),\, e_f(y)\big)
```

- **New combination:** a problem × method cell whose first nonzero year is within the last 2 years.
- **Cluster emergence score** (clusters with at least 10 papers):

```latex
E_c = z(g_c) + z(\Delta g_c) + z(\text{new-author share}_c) - z(\text{median ref age}_c) + z(\overline{\mathrm{nov}}_c)
```

**View:** emerging badges on the map at the playhead year; new combinations flash in the grid in the year they appear; novel-papers layer.

### 6.3 Niches and hidden corners

- Coherence: mean pairwise cosine within the cluster (sample 500 pairs for large clusters).
- External inflow: share of the cluster's received citations that come from other clusters.
- Activity: papers in the last 3 years.

```latex
N_c = z(\mathrm{coh}_c) - z(\mathrm{ext}_c) \quad \text{if activity}_c \ge a_{\min}, \text{ else not a niche}
```

Default a\_min = 10 papers. Low-activity clusters with the same profile are reported separately as dormant.

**View:** niche and dormant badges on the map; selecting one shows its few incoming citation arcs.

### 6.4 Missing bridges and opportunities

**(a) Limitation → innovation matching (priority 1).** For each stated limitation l in cluster X and each innovation i in cluster Y ≠ X published no earlier than l:

```latex
B_{X \to Y} = \operatorname{mean}_{\text{top-}k} \cos\!\big(e_{\mathrm{lim}}(l),\, e_{\mathrm{inn}}(i)\big) \cdot \Big(1 - \frac{F[X,Y]}{\max_Y F[X,Y] + 1}\Big)
```

k = 5. Evidence: the top matching limitation–innovation paper pairs.

**(b) Method transfer.** For a problem cluster P and method cluster m with M\[P, m\] = 0:

```latex
T_{P,m} = \frac{\sum_{P'} \mathrm{sim}(P, P') \cdot \mathbb{1}[M_{P',m} > 0]}{\sum_{P'} \mathrm{sim}(P, P')}
```

sim = cosine between problem-cluster centroids, summed over the 10 nearest problem clusters. Later version: low-rank factorization of M as a recommender.

**(c) Unresolved limitations.** Cluster the limitation vectors inside each cluster. A limitation group with at least 5 papers and no later innovation above the match threshold (default 0.7) is an open problem.

**(d) Semantic-but-not-cited.** Pairs of problem clusters in the top 5% of centroid similarity with citation flow in both directions below the 10th percentile.

**LLM judge.** Only the top 50 candidates per bridge type go to a stronger LLM with the evidence papers' facets. It returns plausibility (1–5), a two-sentence rationale, and any obvious reason it has not been done. The judge reorders; it never adds candidates.

**View:** bridges layer (dashed arcs on the map) with the limitation ↔ innovation pairs in the detail panel, and outlined cells in the grid. Reviewers mark each candidate as worth pursuing, obvious, or wrong; these marks are stored and later used to tune the ranking.

### 6.5 Influence

- **Disruption (CD) index** of paper p, over the n papers citing p or any of its references, with f\_i = 1 if i cites p and b\_i = 1 if i cites any reference of p:

```latex
\mathrm{CD}(p) = \frac{1}{n} \sum_{i=1}^{n} \left( f_i - 2 f_i b_i \right)
```

- **Seeder score:** count of an author's papers among the earliest 10% of clusters whose growth g\_c is above the median.
- **Broker score** (participation coefficient), where k\_{a,c} is author a's papers or citations in cluster c and k\_a their total:

```latex
P_a = 1 - \sum_c \left( \frac{k_{a,c}}{k_a} \right)^2
```

- **Substantive citations:** count only citations S2 marks `isInfluential` or labels as method use, when available.
- **Recent papers** (under 2 years old) have too few citations; use bibliographic coupling (shared references) for their similarity and influence proxies.
- **Institutions:** aggregate author scores through authorships, weighted by 1 / number of institutions per authorship.

**View:** people layer: a selected author's path on the map up to the playhead, with the window's seeders and brokers listed in the detail panel.

## Phase 6b — Visual explorer (first-class, built alongside every phase)

The explorer is the product's main surface, not a reporting add-on: the analytics feed it, and people judge in it. It has **one map and one grid, sharing one time axis**. Share over time, migration, citation flows, emergence and bridges are not separate charts; each is a layer drawn on the map or grid and read through the time axis. It grows from M1 onward, one layer per phase, and every mark drills down to the papers behind it. All layouts are precomputed by the pipeline, so the front end only reads tables.

### Layout

- **Map (center):** the fixed 2D layout of papers for the chosen facet and gloss weight α.
- **Grid (beside the map, linked):** the problem × method grid.
- **Time axis (bottom strip):** total output per year, plus the selected cluster's share. Drag a handle for one year, or brush a window \[t1, t2\]; a play button animates year by year.
- **Layer panel (left):** layer toggles, facet switcher, α slider.
- **Detail panel (right):** the card for whatever is selected (paper, cluster, author, bridge) with its evidence papers, plus a sortable ranking of clusters by any score for the current window.

&#91;embedded content: explorer layout · layer panel, map, grid, detail panel, time axis\]

The accent color is reserved for what a reviewer should look at: bridge candidates on the map, transfer candidates in the grid, and the playhead. Moving the playhead redraws every active layer at once.

### Layers on the map

Every layer reads the same year or window from the time axis.

| Layer | Shows | Behavior over time | Signal | Data |
| --- | --- | --- | --- | --- |
| Papers | Each paper as a point, cluster color | Appears in its publication year; papers before the window fade | — | `viz_layout` |
| Clusters | Hull and label per cluster | Label size follows share in the window; hover shows a share sparkline | 6.1 share growth | `viz_cluster_year` |
| Activity shift | Where output concentrates | Density contours for the window; compare mode shows gain versus loss between two windows | 6.1 direction | `viz_density` |
| Citation flows | Who builds on whom | Arcs between cluster centroids; width = citations in the window, color = change from the previous window; top-N only, bundled | 5.4 | `viz_flow_year` |
| Author migration | Where people move | Arrows between centroids; width = authors moving in the window; animates with play | 6.1 | `viz_flow_year` |
| Status badges | Emerging, niche or dormant | Each cluster's status as of the playhead year | 6.2, 6.3 | `viz_cluster_year` |
| Novel papers | High-novelty papers | Highlighted in their publication year | 6.2 | `signals` |
| Bridges | Candidate gaps | Dashed arcs between clusters; click opens the limitation ↔ innovation pairs | 6.4 | `signals` |
| People | Seeders and brokers | A selected author's yearly positions as a path ending at the playhead; the side panel lists seeders and brokers for the window | 6.5 | `viz_author_year` |

### Layers on the grid

- Cell color = papers in the window (log scale).
- New combinations flash in the year they first appear.
- Method-transfer candidates are outlined; click for the evidence.
- Selecting a cluster on the map highlights its row or column in the grid and its share line on the time axis, and the reverse.

Inspection views for the earlier phases, so people can check the pipeline itself and not only its conclusions:

| View | Phase | What a reviewer does there |
| --- | --- | --- |
| Corpus view | 2 | Sees paper counts by entry route, hop and year; spots seed bias (for example, a field visible only through the semantic route) |
| Facet review | 3 | Reads the abstract beside its six facets; flags unsupported or invented facets; labels the 200-paper gold set here |
| Cluster review | 5 | Browses a cluster's central papers, label and stability; proposes rename, merge or split |

Design rules:

- **Layers, not views.** A new analytic is added as a layer on the map or grid. A standalone chart is allowed only when it cannot be placed on either, and the reason is written down.
- **One time axis for everything.** Every layer, ranking and card reads the same year or window.
- **One color per cluster everywhere**, assigned once per run and stored in `clusters`.
- **Evidence first:** every mark leads to a paper card showing title, year, authors, the facets and glosses, and a link out.
- **Fixed layout per run and per α:** fit UMAP once on the run's corpus and store 2D coordinates, so the map doesn't move between sessions or years. New papers are placed with UMAP's transform, not a refit.
- **Legibility:** arcs and arrows show the top N per window (default 30) with edge bundling; the layer panel suggests at most three active layers at once.
- **Exports:** a snapshot of the current map, layers and window as PNG or SVG, and an animation of a time range for the field report.

**Gloss weight control.** The per-facet α slider (Phase 4) sits with the global controls and re-draws the landscape map, clusters and bridge rankings at the chosen setting, so a reviewer can see what changes when papers are grouped by concept instead of wording.

**Judgments are data.** Every review action (flag a facet, rename a cluster, rate a bridge) is stored in a `review_actions` table: run\_id, reviewer, view, subject, action, note, timestamp. These feed prompt fixes, cluster relabeling and signal weights. Until the backtest (Phase 8) is built, reviewer ratings are the main evidence of which signals work.

Build:

- **Prototype (from M1, one layer per milestone):** Streamlit or Dash with Plotly, reading DuckDB directly. Order: corpus view (M1); facet review (M2); map with papers, clusters and the time axis, cluster review, grid, activity shift and bridges (M3); the remaining layers with their signals (M4). For a quick labeled map, the DataMapPlot library produces interactive maps from 2D coordinates and labels.
- **Production (after M3):** a small web app (React with deck.gl for the map and its arc and path layers; Vega-Lite or Observable Plot for the grid and time axis), reading Parquet or JSON exports of the `viz_*` tables.
- **Precomputed tables, all keyed by run\_id, α and facet; windows are summed from yearly rows at query time:**
  - `viz_layout` (paper\_id, x, y) and `viz_cluster_layout` (cluster\_id, x, y, color)
  - `viz_cluster_year` (cluster\_id, year, n, share, status, scores json)
  - `viz_flow_year` (kind = citation or migration, year, from\_cluster, to\_cluster, count)
  - `viz_density` (year, density grid)
  - `viz_author_year` (author\_id, year, x, y)
  - `review_actions` (see above)

## Phase 7 — Integrity anomaly module (screening only) - not to be implemented right now

This module reports statistical anomalies with evidence for review by people with an editorial or integrity role. It never labels anyone as having committed misconduct, and its output is never public. Most hits will have innocent explanations.

| Signal | How it is computed | Main innocent explanation |
| --- | --- | --- |
| Irrelevant citations | Share of a paper's references whose abstract similarity to the citing paper is below the field's 1st percentile, and whether they concentrate on a few authors or venues | Broad survey papers, tool or dataset citations |
| Citation ring | Groups of authors or venues whose mutual citation density is far above what their semantic similarity predicts | Small subfields with few active groups |
| Duplicate publication | Pairs of one author's papers with near-identical problem, method and result facets (cosine above 0.95 on all three) | Conference and journal versions of one work |
| Productivity outlier | Papers per year above the 99.5th percentile for the field and median team size | Large consortia (physics, genomics, clinical trials) |
| Abrupt topic jump | Author-year centroid moves more than a threshold distance in one year, into an unrelated cluster | **Name disambiguation errors** (most common), genuine career changes |
| Retracted foundations | Share of a cluster's substantive references that point to retracted works (OpenAlex `is_retracted`) | None; this is reported without suspicion of anyone |
| Text artifacts | On the raw abstract, not the facets: tortured-phrase lists and templated sentence structures shared across unrelated author groups | Non-native writing, shared lab templates |

Rules:

- **Check disambiguation first.** Before any author-level flag, verify the author record (ORCID, institution continuity, co-author overlap). Drop the flag if the record looks merged.
- **Normalize by field and team size.** No global thresholds.
- **Prefer aggregate levels** (venue, special issue, cluster) over named individuals.
- **Output format:** the measured value, the field baseline, and the evidence papers. No suspicion score or adjective.
- **Access:** a separate table with role-based access; excluded from exports and reports by default.
- **Validation:** before use, measure precision against known retractions (the Retraction Watch database, now distributed openly by Crossref) and documented paper-mill cases. Compare hits with existing screeners such as the Problematic Paper Screener rather than duplicating them.
- **Out of reach with abstracts:** image manipulation, impossible statistics, fabricated data.

## Phase 8 — Backtest and evaluation (not to be implemented right now)

The backtest is how we know a signal works: freeze the corpus at year T, run the full pipeline, and check predictions against what happened in T+1 to T+3. A signal that does not beat its simple baseline is cut.

### Protocol

1. Choose cutoffs T = 2018, 2019, 2020, 2021 (rolling origins).
2. For each T, keep only papers published in or before T and citations made in or before T. Fit clusters, matrix and flows on that data only.
3. Run every Phase 6 signal and store the top 50 per signal.
4. Compute outcomes from the full corpus for T+1 to T+3.
5. Report precision@k (k = 10, 20, 50) per signal, averaged over the four cutoffs.

### What counts as a hit

| Signal | Hit in T+1…T+3 |
| --- | --- |
| Emerging cluster | Share at T+3 at least 1.5× share at T |
| Moving away | Share falls for 2 consecutive years |
| Method transfer (P, m) | At least 2 papers appear in cell (P, m) |
| Limitation → innovation (X → Y) | Citation flow X → Y at least triples, or a paper in X cites a matched innovation paper |
| Semantic-but-not-cited | Citation flow between the pair rises above the corpus median |
| Seeder / broker | Author's substantive citations in T+1…T+3 in the top decile |

### Baselines each signal must beat

- Random ranking.
- Popularity: rank by current size or citation count.
- Citation-only: the same signal computed without embeddings (for example, link prediction from common neighbors).

### Leakage rules

- **The LLM judge and cluster labeler know the future.** In backtests, either switch the judge off or use a model whose training cutoff is before T. Report both.
- Embedding and extraction models also saw later papers in training; record this as a known limitation, and check whether results hold when T is close to the model's training cutoff.
- Never reuse cluster IDs from a full-corpus run inside a backtest.

### Human evaluation

For the live run, two experts in the pilot field rate the top 20 of each signal: useful and non-obvious / obvious / wrong. Target for v1: at least 30% useful and non-obvious in the bridge signals.

## Data model

All tables live in DuckDB over Parquet (Postgres is fine if the team prefers). Every derived table carries `run_id`.

| Table | Key fields |
| --- | --- |
| `papers` | paper\_id (OpenAlex W), doi, s2\_id, arxiv\_id, title, abstract\_raw, abstract\_source, year, pub\_date, venue\_id, type, language, is\_retracted, cited\_by\_count |
| `authors` | author\_id (OpenAlex A), s2\_author\_id, orcid, display\_name |
| `authorships` | paper\_id, author\_id, position, institution\_ids\[\] |
| `institutions` | institution\_id, name, country, type |
| `references` | citing\_id, cited\_id, source (openalex / s2), contexts\[\], intents\[\], is\_influential |
| `provenance` | run\_id, paper\_id, route (seed / ai\_report / back / forward / author / semantic), hop, via\_id |
| `facets` | paper\_id, facet, text, model\_id, prompt\_version, created\_at |
| `embeddings` | paper\_id, facet, dim, vector, embedding\_model\_version |
| `clusters` | run\_id, cluster\_id, facet, resolution, label, description, stability, size |
| `paper_cluster` | run\_id, paper\_id, facet, resolution, cluster\_id |
| `pm_matrix` | run\_id, problem\_cluster, method\_cluster, year, count, first\_year |
| `citation_flows` | run\_id, facet, year, from\_cluster, to\_cluster, count |
| `author_year` | run\_id, author\_id, year, facet, cluster\_dist (json), centroid\_ref |
| `signals` | run\_id, signal\_type, subject (json of cluster / author / pair ids), score, rank, features (json), evidence\_paper\_ids\[\], judge\_score, judge\_rationale |
| `integrity_flags` | run\_id, flag\_type, subject, measured\_value, field\_baseline, evidence (json); restricted access |
| `runs` | run\_id, seed\_ids\[\], params (json), cutoff\_year, model versions, spend\_usd, started\_at |

## Milestones and build order

Build in five milestones, each ending in a checkable exit criterion. Pick one pilot field the team or advisers know well, so outputs can be judged by eye from M2 onward.

1. **M1 — Data foundation** (Phases 1–2)
   - Ingestion clients for OpenAlex and S2 with caching, budget stop and backoff.
   - Expansion with provenance for one pilot seed.
   - Explorer skeleton and the corpus view.
   - Exit: a 5,000+ paper corpus with fewer than 10% missing abstracts after S2 fill, under the run's metadata spend cap, and a reviewer has checked its composition in the corpus view.
2. **M2 — Facets and embeddings** (Phases 3–4)
   - Extraction job, embedding job and vector store.
   - Facet review view; the 200-paper gold set is labeled inside it.
   - Exit: gold-set acceptance targets met (Phase 3).
3. **M3 — Core objects and first signals** (Phase 5, 6.1, 6.4a, 6.4b)
   - Stable labeled clusters, problem × method matrix, citation flows, share growth, limitation → innovation, method transfer.
   - Map with time axis (papers, clusters, activity shift, bridges layers), cluster review, and the grid.
   - Exit: an expert works through the top 20 bridges on the map and rates at least a few as worth pursuing; ratings stored in `review_actions`.
4. **M4 — Remaining signals and backtest** (6.2, 6.3, 6.4c–d, 6.5, Phase 8)
   - Each signal ships as its layer (status badges, novel papers, citation flows, author migration, people).
   - Exit: precision@20 reported for every signal at four cutoffs; signals that lose to their baseline are cut or reworked.
5. **M5 — Integrity module and reporting** (Phase 7)
   - Restricted-access flags table, validation against known retractions.
   - Field report generated from the explorer's views and review ratings, with static exports.
   - Exit: measured precision on known cases documented before anyone sees live flags.

## Risks and open questions

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Seed bias | "Field is moving away from X" may only mean the seed's neighborhood ignores X | Semantic expansion route; report signals with and without it |
| Author disambiguation errors | False productivity outliers, fake topic jumps, wrong seeders | ORCID and co-author checks before any author-level output |
| Citation lag | Recent work looks unimportant | Semantic signals for the last 2 years; bibliographic coupling |
| Cluster instability | Signals change between runs | Multi-seed consensus, stability threshold, fixed clusters per run |
| Invented limitations | False bridges and false open problems | Explicit-only prompt rule, null-rate monitoring, gold set |
| LLM future knowledge in backtests | Inflated precision | Judge off, or pre-cutoff model, in backtests |
| Missing abstracts | Gaps in some publishers' fields | S2 fill; `title_only` flag and down-weighting |
| API pricing or terms change | Cost or access shifts | Raw-response cache; snapshot path ready |
| Integrity flags leak or are misread | Harm to named people, legal exposure | Restricted table, no scores or adjectives, human review |

Open questions for the team:

- [ ] Which pilot field and seed papers?
- [ ] Which extraction LLM, and its cost per million tokens?
- [ ] GPU available for embeddings, or a hosted option?
- [ ] DuckDB or Postgres as the main store?
- [ ] Who owns access decisions for the integrity module?

## Sources

- [OpenAlex: Authentication & Pricing](https://developers.openalex.org/api-reference/authentication.md)
- [Semantic Scholar Academic Graph API spec (OpenAPI mirror)](https://gist.github.com/alexandreteles/c8bc00830e97eefa961e26c49aa666e7)
- [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2)
- [EmbeddingGemma overview, Google AI for Developers](https://ai.google.dev/gemma/docs/embeddinggemma?authuser=7)
