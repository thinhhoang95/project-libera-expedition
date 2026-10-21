# Feature: Embeddings and vector store (Phase 4)

Status: implemented · Code: `src/landscape/embed.py` · CLI: `landscape embed` (abstracts are also embedded by `landscape semantic`)

## Model

- **EmbeddingGemma 2** (`google/embeddinggemma-2`, Apache 2.0, 768-d, Matryoshka truncation
  to 512/256/128), loaded through sentence-transformers. Only the text tower is loaded:
  `config_kwargs={"vision_config": None, "audio_config": None}` gives 271M parameters.
- One symmetric prefix for every text: `task: sentence similarity | query: `. Using the same
  prefix everywhere keeps all facets in one task space, which the recipe asks for. The prefix
  is part of the recorded version string
  (`google/embeddinggemma-2|task: sentence similarity | query:|d768`), stored on every vector
  and in `runs.model_versions.embedding`.
- dtype: float32 on CPU, bf16 on CUDA when supported. float16 is never used, because the model
  card warns it produces NaNs.
- No citation-trained embeddings (SPECTER etc.) are used.

### Hardware note

This machine has no CUDA GPU (Intel Iris Xe only), so encoding runs on the CPU
(i9-12900HK). Throughput is about 900 tokens/s. On the pilot that meant about 25 minutes for
5,400 abstracts. The facet texts were much faster (around 60 per second), because they are
short. Profiling shows all of that time in matrix multiplication, so the
model itself is the limit and there is no I/O bottleneck to fix. To keep the pilot within
reach, `max_seq_length` is 384 tokens. A 20k-paper corpus will want a GPU.

## What gets embedded

| Vector | Text |
| --- | --- |
| `abstract` | `title. abstract`, or the title alone for title-only papers |
| `problem`, `method`, `innovation`, `result` | faithful facet text |
| `limitations`, `domain_data` | only when non-null |
| `problem_gloss`, `method_gloss` | gloss text |

Everything is stored at 768 dimensions and unit length.

## Store layout

- `data/shared/vectors/{facet}.parquet`: `paper_id`, `vector` (FixedSizeList<float32, 768>),
  `embedding_model_version`. DuckDB can query these files directly.
- `data/shared/vectors/{facet}.faiss`: an HNSW index (M = 32, inner product) over the same vectors,
  with `{facet}.faiss.ids.parquet` holding the row → paper_id mapping.
- Jobs are incremental: only `(paper_id, facet)` pairs missing for the current model version
  are embedded. If a file contains vectors from another version, it is deleted and rebuilt, so
  versions are never mixed.

## Helpers used downstream

- `truncate(v, d)`: Matryoshka truncation followed by L2 re-normalization. 256-d is used for
  kNN graphs and clustering, and 768-d for precise matching.
- `blended(facet, alpha, ids, dim)`: the gloss blend from the recipe,
  `normalize((1-α)·e_f + α·e_gloss)`, for `problem` and `method`. It is computed on the fly
  and never stored. A paper without a gloss keeps its plain facet vector.

## Derived vectors

- **Author-year centroids** (mean of an author's paper vectors in a year, α = 0, per facet)
  are written by Phase 5 to `data/projects/<run>/vectors/author_year_{facet}_{run_id}.parquet`, keyed
  `author_id|year`.
- **Cluster centroids** (768-d) are written to `data/projects/<run>/vectors/cluster_centroids_{run_id}.parquet`.

## Tag normalization

`normalize_tags` embeds every distinct `problem_class` and `method_family` tag in the run and
merges near-synonyms. The result goes to
`tag_vocab(run_id, facet, tag, canonical, n, sim_to_canonical)`, and Phase 5 uses the
canonical tags for the alternative `pm_tag_matrix`.

The merging is greedy and non-transitive. Tags are visited from most to least frequent. Each
tag joins the most frequent existing canonical tag with cosine > 0.93, or else becomes a
canonical tag itself. There are two reasons for this design, both found on the pilot:

- **Connected components chain.** At 0.9, A~B~C linking collapsed 2,056 problem tags into 154
  and merged "systematic literature review" into "distributionally robust optimization".
- **Specializations score like synonyms.** On this model, true synonyms score 0.94–0.97, but
  so do specializations: robust vs distributionally robust optimization scores 0.962, and RL vs
  deep RL scores 0.974. A tag whose word set is a strict subset or superset of the canonical's
  is therefore never merged.

Pilot result: 2,056 → 1,261 problem classes and 2,460 → 1,621 method families. A few merges
at the edges are still wrong (for example "distributed optimization" → DRO). They are meant to
be reviewed in the facet review view, which is not built yet.

### LLM tag pass (off by default)

`canonicalize_tags_llm` (`LabelParams.canonicalize_tags`) is a second pass that runs once the whole
vocabulary is known. Canonical tags are visited by frequency. Each one is shown its 8 nearest more
frequent canonicals (cosine ≥ 0.75, no subset or superset word sets), and the label model picks a
synonym or none. Merges are written to `tag_vocab` with `merged_by = 'llm'`; the embedding merges are
marked `'embedding'`.

On the pilot (2026-10-21, `deepseek-v4-pro`) it merged only 22 method and 28 problem tags, and most of
those merges were wrong: the tag was mapped to a broader or merely related one ("grey wolf optimization"
→ "metaheuristic optimization", "expectation maximization" → "maximum likelihood estimation"). The pilot
vocabulary was restored to the embedding merge. Most tags are not hidden synonyms anyway: 1,082 of 1,621
method canonicals occur once. Keep the pass off unless it is run with a stronger model and checked.

## Pilot run results

See [pilot-run-dro.md](pilot-run-dro.md).
