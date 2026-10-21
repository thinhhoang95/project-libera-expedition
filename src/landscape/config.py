"""Paths, secrets and default parameters shared by every phase."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("LANDSCAPE_DATA", ROOT / "data"))
# Layout (see docs/features/11-web-app.md):
#   shared/      what every project reuses: the DuckDB file (papers, facets and run-keyed tables), the raw
#                response cache and the paper-level vector store, so a paper is fetched, extracted and
#                embedded once whichever project reaches it
#   projects/<project>/   everything one project builds (cutoff runs <project>@T<year> included)
#   app/         the web app's job queue, review notes and read replica of the database
SHARED = DATA / "shared"
RAW_DIR = SHARED / "raw"  # local stand-in for object storage
VEC_DIR = SHARED / "vectors"
# LANDSCAPE_DB points a process at another database file, e.g. the web app's read replica
DB_PATH = Path(os.environ.get("LANDSCAPE_DB", SHARED / "landscape.duckdb"))
PROJECTS = DATA / "projects"
APP_DIR = DATA / "app"


def project_id(run_id: str) -> str:
    """The project a run belongs to: cutoff runs <project>@T<year> live with their project."""
    return run_id.split("@", 1)[0]


def project_dir(run_id: str, *parts: str, mkdir: bool = True) -> Path:
    """projects/<project>/<parts...>. A run identifier must never become an arbitrary filesystem path."""
    from urllib.parse import quote

    segment = quote(project_id(run_id), safe="")
    if segment in ("", ".", ".."):
        segment = "run-" + segment
    d = PROJECTS.joinpath(segment, *parts)
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d


def run_vectors(run_id: str) -> Path:
    """Run-level vectors (cluster centroids, author-year centroids), cutoff runs included."""
    return project_dir(run_id, "vectors")


# The production models (paper bridges, early work, community warmth). The committed ones were fitted on the
# pilot and serve every project; `scripts/phase8b/fit_models.py` (pipeline step "fit-models") fits a project's
# own into projects/<project>/models/, which then win for that project and its cutoff runs.
DEFAULT_MODELS = Path(__file__).with_name("models")


def model_file(run_id: str, name: str) -> Path:
    own = project_dir(run_id, "models", mkdir=False) / f"{name}.json"
    return own if own.exists() else DEFAULT_MODELS / f"{name}.json"

# Project env files win over the shell environment: a machine-wide OPENAI_API_KEY must not
# shadow the OpenRouter key kept in .env.local.
_ENV = {**os.environ, **dotenv_values(ROOT / ".env"), **dotenv_values(ROOT / ".env.local")}

OPENALEX_API_KEY = _ENV.get("OPENALEX_API_KEY")
# The OpenRouter key is stored as OPENAI_API_KEY in .env.local (sk-or-...).
OPENROUTER_API_KEY = _ENV.get("OPENROUTER_API_KEY") or _ENV.get("OPENAI_API_KEY")
S2_API_KEY = _ENV.get("S2_API_KEY")  # optional; unauthenticated pool is heavily rate limited


@dataclass
class ExpansionParams:
    """Phase 2 defaults (build recipe, 'Default parameters')."""

    max_hops: int = 2
    hop2_keep_per_paper: int = 15
    forward_years: int = 5
    forward_per_seed: int = 1000
    forward_per_paper: int = 100
    author_years: int = 5
    works_per_author: int = 50
    max_authors: int = 300
    # "auto" = the semantic_auto_quantile of the corpus's own nearest-centroid similarity.
    # A fixed cosine (the recipe's 0.75) is not portable across embedding models: with
    # EmbeddingGemma 2 even off-topic papers score ~0.87.
    semantic_threshold: float | str = "auto"
    semantic_auto_quantile: float = 0.10
    semantic_topics: int = 8
    semantic_candidates_per_topic: int = 250
    paper_budget: int = 20_000
    # Per-route caps so one route cannot starve the others.
    route_caps: dict = field(
        default_factory=lambda: {"back": 4000, "forward": 6000, "author": 6000, "semantic": 4000}
    )
    metadata_spend_cap_usd: float = 2.00
    freshness_days: int = 30
    s2_contexts_for: str = "seed_hop1"  # which papers get S2 citation contexts: none | seed_hop1 | all
    s2_time_budget_s: float = 300  # per S2 annotation step; afterwards only cached S2 data is used


@dataclass
class ExtractionParams:
    model: str = "deepseek/deepseek-v4.1-flash"
    prompt_version: str = "v1"
    concurrency: int = 48
    # "Without quantization": exclude fp4/int-style endpoints; fp8 is DeepSeek's native precision.
    quantizations: list = field(default_factory=lambda: ["fp8", "bf16", "fp16", "fp32"])
    llm_spend_cap_usd: float = 5.00


@dataclass
class LabelParams:
    model: str = "deepseek/deepseek-v4-pro"
    concurrency: int = 16
    label_fine: bool = False  # fine clusters get c-TF-IDF terms only unless set
    llm_spend_cap_usd: float = 2.00


@dataclass
class EmbeddingParams:
    model: str = "google/embeddinggemma-2"
    # One symmetric prefix used for every text so all vectors live in the same task space.
    prefix: str = "task: sentence similarity | query: "
    batch_size: int = 32
    store_dim: int = 768
    graph_dim: int = 256

    @property
    def version(self) -> str:
        return f"{self.model}|{self.prefix.strip()}|d{self.store_dim}"


@dataclass
class ClusterParams:
    k: int = 20
    drop_edge_pct: float = 5.0
    resolutions: dict = field(default_factory=lambda: {"coarse": 1.0, "medium": 2.0, "fine": 6.0})
    n_seeds: int = 5
    stability_jaccard: float = 0.6
    min_cluster_size: int = 5
    # unstable clusters this large and at least as coherent as the stable ones' 25th percentile
    # are kept as "hub" clusters and reported separately (see docs/features/05-core-objects.md)
    hub_min_size: int = 30
    hub_min_share: float = 0.01
    facets: list = field(
        default_factory=lambda: ["problem", "method", "innovation", "result", "limitations"]
    )
    gloss_alphas: list = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.8])
    seed_values: list | None = None  # user override for the 5 Leiden seeds


@dataclass
class SignalParams:
    """Phase 6 defaults. Weights are starting values for the Phase 8 backtest to tune."""

    # clusterings that get cluster-level signals (facet, alpha), medium resolution
    cluster_sets: list = field(default_factory=lambda: [["problem", 0.3], ["method", 0.3],
                                                        ["innovation", 0.0], ["result", 0.0]])
    min_papers_per_year: int = 100  # a year is a cutoff once it and the 2 before have this many papers
    top_n: int = 50
    top_n_institutions: int = 30
    # 6.2 / 6.3
    novelty_facets: list = field(default_factory=lambda: ["problem", "method", "innovation", "abstract"])
    emergence_min_size: int = 10
    # "backtested" (default) or "recipe" (z(g)+z(dg)+z(new authors)-z(ref age)+z(novelty)).
    # The Phase 8 backtest found the recipe score anti-predictive on the pilot; see 08-backtest.md.
    emergence_score: str = "backtested"
    emergence_weights: dict = field(default_factory=lambda: {"momentum": 1.0, "dg": 1.0,
                                                             "new_author_share": -1.0})
    direction_min_size: int = 20
    emerging_quantile: float = 0.9
    niche_z: float = 1.0
    a_min: int = 10
    grid_alpha: float = 0.3
    # 6.4
    bridge_alpha: float = 0.5
    lim_topk: int = 5
    lim_flow_penalty_power: float = 0.5  # recipe: 1.0
    generic_limitation_drop: float = 0.2  # drop the most generic 20% of limitation statements
    transfer_neighbors: int = 10
    transfer_method: str = "item_cf"  # "item_cf" (backtested) or "recipe" (centroid-neighbour share)
    paper_bridges_per_cluster: int = 10
    paper_bridges_gems_per_cluster: int = 5
    unresolved_min_papers: int = 5
    unresolved_match: float | str = "auto"  # recipe 0.7; "auto" = quantile of lim->inn similarity
    unresolved_match_quantile: float = 0.99
    limitation_group_quantile: float = 0.9
    semantic_pair_top: float = 0.05
    semantic_pair_flow_pct: float = 10
    bridge_keep: int = 200
    judge_top: int = 50
    judge_model: str = "deepseek/deepseek-v4-pro"
    judge_spend_cap_usd: float = 2.00
    # 6.5
    influence_facet: str = "problem"
    influence_alpha: float = 0.3
    cd_min_citers: int = 5
    cd_min_refs: int = 5
    cd_years_before_first_cutoff: int = 5  # rank CD only for papers from the analysed period
    broker_min_papers: int = 3
    recent_years: int = 2
    coupling_min_shared: int = 2


@dataclass
class RunParams:
    expansion: ExpansionParams = field(default_factory=ExpansionParams)
    extraction: ExtractionParams = field(default_factory=ExtractionParams)
    labels: LabelParams = field(default_factory=LabelParams)
    embedding: EmbeddingParams = field(default_factory=EmbeddingParams)
    clusters: ClusterParams = field(default_factory=ClusterParams)
    signals: SignalParams = field(default_factory=SignalParams)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunParams":
        return cls(
            expansion=ExpansionParams(**d.get("expansion", {})),
            extraction=ExtractionParams(**d.get("extraction", {})),
            labels=LabelParams(**d.get("labels", {})),
            embedding=EmbeddingParams(**d.get("embedding", {})),
            clusters=ClusterParams(**d.get("clusters", {})),
            signals=SignalParams(**d.get("signals", {})),
        )
