# Feature: LLM facet extraction (Phase 3)

Status: implemented · Code: `src/landscape/extract.py`, `src/landscape/llm.py` · CLI: `landscape extract`, `landscape qa`

## What it does

Each corpus abstract is rewritten by a cheap, fast LLM into the recipe's schema:

- Six faithful facets: `problem`, `domain_data`, `method`, `innovation`, `result`, `limitations`.
- Two model-knowledge glosses: `problem_gloss`, `method_gloss`.
- Two canonical tags: `problem_class`, `method_family`.
- `contribution_type` and `has_quantitative_result`.

The raw abstract stays unchanged in `papers.abstract_raw`.

## Model and provider routing

- Model: `deepseek/deepseek-v4.1-flash` on OpenRouter.
- Provider routing: `{"sort": "price", "quantizations": ["fp8", "bf16", "fp16", "fp32"], "require_parameters": true}`.
  This picks the cheapest endpoint that is not quantized below DeepSeek's native FP8 and that
  supports JSON-schema structured output. FP4/MXFP4 endpoints and endpoints with an unknown
  precision are excluded. At build time this routed to **Baidu (fp8)**: $0.035 per M input
  tokens and $0.14 per M output tokens, about **$0.00007 per abstract**.
- `temperature: 0`, `reasoning: {enabled: false}`, and a strict `json_schema` response format
  (`extract.SCHEMA`).
- The spend reported by OpenRouter (`usage.cost`) is summed per run. The job stops at
  `llm_spend_cap_usd` (default $5) and records `runs.spend_usd.llm_extraction`.

The OpenRouter key is read from `OPENAI_API_KEY` in `.env.local` (it is an `sk-or-` key).
Values in `.env.local` take precedence over the shell environment, because this machine also
has an unrelated OpenAI key set globally.

## Prompt

`PROMPT_V1` in `extract.py` is the recipe's v1 prompt, word for word. Every facet row stores
`model_id` and `prompt_version`. Papers already processed with the same pair are skipped, so
changing either one triggers a re-run.

## Run settings

- Async with 48 concurrent requests, written to DuckDB in chunks of 400.
- Papers with no abstract, or one shorter than 100 characters, are not sent. They get
  `facet_status.status = 'title_only'` so later phases can down-weight them.
- A response that fails schema validation is retried once. A malformed-JSON response also gets
  one retry inside the client. After that the paper is logged as `failed` with the error and
  skipped.

## Storage

- `facets(paper_id, facet, text, model_id, prompt_version, created_at)`: one row per field, in
  long format. Booleans are stored as `'true'` / `'false'`, and tags are lowercased.
- `facet_status(paper_id, model_id, prompt_version, status, qa_flags, error, cost_usd, created_at)`.

## Automatic QA (`landscape qa`)

Each extraction is checked automatically:

- **Schema validity:** all fields present with the right types, and `contribution_type` from
  the enum.
- **`too_long`:** a faithful or gloss field over 60 words.
- **`copied`:** a faithful field where more than 80% of its words are copied in order from the
  abstract. This is measured as word runs of three or more matched with `difflib`.
- **`tag_too_long`:** a tag over 4 words.
- **Limitations null rate per field of study:** grouped by the paper's primary OpenAlex topic
  field. A rate near zero would mean the model is inventing limitations.

`copied` is informational. Faithful fields are supposed to reuse the abstract's wording, so a
high copy rate on `problem` and `result` is expected. The flag is there to spot fields that
are pure copy-paste.

## Not done yet

- The 200-paper gold set and its acceptance check (≥90% supported, <5% unsupported
  limitations, ≥85% correct gloss variant) need human labellers in the facet review view
  (Phase 6b), which is not built yet. The data is ready for it: each facet row is keyed by
  paper, model and prompt version.

## Pilot run results

See [pilot-run-dro.md](pilot-run-dro.md).
