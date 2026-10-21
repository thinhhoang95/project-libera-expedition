# Feature: Field report (AI field intelligence)

Status: implemented · Code: `src/landscape/field_report.py`, `report_chat.py`, `model_catalog.py`, `src/landscape/web/views_report.py`, `web/src/pages/FieldReport.tsx`, `web/src/components/{ReportChat,ModelPicker,ReportMarkdown}.tsx` · CLI: `landscape field-report`

A dedicated tab (sidebar › Explore › **Field report**, `g i`) where a strong model reads every product of a
project and writes a field intelligence report. The model chooses the report's structure and length. Every
claim links to the papers and categories behind it.

## Run

```bash
uv run landscape field-report --run-id dro-pilot --model claude              # Claude Opus 5.5
uv run landscape field-report --run-id dro-pilot --model gpt --focus "..."   # GPT-6.1 Sol, with a reader focus
```

In the app, *Write the report* queues the same command as a **quick-lane** job, so it never waits for a
pipeline step and reads the replica. It needs a built portfolio (the Core map template).

**Models.** Every model on the page can be any OpenRouter model. The pickers read OpenRouter's catalogue
(`GET /api/openrouter/models`, cached for an hour in `data/app/openrouter_models.json`) and show, for each
model:
- price per million input and output tokens;
- the discount on cached input;
- the context window. Models whose window cannot hold the prompt are greyed out.

The choice is remembered per browser. The defaults and suggested models are below.

| Use | Default | Suggested alternatives |
| --- | --- | --- |
| Writing the report | `anthropic/claude-opus-5.5` ($4 / $20) | `openai/gpt-6.1-sol` ($2 / $10), Sonnet 5.5, Gemini 3.1 Pro, DeepSeek v4 Pro |
| The chat | `deepseek/deepseek-v4.1-flash` ($0.05 / $0.38, cached input $0.001) | Gemini 3.6 Flash, Haiku 5.5, GPT-6 Luna, DeepSeek v4 Pro, Sonnet 5.5 |

On the CLI, `--model` takes `claude`, `gpt` or any slug. `LIBERA_REPORT_MODEL_CLAUDE` or
`LIBERA_REPORT_MODEL_GPT` overrides a preset.

**Report generation.** The call streams. Reasoning is set to the effort closest to `high` that the model
accepts, read from the catalogue. `max_tokens` is 64000, reasoning included. On the pilot the input is about
60k tokens, so a report costs roughly $0.5–1 with Opus and about half that with GPT-6.1 Sol. The report
prompt is sent once, so OpenAI models get `prompt_cache_options.mode = explicit` with no breakpoint. This
skips the 1.25× cache write that GPT-5.6 and later charge.

## The evidence pack

`evidence_pack(p)` calls the functions behind the web app's views, so the model sees the same numbers as
the reader. It keeps only the rows that carry signal, which makes about 226k characters on the pilot (1.8 MB
raw):

| Section | From | Kept |
| --- | --- | --- |
| project, map | overview | seeds, as-of year, papers, years, routes, external reference share |
| communities | overview | all 25: size, recent papers, share and change, papers per year |
| facet_categories | catalogue | method, innovation, result and limitation categories: top 25 each, with description, terms and yearly counts |
| import_map, imports, coverage_gaps | overview, external-imports | top 30 outside-field and 20 adjacent imports by model score, plus the gaps |
| read_now, reading_paths, tools_to_learn, survey_gaps | people, overview | early work (25); per community 5 path papers and 4 tools; survey gaps and thin coverage |
| entry_routes, entry_effects | overview | strategies with at least 5 entries; the adjusted effects with their intervals |
| opportunities | overview, bridges | gems, limitation → innovation (with the judge's rationale), method transfer and underconnected pairs, each with its timing; cluster pairs ranked 11–30 |
| papers_to_bring_in | paper-bridges | per community, 4 likely papers and 3 hidden gems |
| second_waves | overview | top 15 per kind by momentum |
| people_descriptive_only | people | top 10 seeders, brokers and institutions |
| sampling_audit, direction_robustness, underconnected_all | lab | the audit, every topic's within-route change and sign flip, and the top 20 deficits |
| evidence_ledger | experiments | the portfolio verdict table and its notes, verbatim |

## The prompt

The system prompt (`field_report.SYSTEM`) fixes the reader, the rules of evidence and the link format, but
not a template. The model is asked to:
1. Work privately through what the map is and what it cannot see.
2. Rank the questions researchers ask (shape, real momentum, what to read, ideas moving, openings, how to
   get in, missing surveys and tools, blind spots) and drop those with no usable evidence.
3. Cross-reference products. Convergence (for example, a growing method line plus an underconnected target
   plus a second wave) outranks any single list's top row. It should also flag divergence.
4. Check every claim against the ledger. *Use* products can back a recommendation, *use with care*
   products give a lead to check, and *describe* products never predict. A direction is stated only where
   the within-route change agrees.
5. Keep only what changes a decision. The report opens with a title that states the main finding and a
   bottom line, and ends with what the map cannot tell the reader.

An optional **focus** (for example "a control theorist looking for a two-year direction") shapes the
priorities. The model must still report anything a reader would be wrong to miss.

Links: `[title](paper:W…)` opens the paper card and `[label](cat:<cluster id>)` opens the category's details
sheet. The prompt allows only ids from the pack.

## Artefacts

`data/projects/<project>/reports/`:
- `<stamp>-<model>.json`: the report (markdown, title, model, focus, as-of year, tokens, cost, seconds,
  replica, evidence hash);
- `<stamp>-<model>.evidence.json`: the exact pack the model read, so you can audit any claim.

Spend stays in the report file. The quick lane opens the replica read-only, as `my-paper --read-only` does,
so the spend is not added to `runs.spend_usd`.

## The chat (side panel)

A Cursor-style assistant beside the report (`report_chat.py`, `ReportChat.tsx`). It reads the report and the
full evidence pack the report was written from.

- **Open it** with *Chat* in the header, or press ⌘L (Ctrl+L). Select a passage of the report and press ⌘L,
  or click *Ask about this*, to attach the passage as a quote. Press ⌘L again in the chat to switch to the
  Reports panel (generator, contents, history).
- **Streaming:** the answer appears as it is written, one render per animation frame. The model's thinking
  shows as a collapsible "Thought" line. Esc or the stop button cancels.
- **Each turn** has *Regenerate* and *Copy*, and a footer with the provider, input tokens, cached tokens,
  output tokens, cost and time. The composer shows the conversation's total.
- **Links** work as in the report: `paper:` links open the paper card, `cat:` links open the details sheet.
- **Editing the report:** asked to change the report, the model answers with `report-edit` blocks
  (FIND / REPLACE). Each block shows as a diff card, with:
  - a check that the passage occurs exactly once;
  - *Apply*, *Reject* and *Undo*.

  Edits change the report file (`POST /reports/<id>/edit`). The model's original is kept, and every edit is
  logged. *Restore the model's original* puts it back. Undoing everything clears the "edited" mark.
- **Conversations** are saved beside the report (`<id>.chats.json`, up to 50) and reopen from the history
  menu.

**Prompt caching.** One answer re-sends about 62k tokens of context, so the prompt is laid out to be cached:

1. **A stable prefix.** The system message is three text blocks, always in the same order: the instructions,
   the evidence pack (serialized deterministically from the saved file), then the report. The pack never
   changes, and the report changes only when an edit is applied. The report comes last, so an edit leaves
   the pack's cache valid.
2. **Explicit breakpoints.** `cache_control` sits on the evidence pack (`ttl: 1h`, since reading pauses are
   often longer than 5 minutes), on the report, and on the latest turn, so the conversation so far is cached
   too. OpenRouter passes these to Anthropic, Alibaba and Google, and turns them into
   `prompt_cache_breakpoint` for OpenAI GPT-5.6 and later.
3. **Implicit caching.** DeepSeek, OpenAI, Gemini, Grok and Moonshot reuse the stable prefix with no
   markers.
4. **Sticky routing.** `session_id` (and `prompt_cache_key`) is `<project>:<report>:<conversation>`, so every
   turn goes to the same provider and its cache stays warm.

Measured on the pilot with DeepSeek v4.1 Flash:

| Turn | Input tokens | Read from cache | Cost |
| --- | --- | --- | --- |
| 1 | 63k | 0 | $0.0073 |
| 2 | 64k | 63k | $0.0009 (mostly output) |

The first token arrives in about 3 seconds.

## Limits

- One shot, no tools: the report model and the chat see only the pack. They cannot query a paper's full text or the database.
- The model writes the edit blocks. One whose FIND text is not exactly in the report cannot be applied; the card says so.
- The report is only as good as the map. The sampling and coverage limits in `09-portfolio.md` apply, and the
  prompt makes the model restate them.
- The report has not been evaluated against expert judgment. Treat it as a well-briefed first read, and
  open the linked evidence before acting.
