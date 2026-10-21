# Feature: Web app usability pass (Phase 11b)

Status: implemented 2026-10-21 · Code: `web/src/` only (no API changes) · New files: `components/overlay.ts`,
`components/Toasts.tsx` · Checked: `npm run build` (type check and bundle), then each change in the running app
on `dro-pilot`, light and dark

After facet categories and their details sheets were in (see [11-web-app.md](11-web-app.md)), a review of the
whole app picked the ten changes that most improve everyday use. This batch adds all ten. None of them changes
a number, a ranking or an evidence note: they change how the views load, link and read.

## The ten changes

| # | Change | Before | Now | Where |
| --- | --- | --- | --- | --- |
| 1 | Sheets, drawers and dialogs behave as modals | The page scrolled behind an open sheet; focus stayed on the page | The page behind is locked; focus moves in and returns on close; Tab stays inside; Esc closes only the top one; they slide or fade in; keyboard focus is always visible | `overlay.ts`, `CategorySheet.tsx`, `PaperDrawer.tsx`, `Layout.tsx`, `index.css` |
| 2 | Charts follow the theme | Charts kept the old colours after a light/dark switch; the facet heatmap stayed light in dark mode | Charts redraw on a theme change; heatmap and muted colours are theme tokens; a chart's own axis settings no longer drop the themed gridlines | `Plot.tsx` (`useThemeKey`, `HEAT`, `MUTED`), `index.css` |
| 3 | No blank screens while loading | Changing facet, α or resolution on the map blanked the whole page, controls included | The old map stays under a loading veil; searches keep their previous results; a thin bar at the top shows new loads; pages and Plotly load when first opened; page skeletons instead of a spinner | `api.ts` (`keep`), `Layout.tsx` (`TopProgress`), `main.tsx` (lazy pages), `ui.tsx` (`PageSkeleton`) |
| 4 | One time window per project | Facets, the map and the sheets each had their own window; the map had a different slider | One shared window, kept across reloads; the map uses the same time seeker as the other views | `context.tsx` (`useSharedWindow`), `MapGrid.tsx`, `Explore.tsx`, `CategorySheet.tsx` |
| 5 | Readable map labels | Labels overlapped, were cut off at the edges and drawn under the points | Labels are placed without overlap (selected cluster, then flagged ones, then the largest in the window), turn inward at the edges, and sit above the points on a translucent backing | `MapGrid.tsx` (`placeLabels`) |
| 6 | Notices | A finished job showed only in the bell; "Copy link" said "Copied" even when it failed | Finished or failed jobs pop up with a link to the pipeline, and a finished one refreshes the project's views; queuing a job and copying a link confirm, and a failed copy says so | `Toasts.tsx`, `Layout.tsx` (`Notifications`), `Pipeline.tsx`, `Experiments.tsx` |
| 7 | Smarter search (⌘K) | One request per key press; no sign of loading | Searches after a pause in typing, shows a spinner, highlights the match, and offers actions: theme, sidebar, copy link, shortcuts, pipeline, job queue, new project | `Layout.tsx` (`SearchPalette`), `context.tsx` (one shared theme) |
| 8 | Tables export and fit | CSV only on My paper's warm communities; Δ share cut off in narrow cards | Every table exports what it shows as CSV (corner button on hover); category tables drop the sparkline and size columns in narrow cards; sortable headers announce their sort | `DataTable.tsx` (`downloadCsv`, `narrow`), `Facets.tsx` |
| 9 | Start here shows where you are | The section bar did not mark the current section; stat labels were cut off | The section bar marks the section on screen and keeps it in view; stat labels wrap | `StartHere.tsx` (`useScrollSpy`), `ui.tsx` |
| 10 | Orientation aids | Every browser tab read "Libera Expedition"; My paper was empty until you typed | Tabs name the page and project ("Facets · dro-pilot"); My paper suggests recently opened papers and the field's *Read now* list | `Layout.tsx` (`useDocumentTitle`), `MyPaper.tsx` (`Suggestions`) |

How these behave in use is described in [11-web-app.md](11-web-app.md#facets-details-sheets-and-navigation)
(the paragraphs from *Finding your way* on).

## Behaviour that changed

- A details sheet opened outside Facets starts on the shared window, or the last three complete years if none
  is set. It used to start on every year. This now matches the category tables that open it.
- The map still shows every year until a window is chosen. A map link's `w` sets the shared window.
- The theme is one app-wide value: the sidebar switch and the search actions change the same setting.

## How it was checked

- `npm run build` passes (TypeScript type check and the Vite bundle).
- In the running app on `dro-pilot`: Start here, Facets, Map & grid, My paper, Research lab, Experiments,
  Corpus view, Cluster review and Pipeline load with no console errors.
- Overlays: with a paper open over a sheet, Esc closed the paper, then the sheet; the page scroll was locked
  until both closed; focus returned to the category link that opened the sheet.
- Shared window: "Last 5" on the map showed as 2021–2025 on Facets and in a sheet opened from there, and in
  the map link (`w=2021-2025`).
- Map: switching facet kept the controls on screen; labels no longer overlap, in both themes.
- Theme: switching to dark from the search box redrew the trend chart and the facet heatmap in dark colours.
- CSV: the Start here category table exported its columns as shown (Problem, Papers, window, Share, Δ share).
- Search: typing "wasserstein" listed categories and papers with the match highlighted.

## Limits

- The finished-job notice was not seen live: no pipeline job ran during the check. It reads the same events as
  the bell, so it should fire when one ends.
- The browser preview used for the check blocks clipboard access, so "Copy link" showed its failure notice
  there. A normal browser on localhost allows the copy.
- Label placement estimates text width from character count and font size. Very unusual fonts or zoom levels
  can leave a small overlap, or hide a label that would have fitted.
- The shared window is stored in the browser (`localStorage`), so it is per browser, not per user or server.
