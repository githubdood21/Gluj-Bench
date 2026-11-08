# Phase 8 - results, native history, sharing and workflows

Implemented in the working tree on 2026-11-02. Interactive UI verification remains
with the user; no desktop access was requested. See the
[confirmation checklist](phase-8-verification.md). Cross-hardware qualification and
measured runtime-default calibration remain Phase 9 work.

## Reading results

The primary measurement defaults to the largest measured dataset. Buffered compute
and explicit memory/transfer results offer a size/operation selector. Its median,
observed minimum/maximum and observation count appear before expanded details.
Reference drift and validation scope are shown near the measurement when recorded;
missing legacy scope is explicitly unknown. Raw measurements, inferred descriptions
and experimental tuning guidance remain labeled separately. Full scaling charts,
per-size tables, derived traffic and metadata remain available in details.

Summary rows emphasize Latest and Typical. Typical is the median of matching
recorded run values currently in the result view, rather than a pooled median of
all raw samples or all native history. Opening historical runs adds them to this
view without duplicating an already-open native run. Historical best is optional,
labeled and scoped to matching configurations in the current view. It does not
replace the selected dataset with a peak from another size.

Comparisons use the selected measurement, recorded units and existing workload/
configuration/revision compatibility rules. Changes use neutral higher/lower
throughput or latency/cost wording, descriptive variation and explicit missing/
incompatible baseline reasons. A percentage difference is not a significance test.
Legacy results without resolved identities remain limited to their recorded context.

Cache-oriented shader and Streaming / VRAM are modes of the same GPU local-memory
family. Their names and explanations appear in setup and results. Automatic cache
probes infer transitions; explicit sizes do not guarantee residency. Different
kernels and shader/native copy paths remain separate measurements rather than one
scaling curve. This preserves Phase 7 measurement definitions while clarifying why
there are two choices.

## Durable native history and baselines

Run history uses the Phase 1 immutable records in `run-history/runs/`, with its
compact index. Browsing is newest-first, searchable by workload/ID/name/notes, and
paged at 30 records. Individual native records load on a background thread.
Inspect shows recorded time (UTC, or unknown), completion/source/coverage state,
validation, application/build provenance and raw observation count. Raw observations
are paged 30 at a time with condition, metric value/unit, timing scope, active and
elapsed time, and realization. Full arrays remain in native export.

Names, notes, pins and archive state live in additive `annotations.json`, separate
from measured records. Pinning restores a named baseline after restart; Compare
with this run adds it to the comparison chooser in the current session. Archived
records are hidden by default and can be shown/restored. Archive does not delete
samples or remove a pinned baseline's measured data.

Delete native record requires an explicit second action and removes only that
record and its annotation. Original legacy/PNG source files and the separate latest
`results.json` cache remain unchanged. Clearing the result view does not delete
history. The latest-per-configuration cache no longer silently evicts unrelated
configurations at 128 entries. Repeated native runs remain separate immutable
records. History storage has no automatic age/count eviction.

Storage, malformed metadata, unsupported versions and migration failures surface
errors while preserving prior source files. Annotation names are limited to 256
UTF-8 bytes and notes to 16 KiB; errors do not silently truncate text. Native data
is not normalized in place for display. Imported native records receive new local
IDs and retain original run ID, source chain, time, requested/resolved conditions,
hardware, validation, samples and build provenance. They remain imported comparison
sources; imported pins are cleared until chosen locally.

## Export scope and limits

PNG snapshots now embed exactly their checked summary results by default, with
indices rebased to that exported list. Include all completed results in embedded
data is an explicit option; the picture still shows the checked rows. Older
version-1 PNGs remain importable. PNGs contain summary measurements, not full native
raw records. They support up to 128 displayed rows, 10,000 embedded summaries and
64 MiB encoded/decompressed metadata limits. Oversized payloads fail explicitly;
no sample data is silently removed to fit.

Run history offers selected native JSON export, including complete raw observation
arrays, realizations, blocks, anchors, phase timings and annotations. Checked runs
across history pages define the selection; when none are checked, the inspected
run is exported. Native import/export limit is 256 MiB. Export smaller selections
when this limit is reached. Record loading and native JSON import/export run in the
background; benchmark starts wait for history I/O to finish. Export publishes a
flushed temporary file, preserving an existing destination on pre-publication
failure. Native imports can retain successfully saved earlier records if a later
record encounters a storage error; the source remains unchanged.

## Tasks and queue

Prepare a task offers CPU check, Memory check, Large dataset (256 MiB target within
budgets), and comparison of custom conditions. These create reviewable custom plans;
the user chooses Run custom plan to start. Existing descriptor resolution and
capability checks apply. Unsupported sizes/configurations produce explicit errors.
No task predicts application performance or recommends rerunning due to variability.

Plan status shows each condition, completion state, measurement-time floor and
estimated test-buffer allocation (or an explicit execution-time resolution note).
These are minimums/estimates; setup, references, sweeps and bounded extensions add
time. Skip removes only a selected pending condition. Cancel this condition sends
cancellation for the current request and continues queued work after acknowledgement.
Global Cancel stops the current run and skips all pending conditions. Worker failure
retains failed/skipped status rather than implying completion.

`run-history/latest-plan.json` retains the latest plan status. Restart displays prior
completed/partial status; previously running/pending work is marked cancelled/skipped
and never resumes automatically. Older plan sessions are not a new plan-history
archive; native run history remains the durable measurement source.

## Accessibility and verification scope

History and primary selectors use standard keyboard-accessible widgets. Result rows
have an Open button as an alternative to pointer-only selection. Full names and
configuration context remain in the detail inspector; status/source/validation are
textual and do not depend on color. Charts retain table alternatives. Empty/error
states describe available data and failures without generating rerun advice.
Phase 9 records interactive Windows checks, including native dialogs and the
local desktop layout. Other display scales, assistive technologies and full keyboard
traversal remain unqualified; compilation/unit tests do not prove those behaviors.
Existing platform font/scaling behavior is retained.

## Validation and packaging record

307 workspace tests passed, with 24 hardware/preview tests intentionally ignored.
Strict Clippy, formatting, whitespace checks and the optimized workspace build
passed. One release pair was placed in `build/current/`; all 80 existing current
settings/history files were hash-verified unchanged. Four additional data files
found beside the debug build were copied and hash-verified in
`build/recovered-debug-data/phase-8/` before cleaning Cargo caches. No interactive
UI or live benchmark execution was performed for this phase.

Phase 9 supersedes the phase-local verification limitation above: actual UI, native
history/export, worker recovery and live benchmark checks are documented in
[local qualification](phase-9-qualification.md).
