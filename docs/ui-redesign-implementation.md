# UI presentation redesign

Implemented 2026-11-03, following the [information audit](ui-information-audit.md).
This updates the working tree and current executable pair. The earlier versioned
release ZIP is preserved; this follow-up was not externally published.

## Delivered

| Phase | Outcome |
| --- | --- |
| UI-1 | One selected run scope; typed condition/count/resource models; matching Run action; always-visible searchable test checklist with per-test selection and inline details; custom removal; contextual core experiments and unavailable-mode disclosure. |
| UI-2 | Typed queue rows with status filters/counts and distinct Skip/Cancel condition actions; history columns, status/source filters, labeled inspector/provenance and comparison differences. |
| UI-3 | Large primary value, separate observed range/count; Overview, Datasets, Timing and Technical views; existing compute charts retained; memory/transfer curves and a size/operation/value/range/count table. |
| UI-4 | Native observation filters by condition/metric/unit/timing scope/phase and realization; chronological plot and exact paged rows; completed-only demand details alongside pending counts; compatible fixed-input core-count bars with measured one-core speedup only when available. |
| UI-5 | Short control hints and test definitions; repeated descriptions, application-use narratives and tuning advice removed from the rendered flow; hardware topology/support tables; searchable metadata; concise sharing scope. |
| UI-6 | Workspace regressions, strict Clippy, formatting, optimized build and local desktop checks. See qualification scope below. |

Plan filters change the view, never the executable plan. The full condition count
stays in its summary. Row actions carry the original execution index through
sorting/filtering. Unknown estimates remain labeled; a maximum of known allocations
is explicitly partial when any condition is unresolved. Measurement minima are
not total runtime predictions.

The default result keeps uncertainty and units beside its number. Reported legacy
values do not acquire fabricated raw coverage. Technical values over 180 bytes are
identified as recorded payloads rather than printed as JSON walls; native exports
retain the exact payload. Validation scope and comparison differences remain
inspectable, including unavailable, partial and failed states.

Raw plots use one series and realization at a time. Long plots retain chronological
bucket minima/maxima, bounded to 800 plotted points, and identify this display
reduction. Exact chronological samples remain in the table and native export.
Failed/nonfinite observations are excluded from plotted successful values, while
their status remains available in the table. Charts do not create new measurements.
New history plotting is disabled during active testing; memory chart rendering is
paused while tests run. Existing cached compute charts retain their prior behavior.

Memory curves distinguish operations, use labeled logarithmic dataset axes and show
observed ranges where recorded. Copy paths and cache/streaming modes retain their
separate result identities. Offload counts are labeled as per-observation medians;
no completion fraction is invented from independently aggregated counts. Completed
latency percentiles remain separate from pending work and pending ages.

## Qualification scope

- Automated regressions cover selected scope versus resolved execution, stable row
  indices/actions, unresolved estimates, outlier-preserving display reduction and
  fixed-input/hardware/one-core comparison requirements. Existing compatibility,
  sampling, cancellation, history, import and scoped-export checks remain required.
- Desktop review used `GLUJ_BENCH_DATA_DIR` in an isolated qualification directory.
  It verified compact configured/family summaries, the actual 56-condition CPU task,
  five-page review, filtering to eight DEFLATE conditions while the full plan stayed
  56, and keyboard Tab focus on Inspect.
- Native cancelled history showed Partial coverage, raw count and unrecorded
  validation explicitly. The observation inspector displayed a real five-point
  measured series and exact value/active/elapsed/status rows. Fixed-height pickers
  corrected a stretching issue found in that review.
- CPU cache details showed level, per-instance size, instance count, sharing and
  line size as rows. The ordinary result showed a large primary value and separate
  observed range/count. A process-local 1.25 app scale was also inspected; no Windows
  display or privacy/security settings were changed.
- Review covered the standard 1280x820 client layout and increased app scaling.
  The attempted native corner resize did not establish a smaller usable minimum.
  Exhaustive minimum-window, multi-monitor/DPI and assistive-technology coverage
  remains open; no universal accessibility or frame-time guarantee is claimed.
- Every export dialog and every hardware/result mode was not repeated interactively
  for this presentation follow-up. Existing round-trip and compatibility regressions
  cover preservation; the previous hardware qualification remains separately scoped.

Evidence and check logs are retained under `build/qualification/ui-redesign/`.
The current build has exact source/toolchain provenance in `build/current/build-info.json`.
Original user data is hash-checked before/after; generated Cargo caches are reclaimed
after the final build is copied. No per-iteration executable copies are retained.

## Final checks (2026-11-06)

The latest source passed 319 workspace tests; 25 opt-in tests were ignored. Strict
Clippy, formatting and whitespace checks passed. The interrupted optimized build
was resumed before refreshing the current app. Build provenance and executable
hashes are retained with the local qualification evidence.

All 80 previously recorded user-data files are hash-checked. Small settings/history
files found in the debug output are also preserved in the qualification directory
before generated Cargo artifacts are cleaned. The README benchmark screenshot uses
the verified standard-size custom-plan capture.

The optimized release build passed (revision `55d7a87-source-ab677bdd0cef`). Both
executables in `build/current` match their release artifacts by SHA-256. Final
cleanup reclaimed 5.32 GiB of generated Cargo artifacts; all 80 recorded user-data
files remain unchanged. The preserved earlier release ZIP was not rebuilt.

## Empty-detail follow-up (2026-11-06)

A follow-up audit checked benchmark descriptions, hardware details, plan/resources,
capabilities, comparisons, result tabs, sampling details and native history.

- Replaced the fixed-height About this test disclosure with its short definition
  inline; an empty definition renders no control or placeholder panel.
- Dataset, demand and sampling controls now depend on their actual rendered data.
  Switching to a result without the previously selected tab falls back to Overview.
- Expanded result cards size to content, without the old 540 px minimum. Results
  without supplemental measurements say so briefly.
- Empty observation history no longer reserves a chart, empty selectors, table
  headers or pagination. Empty run and metadata filters show a short empty state.
- Capability/comparison disclosures and history inspectors guard their current
  content/selection. Optional empty status/context lines do not reserve rows.

This change only affects presentation; benchmark execution and saved data formats
are unchanged. Qualification for this follow-up is recorded separately under
`build/qualification/empty-details/`.

Follow-up validation: optimized workspace build, formatting and whitespace checks
passed. Desktop checks verified two inline test definitions, result controls and
the metadata no-match state in the rebuilt current app. Unit tests were not rerun
for this presentation-only follow-up. All 80 original saved-data hashes match.
Generated release caches were cleaned after copying the verified app pair.


## Stable-layout follow-up (2026-11-06)

The layout audit found that short Results/Settings pages could use a preferred
height instead of the full viewport, while benchmark column widths depended on
child content. Default button stretching could also redistribute vertical space.

- The sidebar keeps a fixed width, and action buttons keep a consistent height.
- Results and Settings fill the available viewport and start at its top edge.
- Benchmark catalogue/configuration columns use a fixed 48/52 split of the
  available content width, independent of selected labels or expanded details.
- Selectors and text fields allow their containing columns to set the width.
  Fact rows retain natural text measurement and wrapping.
- Expanded content remains in the existing scroll area. Disclosure content can
  extend the document without reallocating the surrounding page or button sizes.

This is a presentation change. Local build and desktop qualification evidence is
recorded under `build/qualification/stable-layout/`.

Stable-layout validation: optimized workspace build, formatting and whitespace
checks passed. Desktop comparisons at a 1280x820 client verified the Results
heading/toolbar/filter/table/primary-value positions across detail expansion,
benchmark column and Run-button positions across menus/condition review, and
Settings controls across dataset-menu opening. Page documents have explicit
top-left origins. Unit tests were not rerun for this layout-only change.
All 80 saved-data hashes match; release caches were reclaimed after verification.


## Two-plan follow-up (2026-11-06)

This supersedes the earlier four-scope selector. Curated is the default and resolves
standard coverage for the selected family only. Custom resolves the selected
configuration when its list is empty, and the exact saved condition list otherwise.
Adding a condition or preparing a quick plan selects Custom. Clearing the list
returns Custom to the selected configuration. The preview and Run share one resolver.
Curated requires a selected family; an absent selection cannot silently run a category.

Only Custom displays variant/settings/list controls. The old compute setup selector
is a Customize settings checkbox. Curated coverage remains extensive and preserves
the existing paired access conditions. The older whole-category planner remains
available internally for existing task generation; it is not another run mode.

Condition rows now flow within the configuration pane's existing scroll area,
without a nested 300 px viewport. One-page reviews have no pagination controls.
Other short queue/history/dataset lists receive explicit top-left content origins.
Qualification evidence is recorded under `build/qualification/two-plans/`.

Two-plan validation: 147 UI tests passed, five opt-in tests ignored. Strict UI
Clippy, formatting, whitespace and optimized workspace build passed. Desktop
review verified the two-entry selector, Curated eight-condition memory coverage
and Custom one-condition review with no gap or one-page navigation. The current
UI/worker pair matches release artifacts by SHA-256. All 80 saved-data hashes
remain unchanged; generated debug/release caches were reclaimed after copying.


## Lean default workloads (2026-11-06)

This supersedes the earlier extensive sampling and paired curated-access choices.
Default measurements now use six observations with a 250 ms active-duration floor
per observation (at least 1.5 s per condition/size). Shorter runs use three at
100 ms (at least 0.3 s). References/calibration use three at 100 ms. Explicit larger
API budgets and sustained duration requests remain honored. Counterbalanced
visits and the complete-observation/outlier retention rules are unchanged. Native
results record `measurement_policy_revision=bounded-v2`; old records remain intact.

Curated plans use their standard traversal/access and one workload realization.
They no longer add shuffled conditions or three-seed ensembles, including latency,
codec, memory and matrix plans. Core-count, precision, register/buffered and matrix
shape coverage remains. Default selected configurations also use standard access
and a single realization. Custom settings retain randomized access, fixed/fresh
seeds and one/three realizations. Intrinsic benchmark algorithms are unchanged.

Regression checks cover budget floors, per-condition coverage, retained outliers,
Curated isolation from custom randomness and explicit Custom random/seed coverage.
Evidence is retained under `build/qualification/lean-defaults/`.

Lean-default validation: 321 workspace tests passed, 25 opt-in tests ignored.
Strict workspace Clippy, formatting, whitespace and optimized workspace build
passed. UI and worker artifacts in build/current match the final release hashes.
No hardware benchmarks or additional desktop review were performed for this
sampling-policy change. All 80 recorded current-app data files and nine profile
data files are preserved; generated Cargo caches were cleaned after the build.


## Result timestamps and latest-run markers (2026-11-06)

Each Results row and the selected result show the recorded date/time with seconds
and an explicit UTC suffix. New captures retain millisecond timestamps in metadata;
older second-resolution timestamps are supported. Missing times say Time not recorded.
Opening native history preserves existing result timestamps instead of rebasing them.

A New badge identifies completed results from the current/latest run action. All
conditions from one Curated/Custom run share a persistent batch identity. Starting
a new run clears the previous flags immediately; old scores do not become New while
waiting for a replacement. Cancelled/failed conditions publish no new score. The
latest completed batch is recoverable from saved results after restart. Native
history views can match membership by run ID; unrelated imports do not become New
just because their timestamp is newer. Older results without batch IDs fall back
to their recorded run identity when available.

Rows reserve consistent height and badge width, with timestamps and adjacent New
badges below their configuration. Values and Open actions are vertically centered.
The inspector shows the same timestamp/marker beside its summary.
No benchmark sampling, metric values or history schema is changed in this follow-up.
Qualification evidence is under `build/qualification/result-recency/`.

Result-recency validation: 150 UI tests passed; five opt-in tests ignored. Strict
UI Clippy, formatting, whitespace and optimized workspace build passed. Desktop
review verified the four real cache/RAM results from the preview, their shared
New markers after restart, distinct UTC timestamps, adjacent date badges and
centered working Open actions. A final 8 px detail-header spacing adjustment was
compile-checked. The preview results remain in the isolated qualification data.
All 80 original data-file hashes match; 14 profile data files are preserved.
The current UI/worker pair matches release hashes; Cargo caches were reclaimed.


## Test checklist and active-only queue (2026-11-06)

The Tests to run list is always present, without a review disclosure or pagination.
Each test has a Run checkbox and row-local Details expansion. Search filters the
view only; it never removes hidden checked tests from execution. Unchecked tests
stay visible. The summary reports selected/available counts and estimates only the
selected work. Selecting none disables Run with a short explanation.

Checkboxes and expanded details use resolved workload identity (separate for
Curated and Custom), preserving choices through sorting, filtering, custom-list
removal and measurement-policy changes. Only checked runs and their corresponding
suite entries enter execution, preserving order and native identities. Choices
last for the session and cannot be changed during an active test.

The queue appears only while a worker request is in flight. Completed/previous
queue state remains available internally/history without occupying idle screen
space. The Resource details disclosure and its repeated explanatory text are
removed. Test estimates retain their estimated/minimum/partial labels. Minimum
measurement times show fractional seconds instead of rounding a 0.3 s floor to 1 s.
Native protocol/condition identifiers keep their existing names for compatibility;
visible planning and queue labels use tests.

Evidence and check logs are under `build/qualification/test-checklist/`.

Checklist validation: 152 UI tests passed; five opt-in tests ignored. Strict UI
Clippy, formatting, whitespace and optimized workspace build passed. Desktop review
verified row-local details, visible unchecked tests, selected-only estimates,
zero-selection feedback, sampling changes and the active-only queue. An isolated
short L1 run executed exactly one of four available tests; the saved suite contains
one completed L1 entry. The queue stayed hidden afterward. Empty queue filters keep
their controls and selected filter binding; that final adjustment was compile-checked.
The current UI/worker pair matches release hashes. All 80 original data files and
18 profile data files are preserved; generated Cargo caches were reclaimed.

## Equal category buttons (2026-11-07)

All four benchmark category buttons use the same 32 px height and equal horizontal
stretch with a zero preferred width. Both rows therefore divide their available
width equally, regardless of label length or selected category. Other button
groups keep their existing sizing. Palette suggestions are preview-only; the app
and logo colors remain unchanged pending selection.

The optimized workspace build and whitespace check passed. Desktop review verified
equal sizing with CPU memory and GPU performance selected. Evidence is under
`build/qualification/category-buttons/`; the single current executable pair was
updated and generated Cargo caches reclaimed. Original saved data is preserved.
