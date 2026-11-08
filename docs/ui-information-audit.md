# UI information audit and redesign plan

Date: 2026-11-03. Status: implemented in the working tree; local qualification recorded in
[implementation notes](ui-redesign-implementation.md).

Scope update: the user explicitly requested an app-wide reduction in descriptions
and explanations. Numeric evidence leads; concise copy supports it. This applies
to every phase below, including expanded views and contextual help.

## Finding

The attached screenshot is a presentation problem, not a reason to collect fewer
measurements. Configuration, plan scope, execution estimates and implementation
notes are competing for attention. Long pipe-separated condition descriptions are
being used as an interface. Expanding a section frequently reveals another report
instead of a useful inspector.

This audit covers the current Slint UI and the Rust code that assembles its text:
benchmark configuration, plans, queue, results, sampling, history, hardware details,
settings, help and sharing. Findings are based on source inspection, the supplied
screenshot, and the preceding local qualification. This is not a new usability
study or a claim that every display scale has been tested.

The proposed default audience is a hardware enthusiast who understands CPU/GPU,
dataset size and throughput, but should not need to interpret runner IDs, raw
nanoseconds or kernel terminology to operate the app. Technical inspection and full
native exports remain available.

## Presentation rules

- The main view answers: what is selected, what will run, what was measured, and
  whether anything is incomplete or incompatible.
- Use labeled values and small tables for facts; use graphs for relationships.
  Do not turn every metadata field into a card or graph.
- Show one selected plan scope at a time. Keep other scopes reachable without
  displaying their complete previews simultaneously.
- Separate measured results, descriptive variability, inferred interpretations,
  requested settings and resolved execution facts.
- Replace repeated explanations with contextual help. Important failures, partial
  coverage, unavailable modes and comparison restrictions stay visible.
- Remove low-value content from the default display, not from saved measurements.
  Use an explicitly opened Technical details inspector for IDs and exact provenance.
- Use readable units, meaningful precision and friendly labels. Exact values remain
  copyable/exportable. Unknown or unrecorded data never becomes zero or a pass badge.
- Keep Extensive as the default; randomness remains opt-in for custom runs and
  separately measured with equal curated coverage. Never suggest rerunning because
  results vary. These are presentation changes, not changes to benchmark policy.

## Copy and explanation audit: numbers first

Shortening copy is a first-class deliverable across the whole application, not a
final polish pass or permission to move the same paragraphs into an accordion.
Expanded views must also be structured and concise. Do not require users to read
a description to discover a value, unit, selection, limitation or required action.

### Default copy policy

- Prefer a clear label plus a value: `Cores 16`, `Observations 12`, `Pending 63`.
  Do not add a sentence restating what that row already says.
- Descriptions are optional. If a title, unit and control explain the function,
  omit the description. Otherwise use one sentence, targeting 12 words or fewer.
- Control hints target 8 words or fewer. Explain only the ambiguity or consequence
  needed to make that choice; place the hint beside the relevant control.
- Routine success/status messages target 8 words or fewer. Errors state the
  problem and, when known, one concrete corrective action. Detail is available
  separately; essential errors are never truncated to meet a word budget.
- Default panels have at most one short explanatory sentence beyond labels and
  values. Repeated rows do not each get their own explanatory paragraph.
- Help opens to one short definition and, when useful, a labeled formula or
  example. Longer methodology belongs in documentation, linked once by topic.
- Treat these as design targets, not automatic string truncation. Keep scientific
  meaning, accessible labels and necessary qualifiers even when they require more.
- Remove generic application-use claims, motivational guidance, repeated caveats,
  implementation narration and diagnoses that merely restate a graph. Do not
  replace them with confident performance judgments or unmeasured causes.
- Keep qualifiers attached to their numbers: `Minimum measurement`, `Estimated
  buffers`, `Completed-only p95`, `Effective traffic`, `Coverage partial`.
  Concision must not make a minimum look like an ETA or throughput look like
  physical memory-bus traffic. Plain numbers without scope are not enough.

### Additional surfaces in scope

| Surface / source | Proposed treatment |
| --- | --- |
| Page subtitles and family cards in `app-window.slint` | Remove navigation instructions that repeat visible controls. Keep a short purpose only when the family name is ambiguous. Replace long supported-mode inventories with mode count and selector. |
| `description-tldr`, full description, `everyday-use`, `metric-guide` | Remove routine result descriptions and Where it can help sections. Keep one concise test definition in About; metric labels/units and a small legend replace the prose guide. Audit workload help generators in `main.rs` as well as static Slint text. |
| `diagnosis`, `tuning-goal`, `tuning-guidance`, tuning-quality notes | Remove generic advice and narrative summaries. Present measured deltas/drift/coverage as labeled values. Keep a short inference only if it adds distinct, supported information, with an Inferred label and evidence available. |
| Codec, sieve, latency, matrix and transfer explanations | Keep a short workload identity where it changes interpretation, such as Independent tiles or Mean copy cost. Move timing inclusions, validation recipes and algorithm implementation into structured Methodology details. |
| Access/randomness and sustained-run explanations | Show selected pattern, realizations, duration and coverage directly. Explain equal curated coverage once in help. Keep Cache misses not measured beside any interpretation that could otherwise imply counters. |
| CPU/GPU pacing and allocation defaults | Show the requested setting and effective constraints. Use a short hint for pacing or budget meaning; put pause formulas and allocation rules in help. Retain that activity pacing is a target rather than a power/utilization limit. |
| Comparison note and result footer | Show delta with units and metric direction. Replace repeated significance disclaimers with an Observed range label and concise contextual help. Incompatibility uses a short reason plus a structured list of changed fields. |
| Validation, coverage and reference drift | Show separately scoped statuses/values, with the validation recipe available on demand. Do not reduce sampled validation to an unconditional Valid badge. Failures and missing/partial coverage stay prominent. |
| Hardware descriptions, capability explanations | Prefer feature/support rows and values. Remove explanations of what a processor or RAM is; use help only for less familiar properties. |
| Empty, loading, success, import/export and recovery messages | One state and one relevant action. Counts and scope replace procedural paragraphs. Keep storage failures, partial imports and destructive-action consequences explicit. |
| Chart captions, tooltips, table headers and legends | Define each encoding once. Tooltip shows the selected measured values and scope; do not repeat the full chart explanation at every point. |

### Copy examples for implementation

These are presentation examples, not invented benchmark results. Values must come
from the actual selected record or resolved configuration.

| Existing pattern | Proposed display |
| --- | --- |
| Measures cache bandwidth that helps the processor reuse nearby data quickly. | `L1 cache bandwidth` and `GB/s`; omit the sentence. |
| Each realization gets the full selected measurement duration... | `Realizations 3` / `Minimum measurement 9 s`; definition in help. |
| Selected measurement: ... (+4.3%, higher throughput). Observed ranges describe variation... | `Throughput +4.3%` beside baseline/current values and their observed ranges. |
| Validation: boundary_and_interior_outputs_of_each_region_match_precision_specific_scalar_fma... | `Validation: sampled output checks` with Scope/details action. |
| One dependent chain. Cache targets cannot exceed their topology-sized allocation... | `Dependent-read latency` / actual dataset size; allocation constraint in the control/error. |
| Points show median throughput; shaded bands show the sample range. | Legend: `Median` / `Observed range`. |
| Highlights are selected by default. Choose any completed result below... | `Selected 1` / `Embedded summaries 1`; separate `Include all` control. |
| Run ... Completed ... coverage ... raw observations ... app ... build ... source ... | `Completed` / `Coverage complete` / `Observations 12`; exact provenance in its own view. |

### Copy acceptance checks

Every surface in the audit needs an explicit keep/shorten/structure/move/remove
decision. Review the ordinary screen and each expanded state; a collapsed heading
alone does not satisfy this requirement. A copy inventory should include static
Slint text, Rust formatters, capability/error strings and legacy/import fallbacks.

The reader must be able to find the result, units, dataset, observed spread,
coverage and relevant limitation by scanning labeled fields. No result needs a
paragraph to explain another paragraph. Compare before/after screenshots for
repetition and reading burden, not only character count. Verify that removed copy
has not removed a measurement qualifier or changed the meaning of a control.

## Audit and proposed replacements

Priority P0 addresses the screenshot and run control; P1 addresses result reading;
P2 addresses supporting surfaces. Source locations are navigation anchors rather
than frozen line numbers.

| Priority / surface | Current issue | Replacement | Hide, remove or retain |
| --- | --- | --- | --- |
| P0 Configured mode preview | `summary()` plus all `details()` is one always-visible string, repeating selected controls and showing device IDs. | Three labeled values: condition count, minimum measurement time, peak test buffers. A short configuration summary and Review conditions action. | Remove pipe-separated prose; move alignment, per-worker estimates and accounting exclusions into resource details. |
| P0 Family/category previews | Both summaries appear alongside configured and custom plans; one disclosure opens both full lists. | Plan scope selector: Configured / Family / Category / Custom. Show only the selected scope's summary, preview and matching Run action. | Retain every scope and its distinct conditions; eliminate simultaneous duplicate reports. |
| P0 Custom plan | Every condition is expanded into wrapped prose; 56 entries can dominate the page. | Dedicated plan editor with grouped rows, count, search/filter and per-row inspect/remove. Preview first groups with View all; review all conditions before starting remains available. | Show distinguishing options in columns; common target/settings once per group. No silent deduplication or dropped execution conditions. |
| P0 Core-count experiment | Generic supported-workload paragraph appears for every CPU test, including unrelated modes. | Show experiment controls only for supporting families. Display Fixed input and selected core counts; explain a missing prerequisite beside the relevant control. | Remove generic list from unrelated tests. Retain explicit unsupported reasons when the experiment is requested. |
| P0 Queue | Repeats internal benchmark/device IDs, minimum time and buffer estimate for every row; action says Skip / cancel regardless of status. | Status counts and a compact table: Test, distinguishing condition, Status, Action. Running row gets Cancel condition; pending row gets Skip. | Technical estimates in row inspector. Keep global Stop all and truthful cancelled/failed/skipped statuses. |
| P0 Capability details | All unavailable or implementation-only entries for the category can be dumped below a selected test. | Selected-mode availability beside its selector, with a concise reason. Other unavailable modes remain discoverable through Show unavailable modes. | Move sparse/FP8 runner status and backend capability inventories to hardware/technical inspection. Do not imply capability-only entries are runnable. |
| P1 Result overview/context | Median/range/count are already useful, but validation scope and drift form a long sentence; descriptions/comparison explanations repeat. | One primary result, observed range, observation count/coverage and configuration chips. Short validation status with scope on demand. Comparison state has its own concise row. | Remove repeated descriptions and boilerplate. Failed validation, partial/unknown coverage and incompatibility remain visible without expansion. |
| P1 Expanded result panel | One expansion stacks cards, bars, full dataset lists, context, charts, timing, about text and diagnosis into a long page. | Separate views: Overview / Dataset curve / Timing or workload behavior / Technical details. Only applicable views appear. | Keep existing useful charts; stop showing the same values in multiple full-length sections. About this test becomes contextual help. |
| P1 Memory dataset lists | Each size/operation is rendered as a label plus a formatted value paragraph. | Size-versus-throughput curve with read/write/copy toggles, plus a compact table of size, median, range and count. Single-size tests use a table or interval marks. | Do not combine cache/streaming modes, shader/native copy paths or transfer directions into one unlabeled curve. |
| P1 Sampling details | Counts, ranges, ensemble explanation and batch information accumulate as many fact rows. | Coverage summary, then a chronological observation plot and table filtered by dataset, metric, phase and realization. | Reference/calibration/diagnostic observations are separate from main observations. Show per-realization coverage; technical batch geometry is collapsed. |
| P1 GPU offload detail | Completion percentiles, pending categories, ages and time windows are mixed in the sampling list. | Dedicated demand view: offered/completed/pending counts, pending submitted/unsubmitted split, completed-only latency distribution and pending-age summary. | Keep unfinished work visible next to completed-only latency. Window definitions and exact accounting move to help/technical details. |
| P1 History list/inspector | Run ID, enum status, source, validation scope, app/build revision and sample count are concatenated into strings; raw samples are 30 blocks of prose. | List columns: Name/test, Date, Device, Status, Source; selected run opens a structured inspector. Raw observations use a paged table and optional plot. | Run/build IDs, exact timestamps and source reference stay copyable in Provenance. Names, notes, pins and import identity remain intact. |
| P2 Hardware details | Status, cache topology and full device details are three long text blocks. | CPU cache table grouped by level/instance/sharing; GPU precision/feature support matrix; memory capacity/speed rows. | Backend IDs, driver/API raw values and low-level feature flags go to searchable Technical details. Preserve installed versus OS-visible distinction. |
| P2 Settings and test help | Long always-visible explanations, workload lists and numerical pacing explanations interrupt controls. | Group defaults by CPU, GPU, Memory/access and Measurement. A one-line explanation per control with an explicit help expander. | Move formulas and exclusions into help. Keep units, effective selection, constraints and validation errors at the control. |
| P2 Sharing | Long instructions precede the checklist; selection labels repeat full configurations. | Compact selected count, preview and clear export scope; group choices by device/family and show distinguishing settings. | One metadata-preservation note; raw-data export remains clearly available. Explicit Include all remains opt-in. |

## First replacement: the screenshot's plan area

Replace the four competing preview sections with the following structure. Values
below come from the screenshot and illustrate layout, not new runtime predictions.

```text
Run scope   [Configured] [Family] [Category] [Custom (56)]

L1 cache bandwidth                         Extensive
1 condition          >= 3 s measured work       ~459 KiB buffers
16 CPU cores         Sequential                Runtime-selected size

[Review conditions]                          [Run configured]

Resource details >   Measurement policy >
```

Selecting Family replaces this summary with its 8 conditions and at least 72 seconds
of measurement; Category shows 13 and at least 117 seconds. Custom opens the 56-row
plan editor. These figures must be derived from the resolved plan, never duplicated
constants. The minimum is **measured work**, not an estimated total completion time.
One adjacent help sentence explains that preparation, sweeps and validation add time.

The resource inspector distinguishes dataset payload, estimated peak test buffers,
RAM versus VRAM, aggregate versus per-worker geometry, and unestimated driver/setup
overhead. When some allocations resolve only during execution, the summary must say
so; a maximum of known estimates must not look like a complete total or a hard cap.

The conditions table uses Test, Mode/access, Dataset, Workers/target and Status.
It adds operation/path, direction or matrix shape only when relevant. Shared values
belong in a group header; differences belong in rows. Long device names wrap in the
inspector rather than repeating in every condition. Grouping must expose the exact
executable count and keep separate sequential/randomized conditions inspectable.

Keep Add to custom plan close to configuration. Show a compact custom-plan count
while another scope is selected. Task shortcuts should prepare the same reviewable
plan workspace, rather than adding another report beneath the configuration.

## Graph selection and honesty

| Question | Appropriate presentation | Boundaries |
| --- | --- | --- |
| How does performance change with size? | Existing dataset curve, observed-range band, selected point and table | Label logarithmic axes; no invented intermediate measurements. Keep separate modes and accounting definitions. |
| How variable was this run? | Median/range interval for a small sample set; observation-over-time plot when raw observations exist | No stability score or rerun advice. Missing raw data gets Summary only. Separate within-realization variation from differences between realizations. |
| Does work change during a sustained run? | Chronological throughput plot with actual elapsed time and phase labels | Display only measured points; do not invent thermal/power explanations. |
| What changes across core counts? | Throughput versus actual worker count, with optional speedup relative to the measured one-core condition | Only compatible fixed-input geometry; missing one-core reference means no speedup value. |
| Where does GPU time go? | Aligned timing series or labeled duration table | Overlapping inclusive intervals must not be stacked or added into a fake total. Pacing and device/host scopes stay explicit. |
| Did the offload workload keep up? | Counts and offered/completed rates; pending split plus completed-only latency distribution | Derive completion fractions only from a common observation window. Do not divide independently aggregated medians or average percentiles into a new distribution. |
| What will this plan run? | Grouped condition table and counts | No graph needed; no invented full-run ETA or memory gauge implying a measured safety margin. |

All graphs need a textual/table alternative, clear units, labels beyond color,
keyboard-selectable points or an equivalent selector, and a useful empty/partial
state. Never pool different metrics, units, datasets or timing scopes on one axis.
Large raw histories need bounded rendering; display downsampling must preserve
outliers and chronology, be labeled, and leave exact observations in the table/export.

## Implementation phases

These are a new presentation follow-up to the delivered benchmark roadmap. They do
not reopen or renumber its measurement/kernel phases.

### UI-1: Plan and configuration clarity (P0)

- Inventory and edit configuration/page copy using the app-wide copy policy;
  remove repeated descriptions and irrelevant explanations in this first slice.
- Introduce typed plan summary, resource summary and condition-row presentation
  models derived from PreparedPlan. Do not parse the existing formatted strings.
- Add the scope selector and one matching Run action; build the grouped condition
  review/custom editor and contextual experiment/capability presentation.
- Replace repeated resource caveats with one disclosure; keep requested and resolved
  sizes distinct. Keep custom editing and all existing run scopes reachable.
- Acceptance: the 1/8/13/56-condition examples show compact summaries; a 56-condition
  plan creates no 56-paragraph dump. All resolved conditions can be inspected and
  the executed plan matches the selected scope exactly.

### UI-2: Queue and history structure (P0/P1)

- Replace queue/history strings with typed rows and a selected-item inspector.
- Use distinct Skip / Cancel condition / Stop all actions, status filters and
  persisted completion counts. Add a compact configuration-difference view for
  comparisons instead of semicolon-joined incompatibility fields.
- Acceptance: cancelled, failed, partial, imported and legacy runs are distinguishable;
  names/pins/notes survive restart; every comparison blocker remains inspectable.

### UI-3: Result hierarchy and dataset presentation (P1)

- Audit descriptions, everyday-use claims, metric guides, tuning guidance,
  diagnoses, comparison boilerplate and chart captions. Replace narrative with
  labeled evidence, retaining only concise explanations that add meaning.
- Split the expanded panel into applicable views and use one selected-dataset state.
- Keep the median/range/count overview; add concise validation and coverage states.
- Reuse existing compute/timing charts and add memory dataset curves/table views.
  Move inferred interpretation and detailed accounting into clearly labeled sections.
- Acceptance: users can identify the selected size, result, spread and coverage
  without opening Technical details. A result is not repeated across multiple cards
  and prose blocks. Direction, path and cache/streaming identity remain visible.

### UI-4: Observation, sustained and offload inspectors (P1)

- Build filtered raw tables and chronological plots from native records; retain
  realization, phase, timing scope and observation status as structured dimensions.
- Present sustained trends, core-count comparisons and demand accounting only when
  sufficient compatible measured data exists. Keep raw recording unchanged.
- Acceptance: zero completions, pending work, partial cancellation, missing raw data
  and mixed realization coverage all display truthfully. Rendering stays responsive
  on long histories without discarding export data.

### UI-5: Hardware, settings, help and sharing (P2)

- Apply the same copy limits to help and expanded sections; do not relocate word
  dumps. Cover hardware prose, control hints, empty/error states and file messages.
- Convert hardware details into topology/support tables and searchable provenance.
- Shorten control help and snapshot instructions; group export choices and surface
  selected versus include-all scope clearly. Remove irrelevant sections by context.
- Acceptance: unsupported capability reasons and essential limitations remain
  discoverable; JSON/PNG scope and original data preservation remain unchanged.

### UI-6: Visual and behavior qualification

- Check every surface against the copy inventory and word-budget targets, including
  expanded views. Review exceptions for scientific meaning and necessary actions.
- Review screenshots at the standard 1280x850 window, maximized layout, a smaller
  supported layout and increased text/display scaling; establish the actual minimum
  usable size from those checks rather than assuming a new universal minimum.
- Verify keyboard traversal, table alternatives, long names, scrolling, empty/error
  states, legacy results, 56+ conditions and large imported histories.
- Recheck run-scope execution, skip/cancel persistence, compatibility explanations,
  equal curated coverage and selected-only exports after presentation changes.
- Use focused regressions for data-to-view mapping and meaningful workflow behavior;
  visual review is required as well as compilation. Update catalogue screenshots only
  after the design settles; keep a single current build rather than per-step copies.

## Source map

- `crates/gluj-bench-ui/ui/app-window.slint`: hardware details; configuration/plan
  area; queue; settings; history/share surfaces; stacked result-detail sections.
- `crates/gluj-bench-ui/src/families.rs`: PreparedPlan summary/details strings,
  category-wide capability details, core-experiment prerequisites.
- `crates/gluj-bench-ui/src/main.rs`: UI refresh concatenation, configured/custom
  previews, hardware detail formatting, workload help and result presentation.
- `crates/gluj-bench-ui/src/phase_eight.rs`: history/queue string assembly, raw
  observation paging, validation context and compatibility explanations.
- `crates/gluj-bench-ui/src/dataset_details.rs`: selected-size facts, sampling
  coverage, GPU offload request/latency accounting.
- `crates/gluj-bench-ui/src/result_visuals.rs`, `gpu_timings/`, `tuning_visuals.rs`:
  existing chart/fact logic to retain and adapt rather than duplicate.

## Delivery

The user authorized implementation. UI-1 through UI-5 are implemented; UI-6 has
local automated and desktop evidence, with remaining environment coverage recorded
in the implementation notes. The descriptions above remain the design rationale.
No sampling policy, kernel definition, native observation or stored score changed.
