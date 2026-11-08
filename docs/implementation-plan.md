# Gluj-Bench implementation plan

Latest policy update (2026-11-06): defaults now use six observations (three for Shorter). Curated uses one standard-access realization; randomization and seed ensembles are Custom options. See [implementation notes](ui-redesign-implementation.md#lean-default-workloads-2026-11-06).




Created: 2026-10-31

Last status review: 2026-11-03



Status: Phases 1–8 are implemented. Phase 9 completes the local release candidate

**0.5.0-rc.1**, with measured defaults, compatibility fixtures, hardware checks and

release packaging. [Qualification evidence and limitations](phase-9-qualification.md)

distinguish locally verified behavior from the remaining cross-vendor hardware gate.

No external release has been published.



Presentation follow-up: the [UI information audit and redesign plan](ui-information-audit.md)

proposes UI-1 through UI-6 for dense previews, inspectors and technical details.

This follow-up is planned, not implemented; it preserves the measurement policies below.
Its scope includes app-wide concise copy: numbers and labeled facts lead; repeated
descriptions, explanations and guidance are shortened, structured or removed,
including inside expanded/help views.



## Implementation status



This status reflects the current working tree, including uncommitted changes.

Implementation and local qualification are complete. Intel/hybrid CPUs, integrated

GPUs and other GPU vendors/drivers still require physical qualification before

claiming broad hardware coverage. This is an explicit release-candidate limitation.



| Phase | Status | Implemented outcome / remaining work |

| --- | --- | --- |

| 0 â€” Baseline | Reconciled | Coverage ledger and original version-1 results/settings/PNG fixtures reconciled during Phase 9. |

| 1 â€” Contracts and compatibility | Implemented | Fifteen-family contracts, shared resolution, versioned run/sample records, native history, legacy migration and additive worker commands. |

| 2 â€” Measurement | Implemented; locally calibrated | Extensive/Shorter policies, bounded coverage, raw observations, correctness checks, timing scopes and neutral variability reporting. |

| 3 â€” Families and suite plans | Implemented | Fifteen cards, supported mode controls, canonical one-core INT64, curated/custom plans, previews, settings migration and separate configuration results. |

| 4 â€” CPU access patterns | Implemented | Seeded access patterns and ensembles, separate equal-coverage curated conditions, ASCII inputs, latency controls, RAM sizes/sweeps, schedule allocation accounting and consistent copy summaries. |

| 5 â€” Older CPU workloads | Implemented | Buffered scalar/SIMD FP64, data-backed POPCNT, independent-block codec controls and ratios, configurable sieve, explicit matrix geometry/reuse, fixed-input core-count experiments and sustained CPU summaries. |

| 6 â€” GPU vector upgrades | Implemented | Native buffered FP16/FP32/FP64, capability-gated storage, FP32 host placement, seeded block/region/lane access, matched calibration, sustained coverage, pending-request accounting and completed-only latency distributions; reproducible shader builds. |

| 7 â€” GPU memory and matrix | Implemented; local user confirmations passed 2026-11-02 | Custom sizes/sweeps, explicit access and copy paths, fresh discovery provenance, transition intervals, directional transfers/ring/submission controls, supported native matrix tiles/reuse/traversal, sustained coverage and shader reproduction. |

| 8 â€” Results and workflows | Implemented; local interactive checks recorded | Largest/selected dataset summaries, observed spread, latest/typical emphasis, named/pinned baselines, native history/observations, archive/delete/export, selected-only PNGs with explicit include-all, raw native JSON round trips, task preparation and per-condition queue controls. |

| 9 â€” Qualification and release | Local RC complete; external hardware gate open | 314 workspace tests, 56 effective local hardware cases, measured defaults, compatibility, UI verification and release package. Physical cross-vendor qualification remains open. |



Phase 6 verification (2026-11-01): **291 tests passed**, 24 intentionally ignored;

strict Clippy, formatting, workspace release build and shader validation/reproduction

passed. A live GPU fixture validated 192 precision/placement/access/loop/launch

combinations. Twenty-seven release-worker cases passed, covering register/buffered

precisions, all access modes, equal-coverage ensembles, fresh seeds, sweeps, all

FP32 host shares, overload, sustained execution and partial cancellation. Native

records matched terminal responses. The cases included an Extensive FP16 run meeting

its full 12-observation floor and dynamic demand. Desktop checks verified buffered

FP16 execution, restored custom size/access controls, the paired Curated mode preview,

and completed-only percentiles alongside pending request counts/ages. Cross-vendor qualification

and runtime calibration remain Phase 9 gates. Earlier CPU/Phase 4 checks remain valid.



Inputs:



- [Product review requirements](product-review-requirements.md).

- [CPU/GPU catalogue audit](benchmark-catalogue-audit.md).

- The product review in this conversation; subsequent user requirements override

  its earlier suggestions to request confirmation runs or reruns.



## Confirmed product decisions



1. Consolidate the catalogue into the proposed approximately fifteen family cards.

   Supported precision, execution path, target, access pattern, placement, direction,

   and worker count become modes/conditions within a family.

2. Extensive measurement is the default. Offer an explicitly selected shorter run.

   Both policies handle variability automatically and report actual coverage.

3. Implement all accepted review improvements in stages: measurement, workloads,

   catalogue, results, history, export, and task-oriented workflows.

4. Never suggest rerunning or retesting because measurements vary, references drift,

   calibration is uncertain, or a difference is small. A manual rerun is the user's

   choice. Show the numbers and their measured spread.

5. Introduce reproducible workload/access randomness that can exercise reduced

   locality and working-set turnover. Random numeric values alone do not satisfy

   this requirement.

6. Access randomness is an explicit Custom setting. Curated uses standard access
   with one realization; randomized access and additional seeds remain available
   in Custom as separate conditions.



## Non-negotiable behavior



- Preserve sequential baselines and meaningful distinct measurements. Consolidating

  cards must not silently reduce suite coverage or merge unlike scores.

- Use bounded automatic measurement. Preserve genuine variability and slow samples;

  do not run until a favorable stability threshold passes.

- Retain median results, raw chronological samples, actual configuration, sample

  coverage, workload revisions, timing scope and validation status.

- Distinguish within-realization variation from variation between random workloads.

- Label inferred explanations as inferences. A reference gap is neither measured

  stall time nor a prediction of recoverable performance or clock headroom.

- Do not claim measured cache misses, cache residency, energy savings or thermal

  causes without the corresponding evidence.

- Keep cancellation available during preparation, measurement and validation. A

  cancelled or failed condition must not become a successful score or a new best.

- Preserve saved settings and historical result provenance. Never fabricate raw

  samples or missing metadata for old results.

- Do not apply clock, voltage or external-process affinity changes.



## Target family catalogue



| Family | Main supported conditions to preserve or implement |

| --- | --- |

| CPU memory bandwidth | L1/L2/L3/RAM/custom targets; read/write/copy; explicit cache/store policies |

| CPU dependent-read latency | Cache/RAM targets; scattered/localized access; single pinned core |

| CPU FP32 arithmetic | Scalar/AVX2 paths; supported register/buffer modes; core counts |

| CPU FP64 arithmetic | Scalar/AVX2 paths; register and subsequently buffered modes |

| CPU FP32 matrix compute | Recorded matrix shapes, blocking and reuse |

| CPU INT64 arithmetic | Register/buffer; one/selected cores; core-count experiments |

| CPU ASCII scan | Input distributions, sizes and traversal |

| CPU DEFLATE | Compression/decompression; corpus, level and block size |

| CPU prime sieve | Search limit, footprint and validated complete-sieve work |

| CPU population count | Verified instruction workload and supported data-backed mode |

| CPU AES rounds | Verified round microbenchmark; do not imply complete encryption |

| GPU local-memory bandwidth | Cache-oriented/streaming paths; read/write/copy |

| GPU host-device transfers | H2D/D2H; size and supported queue/ring settings |

| GPU vector compute | FP16/FP32/FP64; supported placement, access and demand modes |

| GPU matrix compute | FP16/INT8; actual signedness, accumulator and tile definition |



Not every condition combination is supported. Capabilities are resolved per mode

and selected device. Register-only execution remains usable when a corresponding

buffered storage capability is missing. Unimplemented FP8/sparse formats belong in

capability/implementation information rather than the normal runnable catalogue.



## Delivery structure and dependencies



Each phase should be delivered as small reviewable changes. An unchecked item is

planned or incompletely verified work, not an assertion that the current

implementation is missing every part of that feature. Partial prerequisites are

identified in the status table; only fully evidenced checklist items are checked.



| Phase | Outcome | Depends on |

| --- | --- | --- |

| 0 | Baseline, coverage ledger and migration fixtures | Confirmed decisions above |

| 1 | Shared family/condition, result and suite contracts | 0 |

| 2 | Verified kernels and extensive bounded measurement | 1 |

| 3 | Family-card catalogue and complete suite execution | 1, 2 |

| 4 | Reproducible CPU access patterns and memory controls | 2, 3 |

| 5 | Upgraded legacy CPU algorithms and matrix experiments | 2, 3; 4 for shared access controls |

| 6 | GPU vector precision, placement and access upgrades | 2, 3, 4's access-pattern specification |

| 7 | GPU bandwidth, transfers and matrix upgrades | 2, 3; 6's reusable GPU infrastructure where applicable |

| 8 | Results interpretation, durable history, sharing and workflows | 1, 2, 3; integrates 4â€“7 as they land |

| 9 | Cross-hardware qualification, documentation and release | Required scope from 0â€“8 |



Minimal storage support lands in phase 1 and neutral result wording in phase 2;

they must not wait for the full phase-8 presentation and history interface.



## Phase 0 â€” Establish the implementation baseline



Primary areas: current working tree, catalogue providers, saved settings/results,

worker protocol, existing unit tests and Windows CI.



- [x] Inventory the existing uncommitted changes without resetting or overwriting

  them. Identify already-implemented work to preserve. The catalogue audit records

  the existing consolidations; implementation has preserved the dirty working tree.

- [x] Capture a coverage ledger mapping every existing descriptor to its actual

  runner, kernel, mode, capability, operation definition and timing scope.

  The [coverage ledger](benchmark-coverage-ledger.md) reconciles every descriptor,

  including compatibility aliases and capability-only entries.

- [x] Capture representative version-1 results/settings and original PNG imports as

  migration fixtures, including scalar, SIMD, single-thread, latency and offload.

  Original fixtures and their provenance are retained in the UI tests/fixtures

  directory; migration, native export and old-PNG imports pass regression checks.

- [x] Record existing build/test failures separately from new failures. Record

  available CPU/GPU validation hardware and gaps.

  Phase 1 preserves the passing baseline. Phase 9 records local hardware, the

  initial INT8 validation failure and its corrected workload revision, and all

  physical qualification gaps.

- [x] Inspect documentation for stale catalogue/protocol descriptions. README and

  protocol documentation were updated through Phases 1â€“3; release-wide

  documentation reconciliation is recorded in Phase 9.



Exit gate: every implemented descriptor has a planned destination; existing work

and data needed for migration are identified. No benchmark consolidation yet.



## Phase 1 â€” Define common contracts and compatibility



Primary areas: `gluj-bench-core` model/provider/registry/protocol, worker request

handling, UI configuration and result persistence.



- [x] Add declarative family and condition descriptors. Advertise supported options,

  valid combinations, defaults, capabilities, metric direction and timing scope.

- [x] Define one resolved condition shared by CLI/UI/worker. Resolve defaults once;

  reject unsupported options rather than silently ignoring or changing them.

- [x] Define condition identity: algorithm/kernel revision, precision/instruction

  path, matrix shape, core/thread placement, dataset/auxiliary footprint, access

  pattern/generator revision, seed policy, placement, pacing, demand policy,

  useful-operation definition, byte accounting and measurement scope.

- [x] Separate workload identity from measurement-policy provenance. Exact timing

  and quality comparison rules must state when policy differences are comparable,

  descriptive only, or incompatible; never silently merge unlike conditions.

- [x] Add versioned raw sample records with block/realization IDs, chronological

  order, useful work, active/elapsed timings, reference anchors, validation and run

  completion status. Preserve existing summary fields for old readers as appropriate.

- [x] Add a versioned run envelope with run ID, timestamp, application/build version,

  hardware/driver context and resolved configuration.

- [x] Add a suite-plan representation: explicit conditions, intended coverage,

  stable execution order/seed, estimates and per-condition status.

- [x] Define additive versus breaking protocol changes. Update protocol version

  where required; report incompatible UI/worker versions clearly.

- [x] Add minimum persistence for new records before new sampling ships. Choose a

  versioned native run format, atomic writes and a compact index. Raw samples must

  not exist only in transient UI memory.

- [x] Implement read/migration adapters for old results/settings. Preserve original

  IDs and revisions, avoid automatic cross-revision equivalence, and distinguish

  local measurements from imported comparison sources.



Exit gate: serialisation round trips, old fixtures load, explicit equivalent CLI/UI

conditions resolve identically, and incompatible conditions produce specific

reasons. Cancelled/failed/legacy-summary-only records have defined semantics.



## Phase 2 â€” Verify measurement and install extensive sampling



Primary areas: core measurement policy, CPU compute/kernels/latency/scaling,

GPU vector/matrix/bandwidth/profile timing, UI progress and quality text.



- [x] Inspect optimized release instructions for legacy scalar/INT64/POPCNT/AES

  kernels. Resolve the POPCNT rotated-input equivalence and check recurrence

  simplification before trusting nominal operation counts.

- [x] Define reference checks and useful-operation/byte counts per live kernel.

  Use independent correctness references and sampled interior/boundary outputs.

- [x] Implement Extensive and Shorter policies through shared configuration. Set

  minimum samples and minimum active duration independently; adding samples must

  not merely divide the same short window into smaller pieces.

- [x] Establish preparation/warmup, measurement blocks, reference anchors and

  bounded extensions. Give references and calibration an explicit policy too.

- [x] Preserve all valid samples and record extension/budget outcomes. Finish with

  measured spread even if variation remains. Work completed inside a timing batch

  is not automatically a set of independent samples.

- [x] Counterbalance/interleave size conditions where appropriate, restoring each

  mode's documented preparation state before timing.

- [x] Distinguish throughput in active execution, elapsed throughput, GPU service,

  offered-load observation windows and setup/validation time.

- [x] Avoid pooling seed-ensemble variation with fixed-realization variation without

  a defined summary rule. Include samples from every declared realization.

- [x] Initially display descriptive statistics. Introduce confidence intervals only

  with a justified method for the actual estimator, sample dependence and stopping

  policy. Do not use a heuristic spread threshold as a significance probability.

- [x] Route legacy fixed sample/duration limits and GPU reference/discovery clamps

  through the policy; remove contradictory hard-coded UI descriptions.

- [x] Remove result-driven repeat/retest/rerun suggestions across live runners,

  visuals, generated guidance, sharing and docs. Use numeric spread/drift facts.

- [x] Replace animated red "Lost" gaps with static neutral reference comparisons.

  Remove unsupported bound classifications, including the sieve's unmeasured one.

- [x] Expose phase-aware progress, a labeled minimum measurement-time estimate

  and actual samples. Preserve responsive cancellation and mark interrupted

  conditions accurately. Calibrate end-to-end time ranges in Phase 9.



Exit gate: deterministic scheduler tests prove limits, coverage and cancellation;

synthetic variable sequences retain their variation; correctness failures cannot

produce a successful score; all implemented runners use the policy or are clearly

unavailable during development. Extensive defaults require hardware calibration

in phase 9 before release.



Implementation notes, measurement budgets and validation scope: [Phase 2](phase-2-measurement.md).

Validation: 252 workspace tests passed (23 explicitly ignored), strict Clippy,

formatting and release build passed; local CPU/GPU, raw persistence, offered-load

and stdio cancellation/busy smoke checks passed. Cross-device calibration remains

the Phase 9 release gate.



## Phase 3 â€” Consolidate cards and preserve suite coverage



Primary areas: provider catalogues, UI `main.rs`, Slint picker/configuration,

queued-run planning, settings migration and result grouping.



- [x] Implement the fifteen-family catalogue above with per-device/mode controls.

  Replace ID/suffix allowlists with advertised capabilities incrementally.

- [x] Merge single-thread INT64 into one-core register mode using one verified

  harness. Preserve old history/provenance; deduplicate identical queued conditions.

- [x] Group scalar/SIMD arithmetic, cache/RAM targets, latency locality, codec

  directions, GPU precision and FP32 offload placement without merging scores.

- [x] Keep vector versus matrix, bandwidth versus latency, and direct host-memory

  compute versus staging transfers as distinct measurements/families.

- [x] Resolve scope/thread defaults consistently. Show actual cores/workers and

  aggregate versus per-worker sizes before execution.

- [x] Add explicit curated suite plans selecting intended scalar/SIMD, one/all-core,

  cache-level, codec-direction and GPU-precision coverage.

- [x] Support sequential and randomized access as distinct Custom conditions.
  Curated now uses standard access and one realization (2026-11-06 update).

- [x] Make run-family/run-category/run-custom-plan behavior clear. Show condition

  count, estimated duration and peak allocation before starting.

- [x] Preserve per-test overrides and provide migration/default handling for the

  new family modes. Hide unsupported combinations rather than exposing inert knobs.

- [x] Move unimplemented FP8/sparse rows into capability/implementation details.



Exit gate: the phase-0 coverage ledger is satisfied, migration fixtures retain

their meaning, one-core INT64 executes only once for identical conditions, and

family grouping does not make unsupported modes appear executable.



Implementation notes: [Phase 3 families and plans](phase-3-families.md).

Validation: 262 automated tests passed (23 intentionally ignored), strict Clippy,

formatting, whitespace checks and the workspace release build passed. Desktop family/mode/custom-plan checks

and a four-condition FP64 family execution passed in an isolated folder. Release

legacy INT64 alias execution preserved request provenance and saved six canonical

one-core register observations. The final visual recheck of the latest

result-grouping changes was stopped by the user; automated regression coverage

passed. Estimates remain explicitly scoped; calibrated

end-to-end duration ranges and cross-device qualification remain Phase 9 gates.



## Phase 4 â€” Add reproducible CPU access patterns



Primary areas: core access-policy specification, CPU buffered vector/INT64/ASCII,

memory bandwidth, latency chains, allocation accounting and result metadata.



- [x] Define a versioned deterministic generator, root seed, derived worker/size/

  block seeds and reproducible ensemble. Allow an explicit fresh-seed choice.

- [x] Expose access randomness in Custom; Curated uses standard access with one
  realization. Explicit randomized conditions retain separate results and identities.

- [x] Implement seeded block permutations with complete coverage, non-power-of-two

  support and bounded cancellable setup. Precompute schedules outside timing where

  practical and account for their memory and timed access costs.

- [x] First integrate shuffled blocks and rotating regions in CPU FP32 buffered

  compute. Preserve useful arithmetic and unambiguous output ownership.

- [x] Add separately identified scattered-read and hot/cold working-set conditions,

  defining coverage, block/region size, turnover, concurrency and reuse distance.

- [x] Extend appropriate controls to INT64, ASCII and bandwidth. Keep sequential

  peak-bandwidth baselines. ASCII content variation includes expected hit counts.

- [x] Extend the existing latency-chain engine with seed ensembles and supported

  locality/working-set controls, preserving dependent one-chain semantics.

- [x] Restore RAM bandwidth custom-size/sweep behavior explicitly, replacing tests

  that currently assert retired options are ignored. Preserve allocation headroom

  and RAM/cache labeling rules.

- [x] Normalize copy-rate presentation across CPU/GPU families: report distinctly

  named useful-payload and accounted read/write traffic where needed. Version

  accounting changes and do not rescale old results without provenance.



Exit gate: reproducible schedules, bounds/full-coverage checks, reference-correct

outputs and allocation accounting pass. No test assumes randomized access must be

slower or claims a measured cache-miss rate without counters.



Implementation notes: [Phase 4 access patterns and accounting](phase-4-access.md).

Validation: 278 workspace tests passed (23 intentionally ignored), strict Clippy,

formatting and whitespace checks passed. Local release-worker smoke passed for

five FP32 access patterns, two INT64 patterns, four ASCII distributions, a two-size

RAM sweep, L3 bandwidth, two latency ensembles, fresh seeds, GPU copy accounting,

matrix preservation and cancellation with partial raw records. Every successful

condition met per-realization coverage and saved its exact native response record.

Desktop controls and separate sequential/shuffled result rows were verified in an

isolated folder. The final workspace release build passed; final release UI

startup and balanced Curated test mode previews were checked. A focused CPU test

pass and strict workspace Clippy also passed after the last timing-metadata fix.



The realized definitions are explicit: rotating regions rotate the starting block

within a complete pass; scattered numeric access uses complete 64-element kernel

blocks; hot/cold visits count repeated work. These are locality workloads, not

hardware cache-miss measurements. Curated cache bandwidth uses 64-byte blocks to

avoid a redundant one-block shuffle. Cross-hardware calibration remains Phase 9.



## Phase 5 â€” Upgrade older CPU workloads and experiments



Implementation notes: [CPU workload definitions and validation](phase-5-cpu-workloads.md).



Primary areas: CPU compute, codec, sieve, FP64, matrix and topology allocation.



- [x] Upgrade FP64 with verified buffered kernels and size/sweep/access options,

  preserving scalar versus SIMD and FMA definitions.

- [x] Upgrade DEFLATE corpus/block size and compression-level controls. Provide

  seeded compressible/mixed/incompressible inputs, separate directional throughput,

  compression ratio, state/reset policy and round-trip correctness validation.

- [x] Keep compressed streams valid: random access order is supported only for

  independently decodable blocks when that mode is implemented.

- [x] Upgrade sieve limit/footprint controls, known-count/reference validation and

  complete-sieve timing. Record range with primes/s. Preserve algorithm dependencies.

- [x] Add supported data-backed population-count conditions with validated useful

  counts. Keep register instruction behavior separately identified.

- [x] Keep AES rounds accurately named and validated. Full AES encryption is a

  separate potential extension, not required to complete this plan.

- [x] Add representative CPU matrix shape/reuse controls with independent output

  checks; retain the current fixed shape as a named historical mode.

- [x] Add user-selected automatic core-count experiments with fixed aggregate work

  and recorded per-worker size/placement. Keep numerical measured outcomes without

  unsolicited rerun/tuning prompts or unmeasured power-saving claims.

- [x] Add sustained mode for supported CPU conditions with throughput over time and

  early/late summaries. Do not attribute declines to temperature without evidence.

- [x] Remove unreachable legacy runner branches only after fixtures/reference

  dependencies and equivalent live-path coverage are checked.



Exit gate: each added control changes the actual live workload, useful work is

validated, changed definitions have revisions, and matrix/core experiments either

hold work constant or clearly identify non-equivalent shapes.



## Phase 6 â€” Upgrade GPU vector compute and placement



Implementation notes: [GPU vector definitions and shader reproduction](phase-6-gpu-vector.md).



Primary areas: GPU compute/profile/vector harnesses, offered-load queue, traffic,

Vulkan feature discovery, GLSL and embedded SPIR-V.



- [x] Extend buffered vector execution to FP16 and FP64 with precision-specific

  storage, arithmetic, operation counts, feature checks and output references.

  Keep unsupported placement/demand options unavailable per precision.

- [x] Integrate existing FP32 offload as a placement mode, preserving supported

  fractions, real aligned splits and independent RAM/VRAM budgets.

- [x] Port the specified shuffled-block and region-turnover controls, then add

  explicit per-lane scattering as a distinct condition. Record coalescing changes

  and guarantee write ownership/no races.

- [x] Preserve maximum-throughput versus offered-load semantics. Report requested/

  delivered work, fulfillment, completed/pending counts, pending ages and request

  windows alongside service/delay timings.

- [x] Add completion-latency distributions only when observation counts support

  them; label unfinished requests and completed-only latency to avoid hiding

  overload behind fast successful completions.

- [x] Incorporate saturation calibration/launch selection into the common policy

  and record actual choices. Calibration failures or changing references yield

  measured limitations, not requests for a rerun.

- [x] Add sustained execution where supported, with active versus elapsed timing

  clearly separated from pacing and demand.

- [x] Regenerate embedded SPIR-V with source changes, record compiler/tool versions

  and make shader validation/reproduction steps documented and repeatable.



Exit gate: precision/placement/access combinations are capability-correct and

reference-validated; source/binary shaders agree; overload and cancellation tests

preserve their counting/timing definitions; register-only support is retained.



## Phase 7 â€” Upgrade GPU memory, transfers and matrix conditions



Implementation notes: [definitions and reproduction](phase-7-gpu-memory.md).

Automated checks: **298 tests passed**, 24 intentionally ignored; strict Clippy,

formatting, the optimized workspace build and nine memory/matrix shader

regeneration/validation checks passed. The Phase 7 release pair was packaged in

`build/current/` (subsequent phases replace only its executables);

80 then-existing settings/history files were preserved unchanged. Generated Cargo caches

were cleaned after packaging.

The user confirmed the local GPU/UI checks on 2026-11-02. Cross-hardware

qualification remains Phase 9 work. The separate Cache/VRAM presentation concern

is addressed by Phase 8 local-memory mode names and explanatory text.



Primary areas: GPU bandwidth discovery/flavors/native copies, staging transfers,

cooperative-matrix harnesses and shaders.



- [x] Add local-memory custom sizes/sweeps and supported access policies. Preserve

  cache-oriented versus stream kernels and shader versus native-copy identities.

  Do not connect different kernels as one same-kernel scaling curve.

- [x] Replace lifetime-opaque discovery reuse with recorded calibration provenance

  and automatic freshness checks appropriate to each run.

- [x] Report transition intervals and heuristic drop strength; do not present the

  existing drop-derived "confidence" as a statistical probability.

- [x] Add H2D/D2H transfer size sweeps, mean per-request cost (batch wall time /

  actual copies) and supported staging-ring or submission controls. Keep directions separate. Concurrent bidirectional work,

  if added, is an explicit new mode with independently validated accounting.

- [x] Add supported GPU matrix tile/reuse/traversal controls and output references,

  with actual input signedness, accumulation type and shape recorded.

- [x] Preserve dense tile-streaming semantics. A conventional global GEMM or a

  changed accumulation algorithm requires its own explicit mode/revision; do not

  label current GPU tile throughput as the same work as CPU full multiplication.

- [x] Add sustained coverage to applicable memory/matrix conditions through the

  shared policy, including total allocation and cancellation behavior.



Exit gate **local user confirmation passed 2026-11-02**: bandwidth/transfer/matrix

results remain distinct, accounting matches executed work, discovery limitations are descriptive, and all new GPU conditions

pass capability and output checks on their supported hardware.



## Phase 8 â€” Results, history, exports and task workflows



Implementation notes: [results/history/workflow definitions](phase-8-results-history.md).

Desktop access was not used; [interactive confirmations](phase-8-verification.md)

remain with the user. **307 tests passed**, 24 intentionally ignored; strict Clippy,

formatting and the optimized workspace build passed. The latest pair is in

`build/current/`; 80 current data files were hash-verified unchanged. Four additional

debug-folder data files were preserved in `build/recovered-debug-data/phase-8/`

before Cargo cache cleanup. Automated regression coverage includes native raw-record

round trips, annotation persistence, selected-only/include-all PNG export, histories

beyond 128 configurations, queue skipping and interrupted-plan restoration.



Primary areas: UI result grouping/charts/tuning visuals/dataset details/GPU timings,

snapshot import/export, native persistence and Slint navigation.



- [x] Make the selected dataset the primary result, initially the largest tested

  size. Keep peaks/worst points secondary and label reference selection explicitly.

- [x] Show median, observed spread, coverage and concise configuration first.

  Move low-value raw metadata and derived traffic to expandable/toggle views.

- [x] Keep raw measurements, inferred interpretations and user-selected experiments

  visually distinct. Show reference drift and validation scope near affected data.

- [x] Show measured deltas with neutral uncertainty descriptions and specific

  incompatibility reasons. Use metric direction for language such as lower latency;

  avoid implying statistical significance from a percentage alone.

- [x] Replace best-first emphasis with latest/typical values; label historical best

  and its scope. Group configurations inside families without deleting history.

- [x] Build durable history browsing and named/pinned baselines on phase-1 storage.

  Preserve timestamps, notes, build/workload provenance and imported-source identity.

  Do not silently evict the oldest records at the existing 128-entry cap.

- [x] Add explicit user-controlled archival/deletion/export actions; preserve prior

  files on migration failure and expose storage errors. Define storage limits for

  large sample histories without silently truncating measured coverage.

- [x] Make snapshot selection determine exported scope by default, with an explicit

  include-all option. Offer native machine-readable export containing selected raw

  samples; document any PNG payload limits rather than silently stripping samples.

- [x] Add task-oriented entry points using existing suite plans: CPU/memory check,

  large-dataset behavior, and comparison of user-chosen configurations. Avoid

  benchmark-driven prompts to rerun or predicted application performance.

- [x] Add queue inspection, per-condition skip/cancel, resolved allocation/runtime

  previews and preserved partial-plan status. Keep the global stop control.

- [x] Review keyboard access, chart alternatives, text scaling, long device names,

  color-independent status labels, empty/error states and crowded mode controls.



Exit gate **local checks recorded; see [verification scope](phase-8-verification.md)**: history survives restarts/migration, selected-only export round trips,

raw sample coverage remains inspectable, basic result reading requires no metadata

expansion, and no generated surface suggests rerunning because performance varies.



## Phase 9 â€” Qualification and release



- [x] Calibrate Extensive and Shorter budgets, reference/calibration coverage,

  sample-block sizes, seed ensemble size and automatic-extension limits using

  measured overhead/runtime across available hardware. Do not promise one runtime

  for every condition or machine.

- [ ] Check at least representative AMD/Intel CPU behavior, including shared-cache

  topology and available hybrid/core-placement cases; supported discrete and

  integrated Vulkan GPUs; and FP16/FP64/cooperative-matrix capability differences.

  Record untested combinations explicitly rather than treating CI as hardware proof.

- [x] Run independent optimized-kernel checks and correctness tests, controlled

  workload comparisons, memory-budget boundaries, low-memory and cancellation

  cases, worker failure/restart and long-history/import cases.

- [x] Assess live UI responsiveness and measure recorder overhead while testing.

  Local navigation/recovery and a synthetic event-cost diagnostic passed; no

  instrumented frame-time or universal overhead guarantee is claimed.

- [x] Run the repository's formatting, strict Clippy, workspace tests and release

  build. Execute hardware tests deliberately; ignored benchmarks are not required

  unit tests to run blindly in ordinary CI.

- [x] Update README, protocol, workload methodology, limitations, catalogue images,

  migration/release notes and package contents. Document exact policy defaults and

  every accounting/kernel revision affecting historical comparisons.

- [x] Reconcile all phase acceptance checklists and the phase-0 coverage ledger.



Local exit gate completed for 0.5.0-rc.1; the cross-vendor checkbox above remains open.

Required features and migrations pass, runtime defaults are backed by

measurements, validation gaps are recorded, and release documentation matches the

actual executable catalogue and controls.



## Engineering decisions that remain gated



These do not reopen the user-confirmed product direction. Resolve them with an

explicit implementation note and supporting validation before the relevant phase

ships. Ask the user if resolution would change scope or visible product behavior.



| Decision | Resolution gate |

| --- | --- |

| Exact Extensive/Shorter duration, sample minima and extension maxima | Phase 2 policy implemented (12 Ã— 250 ms / 6 Ã— 100 ms per realization, bounded extensions); Phase 9 measured defaults qualified on the local AMD CPU/GPU; external hardware calibration remains |

| Statistical intervals or descriptive spread only for each metric | Descriptive output implemented in Phases 2/4, including explicit ensemble pooling and separate native realization statistics; intervals require a justified future method |

| Native run/sample storage layout and protocol revision | Phase 1 implemented immutable version-1 JSON records, rebuildable index and additive transport v2; history browser and native export implemented in Phase 8 |

| Default block sizes, hot/cold mixes, reuse schedules and seed ensemble | Phase 4 definitions recorded in methodology/native options: 64 KiB target (curated cache bandwidth 64 B), one/three Custom realizations (Curated uses one standard realization), full-pass coverage with explicit repeated hot work; Phase 9 local calibration completed; external hardware coverage remains |

| Precision-specific GPU placement/scatter/queue capabilities | Phases 6â€“7 device checks and reference validation |

| Raw sample limits in portable snapshots | Phase 8 implemented 64 MiB PNG / 256 MiB native limits with explicit errors; native export retains full selected records |

| Fixed-work matrix/core-count comparison geometry | Phase 5 measured shapes and phase 7 tile semantics |



## Deliberately outside required scope



- Full AES cipher modes, new sieve algorithms, sparse/FP8 runners and a conventional

  GPU GEMM engine are separate extensions, not prerequisites for this plan.

- Hardware-counter collection, power/temperature telemetry, automatic clock changes

  and predicted application-level speed/energy savings are not added implicitly.

- Optional burst/rest execution can follow sustained support, but is not required

  to finish the accepted scope.

- Full Cartesian-product suites are opt-in custom plans; Extensive means adequate

  coverage of declared conditions, not every possible setting combination.



## Standard implementation checks



Use the repository wrapper and existing CI commands when relevant to a phase:



```powershell

& .\scripts\cargo.ps1 @('fmt', '--all', '--', '--check')

& .\scripts\cargo.ps1 @('clippy', '--workspace', '--all-targets', '--', '-D', 'warnings')

.\scripts\cargo.ps1 test --workspace --locked

.\scripts\cargo.ps1 build --workspace --release --locked

```



Start with focused tests for each change, then run required workspace checks.

Meaningful regressions include condition/suite coverage, compatibility fixtures,

deterministic access schedules, reference correctness, accounting, bounded sample

collection, cancelled/failed-run status and selected-only export. Do not test

assumed performance rankings such as random access always being slower.
