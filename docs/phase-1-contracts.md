# Phase 1 implementation notes

Implemented 2026-10-31. This phase establishes contracts and storage; the new
measurement scheduler and family-card interface are subsequent phases.

## Catalogue and configuration

Core exports version-1 `FamilyDescriptor`, `ConditionDescriptor`,
`ResolvedCondition` and `SuitePlan` records. All 35 current descriptors have a
destination in the fifteen-family catalogue, including disabled placeholders.
Descriptors advertise option domains, defaults, combination rules, supported
devices, per-mode capabilities, metric direction and timing scope. This does not
make a disabled placeholder executable or combine unlike scores.

The registry, CLI, worker resolve request and UI use the same resolver. Providers
can override the compatibility descriptors. CPU capabilities include discovered
core limits and cache-level availability. GPU matrix register and buffered modes
have separate device support, preserving register execution when storage-buffer
features are absent. Phase 3 will expose the capability-aware family picker.
Allocation headroom, matrix alignment and selected-worker cache partitions still
require runtime validation; a size preview is not an allocation guarantee.

Unknown options, invalid values, ignored dataset-size controls, contradictory
register/buffer choices, simultaneous traffic targets and conflicting RAM-budget
controls are rejected. Explicit nonzero CPU core limits require physical-core
selection. CPU physical-core defaults now agree between direct worker and UI
requests; some older direct-worker compute paths defaulted to logical processors.
With a discovered physical-core count, automatic physical allocation resolves to
that explicit count. CPU profile RAM defaults use the UI's 20-percent policy;
an explicit legacy `ram_budget_mib` remains available independently. GPU direct
requests still default to maximum throughput; the UI's traffic setting is an
explicit option, not a hidden resolver default.

The UI omits global settings that do not apply to the selected condition before
resolution. Explicit caller-provided worker/CLI options are validated strictly.
Old settings and benchmark preferences continue through their existing version-1
adapters; phase 3 will migrate their visible catalogue organization.

## Identity and comparisons

Measurement policy is separate from workload options. `ConditionIdentity`
retains resolved options and actual definitions recorded by the runner, including
revisions, accounting, dataset, shape, placement and timing fields. Existing
runner identities are deliberately incomplete: equal legacy records permit
descriptive comparisons, not a claim of equivalent fully specified experiments.
Kernel, device, timing or workload differences return specific incompatibility
reasons. Policy-only changes are descriptive. `compare_runs` additionally checks
completion, hardware/driver context, source and raw-sample coverage.

An application version is recorded for new runs. An optional
`GLUJ_BENCH_BUILD_REVISION` environment value, set when compiling, supplies build
revision provenance. Missing revisions remain missing. Existing UI display
normalization retains `original_benchmark_id`, and comparisons no longer assume
automatic equivalence between renamed historical IDs and current kernels.

## Samples and run records

`RunRecord` format version 1 contains a unique run ID, UTC Unix-millisecond start
time, source, application/build provenance, hardware snapshot, requested and
resolved configuration, status, validation scope, original summary and raw records.
Samples record chronological sequence and offset, block and realization IDs,
metric/unit, useful work, active/elapsed time, reference anchor and validation.
Realizations retain seeds and generator revisions; blocks and reference anchors
are explicit records. Validation rejects unknown versions, inconsistent identity,
invalid references/order/timing, non-finite numbers and successful summaries on
failed/cancelled runs. Suite vectors define execution order and intended coverage;
duplicate exact resolved conditions are rejected.

`BenchmarkProvider::run_with_samples` can preserve partial observations even when
its outcome is cancellation or failure. Existing runners use the default summary
adapter: `sample_coverage = summary_only`, empty raw samples and validation
`not_recorded` unless the runner actually reports validation scope. Phase 2 must
produce real observations through this boundary. Summary statistics are never
expanded into invented samples. `legacy-fixed-v1` policy provenance describes the
requested legacy sample/duration configuration; existing runner clamps and timing
differences remain for phase 2 to replace. No Extensive/Shorter budgets have been
introduced here.

## Native storage and migration

The worker saves every completed, cancelled or failed run before sending its
terminal response, in `run-history/runs/<run-id>.json` beside the executables.
Records are immutable. Publication writes a temporary file, flushes it and uses
an atomic no-clobber hard link; storage therefore requires hard-link support.
`index.json` is a compact, replaceable cache of IDs, source, timestamps, status
and sample counts. Normal writes read the compact cache and reconcile missing
record IDs. `RunStore::rebuild_index` validates authoritative records and recovers
a missing, stale or corrupt index. Malformed/future run files are preserved and
reported. Temporary files are excluded from index recovery. No archive size cap
or automatic eviction is applied.

The existing `results.json` remains a latest-result display/comparison cache. Its
current replacement/128-entry behavior cannot remove native run files. Full
history browsing and native export are phase 8. CLI runs use the same archive.
Storage failure is reported separately from measurement success: the terminal
response carries the complete record, and the UI retains an unsaved record in
session memory and shows the failure. Unsaved session memory is not durable.

Before display normalization, startup copies version-1 results into native
`legacy_local` records. Migration is idempotent by source, ordinal and exact
summary contents. Original IDs/revisions/metrics remain intact; no old timestamp,
configuration, build or samples are fabricated. Original files are not rewritten
by migration. Latest-result cache entries retain their native run IDs so a
restart does not archive the same local measurement again. Snapshot imports archive original summaries as
`imported_comparison`; they never become local measurements. Existing PNG payload
fixtures remain covered by snapshot round-trip tests.

## Protocol and CLI

Transport protocol remains version 2. Additive changes preserve the existing
`data` payload for result consumers:

- `families` returns the declarative catalogue.
- `benchmarks` additionally returns `conditions` alongside the old descriptors.
- `resolve` accepts the same arguments as `run` and returns its resolved condition.
- Terminal run responses additionally contain `run_record` and `run_storage`.

Transport-version mismatches remain errors; the UI explicitly reports mismatched
worker responses. Removing/renaming fields, changing existing command payloads or
timing/unit meanings requires a protocol revision or a separately versioned
contract. Readers must validate run/contract versions independently of transport.

The CLI accepts `families`, `resolve <id>` and
`run <id> --config <json-file> [--json]`. A config file uses `BenchmarkConfig`:

```json
{"options":{"thread_mode":"physical_cores","cpu_core_limit":"4","int64_preset":"register"}}
```

Omitted duration/sample fields receive shared legacy defaults. Unknown config
fields and non-string options fail rather than silently falling back.

## Validation

The pre-change workspace test baseline passed. New regressions cover catalogue
coverage, resolver idempotence and UI/worker parity, invalid combinations, mode
capabilities, suite order/duplicates, immutable writes, index recovery, legacy
fixtures, source identity, raw-record round trips and partial cancellation.
Hardware benchmark smoke tests remain deliberately ignored; this phase makes no
performance or cross-device qualification claims.

Final checks: 233 workspace tests passed, with 22 intentional hardware/UI tests
ignored; strict workspace/all-target Clippy and formatting checks passed. A live
release-worker smoke check returned fifteen families and 35 conditions, resolved
an explicit one-core register configuration and durably saved an unsupported-test
failure without a score. Smoke records are isolated under the ignored
`target/phase-1-worker-smoke/` directory.
