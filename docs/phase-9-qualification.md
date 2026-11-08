# Phase 9 - local qualification and release candidate

Qualification: 2026-11-02–03. Candidate: **0.5.0-rc.1**. This is a locally qualified candidate;
cross-vendor release qualification remains open. No external release was published.
The user's original settings/history were preserved; qualification uses isolated
`GLUJ_BENCH_DATA_DIR` folders under `build/qualification/phase-9/`.

## Hardware and scope

- AMD Ryzen 9 5950X: 16 physical cores / 32 threads; 16 L1/L2 instances and two L3
  instances. This exercises shared-cache topology, not a hybrid-core scheduler.
- Radeon RX 7900 XTX: discrete Vulkan GPU, API 1.4.359, raw driver version 8389024,
  subgroup 64, timestamp period 10 ns. FP16 arithmetic/storage, FP64, INT8 storage
  and dense FP16/INT8 16x16x16 cooperative matrices are advertised locally.
- 40 GiB installed RAM, configured at 2800 MT/s. Windows x64 GNU LLVM target;
  Rust/Cargo 1.97.0, pinned Cargo.lock; shader manifests record glslang 16.6.0.

The remaining hardware matrix is explicit: Intel CPUs and hybrid core placement,
Intel/AMD integrated/shared-memory GPUs, NVIDIA GPUs, other drivers and devices
without the locally available precision/storage/matrix capabilities were not
physically tested. Contract/unit tests cover unavailable-mode behavior, but do not
replace measurements on those devices. OS-wide RAM exhaustion, forced GPU reset,
and thermally controlled power/temperature studies were not performed. Explicit
budget/limit rejection and cancellation were tested without exhausting the system.

## Hardware verification

**56 effective release-worker cases passed** after correction of the INT8 initializer,
plus three CPU cache validation-provenance checks. Cases cover:

- Extensive/Shorter CPU INT64, GPU FP32 buffered compute, GPU memory and directional
  transfer coverage; scalar FP32/FP64, AVX2 FP64, AES rounds, POPCNT, sieve, codec
  compression/decompression, ASCII, fixed-geometry CPU matrix, CPU RAM bandwidth
  and cache/RAM dependent latency.
- Separate sequential/shuffled CPU buffered conditions, equal three-realization
  coverage, FP64/POPCNT buffers, FP16/FP64 GPU buffers, every local-memory access
  pattern, native-copy sweeps, and fresh cache discovery.
- H2D/D2H sweeps with 1/3/8 staging slots, serial/batched submissions and batches
  1/8/32. Throughput and mean per-copy cost remain separately identified.
- FP16/INT8 matrix sequential/shuffled/rotating traversal at reuse 16, with exact
  scalar output references and separate ensemble observations.
- FP32 50% direct host placement with low offered load and 100% host placement with
  an overloaded arrival stream. The latter retained 37 completed and 63 unserved
  requests per representative observation, including submitted/unsubmitted pending
  counts and pending ages; unfinished work was not hidden behind completion latency.
- Fifteen-second sustained CPU and memory coverage; excessive allocation requests,
  native irregular copies, unsupported matrix tiles and register access options
  rejected explicitly before producing a score.

Every successful case required complete real sample coverage, confirmed durable
storage, and exact equality between the terminal native record and its saved JSON.
Independent validation scopes were retained. Cancellation acknowledged in about
0.17 ms in the local stdio check, rejected a competing run as busy, and retained one
completed observation with partial coverage and no successful score. This latency
is an observed protocol result, not a portable timing guarantee.

The first INT8 cases failed output validation and withheld scores. Buffered A was
initialized to positive ones although the independent reference expected signed -1
(or unsigned 255). A now uses the intended byte pattern and B stays 1. All three
INT8 traversal cases passed after the fix. `matrix_profile_revision` is
`tile-streaming-4-signed-input`; input-pattern metadata makes the definition explicit.
No scores were retroactively rescaled or edited. Initial failed evidence is retained.

## Sampling calibration and overhead

Defaults are retained: Extensive 12 observations of at least 250 ms, Shorter six of
at least 100 ms, warmups 250/100 ms. Reference/calibration floors are six/three
observations of at least 100 ms. Extensions are predetermined and bounded at 1,024
batches; spread or favorable speed never decides whether to extend or retain data.
Three-realization curated conditions each receive full coverage. CPU buffered visits
retain forward/reverse/forward blocks. These choices met count/duration coverage
locally; they are not universally optimal calibration claims.

| Case | Main observations | All recorded observations, including calibration/diagnostics | Observed total wall time |
| --- | --- | --- | --- |
| cpu-register-shorter | 6 | 6 | 0.714 s |
| gpu-vector-shorter | 6 | 99 | 16.018 s |
| gpu-memory-shorter | 6 | 6 | 1.316 s |
| transfer-shorter | 6 | 12 | 0.777 s |
| cpu-register-extensive | 12 | 12 | 3.266 s |
| gpu-vector-extensive | 12 | 198 | 31.534 s |
| gpu-memory-extensive | 12 | 12 | 3.697 s |
| transfer-extensive | 12 | 24 | 3.367 s |

GPU vector launch/reference calibration dominates these small-data examples. The UI
therefore states minimum measurement time and explicitly adds preparation, sweeps,
references and validation; 0.6/3 seconds is not an end-to-end promise. Inclusive
phase intervals can overlap and are not summed into a false total. Runtime is
reported rather than turned into a recommendation to rerun.

A deliberate synthetic-recorder diagnostic processed 20,000 events with and without
progress enabled: approximately 1.4 microseconds/event in the unoptimized test build
on this CPU. This measures allocation/locking/accounting overhead for that narrow
case, not full multiworker contention or an application performance correction.
No synthetic diagnostic observations become native benchmark scores. UI navigation,
history I/O and engine recovery were exercised while the application stayed live;
Slint redraw caps and progress coalescing remain in place. No instrumented frame-time
or universal UI-overhead bound is claimed.

## Optimized code, automated checks and compatibility

314 default workspace tests passed; 25 opt-in hardware/preview/diagnostic tests are
explicitly ignored in the default suite. Strict workspace/all-target Clippy,
formatting and whitespace checks passed. All three precision-profile and all nine
memory/matrix shader sources regenerated identically and passed SPIR-V validation.
The complete optimized workspace builds successfully.

Optimized CPU assembly inspection found the retained 4,096-iteration loops with
eight INT64 multiply chains, four POPCNT operations, eight AESENC rounds, and eight
packed FP32 / FP64 FMA chains. Scalar FP32/FP64 emitted scalar multiply/add instructions.
Independent unit and hardware output references cover the resulting work. Evidence
and selected assembly excerpts are in the local qualification folder.

Captured compatibility fixtures cover original version-1 scalar/SIMD/single-thread,
latency and offload summaries, settings missing new fields, and an original PNG
with the old all-results payload. Original bytes remain available while the UI
adapts display copies. [The coverage ledger](benchmark-coverage-ledger.md) reconciles
every descriptor, including aliases and unavailable sparse/FP8 capabilities.

## UI, failure and persistence

Interactive Windows checks verified local-memory family/mode naming, nearby selected
results with a bounded scrolling catalogue, raw history inspection, named notes and
pins, archive/restore and native JSON save/import. Native import preserved raw
observations and annotations and retained ImportedComparison source identity.
Pin/name/notes survived an actual application restart in the isolated folder.

A deliberate child-worker termination during an active run produced Failed plan
status without a successful score; completed native history remained intact.
The UI exposed Reconnect engine and recovered on request. Run actions are disabled
while offline, and plan starts also reject offline execution. Errors/status now
remain visible on the benchmark page. Abrupt worker death can lose observations
not yet committed by the worker; graceful cancellation retains partial observations.

Task previews identify access, dataset and copy/submission controls instead of
showing apparent duplicate modes; device names replace opaque IDs in custom
previews. Long queue lists scroll independently so global Stop remains reachable.
Live cancellation exposed a plan-status defect: cancel-one continued execution but
marked pending conditions skipped. The final candidate preserves Pending/Running
for continuing work; global Stop still skips unstarted conditions. A regression
checks both paths. Installed-memory summaries now use firmware capacity when
available and explicitly label an OS-visible fallback.
Further visual checks, export scope and queue interaction outcomes are recorded in
`build/qualification/phase-9/ui-verification.json` after final packaging checks.

## Reproduction and artifacts

```powershell
python scripts/qualify-local.py --worker build/current/gluj-bench-worker.exe --output build/qualification/new-local-run
.\scripts\compile-profile-shaders.ps1 -Check
.\scripts\compile-memory-shaders.ps1 -Check
.\scripts\cargo.ps1 test --workspace --locked
.\scripts\cargo.ps1 clippy --workspace --all-targets -- -D warnings
```

The qualification script deliberately runs workloads, not just resolver checks.
Use an explicit output folder; it starts one worker and one workload at a time.
`--only` selects a case-name substring. Data isolation does not change workload
semantics. Required checks should be completed without unrelated build activity
competing with performance measurements.

The local package includes executables, license/notice/branding, current screenshots,
protocol and methodology, coverage/qualification/release notes, VERSION.txt and build
provenance. Build provenance records toolchain, target, source SHA256 and a revision
embedded in native records, including an honest dirty-worktree flag. SkipBuild
packaging rejects missing/stale source provenance. Existing versioned packages are
preserved rather than silently replaced. No user data or qualification histories
are included in the ZIP. Generated Cargo caches are reclaimed after packaging.

Final rebuilt executable verified interactively: cancelling one condition preserved
Cancelled/Running/Pending statuses; global Stop left Cancelled/Skipped statuses.
