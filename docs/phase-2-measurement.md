# Phase 2: measurement implementation

Current defaults (2026-11-06): six measured observations at 250 ms each; Shorter uses three at 100 ms. Curated uses standard access with one realization. Randomized access and additional seed realizations are available in Custom. The historical implementation notes below retain the earlier design; see [current follow-up notes](ui-redesign-implementation.md#lean-default-workloads-2026-11-06).


Date: 2026-10-31

Phase 2 installs bounded measurement, raw observation capture, independent correctness checks and descriptive presentation across the live CPU/GPU runners. Hardware-wide calibration of defaults remains the Phase 9 release gate.

## Shared policy

| Policy | Minimum observations per condition/metric | Minimum active time per observation | Warmup budget |
| --- | --- | --- | --- |
| Extensive (default) | 12 | 250 ms | 250 ms |
| Shorter (explicit setting) | 6 | 100 ms | 100 ms |

References, launch tuning and cache/dataset calibration use a separate shared budget: at least six observations for Extensive or three for Shorter, with at least 100 ms per observation. Preparation and complete traversal warmups can exceed these budgets. A requested higher sample count raises total coverage independently of duration; it never divides a fixed small duration into more observations. Counts above 64 or durations above 120 seconds per condition are rejected. Lower requests are raised to the policy floors.

GPU work is accumulated in bounded batches until the observation duration is covered; a 1,024-batch limit fails explicitly if coverage is impossible. Work/pass counts and actual overshoot are retained. CPU runners likewise count actual completed work, excluding deliberate pacing from active rates. This is a predetermined coverage policy, not a stability search: there is no sample rejection, favorable-result selection, or retry caused by spread/drift. Slow valid observations remain in the result. Atomic work and allocation can exceed duration estimates; this is not a hard wall-clock deadline.

The global Shorter setting applies to all queued tests and is persisted. Progress reports actual completed observations. Estimates describe minimum measurement time and explicitly exclude additional preparation, references, calibration and validation; they are not calibrated end-to-end predictions.

## Observations, clocks and phases

Native run records retain chronological raw observations, batch work counts, actual active/elapsed time, units, condition and block identity, fixed-realization identity, reference anchors and policy provenance. Raw samples are captured by runners, never reconstructed from minimum/median/maximum summaries. Parallel CPU workers share a recorder but keep worker identities. Work inside one batch is not presented as independent samples.

CPU buffered size sweeps use three interleaved forward/reverse/forward blocks, with preparation restored for every visit and statistics computed over all observations for each size. Register reference anchors before/after a sweep remain separate from the measured datasets. Existing latency seeds remain recorded. New randomized workload ensembles are deferred to Phase 4; they are not silently mixed into Phase 2 scores.

Timing scopes distinguish CPU active execution, host copy elapsed time, paired GPU timestamp/host elapsed clocks, and offered-load observation windows with aggregate service. GPU/host clocks need not match. Aggregate service can exceed a wall window and is not clamped. Offered throughput counts completions observed before cutoff; pending work remains explicit. Windows with no completions have no invented completion-latency or active-capacity observation. Conditional completion-only metrics can have fewer observations than throughput. Each offered sample starts with an empty queue, so these windows must not be called steady-state request-latency distributions or individual-request percentiles. Further arrival/sustained behavior belongs to Phase 6.

`phase_timings` records named preparation, warmup, reference/calibration and validation intervals with offsets from the run recorder's monotonic origin. These are **inclusive intervals**, can overlap across workers or contain nested intervals, and must not be summed into a total runtime. Untimed validation means excluded from benchmark timing, not free of wall time. Runner wall duration is recorded separately. Older records default to an empty phase list; absence means not recorded.

Missing instrumentation, insufficient count/duration coverage, invalid numeric observations and correctness failures cannot produce successful scores. Valid observations collected before failure/cancellation remain partial native records. Cancellation is checked after provider completion too, preventing a late cancellation from publishing a score. Worker protocol cancellation/busy handling stays responsive while the run holds the registry.

## Kernel checks and revisions

Release assembly inspection on `x86_64-pc-windows-gnullvm`, Rust 1.97.0, verified the 4,096-iteration register loops: eight INT64 multiply/add chains, four POPCNT operations, eight AESENC operations, scalar FP32/FP64 arithmetic, and eight packed AVX2 FMA chains. INT64 black-box barriers prevent recurrence collapse. POPCNT now uses distinct transformed inputs rather than equivalent rotations. AVX2 feedback now normalizes the eight-lane reduction, preventing repeated chunks from overflowing to infinity. Changed register kernels carry revision 5; the measurement harness carries revision 4. Inspection can be reproduced with:

```powershell
& .\scripts\cargo.ps1 @('rustc','-p','gluj-bench-cpu','--lib','--release','--','--emit=asm')
```

Independent checks cover integer recurrences, population count, software AES rounds, floating point references, sieve count, codec round trips, matrix scalar products, ASCII classification, pointer-chain integrity, bandwidth patterns and copies. Floating-point reference calculations avoid the known MinGW FP32 `mul_add` reference discrepancy by using wider arithmetic before rounding. SIMD repeated-chunk tests check finite output.

GPU vectors/matrices use untimed boundary/interior output readback against independent references. Bandwidth and transfer buffers receive known input patterns and validate actual used destinations. Their revision changes to `vulkan-storage-vector-2-known-pattern` prevent automatic comparison with the old initialization/kernel contract. GPU timing records retain per-observation batch/pass counts rather than assuming every sample used the same work. Unsupported hardware paths remain unavailable; their correctness is not inferred from another GPU's result.

## Presentation and compatibility

Results report count, minimum, median, maximum, SD/median and numeric reference drift as descriptive observations. They do not claim confidence intervals or a significance probability. Missing legacy spread/timing data is labeled not recorded. Reference gaps are static neutral comparisons, not animated red loss or inferred stall time. The sieve no longer claims an unmeasured compute-bound classification.

Variability-driven rerun/retest instructions are removed from live generated guidance and legacy presentation. Raw imported archives remain intact. Manual runs and optional user-directed configuration experiments remain available. Phase 1 history and protocol formats remain additive: `phase_timings` and raw timing-scope fields have backward-compatible defaults.

## Validation

Scheduler/contract tests cover independent count/duration floors, bounded extensions, interleaved size coverage, reference separation, worker chronology, progress/pacing propagation, variable and slow observations, invalid/missing samples, partial failures, distinct GPU clocks, inclusive phase intervals and late cancellation. CPU and GPU reference tests exercise correctness and finite arithmetic; hardware-dependent tests are explicitly opt-in. Validation completed on 2026-10-31:

- Workspace tests: **252 passed**, 23 explicitly ignored hardware/manual tests; no failures.
- Strict workspace/all-target Clippy (`-D warnings`), formatting and whitespace checks passed.
- Optimized release build of the complete workspace passed.
- Release worker smoke checks passed on AMD Ryzen 9 5950X / Radeon RX 7900 XTX: CPU INT64 and AVX2 register modes, CPU buffered FP32, GPU register/buffered FP32, GPU cache bandwidth, host/device transfers and offered-load RAM offload. Each completed with full raw coverage and a durable native record; saved JSON matched the response.
- Real stdio concurrency/cancellation check returned `busy` for a second run, acknowledged cancellation promptly and saved the first run as cancelled with its completed observation and no successful score.
- The opt-in GPU precision reference smoke also passed on the available supported precisions. These checks establish local correctness/coverage, not cross-device performance calibration or cooperative-matrix support on other hardware.

Generated inspection/smoke evidence is under `target/phase-2-*`, outside user run history.

## Next phases

Phase 3 consolidates the catalogue into family cards. Phase 4 adds explicit access randomness: customized workloads default to sequential, and curated plans measure sequential and randomized conditions with equal coverage and separate results. Random data values alone do not claim cache eviction/misses. Phase 9 calibrates duration defaults on representative hardware before a release claim.
