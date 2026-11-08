# Gluj-Bench worker protocol

`gluj-bench-worker --stdio` reads compact JSON requests from standard input and writes newline-delimited JSON envelopes to standard output. Diagnostics use standard error exclusively.

Protocol version 2 requests contain a non-empty request ID:

```json
{"protocol":2,"id":"request-1","command":"devices"}
```

Supported commands are `fingerprint`, `devices`, `benchmarks`, `families`, `resolve`, `run`, and `cancel`. A run includes its benchmark configuration. The additive family/condition contracts, native run records, CLI configuration and compatibility rules are documented in [Phase 1 contracts](phase-1-contracts.md). `benchmarks` also returns `conditions`; `families` returns their family grouping. `resolve` accepts the same arguments as `run` without executing it. Terminal run envelopes retain their existing `data` payload and additionally carry `run_record` and `run_storage` (including save failures). Contract/native-format versions are validated independently of transport version 2.

Workload intensity options `cpu_activity_percent` and `gpu_activity_percent` accept 95, 99, or 100 (default 100 for direct worker requests). CPU intensity adds short pauses independently of `cpu_core_limit`; zero or omitted core limit uses all processors selected by `thread_mode`. UI Automatic uses all physical cores. Intentional CPU pauses are excluded from measured rates/latency and results record `cpu_pacing_revision` for comparison compatibility.

```json
{"protocol":2,"id":"run-1","command":"run","arguments":{"benchmark_id":"cpu.bandwidth.cache.l1","target_duration_ms":2000,"samples":5}}
```

For controlled SMT comparisons, `run` accepts a string-valued `options` object. The default is `physical_cores`; `logical_processors` pins one worker to every discovered logical processor while preserving the same aggregate cache-instance working-set size:

```json
{"protocol":2,"id":"run-smt","command":"run","arguments":{"benchmark_id":"cpu.bandwidth.cache.l3","target_duration_ms":2000,"samples":5,"options":{"thread_mode":"logical_processors"}}}
```

GPU runs use the same string-valued options object to select an adapter returned by `devices`:

```json
{"protocol":2,"id":"run-gpu","command":"run","arguments":{"benchmark_id":"gpu.bandwidth.vram","target_duration_ms":2000,"samples":5,"options":{"device_id":"gpu:vulkan:<device-uuid>"}}}
```

GPU bandwidth IDs are `gpu.bandwidth.cache`, `gpu.bandwidth.vram`, and `gpu.bandwidth.host_link`. Cache and GPU-local memory traffic execute through raw Vulkan with embedded SPIR-V, while host-link measurements use raw Vulkan transfer commands and mapped staging buffers. Cache metric names use `estimated_effective_l2.*` and `estimated_effective_l3.*` only when stable empirical tiers are detected; result metadata always marks those identities as inferred.

GPU vector IDs are `gpu.performance.fp16`, `gpu.performance.fp64` and the unified `gpu.performance.fp32.scaling`. The FP32 condition supports `gpu_preset=register` alongside its buffered presets; the former separate `gpu.performance.fp32` is an internal reference/historical ID. These execute through raw Vulkan using embedded SPIR-V; FP16 uses explicit `f16vec2` packed-pair operands. Buffered FP32 metric names record exact workgroup-aligned byte counts. The live resolved descriptor advertises supported controls; actual sizes, calibration, transitions and timing definitions are retained in result metadata. See the [small-dataset investigation](gpu-fp32-small-datasets.md) for the current FP32 sweep behavior.

Matrix families include dense FP16, INT8, and FP8 plus their `gpu.performance.matrix.sparse.*` variants. Dense FP16 and signed or unsigned INT8 execute through raw Vulkan `VK_KHR_cooperative_matrix` pipelines when the driver advertises a compatible MxNxK/type configuration. `gpu.performance.matrix.fp16.scaling` additionally requires 16-bit storage-buffer access and streams FP16 A/B tiles across nominal power-of-two operand sets from 256 KiB to at most 4 GiB. The upper bound is also limited to one-quarter of detected device-local memory on discrete GPUs, one-eighth on other device types, and the per-binding Vulkan limit. Each loaded tile is reused eight times by default; clients can set `options.matrix_tile_reuse` from 1 through 64. The result follows the same `working_set_<bytes>.compute`, `working_set_<bytes>.bandwidth`, transition metric, and `compute_scaling_points` schema as the FP32 profile. It models dense weight-streaming pressure relevant to low-batch neural-network inference, not attention, quantization, framework overhead, or complete LLM execution.

FP8 and structured-sparse rows remain visible but disabled unless a capability-verified implementation exists; dense KHR support is never treated as evidence of sparse acceleration. Results use `operations/s`; vector FMA and matrix multiply-accumulate both count multiplication and addition as separate operations.

### FP32 system-RAM offload profiles

`gpu.performance.fp32.offload.scaling` uses the FP32 scaling kernel with a configurable `ram_offload_percent`: 50, 75 (default), or 100. The share applies to **all input and output bytes**, allocated on a separate host-visible, non-device-local Vulkan heap. Other test arrays use device-local memory. At 100% the test arrays use no VRAM. The test requires a discrete GPU, timestamp support, and compatible host-memory types; BAR-mapped VRAM and shared-memory integrated GPUs are excluded. Previously saved results with percentage-specific IDs are migrated by the UI to the consolidated ID with their percentage preserved.


```json
{"protocol":2,"id":"ram-offload","command":"run","arguments":{"benchmark_id":"gpu.performance.fp32.offload.scaling","target_duration_ms":2000,"samples":5,"options":{"device_id":"gpu:vulkan:<device-uuid>","ram_budget_percent":"20","vram_budget_percent":"25","gpu_activity_percent":"99","ram_offload_percent":"75","dataset_mode":"single","dataset_bytes":"268435456"}}}
```

`ram_budget_percent` accepts 20–80 (default 20), limiting the host region to that share of installed RAM, the host heap, 80% of available RAM, and available RAM minus 2 GiB, with another 16 MiB reserved for allocation overhead. Any VRAM region respects `vram_budget_percent`, also reserving 16 MiB. `max_working_set_bytes` is an optional total-dataset cap, at least 262144 bytes; it also applies to the ordinary FP32 scaling profile. Sweeps must allow at least three tiers; single-dataset mode requires one valid aligned tier. All offload percentages use the same quarter-region alignment, preserving exact shares and matching tier sizes when their allocation caps match. `arithmetic_iterations` retains the existing FP32 scaling behavior (default 64, clamped to 1–1024).

All CPU/GPU compute scaling tests accept `dataset_mode` (`automatic`, `sweep`, or `single`; default automatic). Explicit sweep/single modes require `dataset_bytes` of at least 262144 bytes within the allocation/device limit. An oversized explicit request returns `dataset_exceeds_budget`, with the allowed size. Kernel alignment and matrix shapes can round the actual tested size downward. CPU sizes aggregate workers; GPU FP16 matrix sizes count operands. Results record `dataset_mode` and `requested_dataset_bytes` alongside actual tier names. Single mode produces one tier and does not claim a measured size-dependent memory-pressure transition.

Alongside `working_set_<bytes>.compute`, `.bandwidth`, and `.reference_delta`, each tier includes `.host_bandwidth`: total effective traffic multiplied by the RAM share. It counts nominal RAM input reads plus output writes over GPU timestamp time. Repeated accesses may hit GPU caches; it is not measured PCIe traffic. CPU work, setup fills, output validation and GPU pacing intervals are excluded from timestamps.

GPU compute scaling profiles return `working_set_<bytes>.gpu_execution_time` and `.end_to_end_time`, in `ns`, with measured per-sample statistics. Each sample records a Vulkan GPU timestamp duration and a paired host elapsed duration around the complete measurement call. Both are divided by the calibrated number of complete dataset passes in that sample batch. Host timing includes command recording, submission, completion, query-result retrieval and GPU activity pacing; GPU timestamp timing excludes host overhead and pacing. Setup, warmup and reference runs are excluded.

Metadata records `gpu_timing_method` (`gpu_timestamp_and_host_elapsed_per_pass_v1`), `gpu_timing_definition`, `gpu_timing_batch_passes` (`dataset bytes:passes per sample` tuples), and `gpu_timing_samples_paired` (`true`). Batch lengths may vary across datasets. These are execution/end-to-end durations, not separate memory-stall durations. The old `.gpu_busy_estimate` / `.gpu_wait_estimate` metrics and `gpu_busy_wait_*` metadata are no longer generated. The UI hides retired estimates from older records and shows unavailable timing data as not recorded. Throughput/reference comparisons remain performance ratios. CPU profiles are unchanged.


Metadata records `ram_offload_percent`, `ram_budget_percent`, `host_allocation_budget_bytes`, `host_allocated_test_buffer_bytes`, `device_allocated_test_buffer_bytes`, `host_memory_placement`, `offload_profile_revision`, and `offload_tier_bytes` (`total:host:local` tuples). `result_validation` describes boundary/interior output checks for each measured region and dataset. Explicit allocation does not simulate automatic VRAM overflow or page migration. Qualitative tuning guidance recommends exploring a lower GPU core-frequency limit or reducing application batch size/concurrent GPU jobs, and reports measured reference differences. It keeps the RAM offload share fixed for frequency comparisons and does not generate a numerical clock-reduction estimate. GPU activity pacing is excluded from TOPS, so application-level completion time must be checked for workload reductions. Protocol version remains 2.

The worker accepts one active run, emits an acceptance response, zero or more progress responses, and one terminal result or error using the run request ID:

```json
{"protocol":2,"id":"run-1","type":"success","data":{"accepted":true}}
{"protocol":2,"id":"run-1","type":"progress","data":{"fraction":0.5,"phase":"write","message":"..."}}
{"protocol":2,"id":"run-1","type":"result","data":{}}
```

Cancel an active run using a separate request ID and the original run ID:

```json
{"protocol":2,"id":"cancel-1","command":"cancel","arguments":{"request_id":"run-1"}}
```

The cancel request receives its own acknowledgement. The original run subsequently terminates with a `cancelled` error. Concurrent run attempts return `busy`.

Bandwidth metrics use `bytes/s`. Each metric includes its own sample count, minimum, median, maximum, and standard deviation. CPU RAM copy and host-device transfers count useful payload. GPU cache/VRAM copy counts read-plus-write device-memory traffic so it can be compared with peak memory bandwidth; its useful payload rate is half the reported value. Workload metadata records the convention and buffer sizes.


## Extensive measurement and native observations

Run options accept `measurement_policy` (`extensive`, default, or `shorter`). Requests are resolved to independent duration/count floors: The `bounded-v2` policy uses at least 6 observations of 250 ms per condition for Extensive, and at least 3 of 100 ms for Shorter. References/calibration use 3 observations of 100 ms. Explicit higher sample counts and durations remain honored; legacy requests are accepted and raised to the applicable floors. Variable observations remain in the result; variability never triggers a rerun recommendation.

The additive native `run_record` envelope includes actual raw batch observations with `timing_scope`, policy/condition provenance, reference anchors, coverage and optional `phase_timings`. Phase intervals are inclusive and can overlap; do not sum them as runtime. Old records omit these fields and load with defaults. Paired GPU timing metadata can include `gpu_timing_batch_passes_series` with actual per-observation work; a fixed pass count is emitted only when it is constant. Missing legacy timings are not reconstructed. See [Phase 2 measurement notes](phase-2-measurement.md) for clock semantics, coverage failures, cancellation and correctness checks.


## Phase 3 catalogue compatibility

The desktop groups advertised conditions into fifteen family cards and prepares explicit `SuitePlan` instances for curated/custom queues. Worker benchmark IDs still identify executable modes; family grouping does not combine scores. Unimplemented FP8/sparse descriptors are retained for capability inspection and omitted from runnable UI plans.

Legacy `cpu.performance.single_thread.integer.i64` run/resolve requests map to the canonical `cpu.performance.integer.i64` one-core register condition after validation of the original request contract. Successful summaries use the canonical ID; `run_record.requested_benchmark_id` preserves the legacy request and `resolved.benchmark_id` identifies execution. Historical summaries remain unchanged. Bandwidth descriptors may advertise `estimated_peak_test_buffer_bytes`; CPU algorithm descriptors may advertise the corresponding per-worker estimate. These are planning estimates, not process/driver memory guarantees.

## Phase 4 access and accounting compatibility

CPU buffered FP32/INT64/ASCII and bandwidth condition descriptors advertise
`access_pattern` (`sequential`, `shuffled_blocks`, `rotating_regions`, `scattered`,
`hot_cold`), `access_seed` (u64), `access_seed_policy` (`fixed`/`fresh`),
`access_realizations` (1/3), and `access_block_bytes` (64–1,048,576, multiples of 64).
Register arithmetic rejects buffer access controls. Latency advertises seed/ensemble,
working-set `dataset_bytes` and `locality_block_bytes`; its dependent-chain mode
continues to define the access pattern. ASCII adds `ascii_content`. RAM bandwidth
adds `dataset_mode` (`automatic`/`single`/`sweep`) and `dataset_bytes`.

Fresh seeds freeze once at execution; requested provenance remains intact and the
resolved/native record stores the actual root. Samples/blocks identify realizations;
realization parameters include per-condition descriptive statistics. Every declared
realization must meet the full measurement policy. Summary spread pools equal
coverage explicitly. See [Phase 4 methodology](phase-4-access.md).

Current CPU/GPU copy summaries use useful payload and separately named
`copy_accounted_read_write` metrics (tier-prefixed where appropriate). GPU raw copy
observations retain accounted read/write traffic; their payload summary is derived
by division by two. Accounting revision metadata prevents equivalence with older
results. Archives and imported data are not rescaled. These are additive fields
and separately versioned workload/accounting definitions; transport remains v2.


### Phase 5 CPU controls

FP64 (`fp64_preset`) and POPCNT (`popcnt_preset`) support register/cache/RAM/custom
presets with buffered size/access controls. DEFLATE advertises `codec_corpus`,
`codec_corpus_bytes`, `codec_block_bytes`, `codec_level` and sequential/shuffled
independent-block access. Sieve adds `sieve_limit`. Matrix adds `matrix_shape`
(`legacy`/`square`/`tall`/`wide`), `matrix_reuse` and `matrix_k_block`; explicit
shapes reject conflicting legacy sizing. CPU `sustained_seconds` accepts 0–120.
Supported workloads advertise `core_experiment` (`none`/`fixed_aggregate`), which
records the workload promise; callers still submit one configuration per worker count.
Always inspect the descriptor option domains before constructing requests.

New workload revisions retain actual vector allocation, conservative reserve,
per-worker placement/input shards and validation provenance. Sustained records
retain raw observations and scoped early/late summaries. Transport version remains 2.
See [Phase 5 definitions and timing scopes](phase-5-cpu-workloads.md).


### Phase 6 GPU vector controls

Existing `gpu.performance.fp16` and `.fp64` IDs retain register defaults and advertise
`gpu_preset` (`register`, `small`, `vram`, `custom`, compatibility `legacy`), dataset
size/mode, arithmetic iterations, allocation budget, seed/ensemble and sustained
coverage. Buffer mode on FP16 additionally requires 16-bit buffer storage support.
FP32 profile/offload IDs retain their placement and demand controls.

Buffered vector `access_pattern` accepts `sequential`, `shuffled_blocks`,
`rotating_regions`, or `lane_scattered`. GPU block grain is fixed at 256 vectors;
CPU `access_block_bytes` is not advertised. Register mode rejects buffer access
controls. `sustained_seconds` is 0–120 and keeps active GPU service separate from
wall time, or observation windows for offered load. Precision-inapplicable placement
and demand keys are rejected. Transport remains version 2.

Offered-load metrics add pending submitted/unsubmitted counts, mean/youngest pending
age, scheduled request window and eligible completed-only completion percentiles.
Raw completed-delay arrays live in realization parameters per observation; the
aggregate completion count accompanies percentile summaries. Buffered results carry
precision/access revision and the embedded shader build manifest. See
[Phase 6 definitions and reproduction](phase-6-gpu-vector.md).


### Phase 7 GPU memory, transfer and matrix controls

GPU memory adds `dataset_mode` automatic/single/sweep, explicit `dataset_bytes`,
`memory_operation` all/read/write/copy, `memory_copy_path` shader/native, and
sequential/shuffled_blocks/rotating_regions/lane_scattered access with seed policy
and one/three realizations. Native copy requires sequential access.

Host transfers add the same sizing controls, `transfer_direction`
both/host_to_device/device_to_host, `staging_buffers` 1/3/8, `transfer_batch`
1/8/32, `transfer_submission` serial/batched, RAM/VRAM budgets and sustained
coverage. Serial requires batch 1. Both directions remain separately measured.
Mean request cost is batch wall time divided by completed copies, not an
individual-latency distribution.

Buffered cooperative matrix adds `matrix_tile_shape` from per-device advertised
native tokens, `matrix_tile_reuse` 1-64 and `matrix_traversal`
sequential/shuffled_tiles/rotating_tiles with seed/ensemble coverage. Register
mode rejects traversal controls. All applicable families accept sustained_seconds
0-120. New records retain allocation, calibration, workload/access revision and
shader manifest metadata. Transport remains version 2. See
[Phase 7 methodology](phase-7-gpu-memory.md) and
[pending user confirmations](phase-7-verification.md).


### Phase 8 history and portable exports

Worker transport remains version 2; immutable native run records remain version 1.
UI-only names, notes, pins and archive state live in `run-history/annotations.json`.
Native portable JSON uses `format: gluj-bench-native`, `version: 1`, `records` (full
RunRecord values) and `annotations` keyed by exported run ID. Export selects exact
records, retaining raw arrays, identities, validation and provenance. Import assigns
new local IDs and marks ImportedComparison, preserving original source identity in
source_reference. The native import/export limit is 256 MiB with explicit failure;
no truncation or automatic history eviction is performed.

PNG payload version remains 1. Selected-only is now the default exported summary
scope; selected_result_indices refer to the exported results array. Include-all
remains an explicit choice. Legacy all-results PNG payloads remain supported.
PNG limits remain 64 MiB, 128 displayed rows and 10,000 embedded summaries; PNG is
not full raw-sample export. See [Phase 8 definitions](phase-8-results-history.md).


### Phase 9 qualification and storage location

GLUJ_BENCH_DATA_DIR optionally redirects UI/worker settings, latest-result cache,
run-history and metadata cache; unset/empty retains the normal location. Worker
binaries remain beside the UI. RunRecord.build_revision embeds the qualified
source/toolchain build identifier when provided by the release build. Packages
include build-info.json with source SHA256, Git context, target and toolchain.

The corrected buffered matrix input definition uses matrix_profile_revision
`tile-streaming-4-signed-input` and explicit matrix_input_pattern metadata. Kernel
and transport contracts remain otherwise additive. Worker disconnection is a local
UI transport failure: it marks current plan work failed/skipped without fabricating
a successful score or silently restarting work. Reconnection is explicit.
