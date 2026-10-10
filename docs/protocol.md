# Gluj-Bench worker protocol

`gluj-bench-worker --stdio` reads compact JSON requests from standard input and writes newline-delimited JSON envelopes to standard output. Diagnostics use standard error exclusively.

Protocol version 2 requests contain a non-empty request ID:

```json
{"protocol":2,"id":"request-1","command":"devices"}
```

Supported commands are `devices`, `benchmarks`, `run`, and `cancel`. A run includes its benchmark configuration:

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

GPU performance IDs include the register-resident vector workloads `gpu.performance.fp16`, `gpu.performance.fp32`, and `gpu.performance.fp64`. These execute through raw Vulkan using embedded SPIR-V; FP16 uses explicit `f16vec2` packed-pair operands. `gpu.performance.fp32.scaling` is a separate memory-backed profile that sweeps nominal power-of-two targets from 256 KiB to at most 512 MiB; metric names record the exact workgroup-aligned byte count. It returns paired `working_set_<bytes>.compute` and `working_set_<bytes>.bandwidth` metrics for each tier, plus `bandwidth_transition_working_set` when two consecutive low-noise tiers fall below 90% of the best of the first three tiers. Its `compute_scaling_points` metadata records `working-set bytes:operations/s:bytes/s:retained ratio` tuples for structured consumers.

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

Metadata records `gpu_timing_method` (`gpu_timestamp_and_host_elapsed_per_pass_v1`), `gpu_timing_definition`, `gpu_timing_batch_passes` (`dataset bytes:passes per sample` tuples), and `gpu_timing_samples_paired` (`true`). Batch lengths may vary across datasets. These are execution/end-to-end durations, not separate memory-stall durations. The old `.gpu_busy_estimate` / `.gpu_wait_estimate` metrics and `gpu_busy_wait_*` metadata are no longer generated. The UI hides retired estimates from older records and requires a rerun for timing graphs. Throughput/reference comparisons remain performance ratios. CPU profiles are unchanged.


Metadata records `ram_offload_percent`, `ram_budget_percent`, `host_allocation_budget_bytes`, `host_allocated_test_buffer_bytes`, `device_allocated_test_buffer_bytes`, `host_memory_placement`, `offload_profile_revision`, and `offload_tier_bytes` (`total:host:local` tuples). `result_validation` confirms first/last output checks for each region at the largest tier. Explicit allocation does not simulate automatic VRAM overflow or page migration. Qualitative tuning guidance recommends exploring a lower GPU core-frequency limit or reducing application batch size/concurrent GPU jobs, then retesting. It keeps the RAM offload share fixed for frequency comparisons and does not generate a numerical clock-reduction estimate. GPU activity pacing is excluded from TOPS, so application-level completion time must be checked for workload reductions. Protocol version remains 2.

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
