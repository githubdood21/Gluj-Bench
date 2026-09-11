# Gluj-Bench worker protocol

`gluj-bench-worker --stdio` reads compact JSON requests from standard input and writes newline-delimited JSON envelopes to standard output. Diagnostics use standard error exclusively.

Protocol version 2 requests contain a non-empty request ID:

```json
{"protocol":2,"id":"request-1","command":"devices"}
```

Supported commands are `devices`, `benchmarks`, `run`, and `cancel`. A run includes its benchmark configuration:

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

GPU performance IDs include the vector-shader workloads `gpu.performance.fp16`, `gpu.performance.fp32`, and `gpu.performance.fp64`. These execute through raw Vulkan using embedded SPIR-V; FP16 uses explicit `f16vec2` packed-pair operands. Matrix families include dense FP16, INT8, and FP8 plus their `gpu.performance.matrix.sparse.*` variants. Dense FP16 and signed or unsigned INT8 execute through raw Vulkan `VK_KHR_cooperative_matrix` pipelines when the driver advertises a compatible MxNxK/type configuration. FP8 and structured-sparse rows remain visible but disabled unless a capability-verified implementation exists; dense KHR support is never treated as evidence of sparse acceleration. Results use `operations/s`; vector FMA and matrix multiply-accumulate both count multiplication and addition as separate operations.

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
