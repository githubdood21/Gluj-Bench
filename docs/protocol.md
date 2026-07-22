# Gluj-Bench worker protocol

`gluj-bench-worker --stdio` reads compact JSON requests from standard input and writes newline-delimited JSON envelopes to standard output. Diagnostics use standard error exclusively.

Protocol version 2 requests contain a non-empty request ID:

```json
{"protocol":2,"id":"request-1","command":"devices"}
```

Supported commands are `devices`, `benchmarks`, `run`, and `cancel`. A run includes its benchmark configuration:

```json
{"protocol":2,"id":"run-1","command":"run","arguments":{"benchmark_id":"cpu.bandwidth.cache.l1","target_duration_ms":5000,"samples":5}}
```

For controlled SMT comparisons, `run` accepts a string-valued `options` object. The default is `physical_cores`; `logical_processors` pins one worker to every discovered logical processor while preserving the same aggregate cache-instance working-set size:

```json
{"protocol":2,"id":"run-smt","command":"run","arguments":{"benchmark_id":"cpu.bandwidth.cache.l3","target_duration_ms":5000,"samples":5,"options":{"thread_mode":"logical_processors"}}}
```

GPU runs use the same string-valued options object to select an adapter returned by `devices`:

```json
{"protocol":2,"id":"run-gpu","command":"run","arguments":{"benchmark_id":"gpu.bandwidth.vram","target_duration_ms":5000,"samples":5,"options":{"device_id":"gpu:wgpu:0"}}}
```

GPU bandwidth IDs are `gpu.bandwidth.cache`, `gpu.bandwidth.vram`, and `gpu.bandwidth.host_link`. Cache metric names use `estimated_effective_l2.*` and `estimated_effective_l3.*` only when stable empirical tiers are detected; result metadata always marks those identities as inferred.

GPU compute IDs include the vector-shader workloads `gpu.performance.fp32`, `gpu.performance.fp16`, `gpu.performance.fp64`, `gpu.performance.int32`, and `gpu.performance.int8_packed`, plus `gpu.performance.matrix.fp16` and `gpu.performance.matrix.int8`. Vector and matrix scores are labeled separately. FP16 cooperative-matrix execution requires an exact, capability-reported 16x16x16 FP16-input/FP32-accumulator configuration. INT8 matrix remains visible but disabled when the portable shader frontend cannot safely express the driver-reported 8-bit operands. Results use `operations/s`; vector FMA and matrix multiply-accumulate both count multiplication and addition as separate operations.

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
