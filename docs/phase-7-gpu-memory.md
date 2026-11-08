# Phase 7 - GPU memory, transfers and cooperative matrices

Implemented in the working tree on 2026-11-01. Automated workspace tests and shader
checks pass. Live GPU output and interactive UI verification are pending the user's
[confirmation checklist](phase-7-verification.md). Cross-vendor qualification remains
Phase 9 work. Transport stays version 2; workload revisions distinguish changed work.

## Local memory and cache discovery

Cache-oriented and streaming kernels retain separate identities. `dataset_mode`
accepts `automatic`, `single`, or `sweep`; explicit `dataset_bytes` must fit budgets
and device limits, with a 64 KiB minimum. Sizes align down to 1 KiB, never upward;
sweeps cover powers of two from 64 KiB and include the aligned endpoint. The UI
accepts whole MiB. Invalid oversized requests fail rather than silently shrinking.

`memory_operation` selects `all`, `read`, `write`, or `copy`. `memory_copy_path`
selects `shader` or `native`. Native copies require sequential access; shader
copies retain their workload flavor. Copy summaries show useful payload and
separately identified accounted read/write traffic. Results retain each measured
size and operation, actual allocation, kernel flavor and selected path. Different
kernels/paths must not be interpreted as a single same-kernel scaling curve.

Customized runs default to sequential access. Shader access optionally selects
`shuffled_blocks` (seeded 64-vector/1 KiB blocks), `rotating_regions`, or
`lane_scattered` (a bijection of lanes). Seeds, fixed/fresh policy and one/three
realizations are recorded. Permutations cover the full working set, including
non-power-of-two sizes; bounded multiplication and overflow-safe modular addition
preserve shader indexing. Each realization receives full policy coverage.
Curated plans separately measure sequential and shuffled conditions with equal
three-realization coverage; randomness is a user setting, not the custom default.
These patterns alter locality; they do not guarantee hardware cache eviction or
measure cache-miss counters.

Automatic discovery clears prior calibration and probes again for each run.
Records include device UUID, driver, timestamp, sample/time floors and heuristic
revision. Inferred cache transitions retain adjacent tested-size intervals and
heuristic drop strength, which is not a statistical confidence probability or a
physical cache-capacity guarantee. A healthy probe with no detected tier produces
a bounded cache-oriented measurement with an explicit no-tier finding. Failed
discovery reports its limitation and preserves available probe evidence.

## Host transfers

`transfer_direction` accepts `both`, `host_to_device` or `device_to_host`; both
means separately measured directional conditions, not simultaneous duplex work.
`staging_buffers` accepts 1/3/8. `transfer_submission` accepts `serial` or `batched`;
`transfer_batch` accepts 1/8/32, with serial requiring 1. Size sweeps use the same
explicit sizing rules above. Automatic transfer sizing is bounded at 64 MiB.

Staging allocation and initialization precede measurement. Wall-clock timing
includes copy submission, completion waits and required host visibility. Mean
request cost is measured batch wall time divided by actual completed copies;
it is not an independently timed request-latency distribution. Raw observations
retain throughput and mean cost; summaries retain size and direction. Discrete
GPUs use non-device-local host staging heaps rather than BAR/VRAM staging.
Noncoherent host allocations receive required flush/invalidate handling.

VRAM and RAM percentage budgets account for the GPU buffer and the entire host
ring separately; shared-memory devices also have a conservative available-memory
cap. Reported test-buffer peaks describe allocated buffers, not process RSS.
Partial setup, allocation failures and cancellation release owned Vulkan objects.
Copy and shader write hazards are ordered; see the Vulkan
[synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html)
and [memory property definitions](https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryPropertyFlagBits.html).

## Cooperative matrix conditions

Buffered matrix modes expose only device-supported native tile tokens, including
input and accumulation types: for example `16x16x16:fp16:fp32` or
`16x16x16:sint8:sint32`. FP16 requires FP32 accumulation; unsupported or saturating
variants are not offered. `matrix_tile_shape=automatic` retains the preferred native
tile. `matrix_tile_reuse` accepts 1-64. `matrix_traversal` selects `sequential`,
`shuffled_tiles` or `rotating_tiles`, with recorded seed/ensemble controls.
Register-only modes reject buffer traversal settings and retain their definition.

Traversal permutes complete read-only operand tiles while subgroup output ownership
stays unique. Reuse and padded tile visits remain counted; every realization has
independent scalar output reference checks. Records include actual dimensions,
input signedness, accumulation type, traversal and revision. This is dense
independent tile operand streaming, not conventional global GEMM, and must not be
presented as equivalent to CPU full-matrix multiplication. Supported shapes are
queried using Vulkan cooperative-matrix capabilities; qualification on each actual
GPU/driver is still required.

## Sampling, sustained runs and shader reproduction

Memory, transfer and buffered matrix conditions use the shared Extensive/Shorter
policy, including warmup, full realization coverage and descriptive spread.
`sustained_seconds` accepts 0-120 and extends applicable active measurement floors;
cancellation preserves available partial observations. Variability is reported in
the numbers, with no generated recommendation to rerun.

Nine changed bandwidth/matrix shaders were regenerated with glslang 16.6.0 and
SPIR-V validation. The checked-in `shaders/memory-matrix-build.json` records source
and binary SHA256 hashes, compiler version and defines; results embed the manifest.
From the repository root, using the retained local compiler or `-Compiler <path>`:

```powershell
.\scripts\compile-memory-shaders.ps1
.\scripts\compile-memory-shaders.ps1 -Check
```

Normal application builds use checked-in SPIR-V and do not require glslang.

Phase 9 qualification corrected buffered INT8 initialization to A=-1 (signed) or
255 (unsigned), with B=1, matching the independent reference. The input pattern is
recorded and matrix_profile_revision is tile-streaming-4-signed-input. The three
INT8 traversal conditions passed afterward; FP16 behavior and register modes retain
their definitions. See [qualification](phase-9-qualification.md).
