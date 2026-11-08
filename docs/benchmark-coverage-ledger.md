# Benchmark coverage and baseline ledger

Reconciled 2026-11-02 from the current executable catalogue and provider dispatch.
This ledger preserves implementation distinctions inside the 15 user-facing families.
Workload definitions and validation scopes are specified in phase methodology documents.

| Descriptor | Family / destination | Runner | Count and timing scope | Capability / distinct mode | Locally advertised |
| --- | --- | --- | --- | --- | --- |
| `cpu.bandwidth.cache.l1` | CPU memory bandwidth | CpuBandwidthProvider::run_cache / run_memory -> kernels | payload read/write/copy bytes; CPU active time (pacing excluded) | SIMD selected by CPU features; topology-owned buffers | Yes |
| `cpu.bandwidth.cache.l2` | CPU memory bandwidth | CpuBandwidthProvider::run_cache / run_memory -> kernels | payload read/write/copy bytes; CPU active time (pacing excluded) | SIMD selected by CPU features; topology-owned buffers | Yes |
| `cpu.bandwidth.cache.l3` | CPU memory bandwidth | CpuBandwidthProvider::run_cache / run_memory -> kernels | payload read/write/copy bytes; CPU active time (pacing excluded) | SIMD selected by CPU features; topology-owned buffers | Yes |
| `cpu.bandwidth.memory` | CPU memory bandwidth | CpuBandwidthProvider::run_cache / run_memory -> kernels | payload read/write/copy bytes; CPU active time (pacing excluded) | SIMD selected by CPU features; topology-owned buffers | Yes |
| `cpu.latency.memory.localized` | CPU dependent-read latency | cache_latency::run / latency::run -> Chain::traverse | dependent loads; ns/load; pinned-core elapsed | cache-instance masks or explicit RAM/locality conditions | Yes |
| `cpu.latency.memory` | CPU dependent-read latency | cache_latency::run / latency::run -> Chain::traverse | dependent loads; ns/load; pinned-core elapsed | cache-instance masks or explicit RAM/locality conditions | Yes |
| `cpu.latency.cache.l1` | CPU dependent-read latency | cache_latency::run / latency::run -> Chain::traverse | dependent loads; ns/load; pinned-core elapsed | cache-instance masks or explicit RAM/locality conditions | Yes |
| `cpu.latency.cache.l2` | CPU dependent-read latency | cache_latency::run / latency::run -> Chain::traverse | dependent loads; ns/load; pinned-core elapsed | cache-instance masks or explicit RAM/locality conditions | Yes |
| `cpu.latency.cache.l3` | CPU dependent-read latency | cache_latency::run / latency::run -> Chain::traverse | dependent loads; ns/load; pinned-core elapsed | cache-instance masks or explicit RAM/locality conditions | Yes |
| `cpu.performance.float.f32` | CPU FP32 | compute::run / cpu_workloads buffered runner -> kernels | scalar multiply and add; CPU active time (pacing excluded) | scalar/SIMD distinction retained; hardware gates for AVX2/FMA, POPCNT, AES | Yes |
| `cpu.performance.float.f64` | CPU FP64 | compute::run / cpu_workloads buffered runner -> kernels | scalar multiply and add; CPU active time (pacing excluded) | scalar/SIMD distinction retained; hardware gates for AVX2/FMA, POPCNT, AES | Yes |
| `cpu.performance.prime.sieve` | CPU prime sieve | cpu_workloads sieve runner | primes found per complete bounded sieve; reset/mark/count active time | limit and worker count define work | Yes |
| `cpu.performance.compression.deflate` | CPU DEFLATE | cpu_workloads codec runner -> miniz_oxide | uncompressed payload bytes; separately recorded compressed bytes/ratio | independent blocks; level/corpus/seed controls | Yes |
| `cpu.performance.decompression.deflate` | CPU DEFLATE | cpu_workloads codec runner -> miniz_oxide | uncompressed payload bytes; separately recorded compressed bytes/ratio | independent blocks; level/corpus/seed controls | Yes |
| `cpu.performance.extended.popcnt` | CPU population count | compute::run / cpu_workloads buffered runner -> kernels | one 64-bit population-count instruction; CPU active time (pacing excluded) | scalar/SIMD distinction retained; hardware gates for AVX2/FMA, POPCNT, AES | Yes |
| `cpu.performance.extended.aes` | CPU AES rounds | compute::run / cpu_workloads buffered runner -> kernels | one hardware AES round instruction; CPU active time (pacing excluded) | scalar/SIMD distinction retained; hardware gates for AVX2/FMA, POPCNT, AES | Yes |
| `cpu.performance.avx2.f64_fma` | CPU FP64 | compute::run / cpu_workloads buffered runner -> kernels | one floating-point add or multiply (FMA counts as two); CPU active time (pacing excluded) | scalar/SIMD distinction retained; hardware gates for AVX2/FMA, POPCNT, AES | Yes |
| `cpu.performance.single_thread.integer.i64` | CPU INT64 (legacy alias) | CpuBandwidthProvider::run -> canonical integer.i64, one core | same canonical register work; original alias retained in provenance | retired visible duplicate; compatibility dispatch retained | Yes |
| `cpu.performance.string.ascii` | CPU ASCII scan | ascii_scaling / cpu_workloads | 64-byte record scans; validated character counts | AVX2 or scalar fallback; content/access controls | Yes |
| `cpu.performance.avx2.f32_fma.scaling` | CPU FP32 | compute::run / cpu_workloads buffered runner -> kernels | one FP32 multiply or add; FMA counts as two per lane; CPU active time (pacing excluded) | scalar/SIMD distinction retained; hardware gates for AVX2/FMA, POPCNT, AES | Yes |
| `cpu.performance.matrix.fp32.scaling` | CPU FP32 matrix | compute_scaling -> matrix_scaling / cpu_workloads explicit shapes | 2*M*N*K per full product; blocked effective traffic | AVX2/FMA; explicit global geometry vs legacy per-worker sizes | Yes |
| `cpu.performance.integer.i64` | CPU INT64 | compute::run / cpu_workloads buffered runner -> kernels | one 64-bit integer multiply or add; CPU active time (pacing excluded) | scalar/SIMD distinction retained; hardware gates for AVX2/FMA, POPCNT, AES | Yes |
| `gpu.bandwidth.cache` | GPU local bandwidth | memory_conditions / legacy discovery -> vulkan_bandwidth | payload + separate accounted traffic; GPU timestamps | explicit cache/stream shader vs native copy identities; budgets/device limits | Yes |
| `gpu.bandwidth.vram` | GPU local bandwidth | memory_conditions / legacy discovery -> vulkan_bandwidth | payload + separate accounted traffic; GPU timestamps | explicit cache/stream shader vs native copy identities; budgets/device limits | Yes |
| `gpu.bandwidth.host_link` | GPU host transfers | memory_conditions / legacy discovery -> vulkan_bandwidth | directional payload bytes; host copy elapsed | explicit cache/stream shader vs native copy identities; budgets/device limits | Yes |
| `gpu.performance.fp16` | GPU vector | compute::run -> vulkan_vector / vulkan_compute_profile | precision-specific FMA operations; timestamp service or offered completion/window | FP16 arithmetic/storage, FP64 arithmetic; FP32 host placement/demand controls | Yes |
| `gpu.performance.fp64` | GPU vector | compute::run -> vulkan_vector / vulkan_compute_profile | precision-specific FMA operations; timestamp service or offered completion/window | FP16 arithmetic/storage, FP64 arithmetic; FP32 host placement/demand controls | Yes |
| `gpu.performance.fp32.scaling` | GPU vector | compute::run -> vulkan_vector / vulkan_compute_profile | precision-specific FMA operations; timestamp service or offered completion/window | FP16 arithmetic/storage, FP64 arithmetic; FP32 host placement/demand controls | Yes |
| `gpu.performance.fp32.offload.scaling` | GPU vector | compute::run -> vulkan_vector / vulkan_compute_profile | precision-specific FMA operations; timestamp service or offered completion/window | FP16 arithmetic/storage, FP64 arithmetic; FP32 host placement/demand controls | Yes |
| `gpu.performance.matrix.fp16.scaling` | GPU matrix | cooperative_matrix::run -> register or MatrixProfileHarness | 2*M*N*K per native MMA, tile visits/reuse/padding; GPU timestamps | KHR native shape/type/subgroup capability; sparse/FP8 unavailable | Yes |
| `gpu.performance.matrix.int8.scaling` | GPU matrix | cooperative_matrix::run -> register or MatrixProfileHarness | 2*M*N*K per native MMA, tile visits/reuse/padding; GPU timestamps | KHR native shape/type/subgroup capability; sparse/FP8 unavailable | Yes |
| `gpu.performance.matrix.fp8` | GPU matrix | cooperative_matrix::run -> register or MatrixProfileHarness | 2*M*N*K per native MMA, tile visits/reuse/padding; GPU timestamps | KHR native shape/type/subgroup capability; sparse/FP8 unavailable | Unavailable |
| `gpu.performance.matrix.sparse.fp16` | GPU matrix | cooperative_matrix::run -> register or MatrixProfileHarness | 2*M*N*K per native MMA, tile visits/reuse/padding; GPU timestamps | KHR native shape/type/subgroup capability; sparse/FP8 unavailable | Unavailable |
| `gpu.performance.matrix.sparse.int8` | GPU matrix | cooperative_matrix::run -> register or MatrixProfileHarness | 2*M*N*K per native MMA, tile visits/reuse/padding; GPU timestamps | KHR native shape/type/subgroup capability; sparse/FP8 unavailable | Unavailable |
| `gpu.performance.matrix.sparse.fp8` | GPU matrix | cooperative_matrix::run -> register or MatrixProfileHarness | 2*M*N*K per native MMA, tile visits/reuse/padding; GPU timestamps | KHR native shape/type/subgroup capability; sparse/FP8 unavailable | Unavailable |

## Baseline and migration evidence

- Existing uncommitted work was preserved through all phases; no reset or automatic history rewrite was performed.
- The source audit records the original catalogue consolidations. Phase 1 began from a passing baseline; no unresolved pre-existing compile/test failure was carried into qualification.
- Captured/anonymized version-1 fixtures cover scalar, SIMD, old single-thread INT64, RAM/cache latency and GPU host offload. Settings missing new fields restore additive defaults.
- An original version-1 preview PNG fixture retains the old all-results payload; the current importer reads it unchanged. New exports use selected scope by default.
- Fixtures live under `crates/gluj-bench-ui/tests/fixtures/`; original run files and user settings remain preserved.
- Cross-revision comparisons are gated; fixtures prove compatibility of reading, not equivalence of old/new workloads.

## Qualification hardware and limitations

Local hardware: AMD Ryzen 9 5950X (16 physical cores, two L3 instances), Radeon RX 7900 XTX, Vulkan 1.4.359, driver raw version 8389024; 40 GiB installed RAM at 2800 MT/s.
Local output, sampling, cancellation and UI checks are recorded in [Phase 9 qualification](phase-9-qualification.md). Intel/hybrid CPUs, integrated/shared-memory GPUs, NVIDIA and other Vulkan drivers remain untested hardware paths.
Sparse/FP8 rows have no verified runner and remain outside the runnable family catalogue. No conventional global GPU GEMM is claimed.
