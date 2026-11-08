# CPU and GPU benchmark catalogue audit

Date: 2026-10-31

Scope: the current working tree, including its existing uncommitted implementation
changes. This is a source audit of registration, dispatch, kernels, configuration,
sampling, result metadata, and UI options. No benchmarks, hardware measurements,
release disassembly checks, or runtime tests were executed for this audit.

The recommendations below are the source-audit proposals. The user subsequently
confirmed the family-card direction and full staged scope; see the
[implementation plan](implementation-plan.md) for the resulting delivery phases
and remaining engineering gates. Only documentation is changed by this audit.

## Main findings

1. The original separate CPU/GPU FP32 register and scaling catalogue entries have
   already been consolidated. CPU INT64 and ASCII and GPU FP16/INT8 matrix compute
   also have unified entries. Their older internal kernels are still needed as
   modes or reference implementations; their existence is not another visible test.
2. One clear remaining same-kernel catalogue redundancy is single-thread INT64.
   The separate single-thread entry and the unified INT64 register mode both call
   `integer_chunk`; the unified runner already supports one selected core. Their
   harnesses differ, so preserve historical result provenance when consolidating.
3. Several other entries are configuration variants that can share a family card:
   cache levels, RAM latency locality, floating-point execution paths, precision,
   compression direction, and GPU FP32 memory placement. Preserve their distinct
   measurements instead of deleting useful coverage.
4. The largest upgrade gaps are legacy CPU arithmetic/instruction kernels, fixed
   codec and sieve inputs, fixed-size memory tests, and register-only GPU FP16/FP64.
5. New sampling and access-pattern options should attach to families and supported
   modes, not produce separate randomized/scaling/sustained catalogue entries.

## Current inventory

The provider catalogue contains 35 descriptor entries: 22 CPU/memory and 13 GPU.
Four GPU entries have no executable implementation (FP8 dense and three sparse
formats), leaving 31 implemented descriptors whose actual availability depends on
hardware, memory, and driver support. A descriptor is not a count of distinct
kernels or all the configurations a test can execute.

### CPU and memory: 22 entries

| Current entry / ID | Actual measurement | Audit decision |
| --- | --- | --- |
| L1/L2/L3 bandwidth: `cpu.bandwidth.cache.l1`, `.l2`, `.l3` (3 entries) | Parallel temporal read/write/copy at topology-sized hot working sets | One CPU memory-bandwidth family, with cache-level presets; retain three measured levels |
| System RAM bandwidth: `cpu.bandwidth.memory` | Large streaming buffers; temporal reads, non-temporal writes/copy; cache preparation outside timing | Same family as cache bandwidth, with explicit access/cache-state policy and consistent byte labels |
| L1/L2/L3 latency: `cpu.latency.cache.l1`, `.l2`, `.l3` (3 entries) | One pinned core; warmed dependent pointer chain; target-level sizing | One CPU dependent-read-latency family, retaining level-specific measurements |
| RAM scattered latency: `cpu.latency.memory` | Dependent chain randomized over the RAM working set | RAM target and scattered-locality mode in the latency family |
| RAM localized latency: `cpu.latency.memory.localized` | Dependent chain randomized within 64 KiB blocks, blocks visited sequentially | Locality mode in the same family; not equivalent to scattered RAM latency |
| INT64: `cpu.performance.integer.i64` | Unified register and buffered wrapping multiply/add; single size or sweep | Keep and absorb the separate single-thread entry |
| Single-thread INT64: `cpu.performance.single_thread.integer.i64` | Same `integer_chunk` as the unified register mode; separate one-thread harness | Retire visible duplicate after making one-core register configuration explicit |
| FP32 scalar: `cpu.performance.float.f32` | Eight source-level scalar multiply/add chains | Execution-path option in the FP32 arithmetic family; preserve a separate result identity |
| FP64 scalar: `cpu.performance.float.f64` | Same scalar workload structure at FP64 | Execution-path option in the FP64 arithmetic family |
| FP32 vector: `cpu.performance.avx2.f32_fma.scaling` | Unified register AVX2/FMA and buffered vector workload with single sizes/sweeps | Keep as FP32 arithmetic family; add the scalar option without equating the kernels |
| FP64 AVX2/FMA: `cpu.performance.avx2.f64_fma` | Register-only AVX2 FP64 FMA | Upgrade into FP64 arithmetic family; add buffered/sweep modes when implemented |
| FP32 matrix: `cpu.performance.matrix.fp32.scaling` | Blocked AVX2 matrix multiplication; 32 rows, K=N, private matrices per worker | Keep separate from vector arithmetic; improve shapes/reuse controls |
| ASCII: `cpu.performance.string.ascii` | Unified character-count scan; cache/RAM/custom sizes and sweeps | Keep separate from raw bandwidth; add input and traversal options |
| Sieve: `cpu.performance.prime.sieve` | Reused byte marks for a fixed search through 1,000,000 | Keep algorithm; upgrade limits, sizing, validation, and output units |
| Compression: `cpu.performance.compression.deflate` | Reusable DEFLATE level-6 compressor; fixed patterned input | One DEFLATE family, compression direction |
| Decompression: `cpu.performance.decompression.deflate` | Reusable decoder on a prepared compressed version of patterned input | Same family, decompression direction; independent throughput and validation |
| POPCNT: `cpu.performance.extended.popcnt` | Counted population-count operations plus rotations, multiply/add and accumulation | Keep as instruction microbenchmark; audit emitted instructions and add data-backed mode |
| AES: `cpu.performance.extended.aes` | Eight independent AESENC round chains | Keep as explicitly named AES-round microbenchmark; do not label as complete encryption |

### GPU: 13 entries

| Current entry / ID | Actual measurement | Audit decision |
| --- | --- | --- |
| Effective cache: `gpu.bandwidth.cache` | Inferred tiers from a read sweep, then read/write/copy with cache-oriented kernels | One GPU local-memory-bandwidth family with cache-oriented mode |
| GPU memory: `gpu.bandwidth.vram` | Streaming read/write and native Vulkan buffer-copy measurements beyond inferred cache where possible | Same family with streaming mode; keep path distinctions explicit |
| Host link: `gpu.bandwidth.host_link` | CPU/RAM-to-GPU and GPU-to-CPU staging-ring buffer copies, timed end to end | Keep separate transfer family; add size/queue controls |
| FP16 vector: `gpu.performance.fp16` | Register-resident FP16 vector FMA, currently no buffered dataset/sweep controls | GPU vector-compute family, FP16 precision; upgrade capability-aware buffered path |
| FP64 vector: `gpu.performance.fp64` | Register-resident FP64 vector FMA | Same vector family, FP64 precision; upgrade buffered path |
| FP32 vector: `gpu.performance.fp32.scaling` | Unified register and buffered FP32 vector compute | Same vector family, FP32 precision |
| FP32 offload: `gpu.performance.fp32.offload.scaling` | Same profile shader/arithmetic with 50/75/100% host-memory placement; maximum or offered-load execution | Host-memory placement and demand modes of FP32 vector; preserve execution and timing definitions |
| FP16 matrix: `gpu.performance.matrix.fp16.scaling` | Unified register and memory-backed cooperative FP16 matrix operations, FP32 accumulation | One GPU matrix-compute family, FP16/FP32 mode |
| INT8 matrix: `gpu.performance.matrix.int8.scaling` | Same profile framework with driver-supported signed or unsigned INT8 and 32-bit accumulation | Same matrix family, explicit numeric/accumulator mode; not equivalent performance |
| Dense FP8: `gpu.performance.matrix.fp8` | No implemented supported runner | Move to capability/implementation information rather than normal runnable-test list |
| Sparse FP16/INT8/FP8: `gpu.performance.matrix.sparse.fp16`, `.int8`, `.fp8` (3 entries) | No verified structured-sparse runner | Same capability/implementation treatment; do not count as executable tests |

## Consolidation decisions

### A. Retire a duplicate: single-thread INT64

`compute.rs::run` dispatches ordinary arithmetic to `run_parallel`, but preserves a
separate single-thread branch. Both its single-thread INT64 case and the unified
register INT64 mode use `integer_chunk`. Setting the unified mode to register
arithmetic with one physical core covers the same kernel and target.

Use one configurable harness for one core, selected core count, and core-count
sweeps. Fix warmup and timing policy differences as part of the migration, rather
than assuming historical results from the two harnesses are interchangeable.
Preserve the old single-thread result ID as import/history provenance or a
versioned migration record. Do not enqueue both variants for identical one-core
register settings.

### B. Group scalar and SIMD arithmetic without erasing the difference

CPU FP32 and FP64 source-level scalar kernels have separate multiply/add
expressions, while AVX2 kernels explicitly execute vector FMA. These are not
identical kernels. Scalar results are useful for an execution-path comparison or
hardware without the SIMD capability.

Offer Scalar and AVX2/FMA as execution paths under FP32 and FP64 arithmetic. Keep
instruction path, FMA operation definition, precision, threading and kernel
revision in result identity. Verify release code generation before promising an
exact scalar instruction mix; Rust source shape alone does not establish it.

Do not require a buffered or cache option for a register-only scalar mode until a
matching implementation exists. CPU matrix compute remains its own family.

### C. Group cache/RAM bandwidth and group latency independently

The three CPU cache-bandwidth entries already share `run_cache`; their main
differences are topology and size. RAM shares the measurement infrastructure but
uses different cache preparation and store policies. One family can offer L1,
L2, L3, RAM, custom size, and sweep presets while retaining those policies.

Cache and RAM copy accounting currently differs: CPU cache copy reports read plus
write traffic, while CPU RAM copy reports useful payload. GPU copy reports read
plus write traffic. Standardize the primary label/accounting or expose clearly
named payload and traffic rates. Historical comparisons require an accounting
revision; doubling an old displayed value without provenance is not acceptable.

The five CPU latency entries share the `Chain` implementation, with different
sizing, locality, warmup and batching. One latency family should preserve target
and locality as explicit dimensions. Keep warmed cache defaults and RAM's
beyond-LLC sizing guards. A custom size must not be silently called L2 or RAM
without the corresponding evidence.

Latency is dependent single-chain access. Bandwidth exposes parallel memory
service. Neither replaces the other or the ASCII scan.

### D. Fold FP32 offload into vector-compute placement

The FP32 profile uses the same shader and arithmetic for VRAM-only and host-memory
regions. This is a strong UI/configuration consolidation opportunity: placement
becomes VRAM-only or mixed system RAM/VRAM with the existing supported fractions.
Do not add arbitrary fractions until alignment, placement, validation and budgets
support them.

Preserve maximum-throughput and offered-load modes. Offered-load measurements
include scheduled arrivals, incomplete requests and arrival-to-completion delay;
they cannot be compared directly as active peak throughput. Direct GPU reads and
writes of host memory are not staging copies, so host-link bandwidth stays separate.

The FP16/FP64 register tests can join the same vector family as precision options.
Only FP32 initially supports buffered placement and demand controls. UI controls
must reflect that support instead of silently using FP32 or pretending FP16/FP64
already implement those paths.

### E. Group GPU matrix formats and codec directions

FP16 and INT8 matrix tests already share a harness framework but exercise different
numeric formats and possibly different hardware execution rates. Group their
configuration, not their scores. Record actual signedness, accumulator type,
cooperative tile shape and reuse policy.

Compression and decompression similarly share a DEFLATE family and a corpus, but
must retain separate rates. An optional run-both workflow can prepare the corpus
once outside timing and then measure both directions. It must not substitute
compression speed for decompression performance or combine them into one number.

## Older-test upgrades

### Highest priority: sampling and validity across every family

- UI requests are hard-coded to five samples and a 2,000 ms target. CPU validation
  currently limits samples to 3..20 and duration to 150..30,000 ms. GPU references
  clamp sample count to 3..5 and shorten duration. These are not the proposed
  extensive bounded sampling policy; upgrade their shared configuration first.
- Keep active-work time, end-to-end time, setup/validation time and request-window
  time separate. The existing `elapsed_ns` field has differing scope across paths.
- Store chronological raw samples and realization/block IDs. Current public
  `SampleStatistics` keeps summaries rather than those distributions.
- References and discovery probes need the policy too. GPU cache discovery uses
  three 100 ms samples per size and caches its findings for the provider lifetime.
  Make provenance/freshness explicit and perform required automatic checks within
  the run rather than issuing a rerun recommendation.
- Preserve observed variability; do not discard low samples or repeatedly sample
  until a preferred stability condition occurs. Finish at declared budget limits.
- Verify outputs outside timing and report validation coverage. A checksum is not
  interchangeable with a scalar/reference correctness check.

### Legacy arithmetic and instruction microbenchmarks

Audit release machine code and operation accounting before extending duration.
The register INT64 kernel has long affine recurrences, whereas the newer buffered
INT64 kernel explicitly uses `black_box` inside recurrence rounds to prevent
simplification. This difference warrants verification; it is not proof of a
miscompiled current benchmark.

POPCNT has a particularly concrete source-level issue to investigate: each loop
counts the bits in a value and three rotations of that same value. Rotation
preserves population count. These four mathematical results are identical and
could permit compiler reuse, while the counter reports four POPCNT operations.
Check emitted instructions and replace with genuinely independent operands or a
validated buffer-counting kernel as appropriate. Existing timings have not been
shown by this source audit to be wrong.

Preserve AES-round throughput as a microbenchmark. A useful data-backed upgrade is
full AES encryption with an explicitly named mode, key size, key-setup policy and
reference vectors. Such results need their own workload revision and bytes/s;
AESENC rounds/s are not equivalent to encrypted bytes/s.

### DEFLATE

Currently, the main parallel path uses 2 MiB per worker, with a 128 KiB probe and a
single deterministic patterned input at level 6. Both sizes can fit in a larger
shared cache; calling them cache versus RAM is not established by size alone.
Changing size also changes codec behavior, so the slowdown is not a pure memory
bandwidth diagnosis.

Add total dataset and per-block size controls, compression level, seeded corpus
selection, and compressible/mixed/incompressible content. Report compression ratio
alongside input/output throughput. Keep corpus and block order definitions
separate: shuffling bytes of a compressed stream would invalidate it. Random block
order requires independently decodable blocks with defined state resets.

### Prime sieve

The current fixed one-million-limit byte sieve always uses the same algorithm and
mark footprint. Add a search-limit/footprint preset, complete-sieve time or
sieves/s, and validated prime counts. Retain primes/s only with the range recorded;
the work per discovered prime changes with range.

`run_parallel` starts `memory_ratio` at 1 and only measures alternate sizes for
ASCII/codec workloads. The sieve therefore receives a compute-bound classification
without a size-sensitivity probe. Remove this unsupported classification or replace
it with descriptive measurements. Do not randomly reorder the sieve's dependency
structure merely to create misses. A future segmented sieve is a distinct algorithm
mode, with its segment size and revision recorded.

### CPU bandwidth and latency

RAM bandwidth intentionally ignores retired custom-size options; an existing
smoke test asserts this. Reintroducing custom/sweep behavior is an explicit backend
upgrade, not just enabling a UI control. Preserve allocation limits and meaningful
LLC separation when labeling RAM results.

Expose supported working-set, locality, seed ensemble, pinned-core and preparation
options through the unified families. Existing RAM latency already supplies global
and 64 KiB-local random chains. Extend that foundation rather than adding another
top-level "random RAM test".

### GPU vector, bandwidth and transfers

Extend the buffered vector harness to precision-specific FP16/FP64 shaders, buffer
types, operation counts and validation before exposing their size/sweep options.
Register-only capability must remain usable when a precision's storage feature is
unavailable. Precision selection must never imply unsupported matrix acceleration.

GPU cache versus streaming bandwidth is not just a different buffer size: the
flavors use different accesses per invocation; streaming copy dispatches a native
Vulkan buffer copy while cache copy uses a shader. Keep execution paths labeled;
do not draw one continuous same-kernel scaling curve across them. Add custom sizes,
size sweeps and access-pattern variants only with explicit path identity.

The host-link test currently chooses one safe size and a three-buffer staging ring.
Add a transfer-size sweep, transfer duration per request, and supported queue/ring
controls. Report H2D and D2H independently and offer concurrent bidirectional work
only as an explicit new measured mode. A random transfer order is an order policy,
not a simulated GPU random-read workload.

### CPU and GPU matrix workloads

CPU matrix is a blocked full product with 32 rows and K=N. GPU buffered matrix
streams cooperative tiles with fixed reuse and an accumulator; it is not the same
global GEMM implementation. Keep these definitions visible and avoid treating their
TOPS as identical application-level work.

Upgrade shape/reuse controls and sampled reference validation. CPU core-count
experiments must preserve the intended aggregate matrix work or state clearly when
per-worker shapes change. GPU tile-order randomness should preserve accumulation
dependencies and quantify floating-point reference tolerances.

### Metadata and maintenance debt

- CPU legacy descriptor metadata says aggregate logical processors, but the UI
  sends exact physical-core counts. CLI defaults vary between runner families.
  Unify default resolution and show actual placement, not a stale descriptor scope.
- `compute.rs::run` returns all non-single-thread workloads through `run_parallel`,
  yet its later match still contains old branches for those workloads. Remove dead
  routes after reference/test dependencies are checked. Upgrade the live parallel
  codec runner, not its unreachable allocating predecessor.
- Replace UI ID/suffix allowlists such as `configurable_compute` and `.scaling`
  checks with declarative supported options. Historical IDs can remain stable while
  the product presents family/mode names. A family rename must not break budgets,
  statistics, queueing, saved settings, sharing or comparisons.
- Hide unimplemented GPU formats from the normal test picker while showing truthful
  capability/implementation information. Do not report unavailable implementation
  as proof that the physical GPU lacks a feature.

## Where new randomness fits

| Family/mode | Appropriate extension | Preserve this boundary |
| --- | --- | --- |
| Register arithmetic or AES-round microkernel | Seeded operands, independent chains | No allocated memory footprint to characterize with cache-eviction controls |
| CPU/GPU buffered vector | Shuffled blocks, scattered reads, rotating regions | Record index overhead, actual coverage and GPU coalescing policy |
| CPU/GPU bandwidth | Separate sequential and irregular-access modes | Peak streaming bandwidth is still useful; irregular results need their own identity |
| Dependent-read latency | Seed ensemble, locality and turnover controls | More concurrent chains change the measurement and require an explicit mode |
| ASCII | Varied character distributions and shuffled records/blocks | Fixed bytes scanned and known expected hit counts |
| DEFLATE | Content entropy, independent-block order and active corpus turnover | A valid codec stream and recorded dictionary/reset policy |
| Matrix | Tile/region traversal and working-set turnover | Preserve useful multiplication and accumulation semantics |
| Sieve | Configurable limit/segmentation if implemented | Preserve sieve dependencies; randomizing everything is not a generic upgrade |
| Host transfers | Size, source/destination region and submission order | This measures transfers rather than shader memory-access locality |

## Proposed catalogue after consolidation

Eleven CPU families and four GPU families would represent the implemented coverage
above. This is a UI organization proposal, not a reduction to fifteen measurements.
Each family can execute multiple distinct recorded conditions.

CPU families:

1. Memory bandwidth (L1/L2/L3/RAM/custom; read/write/copy).
2. Dependent-read latency (cache/RAM targets and locality).
3. FP32 arithmetic (scalar/AVX2; supported register/buffer modes).
4. FP64 arithmetic (scalar/AVX2; buffered modes when implemented).
5. FP32 matrix compute.
6. INT64 arithmetic (including one-core and core-count profiles).
7. ASCII scan.
8. DEFLATE (compression/decompression).
9. Prime sieve.
10. Population count.
11. AES rounds (full encryption only as an explicitly implemented algorithm mode).

GPU families:

1. Local-memory bandwidth (cache-oriented/streaming; read/write/copy).
2. Host-device transfers (H2D/D2H).
3. Vector compute (FP16/FP32/FP64; supported placement/access/demand modes).
4. Matrix compute (FP16/INT8; explicit accumulator and tile definition).

Default suite plans should explicitly select supported conditions. "Run all"
currently enqueues one configured run per descriptor; it does not automatically
cover all modes. After family consolidation, a curated suite must not silently
lose all-core versus one-core, scalar versus SIMD, cache levels, both codec
directions, or supported GPU precisions. Custom exhaustive plans should show
condition count, memory needs and estimated time before starting.

Avoid the complete Cartesian product of all knobs by default. Extensive sampling
of a well-defined condition and broad coverage of distinct conditions are separate
requirements; both must be represented in the suite plan.

## Recommended implementation order and acceptance checks

1. Define family/condition metadata, compatible result identity and explicit suite
   plans. Preserve current IDs/history and per-test settings. Gate options by mode.
2. Remove the single-thread INT64 picker duplication using the shared one-core
   register mode and a versioned harness migration.
3. Introduce shared extensive bounded sampling, raw sample retention, validation,
   descriptive spread/drift output and no rerun recommendations across live paths.
4. Consolidate memory, precision, placement and codec controls, maintaining current
   runnable coverage and meaningful separate results.
5. Upgrade legacy operation-count validity, codec/sieve datasets and GPU FP16/FP64
   buffered support; then add appropriate seeded access-pattern conditions.

Required regression coverage:

- Catalogue family/condition uniqueness and correct mode availability.
- One-core INT64 executes once for an identical requested condition.
- Suite plans retain all intended measurement coverage after UI consolidation.
- Old settings and result imports remain readable; incompatible revisions are not
  silently compared.
- CLI/UI resolve identical explicit configurations the same way.
- Unsupported controls fail clearly rather than being ignored or changing precision.
- Operation counts agree with validated useful work and inspected release kernels.
- Automatic sampling is bounded, cancellable and preserves variable samples.
- Access generators replay exactly, respect bounds and validate output.
- No generated result text recommends rerunning because performance varies.

## Source anchors

- [CPU catalogue and dispatch](../crates/gluj-bench-cpu/src/lib.rs): `benchmarks`,
  `run`, `run_cache`, `run_memory`, `validate`.
- [Legacy CPU arithmetic and algorithms](../crates/gluj-bench-cpu/src/compute.rs):
  `DEFINITIONS`, `run`, `run_parallel`, `data_bytes`, `PrimeFinder`,
  `integer_chunk`, `popcnt_inner`, `aes_inner`.
- [Unified CPU compute](../crates/gluj-bench-cpu/src/compute_scaling.rs):
  register dispatch and buffered `measure`.
- [CPU latency](../crates/gluj-bench-cpu/src/latency.rs) and
  [cache latency](../crates/gluj-bench-cpu/src/cache_latency.rs): shared `Chain`,
  distinct layout and measurement policies.
- [GPU vector catalogue and execution](../crates/gluj-bench-gpu/src/compute.rs):
  `descriptors`, `run`, `run_fp32_scaling`.
- [GPU matrix catalogue and execution](../crates/gluj-bench-gpu/src/cooperative_matrix.rs):
  `descriptors`, `MatrixKind`, `run`, `run_matrix_scaling`.
- [GPU bandwidth](../crates/gluj-bench-gpu/src/lib.rs) and
  [Vulkan bandwidth](../crates/gluj-bench-gpu/src/vulkan_bandwidth.rs):
  discovery, cache/stream flavors and native copy paths.
- [UI configuration and suites](../crates/gluj-bench-ui/src/main.rs):
  `configurable_compute`, `run_options`, `start_next`, `run_all`.
