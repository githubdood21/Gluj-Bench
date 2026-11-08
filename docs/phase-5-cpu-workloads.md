# Phase 5 — CPU workloads and experiments

Implemented in the working tree on 2026-11-01. Existing family cards and benchmark
IDs remain; controls select separately identified conditions. GPU expansion is Phase 6.

## Buffered arithmetic and population count

Scalar FP64 and AVX2/FMA FP64 retain their register modes and add L1/L2/L3/RAM/custom
buffers, single sizes and sweeps. Scalar uses separate multiply/add operations;
SIMD uses fused operations. Both count 32 operations per value over 16 rounds,
validate independently and report logical input/output traffic. Their three arrays
occupy 24 bytes per value. Data-backed POPCNT uses one 64-bit input word per count,
checks the result against an independent bit-count reference and retains its
separate register instruction mode. These definitions have new workload revisions.

The five Phase 4 access patterns and fixed/fresh seeds apply to buffered modes.
Customized runs default to sequential. Curated plans measure sequential and shuffled
conditions separately with equal coverage across three realizations. Access patterns
vary locality; they do not measure hardware cache misses. Register mode rejects
irrelevant buffer controls. Cache previews account for complete kernel-block alignment.

## DEFLATE and sieve

Compression and decompression expose seeded compressible/mixed/incompressible
corpora, corpus size, independent block size and compression level 0–9. Each block
is a complete raw DEFLATE stream. Sequential or shuffled block order preserves
stream validity. Library state and output buffers are reused; stream reset/clear
and codec work are timed. Preparation, allocations and independent round-trip
validation are outside timing. Directional throughput counts original bytes per
second. Compression ratio is compressed bytes divided by original bytes; smaller
values mean a smaller stored stream. These are block-codec results, not file/archive
format benchmarks. Corpus content and block boundaries stay fixed across worker counts.

The sieve exposes its inclusive limit and mark-buffer footprint, measures reset,
marking and final prime counting together, and checks known prime counts or an
independent reference. It reports primes/s with the range and primes per complete
sieve. Each worker executes a complete sieve; it is excluded from fixed-aggregate
core experiments. AES remains the verified round-instruction microbenchmark;
its name and metadata do not claim a complete encryption implementation.

## Matrix and core-count experiments

The historical matrix size/sweep mode remains available. Explicit global M×N×K
shapes are square 128×128×128, tall 512×128×128 and wide 128×512×128, with reuse
1/4/16 and K blocking 32/128/256. Each reuse computes a complete overwrite product.
Workers own disjoint A/output row shards and share immutable B. Independent sampled
FP64 dot products validate output. Shape defines allocation; legacy preset/size
controls cannot change an explicit shape. Logical traffic describes the microkernel's
loads/stores, not measured DRAM traffic.

A separate user-invoked core-count experiment queues 1, 2, 4, … workers through
the selected maximum, including a non-power-of-two maximum. It supports custom
single-size buffered FP64/POPCNT, independent codec blocks and explicit matrix
shapes. It preserves aggregate input, seed, geometry and work per sweep while
partitioning it over workers. It excludes register loops, historical matrix scaling
and sieve because those do not promise fixed aggregate work. Fresh seeds freeze
for the whole experiment. Records retain actual worker placement, shard offsets,
shard sizes and shared/private buffer ownership. Equal work per sweep does not
imply equal completed sweep counts during duration-based sampling.

## Sustained coverage and records

CPU controls offer standard, 15-second or 60-second sustained coverage. The worker
accepts integer durations through 120 seconds. Each condition and realization gets
the full duration and sample floor; sustained conditions run contiguously. Reference
measurements keep their separate short budget. Sample splitting rounds duration up
so fractional millisecond targets cannot underfill declared coverage.

Native records retain chronological raw observations and offsets. Upgraded workload
summaries additionally report first/last-third aggregate throughput medians; scoped
per-worker series retain first/last-third means. UI facts label the applicable scope
and counts. Early/late behavior is descriptive and is not attributed to temperature,
power or throttling without telemetry. Variability remains in the data and never
produces a recommendation to rerun.

Allocation metadata enumerates actual test-vector capacities and access schedules,
separately from the conservative allocation reserve. Codec library-managed state
is excluded from enumerated test buffers; its budget reserves 10 MiB per worker.
These values are not process RSS. Kernel revisions, operation/traffic definitions,
input generator revision, seed realizations and worker layout support reproduction.
The unreachable historical single-thread dispatch was removed after compatibility
and live-path checks; its public alias still resolves to canonical one-core INT64.

## Validation

The workspace passed 286 tests (23 intentionally ignored), strict Clippy, formatting
and the optimized build. Forty-one release-worker cases passed, including sustained
POPCNT/AES, partial cancellation, legacy alias dispatch and Extensive coverage. Saved native records
matched terminal responses. The isolated desktop codec run saved 96 observations
across 16 workers and displayed the new corpus/level/block/ratio facts. Automated fixtures cover fixed
input/work across worker counts, all access patterns, corpus/level/direction choices,
codec round trips, full matrix output equivalence and custom sieve references.
Local validation does not complete Phase 9 cross-hardware qualification, timing
calibration or long-duration thermal/power characterization.
