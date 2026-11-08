# Phase 4: reproducible CPU access patterns

Current defaults (2026-11-06): six measured observations at 250 ms each; Shorter uses three at 100 ms. Curated uses standard access with one realization. Randomized access and additional seed realizations are available in Custom. The historical implementation notes below retain the earlier design; see [current follow-up notes](ui-redesign-implementation.md#lean-default-workloads-2026-11-06).


Implemented: 2026-11-01. Validation results are recorded in the implementation plan.

## Controls and curated coverage

Access is an explicit per-test setting for buffered AVX2 FP32, buffered INT64,
ASCII scans and CPU bandwidth. Customized tests default to sequential access.
Register-only arithmetic does not accept buffer access controls; the UI hides them.
CPU matrix compute retains its existing traversal and operation definition.

The Curated test mode action and curated family/category plans pair sequential and shuffled-block conditions with
identical dataset/worker choices, three realizations and the same measurement
policy. They execute and report separately. Register modes remain single conditions.
Curated latency conditions use three seeds of their existing dependent-chain mode;
latency locality is not replaced with an independent-load bandwidth workload.
Curated plans reset custom seed/content/size overrides to their declared defaults.
Cache-bandwidth plans use 64-byte blocks so small cache targets do not collapse to
a single shuffled block identical to sequential traversal. Other curated buffered
conditions use a 64 KiB block target.

The UI offers an access pattern, fixed/fresh root seed, one/three realizations,
block target, ASCII content, and latency working-set/locality controls where
supported. RAM bandwidth supports automatic, one chosen size, and sweep-to-size.
Preferences remain version 1 with additive defaults; the hardware cache revision
advances to refresh advertised options. Old obsolete RAM option names are still
unsupported; current `dataset_mode` / `dataset_bytes` controls are honored.

## Schedule definition

`splitmix64-fisher-yates-v1` uses SplitMix64, rejection-sampled bounded integers
and Fisher–Yates permutations. The default root is `0x6a09e667f3bcc909`. Root,
worker, size and realization derive deterministic seeds. Buffered compute also
derives a block root from the measurement block index; native realization parameters
record it. The seed, generator revision and resolved options make the schedule
reproducible; duration-based pass counts remain measured outcomes.

- **Sequential:** complete spans in address order. Cold RAM bandwidth retains a
  sequential stream with a reproducible rotated start per pass.
- **Shuffled blocks:** a seeded permutation visits every block exactly once per
  pass, including the final partial block. Values within a block remain sequential.
- **Rotating start regions:** complete coverage per pass; the starting block advances
  by one per pass. This rotates traversal order within the allocated footprint; it
  does not allocate a new dataset or guarantee eviction between passes.
- **Scattered kernel blocks:** a seeded permutation of the minimum safe kernel
  block: 64 bytes for bandwidth/ASCII, 64 elements for numeric kernels (768 bytes
  of FP32 arrays or 1,024 bytes of INT64 arrays). This is not per-lane GPU scattering.
- **Hot/cold reuse:** a complete shuffled cold traversal, with one repeated block
  from the first quarter of the footprint after each cold block. The hot blocks
  cycle in order. Useful operations, scanned bytes and copy bytes count every visit.

Numeric block targets round down to complete 64-element kernel groups, with a
minimum of one group. Native records report the actual block footprint. Schedules
support non-power-of-two spans without dropping tails or overlapping output owners.
Each worker owns its input/output buffers. Setup uses fallible allocation and
cancellation checks. Schedule generation is outside active timing; schedule lookup,
range dispatch, repeated hot visits and traversal are inside it. These costs are
part of the chosen workload, so scattered rates are not a peak-bandwidth claim.

Sequential compute cohorts execute the same traversal. Their extra coverage matches
curated randomized conditions; different cohort IDs do not imply different random
sequential workloads. ASCII content stays fixed within the run while access seeds
vary. Bandwidth source initialization and rotated starts derive from the root seed,
rather than a process-global preparation nonce.

## Seeds, statistics and records

`access_seed_policy` is `fixed` by default. `fresh` chooses a root once at execution,
then stores the actual fixed root in the resolved/native record. The original
request retains the fresh-seed choice. Preview resolution does not generate seeds.

`access_realizations` accepts 1 or 3. Each realization receives the full declared
observation count and active-duration floor, including visits across interleaved
compute size blocks. Extensive remains 12 observations of at least 250 ms **per
realization**; Shorter remains 6 of at least 100 ms. Three realizations therefore
increase coverage and runtime instead of dividing an existing short window.

Raw samples and sample blocks reference their actual realization. Native records
retain per-realization/per-condition counts, median, range and population SD.
Timing metadata separates per-realization targets from total ensemble targets and
uses the selected policy for warmup budgets.
Summary metrics pool equally covered observations descriptively; the UI labels
pooled spread/coverage, and does not describe ensemble spread as fixed-workload
repeatability or confidence. A missing realization or insufficient coverage fails
the condition while preserving partial observations. There is no stability-based
sample rejection or rerun recommendation.

Phase interval endpoints now use the same recorder clock, fixing a previously
intermittent nested-interval ordering failure uncovered during validation.

## Workloads, memory and latency

FP32 and INT64 retain their arithmetic references and independent output ownership.
ASCII supports patterned, seeded mixed, no-hit and all-hit inputs. An independent
bytewise count across the scheduled visits is prepared outside timing, recorded per
worker/realization and checked against the measured scan. Hot-region repeated hits
are included. No random-content setting is presented as random memory access.

Compute payload budgets reserve auxiliary schedules for all live realizations.
The size slider accounts for this reserve. Native summaries retain actual schedule
storage, and add it to allocated test-buffer bytes; buffer footprints remain
separate from process RSS, allocator/driver overhead and physical bus traffic.
Plan allocation estimates remain conservative, with unavailable estimates explicit.

RAM bandwidth sizes preserve the available-memory headroom and 1 GiB payload cap.
Sweeps visit powers of two from 64 KiB through the chosen endpoint, retaining every
size's metrics/raw samples and the largest-size summary. Sizes align per selected
worker. Cache-sized custom allocations use preloaded measurement; larger streaming
allocations retain per-pass cache flushing. Actual preparation is recorded. A small
system-memory allocation is labeled cache-sized; it is not evidence of RAM-only
traffic or measured cache residency.

Latency retains one dependent cycle and a single pinned worker. Seeded Sattolo
construction preserves full-node coverage, including localized blocks joined into
one cycle. New seeds are prepared/warmed outside timing. Custom sizes must contain
whole cache-line nodes and fit memory headroom; a cache-target allocation cannot
exceed its topology-derived target. Locality block size is advertised/shown only for localized chains; scattered chains
reject that inapplicable option. Scattered-block workloads also hide the independent
block target because their minimum kernel block defines it.
RAM modes permit explicitly chosen cache-sized footprints with an appropriate label.

Randomized/locality conditions model different reuse and turnover. They do not
claim measured cache misses, guaranteed eviction, or that random access must be slower.

## Accounting and compatibility

Current CPU/GPU bandwidth `copy` summaries report useful payload moved. A separate
`copy_accounted_read_write` metric (with a tier prefix where applicable) reports
twice that amount as logical read/write traffic. This is not a hardware bus counter.
GPU raw copy observations retain their accounted-traffic definition; primary
payload summaries are explicitly derived by dividing by two. CPU raw copy
observations now use useful-payload accounting. Transfers remain directional payload
measurements. Old native files, cached results and imported values are not rescaled.

New accounting carries `useful-payload-plus-logical-traffic-v1`; CPU bandwidth uses
`cpu-bandwidth-3-access-payload`. Buffered FP32, INT64 and ASCII carry their
`*-2-access` revisions; latency uses `2-seed-ensemble` / `sattolo-single-cycle-v2`.
Result grouping and cache replacement include access, seed, content and accounting
identity, keeping sequential/randomized results and incompatible historical values
separate. The worker transport/native schemas remain additive.

## Validation scope

Regressions cover deterministic complete schedules and tails, rotating order,
allocation bounds, declared ensemble coverage, fresh-seed freezing, kernel output
references, ASCII expected hits, dependent-chain integrity, supported options,
curated equal coverage and separate result-cache entries. Release-worker smoke
results and completed workspace checks are listed in the implementation plan.
Cross-hardware performance calibration remains Phase 9 work; GPU access-pattern
implementation remains Phase 6 work.
