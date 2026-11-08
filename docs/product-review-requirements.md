# Product review requirements

Updated 2026-10-31 following the product review and user feedback.

Status: requirements implemented through the local 0.5.0-rc.1 candidate on
2026-11-02. The [qualification report](phase-9-qualification.md) records available
hardware evidence and remaining cross-device release gates. This document retains
the confirmed product requirements; the roadmap records delivery status.

The [CPU/GPU catalogue audit](benchmark-catalogue-audit.md) informs the
[phased implementation plan](implementation-plan.md), which is the delivery source
of truth. The user confirmed the proposed family cards, extensive testing by
default with an optional shorter run, and the full staged review scope on
2026-10-31. Shared sampling infrastructure applies across all families; randomness
controls apply only to modes where they have a defined workload meaning. Numeric
defaults and hardware-dependent support remain validation gates in the plan.

## Measurement policy

Access randomness is an explicit user setting. Customized workloads default to
sequential access. For applicable workloads, curated tests cover sequential and
randomized access as separate measured conditions with equal sampling/duration
coverage and separate results. The user confirmed this distinction on 2026-10-31.

- Tests must be extensive enough to characterize performance within a single run.
  Variability at a fixed dataset size is handled by the test, not delegated to a
  user-requested rerun.
- Never recommend rerunning or retesting because samples vary, a reference drifts,
  calibration is uncertain, or a difference is small. Manual reruns remain a user
  choice. This supersedes the earlier review's proposed repeat/confirmation prompts.
- Report measurements, spread, coverage, and applicable limits directly. Continued
  variability is a measured characteristic; do not discard it to produce an
  apparently stable headline number.
- Reference drift is reported numerically. It must not be presented as evidence
  for a thermal, power, or memory cause without additional measurements.

### Automatic sampling

1. Establish minimum active measurement duration and minimum independent sample
   count per condition, including references and calibration conditions. Increasing
   sample count must not simply divide an unchanged short duration into smaller
   samples.
2. Perform preparation, warmup, and validation outside timed workload execution.
   Record their wall time separately. Cold-start conditions must use their own
   explicit preparation policy rather than the warm-cache policy.
3. Collect multiple blocks of measurements. Distribute reference anchors through
   long sweeps and counterbalance dataset order with the appropriate cache-state
   preparation before each condition.
4. Allow bounded automatic extension where the selected policy needs more coverage.
   Set sample and duration limits before measurement, expose estimated runtime,
   preserve cancellation, and report actual duration and sample count.
5. At the budget limit, finish normally with the collected numbers and observed
   distribution. Do not loop until a stability threshold happens to pass.
6. Preserve chronological samples, measurement-block identifiers, condition,
   seed, and elapsed/active timings. Retain genuine slow samples. Record explicit
   execution errors separately from performance variability.
7. Distinguish variation within a fixed workload realization from differences
   between randomized realizations. Do not pool these silently.
8. Use a documented statistical method appropriate to the displayed estimator and
   any adaptive stopping rule. A five-sample range is not a confidence interval;
   repeated operations inside one timing window are not independent samples.

Initial duration, sample-count, and extension limits require empirical validation
on supported hardware. These requirements do not claim that any finite run can
remove natural variation or guarantee a particular precision.

### Result presentation

- Default to median throughput or latency, dataset size, observed spread, sample
  count, and concise recorded configuration.
- Provide quantiles only where sample coverage supports them. Preserve raw samples
  for inspection and export; label empirical ranges and percentiles accurately.
- Show comparison deltas as measured differences. If comparison uncertainty is
  provided, use neutral descriptions such as "difference within observed
  variation" without asking the user to rerun.
- Move detailed methodology and raw metadata behind expansion controls, while
  keeping significant measurement limitations visible beside the affected number.
- Replace "Lost" and animated red reference gaps with neutral "Below reference"
  or "Retained throughput" language. Reference gaps do not establish recoverable
  performance or measured stall time.
- Use the selected dataset as the main result, initially the largest tested size.
  Offer extrema as secondary diagnostics.
- Remove measurement-driven instructions to repeat, retest, or rerun from backend
  guidance, UI text, sharing output, and documentation. Review occurrences by
  meaning; legitimate "repeated samples" descriptions remain valid.

## Reproducible workload randomness

Numeric values and memory access order are separate controls. Random values alone
do not turn sequential memory accesses into a cache-miss workload. Existing GPU
FP32 seeded value perturbation therefore does not satisfy the new requirement.
Existing RAM latency pointer-chain randomization is a useful implementation
precedent, but dependent latency and parallel throughput remain distinct tests.

### Access conditions

| Condition | Workload definition | Purpose |
| --- | --- | --- |
| Sequential | Existing streaming or blocked traversal | Preserve a useful baseline |
| Shuffled blocks | Visit a seeded permutation of blocks; retain contiguous work inside each block | Exercise reduced inter-block locality while preserving defined vector or GPU-lane access |
| Scattered access | Independent indexed reads over a defined working set, with controlled concurrency | Measure irregular-access throughput |
| Changing working set | Rotate active regions and vary a defined hot/cold distribution | Exercise turnover, reuse distance, and cache pressure |

Cache eviction is workload behavior, not something randomness guarantees. Random
access within a small resident dataset can still hit cache. Record footprint,
reuse policy, and region turnover; do not claim a measured miss percentage without
hardware-counter evidence.

### Implementation constraints

- Use a versioned deterministic random generator, recorded root seed, and explicit
  seed schedule. Derive seeds reproducibly for each worker, dataset, and block.
- The default uses a fixed, recorded ensemble of realizations so two runs execute
  the same workload distribution. A fresh root seed can be an explicit user choice.
- Precompute access schedules outside timed sections where practical. Account for
  index buffers in allocation budgets and report their footprint and timed reads.
  If address generation is performed during execution, record that overhead as
  part of the workload rather than describing it as memory access alone.
- For shuffled full-pass workloads, use permutations with complete coverage and no
  duplicated or omitted blocks. For hot/cold or sampled workloads, define and
  record the intentionally different coverage policy.
- Specify block size, cache-line utilization, region count, hot/cold probability,
  replacement policy, and arithmetic reuse. Keep useful work per element fixed
  across a given condition's dataset sweep.
- Prevent CPU and GPU write races. A repeated random input index must not imply
  multiple unsynchronized writers to the same output location.
- On GPUs, distinguish shuffled workgroup/block order from per-lane scattered
  access. The latter also changes coalescing and transaction efficiency; report
  that condition separately.
- For matrix kernels, begin with shuffled tile traversal that preserves the
  multiplication, accumulation dependencies, and useful operation count. Do not
  silently turn a dense matrix test into an unrelated gather kernel.
- Treat warmed traversal, working-set turnover, and explicit cold-start preparation
  as separate cache-state policies. Eviction preparation is outside timing unless
  turnover is intentionally part of the measured workload. A preparation buffer
  alone does not prove that a particular cache is cold.
- Preserve existing sequential results. Randomized conditions have separate result
  identities and cannot silently replace or rank against sequential conditions.
- Record access-pattern revision, generator revision, root seed or ensemble,
  actual footprint, block size, reuse/turnover settings, placement, concurrency,
  numeric format, and arithmetic definition in results and exports.

### Validation

- Verify deterministic replay and supported non-power-of-two sizes.
- Verify permutation coverage, bounds, alignment, and cancellation during setup.
- Check interior and boundary outputs against references outside timed sections.
- Verify actual operation/traffic accounting, including index traffic.
- Compare sequential, shuffled-block, and scattered conditions at the same useful
  workload size without assuming that randomized access must always be slower.
- Inspect distributions within each realization and across realizations separately.
- Validate supported CPU/GPU paths and keep unimplemented conditions unavailable
  with a specific explanation.

## Other accepted review directions

- Separate measured outcomes, inferred explanations, and proposed experiments.
- Make effective kernel traffic a secondary chart where it is just a rescaling of
  the throughput measurement. Do not present it as physical bus traffic.
- Improve output correctness checks and expose validation coverage accurately.
- Add task-oriented starting points, resolved configuration previews, runtime and
  allocation estimates, and clearer queue controls.
- Keep historical best secondary to typical/latest measured behavior and label
  its scope. Explain incompatible comparisons with specific reasons.
- Add named baselines and durable run history with workload and build provenance.
- Make exported result selection match the user's selection; including all results
  is an explicit choice.
- Add automated CPU core-count experiments, sustained-performance measurements,
  and representative matrix shapes while preserving comparable work definitions.
- Prefer automatic measured experiments over numerical tuning suggestions inferred
  only from a throughput gap. Follow-up runs are user initiated; the app reports
  numbers without recommending reruns.

## Implementation sequence

See the [implementation plan](implementation-plan.md) for the authoritative phase
order, dependencies, migration work and acceptance criteria. It supersedes the
initial abbreviated sequence in this document.

## Methodology references

- [Google Benchmark user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md):
  warmup, timing, and repetitions.
- [Google Benchmark random interleaving](https://github.com/google/benchmark/blob/main/docs/random_interleaving.md):
  distributing repeated measurements across execution order.
- [Intel hardware prefetch tuning](https://intel.github.io/optimization-zone/hardware/HWPrefetchTuning/):
  access-pattern and prefetch behavior; actual behavior varies by processor.
