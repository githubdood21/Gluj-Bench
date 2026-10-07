# Ideas for later

## Automatic CPU core-count sweep

Accepted as a small automated feature idea; not implemented or scheduled.

- Run an existing CPU scaling workload at several physical-core counts, reusing allocation, affinity, sampling, and cancellation support.
- Keep the aggregate dataset size and workload definition fixed across runs. Record per-worker dataset sizes because changing the core count also changes cache residency.
- Show throughput against core count and identify the smallest tested allocation that retains a user-selected share of the best measured throughput, such as 95%.
- Report noisy or borderline results as inconclusive and offer a repeat measurement rather than presenting a definite recommendation.
- Keep the result specific to the tested workload, dataset, and selected cores; do not claim an optimal allocation for other applications or measured power savings.
- Start with one supported CPU scaling workload and a short selection of core counts to keep the workflow lightweight.

## Burst and sustained workload modes

Accepted as an optional workload-duration feature idea; not implemented or scheduled.

- Let users select short bursts separated by idle intervals or a longer sustained run for supported workloads. Expose duration and burst/rest settings with simple defaults.
- Keep the dataset and calculation definition comparable across modes, while recording duration, duty cycle, warmup, and existing activity-pacing settings with the result.
- For burst mode, report active-work throughput separately from elapsed-time throughput, which includes idle intervals. Make the distinction visible in results.
- For sustained mode, collect throughput over time and show whether performance remains steady or declines, with an early-versus-late summary and sample variation.
- Do not attribute a decline to temperature, throttling, or power limits without measurements that establish the cause.
- Preserve cancellation and the UI redraw cap during long runs. Avoid timing each tiny operation or introducing frequent UI updates that interfere with measurement.
- Keep different modes and duration settings distinct in saved comparisons. Start with a single workload before extending support.

## CPU cache and RAM latency

Implemented: separate CPU L1/L2/L3 cache read-latency tests and two RAM tests (fully scattered random-object reads and localized reads within 64 KiB blocks).

- Cache tests use the instance attached to the pinned core, not summed cache capacity across complexes. L1 uses 75% capacity; L2/L3 aim for four times the preceding level, capped at 75% of target capacity, and require more than twice the preceding level for separation.
- Use a randomized dependent pointer chain pinned to a processor core, with working-set sizes based on the detected cache hierarchy.
- Reuse sampling, cancellation, aligned allocation, and processor-affinity infrastructure. Measure latency in a separate phase from bandwidth.
- Validate cache-level separation, compiler behavior, timing overhead, address-translation effects, and repeatability on AMD and Intel hardware.
- Explain that this is measured access latency for the test's conditions, including the processor and memory path.
- Rough development estimate: 4–8 hours for a prototype, 1–2 days for a polished implementation on one machine, and 3–5 days total with cross-hardware validation (hardware available).
- Tentative extra full-suite measurement time: 8–15 seconds, depending on sampling and preparation.

Further cross-hardware validation remains, including AMD/Intel repeatability, shared-cache interference, timing overhead and address-translation effects. An ignored `cache_residency_probe` CPU test can measure a working-set sweep on the current machine when investigating residency anomalies.
