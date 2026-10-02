# Ideas for later

## CPU cache and RAM latency

Deferred while the results display and guided descriptions are refined.

- Add measured L1, L2, L3, and RAM read-access latency in nanoseconds (lower is better).
- Use a randomized dependent pointer chain pinned to a processor core, with working-set sizes based on the detected cache hierarchy.
- Reuse sampling, cancellation, aligned allocation, and processor-affinity infrastructure. Measure latency in a separate phase from bandwidth.
- Validate cache-level separation, compiler behavior, timing overhead, address-translation effects, and repeatability on AMD and Intel hardware.
- Explain that this is measured access latency for the test's conditions, including the processor and memory path.
- Rough development estimate: 4–8 hours for a prototype, 1–2 days for a polished implementation on one machine, and 3–5 days total with cross-hardware validation (hardware available).
- Tentative extra full-suite measurement time: 8–15 seconds, depending on sampling and preparation.

This item records an idea; latency is not implemented or scheduled.
