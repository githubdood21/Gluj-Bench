# Changelog

All notable changes to Gluj-Bench are documented here. The project follows semantic versioning.

## [Unreleased]

### Added

- Measured GPU execution and end-to-end time per dataset pass for FP32 VRAM/RAM-offload and FP16 matrix scaling, with sample ranges, readable time units and recorded batch sizes. The busy/wait percentage model has been retired; throughput/reference comparisons retain their performance meaning.

- One GPU FP32 system-RAM offload scaling test with a configurable 50%, 75%, or 100% RAM share and the remaining data in VRAM.
- Nominal host-traffic graphs, per-tier RAM/VRAM placement, allocation safeguards, and untimed scalar-reference checks for RAM-offload results.

### Changed

- Latest-measurement details now show throughput, sample range and variation cards, sample-spread bars, and workload context. Raw measurements are collapsible; GPU timing notices are limited to scaling tests.

- Tuning details now use measurement cards, throughput comparison bars, allocation bars, a measurement-quality notice, and a retest checklist. Full technical reasoning remains available in a separate disclosure.

- RAM-offload tuning guidance suggests exploring a lower GPU core-frequency limit or smaller GPU work batches, with throughput retesting and application-level validation.

- Added persisted global scaling defaults and session-only per-test overrides, with automatic sweeps, user-sized sweeps, and single-dataset sampling. Queued runs capture effective settings and target hardware. Legacy offload result IDs migrate with their percentage preserved.
- FP32 GPU scaling now explicitly orders initialization and repeated output writes, respects per-descriptor storage-buffer limits, and cleans up partial Vulkan setup failures. Saved comparisons distinguish the updated profile revision.

## [0.2.0] - 2026-10-03

### Added

- CPU AVX2 FP32 vector and blocked FP32 matrix scaling profiles with per-tier throughput and effective bandwidth graphs.
- Configurable CPU physical-core allocation and RAM allocation budgets, alongside GPU activity and VRAM budgets.
- Latest-run tuning guidance with concise summaries and expandable measurements.
- Local JSON result history for compatible hardware and settings comparisons.
- Persistent stop control that cancels the active benchmark and clears queued benchmarks.
- Application logo with a multi-resolution Windows icon.

### Changed

- Redesigned System Information with hardware icons, readable memory figures, capability badges and a compact scan-readiness indicator.
- Unified component results in a scrollable layout with expandable explanations and scaling graphs.
- Renamed navigation to System Information, Benchmarks and Benchmark Results.
- Refined GPU scaling profiles and memory-pressure guidance to use current measurements and the largest tested dataset.
- Moved system-memory results alongside the CPU and standardized readable performance units.

## [0.1.0] - 2026-07-22

### Added

- Native Windows x64 UI and isolated benchmark worker.
- CPU L0–L3 cache and system-memory read, write, and copy bandwidth suites.
- Multithreaded and single-threaded CPU arithmetic, string, prime, codec, AES, POPCNT, and extended-instruction workloads.
- Vendor-neutral GPU cache, VRAM, and host-link bandwidth through wgpu.
- GPU FP32, FP16, FP64, INT32, packed INT8 vector, and capability-gated FP16 cooperative-matrix workloads.
- Per-metric sample statistics and inferred compute-versus-memory-bound diagnostics.
- Versioned worker CLI and newline-delimited JSON protocol.
- MIT licensing and automated Windows release packaging.

[Unreleased]: https://github.com/githubdood21/Gluj-Bench/compare/v0.2.0...HEAD
[0.1.0]: https://github.com/githubdood21/Gluj-Bench/releases/tag/v0.1.0

[0.2.0]: https://github.com/githubdood21/Gluj-Bench/releases/tag/v0.2.0
