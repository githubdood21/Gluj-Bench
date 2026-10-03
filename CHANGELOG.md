# Changelog

All notable changes to Gluj-Bench are documented here. The project follows semantic versioning.

## [Unreleased]

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
