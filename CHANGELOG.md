# Changelog

All notable changes to Gluj-Bench are documented here. The project follows semantic versioning.

## [Unreleased]

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

[Unreleased]: https://github.com/githubdood21/Gluj-Bench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/githubdood21/Gluj-Bench/releases/tag/v0.1.0
