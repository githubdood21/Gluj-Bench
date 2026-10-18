# Gluj-Bench 0.4.0

This release makes the main CPU and GPU compute benchmarks easier to configure, adds shareable result snapshots, and improves navigation responsiveness.

## Customizable benchmarks

- Simple automatic choices and full customization for CPU AVX2 FP32 vector/matrix, INT64 multiply-add and ASCII scan workloads.
- Unified register and memory-backed modes for GPU FP32 vector, FP16 matrix and INT8 matrix compute, plus simplified FP32 system-RAM offload settings.
- Data-size and physical-core sliders, workload-specific defaults, and independent per-test preferences restored across launches.
- Up to three recent configurations per workload and hardware setup, with compatible reruns grouped for latest/best measurements.

## Share and compare results

- Select completed results for a snapshot, with common highlights selected initially.
- Copy the image or save a PNG containing the full completed-results list in its metadata, including unselected results, settings and sample statistics.
- Import an original snapshot PNG as a separate comparison source.
- Show each result's CPU/GPU target and device model, plus recorded RAM capacity and configured transfer speed where available.

## Responsiveness and fixes

- Snapshot generation and file processing run in background threads; changing selections cancels obsolete generation.
- Page navigation and detail toggles update directly on click. Benchmark panels refresh independently and result charts are reused when revisiting rows.
- Fixed per-test override state when switching benchmarks and overlapping snapshot descriptions.

## Download and update

Download **Gluj-Bench-0.4.0-windows-x64.zip**, extract the entire archive into a writable folder, and keep both executables together. The matching `.sha256` file contains the archive checksum.

When updating, keep `results.json`, `settings.json` and `benchmark-settings.json`. Older results retain their recorded measurements; rerun a benchmark to capture hardware fields that were not previously recorded. Keep original snapshot PNGs for comparison imports, because sharing services may remove metadata.

Windows x64 and a current Vulkan-capable graphics driver are required for GPU benchmarks. Unsupported workloads remain disabled with an explanation. FP8 and sparse matrix workloads remain capability discovery entries without verified runners; SMT selection remains a future feature.

Local validation: 183 workspace tests passed, strict workspace Clippy and formatting checks passed, navigation and image-preview checks passed, and the optimized Windows release build and packaged ZIP integrity were verified.
