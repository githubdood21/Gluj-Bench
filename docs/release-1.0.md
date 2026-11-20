# Gluj-Bench 1.0.0

Gluj-Bench measures the work your hardware completes. All benchmarks in 1.0.0 are synthetic, with results reported in TOPS, GB/s, nanoseconds per access, and workload-specific rates. The dataset and configuration stay alongside each measurement; there is no combined CPU or GPU score.

- **Run a curated set or save your own.** Choose among fifteen CPU, GPU, and memory workload families. Curated sets are read-only. Use **New set** to combine tests with individually configured precision, core allocation, dataset size, access order, and memory placement.
- **See throughput as datasets grow.** Interactive compute graphs show measured points, a compute reference when available, and the throughput drop at the largest dataset when comparable.
- **Compare and share measured results.** Browse run history and observations, name or pin baselines, export raw native records, or save a PNG snapshot with comparison data.
- **Find the important controls faster.** The brass-and-iris workbench groups test sets by workload, keeps editing in a fixed side panel, and provides short hover hints, consistent dropdowns, and run controls during execution.

Real rendering workloads, inference with real LLM models, multi-GPU and multi-CPU tests, and storage benchmarks are planned for future releases. They are not included in 1.0.0.

## Windows x64 download

Extract the ZIP into a writable folder. Launch `gluj-bench-ui.exe` with `gluj-bench-worker.exe` alongside it. A Vulkan-capable graphics driver is required for GPU workloads.

When updating, carry over `run-history/`, `results.json`, `settings.json`, and `benchmark-settings.json`. Existing results retain their recorded definitions and context. Use matching workloads, units, and settings when comparing runs.

The release was tested locally on a Ryzen 9 5950X and Radeon RX 7900 XTX. See the [local qualification and remaining hardware coverage](phase-9-qualification.md) and the [changelog](../CHANGELOG.md).
