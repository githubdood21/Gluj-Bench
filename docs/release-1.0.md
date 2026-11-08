# Gluj-Bench 1.0.0

Gluj-Bench measures the work your hardware completes. Results are reported in TOPS, GB/s, nanoseconds per access, and workload-specific rates, with the dataset and configuration alongside each measurement.

- **Run a Curated plan or build your own.** Choose among fifteen CPU, GPU, and memory workload families. Configure supported precision, core allocation, dataset size, access order, and memory placement in Custom.
- **See throughput as datasets grow.** Interactive compute graphs show measured points, a compute reference when available, and the throughput drop at the largest dataset when comparable.
- **Compare and share measured results.** Browse run history and observations, name or pin baselines, export raw native records, or save a PNG snapshot with comparison data.
- **Find the important controls faster.** The brass-and-iris interface has a visible test checklist, compact settings, context-specific history actions, and a persistent Stop all control during runs.

## Windows x64 download

Extract the ZIP into a writable folder. Launch `gluj-bench-ui.exe` with `gluj-bench-worker.exe` alongside it. A Vulkan-capable graphics driver is required for GPU workloads.

When updating, carry over `run-history/`, `results.json`, `settings.json`, and `benchmark-settings.json`. Existing results retain their recorded definitions and context. Use matching workloads, units, and settings when comparing runs.

The release was tested locally on a Ryzen 9 5950X and Radeon RX 7900 XTX. See the [local qualification and remaining hardware coverage](phase-9-qualification.md) and the [changelog](../CHANGELOG.md).
