# Gluj-Bench

[![CI](https://github.com/githubdood21/Gluj-Bench/actions/workflows/ci.yml/badge.svg)](https://github.com/githubdood21/Gluj-Bench/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/githubdood21/Gluj-Bench?display_name=tag)](https://github.com/githubdood21/Gluj-Bench/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Windows%20x64-5d91ff)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-3dd6c6)](LICENSE)

<p align="center">
  <img src="design/logo-drafts/loop-chip-g-preview.png" alt="Gluj-Bench chip and retest-loop logo" width="112">
</p>

Gluj-Bench is a free, vendor-neutral hardware benchmark for Windows. It measures CPU and GPU calculation throughput, cache and memory bandwidth, and how performance changes as a workload grows beyond cache into RAM or VRAM.

Version **0.2.0** focuses on understanding the hardware available to a workload. Results are real metrics with units, repeated samples and recorded settings. There is no combined ranking or synthetic performance score, and no prediction of game frame rates or AI token speeds.

## Get started

1. Download the [latest Windows x64 release](https://github.com/githubdood21/Gluj-Bench/releases/latest).
2. Extract the entire ZIP into a writable folder. Keep `gluj-bench-ui.exe` and `gluj-bench-worker.exe` together.
3. Launch `gluj-bench-ui.exe` and wait for the hardware scan to finish. The app starts its worker in the background without opening a console window.
4. Open **Settings** to choose your CPU allocation, GPU activity and scaling memory budgets.
5. Open **Benchmarks**, select a category and benchmark, then run it or queue the available benchmarks in that category.
6. Review **Benchmark Results**. Expand the explanations, scaling graphs or tuning measurements for more detail.

A floating **Stop all tests** control remains accessible on every page while a benchmark runs. It requests cancellation of the active benchmark and clears the queue; it shows **Stopping...** until cancellation completes.

## Requirements

- Windows x64 and an x64 processor. CPU topology discovery and thread affinity currently use Windows APIs.
- A graphics driver with Vulkan compute support for GPU benchmarks. CPU and RAM benchmarks remain usable when no compatible GPU is available.
- AVX2 and FMA support for the CPU AVX2 vector and FP32 matrix scaling benchmarks.
- Driver-exposed cooperative-matrix capabilities for the corresponding GPU matrix benchmarks.
- Enough available RAM or GPU memory for the chosen allocation budget.

The release does not require a manufacturer-specific compute SDK. Unsupported benchmarks stay visible with an explanation of the missing capability or implementation.

## The application

### System Information

<p align="center">
  <img src="SysInfo.png" alt="System Information showing scanned hardware, processor details and capability badges" width="1000">
</p>

Hardware cards show your processor, system memory and detected graphics devices with component icons, readable capacities and capability badges. **Details** expands the underlying hardware properties and CPU cache topology.

A checkmark indicates that hardware has been scanned and benchmarks are available. The **available memory** figure reflects the hardware scan; it is not a live utilization monitor. **Refresh hardware** updates the discovered capabilities.

### Benchmarks

<p align="center">
  <img src="Benchmarks.png" alt="Benchmarks page with CPU scaling workloads and benchmark details" width="1000">
</p>

Choose between CPU bandwidth, CPU performance, GPU bandwidth and GPU performance. GPU benchmarks use the selected graphics adapter. Benchmark descriptions explain what each workload measures and where that type of calculation or memory access is useful before you run it.

Runs use five repeated samples by default. A scaling benchmark repeats sampling at each dataset size, so it can take substantially longer than a single throughput test.

### Benchmark Results

<p align="center">
  <img src="ScalingCompute.png" alt="FP16 matrix scaling results with compute throughput and effective traffic graphs" width="1000">
</p>

The selected component has one scrollable results page without workload pagination. CPU cache and system RAM measurements appear together. Memory rows show separate **read**, **write** and **copy** values in GB/s; compute rows show the latest and best compatible measurements.

Descriptions and tuning guidance start with a short takeaway. Expand their dropdowns to inspect the explanation, statistics and reasoning. Scaling rows feature the **largest tested dataset**, rather than the fastest small dataset that fits in cache.

## What is measured

| Category | Benchmarks |
| --- | --- |
| CPU bandwidth | L1, L2 and L3 data-cache bandwidth; system RAM read, write and copy bandwidth |
| CPU performance | FP32/FP64 arithmetic, AVX2/FMA, integer arithmetic, single-thread integer throughput, ASCII scanning, prime sieve, DEFLATE compression/decompression, AES rounds and POPCNT |
| CPU scaling | AVX2 FP32 vector throughput and blocked FP32 matrix throughput across increasing aggregate datasets |
| GPU bandwidth | Estimated effective cache bandwidth, GPU-accessible memory read/write/copy bandwidth, and host-device transfers |
| GPU performance | Register-resident FP16, FP32 and FP64 vector throughput; supported dense FP16 and INT8 matrix calculations |
| GPU scaling | Memory-backed FP32 vector and dense FP16 matrix profiles across increasing working sets |

FP8 and sparse matrix families are also represented in capability discovery. Their rows remain disabled when a compatible, verified runner is unavailable; dense matrix support alone does not imply sparse acceleration.

### Reading the units

- **TOPS:** trillions of operations per second. Floating-point multiply and add count separately, so one FMA per lane counts as two operations. Other operation types have their own definitions in the benchmark details; their rates are not interchangeable.
- **GB/s:** billions of bytes processed per second. Read, write and copy are distinct operations, with traffic accounting described in the details.
- **KiB, MiB and GiB:** binary dataset sizes. A GiB is 1,073,741,824 bytes; memory capacities and working sets use these units.
- **Sample statistics:** medians, minimums, maximums and variation describe repeated measurements of the same workload.

A throughput result describes that kernel, numeric format, dataset and configuration. It is not a measurement of every task the component can perform.

## Allocation and activity settings

Settings save beside the executable and apply to the next run. Controls are locked while benchmarking.

| Setting | Choices / default | Effect |
| --- | --- | --- |
| CPU worker allocation | Gentle 50%, Balanced 75% (default), Full 100% | In Automatic mode, selects that share of physical CPU cores, with at least one worker |
| CPU physical cores | Automatic (default), or an exact count | Uses one pinned worker per selected physical core, excluding SMT siblings; an exact count overrides the percentage preset |
| CPU scaling RAM budget | 20-80% of installed RAM; default 20% | Limits combined buffers for CPU vector and matrix scaling |
| GPU activity pacing | Gentle 50%, Balanced 75% (default), Full 100% | Adds cancellable idle intervals between GPU submissions; reduced modes also target shorter batches |
| GPU scaling VRAM budget | 20-80% of reported GPU memory; default 25% | Limits test-buffer allocation for GPU scaling profiles |

Single-thread CPU tests always use one worker. Ordinary CPU cache and RAM bandwidth tests keep their own dataset sizing; the percentage RAM budget applies to the scaling profiles.

The CPU scaling allocation also uses no more than 80% of currently available RAM and leaves at least 2 GiB available. GPU profiles reserve allocation overhead, and the matrix profile accounts for its output buffer. Vulkan storage-buffer limits, alignment and shared-memory limits on integrated GPUs can reduce the actual tested size. Concurrent applications can change memory availability or cause an allocation to fail.

Reduced allocation or activity can improve responsiveness and may reduce heat or power use, but it can also lower measured throughput. GPU timestamp measurements exclude pacing intervals; wall-clock transfer measurements include them. Settings do not impose a temperature limit or guarantee stability.

## Scaling performance

A scaling profile measures progressively larger datasets to expose how cache reuse and memory traffic affect calculation throughput. Each profile measures a register-resident compute reference before and after the sweep. The **latest post-sweep reference** drives the reported delta and tuning guidance; the initial reference is used to assess drift. These are measured references, not theoretical peaks from a hardware specification.

Expanded results include:

- Compute throughput and effective traffic graphs against logarithmic dataset size.
- Median points, minimum-maximum sample ranges and a current-run compute reference.
- A dataset selector with exact sample statistics and, for CPU matrices, per-worker dimensions.
- The largest tested allocation, its throughput and the change against the reference.
- A sustained memory-pressure transition when the recorded samples support that inference.

### CPU vector and matrix profiles

The AVX2 FP32 vector profile gives each worker two input arrays and one output array. It performs 16 FMAs per value, keeping a fixed compute-to-data ratio of about 2.67 operations per byte of effective traffic.

The FP32 matrix profile uses blocked AVX2/FMA multiplication. Each worker owns A[32,K], B[K,N] and C[32,N], with K=N increasing through the sweep. The fixed 32-row batch reuses weights across rows. Dataset sizes are the **aggregate allocation across all workers**, not the size per core.

Matrix GB/s counts effective accesses within the kernel, including data served repeatedly from cache. Vector traffic counts two input reads and an output write, excluding write allocation and cache-line writeback. Neither figure is direct RAM-controller utilization. Changing the core count also changes the per-worker matrix dimensions; inspect those shapes and the aggregate dataset size when comparing configurations.

### GPU vector and matrix profiles

The FP32 vector profile streams progressively larger memory-backed arrays. The dense FP16 matrix profile streams cooperative-matrix operands with reuse. Both work toward the selected VRAM budget and report the actual aligned size reached, subject to device limits.

Their effective traffic can include cache-served data. A throughput drop as the working set grows suggests memory pressure for that workload; it does not directly measure GPU stall time or the exact percentage of memory congestion.

## Tuning guidance: change one thing, then retest

Guidance uses the **latest run**, not the historical best measurement. Noisy samples or uncertain evidence may require another run instead of a numerical suggestion.

**CPU:** stable, sustained memory-pressure evidence with multiple physical cores can suggest an exploratory trial at roughly 75% of the current core count. Rerun at the same largest aggregate dataset. Keep the reduction if throughput remains within 5% of the original result or improves; restore more cores if the loss is larger. The app does not measure fewer-core performance automatically or establish an optimal core count.

**GPU:** supported scaling evidence can suggest a further reduction to the current core-frequency limit. The trial is relative to the configuration already measured, including any existing limit. Adjust in small steps and rerun the same dataset, aiming for no more than 5% throughput loss. Memory-bandwidth estimates are explanatory gaps, not memory-clock targets.

Power use, temperatures and performance after a change are not measured by the suggestion. For another application, change its worker count or affinity manually and test its own workload. **Gluj-Bench never applies CPU/GPU clock, voltage or external-process affinity changes.**

### Example use case: checking a GPU frequency limit

A local LLM may generate tokens at a rate limited by memory bandwidth. Gluj-Bench can help investigate a similar compute/memory imbalance: run the **FP16 matrix compute scaling** test (or the **FP32 compute scaling profile**), inspect the throughput drop at larger datasets, then use the latest tuning guidance to choose a small core-frequency-limit trial. Rerun the same dataset and settings to check whether at least 95% of its original throughput remains.

For illustration, an actual LLM workload might deliver **100 tokens/s at 400 W** without a limit and **95 tokens/s at 250 W** with a lower core-frequency limit and unchanged memory clock: 5% less throughput for 37.5% less power. Reported GPU utilization could stay at 100% even while compute waits for memory. These are example numbers, not Gluj-Bench measurements or predicted savings.

Validate the change in the actual application: prompt processing can slow down because it often depends more on compute throughput. Gluj-Bench measures hardware benchmark kernels; it does not optimize LLMs, predict token speeds, measure power or apply clock changes. Note that prefill speeds will see a larger drop in perfomance by doing this, See NVIDIA's [prefill and decode explanation](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) for context.

## Saved results and comparisons

Results and settings are portable files beside the executables. Hardware metadata uses the Windows application-data folder:

| File | Location | Purpose |
| --- | --- | --- |
| `results.json` | Beside the executables | Latest result per benchmark, hardware setup and allocation settings, up to 128 entries |
| `settings.json` | Beside the executables | Allocation budgets, CPU core selection and activity presets |
| `hardware-metadata.json` | `%APPDATA%/Gluj-Bench/` | Cached hardware metadata for startup and comparison context |

Keep `results.json` and `settings.json` when updating or moving the app. The application folder must be writable to persist them. Hardware metadata is refreshed separately.

Run a benchmark again, then choose a saved setup under **Compare with**. Compute rows show percentage **Change**; the memory display switches between **Latest**, **Best** and **Change**. A positive change means higher throughput. The displayed best measurement comes from compatible runs in the current session; the saved file is a compact latest-result history, not an unlimited archive of every run.

Comparisons require compatible units, workload definitions and recorded settings, including CPU allocation and scaling budgets. Different allocations are not silently treated as equivalent. **Clear view** clears the displayed runs while preserving the saved file. Save/load errors appear on the results page.

Detected component properties identify saved setups. RAM modules or timings not exposed by discovery cannot be distinguished reliably. Retesting can still compare against the results loaded when the app started.

## Build from source

The application uses **Rust and Slint** for the desktop UI. GPU execution uses **raw Vulkan through `ash`**, with embedded SPIR-V kernels. A separate worker process runs discovery and measurements.

The workspace requires Rust 1.95 or newer. `rust-toolchain.toml` selects stable Rust, Clippy, rustfmt and the Windows GNU LLVM target. The packaging wrapper uses LLVM-MinGW for its compiler, linker and static runtime settings.

Install the prerequisites in PowerShell, then reopen your terminal:

```powershell
winget install --id Rustlang.Rustup --exact
winget install --id MartinStorsjo.LLVM-MinGW.MSVCRT --exact
```

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\cargo.ps1 build --workspace --locked
powershell -ExecutionPolicy Bypass -File .\scripts\cargo.ps1 test --workspace --locked
powershell -ExecutionPolicy Bypass -File .\scripts\cargo.ps1 clippy --workspace --all-targets -- -D warnings
powershell -ExecutionPolicy Bypass -File .\scripts\cargo.ps1 run -p gluj-bench-ui
```

Wrapper builds place the executables in `target/x86_64-pc-windows-gnullvm/debug/`. Build the workspace first so the worker is beside the UI. The first build downloads dependencies. Precompiled SPIR-V files are included; ordinary builds do not require a shader compiler.

VS Code provides a **Gluj-Bench UI (Debug)** launch configuration that builds the workspace before starting the UI. Install the recommended rust-analyzer and CodeLLDB extensions for that workflow.

### Package version 0.2.0

```powershell
.\scripts\package-release.ps1 -Version 0.2.0
```

The script verifies the Cargo version, builds the release workspace, and creates the Windows x64 ZIP and SHA-256 checksum under `dist/`. Pass `-SkipBuild` only when the matching release binaries have already been built. The Windows icon and version metadata are embedded in the UI executable; no separate icon installation is required.

### Workspace layout

| Package | Responsibility |
| --- | --- |
| `gluj-bench-core` | Devices, benchmark definitions, results, sampling metadata, cancellation and protocol |
| `gluj-bench-cpu` | Windows CPU topology, pinned compute/bandwidth kernels and CPU scaling profiles |
| `gluj-bench-gpu` | Vulkan discovery, compute/matrix kernels, memory profiles and scaling analysis |
| `gluj-bench-worker` | Isolated benchmark process, hardware discovery, CLI and JSON request host |
| `gluj-bench-ui` | Slint interface, settings, graphs, result persistence and worker management |

### Worker CLI

```powershell
.\gluj-bench-worker.exe devices --json
.\gluj-bench-worker.exe benchmarks --json
.\gluj-bench-worker.exe run cpu.performance.avx2.f32_fma.scaling --json
.\gluj-bench-worker.exe --stdio
```

The worker's standard-I/O interface uses versioned newline-delimited JSON requests, progress responses and cancellation. UI settings are sent as run options; direct CLI runs use backend defaults and do not read the UI's `settings.json`. See [the protocol documentation](docs/protocol.md) for the request format.

## Project information

Gluj-Bench is actively developed. Measurements depend on the workload, driver, background activity, power state and configuration; benchmark definitions can change between versions. Compare matching configurations and review the recorded context when interpreting a result.

For changes, see [CHANGELOG.md](CHANGELOG.md). Report reproducible issues through [GitHub Issues](https://github.com/githubdood21/Gluj-Bench/issues), and follow [SECURITY.md](SECURITY.md) for security reports.

Gluj-Bench is licensed under the [MIT License](LICENSE) and provided without warranty.
