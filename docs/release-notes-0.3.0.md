# Gluj-Bench 0.3.0

This release adds CPU cache and RAM latency measurements, configurable GPU RAM-offload testing, and clearer interactive results. It also introduces GPLv3 licensing and a separate branding policy.

**CPU cache and RAM latency**

- Added separate L1, L2 and L3 cache read-latency tests using warmed dependent reads on one pinned CPU core. Working sets use that core's cache instance rather than combined cache capacity across processor complexes.
- Added two RAM tests: **random-object latency** for widely scattered pointer reads, and **localized read latency** for reads randomized within 64 KiB blocks to reduce address-translation pressure.
- Results show nanoseconds per access, sample variation and compatible saved comparisons. Lower latency is better. These are observed access times under each test's conditions, rather than advertised RAM timings or exact hardware cache-hit counts.

**GPU RAM offload and scaling**

- Added an FP32 GPU compute-scaling test with configurable **50%, 75% or 100% system-RAM placement**, with the remaining data in VRAM.
- Added placement details, nominal host-traffic graphs, allocation safeguards and output-validation checks.
- Added measured GPU execution and end-to-end time per dataset pass for supported scaling tests, replacing the previous busy/wait percentage model.
- Added saved global scaling defaults and per-test overrides, supporting automatic sweeps, user-sized sweeps and single-dataset runs.
- Improved FP32 scaling synchronization, buffer-limit handling and cleanup after partial Vulkan setup failures.

**Clearer results and responsive graphs**

- Graphs now support exact-value hover inspection, guides to both axes, unit-labelled Y-axis values and compact tooltips.
- Measurement and tuning details now use clearer cards, comparison bars, sample-spread displays and expandable technical explanations.
- UI redraws are capped at **60 FPS normally** and **30 FPS while the worker is busy**. Idle windows remain event-driven; benchmark workloads and measurement timing are unaffected by this UI cap.

**License and branding**

- Changed this revision to **GPL-3.0-only** and added a separate policy requiring distinct product branding for independently distributed forks.
- Release packages include the license, project notice and branding policy. Earlier MIT releases retain their original permissions.

Validated with **122 passing tests**, strict Clippy checks and a smoke test of the packaged Windows x64 worker.

[Changes covered by this release note](https://github.com/githubdood21/Gluj-Bench/compare/ba6436a...1561191): `44a87d7`, `b4e9b5f`, `0b40d3b`, and `1561191`.
