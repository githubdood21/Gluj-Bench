# Gluj-Bench 0.5.0-rc.1

Local release candidate, 2026-11-03. [Qualification evidence and limits](phase-9-qualification.md).
This candidate is qualified on the available Ryzen 9 5950X / Radeon RX 7900 XTX;
Intel/hybrid CPUs, integrated GPUs and other GPU vendors/drivers remain qualification gaps.

## User-visible changes

- Fifteen workload families, with precision, placement, shape and access choices inside
  each family. Cache-oriented and Streaming / VRAM remain distinct local-memory modes.
- Extensive coverage by default, an optional Shorter run, raw observations, descriptive
  spread and sustained conditions. Performance variation never generates rerun advice.
- Explicit seeded access randomness; sequential custom defaults and separate equally
  covered sequential/shuffled curated conditions.
- Configurable legacy CPU workloads, GPU FP16/FP32/FP64 buffers, native matrix tiles,
  memory sizes/sweeps and directional transfers with ring/submission controls.
- Selected-dataset medians, latest/typical summaries, compatibility explanations,
  named/pinned native history, raw observation browsing, archive/delete and scoped exports.
- Task preparation, reviewable plans, per-condition skip/cancel, persistent partial-plan
  status and a reachable global Stop. Worker disconnects recover through Reconnect engine.
  Cancelling one condition preserves pending status for work that continues; global
  Stop cancels the active condition and skips unstarted work.
- Optional GLUJ_BENCH_DATA_DIR isolates portable profiles or qualification data while
  retaining executable pairing. No environment override keeps the existing data location.

## Definitions and migration

Worker protocol remains 2; native run and PNG formats remain 1. New fields have
backward-compatible defaults. Old IDs/revisions and original files are preserved.
Legacy summaries cannot claim raw coverage. Imported runs remain comparison sources.
Kernel, access and traffic revisions gate comparisons; old archives are not rescaled.

Copy summaries use useful payload, with accounted read/write traffic separate.
GPU host-copy mean request cost is batch wall time / completed copies, not an
individual-request latency percentile. Cache transitions are inferred size intervals
with heuristic drop strength, not probability confidence. Dense GPU matrix tiles
are not global GEMM or equivalent to the CPU full-matrix workload.

The buffered INT8 initializer/reference mismatch found during qualification is fixed;
tile-streaming-4-signed-input records the new explicit signed/unsigned input pattern.
CPU/GPU memory and pointer-chain results now expose their existing validation scope.
System information distinguishes installed RAM from OS-visible capacity.

PNG export defaults to checked summaries, with explicit include-all. Use native JSON
for complete selected raw records. Limits are explicit (64 MiB PNG / 256 MiB native);
no samples are silently stripped. Names/notes/pins/archive state are separate from
immutable measurements. No automatic 128-configuration eviction remains.

## Package and source

Keep UI and worker together in a writable folder. Read build-info.json for exact
source/toolchain provenance. Local qualification data and user settings are excluded.
The repository working tree contains the corresponding source; this package does not
imply that the uncommitted candidate has been published on GitHub.
