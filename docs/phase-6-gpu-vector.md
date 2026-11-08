# Phase 6 — GPU vector precision, access and demand

Implemented in the working tree on 2026-11-01. GPU vector compute remains one
family. Existing FP16/FP64 IDs retain register mode and gain buffered choices;
FP32 retains its unified profile ID and direct system-RAM placement mode.

## Precision and placement

Buffered kernels store four-lane vectors in native binary16, binary32 or binary64.
Three arrays (two inputs and an output) occupy 24, 48 or 96 bytes per invocation.
Each arithmetic loop executes eight independent four-lane FMAs: 64 operations,
counting multiplication and addition separately. Allocation and effective traffic
include both reads and the write. Indexing, loads, stores and synchronization are
inside GPU timing, but are not counted as arithmetic operations.

FP16 rounds every FMA and the final left-associated sum to binary16. Its recurrence
uses representable coefficients that remain finite through the supported 1–1024
loops; it still reads both input arrays. FP32 and FP64 use their native arithmetic.
Untimed boundary/interior readbacks from every active region and dataset use
independent precision-specific scalar references. Buffered relative tolerances are
0.3% for FP16, 0.01% for FP32 and 1e-10 for FP64. This is sampled validation, not a
claim of full-buffer validation. Register kernels retain their existing definitions.

Buffered FP16 requires both shaderFloat16 and storageBuffer16BitAccess; register
FP16 requires arithmetic support. FP64 requires shaderFloat64. Buffered shader execution requires Vulkan 1.1 or newer. Vulkan 1.1 FP16
execution enables VK_KHR_shader_float16_int8; Vulkan 1.2 promotes that extension.
Domains/capabilities and UI presets keep unsupported buffered modes unavailable.
FP16/FP64 do not advertise direct RAM placement or offered-load controls. FP32
supports 50%, 75% and 100% direct host placement with independently bounded RAM
and VRAM regions and exact aligned splits. Host placement remains shader access
to RAM, rather than a timed staging-copy benchmark.

The Vulkan feature distinction follows the primary specification:
[FP16 arithmetic](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderFloat16Int8Features.html)
and [16-bit buffer storage](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDevice16BitStorageFeatures.html).

## Access conditions and seed coverage

Customized runs remain sequential by default. Curated plans separately measure
sequential and shuffled blocks, each with three realizations and the full policy
coverage. Explicit GPU choices are:

- Sequential: contiguous invocation-to-vector mapping.
- Shuffled blocks: a seeded affine bijection of 256-vector blocks within each
  descriptor chunk; adjacent lanes stay adjacent within a block.
- Rotating regions: shift the first block for successive dispatches. Recorded
  batch iterations rotate; offered requests use cyclic slot-specific starts.
- Lane scattering: the block permutation plus a seeded odd affine permutation
  of the 256 vector lanes; this changes coalescing separately from block order.

Every mapping covers the same complete dataset per pass and gives each invocation
exclusive output ownership. No access-schedule buffer is allocated. Block grain
is fixed at 256 four-lane vectors (2/4/8 KiB per binding by precision), rather than
reusing CPU block-size controls. Arithmetic perturbation remains tied to physical
vector index within each descriptor chunk, so traversal does not change useful values or operation counts.
Integer parameter bounds prevent shader index multiplication overflow. Chunks
and memory regions preserve ownership and alignment independently.

Fresh roots resolve once per run. Native records retain the root, derived seed,
dataset and realization, generator revision, raw samples, and within-realization
statistics. Affine traversal is not Fisher–Yates sampling. Rotating a complete pass
changes address order without guaranteeing eviction; no hardware cache-miss,
physical DRAM or PCIe-counter claim is made.

## Calibration, sustained execution and time

Launch selection uses the common reference sampling/warmup budget, records all
probes and selected 32/64/128/256-thread shape, and checks the repeated baseline.
Same-precision, same-access, same-kernel VRAM saturation probes record repeatability, a rising
boundary and the selected peak. A single chosen dataset bounds its calibration
range and is never enlarged to the detected peak. Sweeps retain independently
measured points with eight intervals per doubling. Unstable references remain
measured limitations in metadata and never suggest rerunning.

GPU controls offer 15/60-second sustained coverage (worker range
0–120 seconds). Each dataset and realization receives its full sample count and
measurement-duration floor. Uncapped throughput uses GPU timestamp service time;
wall time separately includes host submission/wait and workload pacing. Offered
load uses observation-window time and does not apply activity pacing. Raw offsets
and scoped first/last-third means retain the chronology; no thermal cause is inferred.
Long runs and cross-device runtime calibration remain Phase 9 qualification work.

## Offered load and completion distributions

FP32 host placement retains independently scheduled arrivals and eight physical
submission slots. More arrivals accumulate in the logical backlog. The result
reports requested/delivered work, fulfillment, offered/submitted/completed counts,
submitted and unsubmitted pending requests, scheduled request and actual observation
windows, and oldest/mean/youngest pending age at cutoff. Drain work after cutoff
receives no throughput or completion credit. Each observation starts empty.

Completion delay is host-observed arrival-to-completion time, including submission,
queueing and polling. GPU service time remains separate. Native realization fields
retain each window's completed-only delay observations. Dataset details show pooled
nearest-rank p50/p90 only with at least 20 completions, p95 with 40, and p99 with 100.
The completion count is shown; unavailable percentiles are omitted. Pending requests
are excluded from these percentiles and remain explicitly visible in counts and
ages. These thresholds are display eligibility rules, not confidence intervals or
claims that a percentile has converged. Summary quantiles pool completed requests
across windows and equal-coverage realizations, not quantiles of window means.

## Shader reproduction and validation

The three buffered GLSL shaders and embedded SPIR-V are versioned together.
`profile-build.json` records glslang 16.6.0, Vulkan 1.1 / SPIR-V 1.3 target, validation
command and SHA-256 source/binary hashes. Buffered results embed that build manifest.
Ordinary application builds continue to use the embedded binaries.

From the repository root, run:

```powershell
./scripts/compile-profile-shaders.ps1 -Compiler C:/path/to/glslang.exe
./scripts/compile-profile-shaders.ps1 -Compiler C:/path/to/glslang.exe -Check
```

Generation invokes `--spirv-val`; Check regenerates in the target scratch directory
and compares binaries and the compiler/hash manifest without modifying checked-in
artifacts. Compiler upgrades require regeneration and review of the manifest.

Local validation includes workspace tests and strict Clippy, a live GPU fixture
covering 192 precision/placement/access/loop/launch combinations, and worker/desktop
checks recorded in the roadmap. The fixture validates whole-pass ownership and
sampled numeric output; it does not replace cross-vendor qualification.

Validation on 2026-11-01: 291 workspace tests passed (24 intentionally ignored),
strict Clippy, formatting, release build and shader reproduction passed. The explicit
live fixture validated 192 configurations on an AMD Radeon RX 7900 XTX. Twenty-seven
release-worker cases passed, including Extensive coverage, dynamic demand, overload,
three-realization ensembles, sustained FP16/FP64 and partial offered cancellation.
Each saved native record matched the terminal record. The overload fixture retained
384 completed-delay observations alongside a sample-median 36 pending requests
(8 submitted, 28 unsubmitted) and 64% fulfillment. These are local fixture outcomes,
not performance guarantees. Desktop review exercised buffered FP16, restored custom
size/access, the paired Curated preview, and the actual overload record showing
completed-only percentiles alongside pending counts and ages. The arrival-rate note
reports the selected demand without recommending another run. Small offered-workload warmup now uses
calibrated batches to satisfy its budget while request measurement remains unbatched.
