# FP32 small-dataset investigation — 2026-10-19

Hardware: the local Radeon RX 7900 XTX / Ryzen 9 5950X system. These observations do not establish results for other GPU vendors.

## What 288 KiB actually launches

The FP32 profile assigns one invocation to two vec4 input reads and one vec4 output write: 48 bytes per invocation. 288 KiB therefore contains 6,144 invocations. The original 256-thread launch uses 24 workgroups, split 18 + 6 for 75% RAM offload. Sixty-four arithmetic iterations execute 25,165,824 counted FP32 operations per complete pass. Passes reuse their output and retain a write-after-write memory dependency.

## Placement control, original 256-thread launch

All cases below have the same aggregate allocation, arithmetic, synchronization and timing method. Median GPU times include dispatch and memory-dependency costs; they do not measure bare memory-access latency.

| 288 KiB placement | GPU time/pass | Throughput |
| --- | ---: | ---: |
| All VRAM | 3.85 µs | 6.54 TOPS |
| Both inputs in RAM, output in VRAM | 10.90 µs | 2.31 TOPS |
| Both inputs in VRAM, output in RAM | 7.54 µs | 3.34 TOPS |
| All RAM | 14.68 µs | 1.71 TOPS |

The separate mixed-region control measured 75% RAM at 11.71 µs / 2.15 TOPS with 256-thread groups. Thirty-two-thread groups measured 11.35 µs / 2.22 TOPS. This is a modest launch-shape improvement, not the 6.5+ TOPS achieved by changing placement to VRAM.

## Does the host cost stay constant?

The following direction controls use 32-thread groups throughout, 64 arithmetic iterations, warmup excluded, and nine samples per case:

| Aggregate dataset | All VRAM | RAM inputs only | RAM output only | All RAM |
| --- | ---: | ---: | ---: | ---: |
| 288 KiB | 3.33 µs | 10.45 µs | 6.97 µs | 14.13 µs |
| 1.97 MiB | 6.90 µs | 53.58 µs | 32.68 µs | 78.91 µs |
| 63.98 MiB | 126.51 µs | 1,583.08 µs | 846.89 µs | 2,205.58 µs |

Both reads and writes contribute. Costs grow with bytes, so the difference is not just fixed launch latency. This does not separate PCIe transfer time, cache behavior, GPU scheduling, or memory-dependency effects. Host traffic remains nominal; no physical PCIe counters were collected. The controls retain output synchronization and do not enlarge or duplicate data to inflate throughput.

## Changes based on the evidence

- Probe 32/64/128/256-thread launch shapes on the smallest and up-to-2-MiB datasets. Bracket the probes with the original launch, reject noisy/drifting comparisons, and require more than 2% aggregate improvement. Keep the selected shape fixed during the measured sweep. Record the probes and selected size.
- Preserve allocation alignment, arithmetic per value, physical placement and output dependencies. Register references remain separate measurements.
- Legacy paced results retain their active-GPU timing domain. Configured traffic now uses fixed-rate arrivals independent of completion, with eight physical Vulkan requests in flight and further arrivals tracked as logical backlog. Main graphs use completed average compute/traffic. TOPS retains the measured compute reference; requested host traffic is dashed on the traffic graph. Request completion delay and oldest pending age expose overload; GPU service time remains diagnostic. Pending includes submitted and not yet submitted requests at cutoff. Drain earns no throughput credit; samples start empty. Host polling and submission contribute to request delay, so it is not a pure memory-latency counter.
- Display GPU time per pass with the throughput comparison. It includes computation, memory accesses, dispatch and synchronization; the UI does not call the whole interval pure memory latency.

## Validation after launch tuning

The smallest-dataset tuning smoke selected 32-thread groups for both placement cases. The bracketed 256-thread VRAM control was approximately 6.58 TOPS; the selected launch measured 7.59 TOPS. For 75% RAM the bracketed control was approximately 2.14 TOPS and the selected launch measured 2.22 TOPS. The RAM path remains the dominant difference; launch tuning does not turn RAM performance into VRAM performance. At 1.97 MiB a separate paced validation retained 256 threads because the measured candidate improvement was below the 2% threshold.

Workgroup specialization uses the Vulkan 1.1-compatible WorkgroupSize specialization constant representation. A shader test rejects LocalSizeId, which would require maintenance4 support to be enabled. See the [Khronos specialization documentation](https://docs.vulkan.org/spec/latest/chapters/pipelines.html).

## Reported start after saturation calibration

Normal FP32 VRAM and RAM-offload runs now use hidden, uncapped VRAM probes of the same kernel up to 200 MiB within the allocation budget and any explicit safety cap. The earliest repeatable sample within 3% of the best stable median defines the reported start. The calibration owner is released before allocating the reported test, and its launch shape is retained. The smaller probes appear only in diagnostic metadata, not as result metrics, graph points, or selectable datasets. A requested target below the start is raised within the existing allocation limit. An early plateau can therefore keep a smaller start on an iGPU; this behavior is covered by selection-policy tests, not an iGPU hardware run.

The local hardware check selected about 64 MiB. The VRAM sweep then reported 41.44, 42.35 and 43.24 TOPS at approximately 64, 128 and 256 MiB. The matched VRAM dataset reference is used in the result bar, chart and summary; the separate register reference remains recorded. If throughput is still rising at the probe limit, or peak samples are noisy, the UI describes the result as a best observed reference and records the uncertainty. Neither the selected size nor the probe range establishes L2 capacity.

## Reproduce

Run this PowerShell command for launch-size/arithmetic controls:

```powershell
& ./scripts/cargo.ps1 @('test','-p','gluj-bench-gpu','--locked','profile_small_dataset_diagnostic','--','--ignored','--nocapture')
```

Replace the test filter with `profile_host_direction_diagnostic` for read/write placement controls, or `profile_launch_tuning_smoke` for the selected launch. Run GPU diagnostics sequentially. Buffer-placement masks are diagnostic controls, not new user benchmark modes.
