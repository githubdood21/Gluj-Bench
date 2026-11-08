# Phase 3: family catalogue and explicit plans

Date: 2026-10-31

The picker now shows fifteen families across the four existing categories. Every implemented runner remains a distinct mode/condition, with its own units, kernel identity, settings and result. Family membership comes from advertised condition descriptors, with the Phase 1 compatibility mapping as a fallback. Precision/instruction path, memory target, latency locality and codec direction select modes within a family; existing supported dataset/placement/worker controls stay with their mode.

## Execution choices

- **Run configured mode** runs the selected mode and its effective settings.
- **Run curated family** creates an explicit capability-checked plan for that family.
- **Run curated category** plans all implemented supported modes in that category.
- **Add configured mode** snapshots settings and the selected adapter into a session custom plan. Add more configured modes, inspect the plan, then use **Run custom plan**. Clearing this plan does not remove history.

Curated CPU performance plans cover one and all physical cores, scalar/SIMD arithmetic, and separate compression/decompression. Supported register and buffered modes are separate conditions. Memory plans retain cache-level/RAM targets and latency localities; each bandwidth runner still measures its read/write/copy directions. GPU vector plans preserve FP16/FP32/FP64 and FP32 direct host-memory placement, and matrix plans preserve supported FP16/INT8 modes. Direct host-memory compute and staging transfers remain different families. Vector and matrix compute remain different families.

Identical resolved configurations are executed only once, including one-core machines where one/all-core settings coincide. Resolution checks advertised per-mode capabilities before execution. A register-capable device without buffered storage support retains register execution; unsupported data presets are hidden. Unimplemented FP8/sparse entries appear in capability/implementation details, not runnable cards or queues. Unsupported real modes have explanations and do not enter curated plans.

Plans use the shared `SuitePlan` contract, stable catalogue order, resolved device/configuration, intended coverage, estimates and pending/running/terminal state. A worker-reported condition failure stops remaining queued work; unstarted conditions are skipped. Plans execute serially. The global Extensive/Shorter policy applies consistently to every queued condition.

## Preview and estimates

Before execution, previews show logical condition count, a minimum measurement-time estimate, allocation estimates where available, and inspectable planned conditions. Condition details identify the device, physical workers or a single pinned latency worker, and aggregate/per-worker dataset targets before alignment. A single logical sweep condition can contain many dataset measurements: its preview duration is a **lower bound**, not a total ETA. Preparation, warmup, references/calibration and validation add time. Phase 9 remains responsible for calibrated end-to-end estimates.

Memory figures are planning estimates for test buffers, not process RSS or guaranteed memory caps. Bandwidth providers advertise their buffer estimates. CPU register/algorithm estimates account for selected workers. Explicit buffered compute includes a conservative 512 MiB auxiliary allowance; GPU matrix operands additionally reserve an estimated output allowance. These figures exclude driver/setup overhead. Modes with dynamically selected or unadvertised footprints show an explicit unavailable/partially estimated indication. Actual aligned allocations remain recorded by the runner; estimates do not rewrite actual results.

Expensive plan/preset previews are not rebuilt on each progress refresh while a benchmark is active.

## INT64 compatibility and saved settings

The legacy `cpu.performance.single_thread.integer.i64` request validates its original option contract, resolves to `cpu.performance.integer.i64` with one physical core, register preset and register dataset mode, then executes the canonical harness. The native record retains the requested legacy ID and the resolved canonical ID. Run-record validation permits this specific alias and still rejects unrelated benchmark/device mismatches. Old result archives are not rewritten or declared equivalent without provenance.

Version-1 per-test preferences remain readable. An additive `family_modes` map remembers each family's selected mode. Per-runner custom settings stay intact when switching modes/families. Legacy single-thread overrides migrate to one-core register settings; an existing canonical INT64 preference takes precedence. A saved mode unsupported on the selected adapter falls back to a supported mode while retaining the preference for future devices.

The existing hardware cache version is advanced so new capability/estimate metadata is refreshed. Custom plans are session state in this phase; persistent editable plan management belongs to later workflow/history work.

## Separate result configurations

Result grouping always requires compatible recorded settings, including worker count. Scalar one/all-core runs therefore produce separate rows. The old latest-three-configurations display cap is removed: register/buffer and one/all-core conditions from one family plan must all remain visible. Repeated equivalent configurations still share latest/best statistics. The compact results cache keeps its existing 128-entry bound; immutable native run history remains the archive.

## Validation

Automated regressions cover all fifteen families and all thirty implemented catalogue destinations, one/all-core scalar/SIMD and codec-direction coverage, exact-condition deduplication, register-only GPU capabilities, mode fallback, version-1 preference migration, alias resolution/record validation, plan terminal state and separate scalar/configuration result rows.

Desktop checks used an isolated executable/settings/history folder. Family selection, scalar/SIMD switching, scrollable previews and custom-plan addition worked. A real FP64 family plan executed scalar and AVX2 once each on one and sixteen physical cores, producing four complete native records. Release alias execution and final build/check results are recorded in the implementation plan.

## Later workload changes

This phase does not introduce access randomness. Phase 4 adds it as an explicit setting: customized tests default to sequential; curated suites measure supported sequential and randomized conditions separately with equal coverage. Unsupported precision/data combinations remain unavailable until their implementation phases.
