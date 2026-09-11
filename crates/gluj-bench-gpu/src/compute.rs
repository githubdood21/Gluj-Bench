use super::{
    AdapterRecord, GPU_PRECONDITION_MS, WORKGROUP_SIZE, common_metadata, device_metadata, done,
    duration_ns, ensure_not_cancelled, insert_clock_policy, per_sample_duration, statistics,
};
use crate::vulkan_vector::VulkanVectorHarness;
use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult,
    CancellationToken, Metric, ProgressCallback, ProgressUpdate,
};
use std::{collections::BTreeMap, time::Instant};

const FP32_ID: &str = "gpu.performance.fp32";
const FP16_ID: &str = "gpu.performance.fp16";
const FP64_ID: &str = "gpu.performance.fp64";
const LOOP_COUNT: u32 = 1024;
const PROBE_LOOP_COUNT: u32 = 2048;
const AUTOTUNE_LOOP_COUNT: u32 = 128;
const OPERATIONS_PER_LOOP_PER_INVOCATION: u64 = 64;
const COMPUTE_BOUND_RATIO: f64 = 0.80;
const WORKGROUP_CANDIDATES: [u32; 5] = [512, 1024, 2048, 4096, 8192];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ComputeKind {
    Fp16,
    Fp32,
    Fp64,
}

impl ComputeKind {
    fn id(self) -> &'static str {
        match self {
            Self::Fp16 => FP16_ID,
            Self::Fp32 => FP32_ID,
            Self::Fp64 => FP64_ID,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "FP16 vector shader performance",
            Self::Fp32 => "FP32 vector shader performance",
            Self::Fp64 => "FP64 vector shader performance",
        }
    }

    fn data_type(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
            Self::Fp64 => "fp64",
        }
    }

    fn supported_by_vulkan(self, record: &AdapterRecord) -> bool {
        match self {
            Self::Fp16 => record.vulkan.shader_float16,
            Self::Fp32 => true,
            Self::Fp64 => record.vulkan.shader_float64,
        }
    }

    fn unsupported_reason(self) -> &'static str {
        match self {
            Self::Fp16 => "vulkan_shader_float16_unsupported",
            Self::Fp32 => "",
            Self::Fp64 => "vulkan_shader_float64_unsupported",
        }
    }

    fn accumulator_chains(self) -> u32 {
        if self == Self::Fp16 { 16 } else { 8 }
    }
}

const KINDS: [ComputeKind; 3] = [ComputeKind::Fp16, ComputeKind::Fp32, ComputeKind::Fp64];

pub(super) fn kind(id: &str) -> Option<ComputeKind> {
    KINDS.into_iter().find(|kind| kind.id() == id)
}

pub(super) fn descriptors(adapters: &[AdapterRecord]) -> Vec<BenchmarkDescriptor> {
    KINDS
        .into_iter()
        .enumerate()
        .map(|(index, kind)| {
            let supported_device_ids = adapters
                .iter()
                .filter(|record| {
                    record.supports_vulkan_timestamps() && kind.supported_by_vulkan(record)
                })
                .map(|record| record.id.clone())
                .collect::<Vec<_>>();
            let available = !supported_device_ids.is_empty();
            let unavailable_reason = if available {
                ""
            } else if adapters
                .iter()
                .any(AdapterRecord::supports_vulkan_timestamps)
            {
                kind.unsupported_reason()
            } else if adapters.is_empty() {
                "adapter_not_found"
            } else {
                "timestamp_query_unsupported"
            };
            let mut metadata = BTreeMap::new();
            metadata.insert(
                "instruction_class".into(),
                "vector_fused_multiply_add".into(),
            );
            metadata.insert("execution_domain".into(), "vector_shader".into());
            metadata.insert("api".into(), "Vulkan".into());
            metadata.insert("shader_format".into(), "embedded_spirv".into());
            metadata.insert("operation_counting".into(), operation_definition().into());
            BenchmarkDescriptor {
                id: kind.id().into(),
                name: kind.name().into(),
                category: BenchmarkCategory::Gpu,
                workload: "Register-resident vector fused multiply-add throughput".into(),
                data_type: kind.data_type().into(),
                unit: "operations/s".into(),
                supported_device_ids,
                available,
                unavailable_reason: unavailable_reason.into(),
                suite_id: "gpu.performance".into(),
                display_order: 100 + index as u32,
                metadata,
            }
        })
        .collect()
}

pub(super) fn run(
    record: &AdapterRecord,
    benchmark_id: &str,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
) -> Result<BenchmarkResult, BenchmarkError> {
    let started = Instant::now();
    let kind = kind(benchmark_id).ok_or_else(|| {
        BenchmarkError::new(
            "unsupported_benchmark",
            format!("Unknown GPU compute benchmark '{benchmark_id}'."),
        )
    })?;
    if !kind.supported_by_vulkan(record) {
        return Err(BenchmarkError::new(
            kind.unsupported_reason(),
            format!(
                "{} is not supported by the selected Vulkan device.",
                kind.name()
            ),
        ));
    }
    ensure_not_cancelled(cancellation)?;
    progress(ProgressUpdate {
        fraction: 0.02,
        phase: "gpu_compute_setup".into(),
        message: format!("Creating the Vulkan {} pipeline", kind.data_type()),
    });
    let workgroups = autotune_workgroups(record, kind, cancellation)?;
    let harness = VulkanVectorHarness::new(record, kind, workgroups, LOOP_COUNT)?;

    progress(ProgressUpdate {
        fraction: 0.12,
        phase: "gpu_compute_precondition".into(),
        message: "Saturating GPU execution cores before measurement".into(),
    });
    let initial_ns = harness.measure(1)?;
    let warm_iterations = iterations_for_target(initial_ns, GPU_PRECONDITION_MS / 1000.0);
    let _ = harness.measure(warm_iterations)?;

    progress(ProgressUpdate {
        fraction: 0.20,
        phase: "gpu_compute_diagnosis".into(),
        message: "Checking arithmetic-intensity sensitivity".into(),
    });
    let probe = VulkanVectorHarness::new(record, kind, workgroups, PROBE_LOOP_COUNT)?;
    let primary_probe_rate = measure_rate_for_target(&harness, 0.20)?;
    let higher_intensity_rate = measure_rate_for_target(&probe, 0.20)?;
    let sensitivity_ratio = (primary_probe_rate / higher_intensity_rate).clamp(0.0, 2.0);

    let sample_count = config.samples.max(1);
    let target_per_sample = per_sample_duration(config).as_secs_f64();
    let calibrated_ns = harness.measure(1)?;
    let sample_iterations = iterations_for_target(calibrated_ns, target_per_sample);
    let mut values = Vec::with_capacity(sample_count as usize);
    for sample in 0..sample_count {
        ensure_not_cancelled(cancellation)?;
        progress(ProgressUpdate {
            fraction: 0.25 + sample as f64 / sample_count as f64 * 0.70,
            phase: "gpu_compute_measurement".into(),
            message: format!(
                "Measuring {} Vulkan throughput sample {}/{}",
                kind.data_type(),
                sample + 1,
                sample_count
            ),
        });
        let elapsed_ns = harness.measure(sample_iterations)?;
        values.push(
            harness.operations_per_dispatch() as f64 * sample_iterations as f64
                / (elapsed_ns / 1e9),
        );
    }
    let sample_statistics = statistics(&values);
    let implied_output_bandwidth = sample_statistics.median * 16.0
        / (LOOP_COUNT as f64 * OPERATIONS_PER_LOOP_PER_INVOCATION as f64);
    let mut metadata = common_metadata(record, "vulkan_gpu_timestamp");
    insert_clock_policy(&mut metadata);
    metadata.insert("data_type".into(), kind.data_type().into());
    metadata.insert(
        "instruction_class".into(),
        "vector_fused_multiply_add".into(),
    );
    metadata.insert("operation_definition".into(), operation_definition().into());
    metadata.insert("operations_per_loop_per_invocation".into(), "64".into());
    metadata.insert("loop_count".into(), LOOP_COUNT.to_string());
    metadata.insert(
        "independent_accumulator_chains".into(),
        kind.accumulator_chains().to_string(),
    );
    metadata.insert("selected_workgroups".into(), workgroups.to_string());
    metadata.insert(
        "selected_invocations".into(),
        (workgroups as u64 * WORKGROUP_SIZE).to_string(),
    );
    metadata.insert("api".into(), "Vulkan".into());
    metadata.insert("execution_backend".into(), "raw-vulkan".into());
    metadata.insert("shader_format".into(), "embedded_spirv".into());
    metadata.insert("shader_source_language".into(), "GLSL".into());
    metadata.insert("vector_kernel_revision".into(), "vulkan-packed-1".into());
    metadata.insert(
        "vector_operand_shape".into(),
        match kind {
            ComputeKind::Fp16 => "f16vec2 packed pairs",
            ComputeKind::Fp32 => "vec4",
            ComputeKind::Fp64 => "dvec4",
        }
        .into(),
    );
    metadata.insert(
        "machine_isa_verified".into(),
        "false; SPIR-V type and operation semantics verified".into(),
    );
    metadata.insert(
        "native_acceleration".into(),
        match kind {
            ComputeKind::Fp16 => "explicit_f16vec2_packed_pair_workload",
            ComputeKind::Fp32 => "core_shader_instruction_set",
            ComputeKind::Fp64 => "vulkan_shader_float64_capability_reported",
        }
        .into(),
    );
    metadata.insert(
        "bound_classification".into(),
        classify(sensitivity_ratio).into(),
    );
    metadata.insert(
        "memory_sensitivity_ratio".into(),
        format!("{sensitivity_ratio:.4}"),
    );
    metadata.insert(
        "implied_output_bandwidth_bytes_per_second".into(),
        format!("{implied_output_bandwidth:.0}"),
    );
    metadata.insert("working_set_bytes".into(), harness.output_size.to_string());
    progress(done("Vulkan GPU compute benchmark completed."));
    let mut result_device_metadata = device_metadata(record);
    result_device_metadata.insert("execution_backend".into(), "raw-vulkan".into());
    Ok(BenchmarkResult {
        benchmark_id: kind.id().into(),
        device_id: record.id.clone(),
        elapsed_ns: duration_ns(started.elapsed()),
        metrics: vec![Metric {
            name: "throughput".into(),
            value: sample_statistics.median,
            unit: "operations/s".into(),
            statistics: sample_statistics,
        }],
        workload_metadata: metadata,
        device_metadata: result_device_metadata,
    })
}

fn classify(sensitivity_ratio: f64) -> &'static str {
    if sensitivity_ratio < COMPUTE_BOUND_RATIO {
        "memory_bandwidth_bound"
    } else {
        "compute_bound"
    }
}

fn operation_definition() -> &'static str {
    "one scalar fused multiply-add equals two operations per vector lane"
}

fn autotune_workgroups(
    record: &AdapterRecord,
    kind: ComputeKind,
    cancellation: &CancellationToken,
) -> Result<u32, BenchmarkError> {
    let mut best = None;
    for workgroups in WORKGROUP_CANDIDATES {
        ensure_not_cancelled(cancellation)?;
        let harness = VulkanVectorHarness::new(record, kind, workgroups, AUTOTUNE_LOOP_COUNT)?;
        let rate = measure_rate_for_target(&harness, 0.04)?;
        if best.is_none_or(|(_, best_rate)| rate > best_rate) {
            best = Some((workgroups, rate));
        }
    }
    best.map(|(workgroups, _)| workgroups).ok_or_else(|| {
        BenchmarkError::new(
            "compute_limits_unsupported",
            "The adapter cannot dispatch the minimum Vulkan saturation workload.",
        )
    })
}

fn measure_rate_for_target(
    harness: &VulkanVectorHarness,
    target_seconds: f64,
) -> Result<f64, BenchmarkError> {
    let trial_ns = harness.measure(1)?;
    let iterations = iterations_for_target(trial_ns, target_seconds);
    let elapsed_ns = harness.measure(iterations)?;
    Ok(harness.operations_per_dispatch() as f64 * iterations as f64 / (elapsed_ns / 1e9))
}

fn iterations_for_target(per_dispatch_ns: f64, target_seconds: f64) -> u32 {
    (target_seconds * 1e9 / per_dispatch_ns)
        .ceil()
        .clamp(1.0, 4096.0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_keep_optional_hardware_types_visible() {
        let descriptors = descriptors(&[]);
        assert_eq!(descriptors.len(), 3);
        assert!(descriptors.iter().all(|descriptor| !descriptor.available));
        assert_eq!(descriptors[0].id, FP16_ID);
        assert_eq!(descriptors[1].id, FP32_ID);
        assert_eq!(descriptors[2].id, FP64_ID);
        assert_eq!(descriptors[0].metadata.get("api").unwrap(), "Vulkan");
    }

    #[test]
    fn operation_accounting_is_consistent_across_types() {
        assert_eq!(OPERATIONS_PER_LOOP_PER_INVOCATION, 64);
        assert!(operation_definition().contains("two operations"));
        assert_eq!(ComputeKind::Fp16.accumulator_chains(), 16);
    }

    #[test]
    fn bound_classification_threshold_matches_cpu_suite() {
        assert_eq!(classify(0.79), "memory_bandwidth_bound");
        assert_eq!(classify(0.80), "compute_bound");
    }

    #[test]
    fn iteration_calibration_is_bounded() {
        assert_eq!(iterations_for_target(1_000_000.0, 1.0), 1000);
        assert_eq!(iterations_for_target(1.0, 10.0), 4096);
    }
}
