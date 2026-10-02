use super::{
    AdapterRecord, GPU_PRECONDITION_MS, WORKGROUP_SIZE, common_metadata, device_metadata, done,
    duration_ns, ensure_not_cancelled, insert_clock_policy, per_sample_duration, statistics,
};
use crate::vulkan_compute_profile::VulkanComputeProfileHarness;
use crate::vulkan_vector::VulkanVectorHarness;
use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult,
    CancellationToken, Metric, ProgressCallback, ProgressUpdate, SampleStatistics,
};
use std::{collections::BTreeMap, time::Instant};

const FP32_ID: &str = "gpu.performance.fp32";
const FP16_ID: &str = "gpu.performance.fp16";
const FP64_ID: &str = "gpu.performance.fp64";
const FP32_SCALING_ID: &str = "gpu.performance.fp32.scaling";
const PROFILE_LOOP_COUNT: u32 = 64;
const PROFILE_MIN_SIZE: u64 = 256 * 1024;
const PROFILE_DROP_RATIO: f64 = 0.90;
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

pub(super) fn is_benchmark(id: &str) -> bool {
    kind(id).is_some() || id == FP32_SCALING_ID
}

pub(super) fn descriptors(adapters: &[AdapterRecord]) -> Vec<BenchmarkDescriptor> {
    let mut descriptors = KINDS
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
        .collect::<Vec<_>>();
    let supported_device_ids = adapters
        .iter()
        .filter(|record| record.supports_vulkan_timestamps())
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();
    let available = !supported_device_ids.is_empty();
    descriptors.push(BenchmarkDescriptor {
        id: FP32_SCALING_ID.into(),
        name: "FP32 compute scaling profile".into(),
        category: BenchmarkCategory::Gpu,
        workload: "Memory-backed FP32 FMA throughput across increasing device-local working sets"
            .into(),
        data_type: "fp32".into(),
        unit: "operations/s".into(),
        supported_device_ids,
        available,
        unavailable_reason: if available {
            ""
        } else if adapters.is_empty() {
            "adapter_not_found"
        } else {
            "timestamp_query_unsupported"
        }
        .into(),
        suite_id: "gpu.performance".into(),
        display_order: 103,
        metadata: BTreeMap::from([
            (
                "execution_domain".into(),
                "memory_backed_vector_compute".into(),
            ),
            ("profile_axis".into(), "working_set_bytes".into()),
            (
                "arithmetic_intensity".into(),
                format!(
                    "{:.4} operations/byte",
                    PROFILE_LOOP_COUNT as f64 * 64.0 / 48.0
                ),
            ),
            ("shader_format".into(), "embedded_spirv".into()),
        ]),
    });
    descriptors
}

pub(super) fn run(
    record: &AdapterRecord,
    benchmark_id: &str,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
) -> Result<BenchmarkResult, BenchmarkError> {
    if benchmark_id == FP32_SCALING_ID {
        return run_fp32_scaling(record, config, cancellation, progress);
    }
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
    let warm_iterations = iterations_for_target(
        initial_ns,
        gluj_bench_core::gpu_burst_duration(std::time::Duration::from_secs_f64(
            GPU_PRECONDITION_MS / 1000.0,
        ))
        .as_secs_f64(),
    );
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

#[derive(Debug)]
struct ProfilePoint {
    working_set_bytes: u64,
    compute: SampleStatistics,
    bandwidth: SampleStatistics,
    retained_ratio: f64,
}

fn run_fp32_scaling(
    record: &AdapterRecord,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    callback: &mut ProgressCallback<'_>,
) -> Result<BenchmarkResult, BenchmarkError> {
    let started = Instant::now();
    ensure_not_cancelled(cancellation)?;
    let reference = {
        let mut reference_progress = |mut update: ProgressUpdate| {
            update.fraction *= 0.20;
            update.message = format!("Compute reference: {}", update.message);
            callback(update);
        };
        run(
            record,
            FP32_ID,
            &crate::scaling::reference_config(config),
            cancellation,
            &mut reference_progress,
        )?
    };
    let mut sweep_progress = |mut update: ProgressUpdate| {
        update.fraction = 0.20 + update.fraction * 0.75;
        callback(update);
    };
    let progress = &mut sweep_progress;
    let loop_count = config
        .options
        .get("arithmetic_iterations")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(PROFILE_LOOP_COUNT)
        .clamp(1, 1024);
    let budget_percent = gluj_bench_core::vram_budget_percent(config)?;
    let allocation_budget =
        gluj_bench_core::vram_budget_bytes(record.vulkan.device_local_memory_bytes, budget_percent);
    let maximum = (record.vulkan.max_storage_buffer_range.saturating_mul(3))
        .min(allocation_budget.saturating_sub(16 * 1024 * 1024));
    let requested_sizes = profile_sizes(maximum);
    if requested_sizes.len() < 3 {
        return Err(BenchmarkError::new(
            "insufficient_gpu_memory",
            "At least three working-set tiers are required for a compute scaling profile.",
        ));
    }
    progress(ProgressUpdate {
        fraction: 0.01,
        phase: "gpu_compute_profile_setup".into(),
        message: format!(
            "Allocating an FP32 profile working set up to {} MiB",
            maximum / (1024 * 1024)
        ),
    });
    let harness = VulkanComputeProfileHarness::new(record, maximum)?;
    let sizes = requested_sizes
        .into_iter()
        .map(|size| harness.actual_working_set(size))
        .collect::<Vec<_>>();
    let sample_count = config.samples.clamp(2, 5);
    let target_seconds =
        (config.target_duration_ms as f64 / 1000.0 / sizes.len() as f64 / sample_count as f64)
            .clamp(0.01, 0.10);
    let target_seconds =
        gluj_bench_core::gpu_burst_duration(std::time::Duration::from_secs_f64(target_seconds))
            .as_secs_f64();
    let arithmetic_intensity = loop_count as f64 * 64.0 / 48.0;
    let mut raw_points = Vec::with_capacity(sizes.len());
    for (index, working_set) in sizes.iter().copied().enumerate() {
        ensure_not_cancelled(cancellation)?;
        progress(ProgressUpdate {
            fraction: 0.04 + index as f64 / sizes.len() as f64 * 0.91,
            phase: "gpu_compute_profile_sweep".into(),
            message: format!(
                "Measuring FP32 compute with a {} working set ({}/{})",
                format_size(working_set),
                index + 1,
                sizes.len()
            ),
        });
        let _ = harness.measure(working_set, loop_count, 1)?;
        let calibration_ns = harness.measure(working_set, loop_count, 1)?;
        let iterations = iterations_for_target(calibration_ns, target_seconds);
        let operations = harness.operations_per_dispatch(working_set, loop_count) as f64;
        let traffic = harness.traffic_bytes_per_dispatch(working_set) as f64;
        let mut compute_values = Vec::with_capacity(sample_count as usize);
        let mut bandwidth_values = Vec::with_capacity(sample_count as usize);
        for _ in 0..sample_count {
            ensure_not_cancelled(cancellation)?;
            let elapsed_ns = harness.measure(working_set, loop_count, iterations)?;
            let seconds = elapsed_ns / 1e9;
            compute_values.push(operations * iterations as f64 / seconds);
            bandwidth_values.push(traffic * iterations as f64 / seconds);
        }
        raw_points.push((
            working_set,
            statistics(&compute_values),
            statistics(&bandwidth_values),
        ));
    }
    let (baseline, baseline_statistics) = raw_points
        .iter()
        .take(3)
        .max_by(|(_, left, _), (_, right, _)| left.median.total_cmp(&right.median))
        .map(|(_, compute, _)| (compute.median, compute.clone()))
        .unwrap_or_default();
    let points = raw_points
        .into_iter()
        .map(|(working_set_bytes, compute, bandwidth)| ProfilePoint {
            working_set_bytes,
            retained_ratio: (compute.median / baseline).clamp(0.0, 2.0),
            compute,
            bandwidth,
        })
        .collect::<Vec<_>>();
    let transition_index = sustained_transition(&points);
    let mut metrics = vec![Metric {
        name: "cache_resident_compute".into(),
        value: baseline,
        unit: "operations/s".into(),
        statistics: baseline_statistics,
    }];
    for point in &points {
        metrics.push(Metric {
            name: format!("working_set_{}.compute", point.working_set_bytes),
            value: point.compute.median,
            unit: "operations/s".into(),
            statistics: point.compute.clone(),
        });
        metrics.push(Metric {
            name: format!("working_set_{}.bandwidth", point.working_set_bytes),
            value: point.bandwidth.median,
            unit: "bytes/s".into(),
            statistics: point.bandwidth.clone(),
        });
    }
    if let Some(index) = transition_index {
        metrics.push(Metric {
            name: "bandwidth_transition_working_set".into(),
            value: points[index].working_set_bytes as f64,
            unit: "bytes".into(),
            statistics: statistics(&[points[index].working_set_bytes as f64]),
        });
    }
    let mut metadata = common_metadata(record, "vulkan_gpu_timestamp");
    insert_clock_policy(&mut metadata);
    metadata.insert("execution_backend".into(), "raw-vulkan".into());
    metadata.insert(
        "execution_domain".into(),
        "memory_backed_vector_compute".into(),
    );
    metadata.insert("shader_format".into(), "embedded_spirv".into());
    metadata.insert("shader_source_language".into(), "GLSL".into());
    metadata.insert(
        "compute_profile_revision".into(),
        "working-set-sweep-1".into(),
    );
    metadata.insert("data_type".into(), "fp32".into());
    metadata.insert("arithmetic_iterations".into(), loop_count.to_string());
    metadata.insert(
        "allocation_budget_bytes".into(),
        allocation_budget.to_string(),
    );
    metadata.insert(
        "allocated_test_buffer_bytes".into(),
        harness.maximum_working_set_bytes.to_string(),
    );
    metadata.insert("allocation_limit_note".into(), "The selected VRAM budget is limited by the device's storage-buffer range, with 16 MiB reserved for allocation overhead.".into());
    metadata.insert(
        "arithmetic_intensity_operations_per_byte".into(),
        format!("{arithmetic_intensity:.4}"),
    );
    metadata.insert("bytes_per_element".into(), "48".into());
    metadata.insert(
        "byte_definition".into(),
        "two_16_byte_input_reads_plus_one_16_byte_output_write".into(),
    );
    metadata.insert(
        "baseline_operations_per_second".into(),
        format!("{baseline:.0}"),
    );
    metadata.insert("tested_tier_count".into(), points.len().to_string());
    metadata.insert(
        "profile_sample_count_per_tier".into(),
        sample_count.to_string(),
    );
    metadata.insert(
        "transition_threshold".into(),
        "two consecutive low-noise tiers below 90 percent of the small-set baseline with no later two-tier recovery".into(),
    );
    metadata.insert(
        "bandwidth_transition_status".into(),
        if transition_index.is_some() {
            "observed"
        } else {
            "not_observed_within_tested_range"
        }
        .into(),
    );
    if let Some(index) = transition_index {
        metadata.insert(
            "bandwidth_transition_working_set_bytes".into(),
            points[index].working_set_bytes.to_string(),
        );
        metadata.insert(
            "bandwidth_transition_retained_ratio".into(),
            format!("{:.4}", points[index].retained_ratio),
        );
    }
    metadata.insert(
        "compute_scaling_points".into(),
        points
            .iter()
            .map(|point| {
                format!(
                    "{}:{:.0}:{:.0}:{:.4}",
                    point.working_set_bytes,
                    point.compute.median,
                    point.bandwidth.median,
                    point.retained_ratio
                )
            })
            .collect::<Vec<_>>()
            .join(","),
    );
    drop(harness);
    let check = {
        let mut check_progress = |mut update: ProgressUpdate| {
            // The final check occupies the last five percent of progress.
            update.fraction = 0.95 + update.fraction * 0.05;
            update.message = format!("Checking compute reference drift: {}", update.message);
            callback(update);
        };
        run(
            record,
            FP32_ID,
            &crate::scaling::reference_config(config),
            cancellation,
            &mut check_progress,
        )?
    };
    let reference = crate::scaling::validate_reference(reference, check);
    crate::scaling::add_reference_analysis(
        &mut metrics,
        &mut metadata,
        &reference,
        transition_index.is_some(),
    );
    callback(done("FP32 compute scaling profile completed."));
    let mut result_device_metadata = device_metadata(record);
    result_device_metadata.insert("execution_backend".into(), "raw-vulkan".into());
    Ok(BenchmarkResult {
        benchmark_id: FP32_SCALING_ID.into(),
        device_id: record.id.clone(),
        elapsed_ns: duration_ns(started.elapsed()),
        metrics,
        workload_metadata: metadata,
        device_metadata: result_device_metadata,
    })
}

fn profile_sizes(maximum: u64) -> Vec<u64> {
    let mut sizes = Vec::new();
    let mut size = PROFILE_MIN_SIZE;
    while size <= maximum {
        sizes.push(size);
        let Some(next) = size.checked_mul(2) else {
            break;
        };
        size = next;
    }
    if maximum >= PROFILE_MIN_SIZE && sizes.last().copied() != Some(maximum) {
        sizes.push(maximum);
    }
    sizes
}

fn sustained_transition(points: &[ProfilePoint]) -> Option<usize> {
    points.windows(2).enumerate().find_map(|(index, pair)| {
        let is_drop = pair[0].retained_ratio < PROFILE_DROP_RATIO
            && pair[1].retained_ratio < PROFILE_DROP_RATIO
            && significant_drop(&pair[0])
            && significant_drop(&pair[1]);
        let later_recovery = points[index + 2..].windows(2).any(|recovery| {
            recovery[0].retained_ratio >= PROFILE_DROP_RATIO
                && recovery[1].retained_ratio >= PROFILE_DROP_RATIO
        });
        (is_drop && !later_recovery).then_some(index)
    })
}

fn significant_drop(point: &ProfilePoint) -> bool {
    let relative_noise = if point.compute.median > 0.0 {
        point.compute.standard_deviation / point.compute.median
    } else {
        1.0
    };
    1.0 - point.retained_ratio > (relative_noise * 2.0).max(0.02)
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} KiB", bytes as f64 / 1024.0)
    }
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
        assert_eq!(descriptors.len(), 4);
        assert!(descriptors.iter().all(|descriptor| !descriptor.available));
        assert_eq!(descriptors[0].id, FP16_ID);
        assert_eq!(descriptors[1].id, FP32_ID);
        assert_eq!(descriptors[2].id, FP64_ID);
        assert_eq!(descriptors[3].id, FP32_SCALING_ID);
        assert_eq!(descriptors[0].metadata.get("api").unwrap(), "Vulkan");
        assert_eq!(
            descriptors[3].metadata.get("profile_axis").unwrap(),
            "working_set_bytes"
        );
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

    #[test]
    fn profile_sizes_are_power_of_two_tiers_within_the_limit() {
        assert_eq!(
            profile_sizes(2 * 1024 * 1024),
            vec![256 * 1024, 512 * 1024, 1024 * 1024, 2 * 1024 * 1024]
        );
        assert!(profile_sizes(PROFILE_MIN_SIZE - 1).is_empty());
    }

    #[test]
    fn profile_reaches_selected_budget_above_one_gib() {
        let maximum = 5 * 1024 * 1024 * 1024 - 16 * 1024 * 1024;
        let sizes = profile_sizes(maximum);
        assert_eq!(sizes.last(), Some(&maximum));
        assert!(sizes.contains(&(4 * 1024 * 1024 * 1024)));
        assert!(sizes.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn transition_requires_two_consecutive_significant_drops() {
        let point = |working_set_bytes, retained_ratio, noise| ProfilePoint {
            working_set_bytes,
            retained_ratio,
            compute: SampleStatistics {
                sample_count: 3,
                minimum: 100.0 * (1.0 - noise),
                median: 100.0,
                maximum: 100.0 * (1.0 + noise),
                standard_deviation: 100.0 * noise,
            },
            bandwidth: SampleStatistics::default(),
        };
        let isolated_drop = [
            point(1, 1.0, 0.01),
            point(2, 0.85, 0.01),
            point(4, 0.95, 0.01),
        ];
        assert_eq!(sustained_transition(&isolated_drop), None);

        let sustained_drop = [
            point(1, 1.0, 0.01),
            point(2, 0.85, 0.01),
            point(4, 0.82, 0.01),
        ];
        assert_eq!(sustained_transition(&sustained_drop), Some(1));

        let noisy_drop = [
            point(1, 1.0, 0.01),
            point(2, 0.85, 0.10),
            point(4, 0.82, 0.01),
        ];
        assert_eq!(sustained_transition(&noisy_drop), None);

        let noisy_confirmation = [
            point(1, 1.0, 0.01),
            point(2, 0.85, 0.01),
            point(4, 0.82, 0.10),
        ];
        assert_eq!(sustained_transition(&noisy_confirmation), None);

        let temporary_dip_then_recovery = [
            point(1, 1.0, 0.01),
            point(2, 0.82, 0.01),
            point(4, 0.84, 0.01),
            point(8, 0.96, 0.01),
            point(16, 0.94, 0.01),
            point(32, 0.70, 0.01),
            point(64, 0.68, 0.01),
        ];
        assert_eq!(sustained_transition(&temporary_dip_then_recovery), Some(5));
    }
}
