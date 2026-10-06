use ash::{Entry, vk};
use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult,
    CancellationToken, Metric, ProgressCallback, ProgressUpdate, SampleStatistics,
};
use std::{collections::BTreeMap, time::Instant};

use super::{
    AdapterRecord, GPU_PRECONDITION_MS, done, duration_ns, ensure_not_cancelled,
    insert_clock_policy, per_sample_duration, statistics,
};

pub(super) const FP16_MATRIX_ID: &str = "gpu.performance.matrix.fp16";
pub(super) const FP16_MATRIX_SCALING_ID: &str = "gpu.performance.matrix.fp16.scaling";
pub(super) const INT8_MATRIX_ID: &str = "gpu.performance.matrix.int8";
pub(super) const FP8_MATRIX_ID: &str = "gpu.performance.matrix.fp8";
pub(super) const SPARSE_FP16_MATRIX_ID: &str = "gpu.performance.matrix.sparse.fp16";
pub(super) const SPARSE_INT8_MATRIX_ID: &str = "gpu.performance.matrix.sparse.int8";
pub(super) const SPARSE_FP8_MATRIX_ID: &str = "gpu.performance.matrix.sparse.fp8";

const FP16_SPV: &[u8] = include_bytes!("../shaders/spv/matrix_fp16.spv");
const FP16_PROFILE_SPV: &[u8] = include_bytes!("../shaders/spv/matrix_profile_fp16.spv");
const SINT8_SPV: &[u8] = include_bytes!("../shaders/spv/matrix_int8.spv");
const UINT8_SPV: &[u8] = include_bytes!("../shaders/spv/matrix_uint8.spv");
const MATRIX_WORKGROUP_SIZE: u64 = 64;
const MATRIX_LOOP_COUNT: u64 = 256;
const MATRIX_PROFILE_REUSE: u32 = 8;
const MATRIX_PROFILE_MIN_SIZE: u64 = 256 * 1024;
const WORKGROUP_CANDIDATES: [u32; 5] = [512, 1024, 2048, 4096, 8192];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub input: vk::ComponentTypeKHR,
    pub accumulator: vk::ComponentTypeKHR,
}

impl MatrixShape {
    fn operations_per_mma(self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct CooperativeSupport {
    pub fp16: Option<MatrixShape>,
    pub int8: Option<MatrixShape>,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixKind {
    Fp16,
    Sint8,
    Uint8,
}

impl MatrixKind {
    fn for_benchmark(
        record: &AdapterRecord,
        id: &str,
    ) -> Result<(Self, MatrixShape), BenchmarkError> {
        match id {
            FP16_MATRIX_ID | FP16_MATRIX_SCALING_ID => record.cooperative.fp16.map(|shape| (Self::Fp16, shape)).ok_or_else(|| {
                unsupported(record, "fp16_cooperative_matrix_unsupported", "The selected GPU does not advertise a Vulkan cooperative-matrix configuration with FP16 inputs and FP32 accumulation.")
            }),
            INT8_MATRIX_ID => {
                let shape = record.cooperative.int8.ok_or_else(|| unsupported(
                    record,
                    "int8_cooperative_matrix_unsupported",
                    "The selected GPU does not advertise a Vulkan cooperative-matrix configuration with INT8 inputs and 32-bit accumulation.",
                ))?;
                match (shape.input, shape.accumulator) {
                    (vk::ComponentTypeKHR::SINT8, vk::ComponentTypeKHR::SINT32) => Ok((Self::Sint8, shape)),
                    (vk::ComponentTypeKHR::UINT8, vk::ComponentTypeKHR::UINT32) => Ok((Self::Uint8, shape)),
                    _ => Err(BenchmarkError::new(
                        "int8_cooperative_matrix_type_pair_unsupported",
                        "The GPU reported a mixed signedness INT8 cooperative-matrix configuration that the embedded kernels do not misrepresent as a compatible workload.",
                    )),
                }
            }
            FP8_MATRIX_ID => Err(BenchmarkError::new(
                "fp8_cooperative_matrix_unsupported",
                "No FP8 cooperative-matrix configuration is exposed by the current Vulkan device and driver.",
            )),
            SPARSE_FP16_MATRIX_ID | SPARSE_INT8_MATRIX_ID | SPARSE_FP8_MATRIX_ID => Err(BenchmarkError::new(
                "sparse_matrix_extension_unsupported",
                "VK_KHR_cooperative_matrix defines dense matrix operations. This device does not expose a capability-verified structured-sparse Vulkan execution path.",
            )),
            _ => Err(BenchmarkError::new("unsupported_benchmark", format!("Unknown cooperative-matrix benchmark '{id}'."))),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fp16 => "FP16",
            Self::Sint8 | Self::Uint8 => "INT8",
        }
    }

    fn input_label(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Sint8 => "sint8",
            Self::Uint8 => "uint8",
        }
    }

    fn accumulator_label(self) -> &'static str {
        match self {
            Self::Fp16 => "fp32",
            Self::Sint8 => "sint32",
            Self::Uint8 => "uint32",
        }
    }

    fn shader(self) -> &'static [u8] {
        match self {
            Self::Fp16 => FP16_SPV,
            Self::Sint8 => SINT8_SPV,
            Self::Uint8 => UINT8_SPV,
        }
    }
}

pub(super) fn is_matrix_benchmark(id: &str) -> bool {
    matches!(
        id,
        FP16_MATRIX_ID
            | FP16_MATRIX_SCALING_ID
            | INT8_MATRIX_ID
            | FP8_MATRIX_ID
            | SPARSE_FP16_MATRIX_ID
            | SPARSE_INT8_MATRIX_ID
            | SPARSE_FP8_MATRIX_ID
    )
}

pub(super) fn descriptors(adapters: &[AdapterRecord]) -> Vec<BenchmarkDescriptor> {
    let mut descriptors = [
        (
            FP16_MATRIX_ID,
            "Dense FP16 matrix performance",
            "fp16 matrix multiply-accumulate",
            "fp16",
            "dense",
        ),
        (
            INT8_MATRIX_ID,
            "Dense INT8 matrix performance",
            "int8 matrix multiply-accumulate with 32-bit accumulation",
            "int8",
            "dense",
        ),
        (
            FP8_MATRIX_ID,
            "Dense FP8 matrix performance",
            "fp8 matrix multiply-accumulate",
            "fp8",
            "dense",
        ),
        (
            SPARSE_FP16_MATRIX_ID,
            "Sparse FP16 matrix performance",
            "structured-sparse fp16 matrix multiply-accumulate",
            "fp16",
            "sparse",
        ),
        (
            SPARSE_INT8_MATRIX_ID,
            "Sparse INT8 matrix performance",
            "structured-sparse int8 matrix multiply-accumulate",
            "int8",
            "sparse",
        ),
        (
            SPARSE_FP8_MATRIX_ID,
            "Sparse FP8 matrix performance",
            "structured-sparse fp8 matrix multiply-accumulate",
            "fp8",
            "sparse",
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (id, name, data_type, format, density))| {
        let supported_device_ids = adapters
            .iter()
            .filter(|record| matrix_available(record, id))
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        let hardware_supported_device_ids = adapters
            .iter()
            .filter(|record| hardware_support(record, id))
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        let available = !supported_device_ids.is_empty();
        let unavailable_reason = if available {
            String::new()
        } else if adapters.is_empty() {
            "adapter_not_found".into()
        } else if density == "sparse" {
            "sparse_matrix_extension_unsupported".into()
        } else if format == "fp8" {
            "fp8_cooperative_matrix_unsupported".into()
        } else {
            adapters
                .iter()
                .find_map(|record| {
                    (!record.cooperative.reason.is_empty())
                        .then_some(record.cooperative.reason.clone())
                })
                .unwrap_or_else(|| "cooperative_matrix_format_unsupported".into())
        };
        let mut metadata = BTreeMap::new();
        metadata.insert("execution_domain".into(), "cooperative_matrix".into());
        metadata.insert("execution_backend".into(), "raw-vulkan".into());
        metadata.insert("api".into(), "VK_KHR_cooperative_matrix".into());
        metadata.insert("shader_format".into(), "embedded_spirv".into());
        metadata.insert("matrix_density".into(), density.into());
        metadata.insert("matrix_input_format".into(), format.into());
        metadata.insert(
            "operation_counting".into(),
            "one MxNxK matrix multiply-accumulate equals 2*M*N*K operations".into(),
        );
        metadata.insert("capability_gated".into(), "true".into());
        metadata.insert(
            "hardware_supported_device_ids".into(),
            hardware_supported_device_ids.join(","),
        );
        metadata.insert(
            "hardware_supported".into(),
            (!hardware_supported_device_ids.is_empty()).to_string(),
        );
        BenchmarkDescriptor {
            id: id.into(),
            name: name.into(),
            category: BenchmarkCategory::Gpu,
            workload: "Register-resident cooperative matrix multiply-accumulate throughput".into(),
            data_type: data_type.into(),
            unit: "operations/s".into(),
            supported_device_ids,
            available,
            unavailable_reason,
            suite_id: "gpu.performance".into(),
            display_order: 120 + index as u32 * 2,
            metadata,
        }
    })
    .collect::<Vec<_>>();
    let supported_device_ids = adapters
        .iter()
        .filter(|record| matrix_available(record, FP16_MATRIX_SCALING_ID))
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();
    let available = !supported_device_ids.is_empty();
    descriptors.insert(
        1,
        BenchmarkDescriptor {
            id: FP16_MATRIX_SCALING_ID.into(),
            name: "FP16 matrix compute scaling".into(),
            category: BenchmarkCategory::Gpu,
            workload:
                "Measures FP16 matrix calculation speed as data grows from cache-sized sets toward your selected VRAM budget"
                    .into(),
            data_type: "fp16".into(),
            unit: "operations/s".into(),
            supported_device_ids,
            available,
            unavailable_reason: if available {
                String::new()
            } else if adapters.is_empty() {
                "adapter_not_found".into()
            } else if adapters
                .iter()
                .any(|record| record.cooperative.fp16.is_some())
            {
                "storage_buffer_16bit_access_unsupported".into()
            } else {
                "fp16_cooperative_matrix_unsupported".into()
            },
            suite_id: "gpu.performance".into(),
            display_order: 121,
            metadata: BTreeMap::from([
                (
                    "execution_domain".into(),
                    "memory_backed_cooperative_matrix".into(),
                ),
                ("execution_backend".into(), "raw-vulkan".into()),
                ("api".into(), "VK_KHR_cooperative_matrix".into()),
                ("profile_axis".into(), "operand_working_set_bytes".into()),
                (
                    "model".into(),
                    "dense_weight_streaming_with_tile_reuse".into(),
                ),
            ]),
        },
    );
    descriptors
}

fn hardware_support(record: &AdapterRecord, id: &str) -> bool {
    match id {
        FP16_MATRIX_ID | FP16_MATRIX_SCALING_ID => record.cooperative.fp16.is_some(),
        INT8_MATRIX_ID => record.cooperative.int8.is_some(),
        _ => false,
    }
}

fn matrix_available(record: &AdapterRecord, id: &str) -> bool {
    record.supports_vulkan_timestamps()
        && MatrixKind::for_benchmark(record, id).is_ok()
        && (id != FP16_MATRIX_SCALING_ID || record.vulkan.storage_buffer_16bit_access)
}

pub(super) fn run(
    record: &AdapterRecord,
    benchmark_id: &str,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
) -> Result<BenchmarkResult, BenchmarkError> {
    let (kind, shape) = MatrixKind::for_benchmark(record, benchmark_id)?;
    if benchmark_id == FP16_MATRIX_SCALING_ID {
        return run_fp16_matrix_scaling(record, kind, shape, config, cancellation, progress);
    }
    if !record.supports_vulkan_timestamps() {
        return Err(BenchmarkError::new(
            "timestamp_query_unsupported",
            "This GPU does not expose Vulkan timestamps required for matrix timing.",
        ));
    }
    if record.vulkan.subgroup_size == 0
        || !MATRIX_WORKGROUP_SIZE.is_multiple_of(record.vulkan.subgroup_size as u64)
    {
        return Err(BenchmarkError::new(
            "subgroup_size_unsupported",
            format!(
                "The Vulkan subgroup size {} does not divide the {}-thread matrix workgroup.",
                record.vulkan.subgroup_size, MATRIX_WORKGROUP_SIZE,
            ),
        ));
    }

    let started = Instant::now();
    ensure_not_cancelled(cancellation)?;
    progress(ProgressUpdate {
        fraction: 0.02,
        phase: "gpu_matrix_setup".into(),
        message: format!(
            "Creating the raw Vulkan {} cooperative-matrix pipeline",
            kind.label()
        ),
    });
    let context = VulkanMatrixContext::new(record, kind)?;
    let workgroups = autotune_workgroups(&context, kind, shape, cancellation)?;
    let harness = MatrixHarness::new(&context, kind, shape, workgroups)?;

    progress(ProgressUpdate {
        fraction: 0.12,
        phase: "gpu_matrix_precondition".into(),
        message: "Saturating the GPU matrix execution units before measurement".into(),
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
    let sample_count = config.samples.max(1);
    let calibrated_ns = harness.measure(1)?;
    let sample_iterations =
        iterations_for_target(calibrated_ns, per_sample_duration(config).as_secs_f64());
    let operations_per_dispatch = harness.operations_per_dispatch();
    let mut values = Vec::with_capacity(sample_count as usize);
    for sample in 0..sample_count {
        ensure_not_cancelled(cancellation)?;
        progress(ProgressUpdate {
            fraction: 0.20 + sample as f64 / sample_count as f64 * 0.75,
            phase: "gpu_matrix_measurement".into(),
            message: format!(
                "Measuring {} matrix throughput sample {}/{}",
                kind.label(),
                sample + 1,
                sample_count
            ),
        });
        let elapsed_ns = harness.measure(sample_iterations)?;
        values.push(operations_per_dispatch as f64 * sample_iterations as f64 / (elapsed_ns / 1e9));
    }
    let sample_statistics = statistics(&values);
    let mut metadata = matrix_metadata(record, kind, shape);
    insert_clock_policy(&mut metadata);
    metadata.insert("selected_workgroups".into(), workgroups.to_string());
    metadata.insert("working_set_bytes".into(), harness.output_size.to_string());
    metadata.insert(
        "matrix_mma_per_subgroup".into(),
        MATRIX_LOOP_COUNT.to_string(),
    );
    metadata.insert(
        "operations_per_matrix_mma".into(),
        shape.operations_per_mma().to_string(),
    );
    metadata.insert(
        "operation_definition".into(),
        format!(
            "one {}x{}x{} matrix multiply-accumulate equals {} operations",
            shape.m,
            shape.n,
            shape.k,
            shape.operations_per_mma()
        ),
    );
    metadata.insert("bound_classification".into(), "compute_bound".into());
    metadata.insert("diagnostic_domain".into(), "gpu".into());
    metadata.insert("classification_method".into(), "register-resident repeated cooperative matrix multiply-accumulate with only final result stores".into());
    metadata.insert("classification_is_inference".into(), "true".into());
    metadata.insert(
        "gpu_core_saturation_policy".into(),
        "autotune normalized matrix throughput across 512 to 8192 workgroups".into(),
    );
    progress(done(&format!(
        "GPU {} matrix benchmark completed.",
        kind.label()
    )));
    Ok(BenchmarkResult {
        benchmark_id: benchmark_id.into(),
        device_id: record.id.clone(),
        elapsed_ns: duration_ns(started.elapsed()),
        metrics: vec![Metric {
            name: "throughput".into(),
            value: sample_statistics.median,
            unit: "operations/s".into(),
            statistics: sample_statistics,
        }],
        workload_metadata: metadata,
        device_metadata: matrix_device_metadata(record),
    })
}

#[derive(Debug)]
struct MatrixProfilePoint {
    working_set_bytes: u64,
    compute: SampleStatistics,
    bandwidth: SampleStatistics,
    retained_ratio: f64,
}

fn run_fp16_matrix_scaling(
    record: &AdapterRecord,
    kind: MatrixKind,
    shape: MatrixShape,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    callback: &mut ProgressCallback<'_>,
) -> Result<BenchmarkResult, BenchmarkError> {
    if !record.supports_vulkan_timestamps() {
        return Err(BenchmarkError::new(
            "timestamp_query_unsupported",
            "This GPU does not expose Vulkan timestamps required for matrix timing.",
        ));
    }
    if !record.vulkan.storage_buffer_16bit_access {
        return Err(BenchmarkError::new(
            "storage_buffer_16bit_access_unsupported",
            "The selected GPU cannot load FP16 cooperative-matrix operands from Vulkan storage buffers.",
        ));
    }
    if record.vulkan.subgroup_size == 0
        || !MATRIX_WORKGROUP_SIZE.is_multiple_of(record.vulkan.subgroup_size as u64)
    {
        return Err(BenchmarkError::new(
            "subgroup_size_unsupported",
            "The selected GPU subgroup size is incompatible with the matrix profile kernel.",
        ));
    }
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
            FP16_MATRIX_ID,
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
    let context = VulkanMatrixContext::new(record, kind)?;
    progress(ProgressUpdate {
        fraction: 0.01,
        phase: "gpu_matrix_profile_setup".into(),
        message: "Selecting a saturation width for the FP16 matrix profile".into(),
    });
    let workgroups = autotune_workgroups(&context, kind, shape, cancellation)?;
    let bytes_per_tile = matrix_profile_bytes_per_tile(shape);
    let max_tiles_per_binding = (record.vulkan.max_storage_buffer_range
        / (shape.m as u64 * shape.k as u64 * 2))
        .min(record.vulkan.max_storage_buffer_range / (shape.k as u64 * shape.n as u64 * 2));
    let binding_limit = max_tiles_per_binding.saturating_mul(bytes_per_tile);
    let budget_percent = gluj_bench_core::vram_budget_percent(config)?;
    let allocation_budget =
        gluj_bench_core::vram_budget_bytes(record.vulkan.device_local_memory_bytes, budget_percent);
    let active_subgroups = workgroups as u64 * MATRIX_WORKGROUP_SIZE / context.subgroup_size as u64;
    let output_bytes = active_subgroups * shape.m as u64 * shape.n as u64 * 4;
    let mut memory_limit = allocation_budget
        .saturating_sub(output_bytes)
        .saturating_sub(16 * 1024 * 1024);
    if record.vulkan.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        memory_limit = memory_limit.min(record.vulkan.device_local_memory_bytes / 8);
    }
    let maximum = binding_limit.min(memory_limit);
    let (maximum, single_dataset) = gluj_bench_core::configured_dataset(config, maximum)?;
    let requested_sizes = if single_dataset {
        vec![maximum]
    } else {
        matrix_profile_sizes(maximum)
    };
    if requested_sizes.len() < if single_dataset { 1 } else { 3 } {
        return Err(BenchmarkError::new(
            "insufficient_gpu_memory",
            "At least three FP16 operand-size tiers are required for a matrix scaling profile.",
        ));
    }
    progress(ProgressUpdate {
        fraction: 0.04,
        phase: "gpu_matrix_profile_setup".into(),
        message: format!(
            "Allocating FP16 matrix operands up to {}",
            format_profile_size(maximum)
        ),
    });
    let harness = MatrixProfileHarness::new(&context, shape, workgroups, maximum)?;
    let sizes = requested_sizes
        .into_iter()
        .map(|size| harness.actual_working_set(size))
        .collect::<Vec<_>>();
    let mut sizes = sizes;
    sizes.dedup();
    let reuse_count = config
        .options
        .get("matrix_tile_reuse")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(MATRIX_PROFILE_REUSE)
        .clamp(1, 64);
    let sample_count = config.samples.clamp(2, 5);
    let target_seconds =
        (config.target_duration_ms as f64 / 1000.0 / sizes.len() as f64 / sample_count as f64)
            .clamp(0.01, 0.08);
    let target_seconds =
        gluj_bench_core::gpu_burst_duration(std::time::Duration::from_secs_f64(target_seconds))
            .as_secs_f64();
    let mut timing_metrics = Vec::new();
    let mut timing_batches = Vec::new();
    let mut raw_points = Vec::with_capacity(sizes.len());
    for (index, working_set) in sizes.iter().copied().enumerate() {
        ensure_not_cancelled(cancellation)?;
        progress(ProgressUpdate {
            fraction: 0.06 + index as f64 / sizes.len() as f64 * 0.89,
            phase: "gpu_matrix_profile_sweep".into(),
            message: format!(
                "Measuring FP16 matrix throughput with {} of operands ({}/{})",
                format_profile_size(working_set),
                index + 1,
                sizes.len()
            ),
        });
        let _ = harness.measure(working_set, reuse_count, 1)?;
        let calibration_ns = harness.measure(working_set, reuse_count, 1)?;
        let iterations = iterations_for_target(calibration_ns, target_seconds);
        let operations = harness.operations_per_dispatch(working_set, reuse_count) as f64;
        let traffic = harness.traffic_bytes_per_dispatch(working_set) as f64;
        let mut compute_values = Vec::with_capacity(sample_count as usize);
        let mut bandwidth_values = Vec::with_capacity(sample_count as usize);
        let mut gpu_times = Vec::with_capacity(sample_count as usize);
        let mut end_to_end_times = Vec::with_capacity(sample_count as usize);
        for _ in 0..sample_count {
            ensure_not_cancelled(cancellation)?;
            let sample_started = Instant::now();
            let elapsed_ns = harness.measure(working_set, reuse_count, iterations)?;
            let end_to_end_ns = sample_started.elapsed().as_secs_f64() * 1e9;
            gpu_times.push(crate::timing::per_pass(elapsed_ns, iterations));
            end_to_end_times.push(crate::timing::per_pass(end_to_end_ns, iterations));
            let seconds = elapsed_ns / 1e9;
            compute_values.push(operations * iterations as f64 / seconds);
            bandwidth_values.push(traffic * iterations as f64 / seconds);
        }
        crate::timing::append_metrics(
            &mut timing_metrics,
            working_set,
            &gpu_times,
            &end_to_end_times,
        );
        timing_batches.push(format!("{working_set}:{iterations}"));
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
        .map(
            |(working_set_bytes, compute, bandwidth)| MatrixProfilePoint {
                working_set_bytes,
                retained_ratio: (compute.median / baseline).clamp(0.0, 2.0),
                compute,
                bandwidth,
            },
        )
        .collect::<Vec<_>>();
    let transition_analysis_minimum = harness
        .active_subgroups
        .saturating_mul(bytes_per_tile)
        .saturating_mul(reuse_count as u64);
    let transition_index = matrix_profile_transition(&points, transition_analysis_minimum);
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
    metrics.extend(timing_metrics);
    if let Some(index) = transition_index {
        metrics.push(Metric {
            name: "bandwidth_transition_working_set".into(),
            value: points[index].working_set_bytes as f64,
            unit: "bytes".into(),
            statistics: statistics(&[points[index].working_set_bytes as f64]),
        });
    }
    let mut metadata = matrix_metadata(record, kind, shape);
    metadata.insert(
        "gpu_timing_method".into(),
        "gpu_timestamp_and_host_elapsed_per_pass_v1".into(),
    );
    metadata.insert(
        "gpu_timing_definition".into(),
        gluj_bench_core::GPU_TIMING_EXPLANATION.into(),
    );
    metadata.insert("gpu_timing_batch_passes".into(), timing_batches.join(","));
    metadata.insert("gpu_timing_samples_paired".into(), "true".into());
    insert_clock_policy(&mut metadata);
    metadata.insert(
        "execution_domain".into(),
        "memory_backed_cooperative_matrix".into(),
    );
    metadata.insert(
        "matrix_profile_revision".into(),
        "weight-streaming-1".into(),
    );
    metadata.insert("matrix_tile_reuse".into(), reuse_count.to_string());
    metadata.insert("operand_bytes_per_tile".into(), bytes_per_tile.to_string());
    metadata.insert("tested_tier_count".into(), points.len().to_string());
    metadata.insert(
        "allocation_budget_bytes".into(),
        allocation_budget.to_string(),
    );
    metadata.insert(
        "allocated_test_buffer_bytes".into(),
        (harness.maximum_working_set_bytes + output_bytes).to_string(),
    );
    metadata.insert(
        "profile_sample_count_per_tier".into(),
        sample_count.to_string(),
    );
    metadata.insert(
        "maximum_operand_working_set_bytes".into(),
        maximum.to_string(),
    );
    metadata.insert(
        "transition_analysis_minimum_working_set_bytes".into(),
        transition_analysis_minimum.to_string(),
    );
    metadata.insert("workload_model".into(), "dense weight-streaming matrix tiles with configurable reuse; representative of memory pressure in low-batch neural-network inference, not an end-to-end LLM".into());
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
            update.fraction = 0.95 + update.fraction * 0.05;
            update.message = format!("Checking compute reference drift: {}", update.message);
            callback(update);
        };
        run(
            record,
            FP16_MATRIX_ID,
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
    callback(done("FP16 matrix compute scaling completed."));
    Ok(BenchmarkResult {
        benchmark_id: FP16_MATRIX_SCALING_ID.into(),
        device_id: record.id.clone(),
        elapsed_ns: duration_ns(started.elapsed()),
        metrics,
        workload_metadata: metadata,
        device_metadata: matrix_device_metadata(record),
    })
}

fn matrix_profile_sizes(maximum: u64) -> Vec<u64> {
    let mut sizes = Vec::new();
    let mut size = MATRIX_PROFILE_MIN_SIZE;
    while size <= maximum {
        sizes.push(size);
        let Some(next) = size.checked_mul(2) else {
            break;
        };
        size = next;
    }
    if !sizes.is_empty() && sizes.last().copied() != Some(maximum) {
        sizes.push(maximum);
    }
    sizes
}

fn matrix_profile_transition(
    points: &[MatrixProfilePoint],
    minimum_working_set_bytes: u64,
) -> Option<usize> {
    points.windows(2).enumerate().find_map(|(index, pair)| {
        let is_drop = pair[0].working_set_bytes >= minimum_working_set_bytes
            && pair[0].retained_ratio < 0.90
            && pair[1].retained_ratio < 0.90
            && matrix_profile_drop_is_significant(&pair[0])
            && matrix_profile_drop_is_significant(&pair[1]);
        let later_recovery = points[index + 2..].windows(2).any(|recovery| {
            recovery[0].retained_ratio >= 0.90 && recovery[1].retained_ratio >= 0.90
        });
        (is_drop && !later_recovery).then_some(index)
    })
}

fn matrix_profile_drop_is_significant(point: &MatrixProfilePoint) -> bool {
    let noise = if point.compute.median > 0.0 {
        point.compute.standard_deviation / point.compute.median
    } else {
        1.0
    };
    1.0 - point.retained_ratio > (noise * 2.0).max(0.02)
}

fn matrix_profile_bytes_per_tile(shape: MatrixShape) -> u64 {
    (shape.m as u64 * shape.k as u64 + shape.k as u64 * shape.n as u64) * 2
}

fn format_profile_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} KiB", bytes as f64 / 1024.0)
    }
}

fn autotune_workgroups(
    context: &VulkanMatrixContext,
    kind: MatrixKind,
    shape: MatrixShape,
    cancellation: &CancellationToken,
) -> Result<u32, BenchmarkError> {
    let mut best = None;
    for workgroups in WORKGROUP_CANDIDATES {
        ensure_not_cancelled(cancellation)?;
        let harness = MatrixHarness::new(context, kind, shape, workgroups)?;
        let trial_ns = harness.measure(1)?;
        let iterations = iterations_for_target(trial_ns, 0.04);
        let elapsed_ns = harness.measure(iterations)?;
        let rate =
            harness.operations_per_dispatch() as f64 * iterations as f64 / (elapsed_ns / 1e9);
        if best.is_none_or(|(_, best_rate)| rate > best_rate) {
            best = Some((workgroups, rate));
        }
    }
    best.map(|(workgroups, _)| workgroups).ok_or_else(|| {
        BenchmarkError::new(
            "compute_limits_unsupported",
            "No cooperative-matrix saturation workload could be dispatched.",
        )
    })
}

fn iterations_for_target(per_dispatch_ns: f64, target_seconds: f64) -> u32 {
    (target_seconds * 1e9 / per_dispatch_ns)
        .ceil()
        .clamp(1.0, 4096.0) as u32
}

struct Buffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
}

struct VulkanMatrixContext {
    _entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    timestamp_period_ns: f64,
    timestamp_valid_bits: u32,
    subgroup_size: u32,
}

impl VulkanMatrixContext {
    fn new(record: &AdapterRecord, kind: MatrixKind) -> Result<Self, BenchmarkError> {
        // SAFETY: every handle is owned by the returned context and released by Drop.
        unsafe { Self::new_inner(record, kind) }
    }

    unsafe fn new_inner(record: &AdapterRecord, kind: MatrixKind) -> Result<Self, BenchmarkError> {
        let entry = unsafe { Entry::load() }
            .map_err(|problem| error("vulkan_loader_unavailable", problem))?;
        let app = c"Gluj-Bench";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app)
            .application_version(1)
            .engine_name(app)
            .engine_version(1)
            .api_version(record.vulkan.api_version.min(vk::API_VERSION_1_3));
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }
        .map_err(|problem| error("vulkan_instance_failed", problem))?;
        let physical_device =
            match unsafe { find_physical_device(&instance, &record.vulkan.device_uuid) } {
                Ok(device) => device,
                Err(problem) => {
                    unsafe { instance.destroy_instance(None) };
                    return Err(problem);
                }
            };
        let priorities = [1.0_f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(record.vulkan.compute_queue_family)
            .queue_priorities(&priorities)];
        let extension_names = [ash::khr::cooperative_matrix::NAME.as_ptr()];
        let mut cooperative =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut numeric = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(kind == MatrixKind::Fp16)
            .shader_int8(kind != MatrixKind::Fp16);
        let mut storage_16bit = vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(
                kind == MatrixKind::Fp16 && record.vulkan.storage_buffer_16bit_access,
            );
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extension_names)
            .push_next(&mut cooperative)
            .push_next(&mut numeric)
            .push_next(&mut storage_16bit);
        let device = match unsafe { instance.create_device(physical_device, &device_info, None) } {
            Ok(device) => device,
            Err(problem) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error("vulkan_matrix_device_failed", problem));
            }
        };
        let queue = unsafe { device.get_device_queue(record.vulkan.compute_queue_family, 0) };
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(record.vulkan.compute_queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|problem| error("vulkan_command_pool_failed", problem))?;
        let command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|problem| error("vulkan_command_buffer_failed", problem))?[0];
        let query_pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .map_err(|problem| error("vulkan_query_pool_failed", problem))?;
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        Ok(Self {
            _entry: entry,
            instance,
            device,
            queue,
            command_pool,
            command_buffer,
            query_pool,
            memory_properties,
            timestamp_period_ns: record.vulkan.timestamp_period_ns as f64,
            timestamp_valid_bits: record.vulkan.timestamp_valid_bits,
            subgroup_size: record.vulkan.subgroup_size,
        })
    }

    fn create_output(&self, size: u64) -> Result<Buffer, BenchmarkError> {
        self.create_buffer(size, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer, BenchmarkError> {
        // SAFETY: local allocation is bound once and freed with its buffer by the harness.
        unsafe {
            let handle = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(|problem| error("insufficient_gpu_memory", problem))?;
            let requirements = self.device.get_buffer_memory_requirements(handle);
            let memory_type = (0..self.memory_properties.memory_type_count)
                .find(|index| {
                    requirements.memory_type_bits & (1 << index) != 0
                        && self.memory_properties.memory_types[*index as usize]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .ok_or_else(|| {
                    BenchmarkError::new(
                        "vulkan_memory_type_unavailable",
                        "No device-local Vulkan memory type is available for matrix output.",
                    )
                })?;
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type),
                    None,
                )
                .map_err(|problem| error("insufficient_gpu_memory", problem))?;
            self.device
                .bind_buffer_memory(handle, memory, 0)
                .map_err(|problem| error("vulkan_buffer_bind_failed", problem))?;
            Ok(Buffer { handle, memory })
        }
    }

    fn measure<F>(&self, record: F) -> Result<f64, BenchmarkError>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer),
    {
        // SAFETY: one command buffer is reused only after its previous queue submission is idle.
        unsafe {
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|problem| error("vulkan_command_reset_failed", problem))?;
            self.device
                .begin_command_buffer(
                    self.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|problem| error("vulkan_command_begin_failed", problem))?;
            self.device
                .cmd_reset_query_pool(self.command_buffer, self.query_pool, 0, 2);
            self.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                0,
            );
            record(&self.device, self.command_buffer);
            self.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                1,
            );
            self.device
                .end_command_buffer(self.command_buffer)
                .map_err(|problem| error("vulkan_command_end_failed", problem))?;
            let activity_started = std::time::Instant::now();
            let command_buffers = [self.command_buffer];
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&command_buffers)],
                    vk::Fence::null(),
                )
                .map_err(|problem| error("vulkan_queue_submit_failed", problem))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|problem| error("device_lost", problem))?;
            gluj_bench_core::pace_gpu(activity_started.elapsed())?;
            let mut timestamps = [0_u64; 2];
            self.device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64,
                )
                .map_err(|problem| error("vulkan_timestamp_read_failed", problem))?;
            let ticks =
                wrapped_timestamp_delta(timestamps[0], timestamps[1], self.timestamp_valid_bits);
            let elapsed = ticks as f64 * self.timestamp_period_ns;
            if elapsed <= 0.0 || !elapsed.is_finite() {
                return Err(BenchmarkError::new(
                    "device_lost",
                    "The Vulkan matrix timestamp interval was invalid.",
                ));
            }
            Ok(elapsed)
        }
    }
}

impl Drop for VulkanMatrixContext {
    fn drop(&mut self) {
        // SAFETY: the context exclusively owns these objects and waits before destruction.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct MatrixHarness<'a> {
    context: &'a VulkanMatrixContext,
    workgroups: u32,
    shape: MatrixShape,
    output_size: u64,
    output: Buffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl<'a> MatrixHarness<'a> {
    fn new(
        context: &'a VulkanMatrixContext,
        kind: MatrixKind,
        shape: MatrixShape,
        workgroups: u32,
    ) -> Result<Self, BenchmarkError> {
        let subgroups = MATRIX_WORKGROUP_SIZE / context.subgroup_size as u64;
        let output_size = workgroups as u64 * subgroups * shape.m as u64 * shape.n as u64 * 4;
        let output = context.create_output(output_size)?;
        // SAFETY: created objects are owned by the harness and destroyed by Drop.
        unsafe {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let descriptor_layout = context
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|problem| error("vulkan_descriptor_layout_failed", problem))?;
            let set_layouts = [descriptor_layout];
            let pipeline_layout = context
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
                .map_err(|problem| error("vulkan_pipeline_layout_failed", problem))?;
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)];
            let descriptor_pool = context
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(|problem| error("vulkan_descriptor_pool_failed", problem))?;
            let descriptor_set = context
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts),
                )
                .map_err(|problem| error("vulkan_descriptor_set_failed", problem))?[0];
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(output.handle)
                .range(output_size)];
            context.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info)],
                &[],
            );
            let words = shader_words(kind.shader())?;
            let module = context
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(|problem| error("vulkan_shader_module_failed", problem))?;
            let specialization_values = [shape.m, shape.n, shape.k];
            let specialization_entries = [
                vk::SpecializationMapEntry {
                    constant_id: 0,
                    offset: 0,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 1,
                    offset: 4,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 2,
                    offset: 8,
                    size: 4,
                },
            ];
            let specialization = vk::SpecializationInfo::default()
                .map_entries(&specialization_entries)
                .data(words_as_bytes(&specialization_values));
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .specialization_info(&specialization);
            let pipeline_result = context.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            );
            context.device.destroy_shader_module(module, None);
            let pipeline = pipeline_result
                .map_err(|(_, problem)| error("vulkan_matrix_pipeline_failed", problem))?[0];
            let harness = Self {
                context,
                workgroups,
                shape,
                output_size,
                output,
                descriptor_pool,
                descriptor_layout,
                descriptor_set,
                pipeline_layout,
                pipeline,
            };
            harness.measure(1)?;
            Ok(harness)
        }
    }

    fn operations_per_dispatch(&self) -> u64 {
        let subgroups = MATRIX_WORKGROUP_SIZE / self.context.subgroup_size as u64;
        self.workgroups as u64 * subgroups * MATRIX_LOOP_COUNT * self.shape.operations_per_mma()
    }

    fn measure(&self, iterations: u32) -> Result<f64, BenchmarkError> {
        self.context.measure(|device, command_buffer| unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            for _ in 0..iterations {
                device.cmd_dispatch(command_buffer, self.workgroups, 1, 1);
            }
        })
    }
}

impl Drop for MatrixHarness<'_> {
    fn drop(&mut self) {
        // SAFETY: measurements wait for the queue and the harness owns these child objects.
        unsafe {
            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.context
                .device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.context.device.destroy_buffer(self.output.handle, None);
            self.context.device.free_memory(self.output.memory, None);
        }
    }
}

struct MatrixProfileHarness<'a> {
    context: &'a VulkanMatrixContext,
    shape: MatrixShape,
    workgroups: u32,
    active_subgroups: u64,
    bytes_per_tile: u64,
    maximum_working_set_bytes: u64,
    input_a: Buffer,
    input_b: Buffer,
    output: Buffer,
    output_size: u64,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl<'a> MatrixProfileHarness<'a> {
    fn new(
        context: &'a VulkanMatrixContext,
        shape: MatrixShape,
        workgroups: u32,
        requested_working_set: u64,
    ) -> Result<Self, BenchmarkError> {
        let bytes_a_per_tile = shape.m as u64 * shape.k as u64 * 2;
        let bytes_b_per_tile = shape.k as u64 * shape.n as u64 * 2;
        let bytes_per_tile = bytes_a_per_tile + bytes_b_per_tile;
        let tile_count = (requested_working_set / bytes_per_tile).max(1);
        if tile_count > u32::MAX as u64 {
            return Err(BenchmarkError::new(
                "matrix_profile_too_large",
                "The requested matrix operand set exceeds the shader tile-index range.",
            ));
        }
        let input_a_size = tile_count * bytes_a_per_tile;
        let input_b_size = tile_count * bytes_b_per_tile;
        let maximum_working_set_bytes = tile_count * bytes_per_tile;
        let active_subgroups =
            workgroups as u64 * MATRIX_WORKGROUP_SIZE / context.subgroup_size as u64;
        let output_size = active_subgroups * shape.m as u64 * shape.n as u64 * 4;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let input_a = context.create_buffer(input_a_size, usage)?;
        let input_b = context.create_buffer(input_b_size, usage)?;
        let output = context.create_buffer(output_size, usage)?;
        // SAFETY: all created Vulkan objects are owned by the returned harness.
        unsafe {
            let bindings = [0_u32, 1, 2].map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            });
            let descriptor_layout = context
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|problem| error("vulkan_descriptor_layout_failed", problem))?;
            let set_layouts = [descriptor_layout];
            let push_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(16)];
            let pipeline_layout = context
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_ranges),
                    None,
                )
                .map_err(|problem| error("vulkan_pipeline_layout_failed", problem))?;
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3)];
            let descriptor_pool = context
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(|problem| error("vulkan_descriptor_pool_failed", problem))?;
            let descriptor_set = context
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts),
                )
                .map_err(|problem| error("vulkan_descriptor_set_failed", problem))?[0];
            let buffer_infos = [
                vk::DescriptorBufferInfo::default()
                    .buffer(input_a.handle)
                    .range(input_a_size),
                vk::DescriptorBufferInfo::default()
                    .buffer(input_b.handle)
                    .range(input_b_size),
                vk::DescriptorBufferInfo::default()
                    .buffer(output.handle)
                    .range(output_size),
            ];
            let writes = [0_u32, 1, 2].map(|binding| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[binding as usize]))
            });
            context.device.update_descriptor_sets(&writes, &[]);
            let words = shader_words(FP16_PROFILE_SPV)?;
            let module = context
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(|problem| error("vulkan_shader_module_failed", problem))?;
            let specialization_values = [shape.m, shape.n, shape.k];
            let specialization_entries = [
                vk::SpecializationMapEntry {
                    constant_id: 0,
                    offset: 0,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 1,
                    offset: 4,
                    size: 4,
                },
                vk::SpecializationMapEntry {
                    constant_id: 2,
                    offset: 8,
                    size: 4,
                },
            ];
            let specialization = vk::SpecializationInfo::default()
                .map_entries(&specialization_entries)
                .data(words_as_bytes(&specialization_values));
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .specialization_info(&specialization);
            let pipeline_result = context.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            );
            context.device.destroy_shader_module(module, None);
            let pipeline = pipeline_result
                .map_err(|(_, problem)| error("vulkan_matrix_pipeline_failed", problem))?[0];
            let harness = Self {
                context,
                shape,
                workgroups,
                active_subgroups,
                bytes_per_tile,
                maximum_working_set_bytes,
                input_a,
                input_b,
                output,
                output_size,
                descriptor_pool,
                descriptor_layout,
                descriptor_set,
                pipeline_layout,
                pipeline,
            };
            context.measure(|device, command_buffer| {
                device.cmd_fill_buffer(
                    command_buffer,
                    harness.input_a.handle,
                    0,
                    input_a_size,
                    0x3c00_3c00,
                );
                device.cmd_fill_buffer(
                    command_buffer,
                    harness.input_b.handle,
                    0,
                    input_b_size,
                    0x3c00_3c00,
                );
                device.cmd_fill_buffer(command_buffer, harness.output.handle, 0, output_size, 0);
            })?;
            Ok(harness)
        }
    }

    fn actual_working_set(&self, requested: u64) -> u64 {
        (requested.min(self.maximum_working_set_bytes) / self.bytes_per_tile).max(1)
            * self.bytes_per_tile
    }

    fn tile_iterations(&self, working_set: u64) -> u64 {
        let tiles = working_set / self.bytes_per_tile;
        tiles.div_ceil(self.active_subgroups).max(1)
    }

    fn operations_per_dispatch(&self, working_set: u64, reuse_count: u32) -> u64 {
        self.active_subgroups
            * self.tile_iterations(working_set)
            * reuse_count as u64
            * self.shape.operations_per_mma()
    }

    fn traffic_bytes_per_dispatch(&self, working_set: u64) -> u64 {
        self.active_subgroups * self.tile_iterations(working_set) * self.bytes_per_tile
            + self.output_size
    }

    fn measure(
        &self,
        working_set: u64,
        reuse_count: u32,
        iterations: u32,
    ) -> Result<f64, BenchmarkError> {
        let tile_count = (working_set / self.bytes_per_tile) as u32;
        let tile_iterations = u32::try_from(self.tile_iterations(working_set)).map_err(|_| {
            BenchmarkError::new(
                "matrix_profile_too_large",
                "The matrix profile requires more tile iterations than the shader can index.",
            )
        })?;
        let params = matrix_profile_parameters(
            tile_count,
            tile_iterations,
            reuse_count,
            self.active_subgroups as u32,
        );
        self.context.measure(|device, command_buffer| unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &params,
            );
            for _ in 0..iterations {
                device.cmd_dispatch(command_buffer, self.workgroups, 1, 1);
            }
        })
    }
}

impl Drop for MatrixProfileHarness<'_> {
    fn drop(&mut self) {
        // SAFETY: measurements wait for the queue and the harness owns these child objects.
        unsafe {
            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.context
                .device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            for buffer in [&self.input_a, &self.input_b, &self.output] {
                self.context.device.destroy_buffer(buffer.handle, None);
                self.context.device.free_memory(buffer.memory, None);
            }
        }
    }
}

fn matrix_profile_parameters(
    tile_count: u32,
    tile_iterations: u32,
    reuse_count: u32,
    total_subgroups: u32,
) -> [u8; 16] {
    let mut bytes = [0_u8; 16];
    for (index, value) in [tile_count, tile_iterations, reuse_count, total_subgroups]
        .into_iter()
        .enumerate()
    {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn matrix_metadata(
    record: &AdapterRecord,
    kind: MatrixKind,
    shape: MatrixShape,
) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    metadata.insert("adapter_name".into(), record.vulkan.name.clone());
    metadata.insert("backend".into(), "Vulkan".into());
    metadata.insert("execution_backend".into(), "raw-vulkan".into());
    metadata.insert(
        "vulkan_device_uuid".into(),
        record.vulkan.device_uuid.clone(),
    );
    metadata.insert(
        "device_type".into(),
        record.vulkan.device_type_label().into(),
    );
    metadata.insert("timing_domain".into(), "gpu_timestamp".into());
    metadata.insert("execution_domain".into(), "cooperative_matrix".into());
    metadata.insert("api".into(), "VK_KHR_cooperative_matrix".into());
    metadata.insert("shader_format".into(), "embedded_spirv".into());
    metadata.insert("shader_source_language".into(), "GLSL".into());
    metadata.insert(
        "matrix_kernel_revision".into(),
        "vulkan-khr-cooperative-1".into(),
    );
    metadata.insert("matrix_m".into(), shape.m.to_string());
    metadata.insert("matrix_n".into(), shape.n.to_string());
    metadata.insert("matrix_k".into(), shape.k.to_string());
    metadata.insert("input_type".into(), kind.input_label().into());
    metadata.insert("accumulator_type".into(), kind.accumulator_label().into());
    metadata.insert("result_type".into(), kind.accumulator_label().into());
    metadata.insert(
        "subgroup_size".into(),
        record.vulkan.subgroup_size.to_string(),
    );
    metadata.insert("workgroup_size".into(), MATRIX_WORKGROUP_SIZE.to_string());
    metadata
}

fn matrix_device_metadata(record: &AdapterRecord) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    metadata.insert("device_id".into(), record.id.clone());
    metadata.insert("name".into(), record.vulkan.name.clone());
    metadata.insert("backend".into(), "Vulkan".into());
    metadata.insert("execution_backend".into(), "raw-vulkan".into());
    metadata.insert(
        "vulkan_device_uuid".into(),
        record.vulkan.device_uuid.clone(),
    );
    metadata.insert(
        "device_type".into(),
        record.vulkan.device_type_label().into(),
    );
    metadata
}

fn unsupported(record: &AdapterRecord, fallback_code: &str, message: &str) -> BenchmarkError {
    BenchmarkError::new(
        if record.cooperative.reason.is_empty() {
            fallback_code
        } else {
            &record.cooperative.reason
        },
        message,
    )
}

fn shader_words(bytes: &[u8]) -> Result<Vec<u32>, BenchmarkError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(BenchmarkError::new(
            "invalid_embedded_shader",
            "Embedded matrix SPIR-V is not word-aligned.",
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

unsafe fn find_physical_device(
    instance: &ash::Instance,
    uuid: &str,
) -> Result<vk::PhysicalDevice, BenchmarkError> {
    for physical_device in unsafe { instance.enumerate_physical_devices() }
        .map_err(|problem| error("vulkan_device_enumeration_failed", problem))?
    {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties) };
        if hex_bytes(&id.device_uuid) == uuid {
            return Ok(physical_device);
        }
    }
    Err(BenchmarkError::new(
        "adapter_not_found",
        "The selected Vulkan adapter UUID is no longer present.",
    ))
}

fn wrapped_timestamp_delta(start: u64, end: u64, valid_bits: u32) -> u64 {
    if valid_bits == 0 || valid_bits >= 64 {
        end.wrapping_sub(start)
    } else {
        end.wrapping_sub(start) & ((1_u64 << valid_bits) - 1)
    }
}

fn words_as_bytes(words: &[u32; 3]) -> &[u8] {
    // SAFETY: u32 has no padding and the byte slice borrows the input array.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 12) }
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn error(code: &str, problem: impl std::fmt::Display) -> BenchmarkError {
    BenchmarkError::new(code, problem.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_catalog_exposes_requested_dense_and_sparse_formats() {
        let ids = descriptors(&[])
            .into_iter()
            .map(|descriptor| descriptor.id)
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                FP16_MATRIX_ID,
                FP16_MATRIX_SCALING_ID,
                INT8_MATRIX_ID,
                FP8_MATRIX_ID,
                SPARSE_FP16_MATRIX_ID,
                SPARSE_INT8_MATRIX_ID,
                SPARSE_FP8_MATRIX_ID
            ]
        );
    }

    #[test]
    fn embedded_matrix_shaders_are_spirv() {
        for shader in [FP16_SPV, FP16_PROFILE_SPV, SINT8_SPV, UINT8_SPV] {
            assert_eq!(&shader[..4], &[0x03, 0x02, 0x23, 0x07]);
        }
    }

    #[test]
    fn matrix_operation_count_uses_discovered_shape() {
        let shape = MatrixShape {
            m: 16,
            n: 8,
            k: 32,
            input: vk::ComponentTypeKHR::FLOAT16,
            accumulator: vk::ComponentTypeKHR::FLOAT32,
        };
        assert_eq!(shape.operations_per_mma(), 8192);
    }

    #[test]
    fn timestamp_wrap_respects_valid_width() {
        assert_eq!(wrapped_timestamp_delta(250, 5, 8), 11);
    }

    #[test]
    fn matrix_profile_uses_power_of_two_targets_and_fp16_operand_bytes() {
        let partial_limit = 3 * 1024 * 1024;
        let tiers = matrix_profile_sizes(partial_limit);
        assert_eq!(tiers.last(), Some(&partial_limit));
        assert!(tiers.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(
            matrix_profile_sizes(1024 * 1024),
            [256 * 1024, 512 * 1024, 1024 * 1024]
        );
        let shape = MatrixShape {
            m: 16,
            n: 16,
            k: 16,
            input: vk::ComponentTypeKHR::FLOAT16,
            accumulator: vk::ComponentTypeKHR::FLOAT32,
        };
        assert_eq!(matrix_profile_bytes_per_tile(shape), 1024);
        assert_eq!(
            shape.operations_per_mma() * MATRIX_PROFILE_REUSE as u64,
            65_536
        );
    }

    #[test]
    fn matrix_profile_ignores_a_drop_that_later_recovers() {
        let point = |working_set_bytes, retained_ratio| MatrixProfilePoint {
            working_set_bytes,
            retained_ratio,
            compute: SampleStatistics {
                sample_count: 3,
                minimum: 99.0,
                median: 100.0,
                maximum: 101.0,
                standard_deviation: 1.0,
            },
            bandwidth: SampleStatistics::default(),
        };
        let points = [
            point(1, 1.0),
            point(2, 0.75),
            point(4, 0.72),
            point(8, 0.96),
            point(16, 0.94),
            point(32, 0.55),
            point(64, 0.53),
        ];
        assert_eq!(matrix_profile_transition(&points, 8), Some(5));
    }
}
