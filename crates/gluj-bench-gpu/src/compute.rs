use super::{
    AdapterRecord, GPU_PRECONDITION_MS, WORKGROUP_SIZE, common_metadata, device_metadata, done,
    duration_ns, ensure_not_cancelled, insert_clock_policy, per_sample_duration, statistics,
};
use crate::vulkan_compute_profile::{Precision, VulkanComputeProfileHarness};
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
const OFFLOAD_ID: &str = "gpu.performance.fp32.offload.scaling";
fn offload_percent(id: &str, config: &BenchmarkConfig) -> Result<u32, BenchmarkError> {
    if id == OFFLOAD_ID {
        let percent = config
            .options
            .get("ram_offload_percent")
            .map(|s| s.parse::<u32>())
            .transpose()
            .map_err(|_| {
                BenchmarkError::new(
                    "invalid_config",
                    "RAM offload must be 50, 75, or 100 percent.",
                )
            })?
            .unwrap_or(75);
        if ![50, 75, 100].contains(&percent) {
            return Err(BenchmarkError::new(
                "invalid_config",
                "RAM offload must be 50, 75, or 100 percent.",
            ));
        }
        return Ok(percent);
    }
    Ok(0)
}

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
    fn precision(self) -> Precision {
        match self {
            Self::Fp16 => Precision::Fp16,
            Self::Fp32 => Precision::Fp32,
            Self::Fp64 => Precision::Fp64,
        }
    }
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

    pub(super) fn supported_by_vulkan(self, record: &AdapterRecord) -> bool {
        match self {
            Self::Fp16 => {
                record.vulkan.shader_float16
                    && (record.vulkan.api_version >= ash::vk::API_VERSION_1_2
                        || record
                            .vulkan
                            .extensions
                            .contains("VK_KHR_shader_float16_int8"))
            }
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
    kind(id).is_some() || id == FP32_SCALING_ID || id == OFFLOAD_ID
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
                workload: "Native-precision vector FMA: register throughput or supported buffered datasets".into(),
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
        .filter(|descriptor| descriptor.id != FP32_ID)
        .collect::<Vec<_>>();
    let supported_device_ids = adapters
        .iter()
        .filter(|record| record.supports_vulkan_timestamps())
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();
    let available = !supported_device_ids.is_empty();
    descriptors.push(BenchmarkDescriptor {
        id: FP32_SCALING_ID.into(),
        name: "FP32 vector compute".into(),
        category: BenchmarkCategory::Gpu,
        workload: "GPU FP32 calculations with register, small-data, VRAM or custom working sets"
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
    {
        let mut descriptor = descriptors.last().unwrap().clone();
        descriptor.id = OFFLOAD_ID.into();
        descriptor.name = "FP32 system-RAM offload scaling".into();
        descriptor.workload =
            "GPU FP32 calculation with a configurable share of input/output data in system RAM"
                .into();
        descriptor.supported_device_ids = adapters
            .iter()
            .filter(|record| {
                record.supports_vulkan_timestamps()
                    && record.vulkan.device_type == ash::vk::PhysicalDeviceType::DISCRETE_GPU
                    && record.vulkan.host_memory_heap_bytes > 0
            })
            .map(|record| record.id.clone())
            .collect();
        descriptor.available = !descriptor.supported_device_ids.is_empty();
        descriptor.unavailable_reason = if descriptor.available {
            ""
        } else {
            "discrete_gpu_with_separate_host_memory_and_timestamps_required"
        }
        .into();
        descriptor.display_order = 104;
        descriptor.metadata.insert(
            "execution_domain".into(),
            "direct_host_memory_vector_compute".into(),
        );
        descriptor
            .metadata
            .insert("ram_offload_options".into(), "50,75,100".into());
        descriptors.push(descriptor);
    }
    descriptors
}

pub(super) fn run(
    record: &AdapterRecord,
    benchmark_id: &str,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
) -> Result<BenchmarkResult, BenchmarkError> {
    let _measurement_scope = gluj_bench_core::MeasurementScope::enter(config, benchmark_id);
    if benchmark_id == FP32_SCALING_ID
        && config.options.get("gpu_preset").map(String::as_str) == Some("register")
    {
        let mut result = run(record, FP32_ID, config, cancellation, progress)?;
        result.benchmark_id = FP32_SCALING_ID.into();
        result
            .workload_metadata
            .insert("gpu_preset".into(), "register".into());
        result
            .workload_metadata
            .insert("dataset_mode".into(), "register".into());
        return Ok(result);
    }
    if benchmark_id == FP32_SCALING_ID
        || benchmark_id == OFFLOAD_ID
        || (matches!(benchmark_id, FP16_ID | FP64_ID)
            && config
                .options
                .get("gpu_preset")
                .is_some_and(|p| p != "register"))
    {
        return run_fp32_scaling(record, benchmark_id, config, cancellation, progress);
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
    let primary_probe_rate = measure_rate_for_target(&harness, 0.20, cancellation)?;
    let higher_intensity_rate = measure_rate_for_target(&probe, 0.20, cancellation)?;
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
        let observed = gluj_bench_core::measure_batches(
            gluj_bench_core::sample_target_ns(config),
            cancellation,
            || harness.measure(sample_iterations),
        )?;
        let elapsed_ns = observed.active_ns;
        let measured_iterations = sample_iterations.saturating_mul(observed.batches);

        values.push(
            harness.operations_per_dispatch() as f64 * measured_iterations as f64
                / (elapsed_ns / 1e9),
        );
        gluj_bench_core::record_observation(
            "register",
            "compute",
            *values.last().unwrap(),
            "operations/s",
            harness
                .operations_per_dispatch()
                .saturating_mul(u64::from(measured_iterations)),
            "operations",
            elapsed_ns as u64,
            observed.elapsed_ns,
        );
    }
    // All dispatches in an observation are one sample, not independent repeats.
    let sample_statistics = statistics(&values);
    ensure_not_cancelled(cancellation)?;
    harness.validate_output()?;
    let implied_output_bandwidth = sample_statistics.median * 16.0
        / (LOOP_COUNT as f64 * OPERATIONS_PER_LOOP_PER_INVOCATION as f64);
    let mut metadata = common_metadata(record, "vulkan_gpu_timestamp");
    if matches!(kind, ComputeKind::Fp16 | ComputeKind::Fp64) {
        metadata.insert("gpu_preset".into(), "register".into());
        metadata.insert("dataset_mode".into(), "register".into());
    }
    metadata.insert("result_validation".into(), "boundary and interior output words checked against the precision-specific scalar FMA reference; FP16 relative tolerance 1%, FP32/FP64 0.01%; outside timing".into());
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
    metadata.insert("vector_kernel_revision".into(), "vulkan-packed-2".into());
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
    benchmark_id: &str,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    callback: &mut ProgressCallback<'_>,
) -> Result<BenchmarkResult, BenchmarkError> {
    let kind = if benchmark_id == FP16_ID {
        ComputeKind::Fp16
    } else if benchmark_id == FP64_ID {
        ComputeKind::Fp64
    } else {
        ComputeKind::Fp32
    };
    let precision = kind.precision();
    let started = Instant::now();
    let realizations = config
        .options
        .get("access_realizations")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1);
    let root = config
        .options
        .get("access_seed")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(gluj_bench_core::DEFAULT_ACCESS_SEED);
    let access_mode = match config
        .options
        .get("access_pattern")
        .map(String::as_str)
        .unwrap_or("sequential")
    {
        "sequential" => 0,
        "shuffled_blocks" => 1,
        "rotating_regions" => 2,
        "lane_scattered" => 3,
        _ => {
            return Err(BenchmarkError::new(
                "invalid_config",
                "Unsupported GPU access pattern.",
            ));
        }
    };
    ensure_not_cancelled(cancellation)?;
    let offload = offload_percent(benchmark_id, config)?;
    let traffic_limit = if offload > 0 {
        crate::traffic::TrafficLimit::from_config(config)?
    } else {
        None
    };
    if offload > 0
        && (record.vulkan.device_type != ash::vk::PhysicalDeviceType::DISCRETE_GPU
            || record.vulkan.host_memory_heap_bytes == 0)
    {
        return Err(BenchmarkError::new(
            "host_offload_unsupported",
            "This PCIe offload profile requires a discrete GPU with a separate host-visible system-memory heap.",
        ));
    }
    // Validate settings and allocation limits before running the reference kernel.
    let budget_percent = gluj_bench_core::vram_budget_percent(config)?;
    let vram_budget =
        gluj_bench_core::vram_budget_bytes(record.vulkan.device_local_memory_bytes, budget_percent)
            .saturating_sub(16 * 1024 * 1024);
    let ram_percent = ram_budget_percent(config)?;
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let ram_budget = gluj_bench_core::gpu_offload_ram_budget(
        system.total_memory(),
        system.available_memory(),
        record.vulkan.host_memory_heap_bytes,
        ram_percent,
    );
    let allocation_budget =
        gluj_bench_core::gpu_mixed_allocation_limit(vram_budget, ram_budget, offload);
    let mut maximum = record
        .vulkan
        .max_storage_buffer_range
        .saturating_mul(3)
        .min(allocation_budget);
    let (mut minimum, _, mut alignment) = gluj_bench_core::gpu_fp32_dataset_range(
        record.vulkan.device_local_memory_bytes,
        record.vulkan.max_storage_buffer_range,
        record.vulkan.min_storage_buffer_offset_alignment,
        budget_percent,
        None,
    );
    if offload == 0 {
        let shared = matches!(
            record.vulkan.device_type,
            ash::vk::PhysicalDeviceType::INTEGRATED_GPU | ash::vk::PhysicalDeviceType::CPU
        )
        .then_some(system.available_memory());
        maximum = maximum.min(
            gluj_bench_core::gpu_fp32_dataset_range(
                record.vulkan.device_local_memory_bytes,
                record.vulkan.max_storage_buffer_range,
                record.vulkan.min_storage_buffer_offset_alignment,
                budget_percent,
                shared,
            )
            .1,
        );
    }
    if offload > 0 && config.options.contains_key("gpu_offload_preset") {
        let range = gluj_bench_core::gpu_fp32_offload_dataset_range(
            vram_budget,
            ram_budget,
            record.vulkan.max_storage_buffer_range,
            record.vulkan.min_storage_buffer_offset_alignment,
            offload,
        );
        minimum = range.0;
        maximum = maximum.min(range.1);
        alignment = range.2;
    }
    if let Some(value) = config.options.get("max_working_set_bytes") {
        let cap = value
            .parse::<u64>()
            .ok()
            .filter(|cap| *cap >= PROFILE_MIN_SIZE)
            .ok_or_else(|| {
                BenchmarkError::new(
                    "invalid_config",
                    "max_working_set_bytes must be an integer of at least 262144 bytes.",
                )
            })?;
        maximum = maximum.min(cap);
    }
    let safe_maximum = maximum;
    let (maximum, single_dataset) = if offload == 0 {
        configured_fp32_dataset(config, maximum, minimum)?
    } else {
        configured_gpu_dataset(config, maximum, minimum, "gpu_offload_preset")?
    };
    let requested_maximum = maximum;
    let maximum = if offload == 0 || config.options.contains_key("gpu_offload_preset") {
        maximum / alignment * alignment
    } else {
        maximum
    };
    let requested_sizes = if single_dataset {
        vec![maximum]
    } else {
        profile_sizes(maximum)
            .into_iter()
            .map(|size| {
                if offload > 0 && config.options.contains_key("gpu_offload_preset") {
                    size.max(minimum)
                } else {
                    size
                }
            })
            .collect()
    };
    if maximum
        < if offload == 0 {
            minimum
        } else {
            PROFILE_MIN_SIZE
        }
        || requested_sizes.len()
            < if single_dataset
                || config.options.contains_key("gpu_preset")
                || config.options.contains_key("gpu_offload_preset")
            {
                1
            } else {
                3
            }
    {
        return Err(BenchmarkError::new(
            "insufficient_gpu_memory",
            "At least three working-set tiers are required within the RAM/VRAM budgets and storage-buffer limits.",
        ));
    }
    let reference = {
        let mut reference_progress = |mut update: ProgressUpdate| {
            update.fraction *= 0.20;
            update.message = format!("Compute reference: {}", update.message);
            callback(update);
        };
        run(
            record,
            kind.id(),
            &{
                let mut reference = crate::scaling::reference_config(config);
                reference
                    .options
                    .insert("gpu_preset".into(), "register".into());
                reference
                    .options
                    .insert("dataset_mode".into(), "register".into());
                reference
            },
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
    progress(ProgressUpdate {
        fraction: 0.01,
        phase: "gpu_compute_profile_setup".into(),
        message: format!(
            "Allocating an GPU vector profile working set up to {} MiB",
            maximum / (1024 * 1024)
        ),
    });
    let probe_limit = vram_budget
        .min(record.vulkan.max_storage_buffer_range.saturating_mul(3))
        .min(safe_maximum)
        .min(200 * 1024 * 1024)
        .min(if single_dataset { maximum } else { u64::MAX })
        .min(if offload == 0 { safe_maximum } else { u64::MAX });
    progress(ProgressUpdate {
        fraction: 0.02, phase: "gpu_compute_saturation".into(),
        message: "Finding a repeatable GPU vector dataset peak; undersized warm-up probes are excluded from results.".into(),
    });
    let saturation = calibrate_precision_peak(
        record,
        probe_limit,
        loop_count,
        cancellation,
        precision,
        access_mode,
        root,
    )?;
    let start = saturation.start.div_ceil(alignment) * alignment;
    let start = start
        .max(minimum)
        .min(saturation.limit / alignment * alignment)
        .min(safe_maximum / alignment * alignment);
    // User-selected datasets are never enlarged to the calibration peak.
    let mut harness = if offload == 0 {
        if precision == Precision::Fp32 {
            VulkanComputeProfileHarness::new(record, maximum)?
        } else {
            VulkanComputeProfileHarness::new_precision(record, maximum, precision, 0)?
        }
    } else {
        VulkanComputeProfileHarness::new_with_offload(record, maximum, offload)?
    };
    harness.select_workgroup_size(saturation.workgroup)?;
    let maximum = harness.actual_working_set(maximum);
    let start = harness.actual_working_set(start).min(maximum);
    let mut sizes = if single_dataset {
        vec![maximum]
    } else {
        dense_profile_sizes(start, maximum)
    }
    .into_iter()
    .map(|size| harness.actual_working_set(size))
    .collect::<Vec<_>>();
    sizes.dedup();
    let launch_tuning = saturation.launch_tuning.clone();
    let sample_count = config.samples.max(1);
    let target_seconds = gluj_bench_core::sample_duration(config).as_secs_f64();
    let arithmetic_intensity = loop_count as f64 * 64.0 / precision.bytes() as f64;
    let mut timing_metrics = Vec::new();
    let mut timing_batches = Vec::new();
    let mut timing_batch_series = Vec::new();
    let mut dispatch_workgroups = Vec::new();
    progress(ProgressUpdate {
        fraction: 0.04,
        phase: "measurement_plan".into(),
        message: gluj_bench_core::measurement_estimate(config, sizes.len()),
    });
    let mut raw_points = Vec::with_capacity(sizes.len());
    for (index, working_set) in sizes.iter().copied().enumerate() {
        ensure_not_cancelled(cancellation)?;
        progress(ProgressUpdate {
            fraction: 0.04 + index as f64 / sizes.len() as f64 * 0.91,
            phase: "gpu_compute_profile_sweep".into(),
            message: format!(
                "Measuring GPU vector compute with a {} working set ({}/{})",
                format_size(working_set),
                index + 1,
                sizes.len()
            ),
        });
        let cold_ns = harness.measure(working_set, loop_count, 1)?;
        let warm_iterations = iterations_for_target(
            cold_ns,
            gluj_bench_core::measurement_budget(config).warmup_ns as f64 / 1e9,
        );
        {
            let _warmup = gluj_bench_core::phase_timer("warmup");
            gluj_bench_core::measure_batches(
                gluj_bench_core::measurement_budget(config).warmup_ns,
                cancellation,
                || harness.measure(working_set, loop_count, warm_iterations),
            )?;
        }
        ensure_not_cancelled(cancellation)?;
        let calibration_ns = harness.measure(working_set, loop_count, warm_iterations)?;
        let iterations =
            iterations_for_target(calibration_ns / f64::from(warm_iterations), target_seconds);
        let pass_rate = traffic_limit.map(|limit| limit.for_dataset(working_set));
        let iterations = if pass_rate.is_some() { 1 } else { iterations };
        // Prepare and warm the exact batch. Timed samples reuse its recording,
        // avoiding CPU recording gaps that can let clocks fall between samples.
        let _ = harness.measure(working_set, loop_count, iterations)?;
        let operations = harness.operations_per_dispatch(working_set, loop_count) as f64;
        let traffic = harness.traffic_bytes_per_dispatch(working_set) as f64;
        let mut compute_values = Vec::with_capacity(sample_count as usize);
        let mut bandwidth_values = Vec::with_capacity(sample_count as usize);
        let mut actual_batch_counts = Vec::new();
        let mut gpu_times = Vec::with_capacity(sample_count as usize);
        let mut end_to_end_times = Vec::with_capacity(sample_count as usize);
        let mut pass_rates = Vec::new();
        let mut active_compute_values = Vec::new();
        let mut completed_delays = Vec::new();
        let mut demand_series: BTreeMap<&str, (String, Vec<f64>)> = BTreeMap::new();
        for realization in 0..realizations {
            let seed = gluj_bench_core::derive_seed(root, 0, working_set, realization as u64);
            let _realization = gluj_bench_core::MeasurementScope::realization(
                seed,
                BTreeMap::from([
                    ("root_seed".into(), root.to_string()),
                    ("dataset_bytes".into(), working_set.to_string()),
                    (
                        "generator_revision".into(),
                        "gpu-block-affine-lanes-1".into(),
                    ),
                    ("realization".into(), realization.to_string()),
                ]),
            );
            harness.configure_access(access_mode, seed, 0);
            {
                let _warmup = gluj_bench_core::phase_timer("warmup");
                gluj_bench_core::measure_batches(
                    gluj_bench_core::measurement_budget(config).warmup_ns,
                    cancellation,
                    || harness.measure(working_set, loop_count, warm_iterations),
                )?;
            }
            let mut offered_queue = if pass_rate.is_some() {
                Some(harness.offered_queue(working_set, loop_count)?)
            } else {
                None
            };

            for sample in 0..sample_count {
                if pass_rate.is_none() {
                    harness.configure_access(access_mode, seed, sample);
                }
                ensure_not_cancelled(cancellation)?;
                if let (Some(rate), Some(queue)) = (pass_rate, offered_queue.as_mut()) {
                    let observed = queue.observe(
                        rate.0,
                        target_seconds.max(cold_ns / 1e9 * 2.0),
                        cancellation,
                    )?;
                    let count = observed.completed as f64;
                    completed_delays.extend_from_slice(&observed.completion_delays_ns);
                    if let Some(recorder) = gluj_bench_core::observation_recorder() {
                        recorder.with_realization(
                            seed,
                            BTreeMap::from([(
                                format!("completed_only_delays_ns:sample-{sample}"),
                                serde_json::to_string(&observed.completion_delays_ns)
                                    .expect("delays"),
                            )]),
                        );
                    }
                    if observed.completed > 0
                        && (!observed.gpu_ns.is_finite() || observed.gpu_ns <= 0.0)
                    {
                        return Err(BenchmarkError::new(
                            "invalid_timing",
                            "Completed GPU requests did not provide positive finite service timestamps.",
                        ));
                    }
                    gluj_bench_core::record_observation(
                        &format!("offered-window-bytes-{working_set}"),
                        "completed_compute_over_observation_window",
                        operations * count * 1e9 / observed.window_ns,
                        "operations/s",
                        (operations as u64).saturating_mul(observed.completed),
                        "operations",
                        observed.gpu_ns as u64,
                        observed.window_ns as u64,
                    );
                    let seconds = observed.window_ns / 1e9;
                    compute_values.push(operations * count / seconds);
                    bandwidth_values.push(traffic * count / seconds);
                    pass_rates.push(count / seconds);
                    if count > 0.0 {
                        active_compute_values.push(operations * count / (observed.gpu_ns / 1e9));
                        gpu_times.push(observed.gpu_ns / count);
                        end_to_end_times.push(observed.completion_ns / count);
                        gluj_bench_core::record_observation(
                            &format!("offered-window-bytes-{working_set}"),
                            "offered_mean_completion_delay",
                            observed.completion_ns / count,
                            "ns",
                            observed.completed,
                            "completed_requests",
                            observed.gpu_ns as u64,
                            observed.window_ns as u64,
                        );
                    }
                    for (suffix, unit, value) in [
                        ("requested_compute", "operations/s", operations * rate.0),
                        ("requested_bandwidth", "bytes/s", traffic * rate.0),
                        (
                            "requested_host_bandwidth",
                            "bytes/s",
                            traffic * f64::from(offload) / 100.0 * rate.0,
                        ),
                        ("offered_pass_rate", "passes/s", rate.0),
                        ("offered_requests", "requests", observed.offered as f64),
                        ("submitted_requests", "requests", observed.submitted as f64),
                        ("completed_requests", "requests", count),
                        (
                            "unserved_requests",
                            "requests",
                            observed.offered.saturating_sub(observed.completed) as f64,
                        ),
                        (
                            "demand_fulfillment",
                            "%",
                            count / observed.offered as f64 * 100.0,
                        ),
                        ("queue_wait", "ns", observed.queue_wait_ns / count.max(1.0)),
                        ("oldest_pending_age", "ns", observed.oldest_pending_ns),
                        ("mean_pending_age", "ns", observed.mean_pending_ns),
                        ("youngest_pending_age", "ns", observed.youngest_pending_ns),
                        ("observation_window", "ns", observed.window_ns),
                        ("request_window", "ns", observed.request_window_ns),
                        (
                            "pending_submitted_requests",
                            "requests",
                            observed.submitted.saturating_sub(observed.completed) as f64,
                        ),
                        (
                            "pending_unsubmitted_requests",
                            "requests",
                            observed.offered.saturating_sub(observed.submitted) as f64,
                        ),
                    ] {
                        if (suffix == "queue_wait" && observed.completed == 0)
                            || (suffix == "demand_fulfillment" && observed.offered == 0)
                        {
                            continue;
                        }
                        gluj_bench_core::record_observation(
                            &format!("offered-window-bytes-{working_set}"),
                            &format!("offered_{suffix}"),
                            value,
                            unit,
                            observed.completed,
                            "completed_requests",
                            observed.gpu_ns as u64,
                            observed.window_ns as u64,
                        );
                        demand_series
                            .entry(suffix)
                            .or_insert_with(|| (unit.into(), Vec::new()))
                            .1
                            .push(value);
                    }
                } else {
                    let observed = gluj_bench_core::measure_batches(
                        gluj_bench_core::sample_target_ns(config),
                        cancellation,
                        || harness.measure(working_set, loop_count, iterations),
                    )?;
                    let iterations = iterations.saturating_mul(observed.batches);
                    actual_batch_counts.push(iterations);
                    let elapsed_ns = observed.active_ns;
                    let end_to_end_ns = observed.elapsed_ns as f64;
                    gpu_times.push(crate::timing::per_pass(elapsed_ns, iterations));
                    end_to_end_times.push(crate::timing::per_pass(end_to_end_ns, iterations));
                    let seconds = elapsed_ns / 1e9;
                    active_compute_values.push(operations * f64::from(iterations) / seconds);
                    pass_rates.push(f64::from(iterations) / (end_to_end_ns / 1e9));
                    compute_values.push(operations * iterations as f64 / seconds);
                    bandwidth_values.push(traffic * iterations as f64 / seconds);
                    gluj_bench_core::record_observation(
                        &format!("bytes-{working_set}"),
                        "compute",
                        *compute_values.last().unwrap(),
                        "operations/s",
                        harness
                            .operations_per_dispatch(working_set, loop_count)
                            .saturating_mul(u64::from(iterations)),
                        "operations",
                        elapsed_ns as u64,
                        observed.elapsed_ns,
                    );
                    gluj_bench_core::record_observation(
                        &format!("bytes-{working_set}"),
                        "accounted_bandwidth",
                        *bandwidth_values.last().unwrap(),
                        "bytes/s",
                        harness
                            .traffic_bytes_per_dispatch(working_set)
                            .saturating_mul(u64::from(iterations)),
                        "accounted_bytes",
                        elapsed_ns as u64,
                        observed.elapsed_ns,
                    );
                }
            }
            ensure_not_cancelled(cancellation)?;
            drop(offered_queue);
            harness.validate_output(working_set, loop_count)?;
        } // each realization retains full policy coverage
        crate::timing::append_metrics(
            &mut timing_metrics,
            working_set,
            &gpu_times,
            &end_to_end_times,
        );
        for (suffix, value) in completion_quantiles(&completed_delays) {
            timing_metrics.push(Metric {
                name: format!("working_set_{working_set}.completed_only_completion_{suffix}"),
                value,
                unit: "ns".into(),
                statistics: statistics(&[value]),
            });
        }
        if !completed_delays.is_empty() {
            let value = completed_delays.len() as f64;
            timing_metrics.push(Metric {
                name: format!("working_set_{working_set}.completion_observation_count"),
                value,
                unit: "requests".into(),
                statistics: statistics(&[value]),
            });
        }
        let pass_stats = statistics(&pass_rates);
        timing_metrics.push(Metric {
            name: format!("working_set_{working_set}.pass_rate"),
            value: pass_stats.median,
            unit: "passes/s".into(),
            statistics: pass_stats,
        });
        if traffic_limit.is_some() && !active_compute_values.is_empty() {
            for (suffix, factor, unit) in [
                ("active_compute", 1.0, "operations/s"),
                ("active_bandwidth", traffic / operations, "bytes/s"),
                (
                    "active_host_bandwidth",
                    traffic / operations * f64::from(offload) / 100.0,
                    "bytes/s",
                ),
            ] {
                let values = active_compute_values
                    .iter()
                    .map(|n| n * factor)
                    .collect::<Vec<_>>();
                let stats = statistics(&values);
                timing_metrics.push(Metric {
                    name: format!("working_set_{working_set}.{suffix}"),
                    value: stats.median,
                    unit: unit.into(),
                    statistics: stats,
                });
            }
        }
        for (suffix, (unit, values)) in demand_series {
            let stats = statistics(&values);
            timing_metrics.push(Metric {
                name: format!("working_set_{working_set}.{suffix}"),
                value: stats.median,
                unit,
                statistics: stats,
            });
        }
        if let Some(&first) = actual_batch_counts.first() {
            if actual_batch_counts.iter().all(|value| *value == first) {
                timing_batches.push(format!("{working_set}:{first}"));
            }
            timing_batch_series.push(format!(
                "{working_set}:{}",
                actual_batch_counts
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("/")
            ));
        }
        dispatch_workgroups.push(format!(
            "{working_set}:{}",
            harness
                .dispatch_workgroups(working_set)
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("+")
        ));
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
            retained_ratio: if baseline > 0.0 {
                (compute.median / baseline).clamp(0.0, 2.0)
            } else {
                0.0
            },
            compute,
            bandwidth,
        })
        .collect::<Vec<_>>();
    let transition_index = sustained_transition(&points);
    let mut metrics = vec![Metric {
        name: if offload == 0 {
            "cache_resident_compute"
        } else {
            "small_set_compute"
        }
        .into(),
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
    let mut metadata = common_metadata(record, "vulkan_gpu_timestamp");
    metadata.insert("result_validation".into(),"boundary and interior output vectors in every tested region and dataset match the scalar GPU vector FMA reference; checks outside timing".into());
    metadata.insert(
        "gpu_timing_method".into(),
        "gpu_timestamp_and_host_elapsed_per_pass_v1".into(),
    );
    metadata.insert(
        "gpu_timing_definition".into(),
        gluj_bench_core::GPU_TIMING_EXPLANATION.into(),
    );
    metadata.insert("gpu_timing_batch_passes".into(), timing_batches.join(","));
    metadata.insert(
        "gpu_timing_batch_passes_series".into(),
        timing_batch_series.join(","),
    );
    metadata.insert("gpu_saturation_start_bytes".into(), start.to_string());
    metadata.insert(
        "gpu_saturation_probe_limit_bytes".into(),
        saturation.limit.to_string(),
    );
    metadata.insert("gpu_saturation_probe_log".into(), saturation.log);
    metadata.insert("gpu_saturation_policy".into(), "vram_same_kernel; earliest_repeatable_within_3_percent_of_peak; hidden_uncapped_warmup_probes; capped_at_200_mib_and_budgets; no_cache_capacity_claim".into());
    metadata.insert(
        "gpu_saturation_budget_limited".into(),
        (saturation.limit < 200 * 1024 * 1024).to_string(),
    );
    metadata.insert(
        "gpu_saturation_boundary_peak".into(),
        saturation.boundary_peak.to_string(),
    );
    metadata.insert(
        "gpu_saturation_peak_repeatable".into(),
        saturation.repeatable.to_string(),
    );
    metrics.push(Metric {
        name: "measured_dataset_compute_peak".into(),
        value: saturation.peak.median,
        unit: "operations/s".into(),
        statistics: saturation.peak,
    });

    metadata.insert(
        "gpu_profile_workgroup_size".into(),
        harness.workgroup_size().to_string(),
    );
    metadata.insert(
        "workgroup_size".into(),
        harness.workgroup_size().to_string(),
    );
    metadata.insert("gpu_profile_launch_tuning".into(), launch_tuning);
    metadata.insert(
        "gpu_profile_workgroups_per_pass".into(),
        dispatch_workgroups.join(","),
    );
    metadata.insert(
        "gpu_profile_warmup_policy".into(),
        "per-tier timed-batch warmup before calibration; excluded from samples".into(),
    );
    metadata.insert("small_dataset_limit".into(), "One invocation per vec4 input pair. Small datasets launch fewer workgroups; RAM and VRAM regions dispatch separately. Launch overhead and limited parallelism can lower small-dataset throughput. Arithmetic work per element stays fixed across sizes; repeated dispatches do not create additional independent workgroups.".into());
    metadata.insert("gpu_timing_samples_paired".into(), "true".into());
    metadata.insert("offered_timing_scope".into(),"per-window means over completed requests; windows without completions have no latency/capacity observation; pending requests remain explicit and are not completion times".into());
    if let Some(limit) = traffic_limit {
        metadata.insert(
            "timing_domain".into(),
            "offered_observation_window_with_gpu_service".into(),
        );
        metadata.insert("traffic_generation_policy".into(), "offered_load_v1".into());
        metadata.insert(
            "gpu_timing_method".into(),
            "gpu_service_and_arrival_to_observed_completion_v1".into(),
        );
        metadata.insert(
            "gpu_activity_policy".into(),
            "arrival_driven_no_activity_pacing".into(),
        );
        metadata.insert("gpu_timing_definition".into(), "GPU timestamps measure per-request service after preceding output dependencies. Request delay measures arrival to host-observed completion including queueing, submission and polling. Throughput credits only completions observed before cutoff; drain excluded.".into());
        metadata.insert("gpu_submission_queue_depth".into(), "8".into());
        metadata.insert("offered_load_accounting".into(), "fixed-rate arrivals independent of completions; eight real in-flight Vulkan requests; excess demand tracked as logical backlog without allocating additional datasets; observed completions credited only before cutoff; drain excluded; outstanding work at cutoff unserved; service timestamps exclude preceding output-dependency wait; host-observed arrival-to-completion latency includes queue/submission/polling".into());

        match limit {
            crate::traffic::TrafficLimit::Static(rate) => {
                metadata.insert("offload_passes_per_second".into(), rate.to_string());
                metadata.insert("offload_traffic_mode".into(), "static".into());
            }
            crate::traffic::TrafficLimit::Dynamic(bytes) => {
                metadata.insert("offload_traffic_bytes_per_second".into(), bytes.to_string());
                metadata.insert("offload_traffic_mode".into(), "dynamic".into());
                metadata.insert("offload_traffic_ramp_revision".into(), "sqrt-1mib-1".into());
            }
        }
        metadata.insert(
            "throughput_time_basis".into(),
            "completed_work_per_observation_window".into(),
        );
    }

    metadata.insert("gpu_command_recording_policy".into(), "record and warm exact sample batch before timing; sequential reuse with fresh GPU query resets/results on every submission".into());
    insert_clock_policy(&mut metadata);
    metadata.insert("execution_backend".into(), "raw-vulkan".into());
    metadata.insert(
        "execution_domain".into(),
        "memory_backed_vector_compute".into(),
    );
    if offload > 0 {
        metadata.insert(
            "execution_domain".into(),
            "direct_host_memory_vector_compute".into(),
        );
        metadata.insert("ram_offload_percent".into(), offload.to_string());
        metadata.insert("ram_budget_percent".into(), ram_percent.to_string());
        metadata.insert(
            "host_allocation_budget_bytes".into(),
            ram_budget.to_string(),
        );
        metadata.insert(
            "host_allocated_test_buffer_bytes".into(),
            (harness.maximum_working_set_bytes / 100 * offload as u64
                + harness.maximum_working_set_bytes % 100 * offload as u64 / 100)
                .to_string(),
        );
        metadata.insert(
            "device_allocated_test_buffer_bytes".into(),
            (harness.maximum_working_set_bytes * (100 - offload) as u64 / 100).to_string(),
        );
        metadata.insert(
            "host_memory_placement".into(),
            "HOST_VISIBLE on a non-DEVICE_LOCAL heap; BAR-mapped VRAM excluded".into(),
        );
        metadata.insert(
            "offload_profile_revision".into(),
            "explicit-host-regions-1".into(),
        );
        metadata.insert(
            "offload_tier_bytes".into(),
            points
                .iter()
                .map(|point| {
                    format!(
                        "{}:{}:{}",
                        point.working_set_bytes,
                        point.working_set_bytes * offload as u64 / 100,
                        point.working_set_bytes * (100 - offload) as u64 / 100
                    )
                })
                .collect::<Vec<_>>()
                .join(","),
        );
        metadata.insert("host_traffic_definition".into(), "nominal two input reads plus one output write for the RAM region; GPU caches can serve repeated accesses; not measured PCIe bus counters".into());
        for point in &points {
            let fraction = offload as f64 / 100.0;
            let mut stats = point.bandwidth.clone();
            stats.minimum *= fraction;
            stats.median *= fraction;
            stats.maximum *= fraction;
            stats.standard_deviation *= fraction;
            metrics.push(Metric {
                name: format!("working_set_{}.host_bandwidth", point.working_set_bytes),
                value: stats.median,
                unit: "bytes/s".into(),
                statistics: stats,
            });
        }
    }
    metadata.insert("shader_format".into(), "embedded_spirv".into());
    metadata.insert("shader_source_language".into(), "GLSL".into());
    metadata.insert(
        "compute_profile_revision".into(),
        "buffered-precision-access-8".into(),
    );
    metadata.insert("data_type".into(), kind.data_type().into());
    metadata.insert(
        "access_pattern".into(),
        config
            .options
            .get("access_pattern")
            .cloned()
            .unwrap_or_else(|| "sequential".into()),
    );
    metadata.insert("access_seed".into(), root.to_string());
    metadata.insert("access_realizations".into(), realizations.to_string());
    metadata.insert("completion_latency_distribution".into(),"completed-only host-observed completion delays; nearest-rank p50/p90 require 20 completions, p95 40, p99 100; pooled across equal-coverage windows/realizations; pending requests excluded and separately counted; quantiles are not confidence intervals".into());
    metadata.insert(
        "shader_build_manifest".into(),
        include_str!("../shaders/profile-build.json").into(),
    );
    metadata.insert(
        "calibration_access_pattern".into(),
        config
            .options
            .get("access_pattern")
            .cloned()
            .unwrap_or_else(|| "sequential".into()),
    );
    metadata.insert("access_revision".into(), "gpu-block-affine-lanes-1".into());
    metadata.insert("access_ownership".into(),"bijective permutation within each descriptor chunk; each invocation owns one output vector; shuffled blocks preserve adjacent lanes, lane_scattered changes coalescing; rotating full-pass start does not measure cache eviction".into());
    metadata.insert(
        "storage_bytes_per_value".into(),
        (precision.vector_bytes() / 4).to_string(),
    );
    metadata.insert("operation_definition".into(),"8 independent four-lane FMAs per loop; 64 operations; FP16 rounds each FMA and output sum to binary16, GPU vector/64 use native arithmetic".into());
    if gluj_bench_core::gpu_buffered_vector(benchmark_id) {
        metadata.insert(
            "dataset_mode".into(),
            if single_dataset { "single" } else { "sweep" }.into(),
        );
        let preset_key = if offload > 0 {
            "gpu_offload_preset"
        } else {
            "gpu_preset"
        };
        if let Some(preset) = config.options.get(preset_key) {
            metadata.insert(preset_key.into(), preset.clone());
        }
        metadata.insert(
            "requested_dataset_bytes".into(),
            requested_maximum.to_string(),
        );
        metadata.insert(
            "resolved_dataset_bytes".into(),
            sizes.last().unwrap().to_string(),
        );
    }
    metadata.insert("arithmetic_iterations".into(), loop_count.to_string());
    metadata.insert(
        "allocation_budget_bytes".into(),
        allocation_budget.to_string(),
    );
    metadata.insert(
        "allocated_test_buffer_bytes".into(),
        harness.maximum_working_set_bytes.to_string(),
    );
    metadata.insert("allocation_limit_note".into(), if offload == 0 {
        "VRAM budget limited by storage-buffer range and alignment, with 16 MiB reserved for overhead. Shared-memory GPUs also retain available system-memory headroom."
    } else {
        "RAM budget is limited by the host heap and 80% of available RAM, leaving at least 2 GiB and reserving 16 MiB for overhead. Any VRAM region also respects the VRAM budget. Driver storage-buffer limits and alignment cap the total."
    }.into());
    metadata.insert(
        "arithmetic_intensity_operations_per_byte".into(),
        format!("{arithmetic_intensity:.4}"),
    );
    metadata.insert("bytes_per_element".into(), precision.bytes().to_string());
    metadata.insert(
        "shader_access_width_bytes".into(),
        precision.vector_bytes().to_string(),
    );
    metadata.insert(
        "byte_definition".into(),
        format!(
            "two_{}_byte_input_reads_plus_one_{}_byte_output_write",
            precision.vector_bytes(),
            precision.vector_bytes()
        ),
    );
    metadata.insert(
        "baseline_operations_per_second".into(),
        format!("{baseline:.0}"),
    );
    metadata.insert("tested_tier_count".into(), points.len().to_string());
    metadata.insert("gpu_profile_points_per_doubling".into(), "8".into());
    metadata.insert(
        "gpu_profile_sample_target_ms".into(),
        format!("{:.3}", target_seconds * 1000.0),
    );
    metadata.insert("gpu_profile_sampling_policy".into(), "eight logarithmic intervals per size doubling; preserve coarse-grid sample duration; every plotted point is independently measured; no synthetic smoothing".into());

    metadata.insert(
        "profile_sample_count_per_tier".into(),
        (sample_count * realizations).to_string(),
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
        if offload > 0 && index > 0 {
            metadata.insert(
                "bandwidth_transition_previous_working_set_bytes".into(),
                points[index - 1].working_set_bytes.to_string(),
            );
            metadata.insert(
                "bandwidth_transition_host_bytes".into(),
                (points[index].working_set_bytes * offload as u64 / 100).to_string(),
            );
            metadata.insert(
                "bandwidth_transition_previous_host_bytes".into(),
                (points[index - 1].working_set_bytes * offload as u64 / 100).to_string(),
            );
            metadata.insert("bandwidth_transition_interpretation".into(), "First sampled sustained slowdown on the direct host-memory path, bracketed by adjacent dataset sizes; not a measured cache capacity or a named cache level. Total dataset includes both inputs and output across RAM and VRAM. Host-memory traffic can hit GPU caches and need not follow the same cache path as local VRAM.".into());
        }
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
    {
        ensure_not_cancelled(cancellation)?;
        harness.validate_output(points.last().unwrap().working_set_bytes, loop_count)?;
        metadata.insert(
            "result_validation".into(),
            "boundary_and_interior_outputs_of_each_region_match_precision_specific_scalar_fma"
                .into(),
        );
    }
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
            kind.id(),
            &{
                let mut reference = gluj_bench_core::reference_config(config, "reference-after");
                reference
                    .options
                    .insert("gpu_preset".into(), "register".into());
                reference
                    .options
                    .insert("dataset_mode".into(), "register".into());
                reference
            },
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
    if offload > 0 {
        // Host-path results support placement/workload advice, not core-clock trials.
        for key in [
            "suggested_core_underclock_trial_percent",
            "suggested_core_frequency_limit_reduction_percent",
            "frequency_trial_headroom_percent",
            "frequency_trial_anchor_operations_per_second",
            "idealized_core_headroom_percent",
            "idealized_bandwidth_increase_percent",
            "gpu_core_limit_trial_reductions_percent",
            "efficiency_search_revision",
            "tuning_confidence",
        ] {
            metadata.remove(key);
        }
        metadata.insert(
            "tuning_guidance".into(),
            gluj_bench_core::GPU_OFFLOAD_TUNING_GUIDANCE.into(),
        );
    }
    callback(done("GPU vector compute scaling profile completed."));
    let mut result_device_metadata = device_metadata(record);
    result_device_metadata.insert("execution_backend".into(), "raw-vulkan".into());
    Ok(BenchmarkResult {
        benchmark_id: if offload > 0 {
            OFFLOAD_ID
        } else {
            benchmark_id
        }
        .into(),
        device_id: record.id.clone(),
        elapsed_ns: duration_ns(started.elapsed()),
        metrics,
        workload_metadata: metadata,
        device_metadata: result_device_metadata,
    })
}

fn configured_fp32_dataset(
    config: &BenchmarkConfig,
    limit: u64,
    minimum: u64,
) -> Result<(u64, bool), BenchmarkError> {
    configured_gpu_dataset(config, limit, minimum, "gpu_preset")
}

pub(super) fn configured_gpu_dataset(
    config: &BenchmarkConfig,
    limit: u64,
    minimum: u64,
    preset_key: &str,
) -> Result<(u64, bool), BenchmarkError> {
    let preset = config
        .options
        .get(preset_key)
        .map(String::as_str)
        .unwrap_or("legacy");
    if preset == "legacy" {
        return gluj_bench_core::configured_dataset(config, limit);
    }
    let mode = config
        .options
        .get("dataset_mode")
        .map(String::as_str)
        .unwrap_or("single");
    if !["single", "sweep", "automatic"].contains(&mode) {
        return Err(BenchmarkError::new(
            "invalid_config",
            "GPU dataset mode must be single or sweep.",
        ));
    }
    let bytes = match preset {
        "small" => minimum,
        "vram" => limit,
        "budget" if preset_key == "gpu_offload_preset" => limit,
        "custom" => config
            .options
            .get("dataset_bytes")
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| {
                BenchmarkError::new("invalid_config", "Choose a positive GPU dataset size.")
            })?,
        _ => {
            return Err(BenchmarkError::new(
                "invalid_config",
                "Unknown GPU data preset.",
            ));
        }
    };
    if bytes < minimum {
        return Err(BenchmarkError::new(
            "invalid_config",
            format!("GPU dataset must be at least {minimum} bytes."),
        ));
    }
    if bytes > limit {
        return Err(BenchmarkError::new(
            "dataset_exceeds_budget",
            format!(
                "GPU dataset is {bytes} bytes; the selected VRAM budget and device limits allow {limit} bytes."
            ),
        ));
    }
    Ok((bytes, mode == "single"))
}

fn ram_budget_percent(config: &BenchmarkConfig) -> Result<u32, BenchmarkError> {
    let percent = config
        .options
        .get("ram_budget_percent")
        .map(|s| s.parse::<u32>())
        .transpose()
        .map_err(|_| {
            BenchmarkError::new(
                "invalid_config",
                "RAM budget must be between 20 and 80 percent.",
            )
        })?
        .unwrap_or(20);
    if !(20..=80).contains(&percent) {
        return Err(BenchmarkError::new(
            "invalid_config",
            "RAM budget must be between 20 and 80 percent.",
        ));
    }
    Ok(percent)
}

struct DatasetPeak {
    start: u64,
    limit: u64,
    peak: SampleStatistics,
    workgroup: u64,
    launch_tuning: String,
    log: String,
    boundary_peak: bool,
    repeatable: bool,
}

fn peak_start_index(probes: &[(u64, SampleStatistics)]) -> usize {
    let stable = |s: &SampleStatistics| {
        s.sample_count >= 3
            && s.median.is_finite()
            && s.median > 0.0
            && s.standard_deviation / s.median <= 0.03
    };
    let peak = probes
        .iter()
        .enumerate()
        .filter(|(_, (_, s))| stable(s))
        .max_by(|a, b| a.1.1.median.total_cmp(&b.1.1.median))
        .or_else(|| {
            probes
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.1.median.total_cmp(&b.1.1.median))
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    probes
        .iter()
        .position(|(_, s)| stable(s) && s.median >= probes[peak].1.median * 0.97)
        .unwrap_or(peak)
}

fn calibrate_precision_peak(
    record: &AdapterRecord,
    limit: u64,
    loops: u32,
    cancellation: &CancellationToken,
    precision: Precision,
    access_mode: u32,
    root: u64,
) -> Result<DatasetPeak, BenchmarkError> {
    if limit < PROFILE_MIN_SIZE {
        return Err(BenchmarkError::new(
            "insufficient_gpu_memory",
            "The VRAM budget is too small to calibrate a safe FP32 starting size. Increase the VRAM allocation budget.",
        ));
    }
    let _calibration_scope = gluj_bench_core::MeasurementScope::phase("calibration-dataset-peak");
    let budget = gluj_bench_core::current_reference_budget();
    // This owner is dropped before allocating the actual (possibly RAM-backed) test.
    let mut control = VulkanComputeProfileHarness::new_precision(record, limit, precision, 0)?;
    let limit = control.actual_working_set(limit);
    let mut sizes = vec![
        control.actual_working_set(PROFILE_MIN_SIZE),
        control.actual_working_set(512 * 1024),
    ];
    for mib in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 200] {
        if mib * 1024 * 1024 <= limit {
            sizes.push(control.actual_working_set(mib * 1024 * 1024));
        }
    }
    sizes.push(limit);
    sizes.sort_unstable();
    sizes.dedup();
    control.configure_access(
        access_mode,
        gluj_bench_core::derive_seed(root, 0, limit, 0),
        0,
    );
    let launch_tuning = control.autotune_workgroups(&sizes, loops, cancellation)?;
    let mut probes = Vec::new();
    for &size in &sizes {
        control.configure_access(
            access_mode,
            gluj_bench_core::derive_seed(root, 0, size, 0),
            0,
        );
        ensure_not_cancelled(cancellation)?;
        let cold = control.measure(size, loops, 1)?;
        let passes = iterations_for_target(cold, budget.minimum_sample_ns as f64 / 1e9);
        for _ in 0..2 {
            ensure_not_cancelled(cancellation)?;
            control.measure(size, loops, passes)?;
        }
        let mut values = Vec::new();
        for _ in 0..budget.minimum_samples {
            ensure_not_cancelled(cancellation)?;
            let observed =
                gluj_bench_core::measure_batches(budget.minimum_sample_ns, cancellation, || {
                    control.measure(size, loops, passes)
                })?;
            let ns = observed.active_ns;
            let passes = passes.saturating_mul(observed.batches);
            values.push(
                control.operations_per_dispatch(size, loops) as f64 * f64::from(passes)
                    / (ns / 1e9),
            );
            gluj_bench_core::record_observation(
                &format!("calibration-bytes-{size}"),
                "compute",
                *values.last().unwrap(),
                "operations/s",
                control
                    .operations_per_dispatch(size, loops)
                    .saturating_mul(u64::from(passes)),
                "operations",
                ns as u64,
                observed.elapsed_ns,
            );
        }
        probes.push((size, statistics(&values)));
    }
    let index = peak_start_index(&probes);
    let start = probes[index].0;
    control.validate_output(start, loops)?;
    let repeatable = probes[index].1.standard_deviation / probes[index].1.median <= 0.03;
    let boundary_peak = probes.len() > 1
        && probes.last().is_some_and(|(_, s)| {
            s.median >= probes.iter().map(|(_, s)| s.median).fold(0.0, f64::max) * 0.97
                && s.median > probes[probes.len() - 2].1.median * 1.03
        });
    let log = probes
        .iter()
        .map(|(size, s)| {
            format!(
                "{size}:{:.0}(cv={:.2}%)",
                s.median,
                s.standard_deviation / s.median * 100.0
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(DatasetPeak {
        start,
        limit,
        peak: probes[index].1.clone(),
        workgroup: control.workgroup_size(),
        launch_tuning,
        log,
        boundary_peak,
        repeatable,
    })
}

/// Keep each point's original sampling budget as the sweep grows denser.
#[cfg(test)]
fn coarse_profile_point_count(start: u64, maximum: u64, single: bool) -> usize {
    if single || maximum <= start {
        1
    } else {
        (maximum as f64 / start as f64).log2().ceil() as usize + 1
    }
}

fn dense_profile_sizes(start: u64, maximum: u64) -> Vec<u64> {
    if start == 0 || start > maximum {
        return Vec::new();
    }
    let mut sizes = vec![start];
    for step in 1..=512 {
        let size = (start as f64 * 2_f64.powf(f64::from(step) / 8.0)).round() as u64;
        if size >= maximum {
            break;
        }
        if size > *sizes.last().unwrap() {
            sizes.push(size);
        }
    }
    if sizes.last().copied() != Some(maximum) {
        sizes.push(maximum);
    }
    sizes
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
    // The small-size occupancy ramp is not a memory-pressure drop. Begin looking
    // only after the fastest repeatable tier, even if just one tier reaches it.
    let peak = points
        .iter()
        .enumerate()
        .filter(|(_, point)| {
            point.compute.median.is_finite()
                && point.compute.median > 0.0
                && point.compute.standard_deviation / point.compute.median <= 0.10
        })
        .max_by(|(_, left), (_, right)| left.retained_ratio.total_cmp(&right.retained_ratio))
        .map(|(index, _)| index)?;
    points
        .windows(2)
        .enumerate()
        .skip(peak + 1)
        .find_map(|(index, pair)| {
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
        let rate = measure_rate_for_target(&harness, 0.04, cancellation)?;
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
    cancellation: &CancellationToken,
) -> Result<f64, BenchmarkError> {
    let _scope = gluj_bench_core::MeasurementScope::phase("calibration-vector-launch");
    let budget = gluj_bench_core::current_reference_budget();
    let target = (target_seconds * 1e9).max(budget.minimum_sample_ns as f64) as u64;
    let trial_ns = harness.measure(1)?;
    let iterations = iterations_for_target(
        trial_ns,
        gluj_bench_core::gpu_burst_duration(std::time::Duration::from_nanos(target)).as_secs_f64(),
    );
    let mut values = Vec::new();
    for _ in 0..budget.minimum_samples {
        let observed =
            gluj_bench_core::measure_batches(target, cancellation, || harness.measure(iterations))?;
        let completed = iterations.saturating_mul(observed.batches);
        let work = harness
            .operations_per_dispatch()
            .saturating_mul(u64::from(completed));
        let value = work as f64 * 1e9 / observed.active_ns;
        values.push(value);
        gluj_bench_core::record_observation(
            &format!("dispatch-ops-{}", harness.operations_per_dispatch()),
            "compute",
            value,
            "operations/s",
            work,
            "operations",
            observed.active_ns as u64,
            observed.elapsed_ns,
        );
    }
    Ok(statistics(&values).median)
}

fn iterations_for_target(per_dispatch_ns: f64, target_seconds: f64) -> u32 {
    (target_seconds * 1e9 / per_dispatch_ns)
        .ceil()
        .clamp(1.0, 4096.0) as u32
}

fn completion_quantiles(values: &[f64]) -> Vec<(&'static str, f64)> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    [
        ("p50", 50, 20),
        ("p90", 90, 20),
        ("p95", 95, 40),
        ("p99", 99, 100),
    ]
    .into_iter()
    .filter(|(_, _, minimum)| sorted.len() >= *minimum)
    .map(|(name, percent, _)| (name, sorted[(sorted.len() * percent).div_ceil(100) - 1]))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offload_presets_validate_data_sizes_and_exclude_register_mode() {
        let mut config = BenchmarkConfig::default();
        let resolve =
            |c: &BenchmarkConfig| configured_gpu_dataset(c, 1032192, 294912, "gpu_offload_preset");
        config
            .options
            .insert("gpu_offload_preset".into(), "budget".into());
        config.options.insert("dataset_mode".into(), "sweep".into());
        assert_eq!(resolve(&config).unwrap(), (1032192, false));
        config
            .options
            .insert("gpu_offload_preset".into(), "small".into());
        config
            .options
            .insert("dataset_mode".into(), "single".into());
        assert_eq!(resolve(&config).unwrap(), (294912, true));
        config
            .options
            .insert("gpu_offload_preset".into(), "custom".into());
        config
            .options
            .insert("dataset_bytes".into(), "1032193".into());
        assert_eq!(resolve(&config).unwrap_err().code, "dataset_exceeds_budget");
        config
            .options
            .insert("dataset_bytes".into(), "294911".into());
        assert!(resolve(&config).is_err());
        config
            .options
            .insert("gpu_offload_preset".into(), "register".into());
        assert!(resolve(&config).is_err());
    }

    #[test]
    #[ignore = "runs fixed-arrival FP32 direct-host compute and validates achieved traffic and outputs"]
    fn gpu_offload_traffic_limit_smoke() {
        use gluj_bench_core::BenchmarkProvider;
        let mut provider = crate::GpuBandwidthProvider::discover();
        let device = provider
            .adapters
            .iter()
            .find(|r| {
                r.supports_vulkan_timestamps()
                    && r.vulkan.device_type == ash::vk::PhysicalDeviceType::DISCRETE_GPU
                    && r.vulkan.host_memory_heap_bytes > 0
            })
            .unwrap()
            .id
            .clone();
        let mut config = BenchmarkConfig {
            target_duration_ms: 150,
            samples: 3,
            ..Default::default()
        };
        config.options.extend([
            ("device_id".into(), device),
            ("gpu_offload_preset".into(), "custom".into()),
            ("dataset_mode".into(), "single".into()),
            ("dataset_bytes".into(), "2097152".into()),
            ("ram_offload_percent".into(), "75".into()),
            ("offload_passes_per_second".into(), "100".into()),
        ]);
        let result = provider
            .run(
                OFFLOAD_ID,
                &config,
                &CancellationToken::default(),
                &mut |_| {},
            )
            .unwrap();
        let size = result.workload_metadata["resolved_dataset_bytes"]
            .parse::<u64>()
            .unwrap();
        assert_eq!(result.workload_metadata["offload_passes_per_second"], "100");
        assert_eq!(
            result.workload_metadata["throughput_time_basis"],
            "completed_work_per_observation_window"
        );
        let value = |suffix: &str| {
            result
                .metrics
                .iter()
                .find(|m| m.name == format!("working_set_{size}.{suffix}"))
                .unwrap()
                .value
        };
        let rate = value("pass_rate");
        assert!(rate > 0.0 && rate <= 100.01);
        let saturated_start = result.workload_metadata["gpu_saturation_start_bytes"]
            .parse::<u64>()
            .unwrap();
        assert!(size >= saturated_start);
        assert!(saturated_start <= 200 * 1024 * 1024);
        println!(
            "SATURATION,start={saturated_start},resolved={size},probes={}",
            result.workload_metadata["gpu_saturation_probe_log"]
        );
        let expected_host = size as f64 * 0.75 * rate;
        assert!((value("host_bandwidth") / expected_host - 1.0).abs() < 0.01);
        assert!(value("active_compute") >= value("compute"));
        let intensity = 64.0 * 64.0 / 48.0;
        assert!(
            (value("active_compute") / value("active_host_bandwidth") - intensity / 0.75).abs()
                < 0.001
        );
        assert!(
            (value("active_bandwidth") / value("active_host_bandwidth") - 1.0 / 0.75).abs() < 0.001
        );
        println!(
            "TUNED,size={size},group={},active_tops={:.3},active_host_gbs={:.3},gpu_us={:.3},elapsed_us={:.3}; {}",
            result.workload_metadata["gpu_profile_workgroup_size"],
            value("active_compute") / 1e12,
            value("active_host_bandwidth") / 1e9,
            value("gpu_execution_time") / 1e3,
            value("end_to_end_time") / 1e3,
            result.workload_metadata["gpu_profile_launch_tuning"]
        );

        assert!(result.workload_metadata["result_validation"].starts_with("boundary_and_interior"));
        config.options.remove("offload_passes_per_second");
        let target = 8_u64 * 1024 * 1024;
        config.options.insert(
            "offload_traffic_bytes_per_second".into(),
            target.to_string(),
        );
        config.options.insert("dataset_mode".into(), "sweep".into());
        config
            .options
            .insert("dataset_bytes".into(), (4_u64 * 1024 * 1024).to_string());
        let result = provider
            .run(
                OFFLOAD_ID,
                &config,
                &CancellationToken::default(),
                &mut |_| {},
            )
            .unwrap();
        let rates = result
            .metrics
            .iter()
            .filter_map(|metric| {
                let size = metric
                    .name
                    .strip_prefix("working_set_")?
                    .strip_suffix(".pass_rate")?
                    .parse::<u64>()
                    .ok()?;
                Some((size, metric.value))
            })
            .collect::<Vec<_>>();
        assert!(!rates.is_empty());
        let start = result.workload_metadata["gpu_saturation_start_bytes"]
            .parse::<u64>()
            .unwrap();
        assert!(rates.iter().all(|(size, _)| *size >= start));
        for (size, rate) in rates {
            assert!(
                rate > 0.0
                    && rate * size as f64
                        <= gluj_bench_core::dynamic_offload_traffic(target as f64, size) * 1.001
            );
        }
        assert_eq!(result.workload_metadata["offload_traffic_mode"], "dynamic");
        assert!(
            !result
                .workload_metadata
                .contains_key("offload_passes_per_second")
        );
        assert!(result.workload_metadata["result_validation"].starts_with("boundary_and_interior"));
    }

    #[test]
    #[ignore = "runs direct system-RAM GPU compute for every supported offload share"]
    fn gpu_offload_modes_smoke() {
        use gluj_bench_core::BenchmarkProvider;
        let mut provider = crate::GpuBandwidthProvider::discover();
        let device = provider
            .adapters
            .iter()
            .find(|r| {
                r.supports_vulkan_timestamps()
                    && r.vulkan.device_type == ash::vk::PhysicalDeviceType::DISCRETE_GPU
                    && r.vulkan.host_memory_heap_bytes > 0
            })
            .expect("A discrete GPU with direct host-memory support is required")
            .id
            .clone();
        let mut config = BenchmarkConfig {
            target_duration_ms: 150,
            samples: 3,
            ..Default::default()
        };
        config.options.insert("device_id".into(), device);
        config
            .options
            .insert("max_working_set_bytes".into(), "1048576".into());
        config
            .options
            .insert("dataset_bytes".into(), "737281".into());
        for share in [50_u64, 75, 100] {
            config
                .options
                .insert("ram_offload_percent".into(), share.to_string());
            for (preset, mode) in [
                ("small", "single"),
                ("custom", "single"),
                ("custom", "sweep"),
                ("budget", "single"),
                ("budget", "sweep"),
            ] {
                config
                    .options
                    .insert("gpu_offload_preset".into(), preset.into());
                config.options.insert("dataset_mode".into(), mode.into());
                let result = provider
                    .run(
                        OFFLOAD_ID,
                        &config,
                        &CancellationToken::default(),
                        &mut |_| {},
                    )
                    .unwrap();
                assert_eq!(result.benchmark_id, OFFLOAD_ID);
                assert_eq!(result.workload_metadata["gpu_offload_preset"], preset);
                let resolved = result.workload_metadata["resolved_dataset_bytes"]
                    .parse::<u64>()
                    .unwrap();
                assert_eq!(resolved % 49152, 0);
                assert!(
                    resolved
                        >= result.workload_metadata["gpu_saturation_start_bytes"]
                            .parse::<u64>()
                            .unwrap()
                );
                assert_eq!(
                    result.workload_metadata["host_allocated_test_buffer_bytes"]
                        .parse::<u64>()
                        .unwrap(),
                    resolved * share / 100
                );
                assert_eq!(
                    result.workload_metadata["device_allocated_test_buffer_bytes"]
                        .parse::<u64>()
                        .unwrap(),
                    resolved * (100 - share) / 100
                );
                let tiers = result
                    .metrics
                    .iter()
                    .filter(|m| m.name.ends_with(".compute"))
                    .collect::<Vec<_>>();
                assert!(!tiers.is_empty());
                assert!(tiers.iter().all(|m| m.value > 0.0 && m.value.is_finite()));
                if mode == "single" {
                    assert_eq!(tiers.len(), 1);
                }
                assert!(
                    result
                        .metrics
                        .iter()
                        .any(|m| m.name.ends_with(".gpu_execution_time"))
                );
            }
        }
    }

    #[test]
    fn descriptors_keep_optional_hardware_types_visible() {
        let descriptors = descriptors(&[]);
        assert_eq!(descriptors.len(), 4);
        assert!(descriptors.iter().all(|descriptor| !descriptor.available));
        assert_eq!(descriptors[0].id, FP16_ID);
        assert_eq!(descriptors[1].id, FP64_ID);
        assert_eq!(descriptors[2].id, FP32_SCALING_ID);
        assert!(
            !descriptors
                .iter()
                .any(|descriptor| descriptor.id == FP32_ID)
        );
        assert_eq!(descriptors[0].metadata.get("api").unwrap(), "Vulkan");
        assert_eq!(
            descriptors[2].metadata.get("profile_axis").unwrap(),
            "working_set_bytes"
        );
    }

    #[test]
    fn gpu_presets_validate_custom_sizes_and_keep_legacy_requests() {
        let mut config = BenchmarkConfig::default();
        assert_eq!(
            configured_fp32_dataset(&config, 1044480, 270336).unwrap(),
            (1044480, false)
        );
        config.options.insert("gpu_preset".into(), "custom".into());
        config
            .options
            .insert("dataset_mode".into(), "single".into());
        config
            .options
            .insert("dataset_bytes".into(), "524289".into());
        assert_eq!(
            configured_fp32_dataset(&config, 1044480, 270336).unwrap(),
            (524289, true)
        );
        config
            .options
            .insert("dataset_bytes".into(), "1048576".into());
        assert_eq!(
            configured_fp32_dataset(&config, 1044480, 270336)
                .unwrap_err()
                .code,
            "dataset_exceeds_budget"
        );
        config
            .options
            .insert("dataset_bytes".into(), "262144".into());
        assert!(configured_fp32_dataset(&config, 1044480, 270336).is_err());
        config.options.insert("gpu_preset".into(), "small".into());
        assert_eq!(
            configured_fp32_dataset(&config, 1044480, 270336).unwrap(),
            (270336, true)
        );
        config.options.insert("gpu_preset".into(), "vram".into());
        config.options.insert("dataset_mode".into(), "sweep".into());
        assert_eq!(
            configured_fp32_dataset(&config, 1044480, 270336).unwrap(),
            (1044480, false)
        );
        config.options.insert("gpu_preset".into(), "l2".into());
        assert!(configured_fp32_dataset(&config, 1044480, 270336).is_err());
    }

    #[test]
    #[ignore = "runs GPU FP32 register, small-data and custom-size workloads on the current machine"]
    fn gpu_fp32_modes_smoke() {
        use gluj_bench_core::BenchmarkProvider;
        let mut provider = crate::GpuBandwidthProvider::discover();
        let adapter = provider
            .adapters
            .iter()
            .find(|record| {
                record.supports_vulkan_timestamps()
                    && record.vulkan.device_type != ash::vk::PhysicalDeviceType::CPU
            })
            .expect("A Vulkan GPU with timestamps is required for this hardware test");
        let device_id = adapter.id.clone();
        let mut config = BenchmarkConfig {
            target_duration_ms: 150,
            samples: 3,
            ..Default::default()
        };
        config.options.insert("device_id".into(), device_id);
        config
            .options
            .insert("max_working_set_bytes".into(), "1048576".into());
        config
            .options
            .insert("dataset_bytes".into(), "524289".into());
        for (preset, mode) in [
            ("register", "register"),
            ("small", "single"),
            ("custom", "single"),
            ("custom", "sweep"),
        ] {
            config.options.insert("gpu_preset".into(), preset.into());
            config.options.insert("dataset_mode".into(), mode.into());
            let result = provider
                .run(
                    FP32_SCALING_ID,
                    &config,
                    &CancellationToken::default(),
                    &mut |_| {},
                )
                .unwrap();
            assert_eq!(result.benchmark_id, FP32_SCALING_ID);
            assert_eq!(result.workload_metadata["gpu_preset"], preset);
            if preset == "register" {
                assert!(
                    result
                        .metrics
                        .iter()
                        .any(|metric| metric.name == "throughput" && metric.value > 0.0)
                );
                assert_eq!(result.workload_metadata["dataset_mode"], "register");
                assert!(
                    !result
                        .metrics
                        .iter()
                        .any(|metric| metric.name.starts_with("working_set_"))
                );
            } else {
                let tiers = result
                    .metrics
                    .iter()
                    .filter(|metric| metric.name.ends_with(".compute"))
                    .collect::<Vec<_>>();
                assert!(!tiers.is_empty());
                if mode == "single" {
                    assert_eq!(tiers.len(), 1);
                }
                assert!(
                    tiers
                        .iter()
                        .all(|metric| metric.value.is_finite() && metric.value > 0.0)
                );
                assert!(
                    result
                        .metrics
                        .iter()
                        .any(|metric| metric.name.ends_with(".gpu_execution_time"))
                );
                assert!(
                    result.workload_metadata["resolved_dataset_bytes"]
                        .parse::<u64>()
                        .unwrap()
                        >= result.workload_metadata["gpu_saturation_start_bytes"]
                            .parse::<u64>()
                            .unwrap()
                );
            }
        }
    }

    #[test]
    fn offload_budgets_limit_each_region_and_full_ram_needs_no_vram() {
        let gib = 1024 * 1024 * 1024;
        let overhead = 16 * 1024 * 1024;
        assert_eq!(
            gluj_bench_core::gpu_offload_ram_budget(64 * gib, 8 * gib, 32 * gib, 80),
            6 * gib - overhead
        );
        assert_eq!(
            gluj_bench_core::gpu_offload_ram_budget(64 * gib, gib, 32 * gib, 80),
            0
        );
        assert_eq!(
            gluj_bench_core::gpu_offload_ram_budget(64 * gib, 40 * gib, gib, 80),
            gib - overhead
        );
        assert_eq!(
            gluj_bench_core::gpu_mixed_allocation_limit(2 * gib, 6 * gib, 50),
            4 * gib
        );
        assert_eq!(
            gluj_bench_core::gpu_mixed_allocation_limit(2 * gib, 3 * gib, 75),
            4 * gib
        );
        assert_eq!(
            gluj_bench_core::gpu_mixed_allocation_limit(0, 3 * gib, 100),
            3 * gib
        );
        assert_eq!(
            gluj_bench_core::gpu_mixed_allocation_limit(2 * gib, 0, 0),
            2 * gib
        );
        let mut config = BenchmarkConfig::default();
        assert_eq!(ram_budget_percent(&config).unwrap(), 20);
        for invalid in ["19", "81", "invalid"] {
            config
                .options
                .insert("ram_budget_percent".into(), invalid.into());
            assert_eq!(
                ram_budget_percent(&config).unwrap_err().code,
                "invalid_config"
            );
        }
        let catalog = descriptors(&[]);
        let descriptor = catalog.iter().find(|d| d.id == OFFLOAD_ID).unwrap();
        assert!(!descriptor.available);
        assert!(descriptor.supported_device_ids.is_empty());
        for percent in [50, 75, 100] {
            config
                .options
                .insert("ram_offload_percent".into(), percent.to_string());
            assert_eq!(offload_percent(OFFLOAD_ID, &config).unwrap(), percent);
        }
        config
            .options
            .insert("ram_offload_percent".into(), "60".into());
        assert!(offload_percent(OFFLOAD_ID, &config).is_err());
        assert!(is_benchmark(OFFLOAD_ID));
        assert_eq!(
            catalog.iter().filter(|d| d.id.contains("offload")).count(),
            1
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
    #[ignore = "GPU FP32 calibrated start and reported tiers on the current machine"]
    fn gpu_fp32_peak_start_smoke() {
        use gluj_bench_core::BenchmarkProvider;
        let mut provider = crate::GpuBandwidthProvider::discover();
        let device = provider
            .adapters
            .iter()
            .find(|r| {
                r.supports_vulkan_timestamps()
                    && r.vulkan.device_type != ash::vk::PhysicalDeviceType::CPU
            })
            .unwrap()
            .id
            .clone();
        let mut config = BenchmarkConfig {
            samples: 3,
            target_duration_ms: 150,
            ..Default::default()
        };
        config.options.extend([
            ("device_id".into(), device),
            ("gpu_preset".into(), "custom".into()),
            ("dataset_mode".into(), "sweep".into()),
            ("dataset_bytes".into(), (256_u64 * 1024 * 1024).to_string()),
        ]);
        let result = provider
            .run(
                FP32_SCALING_ID,
                &config,
                &CancellationToken::default(),
                &mut |_| {},
            )
            .unwrap();
        let start = result.workload_metadata["gpu_saturation_start_bytes"]
            .parse::<u64>()
            .unwrap();
        let tiers = result
            .metrics
            .iter()
            .filter(|m| m.name.ends_with(".compute"))
            .map(|m| (m.name.clone(), m.value / 1e12))
            .collect::<Vec<_>>();
        assert!(tiers.iter().all(|(name, _)| {
            name.strip_prefix("working_set_")
                .unwrap()
                .strip_suffix(".compute")
                .unwrap()
                .parse::<u64>()
                .unwrap()
                >= start
        }));
        assert!(start <= 200 * 1024 * 1024);
        assert_eq!(
            result.workload_metadata["gpu_profile_points_per_doubling"],
            "8"
        );
        assert!(tiers.len() >= 3);
        assert!(
            result
                .metrics
                .iter()
                .any(|m| m.name == "measured_dataset_compute_peak" && m.value > 0.0)
        );
        println!(
            "FP32_PEAK,start={start},tiers={tiers:?},repeatable={},boundary_rising={}",
            result.workload_metadata["gpu_saturation_peak_repeatable"],
            result.workload_metadata["gpu_saturation_boundary_peak"]
        );
    }

    #[test]
    fn peak_start_excludes_occupancy_ramp_but_allows_small_gpu_plateaus() {
        let probes = |values: &[f64]| {
            values
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u64 + 1, statistics(&[*v; 5])))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            peak_start_index(&probes(&[7.0, 25.0, 40.0, 49.0, 50.0, 44.0, 35.0])),
            3
        );
        assert_eq!(peak_start_index(&probes(&[10.0, 10.1, 9.9, 8.0])), 0);
        assert_eq!(peak_start_index(&probes(&[7.0, 25.0, 40.0, 50.0])), 3);
        let mut noisy = probes(&[7.0, 100.0, 45.0, 44.5]);
        noisy[1].1.standard_deviation = 30.0;
        assert_eq!(peak_start_index(&noisy), 2);
        assert_eq!(peak_start_index(&probes(&[1.0])), 0);
    }

    #[test]
    fn dense_curve_measures_eight_intervals_per_doubling_and_keeps_endpoints() {
        let start = 64 * 1024 * 1024;
        let sizes = dense_profile_sizes(start, 4 * start);
        assert_eq!(sizes.len(), 17);
        assert_eq!(sizes[0], start);
        assert_eq!(sizes[8], 2 * start);
        assert_eq!(sizes[16], 4 * start);
        assert!(sizes.windows(2).all(|p| p[0] < p[1]));
        assert!(
            sizes
                .windows(2)
                .all(|p| (p[1] as f64 / p[0] as f64 - 2_f64.powf(0.125)).abs() < 1e-6)
        );
        assert_eq!(coarse_profile_point_count(start, 4 * start, false), 3);
        assert_eq!(coarse_profile_point_count(start, 4 * start, true), 1);
        let partial = dense_profile_sizes(start + 49152, 5 * start + 123);
        assert_eq!(partial.last(), Some(&(5 * start + 123)));
        assert_eq!(dense_profile_sizes(start, start), vec![start]);
        assert!(dense_profile_sizes(start, start - 1).is_empty());
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

        let occupancy_ramp_then_drop = [
            point(1, 0.3, 0.01),
            point(2, 0.5, 0.01),
            point(4, 1.0, 0.01),
            point(8, 0.85, 0.01),
            point(16, 0.5, 0.01),
            point(32, 0.2, 0.01),
        ];
        assert_eq!(sustained_transition(&occupancy_ramp_then_drop), Some(3));
        assert_eq!(sustained_transition(&occupancy_ramp_then_drop[..3]), None);
    }
    #[test]
    fn completed_only_quantiles_require_enough_observations_and_use_nearest_rank() {
        assert!(completion_quantiles(&[1.; 19]).is_empty());
        let values = (1..=100).rev().map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            completion_quantiles(&values),
            vec![("p50", 50.), ("p90", 90.), ("p95", 95.), ("p99", 99.)]
        );
        assert_eq!(completion_quantiles(&values[..20]).len(), 2);
        assert_eq!(completion_quantiles(&values[..40]).len(), 3);
    }
}
