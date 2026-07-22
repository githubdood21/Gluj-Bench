use super::{
    AdapterRecord, GPU_PRECONDITION_MS, GpuContext, WORKGROUP_SIZE, common_metadata,
    create_buffer_checked, device_metadata, done, duration_ns, ensure_not_cancelled,
    insert_clock_policy, map_read, per_sample_duration, statistics, wait_for_gpu,
};
use crate::compute_shader;
use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult,
    CancellationToken, Metric, ProgressCallback, ProgressUpdate,
};
use std::{collections::BTreeMap, time::Instant};
use wgpu::util::DeviceExt;

const FP32_ID: &str = "gpu.performance.fp32";
const FP16_ID: &str = "gpu.performance.fp16";
const FP64_ID: &str = "gpu.performance.fp64";
const INT32_ID: &str = "gpu.performance.int32";
const INT8_ID: &str = "gpu.performance.int8_packed";
const LOOP_COUNT: u32 = 1024;
const PROBE_LOOP_COUNT: u32 = 2048;
const AUTOTUNE_LOOP_COUNT: u32 = 128;
const OPERATIONS_PER_LOOP_PER_INVOCATION: u64 = 64;
const COMPUTE_BOUND_RATIO: f64 = 0.80;
const WORKGROUP_CANDIDATES: [u32; 5] = [512, 1024, 2048, 4096, 8192];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ComputeKind {
    Fp32,
    Fp16,
    Fp64,
    Int32,
    Int8Packed,
}

impl ComputeKind {
    fn id(self) -> &'static str {
        match self {
            Self::Fp32 => FP32_ID,
            Self::Fp16 => FP16_ID,
            Self::Fp64 => FP64_ID,
            Self::Int32 => INT32_ID,
            Self::Int8Packed => INT8_ID,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "FP32 vector shader performance",
            Self::Fp16 => "FP16 vector shader performance",
            Self::Fp64 => "FP64 vector shader performance",
            Self::Int32 => "INT32 vector shader performance",
            Self::Int8Packed => "Packed INT8 vector dot-product performance",
        }
    }

    fn data_type(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Fp64 => "fp64",
            Self::Int32 => "uint32",
            Self::Int8Packed => "packed int8 with int32 accumulation",
        }
    }

    fn workload(self) -> &'static str {
        match self {
            Self::Fp32 | Self::Fp16 | Self::Fp64 => {
                "Register-resident vector fused multiply-add throughput"
            }
            Self::Int32 => "Register-resident vector integer multiply-add throughput",
            Self::Int8Packed => "Packed four-lane signed INT8 dot-product accumulation throughput",
        }
    }

    fn shader(self) -> &'static str {
        match self {
            Self::Fp32 => compute_shader::FP32,
            Self::Fp16 => compute_shader::FP16,
            Self::Fp64 => compute_shader::FP64,
            Self::Int32 => compute_shader::INT32,
            Self::Int8Packed => compute_shader::INT8_PACKED,
        }
    }

    fn required_feature(self) -> wgpu::Features {
        match self {
            Self::Fp16 => wgpu::Features::SHADER_F16,
            Self::Fp64 => wgpu::Features::SHADER_F64,
            _ => wgpu::Features::empty(),
        }
    }

    fn unsupported_reason(self) -> &'static str {
        match self {
            Self::Fp16 => "shader_f16_unsupported",
            Self::Fp64 => "shader_f64_unsupported",
            _ => "",
        }
    }

    fn output_element_bytes(self) -> u64 {
        match self {
            Self::Fp16 => 8,
            Self::Fp64 => 32,
            _ => 16,
        }
    }

    fn instruction_class(self) -> &'static str {
        match self {
            Self::Fp32 | Self::Fp16 | Self::Fp64 => "vector_fused_multiply_add",
            Self::Int32 => "vector_integer_multiply_plus_add",
            Self::Int8Packed => "dot4i8packed_plus_int32_accumulate",
        }
    }
}

const KINDS: [ComputeKind; 5] = [
    ComputeKind::Fp32,
    ComputeKind::Fp16,
    ComputeKind::Fp64,
    ComputeKind::Int32,
    ComputeKind::Int8Packed,
];

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
                    record.features.contains(wgpu::Features::TIMESTAMP_QUERY)
                        && record.features.contains(kind.required_feature())
                })
                .map(|record| record.id.clone())
                .collect::<Vec<_>>();
            let available = !supported_device_ids.is_empty();
            let unavailable_reason = if available {
                ""
            } else if adapters
                .iter()
                .any(|record| record.features.contains(wgpu::Features::TIMESTAMP_QUERY))
            {
                kind.unsupported_reason()
            } else if adapters.is_empty() {
                "adapter_not_found"
            } else {
                "timestamp_query_unsupported"
            };
            let mut metadata = BTreeMap::new();
            metadata.insert("instruction_class".into(), kind.instruction_class().into());
            metadata.insert("execution_domain".into(), "vector_shader".into());
            metadata.insert(
                "operation_counting".into(),
                operation_definition(kind).into(),
            );
            if kind == ComputeKind::Int8Packed {
                metadata.insert("native_acceleration".into(), "unverified".into());
            }
            BenchmarkDescriptor {
                id: kind.id().into(),
                name: kind.name().into(),
                category: BenchmarkCategory::Gpu,
                workload: kind.workload().into(),
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
    if !record.features.contains(kind.required_feature()) {
        return Err(BenchmarkError::new(
            kind.unsupported_reason(),
            format!(
                "{} is disabled because the selected GPU does not expose the required hardware shader capability.",
                kind.name()
            ),
        ));
    }
    ensure_not_cancelled(cancellation)?;
    progress(ProgressUpdate {
        fraction: 0.02,
        phase: "gpu_compute_setup".into(),
        message: format!("Preparing and validating the {} kernel", kind.data_type()),
    });
    let context = GpuContext::request_with_features(record, true, kind.required_feature())?;
    let workgroups = autotune_workgroups(&context, kind, cancellation)?;
    let harness = ComputeHarness::new(&context, kind, workgroups, LOOP_COUNT)?;

    progress(ProgressUpdate {
        fraction: 0.12,
        phase: "gpu_compute_precondition".into(),
        message: "Saturating GPU execution cores before measurement".into(),
    });
    let initial_ns = harness.measure(&context, 1)?;
    let warm_iterations = iterations_for_target(initial_ns, GPU_PRECONDITION_MS / 1000.0);
    let _ = harness.measure(&context, warm_iterations)?;

    progress(ProgressUpdate {
        fraction: 0.20,
        phase: "gpu_compute_diagnosis".into(),
        message: "Checking arithmetic-intensity sensitivity".into(),
    });
    let probe = ComputeHarness::new(&context, kind, workgroups, PROBE_LOOP_COUNT)?;
    let primary_probe_rate = measure_rate_for_target(&context, &harness, 0.20)?;
    let higher_intensity_rate = measure_rate_for_target(&context, &probe, 0.20)?;
    let sensitivity_ratio = (primary_probe_rate / higher_intensity_rate).clamp(0.0, 2.0);

    let sample_count = config.samples.max(1);
    let target_per_sample = per_sample_duration(config).as_secs_f64();
    let calibrated_ns = harness.measure(&context, 1)?;
    let sample_iterations = iterations_for_target(calibrated_ns, target_per_sample);
    let mut values = Vec::with_capacity(sample_count as usize);
    for sample in 0..sample_count {
        ensure_not_cancelled(cancellation)?;
        progress(ProgressUpdate {
            fraction: 0.25 + sample as f64 / sample_count as f64 * 0.70,
            phase: "gpu_compute_measurement".into(),
            message: format!(
                "Measuring {} throughput sample {}/{}",
                kind.data_type(),
                sample + 1,
                sample_count
            ),
        });
        let elapsed_ns = harness.measure(&context, sample_iterations)?;
        values.push(
            harness.operations_per_dispatch() as f64 * sample_iterations as f64
                / (elapsed_ns / 1e9),
        );
    }
    let sample_statistics = statistics(&values);
    let implied_output_bandwidth = sample_statistics.median * kind.output_element_bytes() as f64
        / (LOOP_COUNT as f64 * OPERATIONS_PER_LOOP_PER_INVOCATION as f64);
    let classification = classify(sensitivity_ratio);
    let mut metadata = common_metadata(record, "gpu_timestamp");
    insert_clock_policy(&mut metadata);
    metadata.insert("data_type".into(), kind.data_type().into());
    metadata.insert("instruction_class".into(), kind.instruction_class().into());
    metadata.insert(
        "operation_definition".into(),
        operation_definition(kind).into(),
    );
    metadata.insert("operations_per_loop_per_invocation".into(), "64".into());
    metadata.insert("loop_count".into(), LOOP_COUNT.to_string());
    metadata.insert("independent_accumulator_chains".into(), "8".into());
    metadata.insert("selected_workgroups".into(), workgroups.to_string());
    metadata.insert(
        "selected_invocations".into(),
        (workgroups as u64 * WORKGROUP_SIZE).to_string(),
    );
    metadata.insert(
        "gpu_core_saturation_policy".into(),
        "autotune_normalized_throughput_across_512_to_8192_workgroups".into(),
    );
    metadata.insert("bound_classification".into(), classification.into());
    metadata.insert(
        "memory_sensitivity_ratio".into(),
        format!("{sensitivity_ratio:.4}"),
    );
    metadata.insert(
        "large_data_slowdown_percent".into(),
        format!("{:.2}", (1.0 - sensitivity_ratio.min(1.0)) * 100.0),
    );
    metadata.insert(
        "classification_method".into(),
        "normalized throughput at 1024 loops divided by throughput at 2048 loops; a ratio below 0.80 indicates memory/dispatch influence".into(),
    );
    metadata.insert("classification_is_inference".into(), "true".into());
    metadata.insert("diagnostic_domain".into(), "gpu".into());
    metadata.insert(
        "implied_output_bandwidth_bytes_per_second".into(),
        format!("{implied_output_bandwidth:.0}"),
    );
    metadata.insert("working_set_bytes".into(), harness.output_size.to_string());
    metadata.insert(
        "shader_capability_verified_before_dispatch".into(),
        "true".into(),
    );
    metadata.insert(
        "native_acceleration".into(),
        native_acceleration(kind).into(),
    );
    progress(done("GPU compute benchmark completed."));
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
        device_metadata: device_metadata(record),
    })
}

fn classify(sensitivity_ratio: f64) -> &'static str {
    if sensitivity_ratio < COMPUTE_BOUND_RATIO {
        "memory_bandwidth_bound"
    } else {
        "compute_bound"
    }
}

fn native_acceleration(kind: ComputeKind) -> &'static str {
    match kind {
        ComputeKind::Fp16 | ComputeKind::Fp64 => "hardware_capability_reported",
        ComputeKind::Int8Packed => "unverified_backend_may_lower_or_polyfill",
        _ => "core_shader_instruction_set",
    }
}

fn operation_definition(kind: ComputeKind) -> &'static str {
    match kind {
        ComputeKind::Fp32 | ComputeKind::Fp16 | ComputeKind::Fp64 => {
            "one scalar fused multiply-add equals two operations per vector lane"
        }
        ComputeKind::Int32 => {
            "one scalar integer multiply plus one scalar integer add equals two operations per vector lane"
        }
        ComputeKind::Int8Packed => {
            "one dot4I8Packed accumulation equals four INT8 multiplies plus four INT32 adds (eight operations)"
        }
    }
}

fn autotune_workgroups(
    context: &GpuContext,
    kind: ComputeKind,
    cancellation: &CancellationToken,
) -> Result<u32, BenchmarkError> {
    let max_workgroups = context.device.limits().max_compute_workgroups_per_dimension;
    let mut best = None;
    for workgroups in WORKGROUP_CANDIDATES
        .into_iter()
        .filter(|candidate| *candidate <= max_workgroups)
    {
        ensure_not_cancelled(cancellation)?;
        let harness = ComputeHarness::new(context, kind, workgroups, AUTOTUNE_LOOP_COUNT)?;
        let rate = measure_rate_for_target(context, &harness, 0.04)?;
        if best.is_none_or(|(_, best_rate)| rate > best_rate) {
            best = Some((workgroups, rate));
        }
    }
    best.map(|(workgroups, _)| workgroups).ok_or_else(|| {
        BenchmarkError::new(
            "compute_limits_unsupported",
            "The adapter cannot dispatch the minimum GPU saturation workload.",
        )
    })
}

fn measure_rate_for_target(
    context: &GpuContext,
    harness: &ComputeHarness,
    target_seconds: f64,
) -> Result<f64, BenchmarkError> {
    let trial_ns = harness.measure(context, 1)?;
    let iterations = iterations_for_target(trial_ns, target_seconds);
    let elapsed_ns = harness.measure(context, iterations)?;
    Ok(harness.operations_per_dispatch() as f64 * iterations as f64 / (elapsed_ns / 1e9))
}

fn iterations_for_target(per_dispatch_ns: f64, target_seconds: f64) -> u32 {
    (target_seconds * 1e9 / per_dispatch_ns)
        .ceil()
        .clamp(1.0, 4096.0) as u32
}

struct ComputeHarness {
    kind: ComputeKind,
    workgroups: u32,
    loop_count: u32,
    output_size: u64,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    query_set: wgpu::QuerySet,
    query_resolve: wgpu::Buffer,
    query_readback: wgpu::Buffer,
}

impl ComputeHarness {
    fn new(
        context: &GpuContext,
        kind: ComputeKind,
        workgroups: u32,
        loop_count: u32,
    ) -> Result<Self, BenchmarkError> {
        let invocation_count = workgroups as u64 * WORKGROUP_SIZE;
        let output_size = invocation_count
            .checked_mul(kind.output_element_bytes())
            .ok_or_else(|| BenchmarkError::new("size_overflow", "GPU output size overflowed."))?;
        let output = create_buffer_checked(
            context,
            &wgpu::BufferDescriptor {
                label: Some("Gluj-Bench GPU compute output"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let params = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gluj-Bench GPU compute parameters"),
                contents: &compute_parameters(loop_count, invocation_count as u32),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gluj-Bench GPU compute layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Gluj-Bench GPU compute pipeline layout"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
        let error_scope = context
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let module = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Gluj-Bench GPU compute shader"),
                source: wgpu::ShaderSource::Wgsl(kind.shader().into()),
            });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gluj-Bench GPU compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(problem) = pollster::block_on(error_scope.pop()) {
            return Err(BenchmarkError::new(
                "shader_compilation_failed",
                format!("The capability-gated GPU shader was rejected safely: {problem}"),
            ));
        }
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Gluj-Bench GPU compute bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Gluj-Bench GPU compute timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let query_resolve = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gluj-Bench GPU compute timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let query_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gluj-Bench GPU compute timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let harness = Self {
            kind,
            workgroups,
            loop_count,
            output_size,
            pipeline,
            bind_group,
            query_set,
            query_resolve,
            query_readback,
        };
        harness.warm(context)?;
        Ok(harness)
    }

    fn operations_per_dispatch(&self) -> u64 {
        self.workgroups as u64
            * WORKGROUP_SIZE
            * self.loop_count as u64
            * OPERATIONS_PER_LOOP_PER_INVOCATION
    }

    fn warm(&self, context: &GpuContext) -> Result<(), BenchmarkError> {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench GPU compute warmup"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
        context.queue.submit([encoder.finish()]);
        wait_for_gpu(&context.device)
    }

    fn measure(&self, context: &GpuContext, iterations: u32) -> Result<f64, BenchmarkError> {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench timestamped GPU compute"),
            });
        {
            let timestamp_writes = wgpu::ComputePassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gluj-Bench GPU compute pass"),
                timestamp_writes: Some(timestamp_writes),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            for _ in 0..iterations {
                pass.dispatch_workgroups(self.workgroups, 1, 1);
            }
        }
        context.queue.submit([encoder.finish()]);
        wait_for_gpu(&context.device)?;
        let mut resolve = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench GPU compute timestamp resolve"),
            });
        resolve.resolve_query_set(&self.query_set, 0..2, &self.query_resolve, 0);
        resolve.copy_buffer_to_buffer(&self.query_resolve, 0, &self.query_readback, 0, 16);
        context.queue.submit([resolve.finish()]);
        let bytes = map_read(&context.device, &self.query_readback, 16)?;
        let start = u64::from_le_bytes(bytes[0..8].try_into().expect("timestamp width"));
        let end = u64::from_le_bytes(bytes[8..16].try_into().expect("timestamp width"));
        drop(bytes);
        self.query_readback.unmap();
        let elapsed_ns =
            end.wrapping_sub(start) as f64 * context.queue.get_timestamp_period() as f64;
        if elapsed_ns <= 0.0 || !elapsed_ns.is_finite() {
            return Err(BenchmarkError::new(
                "device_lost",
                "GPU compute timestamp interval was not finite and positive.",
            ));
        }
        let _ = self.kind;
        Ok(elapsed_ns)
    }
}

fn compute_parameters(loop_count: u32, invocation_count: u32) -> [u8; 16] {
    let mut bytes = [0_u8; 16];
    bytes[0..4].copy_from_slice(&loop_count.to_le_bytes());
    bytes[4..8].copy_from_slice(&0x9e37_79b9_u32.to_le_bytes());
    bytes[8..12].copy_from_slice(&invocation_count.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_keep_optional_hardware_types_visible() {
        let descriptors = descriptors(&[]);
        assert_eq!(descriptors.len(), 5);
        assert!(descriptors.iter().all(|descriptor| !descriptor.available));
        assert_eq!(descriptors[1].id, FP16_ID);
        assert_eq!(descriptors[2].id, FP64_ID);
    }

    #[test]
    fn operation_accounting_is_consistent_across_types() {
        assert_eq!(OPERATIONS_PER_LOOP_PER_INVOCATION, 64);
        assert!(operation_definition(ComputeKind::Fp32).contains("two operations"));
        assert!(operation_definition(ComputeKind::Int8Packed).contains("eight operations"));
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
