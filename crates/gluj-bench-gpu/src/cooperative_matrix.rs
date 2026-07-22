use ash::{Entry, vk};
use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult,
    CancellationToken, Metric, ProgressCallback, ProgressUpdate,
};
use std::{collections::BTreeMap, ffi::CStr, time::Instant};
use wgpu::util::DeviceExt;

use super::{
    AdapterRecord, GPU_PRECONDITION_MS, GpuContext, common_metadata, create_buffer_checked,
    device_metadata, done, duration_ns, ensure_not_cancelled, insert_clock_policy, map_read,
    per_sample_duration, statistics, wait_for_gpu,
};

pub(super) const FP16_MATRIX_ID: &str = "gpu.performance.matrix.fp16";
pub(super) const INT8_MATRIX_ID: &str = "gpu.performance.matrix.int8";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub input: vk::ComponentTypeKHR,
    pub accumulator: vk::ComponentTypeKHR,
}

#[derive(Debug, Clone, Default)]
pub(super) struct CooperativeSupport {
    pub fp16: Option<MatrixShape>,
    pub int8: Option<MatrixShape>,
    pub reason: String,
}

pub(super) fn discover() -> BTreeMap<String, CooperativeSupport> {
    discover_inner().unwrap_or_default()
}

pub(super) fn is_matrix_benchmark(id: &str) -> bool {
    matches!(id, FP16_MATRIX_ID | INT8_MATRIX_ID)
}

pub(super) fn descriptors(adapters: &[AdapterRecord]) -> Vec<BenchmarkDescriptor> {
    [
        (
            FP16_MATRIX_ID,
            "FP16 cooperative-matrix performance",
            "fp16 matrix multiply-accumulate",
            true,
        ),
        (
            INT8_MATRIX_ID,
            "INT8 cooperative-matrix performance",
            "int8 matrix multiply-accumulate with int32 accumulation",
            false,
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (id, name, data_type, fp16))| {
        let supported_device_ids = adapters
            .iter()
            .filter(|record| fp16 && supports_fp16_wgpu(record))
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        let available = !supported_device_ids.is_empty();
        let unavailable_reason = if available {
            String::new()
        } else if adapters.is_empty() {
            "adapter_not_found".into()
        } else if !fp16
            && adapters
                .iter()
                .any(|record| record.cooperative.int8.is_some())
        {
            "int8_cooperative_matrix_shader_frontend_unsupported".into()
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
        metadata.insert("api".into(), "VK_KHR_cooperative_matrix".into());
        metadata.insert(
            "operation_counting".into(),
            "one MxNxK matrix multiply-accumulate equals 2*M*N*K operations".into(),
        );
        metadata.insert("capability_gated".into(), "true".into());
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
            display_order: 120 + index as u32,
            metadata,
        }
    })
    .collect()
}

const MATRIX_WORKGROUP_SIZE: u64 = 64;
const MATRIX_LOOP_COUNT: u64 = 256;
const MATRIX_OPERATIONS_PER_INSTRUCTION: u64 = 2 * 16 * 16 * 16;
const WORKGROUP_CANDIDATES: [u32; 5] = [512, 1024, 2048, 4096, 8192];

pub(super) fn run(
    record: &AdapterRecord,
    benchmark_id: &str,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
) -> Result<BenchmarkResult, BenchmarkError> {
    if benchmark_id == INT8_MATRIX_ID {
        return Err(BenchmarkError::new(
            "int8_cooperative_matrix_shader_frontend_unsupported",
            "The Vulkan driver exposes INT8 cooperative matrices, but wgpu/WGSL cannot safely express 8-bit cooperative-matrix operands yet.",
        ));
    }
    if benchmark_id != FP16_MATRIX_ID {
        return Err(BenchmarkError::new(
            "unsupported_benchmark",
            format!("Unknown cooperative-matrix benchmark '{benchmark_id}'."),
        ));
    }
    if !supports_fp16_wgpu(record) {
        return Err(BenchmarkError::new(
            "fp16_cooperative_matrix_unsupported",
            "The selected GPU does not expose a safe 16x16x16 FP16-input/FP32-accumulator cooperative-matrix path.",
        ));
    }

    let started = Instant::now();
    ensure_not_cancelled(cancellation)?;
    progress(ProgressUpdate {
        fraction: 0.02,
        phase: "gpu_matrix_setup".into(),
        message: "Compiling and validating the FP16 cooperative-matrix kernel".into(),
    });
    let required = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
        | wgpu::Features::SHADER_F16
        | wgpu::Features::SUBGROUP;
    let context = GpuContext::request_with_features(record, true, required)?;
    let (workgroups, subgroup_size) = autotune_workgroups(&context, cancellation)?;
    let harness = MatrixHarness::new(&context, workgroups)?;

    progress(ProgressUpdate {
        fraction: 0.12,
        phase: "gpu_matrix_precondition".into(),
        message: "Saturating the GPU matrix execution units before measurement".into(),
    });
    let initial_ns = harness.measure(&context, 1)?;
    let warm_iterations = iterations_for_target(initial_ns, GPU_PRECONDITION_MS / 1000.0);
    let _ = harness.measure(&context, warm_iterations)?;

    let sample_count = config.samples.max(1);
    let calibrated_ns = harness.measure(&context, 1)?;
    let sample_iterations =
        iterations_for_target(calibrated_ns, per_sample_duration(config).as_secs_f64());
    let operations_per_dispatch = harness.operations_per_dispatch(subgroup_size)?;
    let mut values = Vec::with_capacity(sample_count as usize);
    for sample in 0..sample_count {
        ensure_not_cancelled(cancellation)?;
        progress(ProgressUpdate {
            fraction: 0.20 + sample as f64 / sample_count as f64 * 0.75,
            phase: "gpu_matrix_measurement".into(),
            message: format!(
                "Measuring FP16 matrix throughput sample {}/{}",
                sample + 1,
                sample_count
            ),
        });
        let elapsed_ns = harness.measure(&context, sample_iterations)?;
        values.push(operations_per_dispatch as f64 * sample_iterations as f64 / (elapsed_ns / 1e9));
    }
    let sample_statistics = statistics(&values);
    let mut metadata = common_metadata(record, "gpu_timestamp");
    insert_clock_policy(&mut metadata);
    metadata.insert("execution_domain".into(), "cooperative_matrix".into());
    metadata.insert("api".into(), "VK_KHR_cooperative_matrix via wgpu".into());
    metadata.insert("matrix_m".into(), "16".into());
    metadata.insert("matrix_n".into(), "16".into());
    metadata.insert("matrix_k".into(), "16".into());
    metadata.insert("input_type".into(), "fp16".into());
    metadata.insert("accumulator_type".into(), "fp32".into());
    metadata.insert("result_type".into(), "fp32".into());
    metadata.insert("subgroup_size".into(), subgroup_size.to_string());
    metadata.insert("workgroup_size".into(), MATRIX_WORKGROUP_SIZE.to_string());
    metadata.insert("selected_workgroups".into(), workgroups.to_string());
    metadata.insert("working_set_bytes".into(), harness.output_size.to_string());
    metadata.insert(
        "matrix_mma_per_subgroup".into(),
        MATRIX_LOOP_COUNT.to_string(),
    );
    metadata.insert(
        "operations_per_matrix_mma".into(),
        MATRIX_OPERATIONS_PER_INSTRUCTION.to_string(),
    );
    metadata.insert(
        "operation_definition".into(),
        "one 16x16x16 matrix multiply-accumulate equals 8192 operations (4096 multiplies plus 4096 adds)".into(),
    );
    metadata.insert("bound_classification".into(), "compute_bound".into());
    metadata.insert("diagnostic_domain".into(), "gpu".into());
    metadata.insert(
        "classification_method".into(),
        "register-resident repeated matrix multiply-accumulate with only final result stores; arithmetic intensity is intentionally high".into(),
    );
    metadata.insert("classification_is_inference".into(), "true".into());
    metadata.insert(
        "gpu_core_saturation_policy".into(),
        "autotune normalized matrix throughput across 512 to 8192 workgroups".into(),
    );
    progress(done("GPU FP16 matrix benchmark completed."));
    Ok(BenchmarkResult {
        benchmark_id: FP16_MATRIX_ID.into(),
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

fn autotune_workgroups(
    context: &GpuContext,
    cancellation: &CancellationToken,
) -> Result<(u32, u32), BenchmarkError> {
    let max_workgroups = context.device.limits().max_compute_workgroups_per_dimension;
    let mut best = None;
    for workgroups in WORKGROUP_CANDIDATES
        .into_iter()
        .filter(|candidate| *candidate <= max_workgroups)
    {
        ensure_not_cancelled(cancellation)?;
        let harness = MatrixHarness::new(context, workgroups)?;
        let subgroup_size = harness.read_subgroup_size(context)?;
        let trial_ns = harness.measure(context, 1)?;
        let iterations = iterations_for_target(trial_ns, 0.04);
        let elapsed_ns = harness.measure(context, iterations)?;
        let rate = harness.operations_per_dispatch(subgroup_size)? as f64 * iterations as f64
            / (elapsed_ns / 1e9);
        if best.is_none_or(|(_, _, best_rate)| rate > best_rate) {
            best = Some((workgroups, subgroup_size, rate));
        }
    }
    best.map(|(workgroups, subgroup_size, _)| (workgroups, subgroup_size))
        .ok_or_else(|| {
            BenchmarkError::new(
                "compute_limits_unsupported",
                "The adapter cannot dispatch the minimum cooperative-matrix saturation workload.",
            )
        })
}

fn iterations_for_target(per_dispatch_ns: f64, target_seconds: f64) -> u32 {
    (target_seconds * 1e9 / per_dispatch_ns)
        .ceil()
        .clamp(1.0, 4096.0) as u32
}

struct MatrixHarness {
    workgroups: u32,
    output_size: u64,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    subgroup_sizes: wgpu::Buffer,
    subgroup_readback: wgpu::Buffer,
    query_set: wgpu::QuerySet,
    query_resolve: wgpu::Buffer,
    query_readback: wgpu::Buffer,
}

impl MatrixHarness {
    fn new(context: &GpuContext, workgroups: u32) -> Result<Self, BenchmarkError> {
        const MAX_SUBGROUPS_PER_WORKGROUP: u64 = 16;
        const MATRIX_RESULT_ELEMENTS: u64 = 16 * 16;
        let output_size = workgroups as u64
            * MAX_SUBGROUPS_PER_WORKGROUP
            * MATRIX_RESULT_ELEMENTS
            * size_of::<f32>() as u64;
        let output = create_buffer_checked(
            context,
            &wgpu::BufferDescriptor {
                label: Some("Gluj-Bench FP16 matrix output"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )?;
        let subgroup_sizes = create_buffer_checked(
            context,
            &wgpu::BufferDescriptor {
                label: Some("Gluj-Bench matrix subgroup sizes"),
                size: workgroups as u64 * size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let subgroup_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gluj-Bench matrix subgroup readback"),
            size: size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let fp16_tile = [0x20_u8, 0x00].repeat(16 * 16);
        let fp32_tile = vec![0_u8; 16 * 16 * size_of::<f32>()];
        let input_a = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gluj-Bench matrix input A"),
                contents: &fp16_tile,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let input_b = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gluj-Bench matrix input B"),
                contents: &fp16_tile,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let input_c = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gluj-Bench matrix input C"),
                contents: &fp32_tile,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gluj-Bench cooperative-matrix layout"),
                entries: &[
                    storage_layout_entry(0, false),
                    storage_layout_entry(1, false),
                    storage_layout_entry(2, true),
                    storage_layout_entry(3, true),
                    storage_layout_entry(4, true),
                ],
            });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Gluj-Bench cooperative-matrix pipeline layout"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
        let error_scope = context
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let module = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Gluj-Bench FP16 cooperative-matrix shader"),
                source: wgpu::ShaderSource::Wgsl(FP16_SHADER.into()),
            });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gluj-Bench FP16 cooperative-matrix pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(problem) = pollster::block_on(error_scope.pop()) {
            return Err(BenchmarkError::new(
                "matrix_shader_compilation_failed",
                format!(
                    "The capability-gated cooperative-matrix shader was rejected safely: {problem}"
                ),
            ));
        }
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Gluj-Bench cooperative-matrix bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: subgroup_sizes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: input_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: input_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: input_c.as_entire_binding(),
                    },
                ],
            });
        let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Gluj-Bench matrix timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let query_resolve = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gluj-Bench matrix timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let query_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gluj-Bench matrix timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let harness = Self {
            workgroups,
            output_size,
            pipeline,
            bind_group,
            subgroup_sizes,
            subgroup_readback,
            query_set,
            query_resolve,
            query_readback,
        };
        harness.warm(context)?;
        Ok(harness)
    }

    fn operations_per_dispatch(&self, subgroup_size: u32) -> Result<u64, BenchmarkError> {
        if subgroup_size == 0 || !MATRIX_WORKGROUP_SIZE.is_multiple_of(subgroup_size as u64) {
            return Err(BenchmarkError::new(
                "invalid_subgroup_size",
                format!("The GPU reported unsupported subgroup size {subgroup_size}."),
            ));
        }
        Ok(self.workgroups as u64
            * (MATRIX_WORKGROUP_SIZE / subgroup_size as u64)
            * MATRIX_LOOP_COUNT
            * MATRIX_OPERATIONS_PER_INSTRUCTION)
    }

    fn warm(&self, context: &GpuContext) -> Result<(), BenchmarkError> {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench matrix warmup"),
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

    fn read_subgroup_size(&self, context: &GpuContext) -> Result<u32, BenchmarkError> {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench subgroup-size readback"),
            });
        encoder.copy_buffer_to_buffer(
            &self.subgroup_sizes,
            0,
            &self.subgroup_readback,
            0,
            size_of::<u32>() as u64,
        );
        context.queue.submit([encoder.finish()]);
        let bytes = map_read(
            &context.device,
            &self.subgroup_readback,
            size_of::<u32>() as u64,
        )?;
        let subgroup_size = u32::from_le_bytes(bytes[..4].try_into().expect("u32 width"));
        drop(bytes);
        self.subgroup_readback.unmap();
        Ok(subgroup_size)
    }

    fn measure(&self, context: &GpuContext, iterations: u32) -> Result<f64, BenchmarkError> {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench timestamped matrix compute"),
            });
        {
            let timestamp_writes = wgpu::ComputePassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gluj-Bench matrix compute pass"),
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
                label: Some("Gluj-Bench matrix timestamp resolve"),
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
                "GPU matrix timestamp interval was not finite and positive.",
            ));
        }
        Ok(elapsed_ns)
    }
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn supports_fp16_wgpu(record: &AdapterRecord) -> bool {
    record.features.contains(
        wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
            | wgpu::Features::SHADER_F16
            | wgpu::Features::SUBGROUP,
    ) && record.cooperative_matrix_properties.iter().any(|property| {
        property.m_size == 16
            && property.n_size == 16
            && property.k_size == 16
            && property.ab_type == wgpu::CooperativeScalarType::F16
            && property.cr_type == wgpu::CooperativeScalarType::F32
    })
}

fn discover_inner() -> Result<BTreeMap<String, CooperativeSupport>, BenchmarkError> {
    // SAFETY: ash loads the system Vulkan loader and all handles created below are destroyed
    // before this function returns. Every queried structure has the required sType initialized.
    let entry = unsafe { Entry::load() }.map_err(|problem| {
        BenchmarkError::new(
            "vulkan_loader_unavailable",
            format!("Could not load Vulkan for cooperative-matrix discovery: {problem}"),
        )
    })?;
    let app_name = c"Gluj-Bench";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(1)
        .engine_name(app_name)
        .engine_version(1)
        .api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    // SAFETY: create_info only references app_info for this call and uses no custom allocator.
    let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(|problem| {
        BenchmarkError::new(
            "vulkan_instance_failed",
            format!("Could not create the Vulkan discovery instance: {problem}"),
        )
    })?;
    let result = (|| {
        // SAFETY: instance is live for all enumerations and property queries.
        let devices = unsafe { instance.enumerate_physical_devices() }.map_err(vulkan_error)?;
        let extension = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
        let mut discovered = BTreeMap::new();
        for physical_device in devices {
            // SAFETY: physical_device belongs to instance.
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            // SAFETY: Vulkan guarantees a NUL-terminated deviceName array.
            let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: physical_device belongs to instance.
            let extensions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) }
                    .map_err(vulkan_error)?;
            let exposes_extension = extensions.iter().any(|property| {
                // SAFETY: Vulkan guarantees a NUL-terminated extensionName array.
                (unsafe { CStr::from_ptr(property.extension_name.as_ptr()) })
                    == ash::khr::cooperative_matrix::NAME
            });
            if !exposes_extension {
                discovered.insert(
                    name.to_ascii_lowercase(),
                    CooperativeSupport {
                        reason: "cooperative_matrix_extension_unsupported".into(),
                        ..Default::default()
                    },
                );
                continue;
            }
            let mut cooperative_features =
                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            let mut features =
                vk::PhysicalDeviceFeatures2::default().push_next(&mut cooperative_features);
            // SAFETY: feature chain is valid and physical_device belongs to instance.
            unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
            if cooperative_features.cooperative_matrix == vk::FALSE {
                discovered.insert(
                    name.to_ascii_lowercase(),
                    CooperativeSupport {
                        reason: "cooperative_matrix_feature_unsupported".into(),
                        ..Default::default()
                    },
                );
                continue;
            }
            // SAFETY: extension support was checked and the queried physical device is live.
            let configurations = unsafe {
                extension.get_physical_device_cooperative_matrix_properties(physical_device)
            }
            .map_err(vulkan_error)?;
            let fp16 = configurations
                .iter()
                .filter(|property| {
                    property.scope == vk::ScopeKHR::SUBGROUP
                        && property.a_type == vk::ComponentTypeKHR::FLOAT16
                        && property.b_type == vk::ComponentTypeKHR::FLOAT16
                        && matches!(
                            property.c_type,
                            vk::ComponentTypeKHR::FLOAT16 | vk::ComponentTypeKHR::FLOAT32
                        )
                        && property.c_type == property.result_type
                })
                .max_by_key(|property| property.m_size * property.n_size * property.k_size)
                .map(matrix_shape);
            let int8 = configurations
                .iter()
                .filter(|property| {
                    property.scope == vk::ScopeKHR::SUBGROUP
                        && matches!(
                            property.a_type,
                            vk::ComponentTypeKHR::SINT8 | vk::ComponentTypeKHR::UINT8
                        )
                        && property.a_type == property.b_type
                        && property.c_type == vk::ComponentTypeKHR::SINT32
                        && property.result_type == vk::ComponentTypeKHR::SINT32
                })
                .max_by_key(|property| property.m_size * property.n_size * property.k_size)
                .map(matrix_shape);
            let reason = if fp16.is_none() && int8.is_none() {
                "cooperative_matrix_formats_unsupported".into()
            } else {
                String::new()
            };
            discovered.insert(
                name.to_ascii_lowercase(),
                CooperativeSupport { fp16, int8, reason },
            );
        }
        Ok(discovered)
    })();
    // SAFETY: no child Vulkan handles survive this function.
    unsafe { instance.destroy_instance(None) };
    result
}

fn matrix_shape(property: &vk::CooperativeMatrixPropertiesKHR<'_>) -> MatrixShape {
    MatrixShape {
        m: property.m_size,
        n: property.n_size,
        k: property.k_size,
        input: property.a_type,
        accumulator: property.c_type,
    }
}

fn vulkan_error(problem: vk::Result) -> BenchmarkError {
    BenchmarkError::new(
        "vulkan_capability_query_failed",
        format!("A Vulkan cooperative-matrix capability query failed: {problem}"),
    )
}

const FP16_SHADER: &str = r#"
enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read_write> subgroup_sizes: array<u32>;
@group(0) @binding(2) var<storage, read> input_a: array<f16>;
@group(0) @binding(3) var<storage, read> input_b: array<f16>;
@group(0) @binding(4) var<storage, read> input_c: array<f32>;

@compute @workgroup_size(64)
fn main(
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    if (local_index == 0u) {
        subgroup_sizes[workgroup_id.x] = subgroup_size;
    }
    let a = coopLoadT<coop_mat16x16<f16,A>>(&input_a[0], 16u);
    let b = coopLoad<coop_mat16x16<f16,B>>(&input_b[0], 16u);
    var c = coopLoadT<coop_mat16x16<f32,C>>(&input_c[0], 16u);
    for (var iteration = 0u; iteration < 256u; iteration++) {
        c = coopMultiplyAdd(a, b, c);
    }
    let base = (workgroup_id.x * 8u + subgroup_id) * 256u;
    coopStoreT(c, &output[base], 16u);
}
"#;

#[cfg(test)]
fn compile_wgsl(source: &str) -> Result<Vec<u32>, BenchmarkError> {
    let module = naga::front::wgsl::parse_str(source).map_err(|problem| {
        BenchmarkError::new(
            "matrix_shader_compilation_failed",
            format!("Could not parse the cooperative-matrix shader: {problem}"),
        )
    })?;
    let capabilities = naga::valid::Capabilities::all();
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
        .validate(&module)
        .map_err(|problem| {
            BenchmarkError::new(
                "matrix_shader_compilation_failed",
                format!("Could not validate the cooperative-matrix shader: {problem}"),
            )
        })?;
    let options = naga::back::spv::Options {
        lang_version: (1, 6),
        ..Default::default()
    };
    naga::back::spv::write_vec(
        &module,
        &info,
        &options,
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".into(),
        }),
    )
    .map_err(|problem| {
        BenchmarkError::new(
            "matrix_shader_compilation_failed",
            format!("Could not emit the cooperative-matrix SPIR-V: {problem}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_cooperative_shader_compiles_to_spirv() {
        let words = compile_wgsl(FP16_SHADER).expect("valid cooperative matrix shader");
        assert_eq!(words.first().copied(), Some(0x0723_0203));
    }
}
