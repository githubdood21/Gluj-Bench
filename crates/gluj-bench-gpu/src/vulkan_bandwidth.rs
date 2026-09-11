use crate::{
    AdapterRecord, DispatchMeasurement, GPU_PRECONDITION_MS, HOST_STAGING_BUFFER_COUNT,
    KernelFlavor, KernelOperation, WORKGROUP_SIZE, ensure_not_cancelled,
};
use ash::{Entry, vk};
use gluj_bench_core::{BenchmarkError, CancellationToken};
use std::{
    ptr,
    time::{Duration, Instant},
};

const READ_CACHE_SPV: &[u8] = include_bytes!("../shaders/spv/bandwidth_read_cache.spv");
const WRITE_CACHE_SPV: &[u8] = include_bytes!("../shaders/spv/bandwidth_write_cache.spv");
const COPY_CACHE_SPV: &[u8] = include_bytes!("../shaders/spv/bandwidth_copy_cache.spv");
const READ_STREAM_SPV: &[u8] = include_bytes!("../shaders/spv/bandwidth_read_stream.spv");
const WRITE_STREAM_SPV: &[u8] = include_bytes!("../shaders/spv/bandwidth_write_stream.spv");
const COPY_STREAM_SPV: &[u8] = include_bytes!("../shaders/spv/bandwidth_copy_stream.spv");

struct Buffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    coherent: bool,
}

pub(super) struct VulkanBandwidthContext {
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
}

impl VulkanBandwidthContext {
    pub fn new(record: &AdapterRecord) -> Result<Self, BenchmarkError> {
        // SAFETY: handles are owned by this context and released in reverse order by Drop.
        unsafe { Self::new_inner(record) }
    }

    unsafe fn new_inner(record: &AdapterRecord) -> Result<Self, BenchmarkError> {
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
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(record.vulkan.compute_queue_family)
            .queue_priorities(&priorities)];
        let device = match unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                None,
            )
        } {
            Ok(device) => device,
            Err(problem) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error("vulkan_device_failed", problem));
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
        })
    }

    fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Result<Buffer, BenchmarkError> {
        // SAFETY: allocation and binding are local; partial resources are cleaned on failure.
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
                .map_err(allocation_error)?;
            let requirements = self.device.get_buffer_memory_requirements(handle);
            let memory_type = match find_memory_type(
                &self.memory_properties,
                requirements.memory_type_bits,
                required,
                preferred,
            ) {
                Some(index) => index,
                None => {
                    self.device.destroy_buffer(handle, None);
                    return Err(BenchmarkError::new(
                        "vulkan_memory_type_unavailable",
                        format!("No Vulkan memory type satisfies {required:?}."),
                    ));
                }
            };
            let flags = self.memory_properties.memory_types[memory_type as usize].property_flags;
            let memory = match self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            ) {
                Ok(memory) => memory,
                Err(problem) => {
                    self.device.destroy_buffer(handle, None);
                    return Err(allocation_error(problem));
                }
            };
            if let Err(problem) = self.device.bind_buffer_memory(handle, memory, 0) {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(handle, None);
                return Err(error("vulkan_buffer_bind_failed", problem));
            }
            Ok(Buffer {
                handle,
                memory,
                coherent: flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
            })
        }
    }

    fn destroy_buffer(&self, buffer: Buffer) {
        // SAFETY: the queue is idle at all call sites and this context owns both handles.
        unsafe {
            self.device.destroy_buffer(buffer.handle, None);
            self.device.free_memory(buffer.memory, None);
        }
    }

    fn begin(&self) -> Result<(), BenchmarkError> {
        // SAFETY: one command buffer is reused only after every prior submission becomes idle.
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
                .map_err(|problem| error("vulkan_command_begin_failed", problem))
        }
    }

    fn submit_and_wait(&self) -> Result<(), BenchmarkError> {
        // SAFETY: the recorded command buffer and queue belong to the same live device.
        unsafe {
            self.device
                .end_command_buffer(self.command_buffer)
                .map_err(|problem| error("vulkan_command_end_failed", problem))?;
            let buffers = [self.command_buffer];
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&buffers)],
                    vk::Fence::null(),
                )
                .map_err(|problem| error("vulkan_queue_submit_failed", problem))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|problem| error("device_lost", problem))
        }
    }

    fn timestamp_elapsed_ns(&self) -> Result<f64, BenchmarkError> {
        let mut values = [0_u64; 2];
        // SAFETY: the queue is idle and both query slots were written by the last submission.
        unsafe {
            self.device.get_query_pool_results(
                self.query_pool,
                0,
                &mut values,
                vk::QueryResultFlags::TYPE_64,
            )
        }
        .map_err(|problem| error("vulkan_timestamp_read_failed", problem))?;
        let ticks = wrapped_timestamp_delta(values[0], values[1], self.timestamp_valid_bits);
        let elapsed = ticks as f64 * self.timestamp_period_ns;
        if elapsed <= 0.0 || !elapsed.is_finite() {
            return Err(BenchmarkError::new(
                "device_lost",
                format!(
                    "Vulkan timestamp interval was invalid (start={}, end={}).",
                    values[0], values[1]
                ),
            ));
        }
        Ok(elapsed)
    }

    fn record_timestamps<F>(&self, record: F) -> Result<f64, BenchmarkError>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer),
    {
        self.begin()?;
        // SAFETY: query pool and command buffer are live and in the recording state.
        unsafe {
            self.device
                .cmd_reset_query_pool(self.command_buffer, self.query_pool, 0, 2);
            self.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                0,
            );
        }
        record(&self.device, self.command_buffer);
        // SAFETY: same live recording objects as above.
        unsafe {
            self.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                1,
            );
        }
        self.submit_and_wait()?;
        self.timestamp_elapsed_ns()
    }
}

impl Drop for VulkanBandwidthContext {
    fn drop(&mut self) {
        // SAFETY: this context exclusively owns these objects; idle prevents in-flight use.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct KernelHarness<'a> {
    context: &'a VulkanBandwidthContext,
    source: Buffer,
    destination: Buffer,
    checksums: Buffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set: vk::DescriptorSet,
    workgroups_x: u32,
    workgroups_y: u32,
    params: [u32; 4],
    traffic_bytes: u64,
}

impl<'a> KernelHarness<'a> {
    fn new(
        context: &'a VulkanBandwidthContext,
        size: u64,
        dispatch_bytes: u64,
        operation: KernelOperation,
        flavor: KernelFlavor,
    ) -> Result<Self, BenchmarkError> {
        if size == 0
            || !size.is_multiple_of(16)
            || dispatch_bytes < size
            || !dispatch_bytes.is_multiple_of(16)
        {
            return Err(BenchmarkError::new(
                "invalid_working_set",
                "GPU bandwidth buffers and dispatch traffic must be non-zero and 16-byte aligned.",
            ));
        }
        let element_count = size / 16;
        let access_count = dispatch_bytes / 16;
        if element_count > u32::MAX as u64 || access_count > u32::MAX as u64 {
            return Err(BenchmarkError::new(
                "invalid_working_set",
                "GPU bandwidth working set exceeds the shader index range.",
            ));
        }
        let accesses = operation.accesses_per_invocation(flavor);
        if !access_count.is_multiple_of(accesses) {
            return Err(BenchmarkError::new(
                "invalid_working_set",
                "GPU traffic does not divide evenly across the selected kernel.",
            ));
        }
        let invocations = access_count / accesses;
        let total_workgroups = invocations.div_ceil(WORKGROUP_SIZE);
        let workgroups_x = total_workgroups.min(65_535) as u32;
        let workgroups_y = total_workgroups.div_ceil(workgroups_x as u64) as u32;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let source = context.create_buffer(
            size,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        let destination = context.create_buffer(
            size,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        let checksum_size = if operation == KernelOperation::Read {
            (invocations * 4).max(4)
        } else {
            4
        };
        let checksums = context.create_buffer(
            checksum_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        // SAFETY: all Vulkan objects below are owned by the harness and destroyed by Drop.
        unsafe {
            let bindings = [
                descriptor_binding(0),
                descriptor_binding(1),
                descriptor_binding(2),
            ];
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
                    .buffer(source.handle)
                    .range(size),
                vk::DescriptorBufferInfo::default()
                    .buffer(destination.handle)
                    .range(size),
                vk::DescriptorBufferInfo::default()
                    .buffer(checksums.handle)
                    .range(checksum_size),
            ];
            let writes = [0_u32, 1, 2].map(|binding| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[binding as usize]))
            });
            context.device.update_descriptor_sets(&writes, &[]);
            let words = shader_words(operation, flavor)?;
            let module = context
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(|problem| error("vulkan_shader_module_failed", problem))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline_result = context.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            );
            context.device.destroy_shader_module(module, None);
            let pipeline = pipeline_result
                .map_err(|(_, problem)| error("vulkan_compute_pipeline_failed", problem))?[0];
            let harness = Self {
                context,
                source,
                destination,
                checksums,
                descriptor_pool,
                descriptor_layout,
                pipeline_layout,
                pipeline,
                descriptor_set,
                workgroups_x,
                workgroups_y,
                params: [
                    element_count as u32,
                    0x9e37_79b9,
                    workgroups_x * WORKGROUP_SIZE as u32,
                    access_count as u32,
                ],
                traffic_bytes: dispatch_bytes,
            };
            harness.measure(1)?;
            Ok(harness)
        }
    }

    fn measure(&self, iterations: u32) -> Result<f64, BenchmarkError> {
        let push = words_as_bytes(&self.params);
        self.context
            .record_timestamps(|device, command_buffer| unsafe {
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
                    push,
                );
                for _ in 0..iterations {
                    device.cmd_dispatch(command_buffer, self.workgroups_x, self.workgroups_y, 1);
                }
            })
    }
}

impl Drop for KernelHarness<'_> {
    fn drop(&mut self) {
        // SAFETY: construction warmed and waited the queue; all child objects are owned here.
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
        }
        let source = std::mem::replace(&mut self.source, null_buffer());
        let destination = std::mem::replace(&mut self.destination, null_buffer());
        let checksums = std::mem::replace(&mut self.checksums, null_buffer());
        self.context.destroy_buffer(source);
        self.context.destroy_buffer(destination);
        self.context.destroy_buffer(checksums);
    }
}

pub(super) fn measure_gpu_operation(
    context: &VulkanBandwidthContext,
    measurement: DispatchMeasurement,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, BenchmarkError> {
    if measurement.operation == KernelOperation::Copy && measurement.flavor == KernelFlavor::Stream
    {
        return measure_native_copy(context, measurement, cancellation);
    }
    let harness = KernelHarness::new(
        context,
        measurement.size,
        measurement.dispatch_bytes,
        measurement.operation,
        measurement.flavor,
    )?;
    calibrated_samples(
        harness.traffic_bytes * measurement.operation.reported_byte_multiplier(),
        measurement.sample_count,
        measurement.target_per_sample,
        measurement.precondition,
        cancellation,
        |iterations| harness.measure(iterations),
    )
}

fn measure_native_copy(
    context: &VulkanBandwidthContext,
    measurement: DispatchMeasurement,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, BenchmarkError> {
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let source = context.create_buffer(
        measurement.size,
        usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )?;
    let destination = context.create_buffer(
        measurement.size,
        usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )?;
    let result = calibrated_samples(
        measurement.size * 2,
        measurement.sample_count,
        measurement.target_per_sample,
        measurement.precondition,
        cancellation,
        |iterations| {
            context.record_timestamps(|device, command_buffer| unsafe {
                for _ in 0..iterations {
                    device.cmd_copy_buffer(
                        command_buffer,
                        source.handle,
                        destination.handle,
                        &[vk::BufferCopy::default().size(measurement.size)],
                    );
                }
            })
        },
    );
    context.destroy_buffer(source);
    context.destroy_buffer(destination);
    result
}

fn calibrated_samples<F>(
    bytes_per_iteration: u64,
    sample_count: u32,
    target: Duration,
    precondition: bool,
    cancellation: &CancellationToken,
    mut measure: F,
) -> Result<Vec<f64>, BenchmarkError>
where
    F: FnMut(u32) -> Result<f64, BenchmarkError>,
{
    let calibration_iterations = 64;
    let mut per_iteration_ns = measure(calibration_iterations)? / calibration_iterations as f64;
    if per_iteration_ns <= 0.0 || !per_iteration_ns.is_finite() {
        return Err(BenchmarkError::new(
            "device_lost",
            "Vulkan bandwidth timestamp calibration was invalid.",
        ));
    }
    if precondition {
        let warm_iterations = (GPU_PRECONDITION_MS * 1e6 / per_iteration_ns)
            .ceil()
            .clamp(1.0, 4096.0) as u32;
        per_iteration_ns = measure(warm_iterations)? / warm_iterations as f64;
    }
    let iterations = (target.as_secs_f64() * 1e9 / per_iteration_ns)
        .ceil()
        .clamp(1.0, 4096.0) as u32;
    let mut values = Vec::with_capacity(sample_count as usize);
    for _ in 0..sample_count {
        ensure_not_cancelled(cancellation)?;
        let elapsed_ns = measure(iterations)?;
        let bandwidth = bytes_per_iteration as f64 * iterations as f64 / (elapsed_ns / 1e9);
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(BenchmarkError::new(
                "device_lost",
                "Vulkan bandwidth sample was not finite and positive.",
            ));
        }
        values.push(bandwidth);
    }
    Ok(values)
}

pub(super) fn measure_host_to_device(
    context: &VulkanBandwidthContext,
    size: u64,
    samples: u32,
    target: Duration,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, BenchmarkError> {
    let staging = create_staging_ring(context, size, 0xa5)?;
    let destination = context.create_buffer(
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )?;
    let result = wall_clock_copy_samples(size, samples, target, cancellation, |iterations| {
        context.begin()?;
        // SAFETY: all buffers and the recording command buffer belong to this context.
        unsafe {
            for iteration in 0..iterations as usize {
                context.device.cmd_copy_buffer(
                    context.command_buffer,
                    staging[iteration % staging.len()].handle,
                    destination.handle,
                    &[vk::BufferCopy::default().size(size)],
                );
            }
        }
        context.submit_and_wait()
    });
    for buffer in staging {
        context.destroy_buffer(buffer);
    }
    context.destroy_buffer(destination);
    result
}

pub(super) fn measure_device_to_host(
    context: &VulkanBandwidthContext,
    size: u64,
    samples: u32,
    target: Duration,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, BenchmarkError> {
    let source = context.create_buffer(
        size,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    )?;
    let readbacks = create_staging_ring(context, size, 0)?;
    let result = wall_clock_copy_samples(size, samples, target, cancellation, |iterations| {
        context.begin()?;
        // SAFETY: all buffers and the recording command buffer belong to this context.
        unsafe {
            for iteration in 0..iterations as usize {
                context.device.cmd_copy_buffer(
                    context.command_buffer,
                    source.handle,
                    readbacks[iteration % readbacks.len()].handle,
                    &[vk::BufferCopy::default().size(size)],
                );
            }
        }
        context.submit_and_wait()?;
        for buffer in readbacks
            .iter()
            .take((iterations as usize).min(readbacks.len()))
        {
            touch_host_buffer(context, buffer, size)?;
        }
        Ok(())
    });
    context.destroy_buffer(source);
    for buffer in readbacks {
        context.destroy_buffer(buffer);
    }
    result
}

fn wall_clock_copy_samples<F>(
    size: u64,
    samples: u32,
    target: Duration,
    cancellation: &CancellationToken,
    mut copy: F,
) -> Result<Vec<f64>, BenchmarkError>
where
    F: FnMut(u32) -> Result<(), BenchmarkError>,
{
    let trial = Instant::now();
    copy(1)?;
    let trial_seconds = trial.elapsed().as_secs_f64().max(1e-6);
    let iterations = (target.as_secs_f64() / trial_seconds)
        .ceil()
        .clamp(1.0, 4096.0) as u32;
    let mut values = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        ensure_not_cancelled(cancellation)?;
        let started = Instant::now();
        copy(iterations)?;
        values.push(size as f64 * iterations as f64 / started.elapsed().as_secs_f64());
    }
    Ok(values)
}

fn create_staging_ring(
    context: &VulkanBandwidthContext,
    size: u64,
    pattern: u8,
) -> Result<Vec<Buffer>, BenchmarkError> {
    (0..HOST_STAGING_BUFFER_COUNT)
        .map(|_| {
            let buffer = context.create_buffer(
                size,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            initialize_host_buffer(context, &buffer, size, pattern)?;
            Ok(buffer)
        })
        .collect()
}

fn initialize_host_buffer(
    context: &VulkanBandwidthContext,
    buffer: &Buffer,
    size: u64,
    pattern: u8,
) -> Result<(), BenchmarkError> {
    // SAFETY: memory is HOST_VISIBLE, the mapped range is within the allocation, and is unmapped.
    unsafe {
        let mapped = context
            .device
            .map_memory(buffer.memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|problem| error("mapping_failed", problem))?;
        ptr::write_bytes(mapped.cast::<u8>(), pattern, size as usize);
        if !buffer.coherent {
            context
                .device
                .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                    .memory(buffer.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)])
                .map_err(|problem| error("mapping_failed", problem))?;
        }
        context.device.unmap_memory(buffer.memory);
    }
    Ok(())
}

fn touch_host_buffer(
    context: &VulkanBandwidthContext,
    buffer: &Buffer,
    size: u64,
) -> Result<(), BenchmarkError> {
    // SAFETY: memory is HOST_VISIBLE and GPU writes have completed before mapping it.
    unsafe {
        let mapped = context
            .device
            .map_memory(buffer.memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|problem| error("mapping_failed", problem))?;
        if !buffer.coherent {
            context
                .device
                .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                    .memory(buffer.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)])
                .map_err(|problem| error("mapping_failed", problem))?;
        }
        std::hint::black_box(ptr::read_volatile(mapped.cast::<u8>()));
        context.device.unmap_memory(buffer.memory);
    }
    Ok(())
}

fn descriptor_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn shader_words(
    operation: KernelOperation,
    flavor: KernelFlavor,
) -> Result<Vec<u32>, BenchmarkError> {
    let bytes = match (operation, flavor) {
        (KernelOperation::Read, KernelFlavor::Cache) => READ_CACHE_SPV,
        (KernelOperation::Write, KernelFlavor::Cache) => WRITE_CACHE_SPV,
        (KernelOperation::Copy, KernelFlavor::Cache) => COPY_CACHE_SPV,
        (KernelOperation::Read, KernelFlavor::Stream) => READ_STREAM_SPV,
        (KernelOperation::Write, KernelFlavor::Stream) => WRITE_STREAM_SPV,
        (KernelOperation::Copy, KernelFlavor::Stream) => COPY_STREAM_SPV,
    };
    if !bytes.len().is_multiple_of(4) {
        return Err(BenchmarkError::new(
            "invalid_embedded_shader",
            "Embedded bandwidth SPIR-V is not word-aligned.",
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let candidates = (0..properties.memory_type_count).filter(|index| {
        bits & (1 << index) != 0
            && properties.memory_types[*index as usize]
                .property_flags
                .contains(required)
    });
    candidates
        .clone()
        .find(|index| {
            properties.memory_types[*index as usize]
                .property_flags
                .contains(preferred)
        })
        .or_else(|| candidates.into_iter().next())
}

unsafe fn find_physical_device(
    instance: &ash::Instance,
    uuid: &str,
) -> Result<vk::PhysicalDevice, BenchmarkError> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|problem| error("vulkan_device_enumeration_failed", problem))?;
    for physical_device in devices {
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

fn words_as_bytes(words: &[u32; 4]) -> &[u8] {
    // SAFETY: u32 has no padding and the returned slice borrows the source array.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), 16) }
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn null_buffer() -> Buffer {
    Buffer {
        handle: vk::Buffer::null(),
        memory: vk::DeviceMemory::null(),
        coherent: false,
    }
}

fn allocation_error(problem: vk::Result) -> BenchmarkError {
    BenchmarkError::new(
        "insufficient_gpu_memory",
        format!("Vulkan buffer allocation failed: {problem}"),
    )
}

fn error(code: &str, problem: impl std::fmt::Display) -> BenchmarkError {
    BenchmarkError::new(code, problem.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_bandwidth_shaders_are_spirv() {
        for shader in [
            READ_CACHE_SPV,
            WRITE_CACHE_SPV,
            COPY_CACHE_SPV,
            READ_STREAM_SPV,
            WRITE_STREAM_SPV,
            COPY_STREAM_SPV,
        ] {
            assert_eq!(&shader[..4], &[0x03, 0x02, 0x23, 0x07]);
        }
    }

    #[test]
    fn timestamp_delta_wraps_to_valid_width() {
        assert_eq!(wrapped_timestamp_delta(250, 5, 8), 11);
        assert_eq!(wrapped_timestamp_delta(10, 25, 64), 15);
    }
}
