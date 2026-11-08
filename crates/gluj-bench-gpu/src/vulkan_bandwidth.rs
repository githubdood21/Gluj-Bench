use crate::{
    AdapterRecord, DispatchMeasurement, HOST_STAGING_BUFFER_COUNT, KernelFlavor, KernelOperation,
    WORKGROUP_SIZE, ensure_not_cancelled,
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

struct ConstructionGuard<'a> {
    actions: Vec<Box<dyn FnOnce() + 'a>>,
}
impl<'a> ConstructionGuard<'a> {
    fn new() -> Self {
        Self {
            actions: Vec::new(),
        }
    }
    fn buffer(&mut self, context: &'a VulkanBandwidthContext, buffer: &Buffer) {
        let handle = buffer.handle;
        let memory = buffer.memory;
        self.actions.push(Box::new(move || {
            // SAFETY: this guard owns partial construction until disarmed.
            unsafe {
                context.device.destroy_buffer(handle, None);
                context.device.free_memory(memory, None);
            }
        }));
    }
    fn disarm(&mut self) {
        self.actions.clear();
    }
}
impl Drop for ConstructionGuard<'_> {
    fn drop(&mut self) {
        while let Some(action) = self.actions.pop() {
            action();
        }
    }
}

struct BufferGuard<'a> {
    context: &'a VulkanBandwidthContext,
    buffer: Option<Buffer>,
}
impl std::ops::Deref for BufferGuard<'_> {
    type Target = Buffer;
    fn deref(&self) -> &Buffer {
        self.buffer.as_ref().unwrap()
    }
}
impl Drop for BufferGuard<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.context.destroy_buffer(buffer);
        }
    }
}
struct StagingRing<'a> {
    context: &'a VulkanBandwidthContext,
    buffers: Vec<Buffer>,
}
impl std::ops::Deref for StagingRing<'_> {
    type Target = [Buffer];
    fn deref(&self) -> &[Buffer] {
        &self.buffers
    }
}
impl Drop for StagingRing<'_> {
    fn drop(&mut self) {
        for buffer in self.buffers.drain(..) {
            self.context.destroy_buffer(buffer);
        }
    }
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
    queue_family: u32,
    discrete: bool,
    timestamp_period_ns: f64,
    timestamp_valid_bits: u32,
}

impl VulkanBandwidthContext {
    fn prepare_pattern(
        &self,
        buffer: vk::Buffer,
        size: u64,
        pattern: u32,
    ) -> Result<(), BenchmarkError> {
        self.record_timestamps(|device, command| unsafe {
            device.cmd_fill_buffer(command, buffer, 0, size, pattern);
            let barriers = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ)];
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &barriers,
                &[],
                &[],
            );
        })?;
        Ok(())
    }
    fn validate_pattern(
        &self,
        buffer: vk::Buffer,
        size: u64,
        expected: u32,
    ) -> Result<(), BenchmarkError> {
        for (_, word) in crate::validation::read_words(
            &self.device,
            &self.memory_properties,
            self.queue,
            self.queue_family,
            buffer,
            size,
        )? {
            if word != expected {
                return Err(crate::validation::problem(
                    "Transfer output differs from the known source pattern.",
                ));
            }
        }
        Ok(())
    }
    pub fn new(record: &AdapterRecord) -> Result<Self, BenchmarkError> {
        let _phase_timer = gluj_bench_core::phase_timer("preparation");
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
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let mut owner = Self {
            _entry: entry,
            instance,
            device,
            queue,
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            query_pool: vk::QueryPool::null(),
            memory_properties,
            queue_family: record.vulkan.compute_queue_family,
            discrete: record.vulkan.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
            timestamp_period_ns: record.vulkan.timestamp_period_ns as f64,
            timestamp_valid_bits: record.vulkan.timestamp_valid_bits,
        };
        owner.command_pool = unsafe {
            owner.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(record.vulkan.compute_queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|p| error("vulkan_command_pool_failed", p))?;
        owner.command_buffer = unsafe {
            owner.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(owner.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|p| error("vulkan_command_buffer_failed", p))?[0];
        owner.query_pool = unsafe {
            owner.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .map_err(|p| error("vulkan_query_pool_failed", p))?;
        Ok(owner)
    }

    fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Result<Buffer, BenchmarkError> {
        let _phase_timer = gluj_bench_core::phase_timer("preparation");
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
            let memory_type = match if required.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                find_staging_memory_type(
                    &self.memory_properties,
                    requirements.memory_type_bits,
                    preferred,
                    self.discrete,
                )
            } else {
                find_memory_type(
                    &self.memory_properties,
                    requirements.memory_type_bits,
                    required,
                    preferred,
                )
            } {
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
        let activity_started = std::time::Instant::now();
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
                .map_err(|problem| error("device_lost", problem))?;
        }
        gluj_bench_core::pace_gpu(activity_started.elapsed())
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
    access: std::cell::Cell<(u32, u64)>,
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
        let _phase_timer = gluj_bench_core::phase_timer("preparation");
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
        let mut construction = ConstructionGuard::new();
        let source = context.create_buffer(
            size,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        construction.buffer(context, &source);
        let destination = context.create_buffer(
            size,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        construction.buffer(context, &destination);
        let checksum_size = if operation == KernelOperation::Read {
            (invocations * 4).max(4)
        } else {
            4
        };
        let checksums = context.create_buffer(
            checksum_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        construction.buffer(context, &checksums);
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
            construction.actions.push(Box::new(move || {
                context
                    .device
                    .destroy_descriptor_set_layout(descriptor_layout, None)
            }));
            let set_layouts = [descriptor_layout];
            let push_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(32)];
            let pipeline_layout = context
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_ranges),
                    None,
                )
                .map_err(|problem| error("vulkan_pipeline_layout_failed", problem))?;
            construction.actions.push(Box::new(move || {
                context
                    .device
                    .destroy_pipeline_layout(pipeline_layout, None)
            }));
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
            construction.actions.push(Box::new(move || {
                context
                    .device
                    .destroy_descriptor_pool(descriptor_pool, None)
            }));
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
            construction.actions.push(Box::new(move || {
                context.device.destroy_pipeline(pipeline, None)
            }));
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
                access: std::cell::Cell::new((0, 0)),
                traffic_bytes: dispatch_bytes,
            };
            construction.disarm();
            context.record_timestamps(|device, command| {
                device.cmd_fill_buffer(command, harness.source.handle, 0, size, 0xa5a5_a5a5);
                device.cmd_fill_buffer(command, harness.destination.handle, 0, size, 0xcccc_cccc);
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
            })?;
            harness.measure(1)?;
            Ok(harness)
        }
    }

    fn measure(&self, iterations: u32) -> Result<f64, BenchmarkError> {
        let (mode, seed) = self.access.get();
        let (mul, offset) = crate::memory_conditions::permutation(self.params[0] / 64, seed);
        self.context
            .record_timestamps(|device, command_buffer| unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )],
                    &[],
                    &[],
                );
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
                for iteration in 0..iterations {
                    let offset = if mode == 2 {
                        ((u64::from(offset) + u64::from(iteration))
                            % u64::from((self.params[0] / 64).max(1)))
                            as u32
                    } else {
                        offset
                    };
                    let params = [
                        self.params[0],
                        self.params[1],
                        self.params[2],
                        self.params[3],
                        mode,
                        mul,
                        offset,
                        (seed as u32) | 1,
                    ];
                    device.cmd_push_constants(
                        command_buffer,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        words_as_bytes(&params),
                    );
                    if iteration > 0 {
                        device.cmd_pipeline_barrier(
                            command_buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[vk::MemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)],
                            &[],
                            &[],
                        );
                    }
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
    measure_gpu_operation_access(
        context,
        measurement,
        cancellation,
        measurement.flavor == KernelFlavor::Stream,
        0,
        0,
    )
}
#[allow(clippy::too_many_arguments)]
pub(super) fn measure_gpu_operation_access(
    context: &VulkanBandwidthContext,
    measurement: DispatchMeasurement,
    cancellation: &CancellationToken,
    native_copy: bool,
    mode: u32,
    seed: u64,
) -> Result<Vec<f64>, BenchmarkError> {
    let _scope = gluj_bench_core::MeasurementScope::condition(&format!(
        "{:?}:{:?}:bytes-{}",
        measurement.flavor, measurement.operation, measurement.size
    ));
    if measurement.operation == KernelOperation::Copy && native_copy {
        return measure_native_copy(context, measurement, cancellation);
    }
    let harness = KernelHarness::new(
        context,
        measurement.size,
        measurement.dispatch_bytes,
        measurement.operation,
        measurement.flavor,
    )?;
    harness.access.set((mode, seed));
    let values = calibrated_samples(
        harness.traffic_bytes * measurement.operation.reported_byte_multiplier(),
        measurement.sample_count,
        measurement.target_per_sample,
        measurement.precondition,
        cancellation,
        |iterations| harness.measure(iterations),
    )?;
    let (output, size) = if measurement.operation == KernelOperation::Read {
        (
            harness.checksums.handle,
            measurement.dispatch_bytes
                / 16
                / measurement
                    .operation
                    .accesses_per_invocation(measurement.flavor)
                * 4,
        )
    } else {
        (harness.destination.handle, measurement.size)
    };
    for (index, word) in crate::validation::read_words(
        &context.device,
        &context.memory_properties,
        context.queue,
        context.queue_family,
        output,
        size,
    )? {
        let expected = match measurement.operation {
            KernelOperation::Read => 0x9e37_79b9,
            KernelOperation::Write => {
                (0x9e37_79b9 ^ (index / 4) as u32).wrapping_add((index % 4) as u32)
            }
            KernelOperation::Copy => 0xa5a5_a5a5,
        };
        if word != expected {
            return Err(crate::validation::problem(
                "GPU bandwidth output differs from its source/write/checksum reference.",
            ));
        }
    }
    Ok(values)
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
    context.prepare_pattern(source.handle, measurement.size, 0xa5a5_a5a5)?;
    let result = calibrated_samples(
        measurement.size * 2,
        measurement.sample_count,
        measurement.target_per_sample,
        measurement.precondition,
        cancellation,
        |iterations| {
            context.record_timestamps(|device, command_buffer| unsafe {
                for _ in 0..iterations {
                    transfer_dependency(device, command_buffer);
                    device.cmd_copy_buffer(
                        command_buffer,
                        source.handle,
                        destination.handle,
                        &[vk::BufferCopy::default().size(measurement.size)],
                    );
                }
            })
        },
    )
    .and_then(|values| {
        context.validate_pattern(destination.handle, measurement.size, 0xa5a5_a5a5)?;
        Ok(values)
    });
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
    let burst = gluj_bench_core::gpu_burst_duration(target);
    let calibration_iterations = 64;
    let mut per_iteration_ns = measure(calibration_iterations)? / calibration_iterations as f64;
    if per_iteration_ns <= 0.0 || !per_iteration_ns.is_finite() {
        return Err(BenchmarkError::new(
            "device_lost",
            "Vulkan bandwidth timestamp calibration was invalid.",
        ));
    }
    if precondition {
        let target = gluj_bench_core::current_measurement_budget().warmup_ns;
        let warm_iterations = (target as f64 / per_iteration_ns).ceil().clamp(1., 4096.) as u32;
        let _warmup = gluj_bench_core::phase_timer("warmup");
        let observed =
            gluj_bench_core::measure_batches(target, cancellation, || measure(warm_iterations))?;
        per_iteration_ns =
            observed.active_ns / (u64::from(warm_iterations) * u64::from(observed.batches)) as f64;
    }
    let iterations = (burst.as_secs_f64() * 1e9 / per_iteration_ns)
        .ceil()
        .clamp(1.0, 4096.0) as u32;
    let mut values = Vec::with_capacity(sample_count as usize);
    for _ in 0..sample_count {
        ensure_not_cancelled(cancellation)?;
        let observation =
            gluj_bench_core::measure_batches(target.as_nanos() as u64, cancellation, || {
                measure(iterations)
            })?;
        let elapsed_ns = observation.active_ns;
        let completed_iterations = iterations.saturating_mul(observation.batches);
        let bandwidth =
            bytes_per_iteration as f64 * completed_iterations as f64 / (elapsed_ns / 1e9);
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(BenchmarkError::new(
                "device_lost",
                "Vulkan bandwidth sample was not finite and positive.",
            ));
        }
        values.push(bandwidth);
        gluj_bench_core::record_observation(
            &format!("traffic-{bytes_per_iteration}"),
            "local_bandwidth",
            bandwidth,
            "bytes/s",
            bytes_per_iteration.saturating_mul(u64::from(completed_iterations)),
            "accounted_bytes",
            elapsed_ns as u64,
            observation.elapsed_ns,
        );
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
    measure_host_to_device_config(
        context,
        size,
        samples,
        target,
        cancellation,
        TransferOptions::default(),
    )
}
pub(super) fn measure_host_to_device_config(
    context: &VulkanBandwidthContext,
    size: u64,
    samples: u32,
    target: Duration,
    cancellation: &CancellationToken,
    options: TransferOptions,
) -> Result<Vec<f64>, BenchmarkError> {
    let _scope = gluj_bench_core::MeasurementScope::condition("host_to_device");
    let staging = create_staging_ring(context, size, 0xa5, options.ring, cancellation)?;
    let destination = BufferGuard {
        context,
        buffer: Some(context.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?),
    };
    let result = wall_clock_copy_samples(size, samples, target, cancellation, |iterations| {
        let batch = if options.serial {
            1
        } else {
            options.batch.max(1)
        };
        for first in (0..iterations).step_by(batch as usize) {
            ensure_not_cancelled(cancellation)?;
            context.begin()?;
            // SAFETY: each group completes before command-pool or staging-slot reuse.
            unsafe {
                for iteration in first..iterations.min(first + batch) {
                    transfer_dependency(&context.device, context.command_buffer);
                    context.device.cmd_copy_buffer(
                        context.command_buffer,
                        staging[iteration as usize % staging.len()].handle,
                        destination.handle,
                        &[vk::BufferCopy::default().size(size)],
                    );
                }
            }
            context.submit_and_wait()?;
        }
        Ok(())
    })
    .and_then(|values| {
        context.validate_pattern(destination.handle, size, 0xa5a5_a5a5)?;
        Ok(values)
    });
    drop(staging);
    drop(destination);
    result
}

pub(super) fn measure_device_to_host(
    context: &VulkanBandwidthContext,
    size: u64,
    samples: u32,
    target: Duration,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, BenchmarkError> {
    measure_device_to_host_config(
        context,
        size,
        samples,
        target,
        cancellation,
        TransferOptions::default(),
    )
}
pub(super) fn measure_device_to_host_config(
    context: &VulkanBandwidthContext,
    size: u64,
    samples: u32,
    target: Duration,
    cancellation: &CancellationToken,
    options: TransferOptions,
) -> Result<Vec<f64>, BenchmarkError> {
    let _scope = gluj_bench_core::MeasurementScope::condition("device_to_host");
    let source = BufferGuard {
        context,
        buffer: Some(context.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?),
    };
    let readbacks = create_staging_ring(context, size, 0, options.ring, cancellation)?;
    context.prepare_pattern(source.handle, size, 0xa5a5_a5a5)?;
    let mut used_slots = 0usize;
    let result = wall_clock_copy_samples(size, samples, target, cancellation, |iterations| {
        used_slots = used_slots.max((iterations as usize).min(readbacks.len()));
        let batch = if options.serial {
            1
        } else {
            options.batch.max(1)
        };
        for first in (0..iterations).step_by(batch as usize) {
            ensure_not_cancelled(cancellation)?;
            context.begin()?;
            // SAFETY: valid ranges, ordered writes, host visibility before mapped reads.
            unsafe {
                for iteration in first..iterations.min(first + batch) {
                    transfer_dependency(&context.device, context.command_buffer);
                    context.device.cmd_copy_buffer(
                        context.command_buffer,
                        source.handle,
                        readbacks[iteration as usize % readbacks.len()].handle,
                        &[vk::BufferCopy::default().size(size)],
                    );
                }
                context.device.cmd_pipeline_barrier(
                    context.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)],
                    &[],
                    &[],
                );
            }
            context.submit_and_wait()?;
        }
        for buffer in readbacks
            .iter()
            .take((iterations as usize).min(readbacks.len()))
        {
            touch_host_buffer(context, buffer, size)?;
        }
        Ok(())
    });
    let result = result.and_then(|values| {
        for buffer in readbacks.iter().take(used_slots) {
            context.validate_pattern(buffer.handle, size, 0xa5a5_a5a5)?;
        }
        Ok(values)
    });
    drop(source);
    drop(readbacks);
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
    {
        let _warmup = gluj_bench_core::phase_timer("warmup");
        gluj_bench_core::measure_batches(
            gluj_bench_core::current_measurement_budget().warmup_ns,
            cancellation,
            || {
                let started = Instant::now();
                copy(iterations)?;
                Ok(started.elapsed().as_secs_f64() * 1e9)
            },
        )?;
    }
    let mut values = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        ensure_not_cancelled(cancellation)?;
        let observation =
            gluj_bench_core::measure_batches(target.as_nanos() as u64, cancellation, || {
                let started = Instant::now();
                copy(iterations)?;
                Ok(started.elapsed().as_secs_f64() * 1e9)
            })?;
        let completed = u64::from(iterations).saturating_mul(u64::from(observation.batches));
        values.push(size as f64 * completed as f64 * 1e9 / observation.active_ns);
        gluj_bench_core::record_observation(
            &format!("transfer-bytes-{size}"),
            "mean_request_cost",
            observation.active_ns / completed as f64,
            "ns",
            completed,
            "copies",
            observation.active_ns as u64,
            observation.elapsed_ns,
        );
        gluj_bench_core::record_observation(
            &format!("transfer-bytes-{size}"),
            "host_transfer",
            *values.last().unwrap(),
            "bytes/s",
            size.saturating_mul(completed),
            "payload_bytes",
            observation.active_ns as u64,
            observation.elapsed_ns,
        );
    }
    Ok(values)
}

fn create_staging_ring<'a>(
    context: &'a VulkanBandwidthContext,
    size: u64,
    pattern: u8,
    count: usize,
    cancellation: &CancellationToken,
) -> Result<StagingRing<'a>, BenchmarkError> {
    let mut ring = StagingRing {
        context,
        buffers: Vec::new(),
    };
    for _ in 0..count {
        let buffer = context.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        ring.buffers.push(buffer);
        initialize_host_buffer(
            context,
            ring.buffers.last().unwrap(),
            size,
            pattern,
            cancellation,
        )?;
    }
    Ok(ring)
}

fn initialize_host_buffer(
    context: &VulkanBandwidthContext,
    buffer: &Buffer,
    size: u64,
    pattern: u8,
    cancellation: &CancellationToken,
) -> Result<(), BenchmarkError> {
    // SAFETY: memory is HOST_VISIBLE, the mapped range is within the allocation, and is unmapped.
    unsafe {
        let mapped = context
            .device
            .map_memory(buffer.memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|problem| error("mapping_failed", problem))?;
        let initialized = (|| {
            for offset in (0..size as usize).step_by(1024 * 1024) {
                ensure_not_cancelled(cancellation)?;
                ptr::write_bytes(
                    mapped.cast::<u8>().add(offset),
                    pattern,
                    (size as usize - offset).min(1024 * 1024),
                );
            }
            if !buffer.coherent {
                context
                    .device
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(buffer.memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
                    .map_err(|problem| error("mapping_failed", problem))?;
            }
            Ok(())
        })();
        context.device.unmap_memory(buffer.memory);
        initialized?;
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
        let visible = if !buffer.coherent {
            context
                .device
                .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                    .memory(buffer.memory)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)])
                .map_err(|problem| error("mapping_failed", problem))
        } else {
            Ok(())
        };
        if visible.is_ok() {
            std::hint::black_box(ptr::read_volatile(mapped.cast::<u8>()));
        }
        context.device.unmap_memory(buffer.memory);
        visible?;
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

fn words_as_bytes(words: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and the returned slice borrows the source array.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), std::mem::size_of_val(words)) }
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

#[derive(Clone, Copy)]
pub(super) struct TransferOptions {
    pub ring: usize,
    pub batch: u32,
    pub serial: bool,
}
impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            ring: HOST_STAGING_BUFFER_COUNT,
            batch: 4096,
            serial: false,
        }
    }
}
unsafe fn transfer_dependency(device: &ash::Device, command: vk::CommandBuffer) {
    // SAFETY: caller owns a recording command buffer; orders WAW copy destinations.
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
            &[],
            &[],
        );
    }
}

fn find_staging_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    preferred: vk::MemoryPropertyFlags,
    discrete: bool,
) -> Option<u32> {
    let candidates = (0..properties.memory_type_count)
        .filter(|index| {
            let ty = properties.memory_types[*index as usize];
            bits & (1 << index) != 0
                && ty
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                && (!discrete
                    || !properties.memory_heaps[ty.heap_index as usize]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        })
        .collect::<Vec<_>>();
    candidates
        .iter()
        .copied()
        .find(|i| {
            properties.memory_types[*i as usize]
                .property_flags
                .contains(preferred)
        })
        .or_else(|| candidates.first().copied())
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
    #[test]
    fn discrete_staging_excludes_host_visible_bar_vram_and_shared_devices_keep_their_heap() {
        let mut p = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 3,
            memory_heap_count: 2,
            ..Default::default()
        };
        p.memory_heaps[0].flags = vk::MemoryHeapFlags::DEVICE_LOCAL;
        p.memory_types[0] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::DEVICE_LOCAL,
            heap_index: 0,
        };
        p.memory_types[1] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            heap_index: 1,
        };
        p.memory_types[2] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 1,
        };
        assert_eq!(
            find_staging_memory_type(&p, 7, vk::MemoryPropertyFlags::HOST_COHERENT, true),
            Some(2)
        );
        assert_eq!(
            find_staging_memory_type(&p, 3, vk::MemoryPropertyFlags::HOST_COHERENT, true),
            Some(1)
        );
        assert_eq!(
            find_staging_memory_type(&p, 1, vk::MemoryPropertyFlags::HOST_COHERENT, true),
            None
        );
        assert_eq!(
            find_staging_memory_type(&p, 1, vk::MemoryPropertyFlags::HOST_COHERENT, false),
            Some(0)
        );
    }
}
