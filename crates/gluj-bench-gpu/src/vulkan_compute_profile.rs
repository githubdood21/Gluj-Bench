use crate::{AdapterRecord, WORKGROUP_SIZE};
use ash::{Entry, vk};
use gluj_bench_core::BenchmarkError;

const PROFILE_SPV: &[u8] = include_bytes!("../shaders/spv/compute_profile_fp32.spv");
const BYTES_PER_ELEMENT: u64 = 48;
const OPERATIONS_PER_LOOP_PER_ELEMENT: u64 = 64;

struct Region {
    elements: u32,
    first_set: usize,
    first_buffer: usize,
}

struct Buffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
}

pub(super) struct VulkanComputeProfileHarness {
    _entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
    elements_per_chunk: u32,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    query_pool: vk::QueryPool,
    buffers: Vec<Buffer>,
    regions: Vec<Region>,
    element_alignment: u64,
    offload_percent: u32,
    timestamp_period_ns: f64,
    timestamp_valid_bits: u32,
    pub maximum_working_set_bytes: u64,
}

impl VulkanComputeProfileHarness {
    pub fn new(record: &AdapterRecord, requested_working_set: u64) -> Result<Self, BenchmarkError> {
        // SAFETY: every Vulkan handle is owned by the returned harness and destroyed by Drop.
        unsafe { Self::new_inner(record, requested_working_set, 0) }
    }

    pub fn new_with_offload(
        record: &AdapterRecord,
        requested_working_set: u64,
        offload_percent: u32,
    ) -> Result<Self, BenchmarkError> {
        if ![50, 75, 100].contains(&offload_percent) {
            return Err(BenchmarkError::new(
                "invalid_config",
                "RAM offload must be 50, 75, or 100 percent.",
            ));
        }
        // SAFETY: the harness owns all resources, including partially constructed resources.
        unsafe { Self::new_inner(record, requested_working_set, offload_percent) }
    }

    unsafe fn new_inner(
        record: &AdapterRecord,
        requested_working_set: u64,
        offload_percent: u32,
    ) -> Result<Self, BenchmarkError> {
        let element_count =
            requested_working_set / BYTES_PER_ELEMENT / WORKGROUP_SIZE * WORKGROUP_SIZE;
        if element_count < WORKGROUP_SIZE || element_count > u32::MAX as u64 {
            return Err(BenchmarkError::new(
                "invalid_working_set",
                "The compute-profile working set is outside the supported shader index range.",
            ));
        }
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
        let limits = unsafe { instance.get_physical_device_properties(physical_device) }.limits;
        let elements_per_chunk = chunk_element_limit(
            limits.max_compute_work_group_count[0]
                .min((limits.max_storage_buffer_range as u64 / 16 / WORKGROUP_SIZE) as u32),
            limits.min_storage_buffer_offset_alignment,
        );
        if elements_per_chunk == 0 {
            unsafe { instance.destroy_instance(None) };
            return Err(BenchmarkError::new(
                "invalid_dispatch_limit",
                "The device cannot dispatch an aligned compute-profile chunk.",
            ));
        }
        let element_alignment = WORKGROUP_SIZE.max(limits.min_storage_buffer_offset_alignment / 16);
        // All offload profiles use identical tier alignment; mixed shares stay exact.
        let total_alignment = element_alignment * if offload_percent > 0 { 4 } else { 1 };
        let element_count = element_count / total_alignment * total_alignment;
        if element_count == 0 {
            unsafe { instance.destroy_instance(None) };
            return Err(BenchmarkError::new(
                "invalid_working_set",
                "Working set is too small for aligned RAM/VRAM regions.",
            ));
        }
        let maximum_working_set_bytes = element_count * BYTES_PER_ELEMENT;
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
        // Construct the owner before allocating children. Drop also handles setup failures.
        let mut harness = Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_sets: Vec::new(),
            elements_per_chunk,
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            query_pool: vk::QueryPool::null(),
            buffers: Vec::new(),
            regions: Vec::new(),
            element_alignment: total_alignment,
            offload_percent,
            timestamp_period_ns: record.vulkan.timestamp_period_ns as f64,
            timestamp_valid_bits: record.vulkan.timestamp_valid_bits,
            maximum_working_set_bytes,
        };
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let counts = region_elements(element_count as u32, offload_percent);
        let mut chunks = Vec::new();
        for (region_index, count) in counts.into_iter().enumerate() {
            if count == 0 {
                continue;
            }
            let region = Region {
                elements: count,
                first_set: chunks.len(),
                first_buffer: harness.buffers.len(),
            };
            for _ in 0..3 {
                harness.buffers.push(unsafe {
                    create_buffer(
                        &harness.instance,
                        &harness.device,
                        physical_device,
                        count as u64 * 16,
                        usage,
                        region_index == 0 && offload_percent > 0,
                    )
                }?);
            }
            chunks.extend(
                dispatch_chunks(count, elements_per_chunk)
                    .into_iter()
                    .map(|(offset, count)| (region.first_buffer, offset, count)),
            );
            harness.regions.push(region);
        }
        let bindings = [0_u32, 1, 2].map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        harness.descriptor_layout = unsafe {
            harness.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|p| error("vulkan_descriptor_layout_failed", p))?;
        let set_layouts = [harness.descriptor_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        harness.pipeline_layout = unsafe {
            harness.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )
        }
        .map_err(|p| error("vulkan_pipeline_layout_failed", p))?;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3 * chunks.len() as u32)];
        harness.descriptor_pool = unsafe {
            harness.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(chunks.len() as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|p| error("vulkan_descriptor_pool_failed", p))?;
        let chunk_layouts = vec![harness.descriptor_layout; chunks.len()];
        harness.descriptor_sets = unsafe {
            harness.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(harness.descriptor_pool)
                    .set_layouts(&chunk_layouts),
            )
        }
        .map_err(|p| error("vulkan_descriptor_set_failed", p))?;
        for (&set, &(first, offset, count)) in harness.descriptor_sets.iter().zip(&chunks) {
            let infos = [0, 1, 2].map(|index| {
                vk::DescriptorBufferInfo::default()
                    .buffer(harness.buffers[first + index].handle)
                    .offset(offset as u64 * 16)
                    .range(count as u64 * 16)
            });
            let writes = [0_u32, 1, 2].map(|binding| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[binding as usize]))
            });
            unsafe { harness.device.update_descriptor_sets(&writes, &[]) };
        }
        let words = shader_words()?;
        let shader = unsafe {
            harness
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|p| error("vulkan_shader_module_failed", p))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main");
        let pipeline = unsafe {
            harness.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(harness.pipeline_layout)],
                None,
            )
        };
        unsafe { harness.device.destroy_shader_module(shader, None) };
        harness.pipeline = match pipeline {
            Ok(pipelines) => pipelines[0],
            Err((pipelines, problem)) => {
                for pipeline in pipelines {
                    unsafe { harness.device.destroy_pipeline(pipeline, None) };
                }
                return Err(error("vulkan_compute_pipeline_failed", problem));
            }
        };
        harness.query_pool = unsafe {
            harness.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .map_err(|p| error("vulkan_query_pool_failed", p))?;
        harness.command_pool = unsafe {
            harness.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(record.vulkan.compute_queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|p| error("vulkan_command_pool_failed", p))?;
        harness.command_buffer = unsafe {
            harness.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(harness.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|p| error("vulkan_command_buffer_failed", p))?[0];
        harness.initialize()?;
        Ok(harness)
    }

    fn initialize(&self) -> Result<(), BenchmarkError> {
        // SAFETY: buffers have TRANSFER_DST usage and no submission is in flight.
        unsafe {
            self.begin()?;
            for region in &self.regions {
                for index in 0..3 {
                    self.device.cmd_fill_buffer(
                        self.command_buffer,
                        self.buffers[region.first_buffer + index].handle,
                        0,
                        region.elements as u64 * 16,
                        if index == 2 { 0 } else { 0x3f00_0000 },
                    );
                }
            }
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
            self.submit()?;
        }
        Ok(())
    }

    pub fn actual_working_set(&self, requested: u64) -> u64 {
        let elements = (requested.min(self.maximum_working_set_bytes)
            / BYTES_PER_ELEMENT
            / self.element_alignment)
            * self.element_alignment;
        elements.max(self.element_alignment) * BYTES_PER_ELEMENT
    }

    pub fn operations_per_dispatch(&self, working_set: u64, loop_count: u32) -> u64 {
        (working_set / BYTES_PER_ELEMENT) * loop_count as u64 * OPERATIONS_PER_LOOP_PER_ELEMENT
    }

    pub fn traffic_bytes_per_dispatch(&self, working_set: u64) -> u64 {
        working_set
    }

    /// Check the first and last output of both memory regions outside the timed samples.
    pub fn validate_output(&self, working_set: u64, loop_count: u32) -> Result<(), BenchmarkError> {
        let counts = region_elements(
            (working_set / BYTES_PER_ELEMENT) as u32,
            self.offload_percent,
        );
        let counts = counts
            .into_iter()
            .filter(|count| *count > 0)
            .collect::<Vec<_>>();
        // SAFETY: the selected physical device belongs to our instance; the readback allocation
        // is host-visible, and every submission completes before mapping or destroying it.
        unsafe {
            let readback = create_buffer(
                &self.instance,
                &self.device,
                self.physical_device,
                64,
                vk::BufferUsageFlags::TRANSFER_DST,
                true,
            )?;
            let result = (|| {
                self.begin()?;
                self.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                    &[],
                    &[],
                );
                for (index, (region, count)) in self.regions.iter().zip(&counts).enumerate() {
                    let copies = [
                        vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: index as u64 * 32,
                            size: 16,
                        },
                        vk::BufferCopy {
                            src_offset: (*count as u64 - 1) * 16,
                            dst_offset: index as u64 * 32 + 16,
                            size: 16,
                        },
                    ];
                    self.device.cmd_copy_buffer(
                        self.command_buffer,
                        self.buffers[region.first_buffer + 2].handle,
                        readback.handle,
                        &copies,
                    );
                }
                self.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)],
                    &[],
                    &[],
                );
                self.submit()?;
                let mapped = self
                    .device
                    .map_memory(
                        readback.memory,
                        0,
                        vk::WHOLE_SIZE,
                        vk::MemoryMapFlags::empty(),
                    )
                    .map_err(|p| error("vulkan_readback_map_failed", p))?;
                let invalidate =
                    self.device
                        .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                            .memory(readback.memory)
                            .offset(0)
                            .size(vk::WHOLE_SIZE)]);
                let values = if invalidate.is_ok() {
                    let mut values = [0_f32; 16];
                    std::ptr::copy_nonoverlapping(
                        mapped.cast::<f32>(),
                        values.as_mut_ptr(),
                        values.len(),
                    );
                    values
                } else {
                    [f32::NAN; 16]
                };
                self.device.unmap_memory(readback.memory);
                invalidate.map_err(|p| error("vulkan_readback_invalidate_failed", p))?;
                for (region, count) in counts.iter().enumerate() {
                    for (sample, index) in [0, (count - 1) % self.elements_per_chunk]
                        .into_iter()
                        .enumerate()
                    {
                        let expected = expected_output(index, loop_count);
                        for lane in 0..4 {
                            let value = values[region * 8 + sample * 4 + lane];
                            if !value.is_finite()
                                || (value - expected).abs() > expected.abs().max(1.0) * 0.0001
                            {
                                return Err(BenchmarkError::new(
                                    "gpu_result_validation_failed",
                                    "FP32 offload output did not match the scalar FMA reference.",
                                ));
                            }
                        }
                    }
                }
                Ok(())
            })();
            self.device.destroy_buffer(readback.handle, None);
            self.device.free_memory(readback.memory, None);
            result
        }
    }

    pub fn measure(
        &self,
        working_set: u64,
        loop_count: u32,
        iterations: u32,
    ) -> Result<f64, BenchmarkError> {
        let element_count = (working_set / BYTES_PER_ELEMENT) as u32;
        let counts = region_elements(element_count, self.offload_percent);
        let active_counts = counts
            .into_iter()
            .filter(|count| *count > 0)
            .collect::<Vec<_>>();
        let chunks = self
            .regions
            .iter()
            .zip(active_counts)
            .flat_map(|(region, count)| {
                dispatch_chunks(count, self.elements_per_chunk)
                    .into_iter()
                    .enumerate()
                    .map(move |(index, (_, count))| (region.first_set + index, count))
            })
            .collect::<Vec<_>>();
        // SAFETY: the command buffer and all referenced handles are owned by this harness.
        unsafe {
            self.begin()?;
            self.device
                .cmd_reset_query_pool(self.command_buffer, self.query_pool, 0, 2);
            self.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                0,
            );
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            for _ in 0..iterations {
                // Repeated dispatches write the same outputs: order their writes explicitly.
                self.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[],
                );
                for &(index, count) in &chunks {
                    self.device.cmd_bind_descriptor_sets(
                        self.command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipeline_layout,
                        0,
                        &[self.descriptor_sets[index]],
                        &[],
                    );
                    self.device.cmd_push_constants(
                        self.command_buffer,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &parameters(loop_count, count),
                    );
                    self.device.cmd_dispatch(
                        self.command_buffer,
                        (count as u64).div_ceil(WORKGROUP_SIZE) as u32,
                        1,
                        1,
                    );
                }
            }
            self.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                1,
            );
            self.submit()?;
            let mut timestamps = [0_u64; 2];
            self.device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|problem| error("vulkan_timestamp_failed", problem))?;
            let elapsed = timestamp_delta(timestamps[0], timestamps[1], self.timestamp_valid_bits)
                as f64
                * self.timestamp_period_ns;
            if elapsed <= 0.0 || !elapsed.is_finite() {
                return Err(BenchmarkError::new(
                    "vulkan_timestamp_invalid",
                    "The compute-profile timestamp was not finite and positive.",
                ));
            }
            Ok(elapsed)
        }
    }

    unsafe fn begin(&self) -> Result<(), BenchmarkError> {
        unsafe {
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
        }
        .map_err(|problem| error("vulkan_command_reset_failed", problem))?;
        unsafe {
            self.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|problem| error("vulkan_command_begin_failed", problem))
    }

    unsafe fn submit(&self) -> Result<(), BenchmarkError> {
        let activity_started = std::time::Instant::now();
        unsafe { self.device.end_command_buffer(self.command_buffer) }
            .map_err(|problem| error("vulkan_command_end_failed", problem))?;
        let buffers = [self.command_buffer];
        unsafe {
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&buffers)],
                vk::Fence::null(),
            )
        }
        .map_err(|problem| error("vulkan_queue_submit_failed", problem))?;
        unsafe { self.device.queue_wait_idle(self.queue) }
            .map_err(|problem| error("vulkan_device_lost", problem))?;
        gluj_bench_core::pace_gpu(activity_started.elapsed())
    }
}

impl Drop for VulkanComputeProfileHarness {
    fn drop(&mut self) {
        // SAFETY: all resources are exclusively owned and the device is made idle first.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_command_pool(self.command_pool, None);
            for buffer in &self.buffers {
                self.device.destroy_buffer(buffer.handle, None);
                self.device.free_memory(buffer.memory, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub(super) fn host_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|index| {
        let ty = properties.memory_types[*index as usize];
        bits & (1 << index) != 0
            && ty
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            && !properties.memory_heaps[ty.heap_index as usize]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
    })
}

fn region_elements(elements: u32, offload_percent: u32) -> [u32; 2] {
    let host = (elements as u64 * offload_percent as u64 / 100) as u32;
    [host, elements - host]
}

fn expected_output(index: u32, loop_count: u32) -> f32 {
    let perturbation = ((index ^ 0x9e37_79b9) & 255) as f32 * 0.000001;
    let mut values = [0.5 + perturbation, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57];
    let multiplier = 0.5_f32 * 0.000002 + 0.999999;
    for _ in 0..loop_count {
        for (index, value) in values.iter_mut().enumerate() {
            *value = value.mul_add(multiplier, (index + 1) as f32 * 0.000001);
        }
    }
    values.into_iter().sum()
}

unsafe fn create_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
    host: bool,
) -> Result<Buffer, BenchmarkError> {
    let handle = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|p| error("vulkan_buffer_failed", p))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
    let properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let memory_type = if host {
        host_memory_type(&properties, requirements.memory_type_bits)
    } else {
        (0..properties.memory_type_count).find(|index| {
            requirements.memory_type_bits & (1 << index) != 0
                && properties.memory_types[*index as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
    };
    let Some(memory_type) = memory_type else {
        unsafe { device.destroy_buffer(handle, None) };
        return Err(BenchmarkError::new(
            "vulkan_memory_type_unavailable",
            if host {
                "No host-visible buffer-compatible memory type on a separate system-RAM heap. BAR-mapped VRAM is excluded."
            } else {
                "No device-local memory type can back the compute-profile buffers."
            },
        ));
    };
    let memory = match unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    } {
        Ok(memory) => memory,
        Err(problem) => {
            unsafe { device.destroy_buffer(handle, None) };
            return Err(error("vulkan_memory_allocation_failed", problem));
        }
    };
    if let Err(problem) = unsafe { device.bind_buffer_memory(handle, memory, 0) } {
        unsafe {
            device.destroy_buffer(handle, None);
            device.free_memory(memory, None);
        }
        return Err(error("vulkan_buffer_bind_failed", problem));
    }
    Ok(Buffer { handle, memory })
}

unsafe fn find_physical_device(
    instance: &ash::Instance,
    expected_uuid: &str,
) -> Result<vk::PhysicalDevice, BenchmarkError> {
    for device in unsafe { instance.enumerate_physical_devices() }
        .map_err(|problem| error("vulkan_device_enumeration_failed", problem))?
    {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        unsafe { instance.get_physical_device_properties2(device, &mut properties) };
        if hex_bytes(&id.device_uuid) == expected_uuid {
            return Ok(device);
        }
    }
    Err(BenchmarkError::new(
        "vulkan_adapter_not_found",
        "The selected Vulkan device UUID is no longer present.",
    ))
}

fn shader_words() -> Result<Vec<u32>, BenchmarkError> {
    if !PROFILE_SPV.len().is_multiple_of(4) {
        return Err(BenchmarkError::new(
            "invalid_spirv",
            "Embedded compute-profile SPIR-V is not word-aligned.",
        ));
    }
    Ok(PROFILE_SPV
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

fn chunk_element_limit(max_workgroups: u32, offset_alignment: u64) -> u32 {
    // Vulkan buffer-offset alignment is a power of two. Whole workgroups use
    // 4096 bytes per binding; align chunk boundaries for stricter devices too.
    let groups_per_alignment = offset_alignment.max(WORKGROUP_SIZE * 16) / (WORKGROUP_SIZE * 16);
    let groups = (max_workgroups as u64).min(u32::MAX as u64 / WORKGROUP_SIZE);
    (groups / groups_per_alignment * groups_per_alignment * WORKGROUP_SIZE) as u32
}

fn dispatch_chunks(elements: u32, limit: u32) -> Vec<(u32, u32)> {
    (0..elements)
        .step_by(limit as usize)
        .map(|offset| (offset, limit.min(elements - offset)))
        .collect()
}

fn parameters(loop_count: u32, element_count: u32) -> [u8; 16] {
    let mut bytes = [0_u8; 16];
    bytes[0..4].copy_from_slice(&loop_count.to_le_bytes());
    bytes[4..8].copy_from_slice(&element_count.to_le_bytes());
    bytes[8..12].copy_from_slice(&0x9e37_79b9_u32.to_le_bytes());
    bytes
}

fn timestamp_delta(start: u64, end: u64, valid_bits: u32) -> u64 {
    let mask = if valid_bits >= 64 {
        u64::MAX
    } else {
        (1_u64 << valid_bits) - 1
    };
    end.wrapping_sub(start) & mask
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
    fn profile_shader_is_spirv() {
        assert_eq!(shader_words().unwrap()[0], 0x0723_0203);
    }

    #[test]
    fn accounting_matches_shader_contract() {
        assert_eq!(BYTES_PER_ELEMENT, 48);
        assert_eq!(OPERATIONS_PER_LOOP_PER_ELEMENT, 64);
    }

    #[test]
    fn offload_selection_excludes_bar_vram_and_respects_buffer_memory_bits() {
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: 2,
            memory_type_count: 3,
            ..Default::default()
        };
        properties.memory_heaps[0] = vk::MemoryHeap {
            size: 8 << 30,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        properties.memory_heaps[1] = vk::MemoryHeap {
            size: 32 << 30,
            flags: vk::MemoryHeapFlags::empty(),
        };
        properties.memory_types[0] = vk::MemoryType {
            heap_index: 0,
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };
        properties.memory_types[1] = vk::MemoryType {
            heap_index: 0,
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE,
        };
        properties.memory_types[2] = vk::MemoryType {
            heap_index: 1,
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
        };
        assert_eq!(host_memory_type(&properties, 0b111), Some(2));
        assert_eq!(host_memory_type(&properties, 0b011), None);
        properties.memory_heaps[1].flags = vk::MemoryHeapFlags::DEVICE_LOCAL;
        assert_eq!(host_memory_type(&properties, u32::MAX), None);
    }

    #[test]
    fn every_aligned_offload_tier_has_exact_split_and_complete_dispatch_coverage() {
        for alignment in [256, 4096] {
            for quarters in [1, 13, 8192] {
                let elements = alignment * 4 * quarters;
                for percent in [0, 50, 75, 100] {
                    let [host, local] = region_elements(elements, percent);
                    assert_eq!(host + local, elements);
                    assert_eq!(host as u64 * 100, elements as u64 * percent as u64);
                    for count in [host, local] {
                        assert_eq!(count % alignment, 0);
                        let limit = alignment * 7;
                        let chunks = dispatch_chunks(count, limit);
                        assert_eq!(chunks.iter().map(|(_, count)| count).sum::<u32>(), count);
                        assert!(
                            chunks
                                .iter()
                                .all(|(offset, count)| offset % alignment == 0 && *count <= limit)
                        );
                    }
                }
            }
        }
        assert!(expected_output(0, 64).is_finite());
        assert!(expected_output(255, 64) > 4.0);
    }

    #[test]
    fn large_dispatch_chunks_cover_buffers_once_within_device_limits() {
        let limit = chunk_element_limit(65535, 256);
        let elements = (5_u64 * 1024 * 1024 * 1024 / BYTES_PER_ELEMENT / WORKGROUP_SIZE
            * WORKGROUP_SIZE) as u32;
        let chunks = dispatch_chunks(elements, limit);
        assert!(chunks.len() > 1);
        let mut covered = 0;
        for (offset, count) in chunks {
            assert_eq!(offset, covered);
            assert_eq!(offset as u64 * 16 % 256, 0);
            assert_eq!(count as u64 % WORKGROUP_SIZE, 0);
            assert!((count as u64).div_ceil(WORKGROUP_SIZE) <= 65535);
            assert!(offset as u64 * 16 + count as u64 * 16 <= elements as u64 * 16);
            covered += count;
        }
        assert_eq!(covered, elements);
    }

    #[test]
    fn chunk_limit_handles_stricter_alignment_and_large_dispatch_limits() {
        let limit = chunk_element_limit(65535, 65536);
        assert_eq!(limit as u64 * 16 % 65536, 0);
        assert!(limit / WORKGROUP_SIZE as u32 <= 65535);
        assert_eq!(
            chunk_element_limit(u32::MAX, 256) % WORKGROUP_SIZE as u32,
            0
        );
    }
}
