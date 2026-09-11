use crate::{AdapterRecord, WORKGROUP_SIZE, compute::ComputeKind};
use ash::{Entry, vk};
use gluj_bench_core::BenchmarkError;

const FP16_SPV: &[u8] = include_bytes!("../shaders/spv/vector_fp16.spv");
const FP32_SPV: &[u8] = include_bytes!("../shaders/spv/vector_fp32.spv");
const FP64_SPV: &[u8] = include_bytes!("../shaders/spv/vector_fp64.spv");

struct Buffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
}

pub(super) struct VulkanVectorHarness {
    _entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    query_pool: vk::QueryPool,
    output: Buffer,
    timestamp_period_ns: f64,
    timestamp_valid_bits: u32,
    workgroups: u32,
    loop_count: u32,
    pub output_size: u64,
}

impl VulkanVectorHarness {
    pub fn new(
        record: &AdapterRecord,
        kind: ComputeKind,
        workgroups: u32,
        loop_count: u32,
    ) -> Result<Self, BenchmarkError> {
        // SAFETY: every created Vulkan handle is owned by the returned harness and destroyed
        // in reverse dependency order by Drop.
        unsafe { Self::new_inner(record, kind, workgroups, loop_count) }
    }

    unsafe fn new_inner(
        record: &AdapterRecord,
        kind: ComputeKind,
        workgroups: u32,
        loop_count: u32,
    ) -> Result<Self, BenchmarkError> {
        let entry = unsafe { Entry::load() }
            .map_err(|problem| error("vulkan_loader_unavailable", problem))?;
        let app_name = c"Gluj-Bench";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(1)
            .engine_name(app_name)
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
        let priority = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(record.vulkan.compute_queue_family)
            .queue_priorities(&priority)];
        let mut core_features = vk::PhysicalDeviceFeatures::default();
        if kind == ComputeKind::Fp64 {
            core_features.shader_float64 = vk::TRUE;
        }
        let mut float16_features = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(kind == ComputeKind::Fp16);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_features(&core_features)
            .push_next(&mut float16_features);
        let device = match unsafe { instance.create_device(physical_device, &device_info, None) } {
            Ok(device) => device,
            Err(problem) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error("vulkan_device_failed", problem));
            }
        };
        let queue = unsafe { device.get_device_queue(record.vulkan.compute_queue_family, 0) };

        let output_size = workgroups as u64 * WORKGROUP_SIZE * 16;
        let output =
            match unsafe { create_device_buffer(&instance, &device, physical_device, output_size) }
            {
                Ok(buffer) => buffer,
                Err(problem) => {
                    unsafe { device.destroy_device(None) };
                    unsafe { instance.destroy_instance(None) };
                    return Err(problem);
                }
            };

        let descriptor_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_binding),
                None,
            )
        }
        .map_err(|problem| error("vulkan_descriptor_layout_failed", problem))?;
        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let set_layouts = [descriptor_layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_range),
                None,
            )
        }
        .map_err(|problem| error("vulkan_pipeline_layout_failed", problem))?;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|problem| error("vulkan_descriptor_pool_failed", problem))?;
        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(|problem| error("vulkan_descriptor_set_failed", problem))?[0];
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(output.handle)
            .offset(0)
            .range(output_size)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info)];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let shader_words = shader_words(kind)?;
        let shader = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&shader_words),
                None,
            )
        }
        .map_err(|problem| error("vulkan_shader_module_failed", problem))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main");
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];
        let pipeline_result = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
        };
        unsafe { device.destroy_shader_module(shader, None) };
        let pipeline = pipeline_result
            .map_err(|(_, problem)| error("vulkan_compute_pipeline_failed", problem))?[0];
        let query_pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .map_err(|problem| error("vulkan_query_pool_failed", problem))?;
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

        Ok(Self {
            _entry: entry,
            instance,
            device,
            queue,
            command_pool,
            command_buffer,
            descriptor_pool,
            descriptor_layout,
            descriptor_set,
            pipeline_layout,
            pipeline,
            query_pool,
            output,
            timestamp_period_ns: record.vulkan.timestamp_period_ns as f64,
            timestamp_valid_bits: record.vulkan.timestamp_valid_bits,
            workgroups,
            loop_count,
            output_size,
        })
    }

    pub fn operations_per_dispatch(&self) -> u64 {
        self.workgroups as u64 * WORKGROUP_SIZE * self.loop_count as u64 * 64
    }

    pub fn measure(&self, iterations: u32) -> Result<f64, BenchmarkError> {
        let invocation_count = self.workgroups.saturating_mul(WORKGROUP_SIZE as u32);
        let params = parameters(self.loop_count, invocation_count);
        // SAFETY: command buffer and all referenced objects belong to this live device and the
        // queue is waited idle before the command pool is reused.
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
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &params,
            );
            for _ in 0..iterations {
                self.device
                    .cmd_dispatch(self.command_buffer, self.workgroups, 1, 1);
            }
            self.device.cmd_write_timestamp(
                self.command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                1,
            );
            self.device
                .end_command_buffer(self.command_buffer)
                .map_err(|problem| error("vulkan_command_end_failed", problem))?;
            let command_buffers = [self.command_buffer];
            let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submits, vk::Fence::null())
                .map_err(|problem| error("vulkan_queue_submit_failed", problem))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|problem| error("vulkan_device_lost", problem))?;
            let mut timestamps = [0_u64; 2];
            self.device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|problem| error("vulkan_timestamp_failed", problem))?;
            let ticks = timestamp_delta(timestamps[0], timestamps[1], self.timestamp_valid_bits);
            let elapsed_ns = ticks as f64 * self.timestamp_period_ns;
            if elapsed_ns <= 0.0 || !elapsed_ns.is_finite() {
                return Err(BenchmarkError::new(
                    "vulkan_timestamp_invalid",
                    "The Vulkan timestamp interval was not finite and positive.",
                ));
            }
            Ok(elapsed_ns)
        }
    }
}

impl Drop for VulkanVectorHarness {
    fn drop(&mut self) {
        // SAFETY: all handles belong to this device/instance and are destroyed once, in reverse
        // dependency order after waiting for submitted work.
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
            self.device.destroy_buffer(self.output.handle, None);
            self.device.free_memory(self.output.memory, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
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

unsafe fn create_device_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    size: u64,
) -> Result<Buffer, BenchmarkError> {
    let handle = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|problem| error("vulkan_buffer_failed", problem))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let memory_type_index = (0..memory_properties.memory_type_count)
        .find(|index| {
            requirements.memory_type_bits & (1 << index) != 0
                && memory_properties.memory_types[*index as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .ok_or_else(|| {
            BenchmarkError::new(
                "vulkan_device_memory_unavailable",
                "No device-local Vulkan memory type can back the vector output buffer.",
            )
        })?;
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index),
            None,
        )
    }
    .map_err(|problem| error("vulkan_memory_allocation_failed", problem))?;
    unsafe { device.bind_buffer_memory(handle, memory, 0) }
        .map_err(|problem| error("vulkan_buffer_bind_failed", problem))?;
    Ok(Buffer { handle, memory })
}

fn shader_words(kind: ComputeKind) -> Result<Vec<u32>, BenchmarkError> {
    let bytes = match kind {
        ComputeKind::Fp16 => FP16_SPV,
        ComputeKind::Fp32 => FP32_SPV,
        ComputeKind::Fp64 => FP64_SPV,
    };
    if !bytes.len().is_multiple_of(4) {
        return Err(BenchmarkError::new(
            "invalid_spirv",
            "Embedded Vulkan shader byte length is not word-aligned.",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("SPIR-V word")))
        .collect())
}

fn parameters(loop_count: u32, invocation_count: u32) -> [u8; 16] {
    let mut bytes = [0_u8; 16];
    bytes[0..4].copy_from_slice(&loop_count.to_le_bytes());
    bytes[4..8].copy_from_slice(&0x9e37_79b9_u32.to_le_bytes());
    bytes[8..12].copy_from_slice(&invocation_count.to_le_bytes());
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
    fn embedded_shaders_are_spirv_modules() {
        for kind in [ComputeKind::Fp16, ComputeKind::Fp32, ComputeKind::Fp64] {
            assert_eq!(shader_words(kind).unwrap()[0], 0x0723_0203);
        }
    }

    #[test]
    fn timestamp_wrap_respects_valid_bit_count() {
        assert_eq!(timestamp_delta(250, 3, 8), 9);
        assert_eq!(timestamp_delta(10, 20, 64), 10);
    }
}
