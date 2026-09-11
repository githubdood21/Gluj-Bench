use crate::{AdapterRecord, WORKGROUP_SIZE};
use ash::{Entry, vk};
use gluj_bench_core::BenchmarkError;

const PROFILE_SPV: &[u8] = include_bytes!("../shaders/spv/compute_profile_fp32.spv");
const BYTES_PER_ELEMENT: u64 = 48;
const OPERATIONS_PER_LOOP_PER_ELEMENT: u64 = 64;

struct Buffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
}

pub(super) struct VulkanComputeProfileHarness {
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
    input_a: Buffer,
    input_b: Buffer,
    output: Buffer,
    timestamp_period_ns: f64,
    timestamp_valid_bits: u32,
    pub maximum_working_set_bytes: u64,
}

impl VulkanComputeProfileHarness {
    pub fn new(record: &AdapterRecord, requested_working_set: u64) -> Result<Self, BenchmarkError> {
        // SAFETY: every Vulkan handle is owned by the returned harness and destroyed by Drop.
        unsafe { Self::new_inner(record, requested_working_set) }
    }

    unsafe fn new_inner(
        record: &AdapterRecord,
        requested_working_set: u64,
    ) -> Result<Self, BenchmarkError> {
        let element_count = requested_working_set / BYTES_PER_ELEMENT;
        let per_buffer_size = element_count * 16;
        if element_count < WORKGROUP_SIZE || element_count > u32::MAX as u64 {
            return Err(BenchmarkError::new(
                "invalid_working_set",
                "The compute-profile working set is outside the supported shader index range.",
            ));
        }
        let maximum_working_set_bytes = element_count * BYTES_PER_ELEMENT;
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
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let input_a =
            unsafe { create_buffer(&instance, &device, physical_device, per_buffer_size, usage) }?;
        let input_b =
            unsafe { create_buffer(&instance, &device, physical_device, per_buffer_size, usage) }?;
        let output =
            unsafe { create_buffer(&instance, &device, physical_device, per_buffer_size, usage) }?;
        let bindings = [0_u32, 1, 2].map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|problem| error("vulkan_descriptor_layout_failed", problem))?;
        let set_layouts = [descriptor_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )
        }
        .map_err(|problem| error("vulkan_pipeline_layout_failed", problem))?;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
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
        let buffer_infos = [input_a.handle, input_b.handle, output.handle].map(|buffer| {
            vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(per_buffer_size)
        });
        let writes = [0_u32, 1, 2].map(|binding| {
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[binding as usize]))
        });
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        let words = shader_words()?;
        let shader = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|problem| error("vulkan_shader_module_failed", problem))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main");
        let pipeline_result = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            )
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
        let harness = Self {
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
            input_a,
            input_b,
            output,
            timestamp_period_ns: record.vulkan.timestamp_period_ns as f64,
            timestamp_valid_bits: record.vulkan.timestamp_valid_bits,
            maximum_working_set_bytes,
        };
        harness.initialize(per_buffer_size)?;
        Ok(harness)
    }

    fn initialize(&self, per_buffer_size: u64) -> Result<(), BenchmarkError> {
        // SAFETY: buffers have TRANSFER_DST usage and no benchmark submission is in flight.
        unsafe {
            self.begin()?;
            self.device.cmd_fill_buffer(
                self.command_buffer,
                self.input_a.handle,
                0,
                per_buffer_size,
                0x3f00_0000,
            );
            self.device.cmd_fill_buffer(
                self.command_buffer,
                self.input_b.handle,
                0,
                per_buffer_size,
                0x3f00_0000,
            );
            self.device.cmd_fill_buffer(
                self.command_buffer,
                self.output.handle,
                0,
                per_buffer_size,
                0,
            );
            self.submit()?;
        }
        Ok(())
    }

    pub fn actual_working_set(&self, requested: u64) -> u64 {
        let elements =
            (requested.min(self.maximum_working_set_bytes) / BYTES_PER_ELEMENT / WORKGROUP_SIZE)
                * WORKGROUP_SIZE;
        elements.max(WORKGROUP_SIZE) * BYTES_PER_ELEMENT
    }

    pub fn operations_per_dispatch(&self, working_set: u64, loop_count: u32) -> u64 {
        (working_set / BYTES_PER_ELEMENT) * loop_count as u64 * OPERATIONS_PER_LOOP_PER_ELEMENT
    }

    pub fn traffic_bytes_per_dispatch(&self, working_set: u64) -> u64 {
        working_set
    }

    pub fn measure(
        &self,
        working_set: u64,
        loop_count: u32,
        iterations: u32,
    ) -> Result<f64, BenchmarkError> {
        let element_count = (working_set / BYTES_PER_ELEMENT) as u32;
        let workgroups = (element_count as u64).div_ceil(WORKGROUP_SIZE) as u32;
        let params = parameters(loop_count, element_count);
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
                    .cmd_dispatch(self.command_buffer, workgroups, 1, 1);
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
            .map_err(|problem| error("vulkan_device_lost", problem))
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
            for buffer in [&self.input_a, &self.input_b, &self.output] {
                self.device.destroy_buffer(buffer.handle, None);
                self.device.free_memory(buffer.memory, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn create_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
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
    .map_err(|problem| error("vulkan_buffer_failed", problem))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
    let properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let memory_type = (0..properties.memory_type_count)
        .find(|index| {
            requirements.memory_type_bits & (1 << index) != 0
                && properties.memory_types[*index as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .ok_or_else(|| {
            BenchmarkError::new(
                "vulkan_device_memory_unavailable",
                "No device-local memory type can back the compute-profile buffers.",
            )
        })?;
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    }
    .map_err(|problem| error("vulkan_memory_allocation_failed", problem))?;
    unsafe { device.bind_buffer_memory(handle, memory, 0) }
        .map_err(|problem| error("vulkan_buffer_bind_failed", problem))?;
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
}
