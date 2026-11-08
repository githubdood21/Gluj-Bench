mod offered;
use crate::{AdapterRecord, WORKGROUP_SIZE};
use ash::{Entry, vk};
use gluj_bench_core::BenchmarkError;

const PROFILE_SPV: &[u8] = include_bytes!("../shaders/spv/compute_profile_fp32.spv");
const BYTES_PER_ELEMENT: u64 = 48;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Precision {
    Fp16,
    Fp32,
    Fp64,
}
impl Precision {
    pub fn vector_bytes(self) -> u64 {
        match self {
            Self::Fp16 => 8,
            Self::Fp32 => 16,
            Self::Fp64 => 32,
        }
    }
    pub fn bytes(self) -> u64 {
        if self == Self::Fp32 {
            BYTES_PER_ELEMENT
        } else {
            self.vector_bytes() * 3
        }
    }
    fn shader(self) -> &'static [u8] {
        match self {
            Self::Fp16 => include_bytes!("../shaders/spv/compute_profile_fp16.spv"),
            Self::Fp32 => PROFILE_SPV,
            Self::Fp64 => include_bytes!("../shaders/spv/compute_profile_fp64.spv"),
        }
    }
    fn fill(self) -> u32 {
        match self {
            Self::Fp16 => 0x38003800,
            Self::Fp32 => 0x3f000000,
            Self::Fp64 => 0x3fe00000,
        }
    }
}
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
    precision: Precision,
    access: std::cell::Cell<(u32, u64, u32)>,
    _entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    prepared_measurement: std::cell::Cell<Option<(u64, u32, u32)>>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
    elements_per_chunk: u32,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pipelines: Vec<(u64, vk::Pipeline)>,
    query_pool: vk::QueryPool,
    buffers: Vec<Buffer>,
    regions: Vec<Region>,
    element_alignment: u64,
    offload_percent: u32,
    workgroup_size: u64,
    timestamp_period_ns: f64,
    timestamp_valid_bits: u32,
    pub maximum_working_set_bytes: u64,
}

impl VulkanComputeProfileHarness {
    pub fn new(record: &AdapterRecord, requested_working_set: u64) -> Result<Self, BenchmarkError> {
        // SAFETY: every Vulkan handle is owned by the returned harness and destroyed by Drop.
        unsafe {
            Self::new_inner(
                record,
                requested_working_set,
                0,
                WORKGROUP_SIZE,
                7,
                Precision::Fp32,
            )
        }
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
        unsafe {
            Self::new_inner(
                record,
                requested_working_set,
                offload_percent,
                WORKGROUP_SIZE,
                7,
                Precision::Fp32,
            )
        }
    }

    pub fn new_precision(
        record: &AdapterRecord,
        bytes: u64,
        precision: Precision,
        offload: u32,
    ) -> Result<Self, BenchmarkError> {
        if record.vulkan.api_version < vk::API_VERSION_1_1 {
            return Err(BenchmarkError::new(
                "unsupported_vulkan_version",
                "Buffered shaders require Vulkan 1.1 or newer.",
            ));
        }
        if ![0, 50, 75, 100].contains(&offload) || (offload > 0 && precision != Precision::Fp32) {
            return Err(BenchmarkError::new(
                "unsupported_configuration",
                "Only FP32 supports direct host placement.",
            ));
        }
        if (precision == Precision::Fp16
            && (!record.vulkan.shader_float16
                || !record.vulkan.storage_buffer_16bit_access
                || (record.vulkan.api_version < vk::API_VERSION_1_2
                    && !record
                        .vulkan
                        .extensions
                        .contains("VK_KHR_shader_float16_int8"))))
            || (precision == Precision::Fp64 && !record.vulkan.shader_float64)
        {
            return Err(BenchmarkError::new(
                "unsupported_precision",
                "Buffered precision requires native arithmetic and storage support.",
            ));
        }
        // SAFETY: the returned owner destroys all partially constructed resources.
        unsafe { Self::new_inner(record, bytes, offload, WORKGROUP_SIZE, 7, precision) }
    }
    pub fn configure_access(&self, mode: u32, seed: u64, epoch: u32) {
        self.access.set((mode, seed, epoch));
        self.prepared_measurement.set(None);
    }

    unsafe fn new_inner(
        record: &AdapterRecord,
        requested_working_set: u64,
        offload_percent: u32,
        workgroup_size: u64,
        host_buffer_mask: u8,
        precision: Precision,
    ) -> Result<Self, BenchmarkError> {
        if ![32, 64, 128, 256].contains(&workgroup_size) {
            return Err(BenchmarkError::new(
                "invalid_config",
                "Unsupported profile workgroup size.",
            ));
        }
        let element_count =
            requested_working_set / precision.bytes() / WORKGROUP_SIZE * WORKGROUP_SIZE;
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
            ((limits.max_compute_work_group_count[0] as u64 * 32 / WORKGROUP_SIZE) as u32).min(
                (limits.max_storage_buffer_range as u64 / precision.vector_bytes() / WORKGROUP_SIZE)
                    as u32,
            ),
            limits.min_storage_buffer_offset_alignment * 16 / precision.vector_bytes(),
        );
        if elements_per_chunk == 0 {
            unsafe { instance.destroy_instance(None) };
            return Err(BenchmarkError::new(
                "invalid_dispatch_limit",
                "The device cannot dispatch an aligned compute-profile chunk.",
            ));
        }
        let element_alignment = WORKGROUP_SIZE
            .max(limits.min_storage_buffer_offset_alignment / precision.vector_bytes());
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
        let maximum_working_set_bytes = element_count * precision.bytes();
        let priorities = [1.0_f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(record.vulkan.compute_queue_family)
            .queue_priorities(&priorities)];
        let features =
            vk::PhysicalDeviceFeatures::default().shader_float64(precision == Precision::Fp64);
        let mut f16 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(precision == Precision::Fp16);
        let mut storage = vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(precision == Precision::Fp16);
        let extensions = [c"VK_KHR_shader_float16_int8".as_ptr()];
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_features(&features);
        if precision == Precision::Fp16 {
            device_info = device_info.push_next(&mut f16).push_next(&mut storage);
            if record.vulkan.api_version < vk::API_VERSION_1_2 {
                device_info = device_info.enabled_extension_names(&extensions);
            }
        }
        let device = match unsafe { instance.create_device(physical_device, &device_info, None) } {
            Ok(device) => device,
            Err(problem) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error("vulkan_device_failed", problem));
            }
        };
        let queue = unsafe { device.get_device_queue(record.vulkan.compute_queue_family, 0) };
        // Construct the owner before allocating children. Drop also handles setup failures.
        let mut harness = Self {
            precision,
            access: std::cell::Cell::new((0, 0, 0)),
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            prepared_measurement: std::cell::Cell::new(None),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_sets: Vec::new(),
            elements_per_chunk,
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pipelines: Vec::new(),
            query_pool: vk::QueryPool::null(),
            buffers: Vec::new(),
            regions: Vec::new(),
            element_alignment: total_alignment,
            offload_percent,
            workgroup_size,
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
            for binding in 0..3 {
                harness.buffers.push(unsafe {
                    create_buffer(
                        &harness.instance,
                        &harness.device,
                        physical_device,
                        count as u64 * precision.vector_bytes(),
                        usage,
                        region_index == 0
                            && offload_percent > 0
                            && (host_buffer_mask & (1 << binding)) != 0,
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
            .size(32)];
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
                    .offset(offset as u64 * precision.vector_bytes())
                    .range(count as u64 * precision.vector_bytes())
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
        harness.select_workgroup_size(workgroup_size)?;
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

    pub fn workgroup_size(&self) -> u64 {
        self.workgroup_size
    }

    pub(super) fn select_workgroup_size(
        &mut self,
        workgroup_size: u64,
    ) -> Result<(), BenchmarkError> {
        self.prepared_measurement.set(None);
        self.workgroup_size = workgroup_size;
        if let Some((_, pipeline)) = self
            .pipelines
            .iter()
            .find(|(size, _)| *size == workgroup_size)
        {
            self.pipeline = *pipeline;
            return Ok(());
        }
        // SAFETY: no command is in flight; all cached pipelines share this layout.
        // Shader modules can be destroyed once the pipeline has been created.
        let words = words_from(self.precision.shader())?;
        let shader = unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|p| error("vulkan_shader_module_failed", p))?;
        let group_bytes = (workgroup_size as u32).to_le_bytes();
        let entries = [vk::SpecializationMapEntry::default()
            .constant_id(0)
            .offset(0)
            .size(4)];
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&group_bytes);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .specialization_info(&specialization)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main");
        let pipeline = unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(self.pipeline_layout)],
                None,
            )
        };
        unsafe { self.device.destroy_shader_module(shader, None) };
        let pipeline = match pipeline {
            Ok(pipelines) => pipelines[0],
            Err((pipelines, problem)) => {
                for pipeline in pipelines {
                    unsafe { self.device.destroy_pipeline(pipeline, None) };
                }
                return Err(error("vulkan_compute_pipeline_failed", problem));
            }
        };
        self.pipelines.push((workgroup_size, pipeline));
        self.pipeline = pipeline;
        Ok(())
    }

    /// Tune launch shape, preserving bytes, work per value, placement and synchronization.
    pub fn autotune_workgroups(
        &mut self,
        sizes: &[u64],
        loops: u32,
        cancellation: &gluj_bench_core::CancellationToken,
    ) -> Result<String, BenchmarkError> {
        let _calibration_scope =
            gluj_bench_core::MeasurementScope::phase("calibration-launch-shape");
        let budget = gluj_bench_core::current_reference_budget();
        let first = sizes[0];
        let mut probes = vec![
            first,
            self.actual_working_set(sizes[sizes.len() - 1].min(2 * 1024 * 1024)),
        ];
        probes.sort_unstable();
        probes.dedup();
        let mut measurements = Vec::new();
        let mut log = Vec::new();
        for group in [256, 128, 64, 32, 256] {
            self.select_workgroup_size(group)?;
            let mut times = Vec::new();
            let mut stable = true;
            for &size in &probes {
                let cold = self.measure(size, loops, 1)?;
                let passes = (budget.minimum_sample_ns as f64 / cold).clamp(1.0, 4096.0) as u32;
                let warm_batches = budget.warmup_ns.div_ceil(budget.minimum_sample_ns).max(1);
                for _ in 0..warm_batches {
                    if cancellation.is_cancelled() {
                        return Err(BenchmarkError::new(
                            "cancelled",
                            "The benchmark was cancelled.",
                        ));
                    }
                    self.measure(size, loops, passes)?;
                }
                let mut values = Vec::new();
                for _ in 0..budget.minimum_samples {
                    if cancellation.is_cancelled() {
                        return Err(BenchmarkError::new(
                            "cancelled",
                            "The benchmark was cancelled.",
                        ));
                    }
                    let observed = gluj_bench_core::measure_batches(
                        budget.minimum_sample_ns,
                        cancellation,
                        || self.measure(size, loops, passes),
                    )?;
                    let completed = passes.saturating_mul(observed.batches);
                    values.push(observed.active_ns / f64::from(completed));
                    gluj_bench_core::record_observation(
                        &format!("launch-{group}:bytes-{size}"),
                        "dispatch_time",
                        *values.last().unwrap(),
                        "ns",
                        self.operations_per_dispatch(size, loops)
                            .saturating_mul(u64::from(completed)),
                        "operations",
                        observed.active_ns as u64,
                        observed.elapsed_ns,
                    );
                }
                let stats = crate::analysis::statistics(&values);
                stable &= stats.standard_deviation / stats.median < 0.03;
                log.push(format!(
                    "{group}@{size}:{:.1}ns(cv={:.2}%)",
                    stats.median,
                    stats.standard_deviation / stats.median * 100.0
                ));
                times.push(stats.median);
            }
            measurements.push((group, times, stable));
        }
        let baseline = &measurements[0];
        let confirmation = &measurements[4];
        let stable_baseline = baseline.2
            && confirmation.2
            && baseline
                .1
                .iter()
                .zip(&confirmation.1)
                .all(|(a, b)| (a / b - 1.0).abs() <= 0.05);
        let mut selected = 256;
        let mut best: f64 = 1.0;
        if stable_baseline {
            for (group, times, stable) in &measurements[1..4] {
                let score = times
                    .iter()
                    .zip(&baseline.1)
                    .zip(&confirmation.1)
                    .map(|((t, a), b)| (a + b) * 0.5 / t)
                    .sum::<f64>()
                    / times.len() as f64;
                if *stable && score > best.max(1.02_f64) {
                    best = score;
                    selected = *group;
                }
            }
        }
        self.select_workgroup_size(selected)?;
        log.push(format!(
            "selected:{selected};baseline_stable:{stable_baseline}"
        ));
        Ok(log.join(","))
    }

    fn initialize(&self) -> Result<(), BenchmarkError> {
        // SAFETY: buffers have TRANSFER_DST usage and no submission is in flight.
        unsafe {
            self.begin()?;
            for region in &self.regions {
                for index in 0..3 {
                    if !(self.precision == Precision::Fp64 && index != 2) {
                        self.device.cmd_fill_buffer(
                            self.command_buffer,
                            self.buffers[region.first_buffer + index].handle,
                            0,
                            region.elements as u64 * self.precision.vector_bytes(),
                            if index == 2 { 0 } else { self.precision.fill() },
                        );
                    }
                    if self.precision == Precision::Fp64 && index != 2 {
                        let data = vec![0.5_f64; 8192];
                        let bytes = std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), 65536);
                        let total = region.elements as u64 * self.precision.vector_bytes();
                        for offset in (0..total).step_by(65536) {
                            self.device.cmd_update_buffer(
                                self.command_buffer,
                                self.buffers[region.first_buffer + index].handle,
                                offset,
                                &bytes[..(total - offset).min(65536) as usize],
                            );
                        }
                    }
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
            / self.precision.bytes()
            / self.element_alignment)
            * self.element_alignment;
        elements.max(self.element_alignment) * self.precision.bytes()
    }

    pub fn operations_per_dispatch(&self, working_set: u64, loop_count: u32) -> u64 {
        (working_set / self.precision.bytes()) * loop_count as u64 * OPERATIONS_PER_LOOP_PER_ELEMENT
    }

    pub fn traffic_bytes_per_dispatch(&self, working_set: u64) -> u64 {
        working_set
    }

    pub fn dispatch_workgroups(&self, working_set: u64) -> Vec<u32> {
        region_elements(
            (working_set / self.precision.bytes()) as u32,
            self.offload_percent,
        )
        .into_iter()
        .filter(|count| *count > 0)
        .flat_map(|count| dispatch_chunks(count, self.elements_per_chunk))
        .map(|(_, count)| (count as u64).div_ceil(self.workgroup_size) as u32)
        .collect()
    }

    /// Check the first and last output of both memory regions outside the timed samples.
    pub fn validate_output(&self, working_set: u64, loop_count: u32) -> Result<(), BenchmarkError> {
        let _phase_timer = gluj_bench_core::phase_timer("validation");
        let counts = region_elements(
            (working_set / self.precision.bytes()) as u32,
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
                14 * self.precision.vector_bytes(),
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
                    let copies: Vec<_> = gluj_bench_core::validation_indices(*count as usize)
                        .into_iter()
                        .enumerate()
                        .map(|(sample, element)| vk::BufferCopy {
                            src_offset: element as u64 * self.precision.vector_bytes(),
                            dst_offset: (index * 7 + sample) as u64 * self.precision.vector_bytes(),
                            size: self.precision.vector_bytes(),
                        })
                        .collect();
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
                let mut values = vec![0_u8; (14 * self.precision.vector_bytes()) as usize];
                if invalidate.is_ok() {
                    std::ptr::copy_nonoverlapping(
                        mapped.cast::<u8>(),
                        values.as_mut_ptr(),
                        values.len(),
                    );
                }
                self.device.unmap_memory(readback.memory);
                invalidate.map_err(|p| error("vulkan_readback_invalidate_failed", p))?;
                for (region, count) in counts.iter().enumerate() {
                    for (sample, index) in gluj_bench_core::validation_indices(*count as usize)
                        .into_iter()
                        .enumerate()
                    {
                        let expected = expected_precision(
                            self.precision,
                            index as u32 % self.elements_per_chunk,
                            loop_count,
                        );
                        for lane in 0..4 {
                            let scalar_bytes = (self.precision.vector_bytes() / 4) as usize;
                            let start = (region * 28 + sample * 4 + lane) * scalar_bytes;
                            let value = match self.precision {
                                Precision::Fp16 => half::f16::from_bits(u16::from_le_bytes(
                                    values[start..start + 2].try_into().unwrap(),
                                ))
                                .to_f64(),
                                Precision::Fp32 => {
                                    f32::from_le_bytes(values[start..start + 4].try_into().unwrap())
                                        as f64
                                }
                                Precision::Fp64 => {
                                    f64::from_le_bytes(values[start..start + 8].try_into().unwrap())
                                }
                            };
                            if !value.is_finite()
                                || (value - expected).abs()
                                    > expected.abs().max(1.0)
                                        * if self.precision == Precision::Fp16 {
                                            0.003
                                        } else if self.precision == Precision::Fp64 {
                                            1e-10
                                        } else {
                                            0.0001
                                        }
                            {
                                return Err(BenchmarkError::new(
                                    "gpu_result_validation_failed",
                                    "Buffered output did not match its precision-specific scalar FMA reference.",
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
        let key = (working_set, loop_count, iterations);
        // SAFETY: submissions complete before reuse or reset. Query resets and writes
        // are replayed for every submission; only recording is reused, never timing data.
        unsafe {
            if self.prepared_measurement.get() != Some(key) {
                self.record_measurement(
                    self.command_buffer,
                    self.query_pool,
                    working_set,
                    loop_count,
                    iterations,
                    false,
                )?;
                self.prepared_measurement.set(Some(key));
            }
            self.submit_recorded()?;
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

    fn record_measurement(
        &self,
        command_buffer: vk::CommandBuffer,
        query_pool: vk::QueryPool,
        working_set: u64,
        loop_count: u32,
        iterations: u32,
        queued: bool,
    ) -> Result<(), BenchmarkError> {
        // SAFETY: the caller owns these idle buffers/queries; the layout and descriptors belong to self.
        unsafe {
            let element_count = (working_set / self.precision.bytes()) as u32;
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
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|p| error("vulkan_command_reset_failed", p))?;
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .map_err(|p| error("vulkan_command_begin_failed", p))?;
            self.device
                .cmd_reset_query_pool(command_buffer, query_pool, 0, 2);
            if !queued {
                self.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    0,
                );
            }
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            for iteration in 0..iterations {
                // WAW hazards need a memory dependency, not only execution ordering.
                // https://docs.vulkan.org/spec/latest/chapters/synchronization.html
                let writes = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)];
                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &writes,
                    &[],
                    &[],
                );
                if queued && iteration == 0 {
                    self.device.cmd_write_timestamp(
                        command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        query_pool,
                        0,
                    );
                }
                for &(index, count) in &chunks {
                    self.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipeline_layout,
                        0,
                        &[self.descriptor_sets[index]],
                        &[],
                    );
                    self.device.cmd_push_constants(
                        command_buffer,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &access_parameters(loop_count, count, self.access.get(), iteration),
                    );
                    self.device.cmd_dispatch(
                        command_buffer,
                        (count as u64).div_ceil(self.workgroup_size) as u32,
                        1,
                        1,
                    );
                }
            }
            self.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                query_pool,
                1,
            );
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|problem| error("vulkan_command_end_failed", problem))?;
        }
        Ok(())
    }

    unsafe fn begin(&self) -> Result<(), BenchmarkError> {
        self.prepared_measurement.set(None);
        unsafe {
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
        }
        .map_err(|problem| error("vulkan_command_reset_failed", problem))?;
        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &vk::CommandBufferBeginInfo::default())
        }
        .map_err(|problem| error("vulkan_command_begin_failed", problem))
    }

    unsafe fn submit(&self) -> Result<(), BenchmarkError> {
        unsafe { self.device.end_command_buffer(self.command_buffer) }
            .map_err(|problem| error("vulkan_command_end_failed", problem))?;
        unsafe { self.submit_recorded() }
    }

    unsafe fn submit_recorded(&self) -> Result<(), BenchmarkError> {
        let activity_started = std::time::Instant::now();
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
            for (_, pipeline) in &self.pipelines {
                self.device.destroy_pipeline(*pipeline, None);
            }
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
            *value = gluj_bench_core::reference_fma_f32(
                *value,
                multiplier,
                (index + 1) as f32 * 0.000001,
            );
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

#[cfg(test)]
fn shader_words() -> Result<Vec<u32>, BenchmarkError> {
    words_from(PROFILE_SPV)
}
fn words_from(bytes: &[u8]) -> Result<Vec<u32>, BenchmarkError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(BenchmarkError::new(
            "invalid_spirv",
            "Shader is not word aligned.",
        ));
    }
    Ok(bytes
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

fn access_parameters(
    loops: u32,
    count: u32,
    (mode, seed, epoch): (u32, u64, u32),
    iteration: u32,
) -> [u8; 32] {
    let blocks = count / 256;
    let mut mul = ((seed as u32) % 63) * 2 + 1;
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }
    while gcd(mul, blocks.max(1)) != 1 {
        mul = if mul >= 253 { 1 } else { mul + 2 };
    }
    let offset = if mode == 2 {
        epoch.wrapping_add(iteration)
    } else {
        (seed >> 32) as u32
    };
    let words = [
        loops,
        count,
        0x9e3779b9,
        mode,
        mul,
        offset % blocks.max(1),
        ((seed as u32) >> 8) | 1,
        (seed >> 32) as u32,
    ];
    let mut bytes = [0; 32];
    for (word, out) in words.iter().zip(bytes.chunks_exact_mut(4)) {
        out.copy_from_slice(&word.to_le_bytes());
    }
    bytes
}
fn expected_precision(precision: Precision, index: u32, loops: u32) -> f64 {
    match precision {
        Precision::Fp32 => expected_output(index, loops) as f64,
        Precision::Fp64 => {
            let p = ((index ^ 0x9e3779b9) & 255) as f64 * 0.000001;
            let mut values = [0.5 + p, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57];
            let m = 0.5 * 0.000002 + 0.999999;
            for _ in 0..loops {
                for (i, x) in values.iter_mut().enumerate() {
                    *x = x.mul_add(m, (i + 1) as f64 * 0.000001);
                }
            }
            values.iter().sum()
        }
        Precision::Fp16 => {
            let q = |v: f64| half::f16::from_f64(v).to_f64();
            let mut values = [
                q(q(0.5) + q(q(((index ^ 0x9e3779b9) & 255) as f64) * q(0.000001))),
                q(q(0.5) + q(0.01)),
                q(q(0.5) + q(0.02)),
                q(q(0.5) + q(0.03)),
                q(q(0.5) + q(0.04)),
                q(q(0.5) + q(0.05)),
                q(q(0.5) + q(0.06)),
                q(q(0.5) + q(0.07)),
            ];
            for _ in 0..loops {
                for (i, x) in values.iter_mut().enumerate() {
                    *x = q(x.mul_add(0.9990234375, (i + 1) as f64 * 0.00048828125));
                }
            }
            values.into_iter().reduce(|a, b| q(a + b)).unwrap()
        }
    }
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
    #[ignore = "controlled GPU diagnostic: same bytes/arithmetic, RAM placement and group sizes"]
    fn profile_small_dataset_diagnostic() {
        let provider = crate::GpuBandwidthProvider::discover();
        let record = provider
            .adapters
            .iter()
            .find(|r| {
                r.supports_vulkan_timestamps()
                    && r.vulkan.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                    && r.vulkan.host_memory_heap_bytes > 0
            })
            .unwrap();
        println!("DIAGNOSTIC,bytes,ram_percent,group_size,loops,groups,gpu_ns,tops,cv_percent");
        // Interleave launch shapes and placements; reuse the same actual dataset sizes.
        for bytes in [294912, 2064384, 67092480] {
            for group in [256, 64, 128, 32] {
                for share in [0, 75, 100] {
                    // SAFETY: the harness exclusively owns its Vulkan resources.
                    let harness = unsafe {
                        VulkanComputeProfileHarness::new_inner(
                            record,
                            bytes,
                            share,
                            group,
                            7,
                            Precision::Fp32,
                        )
                    }
                    .unwrap();
                    let size = harness.actual_working_set(bytes);
                    assert_eq!(size, bytes);
                    let loops = if bytes == 294912 {
                        vec![1, 64, 512]
                    } else {
                        vec![64]
                    };
                    for loops in loops {
                        let cold = harness.measure(size, loops, 8).unwrap() / 8.0;
                        let passes = (5e6 / cold).clamp(1.0, 4096.0) as u32;
                        for _ in 0..8 {
                            harness.measure(size, loops, passes).unwrap();
                        }
                        let mut values = Vec::new();
                        for _ in 0..7 {
                            values.push(
                                harness.measure(size, loops, passes).unwrap() / f64::from(passes),
                            );
                        }
                        harness.validate_output(size, loops).unwrap();
                        let stats = crate::analysis::statistics(&values);
                        let tops = harness.operations_per_dispatch(size, loops) as f64
                            / stats.median
                            / 1000.0;
                        println!(
                            "DIAGNOSTIC,{size},{share},{group},{loops},{:?},{:.1},{tops:.4},{:.2}",
                            harness.dispatch_workgroups(size),
                            stats.median,
                            stats.standard_deviation / stats.median * 100.0
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "GPU diagnostic: isolate RAM reads and writes with the same data and arithmetic"]
    fn profile_host_direction_diagnostic() {
        let provider = crate::GpuBandwidthProvider::discover();
        let record = provider
            .adapters
            .iter()
            .find(|r| {
                r.supports_vulkan_timestamps()
                    && r.vulkan.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                    && r.vulkan.host_memory_heap_bytes > 0
            })
            .unwrap();
        println!("DIRECTION,bytes,host_mask,group,gpu_ns,tops,cv_percent");
        for bytes in [294912, 2064384, 67092480] {
            for group in [32, 256] {
                for mask in [7, 3, 4, 0] {
                    // SAFETY: exclusively owned resources; mask only chooses physical buffer placement.
                    let harness = unsafe {
                        VulkanComputeProfileHarness::new_inner(
                            record,
                            bytes,
                            100,
                            group,
                            mask,
                            Precision::Fp32,
                        )
                    }
                    .unwrap();
                    let cold = harness.measure(bytes, 64, 8).unwrap() / 8.0;
                    let passes = (5e6 / cold).clamp(1.0, 4096.0) as u32;
                    for _ in 0..24 {
                        harness.measure(bytes, 64, passes).unwrap();
                    }
                    let mut values = Vec::new();
                    for _ in 0..9 {
                        values
                            .push(harness.measure(bytes, 64, passes).unwrap() / f64::from(passes));
                    }
                    harness.validate_output(bytes, 64).unwrap();
                    let stats = crate::analysis::statistics(&values);
                    let tops =
                        harness.operations_per_dispatch(bytes, 64) as f64 / stats.median / 1000.0;
                    println!(
                        "DIRECTION,{bytes},{mask},{group},{:.1},{tops:.4},{:.2}",
                        stats.median,
                        stats.standard_deviation / stats.median * 100.0
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "GPU launch-tuning smoke: validate selected pipelines at the smallest dataset"]
    fn profile_launch_tuning_smoke() {
        let provider = crate::GpuBandwidthProvider::discover();
        let record = provider
            .adapters
            .iter()
            .find(|r| {
                r.supports_vulkan_timestamps()
                    && r.vulkan.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                    && r.vulkan.host_memory_heap_bytes > 0
            })
            .unwrap();
        for share in [0, 75] {
            // SAFETY: exclusively owned resources, standard buffer placement.
            let mut harness = unsafe {
                VulkanComputeProfileHarness::new_inner(
                    record,
                    294912,
                    share,
                    256,
                    7,
                    Precision::Fp32,
                )
            }
            .unwrap();
            let log = harness
                .autotune_workgroups(
                    &[294912],
                    64,
                    &gluj_bench_core::CancellationToken::default(),
                )
                .unwrap();
            assert!([32, 64, 128, 256].contains(&harness.workgroup_size()));
            let ns = harness.measure(294912, 64, 256).unwrap() / 256.0;
            harness.validate_output(294912, 64).unwrap();
            println!(
                "SMALL_TUNED,share={share},group={},gpu_ns={ns:.1},tops={:.3};{log}",
                harness.workgroup_size(),
                harness.operations_per_dispatch(294912, 64) as f64 / ns / 1000.0
            );
        }
    }

    #[test]
    #[ignore = "real GPU fixed-arrival underload/overload and cancellation validation"]
    fn profile_offered_load_smoke() {
        let provider = crate::GpuBandwidthProvider::discover();
        let record = provider
            .adapters
            .iter()
            .find(|r| {
                r.supports_vulkan_timestamps()
                    && r.vulkan.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                    && r.vulkan.host_memory_heap_bytes > 0
            })
            .unwrap();
        let bytes = 64 * 1024 * 1024;
        // SAFETY: exclusive diagnostic resources and standard buffer placement.
        let harness = unsafe {
            VulkanComputeProfileHarness::new_inner(record, bytes, 75, 256, 7, Precision::Fp32)
        }
        .unwrap();
        let bytes = harness.actual_working_set(bytes);
        harness.measure(bytes, 64, 1).unwrap();
        {
            let mut queue = harness.offered_queue(bytes, 64).unwrap();
            let token = gluj_bench_core::CancellationToken::default();
            let low = queue.observe(20.0, 0.2, &token).unwrap();
            let high = queue.observe(10000.0, 0.2, &token).unwrap();
            for (name, sample) in [("low", &low), ("overload", &high)] {
                assert!(sample.completed <= sample.submitted && sample.submitted <= sample.offered);
                println!(
                    "OFFERED,{name},requested={},submitted={},completed={},average_tops={:.3},completion_ms={:.3},oldest_pending_ms={:.3}",
                    sample.offered,
                    sample.submitted,
                    sample.completed,
                    harness.operations_per_dispatch(bytes, 64) as f64 * sample.completed as f64
                        / sample.window_ns
                        / 1000.0,
                    sample.completion_ns / sample.completed.max(1) as f64 / 1e6,
                    sample.oldest_pending_ns / 1e6
                );
            }
            assert_eq!(low.completed, low.offered);
            assert!(high.completed > 0 && high.completed < high.offered / 2);
            assert!(high.oldest_pending_ns > 50e6);
            assert!(
                high.completion_ns / high.completed as f64
                    > low.completion_ns / low.completed as f64
            );
            token.cancel();
            assert_eq!(
                queue.observe(10000.0, 0.2, &token).err().unwrap().code,
                "cancelled"
            );
        }
        harness.validate_output(bytes, 64).unwrap();
    }

    #[test]
    fn profile_shader_is_spirv() {
        assert_eq!(shader_words().unwrap()[0], 0x0723_0203);
        // Vulkan 1.1-compatible specialization uses WorkgroupSize, not LocalSizeId
        // (which would require maintenance4 to be enabled on the logical device).
        let words = shader_words().unwrap();
        let mut offset = 5;
        let mut size_id = false;
        let mut builtin_size = false;
        while offset < words.len() {
            let count = (words[offset] >> 16) as usize;
            let opcode = words[offset] & 0xffff;
            assert_ne!(
                opcode, 331,
                "LocalSizeId needs an additional enabled device feature"
            );
            if opcode == 71 && count >= 4 {
                size_id |= words[offset + 2] == 1 && words[offset + 3] == 0;
                builtin_size |= words[offset + 2] == 11 && words[offset + 3] == 25;
            }
            offset += count;
        }
        assert!(size_id && builtin_size);
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
        // Every cached launch variant must fit the dispatch limit, including 32 threads.
        let limit = chunk_element_limit(65535 * 32 / WORKGROUP_SIZE as u32, 256);
        let elements = (5_u64 * 1024 * 1024 * 1024 / BYTES_PER_ELEMENT / WORKGROUP_SIZE
            * WORKGROUP_SIZE) as u32;
        let chunks = dispatch_chunks(elements, limit);
        assert!(chunks.len() > 1);
        let mut covered = 0;
        for (offset, count) in chunks {
            assert_eq!(offset, covered);
            assert_eq!(offset as u64 * 16 % 256, 0);
            assert_eq!(count as u64 % WORKGROUP_SIZE, 0);
            for group in [32, 64, 128, 256] {
                assert!((count as u64).div_ceil(group) <= 65535);
            }
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
    #[test]
    fn access_mapping_is_a_bijection_and_lane_scattering_changes_lane_locality() {
        for blocks in [1, 2, 3, 7, 17, 64, 127, 256] {
            for mode in 0..4 {
                for seed in [0, 1, 42, u64::MAX] {
                    let params = access_parameters(16, blocks * 256, (mode, seed, 19), 3);
                    let p = params
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .copied()
                        .map(u32::from_le_bytes)
                        .collect::<Vec<_>>();
                    let mut seen = std::collections::BTreeSet::new();
                    for index in 0..blocks * 256 {
                        let mut block = index / 256;
                        let mut lane = index % 256;
                        if mode == 1 || mode == 3 {
                            block = (block * p[4] + p[5]) % blocks;
                        }
                        if mode == 2 {
                            block = (block + p[5]) % blocks;
                        }
                        if mode == 3 {
                            lane = lane.wrapping_mul(p[6]).wrapping_add(p[7]) & 255;
                        }
                        assert!(seen.insert(block * 256 + lane));
                    }
                    assert_eq!(seen.len(), (blocks * 256) as usize);
                }
            }
        }
        let params = access_parameters(1, u32::MAX / 256 * 256, (1, u64::MAX, 0), 0);
        let p = params
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect::<Vec<_>>();
        assert!((u32::MAX as u64 / 256 - 1) * p[4] as u64 + p[5] as u64 <= u32::MAX as u64);
    }
    #[test]
    #[ignore = "executes all supported buffered precisions, access modes and FP32 placements on the current GPU"]
    fn buffered_precision_access_output_smoke() {
        let provider = crate::GpuBandwidthProvider::discover();
        let record = provider
            .adapters
            .iter()
            .find(|r| r.supports_vulkan_timestamps())
            .unwrap();
        for precision in [Precision::Fp16, Precision::Fp32, Precision::Fp64] {
            if precision == Precision::Fp16
                && (!record.vulkan.shader_float16 || !record.vulkan.storage_buffer_16bit_access)
                || precision == Precision::Fp64 && !record.vulkan.shader_float64
            {
                continue;
            }
            let shares = if precision == Precision::Fp32
                && record.vulkan.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                && record.vulkan.host_memory_heap_bytes > 0
            {
                vec![0, 50, 75, 100]
            } else {
                vec![0]
            };
            for share in shares {
                let mut harness =
                    VulkanComputeProfileHarness::new_precision(record, 983040, precision, share)
                        .unwrap();
                for group in [32, 256] {
                    harness.select_workgroup_size(group).unwrap();
                    for mode in 0..4 {
                        for loops in [1, 16, 64, 1024] {
                            harness.configure_access(mode, 0x123456789abcdef, 17);
                            let bytes = harness.actual_working_set(983040);
                            assert!(harness.measure(bytes, loops, 3).unwrap() > 0.);
                            harness.validate_output(bytes, loops).unwrap();
                        }
                    }
                }
                println!(
                    "validated {precision:?}, {share}% host, four access modes, four loop counts, two launch shapes"
                );
            }
        }
    }
}
