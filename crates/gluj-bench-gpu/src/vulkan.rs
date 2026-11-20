use crate::cooperative_matrix::{CooperativeSupport, MatrixShape};
use crate::pcie_info::{PciBusInfo, PcieLinkInfo, WindowsPciIndex};
use ash::{Entry, vk};
use gluj_bench_core::BenchmarkError;
use std::{collections::BTreeSet, ffi::CStr};

#[derive(Debug, Clone)]
pub(super) struct VulkanAdapterInfo {
    pub id: String,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub api_version: u32,
    pub driver_version: u32,
    pub device_uuid: String,
    pub compute_queue_family: u32,
    pub timestamp_valid_bits: u32,
    pub timestamp_period_ns: f32,
    pub subgroup_size: u32,
    pub subgroup_size_control: bool,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub shader_float16: bool,
    pub storage_buffer_16bit_access: bool,
    pub storage_buffer_8bit_access: bool,
    pub shader_int8: bool,
    pub shader_float64: bool,
    pub max_storage_buffer_range: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub device_local_memory_bytes: u64,
    pub host_memory_heap_bytes: u64,
    pub extensions: BTreeSet<String>,
    pub cooperative: CooperativeSupport,
    pub pci_bus_info: Option<PciBusInfo>,
    pub pcie_link: Option<PcieLinkInfo>,
}

impl VulkanAdapterInfo {
    pub fn timestamp_queries(&self) -> bool {
        self.timestamp_valid_bits > 0
    }

    pub fn device_type_label(&self) -> &'static str {
        match self.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "DiscreteGpu",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "IntegratedGpu",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "VirtualGpu",
            vk::PhysicalDeviceType::CPU => "Cpu",
            _ => "Other",
        }
    }
}

pub(super) fn discover() -> Result<Vec<VulkanAdapterInfo>, BenchmarkError> {
    // SAFETY: all Vulkan handles created here are destroyed before returning.
    let entry = unsafe { Entry::load() }.map_err(|problem| {
        BenchmarkError::new(
            "vulkan_loader_unavailable",
            format!("Could not load the Vulkan loader: {problem}"),
        )
    })?;
    // SAFETY: this only queries the loaded Vulkan loader.
    let loader_version = unsafe { entry.try_enumerate_instance_version() }
        .map_err(vulkan_error)?
        .unwrap_or(vk::API_VERSION_1_0);
    let app_name = c"Gluj-Bench";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(1)
        .engine_name(app_name)
        .engine_version(1)
        .api_version(loader_version.min(vk::API_VERSION_1_3));
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    // SAFETY: create_info and app_info live for this call and no custom allocator is used.
    let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(|problem| {
        BenchmarkError::new(
            "vulkan_instance_failed",
            format!("Could not create the Vulkan discovery instance: {problem}"),
        )
    })?;

    let result = discover_with_instance(&entry, &instance);
    // SAFETY: no child Vulkan handles survive discovery.
    unsafe { instance.destroy_instance(None) };
    result
}

fn discover_with_instance(
    entry: &Entry,
    instance: &ash::Instance,
) -> Result<Vec<VulkanAdapterInfo>, BenchmarkError> {
    // SAFETY: instance remains live throughout all queries in this function.
    let physical_devices =
        unsafe { instance.enumerate_physical_devices() }.map_err(vulkan_error)?;
    let cooperative_extension = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    let pci_index = WindowsPciIndex::discover();
    let mut adapters = Vec::new();
    for (enumeration_index, physical_device) in physical_devices.into_iter().enumerate() {
        // SAFETY: physical_device belongs to instance.
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        if properties.device_type == vk::PhysicalDeviceType::CPU {
            continue;
        }
        // SAFETY: Vulkan guarantees a NUL-terminated deviceName array.
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut id_properties = vk::PhysicalDeviceIDProperties::default();
        let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut id_properties)
            .push_next(&mut subgroup_properties);
        // SAFETY: the output chain is valid and physical_device belongs to instance.
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
        let device_uuid = hex_bytes(&id_properties.device_uuid);

        // SAFETY: physical_device belongs to instance.
        let extension_properties =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .map_err(vulkan_error)?;
        let extensions = extension_properties
            .iter()
            .map(|property| {
                // SAFETY: Vulkan guarantees a NUL-terminated extensionName array.
                unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<BTreeSet<_>>();

        let has_subgroup_size_control = properties.api_version >= vk::API_VERSION_1_3
            || extensions.contains("VK_EXT_subgroup_size_control");
        let pci_bus_info = if extensions.contains("VK_EXT_pci_bus_info") {
            let mut pci_props = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
            let mut pci_props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut pci_props);
            // SAFETY: the output chain is valid and physical_device belongs to instance.
            unsafe { instance.get_physical_device_properties2(physical_device, &mut pci_props2) };
            Some(PciBusInfo {
                domain: pci_props.pci_domain,
                bus: pci_props.pci_bus,
                device: pci_props.pci_device,
                function: pci_props.pci_function,
            })
        } else {
            None
        };
        let pcie_link = pci_index.lookup(properties.vendor_id, properties.device_id, pci_bus_info);
        let (subgroup_size_control, min_subgroup_size, max_subgroup_size) =
            if has_subgroup_size_control {
                let mut subgroup_control_props =
                    vk::PhysicalDeviceSubgroupSizeControlProperties::default();
                let mut control_props2 =
                    vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_control_props);
                // SAFETY: the output chain is valid and physical_device belongs to instance.
                unsafe {
                    instance.get_physical_device_properties2(physical_device, &mut control_props2)
                };

                let mut subgroup_control_features =
                    vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
                let mut control_features2 = vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut subgroup_control_features);
                // SAFETY: the output chain is valid and physical_device belongs to instance.
                unsafe {
                    instance.get_physical_device_features2(physical_device, &mut control_features2)
                };

                let supported = subgroup_control_features.subgroup_size_control == vk::TRUE
                    && subgroup_control_props
                        .required_subgroup_size_stages
                        .contains(vk::ShaderStageFlags::COMPUTE);
                (
                    supported,
                    subgroup_control_props.min_subgroup_size,
                    subgroup_control_props.max_subgroup_size,
                )
            } else {
                (
                    false,
                    subgroup_properties.subgroup_size,
                    subgroup_properties.subgroup_size,
                )
            };

        let mut float16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut storage_16bit = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut storage_8bit = vk::PhysicalDevice8BitStorageFeatures::default();
        let shader_float64 = {
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut float16_int8)
                .push_next(&mut storage_16bit)
                .push_next(&mut storage_8bit);
            // SAFETY: the output chain is valid and physical_device belongs to instance.
            unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
            features2.features.shader_float64 == vk::TRUE
        };
        let shader_float16 = float16_int8.shader_float16 == vk::TRUE;
        let shader_int8 = float16_int8.shader_int8 == vk::TRUE;
        let storage_buffer_8bit_access = storage_8bit.storage_buffer8_bit_access == vk::TRUE;
        let storage_buffer_16bit_access = storage_16bit.storage_buffer16_bit_access == vk::TRUE;
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let device_local_memory_bytes = device_local_memory_bytes(&memory_properties);

        let host_memory_heap_bytes =
            crate::vulkan_compute_profile::host_memory_type(&memory_properties, u32::MAX)
                .map(|index| {
                    memory_properties.memory_heaps
                        [memory_properties.memory_types[index as usize].heap_index as usize]
                        .size
                })
                .unwrap_or(0);

        // SAFETY: physical_device belongs to instance.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let Some((compute_queue_family, timestamp_valid_bits)) =
            select_compute_queue_family(&queue_families)
        else {
            continue;
        };

        let cooperative = cooperative_support(
            instance,
            &cooperative_extension,
            physical_device,
            &extensions,
        )
        .unwrap_or_else(|problem| CooperativeSupport {
            reason: problem.code,
            ..Default::default()
        });
        let stable_suffix = if id_properties.device_uuid.iter().any(|byte| *byte != 0) {
            device_uuid.clone()
        } else {
            format!(
                "{:04x}:{:04x}:{enumeration_index}",
                properties.vendor_id, properties.device_id
            )
        };
        adapters.push(VulkanAdapterInfo {
            id: format!("gpu:vulkan:{stable_suffix}"),
            name,
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            device_type: properties.device_type,
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            device_uuid,
            compute_queue_family,
            timestamp_valid_bits,
            timestamp_period_ns: properties.limits.timestamp_period,
            subgroup_size: subgroup_properties.subgroup_size,
            subgroup_size_control,
            min_subgroup_size,
            max_subgroup_size,
            shader_float16,
            storage_buffer_16bit_access,
            storage_buffer_8bit_access,
            shader_int8,
            shader_float64,
            max_storage_buffer_range: properties.limits.max_storage_buffer_range as u64,
            min_storage_buffer_offset_alignment: properties
                .limits
                .min_storage_buffer_offset_alignment,
            device_local_memory_bytes,
            host_memory_heap_bytes,
            extensions,
            cooperative,
            pci_bus_info,
            pcie_link,
        });
    }
    adapters.sort_by_key(|adapter| adapter_priority(adapter.device_type));
    Ok(adapters)
}

fn adapter_priority(device_type: vk::PhysicalDeviceType) -> u8 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        _ => 3,
    }
}

fn select_compute_queue_family(queue_families: &[vk::QueueFamilyProperties]) -> Option<(u32, u32)> {
    queue_families
        .iter()
        .enumerate()
        .filter(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .max_by_key(|(_, family)| {
            (
                // Prefer any compute-capable queue with timestamps over a non-timestamped async queue.
                u32::from(family.timestamp_valid_bits > 0),
                u32::from(!family.queue_flags.contains(vk::QueueFlags::GRAPHICS)),
                family.timestamp_valid_bits,
            )
        })
        .map(|(index, family)| (index as u32, family.timestamp_valid_bits))
}

fn device_local_memory_bytes(properties: &vk::PhysicalDeviceMemoryProperties) -> u64 {
    let heap_count = (properties.memory_heap_count as usize).min(properties.memory_heaps.len());
    let type_count = (properties.memory_type_count as usize).min(properties.memory_types.len());
    let heaps = &properties.memory_heaps[..heap_count];
    let types = &properties.memory_types[..type_count];

    let has_dedicated_type = |heap_index: usize| {
        types.iter().any(|ty| {
            ty.heap_index as usize == heap_index
                && ty
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                && !ty
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        })
    };

    // Discrete GPUs (notably NVIDIA without ReBAR) often expose a small (~214-256 MiB)
    // HOST_VISIBLE PCIe BAR aperture heap also flagged DEVICE_LOCAL alongside the full
    // dedicated VRAM heap. Exclude BAR-only DEVICE_LOCAL heaps when a non-host-visible
    // DEVICE_LOCAL heap exists so physical VRAM is not double-counted.
    let any_dedicated_heap = (0..heap_count).any(|index| {
        heaps[index]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            && has_dedicated_type(index)
    });

    heaps
        .iter()
        .enumerate()
        .filter(|(index, heap)| {
            heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                && (!any_dedicated_heap || has_dedicated_type(*index))
        })
        .map(|(_, heap)| heap.size)
        .sum()
}

fn cooperative_support(
    instance: &ash::Instance,
    extension: &ash::khr::cooperative_matrix::Instance,
    physical_device: vk::PhysicalDevice,
    extensions: &BTreeSet<String>,
) -> Result<CooperativeSupport, BenchmarkError> {
    let extension_name = ash::khr::cooperative_matrix::NAME
        .to_str()
        .unwrap_or_default();
    if !extensions.contains(extension_name) {
        return Ok(CooperativeSupport {
            reason: "cooperative_matrix_extension_unsupported".into(),
            ..Default::default()
        });
    }
    let mut cooperative_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut cooperative_features);
    // SAFETY: the feature chain is valid and physical_device belongs to instance.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
    if cooperative_features.cooperative_matrix == vk::FALSE {
        return Ok(CooperativeSupport {
            reason: "cooperative_matrix_feature_unsupported".into(),
            ..Default::default()
        });
    }
    // SAFETY: extension presence was checked and physical_device is live.
    let configurations =
        unsafe { extension.get_physical_device_cooperative_matrix_properties(physical_device) }
            .map_err(vulkan_error)?;
    let fp16 = configurations
        .iter()
        .filter(|property| {
            property.scope == vk::ScopeKHR::SUBGROUP
                && property.a_type == vk::ComponentTypeKHR::FLOAT16
                && property.b_type == vk::ComponentTypeKHR::FLOAT16
                && property.c_type == vk::ComponentTypeKHR::FLOAT32
                && property.saturating_accumulation == vk::FALSE
                && property.c_type == property.result_type
        })
        .map(matrix_shape)
        .collect::<Vec<_>>();
    let fp16_shapes = fp16;
    let fp16 = fp16_shapes
        .iter()
        .copied()
        .max_by_key(|s| u64::from(s.m) * u64::from(s.n) * u64::from(s.k));
    let int8 = configurations
        .iter()
        .filter(|property| {
            property.scope == vk::ScopeKHR::SUBGROUP
                && property.saturating_accumulation == vk::FALSE
                && matches!(
                    property.a_type,
                    vk::ComponentTypeKHR::SINT8 | vk::ComponentTypeKHR::UINT8
                )
                && property.a_type == property.b_type
                && matches!(
                    (property.a_type, property.c_type),
                    (vk::ComponentTypeKHR::SINT8, vk::ComponentTypeKHR::SINT32)
                        | (vk::ComponentTypeKHR::UINT8, vk::ComponentTypeKHR::UINT32)
                )
                && property.result_type == property.c_type
        })
        .map(matrix_shape)
        .collect::<Vec<_>>();
    let int8_shapes = int8;
    let int8 = int8_shapes
        .iter()
        .copied()
        .max_by_key(|s| u64::from(s.m) * u64::from(s.n) * u64::from(s.k));
    let reason = if fp16.is_none() && int8.is_none() {
        "cooperative_matrix_formats_unsupported".into()
    } else {
        String::new()
    };
    Ok(CooperativeSupport {
        fp16,
        int8,
        reason,
        fp16_shapes,
        int8_shapes,
    })
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

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn format_api_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

fn vulkan_error(problem: vk::Result) -> BenchmarkError {
    BenchmarkError::new(
        "vulkan_capability_query_failed",
        format!("A Vulkan capability query failed: {problem}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_encoding_is_stable_and_unambiguous() {
        assert_eq!(hex_bytes(&[0, 1, 0xfe, 0xff]), "0001feff");
    }

    #[test]
    fn api_versions_are_human_readable() {
        assert_eq!(
            format_api_version(vk::make_api_version(0, 1, 3, 301)),
            "1.3.301"
        );
    }

    #[test]
    fn compute_queue_prefers_timestamped_queue_over_non_timestamped_async_queue() {
        let families = [
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS
                    | vk::QueueFlags::COMPUTE
                    | vk::QueueFlags::TRANSFER,
                queue_count: 1,
                timestamp_valid_bits: 64,
                min_image_transfer_granularity: vk::Extent3D::default(),
            },
            vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                queue_count: 1,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: vk::Extent3D::default(),
            },
        ];
        assert_eq!(select_compute_queue_family(&families), Some((0, 64)));

        let both_timestamped = [
            families[0],
            vk::QueueFamilyProperties {
                timestamp_valid_bits: 36,
                ..families[1]
            },
        ];
        assert_eq!(
            select_compute_queue_family(&both_timestamped),
            Some((1, 36))
        );
    }

    #[test]
    fn device_local_memory_excludes_bar_only_heap_when_dedicated_vram_exists() {
        let mut nvidia_non_rebar = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: 3,
            memory_type_count: 3,
            ..Default::default()
        };
        nvidia_non_rebar.memory_heaps[0] = vk::MemoryHeap {
            size: 8 << 30,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        nvidia_non_rebar.memory_heaps[1] = vk::MemoryHeap {
            size: 32 << 30,
            flags: vk::MemoryHeapFlags::empty(),
        };
        nvidia_non_rebar.memory_heaps[2] = vk::MemoryHeap {
            size: 214 << 20,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        nvidia_non_rebar.memory_types[0] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            heap_index: 0,
        };
        nvidia_non_rebar.memory_types[1] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 1,
        };
        nvidia_non_rebar.memory_types[2] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 2,
        };
        assert_eq!(device_local_memory_bytes(&nvidia_non_rebar), 8 << 30);

        let mut intel_igpu = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: 1,
            memory_type_count: 1,
            ..Default::default()
        };
        intel_igpu.memory_heaps[0] = vk::MemoryHeap {
            size: 16 << 30,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        intel_igpu.memory_types[0] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 0,
        };
        assert_eq!(device_local_memory_bytes(&intel_igpu), 16 << 30);
    }

    #[test]
    fn discrete_gpu_prioritized_ahead_of_integrated_gpu() {
        assert!(
            adapter_priority(vk::PhysicalDeviceType::DISCRETE_GPU)
                < adapter_priority(vk::PhysicalDeviceType::INTEGRATED_GPU)
        );
    }
}
