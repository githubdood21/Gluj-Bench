use crate::cooperative_matrix::{CooperativeSupport, MatrixShape};
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
    pub shader_float16: bool,
    pub shader_int8: bool,
    pub shader_float64: bool,
    pub max_storage_buffer_range: u64,
    pub extensions: BTreeSet<String>,
    pub cooperative: CooperativeSupport,
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

        let mut float16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let shader_float64 = {
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut float16_int8);
            // SAFETY: the output chain is valid and physical_device belongs to instance.
            unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
            features2.features.shader_float64 == vk::TRUE
        };
        let shader_float16 = float16_int8.shader_float16 == vk::TRUE;
        let shader_int8 = float16_int8.shader_int8 == vk::TRUE;

        // SAFETY: physical_device belongs to instance.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let Some((compute_queue_family, timestamp_valid_bits)) = queue_families
            .iter()
            .enumerate()
            .filter(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .max_by_key(|(_, family)| {
                (
                    u32::from(!family.queue_flags.contains(vk::QueueFlags::GRAPHICS)),
                    family.timestamp_valid_bits,
                )
            })
            .map(|(index, family)| (index as u32, family.timestamp_valid_bits))
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
            shader_float16,
            shader_int8,
            shader_float64,
            max_storage_buffer_range: properties.limits.max_storage_buffer_range as u64,
            extensions,
            cooperative,
        });
    }
    Ok(adapters)
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
    Ok(CooperativeSupport { fp16, int8, reason })
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
}
