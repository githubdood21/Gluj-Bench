mod analysis;
mod compute;
mod cooperative_matrix;
mod scaling;
mod vulkan;
mod vulkan_bandwidth;
mod vulkan_compute_profile;
mod vulkan_vector;

pub use analysis::{EffectiveCacheTier, SweepPoint, detect_effective_cache_tiers};

use analysis::{
    CACHE_SWEEP_MAX, MIB, cache_working_set, coefficient_of_variation, statistics, sweep_sizes,
    vram_working_set,
};
use ash::vk;
use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkProvider,
    BenchmarkResult, CancellationToken, DeviceCategory, DeviceDescriptor, Metric, ProgressCallback,
    ProgressUpdate,
};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

const CACHE_ID: &str = "gpu.bandwidth.cache";
const VRAM_ID: &str = "gpu.bandwidth.vram";
const HOST_LINK_ID: &str = "gpu.bandwidth.host_link";
const WORKGROUP_SIZE: u64 = 256;
const GPU_PRECONDITION_MS: f64 = 750.0;
const HOST_STAGING_BUFFER_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelOperation {
    Read,
    Write,
    Copy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelFlavor {
    Cache,
    Stream,
}

#[derive(Debug, Clone, Copy)]
struct DispatchMeasurement {
    size: u64,
    dispatch_bytes: u64,
    operation: KernelOperation,
    flavor: KernelFlavor,
    precondition: bool,
    sample_count: u32,
    target_per_sample: Duration,
}

impl KernelOperation {
    fn accesses_per_invocation(self, flavor: KernelFlavor) -> u64 {
        match (self, flavor) {
            (Self::Read, KernelFlavor::Cache) => 4,
            (Self::Write | Self::Copy, KernelFlavor::Cache) => 1,
            (Self::Read, KernelFlavor::Stream) => 64,
            (Self::Write | Self::Copy, KernelFlavor::Stream) => 16,
        }
    }

    fn reported_byte_multiplier(self) -> u64 {
        if matches!(self, Self::Copy) { 2 } else { 1 }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Copy => "copy",
        }
    }
}

struct AdapterRecord {
    id: String,
    vulkan: vulkan::VulkanAdapterInfo,
    cooperative: cooperative_matrix::CooperativeSupport,
}

impl AdapterRecord {
    fn supports_vulkan_timestamps(&self) -> bool {
        self.vulkan.timestamp_queries()
    }
}

pub struct GpuBandwidthProvider {
    adapters: Vec<AdapterRecord>,
    devices: Vec<DeviceDescriptor>,
    inferred_tiers: BTreeMap<String, Vec<EffectiveCacheTier>>,
    sweep_points: BTreeMap<String, Vec<SweepPoint>>,
}

impl GpuBandwidthProvider {
    pub fn discover() -> Self {
        let (vulkan_adapters, vulkan_problem) = match vulkan::discover() {
            Ok(adapters) => (adapters, None),
            Err(problem) => (Vec::new(), Some(problem)),
        };
        let mut records = Vec::new();
        let mut devices = Vec::new();
        for vulkan in vulkan_adapters {
            let id = vulkan.id.clone();
            let cooperative = vulkan.cooperative.clone();
            let timestamp_queries = vulkan.timestamp_queries();
            let mut properties = BTreeMap::new();
            properties.insert("backend".into(), "Vulkan".into());
            properties.insert("execution_backend".into(), "raw-vulkan".into());
            properties.insert("device_type".into(), vulkan.device_type_label().into());
            properties.insert("vulkan_device_uuid".into(), vulkan.device_uuid.clone());
            properties.insert(
                "vulkan_vendor_id".into(),
                format!("0x{:04x}", vulkan.vendor_id),
            );
            properties.insert(
                "vulkan_device_id".into(),
                format!("0x{:04x}", vulkan.device_id),
            );
            properties.insert(
                "vulkan_api_version".into(),
                vulkan::format_api_version(vulkan.api_version),
            );
            properties.insert(
                "vulkan_driver_version".into(),
                vulkan.driver_version.to_string(),
            );
            properties.insert(
                "vulkan_compute_queue_family".into(),
                vulkan.compute_queue_family.to_string(),
            );
            properties.insert(
                "vulkan_timestamp_valid_bits".into(),
                vulkan.timestamp_valid_bits.to_string(),
            );
            properties.insert(
                "vulkan_timestamp_period_ns".into(),
                vulkan.timestamp_period_ns.to_string(),
            );
            properties.insert(
                "vulkan_subgroup_size".into(),
                vulkan.subgroup_size.to_string(),
            );
            properties.insert("timestamp_queries".into(), timestamp_queries.to_string());
            properties.insert("shader_f16".into(), vulkan.shader_float16.to_string());
            properties.insert(
                "storage_buffer_16bit_access".into(),
                vulkan.storage_buffer_16bit_access.to_string(),
            );
            properties.insert("shader_f64".into(), vulkan.shader_float64.to_string());
            properties.insert("shader_int8".into(), vulkan.shader_int8.to_string());
            properties.insert(
                "cooperative_matrix_fp16".into(),
                cooperative.fp16.is_some().to_string(),
            );
            properties.insert(
                "cooperative_matrix_int8".into(),
                cooperative.int8.is_some().to_string(),
            );
            if let Some(shape) = cooperative.fp16 {
                properties.insert(
                    "cooperative_matrix_fp16_shape".into(),
                    format!("{}x{}x{}", shape.m, shape.n, shape.k),
                );
            }
            if let Some(shape) = cooperative.int8 {
                properties.insert(
                    "cooperative_matrix_int8_shape".into(),
                    format!("{}x{}x{}", shape.m, shape.n, shape.k),
                );
            }
            properties.insert(
                "vulkan_device_extension_count".into(),
                vulkan.extensions.len().to_string(),
            );
            properties.insert(
                "max_buffer_size".into(),
                vulkan.max_storage_buffer_range.to_string(),
            );
            properties.insert(
                "max_storage_buffer_binding_size".into(),
                vulkan.max_storage_buffer_range.to_string(),
            );
            properties.insert(
                "device_local_memory_bytes".into(),
                vulkan.device_local_memory_bytes.to_string(),
            );
            devices.push(DeviceDescriptor {
                id: id.clone(),
                name: vulkan.name.clone(),
                category: DeviceCategory::Gpu,
                available: true,
                status: if timestamp_queries {
                    "Raw Vulkan compute device ready; numeric and matrix workloads are capability-gated.".into()
                } else {
                    "Vulkan device discovered, but GPU-local timestamp queries are unavailable.".into()
                },
                properties,
                caches: Vec::new(),
            });
            records.push(AdapterRecord {
                id,
                vulkan,
                cooperative,
            });
        }
        if devices.is_empty() {
            devices.push(DeviceDescriptor {
                id: "gpu:vulkan:unavailable".into(),
                name: "Vulkan GPU compute".into(),
                category: DeviceCategory::Gpu,
                available: false,
                status: vulkan_problem.map_or_else(
                    || "No compatible Vulkan compute adapter was discovered.".into(),
                    |problem| format!("Vulkan discovery failed: {}", problem.message),
                ),
                properties: BTreeMap::new(),
                caches: Vec::new(),
            });
        }
        Self {
            adapters: records,
            devices,
            inferred_tiers: BTreeMap::new(),
            sweep_points: BTreeMap::new(),
        }
    }

    fn benchmark_descriptors(&self) -> Vec<BenchmarkDescriptor> {
        let local_devices: Vec<_> = self
            .adapters
            .iter()
            .filter(|record| record.supports_vulkan_timestamps())
            .map(|record| record.id.clone())
            .collect();
        let host_devices: Vec<_> = self
            .adapters
            .iter()
            .map(|record| record.id.clone())
            .collect();
        let local_available = !local_devices.is_empty();
        let host_available = !host_devices.is_empty();
        let mut descriptors = vec![
            descriptor(
                CACHE_ID,
                "Estimated effective GPU cache bandwidth",
                "Empirical hot-working-set cache plateau discovery; some unified-memory GPUs have no isolatable tier",
                local_devices.clone(),
                local_available,
                if local_available {
                    ""
                } else {
                    "timestamp_query_unsupported"
                },
                0,
            ),
            descriptor(
                VRAM_ID,
                "GPU-accessible memory bandwidth",
                "Streaming read, write, and copy bandwidth outside an inferred cache tier when one is available",
                local_devices,
                local_available,
                if local_available {
                    ""
                } else {
                    "timestamp_query_unsupported"
                },
                1,
            ),
            descriptor(
                HOST_LINK_ID,
                "Host-device transfer bandwidth",
                "Bidirectional CPU/RAM to GPU transfer bandwidth",
                host_devices,
                host_available,
                if host_available {
                    ""
                } else {
                    "adapter_not_found"
                },
                2,
            ),
        ];
        descriptors.extend(compute::descriptors(&self.adapters));
        descriptors.extend(cooperative_matrix::descriptors(&self.adapters));
        descriptors
    }

    fn selected_adapter(&self, config: &BenchmarkConfig) -> Result<&AdapterRecord, BenchmarkError> {
        if let Some(id) = config.options.get("device_id") {
            return self
                .adapters
                .iter()
                .find(|record| &record.id == id)
                .ok_or_else(|| {
                    BenchmarkError::new(
                        "adapter_not_found",
                        format!("GPU adapter '{id}' was not found."),
                    )
                });
        }
        self.adapters.first().ok_or_else(|| {
            BenchmarkError::new(
                "adapter_not_found",
                "No compatible Vulkan GPU adapter was discovered.",
            )
        })
    }

    fn discover_tiers(
        &mut self,
        adapter_index: usize,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<Vec<EffectiveCacheTier>, BenchmarkError> {
        let adapter_id = self.adapters[adapter_index].id.clone();
        if let Some(tiers) = self.inferred_tiers.get(&adapter_id) {
            return Ok(tiers.clone());
        }
        let context = vulkan_bandwidth::VulkanBandwidthContext::new(&self.adapters[adapter_index])?;
        let limit = local_buffer_limit(&self.adapters[adapter_index]);
        let sizes = sweep_sizes(limit);
        if sizes.len() < 4 {
            return Err(BenchmarkError::new(
                "insufficient_gpu_memory",
                "The adapter limits are too small for cache-tier discovery.",
            ));
        }
        let mut points = Vec::new();
        for (index, size) in sizes.iter().copied().enumerate() {
            ensure_not_cancelled(cancellation)?;
            progress(ProgressUpdate {
                fraction: index as f64 / sizes.len() as f64 * 0.25,
                phase: "gpu_cache_discovery".into(),
                message: if size < MIB {
                    format!("Probing effective cache behavior at {} KiB", size / 1024)
                } else {
                    format!("Probing effective cache behavior at {} MiB", size / MIB)
                },
            });
            let samples = vulkan_bandwidth::measure_gpu_operation(
                &context,
                DispatchMeasurement {
                    size,
                    dispatch_bytes: size.max(64 * MIB),
                    operation: KernelOperation::Read,
                    flavor: KernelFlavor::Cache,
                    precondition: false,
                    sample_count: 3,
                    target_per_sample: Duration::from_millis(100),
                },
                cancellation,
            )?;
            points.push(SweepPoint {
                size_bytes: size,
                bandwidth_bytes_per_second: statistics(&samples).median,
                coefficient_of_variation: coefficient_of_variation(&samples),
            });
        }
        let tiers = detect_effective_cache_tiers(&points);
        self.sweep_points.insert(adapter_id.clone(), points);
        self.inferred_tiers.insert(adapter_id, tiers.clone());
        Ok(tiers)
    }

    fn run_cache(
        &mut self,
        adapter_index: usize,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let started = Instant::now();
        let tiers = match self.discover_tiers(adapter_index, cancellation, progress) {
            Ok(tiers) => tiers,
            Err(problem) if problem.code == "cancelled" => return Err(problem),
            Err(problem) => {
                progress(done("GPU cache tier is not measurable on this adapter."));
                return Ok(self.cache_discovery_result(
                    adapter_index,
                    started,
                    "unavailable",
                    Some(&problem.code),
                ));
            }
        };
        if tiers.is_empty() {
            progress(done(
                "No isolatable GPU cache tier was detected on this adapter.",
            ));
            return Ok(self.cache_discovery_result(adapter_index, started, "not_detected", None));
        }
        let context = vulkan_bandwidth::VulkanBandwidthContext::new(&self.adapters[adapter_index])?;
        let mut metrics = Vec::new();
        let mut metadata = bandwidth_metadata(&self.adapters[adapter_index], "gpu_timestamp");
        insert_clock_policy(&mut metadata);
        metadata.insert("classification".into(), "inferred".into());
        metadata.insert("physical_cache_identity_confirmed".into(), "false".into());
        metadata.insert("plateau_stability_threshold".into(), "10%".into());
        metadata.insert("sustained_drop_threshold".into(), "20%".into());
        metadata.insert("outer_cliff_threshold".into(), "50% sustained".into());
        metadata.insert(
            "working_set_policy".into(),
            "largest_power_of_two_up_to_40_percent_of_inferred_capacity".into(),
        );
        metadata.insert(
            "copy_byte_definition".into(),
            "read_plus_write_device_memory_traffic".into(),
        );
        metadata.insert("cache_read_vectors_per_invocation".into(), "4".into());
        metadata.insert("cache_write_vectors_per_invocation".into(), "1".into());
        metadata.insert("cache_copy_vectors_per_invocation".into(), "1".into());
        if let Some(points) = self.sweep_points.get(&self.adapters[adapter_index].id) {
            metadata.insert(
                "cache_sweep_points".into(),
                points
                    .iter()
                    .map(|point| {
                        format!(
                            "{}:{:.0}:{:.4}",
                            point.size_bytes,
                            point.bandwidth_bytes_per_second,
                            point.coefficient_of_variation
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        let operation_count = tiers.len() * 3;
        let mut completed = 0usize;
        for (tier_index, tier) in tiers.iter().copied().enumerate() {
            let lower = tier_index.checked_sub(1).map(|index| tiers[index]);
            let Some(working_set) = cache_working_set(tier, lower) else {
                metadata.insert(
                    format!("estimated_effective_{}_status", tier_label(tier_index)),
                    "insufficient_tier_separation".into(),
                );
                continue;
            };
            let label = tier_label(tier_index);
            metadata.insert(
                format!("estimated_effective_{label}_classification"),
                "inferred".into(),
            );
            metadata.insert(
                format!("estimated_effective_{label}_last_cached_bytes"),
                tier.last_cached_size.to_string(),
            );
            metadata.insert(
                format!("estimated_effective_{label}_first_uncached_bytes"),
                tier.first_uncached_size.to_string(),
            );
            metadata.insert(
                format!("estimated_effective_{label}_capacity_estimate_bytes"),
                tier.estimated_capacity.to_string(),
            );
            metadata.insert(
                format!("estimated_effective_{label}_confidence"),
                format!("{:.3}", tier.confidence),
            );
            metadata.insert(
                format!("estimated_effective_{label}_working_set_bytes"),
                working_set.to_string(),
            );
            for operation in [
                KernelOperation::Read,
                KernelOperation::Write,
                KernelOperation::Copy,
            ] {
                ensure_not_cancelled(cancellation)?;
                progress(ProgressUpdate {
                    fraction: 0.25 + completed as f64 / operation_count as f64 * 0.75,
                    phase: "gpu_cache_bandwidth".into(),
                    message: format!(
                        "Measuring estimated effective {label} {} bandwidth",
                        operation.name()
                    ),
                });
                let values = vulkan_bandwidth::measure_gpu_operation(
                    &context,
                    DispatchMeasurement {
                        size: working_set,
                        dispatch_bytes: if matches!(operation, KernelOperation::Read) {
                            working_set.max(64 * MIB)
                        } else {
                            working_set
                        },
                        operation,
                        flavor: KernelFlavor::Cache,
                        precondition: true,
                        sample_count: config.samples.max(1),
                        target_per_sample: per_sample_duration(config),
                    },
                    cancellation,
                )?;
                metrics.push(metric(
                    format!("estimated_effective_{label}.{}", operation.name()),
                    values,
                ));
                completed += 1;
            }
        }
        if metrics.is_empty() {
            return Err(BenchmarkError::new(
                "insufficient_tier_separation",
                "Detected cache plateaus could not be isolated from adjacent tiers.",
            ));
        }
        progress(done("GPU effective cache benchmark completed."));
        Ok(BenchmarkResult {
            benchmark_id: CACHE_ID.into(),
            device_id: self.adapters[adapter_index].id.clone(),
            elapsed_ns: duration_ns(started.elapsed()),
            metrics,
            workload_metadata: metadata,
            device_metadata: bandwidth_device_metadata(&self.adapters[adapter_index]),
        })
    }

    fn run_vram(
        &mut self,
        adapter_index: usize,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let started = Instant::now();
        let (tiers, cache_discovery_status) =
            match self.discover_tiers(adapter_index, cancellation, progress) {
                Ok(tiers) if !tiers.is_empty() => (tiers, "detected"),
                Ok(_) => (Vec::new(), "not_detected"),
                Err(problem) if problem.code == "cancelled" => return Err(problem),
                Err(_) => (Vec::new(), "unavailable"),
            };
        let outer = tiers.last().copied();
        let limit = local_buffer_limit(&self.adapters[adapter_index]);
        let Some(working_set) = vram_working_set(outer, limit) else {
            return Err(BenchmarkError::new(
                "insufficient_cache_separation",
                "A GPU-local working set four times larger than the outer inferred cache tier cannot fit within adapter limits.",
            ));
        };
        let context = vulkan_bandwidth::VulkanBandwidthContext::new(&self.adapters[adapter_index])?;
        let mut metrics = Vec::new();
        for (index, operation) in [
            KernelOperation::Read,
            KernelOperation::Write,
            KernelOperation::Copy,
        ]
        .into_iter()
        .enumerate()
        {
            ensure_not_cancelled(cancellation)?;
            progress(ProgressUpdate {
                fraction: 0.25 + index as f64 / 3.0 * 0.75,
                phase: "gpu_local_memory".into(),
                message: format!("Measuring GPU-local {} bandwidth", operation.name()),
            });
            let values = vulkan_bandwidth::measure_gpu_operation(
                &context,
                DispatchMeasurement {
                    size: working_set,
                    dispatch_bytes: working_set,
                    operation,
                    flavor: KernelFlavor::Stream,
                    precondition: true,
                    sample_count: config.samples.max(1),
                    target_per_sample: per_sample_duration(config),
                },
                cancellation,
            )?;
            metrics.push(metric(operation.name().into(), values));
        }
        let mut metadata = bandwidth_metadata(&self.adapters[adapter_index], "gpu_timestamp");
        insert_clock_policy(&mut metadata);
        metadata.insert("working_set_bytes".into(), working_set.to_string());
        metadata.insert(
            "cache_discovery_status".into(),
            cache_discovery_status.into(),
        );
        metadata.insert(
            "working_set_policy".into(),
            if outer.is_some() {
                "four_times_outer_inferred_cache_tier"
            } else {
                "conservative_direct_memory_fallback_without_cache_tier"
            }
            .into(),
        );
        metadata.insert("cache_separation_multiplier".into(), "4".into());
        metadata.insert("minimum_working_set_bytes".into(), (256 * MIB).to_string());
        metadata.insert(
            "copy_byte_definition".into(),
            "read_plus_write_device_memory_traffic".into(),
        );
        metadata.insert(
            "copy_useful_payload_relation".into(),
            "reported_rate_divided_by_2".into(),
        );
        metadata.insert("stream_read_vectors_per_invocation".into(), "64".into());
        metadata.insert(
            "stream_write_copy_vectors_per_invocation".into(),
            "16".into(),
        );
        metadata.insert(
            "read_checksum_policy".into(),
            "one_store_per_invocation".into(),
        );
        metadata.insert("copy_path".into(), "raw_vulkan_copy_command".into());
        progress(done("GPU-local memory benchmark completed."));
        Ok(BenchmarkResult {
            benchmark_id: VRAM_ID.into(),
            device_id: self.adapters[adapter_index].id.clone(),
            elapsed_ns: duration_ns(started.elapsed()),
            metrics,
            workload_metadata: metadata,
            device_metadata: bandwidth_device_metadata(&self.adapters[adapter_index]),
        })
    }

    fn cache_discovery_result(
        &self,
        adapter_index: usize,
        started: Instant,
        status: &str,
        reason: Option<&str>,
    ) -> BenchmarkResult {
        let mut metadata = bandwidth_metadata(&self.adapters[adapter_index], "gpu_timestamp");
        metadata.insert("cache_discovery_status".into(), status.into());
        metadata.insert("physical_cache_identity_confirmed".into(), "false".into());
        metadata.insert(
            "cache_discovery_note".into(),
            "No isolatable effective cache tier was required or observed; this is expected on some unified-memory and integrated GPUs.".into(),
        );
        if let Some(reason) = reason {
            metadata.insert("cache_discovery_reason".into(), reason.into());
        }
        BenchmarkResult {
            benchmark_id: CACHE_ID.into(),
            device_id: self.adapters[adapter_index].id.clone(),
            elapsed_ns: duration_ns(started.elapsed()),
            metrics: vec![metric("effective_cache_tiers_detected".into(), vec![0.0])],
            workload_metadata: metadata,
            device_metadata: bandwidth_device_metadata(&self.adapters[adapter_index]),
        }
    }

    fn run_host_link(
        &self,
        adapter_index: usize,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let started = Instant::now();
        let record = &self.adapters[adapter_index];
        let context = vulkan_bandwidth::VulkanBandwidthContext::new(record)?;
        let system = sysinfo::System::new_all();
        let Some(size) = host_transfer_size(
            record.vulkan.max_storage_buffer_range,
            system.available_memory(),
            record.vulkan.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU,
        ) else {
            return Err(BenchmarkError::new(
                "insufficient_gpu_memory",
                "A safe host-device transfer buffer could not be allocated within current memory limits.",
            ));
        };
        progress(ProgressUpdate {
            fraction: 0.05,
            phase: "host_to_device".into(),
            message: "Measuring Host to Device transfer bandwidth".into(),
        });
        let host_to_device = vulkan_bandwidth::measure_host_to_device(
            &context,
            size,
            config.samples.max(1),
            per_sample_duration(config),
            cancellation,
        )?;
        progress(ProgressUpdate {
            fraction: 0.55,
            phase: "device_to_host".into(),
            message: "Measuring Device to Host transfer bandwidth".into(),
        });
        let device_to_host = vulkan_bandwidth::measure_device_to_host(
            &context,
            size,
            config.samples.max(1),
            per_sample_duration(config),
            cancellation,
        )?;
        let mut metadata = bandwidth_metadata(record, "cpu_wall_clock_end_to_end");
        metadata.insert("transfer_buffer_bytes".into(), size.to_string());
        metadata.insert(
            "host_buffers_first_touched_before_timing".into(),
            "true".into(),
        );
        metadata.insert("allocation_in_timed_region".into(), "false".into());
        metadata.insert(
            "host_transfer_path".into(),
            "preallocated_staging_ring_native_copy".into(),
        );
        metadata.insert(
            "staging_buffer_count".into(),
            HOST_STAGING_BUFFER_COUNT.to_string(),
        );
        metadata.insert(
            "link_classification".into(),
            link_classification(record.vulkan.device_type).into(),
        );
        metadata.insert("link_classification_is_inference".into(), "true".into());
        progress(done("Host-device transfer benchmark completed."));
        Ok(BenchmarkResult {
            benchmark_id: HOST_LINK_ID.into(),
            device_id: record.id.clone(),
            elapsed_ns: duration_ns(started.elapsed()),
            metrics: vec![
                metric("host_to_device".into(), host_to_device),
                metric("device_to_host".into(), device_to_host),
            ],
            workload_metadata: metadata,
            device_metadata: bandwidth_device_metadata(record),
        })
    }
}

impl Default for GpuBandwidthProvider {
    fn default() -> Self {
        Self::discover()
    }
}

impl BenchmarkProvider for GpuBandwidthProvider {
    fn devices(&self) -> Vec<DeviceDescriptor> {
        self.devices.clone()
    }

    fn benchmarks(&self) -> Vec<BenchmarkDescriptor> {
        self.benchmark_descriptors()
    }

    fn run(
        &mut self,
        benchmark_id: &str,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let selected_id = self.selected_adapter(config)?.id.clone();
        let adapter_index = self
            .adapters
            .iter()
            .position(|record| record.id == selected_id)
            .expect("selected adapter must still exist");
        if benchmark_id != HOST_LINK_ID
            && !self.adapters[adapter_index].supports_vulkan_timestamps()
        {
            return Err(BenchmarkError::new(
                "timestamp_query_unsupported",
                "This GPU does not expose Vulkan timestamps required for GPU-local timing.",
            ));
        }
        match benchmark_id {
            CACHE_ID => self.run_cache(adapter_index, config, cancellation, progress),
            VRAM_ID => self.run_vram(adapter_index, config, cancellation, progress),
            HOST_LINK_ID => self.run_host_link(adapter_index, config, cancellation, progress),
            _ if compute::is_benchmark(benchmark_id) => compute::run(
                &self.adapters[adapter_index],
                benchmark_id,
                config,
                cancellation,
                progress,
            ),
            _ if cooperative_matrix::is_matrix_benchmark(benchmark_id) => cooperative_matrix::run(
                &self.adapters[adapter_index],
                benchmark_id,
                config,
                cancellation,
                progress,
            ),
            _ => Err(BenchmarkError::new(
                "unsupported_benchmark",
                format!("GPU provider does not implement '{benchmark_id}'."),
            )),
        }
    }
}

fn descriptor(
    id: &str,
    name: &str,
    workload: &str,
    supported_device_ids: Vec<String>,
    available: bool,
    unavailable_reason: &str,
    display_order: u32,
) -> BenchmarkDescriptor {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "cache_identity_policy".into(),
        "estimated_effective_inferred".into(),
    );
    BenchmarkDescriptor {
        id: id.into(),
        name: name.into(),
        category: BenchmarkCategory::Gpu,
        workload: workload.into(),
        data_type: "u32 vector storage-buffer traffic".into(),
        unit: "bytes/s".into(),
        supported_device_ids,
        available,
        unavailable_reason: unavailable_reason.into(),
        suite_id: "gpu.bandwidth".into(),
        display_order,
        metadata,
    }
}

fn local_buffer_limit(record: &AdapterRecord) -> u64 {
    record.vulkan.max_storage_buffer_range.min(CACHE_SWEEP_MAX)
}

fn host_transfer_size(adapter_limit: u64, available_memory: u64, integrated: bool) -> Option<u64> {
    let mut limit = adapter_limit.min(64 * MIB);
    if integrated {
        limit = limit.min(available_memory / 16);
    }
    let size = limit / 256 * 256;
    (size >= 4 * MIB).then_some(size)
}

fn per_sample_duration(config: &BenchmarkConfig) -> Duration {
    gluj_bench_core::gpu_burst_duration(Duration::from_millis(
        (config.target_duration_ms / config.samples.max(1) as u64).max(1),
    ))
}

fn metric(name: String, values: Vec<f64>) -> Metric {
    let stats = statistics(&values);
    Metric {
        name,
        value: stats.median,
        unit: "bytes/s".into(),
        statistics: stats,
    }
}

fn tier_label(index: usize) -> &'static str {
    if index == 0 { "l2" } else { "l3" }
}

fn link_classification(device_type: vk::PhysicalDeviceType) -> &'static str {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => "probable_pcie",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "shared_memory_or_uma",
        _ => "host_device_path_unknown",
    }
}

fn common_metadata(record: &AdapterRecord, timing_domain: &str) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    metadata.insert("adapter_name".into(), record.vulkan.name.clone());
    metadata.insert("backend".into(), "Vulkan".into());
    metadata.insert("execution_backend".into(), "raw-vulkan".into());
    metadata.insert(
        "vulkan_device_uuid".into(),
        record.vulkan.device_uuid.clone(),
    );
    metadata.insert(
        "device_type".into(),
        record.vulkan.device_type_label().into(),
    );
    metadata.insert("timing_domain".into(), timing_domain.into());
    metadata.insert("shader_access_width_bytes".into(), "16".into());
    metadata.insert("workgroup_size".into(), WORKGROUP_SIZE.to_string());
    metadata
}

fn bandwidth_metadata(record: &AdapterRecord, timing_domain: &str) -> BTreeMap<String, String> {
    let mut metadata = common_metadata(record, timing_domain);
    metadata.insert("execution_backend".into(), "raw-vulkan".into());
    metadata.insert("shader_format".into(), "embedded_spirv".into());
    metadata.insert("shader_source_language".into(), "GLSL".into());
    metadata.insert(
        "bandwidth_kernel_revision".into(),
        "vulkan-storage-vector-1".into(),
    );
    metadata
}

fn insert_clock_policy(metadata: &mut BTreeMap<String, String>) {
    metadata.insert(
        "clock_policy".into(),
        "vendor_neutral_sustained_preconditioning".into(),
    );
    metadata.insert(
        "gpu_precondition_ms".into(),
        format!("{GPU_PRECONDITION_MS:.0}"),
    );
    metadata.insert("driver_clock_lock".into(), "false".into());
}

fn device_metadata(record: &AdapterRecord) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    metadata.insert("device_id".into(), record.id.clone());
    metadata.insert("name".into(), record.vulkan.name.clone());
    metadata.insert("backend".into(), "Vulkan".into());
    metadata.insert("execution_backend".into(), "raw-vulkan".into());
    metadata.insert(
        "vulkan_device_uuid".into(),
        record.vulkan.device_uuid.clone(),
    );
    metadata.insert(
        "device_type".into(),
        record.vulkan.device_type_label().into(),
    );
    metadata
}

fn bandwidth_device_metadata(record: &AdapterRecord) -> BTreeMap<String, String> {
    let mut metadata = device_metadata(record);
    metadata.insert("execution_backend".into(), "raw-vulkan".into());
    metadata
}

fn ensure_not_cancelled(cancellation: &CancellationToken) -> Result<(), BenchmarkError> {
    if cancellation.is_cancelled() {
        Err(BenchmarkError::new(
            "cancelled",
            "The GPU benchmark was cancelled. In-flight GPU work was allowed to complete.",
        ))
    } else {
        Ok(())
    }
}

fn done(message: &str) -> ProgressUpdate {
    ProgressUpdate {
        fraction: 1.0,
        phase: "complete".into(),
        message: message.into(),
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_are_vendor_neutral_and_ordered() {
        let provider = GpuBandwidthProvider::discover();
        let descriptors = provider.benchmark_descriptors();
        assert_eq!(
            descriptors
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            [
                CACHE_ID,
                VRAM_ID,
                HOST_LINK_ID,
                "gpu.performance.fp16",
                "gpu.performance.fp32",
                "gpu.performance.fp64",
                "gpu.performance.fp32.scaling",
                cooperative_matrix::FP16_MATRIX_ID,
                cooperative_matrix::FP16_MATRIX_SCALING_ID,
                cooperative_matrix::INT8_MATRIX_ID,
                cooperative_matrix::FP8_MATRIX_ID,
                cooperative_matrix::SPARSE_FP16_MATRIX_ID,
                cooperative_matrix::SPARSE_INT8_MATRIX_ID,
                cooperative_matrix::SPARSE_FP8_MATRIX_ID,
            ]
        );
        assert!(
            provider
                .devices
                .iter()
                .all(|device| device.id.starts_with("gpu:vulkan:"))
        );
    }

    #[test]
    fn link_labels_do_not_claim_pcie_for_integrated_devices() {
        assert_eq!(
            link_classification(vk::PhysicalDeviceType::DISCRETE_GPU),
            "probable_pcie"
        );
        assert_eq!(
            link_classification(vk::PhysicalDeviceType::INTEGRATED_GPU),
            "shared_memory_or_uma"
        );
    }

    #[test]
    fn integrated_host_buffers_respect_available_memory() {
        assert_eq!(
            host_transfer_size(128 * MIB, 512 * MIB, true),
            Some(32 * MIB)
        );
        assert_eq!(host_transfer_size(128 * MIB, 32 * MIB, true), None);
        assert_eq!(
            host_transfer_size(128 * MIB, 32 * MIB, false),
            Some(64 * MIB)
        );
    }

    #[test]
    fn gpu_local_copy_reports_total_device_memory_traffic() {
        assert_eq!(KernelOperation::Read.reported_byte_multiplier(), 1);
        assert_eq!(KernelOperation::Write.reported_byte_multiplier(), 1);
        assert_eq!(KernelOperation::Copy.reported_byte_multiplier(), 2);
        assert_eq!(
            KernelOperation::Read.accesses_per_invocation(KernelFlavor::Cache),
            4
        );
        assert_eq!(
            KernelOperation::Copy.accesses_per_invocation(KernelFlavor::Stream),
            16
        );
        assert_eq!(
            KernelOperation::Read.accesses_per_invocation(KernelFlavor::Stream),
            64
        );
    }
}
