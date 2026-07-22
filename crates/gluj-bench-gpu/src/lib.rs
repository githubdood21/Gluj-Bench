mod analysis;
mod compute;
mod compute_shader;
mod cooperative_matrix;
mod shader;

pub use analysis::{EffectiveCacheTier, SweepPoint, detect_effective_cache_tiers};

use analysis::{
    CACHE_SWEEP_MAX, MIB, cache_working_set, coefficient_of_variation, statistics, sweep_sizes,
    vram_working_set,
};
use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkProvider,
    BenchmarkResult, CancellationToken, DeviceCategory, DeviceDescriptor, Metric, ProgressCallback,
    ProgressUpdate,
};
use std::{
    collections::BTreeMap,
    sync::mpsc,
    time::{Duration, Instant},
};
use wgpu::util::DeviceExt;

const CACHE_ID: &str = "gpu.bandwidth.cache";
const VRAM_ID: &str = "gpu.bandwidth.vram";
const HOST_LINK_ID: &str = "gpu.bandwidth.host_link";
const WORKGROUP_SIZE: u64 = 256;
const GPU_PRECONDITION_MS: f64 = 750.0;
const HOST_STAGING_BUFFER_COUNT: usize = 3;

#[derive(Debug, Clone, Copy)]
enum KernelOperation {
    Read,
    Write,
    Copy,
}

#[derive(Debug, Clone, Copy)]
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
    fn entry_point(self, flavor: KernelFlavor) -> &'static str {
        match (self, flavor) {
            (Self::Read, KernelFlavor::Cache) => "read_cache",
            (Self::Write, KernelFlavor::Cache) => "write_cache",
            (Self::Copy, KernelFlavor::Cache) => "copy_cache",
            (Self::Read, KernelFlavor::Stream) => "read_stream",
            (Self::Write, KernelFlavor::Stream) => "write_stream",
            (Self::Copy, KernelFlavor::Stream) => "copy_stream",
        }
    }

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
    adapter: wgpu::Adapter,
    info: wgpu::AdapterInfo,
    features: wgpu::Features,
    limits: wgpu::Limits,
    cooperative: cooperative_matrix::CooperativeSupport,
    cooperative_matrix_properties: Vec<wgpu::CooperativeMatrixProperties>,
}

pub struct GpuBandwidthProvider {
    adapters: Vec<AdapterRecord>,
    devices: Vec<DeviceDescriptor>,
    inferred_tiers: BTreeMap<String, Vec<EffectiveCacheTier>>,
    sweep_points: BTreeMap<String, Vec<SweepPoint>>,
}

impl GpuBandwidthProvider {
    pub fn discover() -> Self {
        let mut cooperative_support = cooperative_matrix::discover();
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let enumerated = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let mut adapters: Vec<wgpu::Adapter> = Vec::new();
        for adapter in enumerated {
            let info = adapter.get_info();
            if info.device_type == wgpu::DeviceType::Cpu {
                continue;
            }
            if let Some(existing_index) = adapters
                .iter()
                .position(|existing| existing.get_info().name.eq_ignore_ascii_case(&info.name))
            {
                let existing = &adapters[existing_index];
                let candidate_score = adapter_score(&adapter);
                if candidate_score > adapter_score(existing) {
                    adapters[existing_index] = adapter;
                }
            } else {
                adapters.push(adapter);
            }
        }
        let mut records = Vec::new();
        let mut devices = Vec::new();
        for (index, adapter) in adapters.into_iter().enumerate() {
            let info = adapter.get_info();
            let features = adapter.features();
            let limits = adapter.limits();
            let cooperative_matrix_properties = adapter.cooperative_matrix_properties();
            let id = format!("gpu:wgpu:{index}");
            let cooperative = cooperative_support
                .remove(&info.name.to_ascii_lowercase())
                .unwrap_or_else(|| cooperative_matrix::CooperativeSupport {
                    reason: "vulkan_adapter_match_unavailable".into(),
                    ..Default::default()
                });
            let timestamp_queries = features.contains(wgpu::Features::TIMESTAMP_QUERY);
            let mut properties = BTreeMap::new();
            properties.insert("backend".into(), format!("{:?}", info.backend));
            properties.insert("device_type".into(), format!("{:?}", info.device_type));
            properties.insert("timestamp_queries".into(), timestamp_queries.to_string());
            properties.insert(
                "timestamp_queries_inside_encoders".into(),
                features
                    .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
                    .to_string(),
            );
            properties.insert(
                "shader_f16".into(),
                features.contains(wgpu::Features::SHADER_F16).to_string(),
            );
            properties.insert(
                "shader_f64".into(),
                features.contains(wgpu::Features::SHADER_F64).to_string(),
            );
            properties.insert(
                "cooperative_matrix_fp16".into(),
                cooperative.fp16.is_some().to_string(),
            );
            properties.insert(
                "cooperative_matrix_int8".into(),
                cooperative.int8.is_some().to_string(),
            );
            properties.insert(
                "wgpu_cooperative_matrix_configurations".into(),
                cooperative_matrix_properties.len().to_string(),
            );
            properties.insert("max_buffer_size".into(), limits.max_buffer_size.to_string());
            properties.insert(
                "max_storage_buffer_binding_size".into(),
                limits.max_storage_buffer_binding_size.to_string(),
            );
            devices.push(DeviceDescriptor {
                id: id.clone(),
                name: info.name.clone(),
                category: DeviceCategory::Gpu,
                available: true,
                status: if timestamp_queries {
                    "GPU-local bandwidth, compute, and host-device tests are available through wgpu; optional numeric formats remain capability-gated.".into()
                } else {
                    "Host-device bandwidth is available; GPU-local bandwidth and compute timing are unavailable because timestamp queries are unsupported.".into()
                },
                properties,
                caches: Vec::new(),
            });
            records.push(AdapterRecord {
                id,
                adapter,
                info,
                features,
                limits,
                cooperative,
                cooperative_matrix_properties,
            });
        }
        if devices.is_empty() {
            devices.push(DeviceDescriptor {
                id: "gpu:wgpu:unavailable".into(),
                name: "GPU compute".into(),
                category: DeviceCategory::Gpu,
                available: false,
                status: "No graphics adapter was discovered through wgpu.".into(),
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
            .filter(|record| record.features.contains(wgpu::Features::TIMESTAMP_QUERY))
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
                "Estimated effective L2 / L3 cache bandwidth",
                "Empirical hot-working-set cache plateau discovery and bandwidth",
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
                "GPU-local memory bandwidth",
                "Cache-separated GPU-local read, write, and copy bandwidth",
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
                "No GPU adapter was discovered through wgpu.",
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
        let context = GpuContext::request(&self.adapters[adapter_index], true)?;
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
            let samples = measure_gpu_operation_with_dispatch(
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
        let tiers = self.discover_tiers(adapter_index, cancellation, progress)?;
        if tiers.is_empty() {
            return Err(BenchmarkError::new(
                "cache_tier_not_detected",
                "No stable effective cache bandwidth plateau was detected on this adapter.",
            ));
        }
        let context = GpuContext::request(&self.adapters[adapter_index], true)?;
        let mut metrics = Vec::new();
        let mut metadata = common_metadata(&self.adapters[adapter_index], "gpu_timestamp");
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
                let values = measure_gpu_operation_with_dispatch(
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
            device_metadata: device_metadata(&self.adapters[adapter_index]),
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
        let tiers = self.discover_tiers(adapter_index, cancellation, progress)?;
        let outer = tiers.last().copied();
        let limit = local_buffer_limit(&self.adapters[adapter_index]);
        let Some(working_set) = vram_working_set(outer, limit) else {
            return Err(BenchmarkError::new(
                "insufficient_cache_separation",
                "A GPU-local working set four times larger than the outer inferred cache tier cannot fit within adapter limits.",
            ));
        };
        let context = GpuContext::request(&self.adapters[adapter_index], true)?;
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
            let values = measure_gpu_operation(
                &context,
                working_set,
                operation,
                config.samples.max(1),
                per_sample_duration(config),
                cancellation,
            )?;
            metrics.push(metric(operation.name().into(), values));
        }
        let mut metadata = common_metadata(&self.adapters[adapter_index], "gpu_timestamp");
        insert_clock_policy(&mut metadata);
        metadata.insert("working_set_bytes".into(), working_set.to_string());
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
        metadata.insert(
            "copy_path".into(),
            if context
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
            {
                "native_copy_command"
            } else {
                "wgsl_compute_fallback"
            }
            .into(),
        );
        progress(done("GPU-local memory benchmark completed."));
        Ok(BenchmarkResult {
            benchmark_id: VRAM_ID.into(),
            device_id: self.adapters[adapter_index].id.clone(),
            elapsed_ns: duration_ns(started.elapsed()),
            metrics,
            workload_metadata: metadata,
            device_metadata: device_metadata(&self.adapters[adapter_index]),
        })
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
        let context = GpuContext::request(record, false)?;
        let system = sysinfo::System::new_all();
        let Some(size) = host_transfer_size(
            record.limits.max_buffer_size,
            system.available_memory(),
            record.info.device_type == wgpu::DeviceType::IntegratedGpu,
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
        let host_to_device = measure_host_to_device(
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
        let device_to_host = measure_device_to_host(
            &context,
            size,
            config.samples.max(1),
            per_sample_duration(config),
            cancellation,
        )?;
        let mut metadata = common_metadata(record, "cpu_wall_clock_end_to_end");
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
            link_classification(record.info.device_type).into(),
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
            device_metadata: device_metadata(record),
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
        if (matches!(benchmark_id, CACHE_ID | VRAM_ID)
            || compute::kind(benchmark_id).is_some()
            || cooperative_matrix::is_matrix_benchmark(benchmark_id))
            && !self.adapters[adapter_index]
                .features
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return Err(BenchmarkError::new(
                "timestamp_query_unsupported",
                "This GPU does not expose timestamp queries required for GPU-local timing.",
            ));
        }
        match benchmark_id {
            CACHE_ID => self.run_cache(adapter_index, config, cancellation, progress),
            VRAM_ID => self.run_vram(adapter_index, config, cancellation, progress),
            HOST_LINK_ID => self.run_host_link(adapter_index, config, cancellation, progress),
            _ if compute::kind(benchmark_id).is_some() => compute::run(
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

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    fn request(record: &AdapterRecord, timestamps: bool) -> Result<Self, BenchmarkError> {
        Self::request_with_features(record, timestamps, wgpu::Features::empty())
    }

    fn request_with_features(
        record: &AdapterRecord,
        timestamps: bool,
        extra_features: wgpu::Features,
    ) -> Result<Self, BenchmarkError> {
        if !record.features.contains(extra_features) {
            return Err(BenchmarkError::new(
                "shader_feature_unsupported",
                "The selected GPU does not expose the shader feature required by this benchmark.",
            ));
        }
        let required_features = if timestamps {
            let mut features = wgpu::Features::TIMESTAMP_QUERY;
            if record
                .features
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
            {
                features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
            }
            features | extra_features
        } else {
            extra_features
        };
        let experimental_features =
            if extra_features.intersects(wgpu::Features::all_experimental_mask()) {
                // SAFETY: callers capability-gate experimental workloads before requesting a
                // device. Cooperative-matrix shaders use initialized, bounds-checked buffers,
                // an adapter-reported shape/type tuple, and are validation-error scoped before
                // any command is submitted.
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                wgpu::ExperimentalFeatures::disabled()
            };
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("Gluj-Bench GPU bandwidth device"),
            required_features,
            required_limits: record.limits.clone(),
            experimental_features,
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(record.adapter.request_device(&descriptor))
            .map_err(|problem| BenchmarkError::new("device_lost", problem.to_string()))?;
        Ok(Self { device, queue })
    }
}

fn measure_gpu_operation(
    context: &GpuContext,
    size: u64,
    operation: KernelOperation,
    sample_count: u32,
    target_per_sample: Duration,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, BenchmarkError> {
    if matches!(operation, KernelOperation::Copy)
        && context
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    {
        return measure_native_gpu_copy(
            context,
            size,
            sample_count,
            target_per_sample,
            cancellation,
        );
    }
    measure_gpu_operation_with_dispatch(
        context,
        DispatchMeasurement {
            size,
            dispatch_bytes: size,
            operation,
            flavor: KernelFlavor::Stream,
            precondition: true,
            sample_count,
            target_per_sample,
        },
        cancellation,
    )
}

fn measure_gpu_operation_with_dispatch(
    context: &GpuContext,
    measurement: DispatchMeasurement,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, BenchmarkError> {
    let DispatchMeasurement {
        size,
        dispatch_bytes,
        operation,
        flavor,
        precondition,
        sample_count,
        target_per_sample,
    } = measurement;
    let harness = KernelHarness::new(context, size, dispatch_bytes, operation, flavor)?;
    let calibration_iterations = 64;
    let calibration_elapsed = harness.measure(context, calibration_iterations)?;
    let mut per_dispatch_ns = calibration_elapsed / calibration_iterations as f64;
    if per_dispatch_ns <= 0.0 || !per_dispatch_ns.is_finite() {
        return Err(BenchmarkError::new(
            "device_lost",
            "GPU timestamp calibration remained invalid after a 64-dispatch batch.",
        ));
    }
    if precondition {
        let warm_iterations = (GPU_PRECONDITION_MS * 1e6 / per_dispatch_ns)
            .ceil()
            .clamp(1.0, 4096.0) as u32;
        let warm_elapsed = harness.measure(context, warm_iterations)?;
        per_dispatch_ns = warm_elapsed / warm_iterations as f64;
    }
    let target_ns = target_per_sample.as_secs_f64() * 1e9;
    let iterations = (target_ns / per_dispatch_ns).ceil().clamp(1.0, 4096.0) as u32;
    let mut values = Vec::with_capacity(sample_count as usize);
    for _ in 0..sample_count {
        ensure_not_cancelled(cancellation)?;
        let elapsed_ns = harness.measure(context, iterations)?;
        let bytes = harness.traffic_bytes as f64
            * operation.reported_byte_multiplier() as f64
            * iterations as f64;
        let bandwidth = bytes / (elapsed_ns / 1e9);
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(BenchmarkError::new(
                "device_lost",
                "GPU bandwidth sample was not finite and positive.",
            ));
        }
        values.push(bandwidth);
    }
    Ok(values)
}

struct KernelHarness {
    size: u64,
    traffic_bytes: u64,
    workgroups_x: u32,
    workgroups_y: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    query_set: wgpu::QuerySet,
    query_resolve: wgpu::Buffer,
    query_readback: wgpu::Buffer,
}

impl KernelHarness {
    fn new(
        context: &GpuContext,
        size: u64,
        dispatch_bytes: u64,
        operation: KernelOperation,
        flavor: KernelFlavor,
    ) -> Result<Self, BenchmarkError> {
        if size < 16 || !size.is_multiple_of(16) {
            return Err(BenchmarkError::new(
                "invalid_working_set",
                "GPU working sets must be 16-byte aligned.",
            ));
        }
        let source = create_buffer_checked(
            context,
            &wgpu::BufferDescriptor {
                label: Some("Gluj-Bench source"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let destination = create_buffer_checked(
            context,
            &wgpu::BufferDescriptor {
                label: Some("Gluj-Bench destination"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        if dispatch_bytes < size || !dispatch_bytes.is_multiple_of(16) {
            return Err(BenchmarkError::new(
                "invalid_working_set",
                "GPU dispatch traffic must cover the complete aligned working set.",
            ));
        }
        let element_count = size / 16;
        let access_count = dispatch_bytes / 16;
        let accesses_per_invocation = operation.accesses_per_invocation(flavor);
        if !access_count.is_multiple_of(accesses_per_invocation) {
            return Err(BenchmarkError::new(
                "invalid_working_set",
                "GPU traffic must divide evenly across the selected vector kernel.",
            ));
        }
        let total_invocations = access_count / accesses_per_invocation;
        let total_workgroups = total_invocations.div_ceil(WORKGROUP_SIZE);
        let workgroups_x = total_workgroups.min(65_535) as u32;
        let workgroups_y = total_workgroups.div_ceil(workgroups_x as u64) as u32;
        let params = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gluj-Bench bandwidth parameters"),
                contents: &parameters(
                    element_count as u32,
                    0x9e37_79b9,
                    workgroups_x * WORKGROUP_SIZE as u32,
                    access_count as u32,
                ),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let checksum_size = (total_invocations * 4).max(4);
        let checksums = create_buffer_checked(
            context,
            &wgpu::BufferDescriptor {
                label: Some("Gluj-Bench read checksums"),
                size: checksum_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )?;
        let module = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Gluj-Bench bandwidth shader"),
                source: wgpu::ShaderSource::Wgsl(shader::BANDWIDTH_SHADER.into()),
            });
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Gluj-Bench bandwidth bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Gluj-Bench bandwidth pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gluj-Bench bandwidth pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(operation.entry_point(flavor)),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Gluj-Bench bandwidth bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: source.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: destination.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: checksums.as_entire_binding(),
                    },
                ],
            });
        let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Gluj-Bench timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let query_resolve = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gluj-Bench timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let query_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gluj-Bench timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let harness = Self {
            size,
            traffic_bytes: dispatch_bytes,
            workgroups_x,
            workgroups_y,
            pipeline,
            bind_group,
            query_set,
            query_resolve,
            query_readback,
        };
        harness.warm(context)?;
        Ok(harness)
    }

    fn warm(&self, context: &GpuContext) -> Result<(), BenchmarkError> {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench warmup"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroups_x, self.workgroups_y, 1);
        }
        context.queue.submit([encoder.finish()]);
        wait_for_gpu(&context.device)
    }

    fn measure(&self, context: &GpuContext, iterations: u32) -> Result<f64, BenchmarkError> {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench timestamped bandwidth"),
            });
        {
            let timestamp_writes = wgpu::ComputePassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gluj-Bench bandwidth pass"),
                timestamp_writes: Some(timestamp_writes),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            for _ in 0..iterations {
                pass.dispatch_workgroups(self.workgroups_x, self.workgroups_y, 1);
            }
        }
        context.queue.submit([encoder.finish()]);
        wait_for_gpu(&context.device)?;
        let mut resolve = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench timestamp resolve"),
            });
        resolve.resolve_query_set(&self.query_set, 0..2, &self.query_resolve, 0);
        resolve.copy_buffer_to_buffer(&self.query_resolve, 0, &self.query_readback, 0, 16);
        context.queue.submit([resolve.finish()]);
        let bytes = map_read(&context.device, &self.query_readback, 16)?;
        let start = u64::from_le_bytes(bytes[0..8].try_into().expect("timestamp width"));
        let end = u64::from_le_bytes(bytes[8..16].try_into().expect("timestamp width"));
        drop(bytes);
        self.query_readback.unmap();
        let ticks = end.wrapping_sub(start);
        let period = context.queue.get_timestamp_period() as f64;
        let elapsed = ticks as f64 * period;
        if elapsed <= 0.0 || !elapsed.is_finite() {
            return Err(BenchmarkError::new(
                "device_lost",
                format!(
                    "GPU timestamp query returned an invalid interval (start={start}, end={end}, period_ns={period})."
                ),
            ));
        }
        let _ = self.size;
        Ok(elapsed)
    }
}

fn measure_native_gpu_copy(
    context: &GpuContext,
    size: u64,
    sample_count: u32,
    target_per_sample: Duration,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, BenchmarkError> {
    let harness = NativeCopyHarness::new(context, size)?;
    let calibration_iterations = 16;
    let calibration_elapsed = harness.measure(context, calibration_iterations)?;
    let mut per_copy_ns = calibration_elapsed / calibration_iterations as f64;
    let warm_iterations = (GPU_PRECONDITION_MS * 1e6 / per_copy_ns)
        .ceil()
        .clamp(1.0, 4096.0) as u32;
    let warm_elapsed = harness.measure(context, warm_iterations)?;
    per_copy_ns = warm_elapsed / warm_iterations as f64;
    let target_ns = target_per_sample.as_secs_f64() * 1e9;
    let iterations = (target_ns / per_copy_ns).ceil().clamp(1.0, 4096.0) as u32;
    let mut values = Vec::with_capacity(sample_count as usize);
    for _ in 0..sample_count {
        ensure_not_cancelled(cancellation)?;
        let elapsed_ns = harness.measure(context, iterations)?;
        let traffic_bytes = size as f64 * 2.0 * iterations as f64;
        values.push(traffic_bytes / (elapsed_ns / 1e9));
    }
    Ok(values)
}

struct NativeCopyHarness {
    size: u64,
    source: wgpu::Buffer,
    destination: wgpu::Buffer,
    query_set: wgpu::QuerySet,
    query_resolve: wgpu::Buffer,
    query_readback: wgpu::Buffer,
}

impl NativeCopyHarness {
    fn new(context: &GpuContext, size: u64) -> Result<Self, BenchmarkError> {
        let source = create_buffer_checked(
            context,
            &wgpu::BufferDescriptor {
                label: Some("Gluj-Bench native-copy source"),
                size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let destination = create_buffer_checked(
            context,
            &wgpu::BufferDescriptor {
                label: Some("Gluj-Bench native-copy destination"),
                size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Gluj-Bench native-copy timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let query_resolve = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gluj-Bench native-copy timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let query_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gluj-Bench native-copy timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let harness = Self {
            size,
            source,
            destination,
            query_set,
            query_resolve,
            query_readback,
        };
        harness.submit_copies(context, 1, false)?;
        Ok(harness)
    }

    fn submit_copies(
        &self,
        context: &GpuContext,
        iterations: u32,
        timestamped: bool,
    ) -> Result<(), BenchmarkError> {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench native GPU copy"),
            });
        if timestamped {
            encoder.write_timestamp(&self.query_set, 0);
        }
        for _ in 0..iterations {
            encoder.copy_buffer_to_buffer(&self.source, 0, &self.destination, 0, self.size);
        }
        if timestamped {
            encoder.write_timestamp(&self.query_set, 1);
        }
        context.queue.submit([encoder.finish()]);
        wait_for_gpu(&context.device)
    }

    fn measure(&self, context: &GpuContext, iterations: u32) -> Result<f64, BenchmarkError> {
        self.submit_copies(context, iterations, true)?;
        let mut resolve = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gluj-Bench native-copy timestamp resolve"),
            });
        resolve.resolve_query_set(&self.query_set, 0..2, &self.query_resolve, 0);
        resolve.copy_buffer_to_buffer(&self.query_resolve, 0, &self.query_readback, 0, 16);
        context.queue.submit([resolve.finish()]);
        let bytes = map_read(&context.device, &self.query_readback, 16)?;
        let start = u64::from_le_bytes(bytes[0..8].try_into().expect("timestamp width"));
        let end = u64::from_le_bytes(bytes[8..16].try_into().expect("timestamp width"));
        drop(bytes);
        self.query_readback.unmap();
        let elapsed = end.wrapping_sub(start) as f64 * context.queue.get_timestamp_period() as f64;
        if elapsed <= 0.0 || !elapsed.is_finite() {
            return Err(BenchmarkError::new(
                "device_lost",
                "Native GPU copy timestamp interval was invalid.",
            ));
        }
        Ok(elapsed)
    }
}

fn measure_host_to_device(
    context: &GpuContext,
    size: u64,
    samples: u32,
    target: Duration,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, BenchmarkError> {
    let staging = (0..HOST_STAGING_BUFFER_COUNT)
        .map(|index| {
            create_initialized_host_buffer(
                context,
                &format!("Gluj-Bench upload staging {index}"),
                size,
                wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                0xa5,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let destination = create_buffer_checked(
        context,
        &wgpu::BufferDescriptor {
            label: Some("Gluj-Bench host-to-device destination"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    )?;
    let trial_started = Instant::now();
    copy_from_upload_ring(context, &staging, &destination, size, 1)?;
    let trial = trial_started.elapsed().as_secs_f64().max(1e-6);
    let iterations = (target.as_secs_f64() / trial).ceil().clamp(1.0, 4096.0) as u32;
    let mut values = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        ensure_not_cancelled(cancellation)?;
        let started = Instant::now();
        copy_from_upload_ring(context, &staging, &destination, size, iterations)?;
        let elapsed = started.elapsed().as_secs_f64();
        values.push(size as f64 * iterations as f64 / elapsed);
    }
    Ok(values)
}

fn measure_device_to_host(
    context: &GpuContext,
    size: u64,
    samples: u32,
    target: Duration,
    cancellation: &CancellationToken,
) -> Result<Vec<f64>, BenchmarkError> {
    let source = create_buffer_checked(
        context,
        &wgpu::BufferDescriptor {
            label: Some("Gluj-Bench device-to-host source"),
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let readbacks = (0..HOST_STAGING_BUFFER_COUNT)
        .map(|index| {
            create_initialized_host_buffer(
                context,
                &format!("Gluj-Bench download staging {index}"),
                size,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                0,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let trial_started = Instant::now();
    copy_to_readback_ring(context, &source, &readbacks, size, 1)?;
    let trial = trial_started.elapsed().as_secs_f64().max(1e-6);
    let target_seconds = target.as_secs_f64();
    let iterations = (target_seconds / trial).ceil().clamp(1.0, 4096.0) as u32;
    let mut values = Vec::new();
    for _ in 0..samples {
        ensure_not_cancelled(cancellation)?;
        let started = Instant::now();
        copy_to_readback_ring(context, &source, &readbacks, size, iterations)?;
        let elapsed = started.elapsed().as_secs_f64();
        values.push(size as f64 * iterations as f64 / elapsed);
    }
    Ok(values)
}

fn copy_from_upload_ring(
    context: &GpuContext,
    staging: &[wgpu::Buffer],
    destination: &wgpu::Buffer,
    size: u64,
    iterations: u32,
) -> Result<(), BenchmarkError> {
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gluj-Bench batched host-to-device copies"),
        });
    for iteration in 0..iterations as usize {
        encoder.copy_buffer_to_buffer(&staging[iteration % staging.len()], 0, destination, 0, size);
    }
    context.queue.submit([encoder.finish()]);
    wait_for_gpu(&context.device)
}

fn copy_to_readback_ring(
    context: &GpuContext,
    source: &wgpu::Buffer,
    readbacks: &[wgpu::Buffer],
    size: u64,
    iterations: u32,
) -> Result<(), BenchmarkError> {
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gluj-Bench device-to-host copies"),
        });
    for iteration in 0..iterations {
        let destination = &readbacks[iteration as usize % readbacks.len()];
        encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
    }
    context.queue.submit([encoder.finish()]);
    map_readback_ring(&context.device, readbacks, size, iterations)
}

fn create_initialized_host_buffer(
    context: &GpuContext,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
    pattern: u8,
) -> Result<wgpu::Buffer, BenchmarkError> {
    let buffer = create_buffer_checked(
        context,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: true,
        },
    )?;
    {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .map_err(|problem| BenchmarkError::new("mapping_failed", problem.to_string()))?;
        let chunk = [pattern; 4096];
        for offset in (0..mapped.len()).step_by(chunk.len()) {
            let end = (offset + chunk.len()).min(mapped.len());
            mapped
                .slice(offset..end)
                .copy_from_slice(&chunk[..end - offset]);
        }
    }
    buffer.unmap();
    Ok(buffer)
}

fn map_readback_ring(
    device: &wgpu::Device,
    readbacks: &[wgpu::Buffer],
    size: u64,
    iterations: u32,
) -> Result<(), BenchmarkError> {
    let used = (iterations as usize).min(readbacks.len());
    let mut receivers = Vec::with_capacity(used);
    for buffer in &readbacks[..used] {
        let (sender, receiver) = mpsc::channel();
        buffer
            .slice(0..size)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        receivers.push(receiver);
    }
    wait_for_gpu(device)?;
    for (buffer, receiver) in readbacks[..used].iter().zip(receivers) {
        receiver
            .recv()
            .map_err(|_| {
                BenchmarkError::new("mapping_failed", "GPU mapping callback was dropped.")
            })?
            .map_err(|problem| BenchmarkError::new("mapping_failed", problem.to_string()))?;
        {
            let mapped = buffer
                .slice(0..size)
                .get_mapped_range()
                .map_err(|problem| BenchmarkError::new("mapping_failed", problem.to_string()))?;
            std::hint::black_box(mapped.first().copied());
        }
        buffer.unmap();
    }
    Ok(())
}

fn create_buffer_checked(
    context: &GpuContext,
    descriptor: &wgpu::BufferDescriptor<'_>,
) -> Result<wgpu::Buffer, BenchmarkError> {
    let error_scope = context
        .device
        .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buffer = context.device.create_buffer(descriptor);
    if let Some(problem) = pollster::block_on(error_scope.pop()) {
        return Err(BenchmarkError::new(
            "insufficient_gpu_memory",
            format!("GPU buffer allocation failed: {problem}"),
        ));
    }
    Ok(buffer)
}

fn map_read(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Result<wgpu::BufferView, BenchmarkError> {
    let (sender, receiver) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, 0..size, move |result| {
        let _ = sender.send(result);
    });
    wait_for_gpu(device)?;
    receiver
        .recv()
        .map_err(|_| BenchmarkError::new("mapping_failed", "GPU mapping callback was dropped."))?
        .map_err(|problem| BenchmarkError::new("mapping_failed", problem.to_string()))?;
    buffer
        .get_mapped_range(0..size)
        .map_err(|problem| BenchmarkError::new("mapping_failed", problem.to_string()))
}

fn wait_for_gpu(device: &wgpu::Device) -> Result<(), BenchmarkError> {
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map(|_| ())
        .map_err(|problem| BenchmarkError::new("device_lost", problem.to_string()))
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
    record
        .limits
        .max_buffer_size
        .min(record.limits.max_storage_buffer_binding_size)
        .min(CACHE_SWEEP_MAX)
}

fn host_transfer_size(adapter_limit: u64, available_memory: u64, integrated: bool) -> Option<u64> {
    let mut limit = adapter_limit.min(64 * MIB);
    if integrated {
        limit = limit.min(available_memory / 16);
    }
    let size = limit / 256 * 256;
    (size >= 4 * MIB).then_some(size)
}

fn parameters(element_count: u32, seed: u32, row_width: u32, access_count: u32) -> [u8; 16] {
    let mut bytes = [0; 16];
    bytes[0..4].copy_from_slice(&element_count.to_le_bytes());
    bytes[4..8].copy_from_slice(&seed.to_le_bytes());
    bytes[8..12].copy_from_slice(&row_width.to_le_bytes());
    bytes[12..16].copy_from_slice(&access_count.to_le_bytes());
    bytes
}

fn per_sample_duration(config: &BenchmarkConfig) -> Duration {
    Duration::from_millis((config.target_duration_ms / config.samples.max(1) as u64).max(1))
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

fn link_classification(device_type: wgpu::DeviceType) -> &'static str {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => "probable_pcie",
        wgpu::DeviceType::IntegratedGpu => "shared_memory_or_uma",
        _ => "host_device_path_unknown",
    }
}

fn adapter_score(adapter: &wgpu::Adapter) -> u8 {
    let timestamp = u8::from(adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY)) * 10;
    let backend = match adapter.get_info().backend {
        wgpu::Backend::Vulkan => 5,
        wgpu::Backend::Dx12 => 4,
        wgpu::Backend::Metal => 3,
        wgpu::Backend::Gl => 2,
        _ => 1,
    };
    timestamp + backend
}

fn common_metadata(record: &AdapterRecord, timing_domain: &str) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    metadata.insert("adapter_name".into(), record.info.name.clone());
    metadata.insert("backend".into(), format!("{:?}", record.info.backend));
    metadata.insert(
        "device_type".into(),
        format!("{:?}", record.info.device_type),
    );
    metadata.insert("timing_domain".into(), timing_domain.into());
    metadata.insert("shader_access_width_bytes".into(), "16".into());
    metadata.insert("workgroup_size".into(), WORKGROUP_SIZE.to_string());
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
    metadata.insert("name".into(), record.info.name.clone());
    metadata.insert("backend".into(), format!("{:?}", record.info.backend));
    metadata.insert(
        "device_type".into(),
        format!("{:?}", record.info.device_type),
    );
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
                "gpu.performance.fp32",
                "gpu.performance.fp16",
                "gpu.performance.fp64",
                "gpu.performance.int32",
                "gpu.performance.int8_packed",
                cooperative_matrix::FP16_MATRIX_ID,
                cooperative_matrix::INT8_MATRIX_ID,
            ]
        );
        assert!(
            provider
                .devices
                .iter()
                .all(|device| device.id.starts_with("gpu:wgpu:"))
        );
    }

    #[test]
    fn link_labels_do_not_claim_pcie_for_integrated_devices() {
        assert_eq!(
            link_classification(wgpu::DeviceType::DiscreteGpu),
            "probable_pcie"
        );
        assert_eq!(
            link_classification(wgpu::DeviceType::IntegratedGpu),
            "shared_memory_or_uma"
        );
    }

    #[test]
    fn parameter_buffer_matches_wgsl_uniform_layout() {
        let bytes = parameters(123, 456, 65_280, 4096);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 123);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 456);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 65_280);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 4096);
        assert_eq!(bytes.len(), 16);
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
