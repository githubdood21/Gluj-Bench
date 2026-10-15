mod ascii_scaling;
mod cache_latency;
mod compute;
mod compute_scaling;
mod integer_scaling;
mod kernels;
mod latency;
mod matrix_scaling;
mod pacing;
mod topology;

use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkProvider,
    BenchmarkResult, CancellationToken, DeviceCategory, DeviceDescriptor, Metric, ProgressCallback,
    ProgressUpdate,
};
use kernels::{
    CachePreparation, MeasurementOptions, Operation, cold_cache_method, measure_parallel,
    measure_parallel_sizes, simd_path,
};
use std::collections::BTreeMap;
use sysinfo::System;
use topology::{CpuTopology, ProcessorLocation};

const SUITE_ID: &str = "cpu.bandwidth";
const MIB: u64 = 1024 * 1024;
const MAX_RAM_PAYLOAD: u64 = 1024 * MIB;

pub struct CpuBandwidthProvider {
    topology: Result<CpuTopology, BenchmarkError>,
    cpu_device: DeviceDescriptor,
    memory_device: DeviceDescriptor,
    available_memory: u64,
}

impl CpuBandwidthProvider {
    pub fn discover() -> Self {
        let system = System::new_all();
        let name = system
            .cpus()
            .first()
            .map(|cpu| cpu.brand().trim().to_owned())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "System CPU".to_owned());
        let topology = topology::discover();
        let mut cpu_properties = BTreeMap::new();
        cpu_properties.insert("logical_processors".into(), system.cpus().len().to_string());
        cpu_properties.insert("architecture".into(), std::env::consts::ARCH.into());
        let (physical_cores, caches, status) = match &topology {
            Ok(topology) => {
                cpu_properties.insert(
                    "physical_cores".into(),
                    topology.physical_cores.len().to_string(),
                );
                cpu_properties.insert(
                    "representative_processor_group".into(),
                    topology.representative.group.to_string(),
                );
                cpu_properties.insert(
                    "representative_processor_index".into(),
                    topology.representative.index.to_string(),
                );
                (
                    topology.physical_cores.len(),
                    topology.caches.clone(),
                    "CPU topology and bandwidth capabilities discovered.".to_owned(),
                )
            }
            Err(problem) => (
                System::physical_core_count().unwrap_or(0),
                Vec::new(),
                problem.message.clone(),
            ),
        };
        cpu_properties
            .entry("physical_cores".into())
            .or_insert_with(|| physical_cores.to_string());
        let cpu_device = DeviceDescriptor {
            id: "cpu:system".into(),
            name,
            category: DeviceCategory::Cpu,
            available: topology.is_ok(),
            status,
            properties: cpu_properties,
            caches,
        };
        let mut memory_properties = BTreeMap::new();
        memory_properties.insert("total_bytes".into(), system.total_memory().to_string());
        memory_properties.insert(
            "available_bytes_at_discovery".into(),
            system.available_memory().to_string(),
        );
        let memory_device = DeviceDescriptor {
            id: "memory:system".into(),
            name: "System memory".into(),
            category: DeviceCategory::Memory,
            available: topology.is_ok(),
            status: "Available for aggregate CPU-visible bandwidth measurement.".into(),
            properties: memory_properties,
            caches: Vec::new(),
        };
        Self {
            topology,
            cpu_device,
            memory_device,
            available_memory: system.available_memory(),
        }
    }

    fn cache_for_core(
        topology: &CpuTopology,
        location: ProcessorLocation,
        level: u8,
    ) -> Option<&topology::CacheTarget> {
        let processor_mask = 1usize << location.index;
        topology
            .cache_targets
            .iter()
            .filter(|cache| {
                cache.descriptor.level == level
                    && cache.group == location.group
                    && cache.mask & processor_mask != 0
            })
            .max_by_key(|cache| cache.descriptor.size_bytes)
    }

    fn cache_layout(
        topology: &CpuTopology,
        level: u8,
    ) -> Result<Vec<(ProcessorLocation, usize)>, String> {
        let mut instances: BTreeMap<(u16, usize, u64), Vec<ProcessorLocation>> = BTreeMap::new();
        for location in topology.physical_cores.iter().copied() {
            if let Some(cache) = Self::cache_for_core(topology, location, level) {
                instances
                    .entry((cache.group, cache.mask, cache.descriptor.size_bytes))
                    .or_default()
                    .push(location);
            }
        }
        if instances.is_empty() {
            return Err("cache_level_not_present".into());
        }
        let mut layout = Vec::new();
        for ((_, _, cache_size), locations) in instances {
            let instance_budget = cache_size.saturating_mul(2) / 5;
            let per_core = (instance_budget / locations.len() as u64) / 64 * 64;
            for location in locations {
                let lower_size = (0..level)
                    .rev()
                    .find_map(|lower| {
                        Self::cache_for_core(topology, location, lower)
                            .map(|cache| cache.descriptor.size_bytes)
                    })
                    .unwrap_or(0);
                if per_core <= lower_size.saturating_mul(2) {
                    return Err("insufficient_level_separation".into());
                }
                layout.push((
                    location,
                    usize::try_from(per_core).map_err(|_| "working_set_too_large".to_owned())?,
                ));
            }
        }
        layout.sort_unstable_by_key(|(location, _)| *location);
        Ok(layout)
    }

    fn logical_cache_layout(
        topology: &CpuTopology,
        level: u8,
    ) -> Result<Vec<(ProcessorLocation, usize)>, String> {
        let physical_layout = Self::cache_layout(topology, level)?;
        let mut logical_layout = Vec::new();
        for (physical, bytes) in physical_layout {
            let threads = topology
                .core_threads
                .get(&physical)
                .ok_or_else(|| "logical_topology_unavailable".to_owned())?;
            let per_thread = (bytes / threads.len()) / 64 * 64;
            if per_thread == 0 {
                return Err("working_set_too_small".into());
            }
            logical_layout.extend(threads.iter().copied().map(|thread| (thread, per_thread)));
        }
        logical_layout.sort_unstable_by_key(|(location, _)| *location);
        Ok(logical_layout)
    }

    fn logical_mode(config: &BenchmarkConfig) -> bool {
        if topology::explicit_core_limit(config) {
            return false;
        }
        config.options.get("thread_mode").map(String::as_str) == Some("logical_processors")
    }

    fn ram_payload(&self, topology: &CpuTopology) -> Result<usize, String> {
        let highest = topology
            .caches
            .iter()
            .map(|cache| cache.level)
            .max()
            .unwrap_or(0);
        let aggregate_llc = topology
            .caches
            .iter()
            .filter(|cache| cache.level == highest)
            .map(|cache| cache.size_bytes.saturating_mul(cache.instances as u64))
            .sum::<u64>();
        let desired = (256 * MIB).max(aggregate_llc.saturating_mul(4));
        let safe_cap = (self.available_memory / 16).min(MAX_RAM_PAYLOAD);
        let minimum = aggregate_llc.saturating_mul(2).saturating_add(64 * MIB);
        let payload = desired.min(safe_cap) / 64 * 64;
        if payload < minimum {
            return Err("insufficient_available_memory".into());
        }
        usize::try_from(payload).map_err(|_| "working_set_too_large".into())
    }

    fn configured_ram_payload(
        &self,
        topology: &CpuTopology,
        config: &BenchmarkConfig,
        workers: usize,
        available: u64,
    ) -> Result<usize, BenchmarkError> {
        match config
            .options
            .get("ram_bandwidth_mode")
            .map(String::as_str)
            .unwrap_or("automatic")
        {
            "automatic" => self
                .ram_payload(topology)
                .map_err(|reason| BenchmarkError::new("benchmark_unavailable", reason)),
            "custom" => {
                let requested = config
                    .options
                    .get("ram_bandwidth_bytes")
                    .and_then(|value| value.parse::<u64>().ok())
                    .ok_or_else(|| {
                        BenchmarkError::new(
                            "invalid_config",
                            "Choose a positive RAM bandwidth payload size.",
                        )
                    })?;
                let highest = topology
                    .caches
                    .iter()
                    .map(|cache| cache.level)
                    .max()
                    .unwrap_or(0);
                let llc = topology
                    .caches
                    .iter()
                    .filter(|cache| cache.level == highest)
                    .map(|cache| cache.size_bytes.saturating_mul(cache.instances as u64))
                    .sum();
                let (minimum, maximum) =
                    gluj_bench_core::ram_bandwidth_payload_range(llc, available, workers as u32);
                if requested < minimum {
                    return Err(BenchmarkError::new(
                        "invalid_config",
                        format!(
                            "RAM bandwidth payload must be at least {minimum} bytes to stay beyond the cache range."
                        ),
                    ));
                }
                if requested > maximum {
                    return Err(BenchmarkError::new(
                        "dataset_exceeds_budget",
                        format!(
                            "RAM bandwidth payload is {requested} bytes; available memory allows {maximum} bytes with space for both buffers."
                        ),
                    ));
                }
                let alignment = 64 * workers.max(1) as u64;
                usize::try_from(requested / alignment * alignment).map_err(|_| {
                    BenchmarkError::new(
                        "working_set_too_large",
                        "RAM bandwidth payload exceeds the address range.",
                    )
                })
            }
            _ => Err(BenchmarkError::new(
                "invalid_config",
                "RAM bandwidth size mode must be automatic or custom.",
            )),
        }
    }

    fn cache_descriptor(&self, level: u8) -> BenchmarkDescriptor {
        let id = format!("cpu.bandwidth.cache.l{level}");
        let mut metadata = BTreeMap::new();
        metadata.insert("cache_level".into(), level.to_string());
        metadata.insert("cache_capacity_fraction".into(), "0.40".into());
        metadata.insert("cache_state".into(), "preloaded_before_timing".into());
        let (available, reason) = match &self.topology {
            Ok(topology) => match Self::cache_layout(topology, level) {
                Ok(layout) => {
                    let cache_instances = topology
                        .cache_targets
                        .iter()
                        .filter(|cache| cache.descriptor.level == level)
                        .map(|cache| (cache.group, cache.mask))
                        .collect::<std::collections::BTreeSet<_>>()
                        .len();
                    metadata.insert(
                        "aggregate_working_set_bytes".into(),
                        layout
                            .iter()
                            .map(|(_, bytes)| *bytes as u64)
                            .sum::<u64>()
                            .to_string(),
                    );
                    metadata.insert("thread_count".into(), layout.len().to_string());
                    metadata.insert("cache_instances".into(), cache_instances.to_string());
                    (true, String::new())
                }
                Err(reason) => (false, reason),
            },
            Err(problem) => (false, problem.code.clone()),
        };
        BenchmarkDescriptor {
            id,
            name: format!("L{level} cache bandwidth"),
            category: BenchmarkCategory::Cpu,
            workload: "Physical-core-scaled cache read, write, and copy".into(),
            data_type: "bytes".into(),
            unit: "bytes/s".into(),
            supported_device_ids: if available {
                vec![self.cpu_device.id.clone()]
            } else {
                Vec::new()
            },
            available,
            unavailable_reason: reason,
            suite_id: SUITE_ID.into(),
            display_order: level as u32,
            metadata,
        }
    }

    fn memory_descriptor(&self) -> BenchmarkDescriptor {
        let mut metadata = BTreeMap::new();
        metadata.insert("cache_state".into(), "flushed_before_each_sweep".into());
        metadata.insert("cache_flush_method".into(), cold_cache_method().into());
        metadata.insert("access_order".into(), "sequential_rotated_per_sweep".into());
        let (available, reason) = match &self.topology {
            Ok(topology) => match self.ram_payload(topology) {
                Ok(bytes) => {
                    metadata.insert("working_set_bytes".into(), bytes.to_string());
                    (true, String::new())
                }
                Err(reason) => (false, reason),
            },
            Err(problem) => (false, problem.code.clone()),
        };
        BenchmarkDescriptor {
            id: "cpu.bandwidth.memory".into(),
            name: "System RAM bandwidth".into(),
            category: BenchmarkCategory::Memory,
            workload: "Physical-core-scaled streaming memory access".into(),
            data_type: "bytes".into(),
            unit: "bytes/s".into(),
            supported_device_ids: if available {
                vec![self.memory_device.id.clone()]
            } else {
                Vec::new()
            },
            available,
            unavailable_reason: reason,
            suite_id: SUITE_ID.into(),
            display_order: 4,
            metadata,
        }
    }

    fn validate(config: &BenchmarkConfig) -> Result<(), BenchmarkError> {
        if !(3..=20).contains(&config.samples) {
            return Err(BenchmarkError::new(
                "invalid_config",
                "samples must be between 3 and 20.",
            ));
        }
        if !(150..=30_000).contains(&config.target_duration_ms) {
            return Err(BenchmarkError::new(
                "invalid_config",
                "target_duration_ms must be between 150 and 30000.",
            ));
        }
        if let Some(mode) = config.options.get("thread_mode")
            && mode != "physical_cores"
            && mode != "logical_processors"
        {
            return Err(BenchmarkError::new(
                "invalid_config",
                "thread_mode must be physical_cores or logical_processors.",
            ));
        }
        Ok(())
    }

    fn metric(name: &str, statistics: gluj_bench_core::SampleStatistics) -> Metric {
        Metric {
            name: name.into(),
            value: statistics.median,
            unit: "bytes/s".into(),
            statistics,
        }
    }

    fn run_cache(
        &self,
        level: u8,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let topology = self.topology.as_ref().map_err(Clone::clone)?;
        let logical_mode = Self::logical_mode(config);
        let mut layout = if logical_mode {
            Self::logical_cache_layout(topology, level)
        } else {
            Self::cache_layout(topology, level)
        }
        .map_err(|reason| BenchmarkError::new("benchmark_unavailable", reason))?;
        let selected = topology::selected_workers(topology, config, false)?;
        layout.retain(|(location, _)| selected.contains(location));
        let target = topology.representative_caches.get(&level).unwrap();
        let locations: Vec<_> = layout.iter().map(|(location, _)| *location).collect();
        let full_payload_sizes: Vec<_> = layout.iter().map(|(_, bytes)| *bytes).collect();
        let aggregate_working_set = full_payload_sizes.iter().sum::<usize>();
        let cache_instances = topology
            .cache_targets
            .iter()
            .filter(|cache| cache.descriptor.level == level)
            .map(|cache| (cache.group, cache.mask))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let target_ns = config.target_duration_ms.saturating_mul(1_000_000) / config.samples as u64;
        let mut metrics = Vec::new();
        let mut total_elapsed = 0u64;
        for (index, (name, operation)) in [
            ("read", Operation::Read),
            ("write", Operation::Write),
            ("copy", Operation::Copy),
        ]
        .into_iter()
        .enumerate()
        {
            progress(ProgressUpdate {
                fraction: index as f64 / 3.0,
                phase: name.into(),
                message: format!("Measuring L{level} {name} bandwidth"),
            });
            let payload_sizes: Vec<_> = if matches!(operation, Operation::Copy) {
                full_payload_sizes.iter().map(|bytes| bytes / 2).collect()
            } else {
                full_payload_sizes.clone()
            };
            let (statistics, elapsed) = measure_parallel_sizes(
                operation,
                &locations,
                &payload_sizes,
                MeasurementOptions {
                    sample_count: config.samples,
                    target_ns,
                    streaming: false,
                    preparation: CachePreparation::Hot,
                    byte_multiplier: if matches!(operation, Operation::Copy) {
                        2
                    } else {
                        1
                    },
                },
                cancellation,
            )?;
            total_elapsed = total_elapsed.saturating_add(elapsed);
            metrics.push(Self::metric(name, statistics));
        }
        progress(ProgressUpdate {
            fraction: 1.0,
            phase: "complete".into(),
            message: format!("L{level} bandwidth complete"),
        });
        let mut workload_metadata = BTreeMap::new();
        workload_metadata.insert(
            "aggregate_working_set_bytes".into(),
            aggregate_working_set.to_string(),
        );
        workload_metadata.insert(
            "aggregate_copy_useful_payload_bytes".into(),
            (aggregate_working_set / 2).to_string(),
        );
        workload_metadata.insert(
            "aggregate_copy_traffic_bytes".into(),
            aggregate_working_set.to_string(),
        );
        workload_metadata.insert("access_policy".into(), "temporal".into());
        workload_metadata.insert("cache_state".into(), "preloaded_before_timing".into());
        workload_metadata.insert("preparation_timed".into(), "false".into());
        workload_metadata.insert(
            "target_duration_ms_per_operation".into(),
            config.target_duration_ms.to_string(),
        );
        workload_metadata.insert(
            "target_duration_ms_per_sample".into(),
            (config.target_duration_ms / config.samples as u64).to_string(),
        );
        workload_metadata.insert(
            "cache_instance_partitioning".into(),
            "topology_masks".into(),
        );
        workload_metadata.insert("cross_instance_shared_lines".into(), "none".into());
        workload_metadata.insert("buffer_ownership".into(), "pinned_core_first_touch".into());
        workload_metadata.insert("simd_path".into(), simd_path().into());
        workload_metadata.insert("thread_count".into(), locations.len().to_string());
        workload_metadata.insert(
            "thread_mode".into(),
            if logical_mode {
                "logical_processors"
            } else {
                "physical_cores"
            }
            .into(),
        );
        workload_metadata.insert("cache_instances".into(), cache_instances.to_string());
        workload_metadata.insert(
            "scope".into(),
            if logical_mode {
                "aggregate_logical_processors"
            } else {
                "aggregate_physical_cores"
            }
            .into(),
        );
        workload_metadata.insert(
            "copy_byte_definition".into(),
            "read_plus_write_traffic".into(),
        );
        let mut device_metadata = BTreeMap::new();
        device_metadata.insert("cache_level".into(), level.to_string());
        device_metadata.insert(
            "cache_size_bytes".into(),
            target.descriptor.size_bytes.to_string(),
        );
        device_metadata.insert(
            "cache_line_bytes".into(),
            target.descriptor.line_size_bytes.to_string(),
        );
        device_metadata.insert("cache_instances".into(), cache_instances.to_string());
        Ok(BenchmarkResult {
            benchmark_id: format!("cpu.bandwidth.cache.l{level}"),
            device_id: self.cpu_device.id.clone(),
            elapsed_ns: total_elapsed,
            metrics,
            workload_metadata,
            device_metadata,
        })
    }

    fn run_memory(
        &self,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let topology = self.topology.as_ref().map_err(Clone::clone)?;
        let locations = topology::selected_workers(topology, config, false)?;
        let mut memory = sysinfo::System::new();
        memory.refresh_memory();
        let payload = self.configured_ram_payload(
            topology,
            config,
            locations.len(),
            memory.available_memory(),
        )?;
        let target_ns = config.target_duration_ms.saturating_mul(1_000_000) / config.samples as u64;
        let mut metrics = Vec::new();
        let mut total_elapsed = 0u64;
        for (index, (name, operation)) in [
            ("read", Operation::Read),
            ("write", Operation::Write),
            ("copy", Operation::Copy),
        ]
        .into_iter()
        .enumerate()
        {
            progress(ProgressUpdate {
                fraction: index as f64 / 3.0,
                phase: name.into(),
                message: format!("Measuring RAM {name} bandwidth"),
            });
            let streaming = !matches!(operation, Operation::Read);
            let (statistics, elapsed) = measure_parallel(
                operation,
                &locations,
                payload,
                MeasurementOptions {
                    sample_count: config.samples,
                    target_ns,
                    streaming,
                    preparation: CachePreparation::Cold,
                    byte_multiplier: 1,
                },
                cancellation,
            )?;
            total_elapsed = total_elapsed.saturating_add(elapsed);
            metrics.push(Self::metric(name, statistics));
        }
        progress(ProgressUpdate {
            fraction: 1.0,
            phase: "complete".into(),
            message: "RAM bandwidth complete".into(),
        });
        let mut workload_metadata = BTreeMap::new();
        workload_metadata.insert("payload_bytes".into(), payload.to_string());
        let actual_payload = (payload / locations.len().max(1))
            .max(64)
            .next_multiple_of(64)
            * locations.len().max(1);
        workload_metadata.insert(
            "ram_bandwidth_mode".into(),
            config
                .options
                .get("ram_bandwidth_mode")
                .cloned()
                .unwrap_or_else(|| "automatic".into()),
        );
        workload_metadata.insert("actual_payload_bytes".into(), actual_payload.to_string());
        workload_metadata.insert(
            "allocated_test_buffer_bytes".into(),
            (actual_payload as u64 * 2).to_string(),
        );
        if config
            .options
            .get("ram_bandwidth_mode")
            .is_some_and(|mode| mode == "custom")
            && let Some(requested) = config.options.get("ram_bandwidth_bytes")
        {
            workload_metadata.insert("requested_payload_bytes".into(), requested.clone());
        }
        workload_metadata.insert("write_copy_policy".into(), "non_temporal_streaming".into());
        workload_metadata.insert("read_policy".into(), "temporal".into());
        workload_metadata.insert("cache_state".into(), "flushed_before_each_sweep".into());
        workload_metadata.insert("cache_flush_method".into(), cold_cache_method().into());
        workload_metadata.insert("access_order".into(), "sequential_rotated_per_sweep".into());
        workload_metadata.insert("preparation_timed".into(), "false".into());
        workload_metadata.insert(
            "target_duration_ms_per_operation".into(),
            config.target_duration_ms.to_string(),
        );
        workload_metadata.insert(
            "target_duration_ms_per_sample".into(),
            (config.target_duration_ms / config.samples as u64).to_string(),
        );
        workload_metadata.insert("buffer_ownership".into(), "pinned_core_first_touch".into());
        workload_metadata.insert("simd_path".into(), simd_path().into());
        workload_metadata.insert("thread_count".into(), locations.len().to_string());
        workload_metadata.insert(
            "thread_mode".into(),
            if Self::logical_mode(config) {
                "logical_processors"
            } else {
                "physical_cores"
            }
            .into(),
        );
        workload_metadata.insert("copy_byte_definition".into(), "useful_payload".into());
        let mut device_metadata = BTreeMap::new();
        device_metadata.insert(
            "physical_cores".into(),
            topology.physical_cores.len().to_string(),
        );
        Ok(BenchmarkResult {
            benchmark_id: "cpu.bandwidth.memory".into(),
            device_id: self.memory_device.id.clone(),
            elapsed_ns: total_elapsed,
            metrics,
            workload_metadata,
            device_metadata,
        })
    }
}

impl Default for CpuBandwidthProvider {
    fn default() -> Self {
        Self::discover()
    }
}

impl BenchmarkProvider for CpuBandwidthProvider {
    fn devices(&self) -> Vec<DeviceDescriptor> {
        vec![self.cpu_device.clone(), self.memory_device.clone()]
    }
    fn benchmarks(&self) -> Vec<BenchmarkDescriptor> {
        let mut items: Vec<_> = (1..=3).map(|level| self.cache_descriptor(level)).collect();
        items.push(self.memory_descriptor());
        for pattern in [latency::Pattern::Localized, latency::Pattern::RandomObjects] {
            items.push(latency::descriptor(
                pattern,
                &self.memory_device,
                self.topology.as_ref(),
                self.available_memory,
            ));
        }
        for level in 1..=3 {
            items.push(cache_latency::descriptor(
                level,
                &self.cpu_device,
                self.topology.as_ref(),
            ));
        }
        items.extend(
            compute::descriptors(&self.cpu_device, self.topology.is_ok())
                .into_iter()
                .filter(|item| {
                    item.id != "cpu.performance.avx2.f32_fma"
                        && item.id != ascii_scaling::ID
                        && item.id != integer_scaling::ID
                }),
        );
        items.push(ascii_scaling::descriptor(
            &self.cpu_device,
            self.topology.as_ref().ok(),
        ));
        let mut vector = compute_scaling::descriptor(&self.cpu_device, self.topology.is_ok());
        if let Ok(topology) = &self.topology {
            compute_scaling::cache_previews(&mut vector, topology);
        }
        items.push(vector);
        let mut matrix =
            compute_scaling::matrix_descriptor(&self.cpu_device, self.topology.is_ok());
        if let Ok(topology) = &self.topology {
            compute_scaling::cache_previews(&mut matrix, topology);
        }
        items.push(matrix);
        let mut integer =
            compute_scaling::integer_descriptor(&self.cpu_device, self.topology.is_ok());
        if let Ok(topology) = &self.topology {
            compute_scaling::cache_previews(&mut integer, topology);
        }
        items.push(integer);
        items
    }
    fn run(
        &mut self,
        benchmark_id: &str,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        Self::validate(config)?;
        if let Some(level) = cache_latency::level(benchmark_id) {
            return cache_latency::run(
                level,
                config,
                cancellation,
                progress,
                &self.cpu_device,
                self.topology.as_ref().map_err(Clone::clone)?,
            );
        }
        if let Some(pattern) = latency::Pattern::from_id(benchmark_id) {
            return latency::run(
                pattern,
                config,
                cancellation,
                progress,
                &self.memory_device,
                self.topology.as_ref().map_err(Clone::clone)?,
            );
        }
        if benchmark_id == ascii_scaling::ID {
            return ascii_scaling::run(
                config,
                cancellation,
                progress,
                &self.cpu_device,
                self.topology.as_ref().map_err(Clone::clone)?,
            );
        }
        if benchmark_id == compute_scaling::ID
            || benchmark_id == matrix_scaling::ID
            || benchmark_id == integer_scaling::ID
        {
            return compute_scaling::run(
                benchmark_id,
                config,
                cancellation,
                progress,
                &self.cpu_device,
                self.topology.as_ref().map_err(Clone::clone)?,
            );
        }
        if benchmark_id.starts_with("cpu.performance.") {
            let topology = self.topology.as_ref().map_err(Clone::clone)?;
            return compute::run(
                benchmark_id,
                config,
                cancellation,
                progress,
                topology.representative,
                &self.cpu_device,
                topology,
            );
        }
        if benchmark_id == "cpu.bandwidth.memory" {
            return self.run_memory(config, cancellation, progress);
        }
        if let Some(level) = benchmark_id
            .strip_prefix("cpu.bandwidth.cache.l")
            .and_then(|value| value.parse::<u8>().ok())
            .filter(|level| (1..=3).contains(level))
        {
            return self.run_cache(level, config, cancellation, progress);
        }
        Err(BenchmarkError::new(
            "unsupported_benchmark",
            format!("Unknown CPU bandwidth benchmark '{benchmark_id}'."),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gluj_bench_core::{CacheDescriptor, CacheKind};
    use topology::CacheTarget;

    fn cache(level: u8, size_bytes: u64, mask: usize) -> CacheTarget {
        CacheTarget {
            descriptor: CacheDescriptor {
                level,
                kind: CacheKind::Unified,
                size_bytes,
                line_size_bytes: 64,
                sharing_logical_processors: mask.count_ones(),
                instances: 1,
            },
            group: 0,
            mask,
        }
    }

    #[test]
    fn cache_layout_partitions_each_shared_cache_instance() {
        let cores = [0, 2, 4, 6].map(|index| ProcessorLocation { group: 0, index });
        let mut cache_targets = vec![cache(3, 32 * MIB, 0x0f), cache(3, 32 * MIB, 0xf0)];
        cache_targets.extend(
            [0x03, 0x0c, 0x30, 0xc0]
                .into_iter()
                .map(|mask| cache(2, 512 * 1024, mask)),
        );
        let topology = CpuTopology {
            physical_cores: cores.to_vec(),
            core_threads: cores
                .iter()
                .copied()
                .map(|core| {
                    (
                        core,
                        vec![
                            core,
                            ProcessorLocation {
                                group: core.group,
                                index: core.index + 1,
                            },
                        ],
                    )
                })
                .collect(),
            representative: cores[0],
            representative_caches: BTreeMap::new(),
            cache_targets,
            caches: Vec::new(),
        };

        let layout = CpuBandwidthProvider::cache_layout(&topology, 3).unwrap();
        assert_eq!(layout.len(), 4);
        let assigned = layout.iter().map(|(_, bytes)| bytes).sum::<usize>();
        let budget = 2 * 32 * MIB as usize * 2 / 5;
        assert!(assigned <= budget);
        assert!(budget - assigned < layout.len() * 64);
        assert!(layout.iter().all(|(_, bytes)| *bytes > 2 * 512 * 1024));

        let logical = CpuBandwidthProvider::logical_cache_layout(&topology, 3).unwrap();
        assert_eq!(logical.len(), 8);
        let logical_assigned = logical.iter().map(|(_, bytes)| bytes).sum::<usize>();
        assert!(logical_assigned <= assigned);
        assert!(assigned - logical_assigned < logical.len() * 64);
    }

    #[test]
    fn exposes_all_levels_even_when_unavailable() {
        let provider = CpuBandwidthProvider::discover();
        let ids: Vec<_> = provider
            .benchmarks()
            .into_iter()
            .map(|item| item.id)
            .collect();
        assert!(!ids.contains(&"cpu.bandwidth.cache.l0".into()));
        assert!(!ids.contains(&"cpu.performance.avx2.f32_fma".into()));
        assert_eq!(
            ids.iter()
                .filter(|id| id.as_str() == compute_scaling::ID)
                .count(),
            1
        );
        assert!(ids.contains(&"cpu.bandwidth.cache.l1".into()));
        assert!(ids.contains(&"cpu.bandwidth.cache.l3".into()));
        assert!(ids.contains(&"cpu.bandwidth.memory".into()));
        assert!(ids.contains(&"cpu.latency.memory".into()));
        assert!(ids.contains(&"cpu.latency.memory.localized".into()));
        for level in 1..=3 {
            assert!(ids.contains(&format!("cpu.latency.cache.l{level}")));
        }
    }

    #[test]
    fn ram_bandwidth_custom_payload_validates_without_changing_automatic_sizing() {
        let mut provider = CpuBandwidthProvider::discover();
        provider.available_memory = 16 * 1024 * MIB;
        let workers = [0, 1, 2].map(|index| ProcessorLocation { group: 0, index });
        let topology = CpuTopology {
            representative: workers[0],
            physical_cores: workers.to_vec(),
            core_threads: BTreeMap::new(),
            representative_caches: BTreeMap::new(),
            cache_targets: vec![],
            caches: vec![cache(3, 8 * MIB, 7).descriptor],
        };
        let mut config = BenchmarkConfig::default();
        assert_eq!(
            provider
                .configured_ram_payload(&topology, &config, 3, 8 * 1024 * MIB)
                .unwrap(),
            provider.ram_payload(&topology).unwrap()
        );
        config
            .options
            .insert("ram_bandwidth_mode".into(), "custom".into());
        let (minimum, maximum) =
            gluj_bench_core::ram_bandwidth_payload_range(8 * MIB, 8 * 1024 * MIB, 3);
        config
            .options
            .insert("ram_bandwidth_bytes".into(), (minimum + 65).to_string());
        let actual = provider
            .configured_ram_payload(&topology, &config, 3, 8 * 1024 * MIB)
            .unwrap() as u64;
        assert_eq!(actual, minimum);
        assert_eq!(actual % 192, 0);
        config
            .options
            .insert("ram_bandwidth_bytes".into(), (minimum - 1).to_string());
        assert_eq!(
            provider
                .configured_ram_payload(&topology, &config, 3, 8 * 1024 * MIB)
                .unwrap_err()
                .code,
            "invalid_config"
        );
        config
            .options
            .insert("ram_bandwidth_bytes".into(), (maximum + 1).to_string());
        assert_eq!(
            provider
                .configured_ram_payload(&topology, &config, 3, 8 * 1024 * MIB)
                .unwrap_err()
                .code,
            "dataset_exceeds_budget"
        );
        config
            .options
            .insert("ram_bandwidth_bytes".into(), "garbage".into());
        assert!(
            provider
                .configured_ram_payload(&topology, &config, 3, 8 * 1024 * MIB)
                .is_err()
        );
    }
    #[test]
    fn rejects_unstable_sample_counts() {
        let config = BenchmarkConfig {
            samples: 1,
            ..Default::default()
        };
        assert_eq!(
            CpuBandwidthProvider::validate(&config).unwrap_err().code,
            "invalid_config"
        );
    }

    #[test]
    #[cfg(windows)]
    #[ignore = "runs a short custom-size RAM bandwidth benchmark on the current machine"]
    fn ram_bandwidth_custom_smoke() {
        let mut provider = CpuBandwidthProvider::discover();
        let topology = provider.topology.as_ref().unwrap();
        let highest = topology
            .caches
            .iter()
            .map(|cache| cache.level)
            .max()
            .unwrap_or(0);
        let llc = topology
            .caches
            .iter()
            .filter(|cache| cache.level == highest)
            .map(|cache| cache.size_bytes * cache.instances as u64)
            .sum();
        let (minimum, maximum) =
            gluj_bench_core::ram_bandwidth_payload_range(llc, provider.available_memory, 1);
        assert!(minimum <= maximum);
        let config = BenchmarkConfig {
            target_duration_ms: 150,
            samples: 3,
            options: BTreeMap::from([
                ("cpu_core_limit".into(), "1".into()),
                ("ram_bandwidth_mode".into(), "custom".into()),
                ("ram_bandwidth_bytes".into(), minimum.to_string()),
            ]),
        };
        let result = provider
            .run(
                "cpu.bandwidth.memory",
                &config,
                &CancellationToken::default(),
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(
            result
                .metrics
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>(),
            ["read", "write", "copy"]
        );
        assert!(
            result
                .metrics
                .iter()
                .all(|m| m.value.is_finite() && m.value > 0.0 && m.statistics.sample_count == 3)
        );
        assert_eq!(
            result.workload_metadata["actual_payload_bytes"],
            minimum.to_string()
        );
        assert_eq!(
            result.workload_metadata["allocated_test_buffer_bytes"],
            (minimum * 2).to_string()
        );
        assert_eq!(
            result.workload_metadata["copy_byte_definition"],
            "useful_payload"
        );
        assert_eq!(
            result.workload_metadata["cache_state"],
            "flushed_before_each_sweep"
        );
    }
}
