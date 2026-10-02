mod compute;
mod kernels;
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
        config.options.get("thread_mode").map(String::as_str) == Some("logical_processors")
    }

    fn benchmark_locations(
        topology: &CpuTopology,
        config: &BenchmarkConfig,
    ) -> Vec<ProcessorLocation> {
        if Self::logical_mode(config) {
            topology.core_threads.values().flatten().copied().collect()
        } else {
            topology.physical_cores.clone()
        }
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
        let budget = gluj_bench_core::worker_budget(
            layout.len(),
            gluj_bench_core::workload_percent(config, "cpu_worker_percent")?,
        );
        layout.truncate(budget);
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
        let mut locations = Self::benchmark_locations(topology, config);
        let budget = gluj_bench_core::worker_budget(
            locations.len(),
            gluj_bench_core::workload_percent(config, "cpu_worker_percent")?,
        );
        locations.truncate(budget);
        let payload = self
            .ram_payload(topology)
            .map_err(|reason| BenchmarkError::new("benchmark_unavailable", reason))?;
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
        items.extend(compute::descriptors(
            &self.cpu_device,
            self.topology.is_ok(),
        ));
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
        assert!(ids.contains(&"cpu.bandwidth.cache.l1".into()));
        assert!(ids.contains(&"cpu.bandwidth.cache.l3".into()));
        assert!(ids.contains(&"cpu.bandwidth.memory".into()));
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
}
