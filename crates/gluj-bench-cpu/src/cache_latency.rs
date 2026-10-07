use crate::{
    kernels::statistics,
    latency::{Chain, LOCAL_BYTES, Pattern, SEED},
    topology::{AffinityGuard, CpuTopology},
};
use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult,
    CancellationToken, DeviceDescriptor, Metric, ProgressCallback, ProgressUpdate,
};
use std::{collections::BTreeMap, time::Instant};

const MIN_BATCH_LOADS: usize = 65536;

pub fn level(id: &str) -> Option<u8> {
    match id {
        "cpu.latency.cache.l1" => Some(1),
        "cpu.latency.cache.l2" => Some(2),
        "cpu.latency.cache.l3" => Some(3),
        _ => None,
    }
}

struct CacheLayout {
    bytes: usize,
    stride: usize,
    capacity: u64,
    lower_capacity: u64,
    group: u16,
    mask: usize,
    sharing: u32,
}

fn layout(topology: &CpuTopology, level: u8) -> Result<CacheLayout, BenchmarkError> {
    let unavailable = |message| BenchmarkError::new("benchmark_unavailable", message);
    if !(1..=3).contains(&level) {
        return Err(unavailable(
            "Only L1, L2 and L3 cache latency are supported.",
        ));
    }
    let target = topology.representative_caches.get(&level).ok_or_else(|| {
        unavailable("This cache level was not detected for the pinned processor core.")
    })?;
    let capacity = target.descriptor.size_bytes;
    let stride = u64::from(target.descriptor.line_size_bytes).max(64);
    if !stride.is_power_of_two() || stride > 4096 {
        return Err(unavailable("Unsupported cache-line size."));
    }
    let lower_capacity = if level > 1 {
        topology.representative_caches.get(&(level - 1))
            .filter(|c| c.descriptor.size_bytes > 0)
            .ok_or_else(|| unavailable("The preceding cache level's capacity is unknown; level separation cannot be established."))?
            .descriptor.size_bytes
    } else {
        0
    };
    let ceiling = capacity / 4 * 3;
    // Prefer a modest footprint beyond the preceding level, leaving room for
    // concurrent system traffic in shared caches.
    let desired = if level == 1 {
        ceiling
    } else {
        lower_capacity.saturating_mul(4).min(ceiling)
    };
    let bytes = desired / stride * stride;
    if bytes < 2 * stride || bytes <= lower_capacity.saturating_mul(2) {
        return Err(unavailable(
            "Insufficient cache-level separation: the working set must fit in the target cache and exceed twice the preceding level's capacity.",
        ));
    }
    Ok(CacheLayout {
        bytes: usize::try_from(bytes)
            .map_err(|_| unavailable("Cache working set is too large."))?,
        stride: stride as usize,
        capacity,
        lower_capacity,
        group: target.group,
        mask: target.mask,
        sharing: target.descriptor.sharing_logical_processors,
    })
}

fn metadata(layout: &CacheLayout, level: u8) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("cache_level".into(), level.to_string()),
        ("target_cache_bytes".into(), layout.capacity.to_string()),
        ("cache_processor_group".into(), layout.group.to_string()),
        ("cache_processor_mask".into(), format!("{:x}", layout.mask)),
        (
            "cache_sharing_logical_processors".into(),
            layout.sharing.to_string(),
        ),
        (
            "preceding_cache_bytes".into(),
            layout.lower_capacity.to_string(),
        ),
        (
            "cache_capacity_fraction".into(),
            format!("{:.6}", layout.bytes as f64 / layout.capacity as f64),
        ),
        ("working_set_bytes".into(), layout.bytes.to_string()),
        ("node_stride_bytes".into(), layout.stride.to_string()),
        ("thread_count".into(), "1".into()),
        ("metric_direction".into(), "lower_is_better".into()),
        ("latency_profile_revision".into(), "1".into()),
    ])
}

pub fn descriptor(
    level: u8,
    device: &DeviceDescriptor,
    topology: Result<&CpuTopology, &BenchmarkError>,
) -> BenchmarkDescriptor {
    let mut properties = BTreeMap::new();
    let reason = match topology {
        Ok(topology) => match layout(topology, level) {
            Ok(layout) => {
                properties = metadata(&layout, level);
                String::new()
            }
            Err(problem) => problem.message,
        },
        Err(problem) => problem.message.clone(),
    };
    BenchmarkDescriptor {
        id: format!("cpu.latency.cache.l{level}"),
        name: format!("L{level} cache read latency"),
        category: BenchmarkCategory::Cpu,
        workload: "Single-core warmed dependent pointer reads in a cache-sized working set".into(),
        data_type: "dependent memory access".into(),
        unit: "ns".into(),
        supported_device_ids: if reason.is_empty() {
            vec![device.id.clone()]
        } else {
            vec![]
        },
        available: reason.is_empty(),
        unavailable_reason: reason,
        suite_id: "cpu.bandwidth".into(),
        display_order: 6 + u32::from(level),
        metadata: properties,
    }
}

fn check_cancel(token: &CancellationToken) -> Result<(), BenchmarkError> {
    if token.is_cancelled() {
        Err(BenchmarkError::new(
            "cancelled",
            "Cache latency test cancelled.",
        ))
    } else {
        Ok(())
    }
}

fn batch_cycles(nodes: usize) -> usize {
    MIN_BATCH_LOADS.div_ceil(nodes).max(1)
}

pub fn run(
    level: u8,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
    device: &DeviceDescriptor,
    topology: &CpuTopology,
) -> Result<BenchmarkResult, BenchmarkError> {
    check_cancel(cancellation)?;
    let layout = layout(topology, level)?;
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    if layout.bytes as u64 > system.available_memory() / 16 {
        return Err(BenchmarkError::new(
            "benchmark_unavailable",
            "Insufficient available memory for this cache working set.",
        ));
    }
    let location = topology.representative;
    let _affinity = AffinityGuard::pin(location)?;
    progress(ProgressUpdate {
        fraction: 0.0,
        phase: "prepare".into(),
        message: format!("Preparing L{level} cache pointer chain"),
    });
    let pattern = if level == 1 {
        Pattern::RandomObjects
    } else {
        Pattern::Localized
    };
    let chain = Chain::new(layout.bytes, layout.stride, pattern, cancellation)?;
    progress(ProgressUpdate {
        fraction: 0.05,
        phase: "warmup".into(),
        message: format!("Warming L{level} cache with two complete traversals"),
    });
    chain.traverse_cycles(2, cancellation)?;
    let nodes = layout.bytes / layout.stride;
    let cycles = batch_cycles(nodes);
    let batch_loads = nodes * cycles;
    let target_ns = config.target_duration_ms.saturating_mul(1_000_000) / u64::from(config.samples);
    let mut values = Vec::new();
    let mut elapsed_total = 0u64;
    let mut loads_total = 0u64;
    for sample in 0..config.samples {
        progress(ProgressUpdate {
            fraction: 0.1 + 0.9 * f64::from(sample) / f64::from(config.samples),
            phase: "sample".into(),
            message: format!(
                "L{level} cache latency: sample {} of {}",
                sample + 1,
                config.samples
            ),
        });
        let mut elapsed = 0u64;
        let mut loads = 0u64;
        while elapsed < target_ns || loads == 0 {
            check_cancel(cancellation)?;
            let start = Instant::now();
            chain.traverse_cycles(cycles, cancellation)?;
            elapsed =
                elapsed.saturating_add(start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
            loads = loads.saturating_add(batch_loads as u64);
        }
        values.push(elapsed as f64 / loads as f64);
        elapsed_total = elapsed_total.saturating_add(elapsed);
        loads_total = loads_total.saturating_add(loads);
    }
    let statistics = statistics(&values);
    let mut workload_metadata = metadata(&layout, level);
    for (key, value) in [
        ("thread_mode", "single_pinned_core".into()),
        ("processor_group", location.group.to_string()),
        ("processor_index", location.index.to_string()),
        ("node_count", nodes.to_string()),
        ("dependent_load_count", loads_total.to_string()),
        ("batch_loads", batch_loads.to_string()),
        ("random_seed", SEED.to_string()),
        ("access_order", pattern.access_order().into()),
        ("page_policy", "ordinary_allocator_pages".into()),
        ("cache_state", "two_full_cycle_warmups_no_flush".into()),
        ("preparation_timed", "false".into()),
        ("target_duration_ms", config.target_duration_ms.to_string()),
        ("measurement_scope", "Observed cache-sized dependent-read latency; level residency is inferred from topology and working set, not confirmed by hardware counters. Includes translation, prefetch/cache effects, timing/loop overhead and system interference.".into()),
    ] { workload_metadata.insert(key.into(), value); }
    if level > 1 {
        workload_metadata.insert("locality_block_bytes".into(), LOCAL_BYTES.to_string());
    }
    progress(ProgressUpdate {
        fraction: 1.0,
        phase: "complete".into(),
        message: format!("L{level} cache read latency complete"),
    });
    Ok(BenchmarkResult {
        benchmark_id: format!("cpu.latency.cache.l{level}"),
        device_id: device.id.clone(),
        elapsed_ns: elapsed_total,
        metrics: vec![Metric {
            name: "read_latency".into(),
            value: statistics.median,
            unit: "ns".into(),
            statistics,
        }],
        workload_metadata,
        device_metadata: BTreeMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{CacheTarget, ProcessorLocation};
    use gluj_bench_core::{CacheDescriptor, CacheKind};

    #[test]
    #[ignore = "measures cache residency on the current machine"]
    fn cache_residency_probe() {
        let topology = crate::topology::discover().unwrap();
        let _affinity = AffinityGuard::pin(topology.representative).unwrap();
        let token = CancellationToken::default();
        for bytes in [
            1024 * 1024,
            2 * 1024 * 1024,
            4 * 1024 * 1024,
            8 * 1024 * 1024,
            16 * 1024 * 1024,
            24 * 1024 * 1024,
        ] {
            let chain = Chain::new(bytes, 64, Pattern::Localized, &token).unwrap();
            chain.traverse_cycles(4, &token).unwrap();
            let cycles = batch_cycles(bytes / 64);
            let mut elapsed = 0;
            let mut reads = 0;
            while elapsed < 300_000_000u64 {
                let start = Instant::now();
                chain.traverse_cycles(cycles, &token).unwrap();
                elapsed += start.elapsed().as_nanos() as u64;
                reads += bytes / 64 * cycles;
            }
            println!(
                "{bytes} bytes: {:.3} ns/read",
                elapsed as f64 / reads as f64
            );
        }
    }

    fn topology(sizes: [u64; 3]) -> CpuTopology {
        let core = ProcessorLocation { group: 0, index: 0 };
        let caches = sizes
            .into_iter()
            .enumerate()
            .map(|(i, size)| {
                (
                    i as u8 + 1,
                    CacheTarget {
                        descriptor: CacheDescriptor {
                            level: i as u8 + 1,
                            kind: CacheKind::Unified,
                            size_bytes: size,
                            line_size_bytes: 64,
                            sharing_logical_processors: 1,
                            instances: 1,
                        },
                        group: 0,
                        mask: 1,
                    },
                )
            })
            .collect();
        CpuTopology {
            physical_cores: vec![core],
            core_threads: BTreeMap::new(),
            representative: core,
            representative_caches: caches,
            cache_targets: vec![],
            caches: vec![],
        }
    }

    #[test]
    fn sizing_uses_one_cache_instance_and_separates_preceding_levels() {
        let topology = topology([32 * 1024, 1024 * 1024, 32 * 1024 * 1024]);
        for level in 1..=3 {
            let layout = layout(&topology, level).unwrap();
            assert!(layout.bytes as u64 <= layout.capacity * 3 / 4);
            assert!(layout.bytes as u64 > 2 * layout.lower_capacity);
            assert_eq!(layout.bytes % layout.stride, 0);
            let nodes = layout.bytes / layout.stride;
            assert!(batch_cycles(nodes) * nodes >= MIN_BATCH_LOADS);
        }
        assert_eq!(layout(&topology, 1).unwrap().bytes, 24 * 1024);
        assert_eq!(layout(&topology, 2).unwrap().bytes, 128 * 1024);
        assert_eq!(layout(&topology, 3).unwrap().bytes, 4 * 1024 * 1024);
    }

    #[test]
    fn missing_or_overlapping_levels_have_explanatory_errors() {
        let mut topology = topology([64 * 1024, 128 * 1024, 1024 * 1024]);
        assert!(layout(&topology, 2).is_err());
        topology.representative_caches.remove(&3);
        assert!(layout(&topology, 3).is_err());
        topology.representative_caches.remove(&1);
        assert!(layout(&topology, 2).is_err());
        assert!(level("cpu.latency.cache.l4").is_none());
    }

    #[test]
    fn aggregate_cache_capacity_is_not_used_for_the_pinned_core() {
        let mut topology = topology([32 * 1024, 512 * 1024, 32 * 1024 * 1024]);
        let mut aggregate = topology.representative_caches[&3].descriptor.clone();
        aggregate.instances = 2;
        topology.caches.push(aggregate);
        let layout = layout(&topology, 3).unwrap();
        assert_eq!(layout.capacity, 32 * 1024 * 1024);
        assert_eq!(layout.bytes, 2 * 1024 * 1024);
        assert_eq!(metadata(&layout, 3)["cache_processor_mask"], "1");
    }
}
