use crate::{
    kernels::{AlignedBuffer, statistics},
    topology::{AffinityGuard, CpuTopology},
};
use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult,
    CancellationToken, DeviceDescriptor, Metric, ProgressCallback, ProgressUpdate,
};
use std::{collections::BTreeMap, hint::black_box, time::Instant};

pub const ID: &str = "cpu.latency.memory";
pub const LOCAL_ID: &str = "cpu.latency.memory.localized";
pub(super) const LOCAL_BYTES: usize = 64 * 1024;
const MIB: u64 = 1024 * 1024;
#[cfg(test)]
pub(super) const SEED: u64 = 0x6a09_e667_f3bc_c909;
const CHUNK: usize = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    RandomObjects,
    Localized,
}

impl Pattern {
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            ID => Some(Self::RandomObjects),
            LOCAL_ID => Some(Self::Localized),
            _ => None,
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::RandomObjects => ID,
            Self::Localized => LOCAL_ID,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::RandomObjects => "RAM random-object latency",
            Self::Localized => "RAM read latency (localized)",
        }
    }

    pub(super) fn access_order(self) -> &'static str {
        match self {
            Self::RandomObjects => "sattolo_random_single_cycle",
            Self::Localized => "random_within_64k_blocks_sequential_blocks",
        }
    }
}

fn check_cancel(cancellation: &CancellationToken) -> Result<(), BenchmarkError> {
    if cancellation.is_cancelled() {
        Err(BenchmarkError::new(
            "cancelled",
            "RAM latency test cancelled.",
        ))
    } else {
        Ok(())
    }
}

fn layout(topology: &CpuTopology, available: u64) -> Result<(usize, usize, u64), BenchmarkError> {
    let highest = topology.caches.iter().map(|c| c.level).max().unwrap_or(0);
    let llc = topology
        .caches
        .iter()
        .filter(|c| c.level == highest)
        .fold(0u64, |total, c| {
            total.saturating_add(c.size_bytes.saturating_mul(c.instances as u64))
        });
    if llc == 0 {
        return Err(BenchmarkError::new(
            "benchmark_unavailable",
            "RAM latency needs a detected last-level cache size to select a larger working set.",
        ));
    }
    let stride = topology
        .caches
        .iter()
        .map(|c| u64::from(c.line_size_bytes))
        .max()
        .unwrap_or(64)
        .max(64);
    if !stride.is_power_of_two() || stride > 4096 {
        return Err(BenchmarkError::new(
            "benchmark_unavailable",
            "Unsupported cache-line size.",
        ));
    }
    let desired = (256 * MIB).max(llc.saturating_mul(4));
    let bytes = desired.div_ceil(stride).saturating_mul(stride);
    let cap = (available / 16)
        .min(available.saturating_sub(2048 * MIB))
        .min(1024 * MIB);
    // Do not silently turn a RAM test into a cache-sized test on low-memory machines.
    if bytes > cap {
        return Err(BenchmarkError::new(
            "benchmark_unavailable",
            "Insufficient available memory for a RAM latency working set at least four times the detected last-level cache.",
        ));
    }
    Ok((bytes as usize, stride as usize, llc))
}

pub fn descriptor(
    pattern: Pattern,
    device: &DeviceDescriptor,
    topology: Result<&CpuTopology, &BenchmarkError>,
    available: u64,
) -> BenchmarkDescriptor {
    let mut metadata = BTreeMap::from([
        ("metric_direction".into(), "lower_is_better".into()),
        ("thread_count".into(), "1".into()),
        ("latency_profile_revision".into(), "2-seed-ensemble".into()),
    ]);
    metadata.insert("access_order".into(), pattern.access_order().into());
    if pattern == Pattern::Localized {
        metadata.insert("locality_block_bytes".into(), LOCAL_BYTES.to_string());
    }
    let reason = match topology {
        Ok(topology) => match layout(topology, available) {
            Ok((bytes, stride, _)) => {
                metadata.insert("working_set_bytes".into(), bytes.to_string());
                metadata.insert("node_stride_bytes".into(), stride.to_string());
                String::new()
            }
            Err(e) => e.message,
        },
        Err(e) => e.message.clone(),
    };
    BenchmarkDescriptor {
        id: pattern.id().into(),
        name: pattern.name().into(),
        category: BenchmarkCategory::Memory,
        workload: match pattern {
            Pattern::RandomObjects => {
                "Single-core dependent reads of scattered object-like nodes across RAM"
            }
            Pattern::Localized => "Single-core dependent RAM reads randomized within 64 KiB blocks",
        }
        .into(),
        data_type: "dependent memory access".into(),
        unit: "ns".into(),
        supported_device_ids: if reason.is_empty() {
            vec![device.id.clone()]
        } else {
            vec![]
        },
        available: reason.is_empty(),
        unavailable_reason: reason,
        // Keep this initial test in the existing CPU memory category.
        suite_id: "cpu.bandwidth".into(),
        display_order: if pattern == Pattern::Localized { 5 } else { 6 },
        metadata,
    }
}

fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn random_below(state: &mut u64, bound: usize) -> usize {
    let bound = bound as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = random(state);
        if value >= threshold {
            return (value % bound) as usize;
        }
    }
}

pub(super) struct Chain {
    buffer: AlignedBuffer,
    stride: usize,
    nodes: usize,
}

impl Chain {
    #[cfg(test)]
    pub(super) fn new(
        bytes: usize,
        stride: usize,
        pattern: Pattern,
        cancellation: &CancellationToken,
    ) -> Result<Self, BenchmarkError> {
        Self::new_seeded(bytes, stride, pattern, SEED, LOCAL_BYTES, cancellation)
    }
    pub(super) fn new_seeded(
        bytes: usize,
        stride: usize,
        pattern: Pattern,
        seed: u64,
        locality_bytes: usize,
        cancellation: &CancellationToken,
    ) -> Result<Self, BenchmarkError> {
        let _phase_timer = gluj_bench_core::phase_timer("preparation");
        check_cancel(cancellation)?;
        if bytes < stride
            || !bytes.is_multiple_of(stride)
            || locality_bytes < stride
            || !locality_bytes.is_multiple_of(stride)
        {
            return Err(BenchmarkError::new(
                "invalid_config",
                "Pointer-chain size/locality must contain whole cache-line nodes.",
            ));
        }
        let mut buffer = AlignedBuffer::new(bytes)?;
        let nodes = bytes / stride;
        let base = buffer.as_mut_ptr();
        // Sattolo's shuffle creates exactly one cycle visiting every cache-line node.
        // First touch, shuffle and pointer conversion are all outside measurement.
        for i in 0..nodes {
            if i % CHUNK == 0 {
                check_cancel(cancellation)?;
            }
            unsafe {
                base.add(i * stride).cast::<usize>().write(i * stride);
            }
        }
        let mut state = seed;
        let block_nodes = match pattern {
            Pattern::RandomObjects => nodes,
            Pattern::Localized => locality_bytes / stride,
        };
        for begin in (0..nodes).step_by(block_nodes) {
            check_cancel(cancellation)?;
            let end = (begin + block_nodes).min(nodes);
            for i in (begin + 1..end).rev() {
                if i % CHUNK == 0 {
                    check_cancel(cancellation)?;
                }
                let j = begin + random_below(&mut state, i - begin);
                unsafe {
                    std::ptr::swap(
                        base.add(i * stride).cast::<usize>(),
                        base.add(j * stride).cast::<usize>(),
                    );
                }
            }
        }
        if pattern == Pattern::Localized {
            // Join the disjoint block cycles into one cycle without another allocation.
            // Rewire one edge per block to the next block's original successor.
            // Traversal stays within each block until that block has been visited.
            let first_successor = unsafe { base.cast::<usize>().read() };
            for begin in (0..nodes).step_by(block_nodes) {
                check_cancel(cancellation)?;
                let next = begin + block_nodes;
                let successor = if next < nodes {
                    unsafe { base.add(next * stride).cast::<usize>().read() }
                } else {
                    first_successor
                };
                unsafe {
                    base.add(begin * stride).cast::<usize>().write(successor);
                }
            }
        }
        for i in 0..nodes {
            if i % CHUNK == 0 {
                check_cancel(cancellation)?;
            }
            unsafe {
                let node = base.add(i * stride).cast::<*const u8>();
                let offset = node.cast::<usize>().read();
                node.write(base.add(offset));
            }
        }
        Ok(Self {
            buffer,
            stride,
            nodes,
        })
    }

    pub(super) fn traverse(&self, cancellation: &CancellationToken) -> Result<(), BenchmarkError> {
        self.traverse_cycles(1, cancellation)
    }

    pub(super) fn traverse_cycles(
        &self,
        cycles: usize,
        cancellation: &CancellationToken,
    ) -> Result<(), BenchmarkError> {
        let mut cursor = self.buffer.as_ptr();
        let mut left = self.nodes.checked_mul(cycles).ok_or_else(|| {
            BenchmarkError::new("invalid_config", "Too many pointer-chain reads.")
        })?;
        while left > 0 {
            check_cancel(cancellation)?;
            let count = left.min(CHUNK);
            // SAFETY: the internally constructed cycle points only to aligned nodes in buffer.
            cursor = unsafe { chase(cursor, count) };
            left -= count;
        }
        if cursor != self.buffer.as_ptr() {
            return Err(BenchmarkError::new(
                "validation_failed",
                "Pointer chain did not complete its cycle.",
            ));
        }
        black_box(cursor);
        Ok(())
    }
}

#[inline(never)]
unsafe fn chase(mut cursor: *const u8, count: usize) -> *const u8 {
    // Every address depends on the preceding load. Volatile reads retain the work;
    // eight-way unrolling reduces loop overhead without adding independent reads.
    for _ in 0..count / 8 {
        for _ in 0..8 {
            cursor = unsafe { cursor.cast::<*const u8>().read_volatile() };
        }
    }
    for _ in 0..count % 8 {
        cursor = unsafe { cursor.cast::<*const u8>().read_volatile() };
    }
    cursor
}

pub fn run(
    pattern: Pattern,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
    device: &DeviceDescriptor,
    topology: &CpuTopology,
) -> Result<BenchmarkResult, BenchmarkError> {
    let _measurement_scope = gluj_bench_core::MeasurementScope::enter(config, pattern.id());
    check_cancel(cancellation)?;
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let (default_bytes, stride, llc) = layout(
        topology,
        if config.options.contains_key("dataset_bytes") {
            u64::MAX
        } else {
            system.available_memory()
        },
    )?;
    let bytes = latency_bytes(
        config,
        default_bytes,
        stride,
        (system.available_memory() / 16).min(1024 * MIB),
    )?;
    let access = gluj_bench_core::AccessPolicy::from_config(config)?;
    let locality = config
        .options
        .get("locality_block_bytes")
        .and_then(|s| s.parse().ok())
        .unwrap_or(LOCAL_BYTES);
    let location = topology.representative;
    let _affinity = AffinityGuard::pin(location)?;
    progress(ProgressUpdate {
        fraction: 0.0,
        phase: "prepare".into(),
        message: format!("Preparing {} pointer chain", pattern.name()),
    });
    let mut chain = Chain::new_seeded(
        bytes,
        stride,
        pattern,
        access.derived_seed(0, bytes, 0),
        locality,
        cancellation,
    )?;
    progress(ProgressUpdate {
        fraction: 0.05,
        phase: "warmup".into(),
        message: "Warming and validating a full RAM traversal".into(),
    });
    {
        let _warmup = gluj_bench_core::phase_timer("warmup-and-chain-validation");
        chain.traverse(cancellation)?;
    }
    let target = gluj_bench_core::sample_target_ns(config);
    let mut values = Vec::new();
    let mut elapsed_total = 0u64;
    let mut loads_total = 0u64;
    let mut pacer = crate::pacing::CpuPacer::current(cancellation);
    for sample in 0..config.samples * access.realizations {
        let realization = sample / config.samples;
        if sample > 0 && sample.is_multiple_of(config.samples) {
            chain = Chain::new_seeded(
                bytes,
                stride,
                pattern,
                access.derived_seed(0, bytes, realization),
                locality,
                cancellation,
            )?;
            let _warmup = gluj_bench_core::phase_timer("warmup-and-chain-validation");
            chain.traverse(cancellation)?;
        }
        let observation_started = Instant::now();
        progress(ProgressUpdate {
            fraction: 0.1 + 0.9 * f64::from(sample) / f64::from(config.samples),
            phase: "sample".into(),
            message: format!(
                "{}: sample {} of {}",
                pattern.name(),
                sample + 1,
                config.samples
            ),
        });
        let mut elapsed = 0u64;
        let mut loads = 0u64;
        // Complete cycles avoid sampling a small cache-resident subset. Consequently
        // the requested duration is a minimum target, not a strict runtime bound.
        loop {
            check_cancel(cancellation)?;
            let start = Instant::now();
            chain.traverse(cancellation)?;
            let active = start.elapsed();
            elapsed = elapsed.saturating_add(active.as_nanos().min(u128::from(u64::MAX)) as u64);
            loads += chain.nodes as u64;
            pacer.account(active)?;
            if elapsed >= target {
                break;
            }
        }
        values.push(elapsed as f64 / loads as f64);
        if let Some(recorder) = gluj_bench_core::observation_recorder() {
            recorder
                .with_realization(
                    access.derived_seed(0, bytes, realization),
                    BTreeMap::from([
                        ("root_seed".into(), access.seed.to_string()),
                        ("realization".into(), realization.to_string()),
                        (
                            "generator_revision".into(),
                            "sattolo-single-cycle-v2".into(),
                        ),
                    ]),
                )
                .record(
                    &format!("bytes-{bytes}"),
                    "dependent_latency",
                    *values.last().unwrap(),
                    "ns",
                    loads,
                    "dependent_loads",
                    elapsed,
                    observation_started.elapsed().as_nanos() as u64,
                    "validated",
                );
        }
        elapsed_total = elapsed_total.saturating_add(elapsed);
        loads_total = loads_total.saturating_add(loads);
    }
    let statistics = statistics(&values);
    let mut workload_metadata = BTreeMap::from([
        ("result_validation".into(), "constructed single-cycle pointer chain; complete cycle checked during untimed warmup and measured traversals".into()),
        ("working_set_bytes".into(), bytes.to_string()),
        ("detected_aggregate_llc_bytes".into(), llc.to_string()),
        ("node_stride_bytes".into(), chain.stride.to_string()),
        ("node_count".into(), chain.nodes.to_string()),
        ("dependent_load_count".into(), loads_total.to_string()),
        ("thread_count".into(), "1".into()),
        ("thread_mode".into(), "single_pinned_core".into()),
        ("processor_group".into(), location.group.to_string()),
        ("processor_index".into(), location.index.to_string()),
        ("access_order".into(), pattern.access_order().into()),
        ("random_seed".into(), access.seed.to_string()),
        ("latency_profile_revision".into(), "2-seed-ensemble".into()),
        ("metric_direction".into(), "lower_is_better".into()),
        ("preparation_timed".into(), "false".into()),
        ("cache_state".into(), "full_cycle_warmup_no_flush".into()),
        ("page_policy".into(), "ordinary_allocator_pages".into()),
        ("target_duration_ms".into(), config.target_duration_ms.to_string()),
        ("measurement_scope".into(), "CPU-observed dependent read latency including address translation, loop/timer overhead and OS interference; not DRAM CAS timing or guaranteed isolated NUMA latency".into()),
    ]);
    workload_metadata.insert(
        "latency_pattern".into(),
        match pattern {
            Pattern::RandomObjects => "random_objects",
            Pattern::Localized => "localized",
        }
        .into(),
    );
    if pattern == Pattern::Localized {
        workload_metadata.insert("locality_block_bytes".into(), locality.to_string());
    }
    workload_metadata.insert(
        "working_set_label".into(),
        if bytes as u64 <= llc {
            "custom cache-sized system-memory allocation"
        } else {
            "system-memory allocation; residency not measured"
        }
        .into(),
    );
    workload_metadata.insert("access_seed".into(), access.seed.to_string());
    workload_metadata.insert(
        "access_realizations".into(),
        access.realizations.to_string(),
    );
    workload_metadata.insert(
        "access_generator_revision".into(),
        "sattolo-single-cycle-v2".into(),
    );
    workload_metadata.insert("statistical_summary".into(),"equal observations per seed; pooled descriptive median; separate native realization summaries".into());
    progress(ProgressUpdate {
        fraction: 1.0,
        phase: "complete".into(),
        message: format!("{} complete", pattern.name()),
    });
    Ok(BenchmarkResult {
        benchmark_id: pattern.id().into(),
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

    fn topology(llc: u64) -> CpuTopology {
        use crate::topology::ProcessorLocation;
        let core = ProcessorLocation { group: 0, index: 0 };
        CpuTopology {
            physical_cores: vec![core],
            core_threads: BTreeMap::new(),
            representative: core,
            representative_caches: BTreeMap::new(),
            cache_targets: vec![],
            caches: vec![gluj_bench_core::CacheDescriptor {
                level: 3,
                kind: gluj_bench_core::CacheKind::Unified,
                size_bytes: llc,
                line_size_bytes: 64,
                sharing_logical_processors: 1,
                instances: 1,
            }],
        }
    }

    #[test]
    fn sizing_preserves_ram_scope_and_memory_headroom() {
        assert_eq!(
            layout(&topology(32 * MIB), 8 * 1024 * MIB).unwrap(),
            (256 * MIB as usize, 64, 32 * MIB)
        );
        assert_eq!(
            layout(&topology(96 * MIB), 8 * 1024 * MIB).unwrap().0,
            384 * MIB as usize
        );
        assert!(layout(&topology(32 * MIB), 3 * 1024 * MIB).is_err());
        assert!(layout(&topology(512 * MIB), 64 * 1024 * MIB).is_err());
        assert!(layout(&topology(0), 8 * 1024 * MIB).is_err());
    }

    #[test]
    fn randomized_chain_visits_every_node_once() {
        for nodes in [2, 3, 17, 128, 257] {
            let chain = Chain::new(
                nodes * 64,
                64,
                Pattern::RandomObjects,
                &CancellationToken::default(),
            )
            .unwrap();
            let base = chain.buffer.as_ptr();
            let mut cursor = base;
            let mut visited = vec![false; nodes];
            for _ in 0..nodes {
                let offset = unsafe { cursor.offset_from(base) } as usize;
                assert_eq!(offset % 64, 0);
                let index = offset / 64;
                assert!(index < nodes && !visited[index]);
                visited[index] = true;
                cursor = unsafe { chase(cursor, 1) };
            }
            assert_eq!(cursor, base);
            assert!(visited.into_iter().all(|v| v));
            chain.traverse(&CancellationToken::default()).unwrap();
            chain
                .traverse_cycles(257, &CancellationToken::default())
                .unwrap();
        }
    }

    #[test]
    fn preparation_and_traversal_honor_cancellation() {
        let chain = Chain::new(
            4096,
            64,
            Pattern::RandomObjects,
            &CancellationToken::default(),
        )
        .unwrap();
        let token = CancellationToken::default();
        token.cancel();
        assert!(Chain::new(4096, 64, Pattern::RandomObjects, &token).is_err());
        assert_eq!(chain.traverse(&token).unwrap_err().code, "cancelled");
    }

    #[test]
    fn localized_chain_covers_full_working_set_with_one_transition_per_block() {
        for nodes in [2, 1024, 1025, 3072, 3203] {
            let chain = Chain::new(
                nodes * 64,
                64,
                Pattern::Localized,
                &CancellationToken::default(),
            )
            .unwrap();
            let base = chain.buffer.as_ptr();
            let mut cursor = base;
            let mut visited = vec![false; nodes];
            let mut transitions = 0;
            for _ in 0..nodes {
                let offset = unsafe { cursor.offset_from(base) } as usize;
                assert_eq!(offset % 64, 0);
                let index = offset / 64;
                assert!(index < nodes && !visited[index]);
                visited[index] = true;
                let next = unsafe { chase(cursor, 1) };
                let next_offset = unsafe { next.offset_from(base) } as usize;
                if offset / LOCAL_BYTES != next_offset / LOCAL_BYTES {
                    transitions += 1;
                }
                cursor = next;
            }
            let blocks = nodes.div_ceil(LOCAL_BYTES / 64);
            assert_eq!(transitions, if blocks > 1 { blocks } else { 0 });
            assert_eq!(cursor, base);
            assert!(visited.into_iter().all(|v| v));
            chain.traverse(&CancellationToken::default()).unwrap();
            let cancelled = CancellationToken::default();
            cancelled.cancel();
            assert!(Chain::new(nodes * 64, 64, Pattern::Localized, &cancelled).is_err());
            assert_eq!(chain.traverse(&cancelled).unwrap_err().code, "cancelled");
        }
    }
}

pub(super) fn latency_bytes(
    config: &BenchmarkConfig,
    default: usize,
    stride: usize,
    cap: u64,
) -> Result<usize, BenchmarkError> {
    let bytes = config
        .options
        .get("dataset_bytes")
        .map(|s| s.parse::<u64>())
        .transpose()
        .map_err(|_| BenchmarkError::new("invalid_config", "Invalid latency working set."))?
        .unwrap_or(default as u64);
    if bytes < stride as u64
        || bytes > cap
        || bytes > usize::MAX as u64
        || !bytes.is_multiple_of(stride as u64)
    {
        return Err(BenchmarkError::new(
            "invalid_config",
            "Latency size must contain whole nodes and fit the available-memory headroom.",
        ));
    }
    Ok(bytes as usize)
}

#[cfg(test)]
mod seed_regressions {
    use super::*;
    #[test]
    fn ensemble_chains_preserve_one_cycle_with_custom_locality_and_size() {
        let token = CancellationToken::default();
        for seed in [1, 2, 3] {
            for pattern in [Pattern::RandomObjects, Pattern::Localized] {
                let chain = Chain::new_seeded(64 * 101, 64, pattern, seed, 64 * 7, &token).unwrap();
                let base = chain.buffer.as_ptr();
                let mut cursor = base;
                let mut seen = std::collections::BTreeSet::new();
                for _ in 0..101 {
                    assert!(seen.insert(cursor as usize - base as usize));
                    cursor = unsafe { cursor.cast::<*const u8>().read() };
                }
                assert_eq!(cursor, base);
                assert_eq!(seen.len(), 101);
                chain.traverse_cycles(2, &token).unwrap();
            }
        }
        let config = BenchmarkConfig {
            options: BTreeMap::from([("dataset_bytes".into(), "100".into())]),
            ..Default::default()
        };
        assert!(latency_bytes(&config, 4096, 64, 8192).is_err());
    }
}
