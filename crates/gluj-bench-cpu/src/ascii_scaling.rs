use crate::{
    compute, compute_scaling,
    topology::{CpuTopology, ProcessorLocation},
};
use gluj_bench_core::{
    BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult, CancellationToken,
    DeviceDescriptor, Metric, ProgressCallback, ProgressUpdate,
};
use std::{collections::BTreeMap, hint::black_box};

pub const ID: &str = "cpu.performance.string.ascii";
const RECORD_BYTES: u64 = 64;
const FALLBACK_BYTES: u64 = 64 * 1024;
const PATTERN: &[u8] = b"The quick brown fox searches these ASCII records for the letter e.";

pub(crate) struct Buffers {
    input: Vec<u8>,
    hits: u64,
}
impl Buffers {
    pub(crate) fn new(
        bytes: usize,
        cancellation: &CancellationToken,
    ) -> Result<Self, BenchmarkError> {
        let mut input = Vec::new();
        input.try_reserve_exact(bytes).map_err(|_| {
            BenchmarkError::new("allocation_failed", "Could not allocate the ASCII input.")
        })?;
        for start in (0..bytes).step_by(1024 * 1024) {
            if cancellation.is_cancelled() {
                return Err(BenchmarkError::new(
                    "cancelled",
                    "ASCII input preparation cancelled.",
                ));
            }
            input.extend(
                (start..bytes.min(start + 1024 * 1024)).map(|index| PATTERN[index % PATTERN.len()]),
            );
        }
        Ok(Self { input, hits: 0 })
    }
    pub(crate) fn sweep(&mut self, cancellation: &CancellationToken) -> bool {
        let mut hits = 0u64;
        // Keep cancellation responsive even with multi-GB inputs. Chunk boundaries
        // are record aligned and do not change the scan operation or byte count.
        for chunk in self.input.chunks(1024 * 1024) {
            if cancellation.is_cancelled() {
                return false;
            }
            hits += compute::scan(black_box(chunk)).1;
        }
        self.hits = black_box(hits);
        true
    }
    pub(crate) fn checksum(&self) -> u64 {
        black_box(self.hits)
    }
}

fn cache_bytes(
    topology: &CpuTopology,
    workers: &[ProcessorLocation],
    level: u8,
) -> Result<u64, BenchmarkError> {
    compute_scaling::cache_dataset_aligned(topology, workers, level, RECORD_BYTES)
}
fn automatic_bytes(topology: &CpuTopology, workers: &[ProcessorLocation]) -> u64 {
    cache_bytes(topology, workers, 2).unwrap_or(FALLBACK_BYTES * workers.len() as u64)
}
fn instruction_path() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return "avx2_byte_compare";
    }
    "scalar_byte_compare"
}
pub fn descriptor(
    device: &DeviceDescriptor,
    topology: Option<&CpuTopology>,
) -> BenchmarkDescriptor {
    let mut item = compute::descriptors(device, topology.is_some())
        .into_iter()
        .find(|item| item.id == ID)
        .unwrap();
    item.workload = "ASCII character counting with automatic or fully custom input sizes".into();
    item.metadata
        .insert("kernel_revision".into(), "ascii-scan-1".into());
    item.metadata.insert(
        "operation_definition".into(),
        "one 64-byte ASCII record scanned for the letter e".into(),
    );
    item.metadata
        .insert("profile_axis".into(), "aggregate_working_set_bytes".into());
    if let Some(topology) = topology {
        for level in 1..=3 {
            let sizes = (1..=topology.physical_cores.len())
                .map(|count| {
                    cache_bytes(topology, &topology.physical_cores[..count], level)
                        .map(|bytes| bytes.to_string())
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>()
                .join(",");
            item.metadata
                .insert(format!("cache_l{level}_dataset_sizes"), sizes);
        }
        let sizes = (1..=topology.physical_cores.len())
            .map(|count| automatic_bytes(topology, &topology.physical_cores[..count]).to_string())
            .collect::<Vec<_>>()
            .join(",");
        item.metadata
            .insert("automatic_compute_dataset_sizes".into(), sizes);
    }
    item
}
fn dataset(
    config: &BenchmarkConfig,
    limit: u64,
    topology: &CpuTopology,
    workers: &[ProcessorLocation],
) -> Result<(u64, bool), BenchmarkError> {
    let preset = config
        .options
        .get("ascii_preset")
        .map(String::as_str)
        .unwrap_or("compute");
    if preset == "legacy" {
        return gluj_bench_core::configured_dataset(config, limit);
    }
    let mode = config
        .options
        .get("dataset_mode")
        .map(String::as_str)
        .unwrap_or("single");
    if !["single", "sweep", "automatic"].contains(&mode) {
        return Err(BenchmarkError::new(
            "invalid_config",
            "ASCII dataset mode must be single or sweep.",
        ));
    }
    let requested = match preset {
        "compute" => automatic_bytes(topology, workers).min(limit),
        "ram" => limit,
        "l1" | "l2" | "l3" => cache_bytes(topology, workers, preset.as_bytes()[1] - b'0')?,
        "custom" => config
            .options
            .get("dataset_bytes")
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| {
                BenchmarkError::new(
                    "invalid_config",
                    "Custom ASCII input size must be a positive byte count.",
                )
            })?,
        _ => {
            return Err(BenchmarkError::new(
                "invalid_config",
                "Unknown ASCII input preset.",
            ));
        }
    };
    let alignment = RECORD_BYTES * workers.len() as u64;
    if requested < alignment {
        return Err(BenchmarkError::new(
            "invalid_config",
            "ASCII input must fit at least one 64-byte record per worker.",
        ));
    }
    if requested > limit {
        return Err(BenchmarkError::new(
            "dataset_exceeds_budget",
            format!(
                "Requested ASCII input is {requested} bytes; the RAM budget allows {limit} bytes."
            ),
        ));
    }
    Ok((requested, mode == "single"))
}
fn sizes(maximum: u64, workers: usize, single: bool) -> Vec<u64> {
    let alignment = RECORD_BYTES * workers as u64;
    let maximum = maximum / alignment * alignment;
    if maximum < alignment {
        return vec![];
    }
    if single {
        return vec![maximum];
    }
    let mut result = Vec::new();
    let mut bytes = (4096 * workers as u64).min(maximum);
    while bytes < maximum {
        result.push(bytes);
        bytes = bytes.saturating_mul(2);
    }
    result.push(maximum);
    result
}
pub fn run(
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
    device: &DeviceDescriptor,
    topology: &CpuTopology,
) -> Result<BenchmarkResult, BenchmarkError> {
    let workers = crate::topology::selected_workers(topology, config, false)?;
    if workers.is_empty() {
        return Err(BenchmarkError::new(
            "topology_unavailable",
            "No CPU workers available.",
        ));
    }
    let percent = config
        .options
        .get("ram_budget_percent")
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|_| {
            BenchmarkError::new("invalid_option", "RAM budget must be a whole percentage.")
        })?
        .unwrap_or(20);
    if !(20..=80).contains(&percent) {
        return Err(BenchmarkError::new(
            "invalid_option",
            "RAM budget must be between 20% and 80%.",
        ));
    }
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let limit = gluj_bench_core::ram_allocation_budget(
        system.total_memory(),
        system.available_memory(),
        percent,
    );
    let (requested, single) = dataset(config, limit, topology, &workers)?;
    let tiers = sizes(requested, workers.len(), single);
    if tiers.is_empty() {
        return Err(BenchmarkError::new(
            "insufficient_memory",
            "Not enough RAM for the ASCII input.",
        ));
    }
    let mut metrics = Vec::new();
    let mut elapsed_ns = 0;
    let mut checksum = 0;
    for (index, bytes) in tiers.iter().enumerate() {
        progress(ProgressUpdate {
            fraction: index as f64 / tiers.len() as f64,
            phase: "ascii_scan".into(),
            message: format!(
                "Scanning {bytes} bytes of ASCII input ({}/{})",
                index + 1,
                tiers.len()
            ),
        });
        let (records, bandwidth, elapsed, hash) = compute_scaling::measure(
            &workers,
            *bytes,
            config,
            cancellation,
            compute_scaling::Kernel::Ascii,
        )?;
        elapsed_ns += elapsed;
        checksum ^= hash;
        for (suffix, values, unit) in [
            ("compute", records, "strings/s"),
            ("bandwidth", bandwidth, "bytes/s"),
        ] {
            let stats = crate::kernels::statistics(&values);
            metrics.push(Metric {
                name: format!("working_set_{bytes}.{suffix}"),
                value: stats.median,
                unit: unit.into(),
                statistics: stats,
            });
        }
    }
    let record_tiers = metrics
        .iter()
        .filter(|m| m.name.ends_with(".compute"))
        .collect::<Vec<_>>();
    let baseline = record_tiers
        .iter()
        .take(3)
        .map(|m| m.value)
        .fold(0.0_f64, f64::max);
    let transition = compute_scaling::transition_index(&record_tiers, baseline);
    let resolved = *tiers.last().unwrap();
    let preset = config
        .options
        .get("ascii_preset")
        .map(String::as_str)
        .unwrap_or("compute");
    let mut metadata = BTreeMap::from([
        ("ascii_preset".into(), preset.into()),
        ("dataset_mode".into(), if single { "single" } else { "sweep" }.into()),
        ("kernel_revision".into(), "ascii-scan-1".into()),
        ("harness_revision".into(), "4".into()),
        ("data_type".into(), "64-byte ASCII record".into()),
        ("operation_definition".into(), "one 64-byte ASCII record scanned for the letter e".into()),
        ("record_bytes".into(), RECORD_BYTES.to_string()),
        ("input_pattern".into(), "fixed_ascii_sentence_v1".into()),
        ("instruction_path".into(), instruction_path().into()),
        ("working_set_definition".into(), "aggregate ASCII input buffers across selected workers; no output arrays".into()),
        ("byte_definition".into(), "input bytes read once per complete scan; effective payload, not physical RAM traffic".into()),
        ("thread_count".into(), workers.len().to_string()),
        ("thread_mode".into(), if !crate::topology::explicit_core_limit(config) && config.options.get("thread_mode").is_some_and(|mode| mode == "logical_processors") { "logical_processors" } else { "physical_cores" }.into()),
        ("affinity".into(), "one_worker_per_selected_processor".into()),
        ("cpu_activity_percent".into(), gluj_bench_core::workload_percent(config, "cpu_activity_percent")?.to_string()),
        ("ram_budget_percent".into(), percent.to_string()),
        ("allocation_budget_bytes".into(), limit.to_string()),
        ("available_memory_bytes_at_run".into(), system.available_memory().to_string()),
        ("requested_dataset_bytes".into(), requested.to_string()),
        ("resolved_dataset_bytes".into(), resolved.to_string()),
        ("allocated_test_buffer_bytes".into(), resolved.to_string()),
        ("aggregate_working_set_bytes".into(), resolved.to_string()),
        ("per_thread_working_set_bytes".into(), (resolved / workers.len() as u64).to_string()),
        ("tested_tier_count".into(), tiers.len().to_string()),
        ("profile_sample_count_per_tier".into(), config.samples.to_string()),
        ("target_duration_ms_per_tier".into(), config.target_duration_ms.to_string()),
        ("warmup_ms_per_tier".into(), "100".into()),
        ("buffer_ownership".into(), "pinned_thread_first_touch; disjoint inputs".into()),
        ("baseline_operations_per_second".into(), baseline.to_string()),
        ("bandwidth_transition_status".into(), if transition.is_some() { "observed" } else { "not_observed_within_tested_range" }.into()),
        ("bound_classification".into(), if transition.is_some() { "memory_sensitive" } else { "not_established" }.into()),
        ("classification_is_inference".into(), "true".into()),
        ("tuning_guidance".into(), "Compare this fixed scan at matching input sizes and worker counts. A sustained slowdown indicates input-size sensitivity; cache residency and memory-stall time are not directly measured.".into()),
        ("checksum".into(), checksum.to_string()),
    ]);
    if let Some(index) = transition {
        metadata.insert(
            "bandwidth_transition_working_set_bytes".into(),
            tiers[index].to_string(),
        );
        metadata.insert(
            "bandwidth_transition_retained_ratio".into(),
            (record_tiers[index].value / baseline).to_string(),
        );
    }
    if matches!(preset, "l1" | "l2" | "l3") {
        metadata.insert("cache_capacity_fraction".into(), "0.75".into());
    }
    progress(ProgressUpdate {
        fraction: 1.0,
        phase: "complete".into(),
        message: "ASCII scan complete".into(),
    });
    Ok(BenchmarkResult {
        benchmark_id: ID.into(),
        device_id: device.id.clone(),
        elapsed_ns,
        metrics,
        workload_metadata: metadata,
        device_metadata: BTreeMap::from([("name".into(), device.name.clone())]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn topology() -> CpuTopology {
        let workers = vec![
            ProcessorLocation { group: 0, index: 0 },
            ProcessorLocation { group: 0, index: 1 },
        ];
        CpuTopology {
            representative: workers[0],
            core_threads: workers.iter().map(|w| (*w, vec![*w])).collect(),
            physical_cores: workers,
            representative_caches: BTreeMap::new(),
            caches: vec![],
            cache_targets: vec![crate::topology::CacheTarget {
                group: 0,
                mask: 3,
                descriptor: gluj_bench_core::CacheDescriptor {
                    level: 2,
                    kind: gluj_bench_core::CacheKind::Unified,
                    size_bytes: 1024 * 1024,
                    line_size_bytes: 64,
                    sharing_logical_processors: 2,
                    instances: 1,
                },
            }],
        }
    }
    #[test]
    fn scan_counts_ascii_identically_across_sizes_and_chunk_boundaries() {
        let cancellation = CancellationToken::default();
        for bytes in [64, 128, 4096, 1024 * 1024 + 64] {
            let mut buffers = Buffers::new(bytes, &cancellation).unwrap();
            assert!(buffers.input.is_ascii());
            let expected = buffers.input.iter().filter(|byte| **byte == b'e').count() as u64;
            assert!(buffers.sweep(&cancellation));
            assert_eq!(buffers.checksum(), expected);
            assert_eq!(compute::scan(&buffers.input), (bytes as u64 / 64, expected));
            assert!(buffers.sweep(&cancellation));
            assert_eq!(buffers.checksum(), expected);
        }
        cancellation.cancel();
        assert!(Buffers::new(1024, &cancellation).is_err());
    }
    #[test]
    fn presets_share_caches_validate_requests_and_align_complete_records() {
        let topology = topology();
        let workers = &topology.physical_cores;
        let mut config = BenchmarkConfig::default();
        assert_eq!(
            dataset(&config, 1024 * 1024, &topology, workers).unwrap(),
            (768 * 1024, true)
        );
        assert_eq!(automatic_bytes(&topology, &workers[..1]), 768 * 1024);
        assert_eq!(
            dataset(&config, 65537, &topology, workers).unwrap(),
            (65537, true)
        );
        assert_eq!(sizes(65537, 2, true), vec![65536]);
        let mut absent = topology.clone();
        absent.cache_targets.clear();
        assert_eq!(automatic_bytes(&absent, workers), 128 * 1024);
        config
            .options
            .insert("ascii_preset".into(), "custom".into());
        config.options.insert("dataset_bytes".into(), "129".into());
        assert_eq!(
            dataset(&config, 4096, &topology, workers).unwrap(),
            (129, true)
        );
        assert_eq!(sizes(129, 2, true), vec![128]);
        config.options.insert("dataset_bytes".into(), "127".into());
        assert!(dataset(&config, 4096, &topology, workers).is_err());
        config.options.insert("dataset_bytes".into(), "4097".into());
        assert_eq!(
            dataset(&config, 4096, &topology, workers).unwrap_err().code,
            "dataset_exceeds_budget"
        );
        config.options.insert("ascii_preset".into(), "l1".into());
        assert_eq!(
            dataset(&config, 1024 * 1024, &topology, workers)
                .unwrap_err()
                .code,
            "cache_unavailable"
        );
        config
            .options
            .insert("ascii_preset".into(), "register".into());
        assert!(dataset(&config, 1024 * 1024, &topology, workers).is_err());
        config.options.insert("ascii_preset".into(), "ram".into());
        config.options.insert("dataset_mode".into(), "sweep".into());
        assert_eq!(
            dataset(&config, 65537, &topology, workers).unwrap(),
            (65537, false)
        );
        let tiers = sizes(65537, 2, false);
        assert!(tiers.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(tiers.iter().all(|bytes| bytes % 128 == 0));
        assert_eq!(tiers.last(), Some(&65536));
    }
    #[test]
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[ignore = "runs short ASCII scan benchmarks on the current machine"]
    fn ascii_modes_smoke() {
        let mut registry = gluj_bench_core::BenchmarkRegistry::default();
        registry.add(Box::new(crate::CpuBandwidthProvider::discover()));
        let mut config = BenchmarkConfig {
            target_duration_ms: 150,
            samples: 3,
            ..BenchmarkConfig::default()
        };
        config.options.insert("cpu_core_limit".into(), "1".into());
        config
            .options
            .insert("ram_budget_percent".into(), "20".into());
        config
            .options
            .insert("dataset_bytes".into(), "65537".into());
        for (preset, mode) in [
            ("compute", "single"),
            ("custom", "single"),
            ("custom", "sweep"),
            ("l1", "single"),
        ] {
            config.options.insert("ascii_preset".into(), preset.into());
            config.options.insert("dataset_mode".into(), mode.into());
            let result = registry
                .run(ID, &config, &CancellationToken::default(), &mut |_| {})
                .unwrap();
            assert_eq!(result.workload_metadata["kernel_revision"], "ascii-scan-1");
            assert_eq!(result.workload_metadata["ascii_preset"], preset);
            let records = result
                .metrics
                .iter()
                .filter(|m| m.name.ends_with(".compute"))
                .collect::<Vec<_>>();
            assert!(!records.is_empty());
            if mode == "single" {
                assert_eq!(records.len(), 1);
            }
            for record in records {
                assert_eq!(record.unit, "strings/s");
                assert!(record.value > 0.0);
                let traffic = result
                    .metrics
                    .iter()
                    .find(|m| m.name == record.name.replace(".compute", ".bandwidth"))
                    .unwrap();
                assert!((traffic.value / record.value - 64.0).abs() < 1e-6);
                assert_eq!(record.statistics.sample_count, 3);
            }
            assert!(
                !result
                    .metrics
                    .iter()
                    .any(|m| m.name == "measured_compute_ceiling")
            );
        }
    }
}
