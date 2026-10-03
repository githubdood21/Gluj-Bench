use crate::{
    compute,
    kernels::statistics,
    topology::{AffinityGuard, CpuTopology, ProcessorLocation},
};
use gluj_bench_core::{
    BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult, CancellationToken,
    DeviceDescriptor, Metric, ProgressCallback, ProgressUpdate,
};
use std::{
    collections::BTreeMap,
    hint::black_box,
    sync::{
        Barrier,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
};

pub const ID: &str = "cpu.performance.avx2.f32_fma.scaling";
const REFERENCE: &str = "cpu.performance.avx2.f32_fma";
const REUSE: u64 = 16;
const BLOCK: usize = 64;
const BYTES_PER_ELEMENT: u64 = 12;
const MIB: u64 = 1024 * 1024;

pub fn matrix_descriptor(
    device: &DeviceDescriptor,
    topology_available: bool,
) -> BenchmarkDescriptor {
    let mut descriptor = descriptor(device, topology_available);
    descriptor.id = crate::matrix_scaling::ID.into();
    descriptor.name = "AVX2 FP32 matrix compute scaling".into();
    descriptor.workload =
        "Cache-blocked FP32 matrix multiplication: 32 input rows and increasing weight matrices"
            .into();
    descriptor.display_order = 12;
    descriptor
        .metadata
        .insert("kernel_revision".into(), "avx2-matrix-1".into());
    descriptor
}

enum WorkBuffers {
    Vector(Buffers),
    Matrix(crate::matrix_scaling::Buffers),
}
impl WorkBuffers {
    fn new(elements: usize, worker: usize, matrix: bool) -> Result<Self, BenchmarkError> {
        if matrix {
            Ok(Self::Matrix(crate::matrix_scaling::Buffers::new(
                elements, worker,
            )?))
        } else {
            Ok(Self::Vector(Buffers::new(elements, worker)?))
        }
    }
    fn sweep(&mut self, cancellation: &CancellationToken) -> bool {
        match self {
            Self::Vector(buffers) => {
                buffers.sweep();
                true
            }
            Self::Matrix(buffers) => buffers.sweep(cancellation),
        }
    }
    fn checksum(&self) -> u64 {
        match self {
            Self::Matrix(buffers) => buffers.checksum(),
            Self::Vector(buffers) => {
                u64::from(black_box(buffers.output[0]).to_bits())
                    ^ u64::from(black_box(*buffers.output.last().unwrap()).to_bits())
            }
        }
    }
}

pub fn descriptor(device: &DeviceDescriptor, topology_available: bool) -> BenchmarkDescriptor {
    let mut descriptor = compute::descriptors(device, topology_available)
        .into_iter()
        .find(|d| d.id == REFERENCE)
        .expect("AVX2 reference is registered");
    descriptor.id = ID.into();
    descriptor.name = "AVX2 FP32 vector compute scaling".into();
    descriptor.workload =
        "AVX2/FMA throughput over two input arrays and one output array, from cache to RAM".into();
    descriptor.display_order = 11;
    descriptor.metadata = BTreeMap::from([
        ("profile_axis".into(), "aggregate_working_set_bytes".into()),
        ("kernel_revision".into(), "avx2-streaming-1".into()),
        ("data_type".into(), "fp32".into()),
        ("compute_reference_benchmark".into(), REFERENCE.into()),
        (
            "operation_definition".into(),
            "one FP32 multiply or add; FMA counts as two per lane".into(),
        ),
    ]);
    descriptor
}

fn sizes(maximum: u64, workers: usize) -> Vec<u64> {
    let alignment = BLOCK as u64 * BYTES_PER_ELEMENT * workers as u64;
    let maximum = maximum / alignment * alignment;
    let mut size = 1024 * BYTES_PER_ELEMENT * workers as u64;
    let mut result = Vec::new();
    while size < maximum {
        result.push(size);
        size = size.saturating_mul(2);
    }
    if maximum >= 1024 * BYTES_PER_ELEMENT * workers as u64 {
        result.push(maximum);
    }
    result
}

struct Buffers {
    a: Vec<f32>,
    b: Vec<f32>,
    output: Vec<f32>,
}
impl Buffers {
    fn new(elements: usize, worker: usize) -> Result<Self, BenchmarkError> {
        fn array(elements: usize, value: f32) -> Result<Vec<f32>, BenchmarkError> {
            let mut data = Vec::new();
            data.try_reserve_exact(elements)
                .map_err(|e| BenchmarkError::new("allocation_failed", e.to_string()))?;
            data.resize(elements, value);
            Ok(data)
        }
        Ok(Self {
            a: array(elements, 1.0 + worker as f32 * 0.001)?,
            b: array(elements, 0.999_999_94)?,
            output: array(elements, 0.0)?,
        })
    }
    fn sweep(&mut self) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: run() gates AVX2/FMA, arrays have equal lengths divisible by 64,
        // and disjoint buffers are exclusively owned by this pinned worker.
        unsafe {
            vector_sweep(&self.a, &self.b, &mut self.output);
        }
        black_box(&self.output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vector_sweep(a: &[f32], b: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};
    let addend = _mm256_set1_ps(0.000_001);
    for offset in (0..a.len()).step_by(BLOCK) {
        // Eight independent vectors avoid measuring a single dependency chain.
        let mut values = [_mm256_set1_ps(0.0); 8];
        let mut multipliers = values;
        for (lane, value) in values.iter_mut().enumerate() {
            *value = unsafe { _mm256_loadu_ps(a.as_ptr().add(offset + lane * 8)) };
            multipliers[lane] = unsafe { _mm256_loadu_ps(b.as_ptr().add(offset + lane * 8)) };
        }
        for _ in 0..REUSE {
            for (lane, value) in values.iter_mut().enumerate() {
                *value = _mm256_fmadd_ps(*value, multipliers[lane], addend);
            }
        }
        for (lane, value) in values.iter().enumerate() {
            unsafe {
                _mm256_storeu_ps(output.as_mut_ptr().add(offset + lane * 8), *value);
            }
        }
    }
}

fn measure(
    locations: &[ProcessorLocation],
    bytes: u64,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    matrix: bool,
) -> Result<(Vec<f64>, Vec<f64>, u64, u64), BenchmarkError> {
    let elements = if matrix {
        crate::matrix_scaling::dimension_for_bytes(bytes / locations.len() as u64)
    } else {
        usize::try_from(bytes / locations.len() as u64 / BYTES_PER_ELEMENT).map_err(|_| {
            BenchmarkError::new(
                "allocation_too_large",
                "Working set exceeds the address range.",
            )
        })?
    };
    let operations_per_sweep = if matrix {
        crate::matrix_scaling::operations(elements)
    } else {
        elements as u64 * 2 * REUSE
    };
    let traffic_per_sweep = if matrix {
        crate::matrix_scaling::traffic(elements)
    } else {
        elements as u64 * BYTES_PER_ELEMENT
    };
    let barrier = Barrier::new(locations.len());
    let failed = AtomicBool::new(false);
    let duration_ns = config.target_duration_ms.saturating_mul(1_000_000) / config.samples as u64;
    let results = thread::scope(|scope| {
        let handles = locations
            .iter()
            .enumerate()
            .map(|(index, location)| {
                let (barrier, failed) = (&barrier, &failed);
                scope.spawn(move || -> Result<Vec<(u64, u64, u64)>, BenchmarkError> {
                    let prepared = (|| {
                        let affinity = AffinityGuard::pin(*location)?;
                        let buffers = WorkBuffers::new(elements, index, matrix)?;
                        Ok::<_, BenchmarkError>((affinity, buffers))
                    })();
                    if prepared.is_err() {
                        failed.store(true, Ordering::SeqCst);
                    }
                    // Every worker reaches this barrier even on allocation/affinity failure.
                    barrier.wait();
                    if failed.load(Ordering::SeqCst) {
                        return Err(prepared.err().unwrap_or_else(|| {
                            BenchmarkError::new(
                                "worker_prepare_failed",
                                "Another CPU worker could not allocate memory or set affinity.",
                            )
                        }));
                    }
                    let (_affinity, mut buffers) = prepared?;
                    let warmup = Instant::now();
                    while warmup.elapsed().as_millis() < 100 && !cancellation.is_cancelled() {
                        if !buffers.sweep(cancellation) {
                            break;
                        }
                    }
                    let mut samples = Vec::new();
                    for _ in 0..config.samples {
                        barrier.wait();
                        let start = Instant::now();
                        let mut sweeps = 0_u64;
                        while start.elapsed().as_nanos() < duration_ns as u128
                            && !cancellation.is_cancelled()
                        {
                            if !buffers.sweep(cancellation) {
                                break;
                            }
                            sweeps += 1;
                        }
                        let elapsed = start.elapsed().as_nanos().max(1) as u64;
                        barrier.wait();
                        let checksum = buffers.checksum();
                        samples.push((sweeps, elapsed, checksum));
                    }
                    Ok(samples)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    BenchmarkError::new("worker_failed", "CPU scaling worker panicked.")
                })?
            })
            .collect::<Result<Vec<_>, BenchmarkError>>()
    })?;
    if cancellation.is_cancelled() {
        return Err(BenchmarkError::new(
            "cancelled",
            "CPU scaling test cancelled.",
        ));
    }
    let mut compute = Vec::new();
    let mut bandwidth = Vec::new();
    let mut total_elapsed = 0;
    let mut checksum = 0;
    for index in 0..config.samples as usize {
        let sweeps = results.iter().map(|s| s[index].0).sum::<u64>();
        let elapsed = results.iter().map(|s| s[index].1).max().unwrap().max(1);
        compute.push(sweeps as f64 * operations_per_sweep as f64 * 1e9 / elapsed as f64);
        bandwidth.push(sweeps as f64 * traffic_per_sweep as f64 * 1e9 / elapsed as f64);
        total_elapsed += elapsed;
        checksum ^= results
            .iter()
            .map(|s| s[index].2)
            .fold(0, u64::wrapping_add);
    }
    Ok((compute, bandwidth, total_elapsed, checksum))
}

pub fn run(
    id: &str,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
    device: &DeviceDescriptor,
    topology: &CpuTopology,
) -> Result<BenchmarkResult, BenchmarkError> {
    let matrix = id == crate::matrix_scaling::ID;
    let descriptor = if matrix {
        matrix_descriptor(device, true)
    } else {
        descriptor(device, true)
    };
    if !descriptor.available {
        return Err(BenchmarkError::new(
            "benchmark_unavailable",
            descriptor.unavailable_reason,
        ));
    }
    let locations = crate::topology::selected_workers(topology, config, true)?;
    if locations.is_empty() {
        return Err(BenchmarkError::new(
            "topology_unavailable",
            "No CPU workers available.",
        ));
    }
    let requested_mib = config
        .options
        .get("ram_budget_mib")
        .map(|s| s.parse::<u64>())
        .transpose()
        .map_err(|_| {
            BenchmarkError::new("invalid_option", "ram_budget_mib must be a whole number.")
        })?
        .unwrap_or(1024);
    if !(4..=4096).contains(&requested_mib) {
        return Err(BenchmarkError::new(
            "invalid_option",
            "ram_budget_mib must be between 4 and 4096.",
        ));
    }
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let available = system.available_memory();
    let ram_percent = config
        .options
        .get("ram_budget_percent")
        .map(|s| s.parse::<u32>())
        .transpose()
        .map_err(|_| {
            BenchmarkError::new(
                "invalid_option",
                "ram_budget_percent must be a whole number.",
            )
        })?;
    if ram_percent.is_some_and(|p| !(20..=80).contains(&p)) {
        return Err(BenchmarkError::new(
            "invalid_option",
            "RAM budget must be between 20% and 80%.",
        ));
    }
    let maximum = allocation_budget(system.total_memory(), available, ram_percent, requested_mib);

    let tiers = if matrix {
        crate::matrix_scaling::sizes(maximum, locations.len())
    } else {
        sizes(maximum, locations.len())
    };
    if tiers.len() < 3 {
        return Err(BenchmarkError::new(
            "insufficient_memory",
            "Not enough free RAM for a CPU scaling sweep.",
        ));
    }
    let mut ref_config = config.clone();
    ref_config.target_duration_ms = config.target_duration_ms.clamp(150, 500);
    let mut quiet = |_| {};
    progress(ProgressUpdate {
        fraction: 0.0,
        phase: "cpu_compute_reference".into(),
        message: "Measuring register-resident AVX2 FP32 throughput".into(),
    });
    let initial = compute::run(
        REFERENCE,
        &ref_config,
        cancellation,
        &mut quiet,
        topology.representative,
        device,
        topology,
    )?;
    let mut metrics = Vec::new();
    let mut elapsed = 0;
    let mut checksum = 0;
    for (index, bytes) in tiers.iter().enumerate() {
        progress(ProgressUpdate {
            fraction: 0.05 + 0.85 * index as f64 / tiers.len() as f64,
            phase: if matrix {
                "cpu_matrix_scaling"
            } else {
                "cpu_vector_scaling"
            }
            .into(),
            message: format!(
                "AVX2 FP32: {:.2} MiB total, {:.2} KiB per worker ({}/{})",
                *bytes as f64 / MIB as f64,
                *bytes as f64 / locations.len() as f64 / 1024.0,
                index + 1,
                tiers.len()
            ),
        });
        let (compute, bandwidth, time, hash) =
            measure(&locations, *bytes, config, cancellation, matrix)?;
        elapsed += time;
        checksum ^= hash;
        for (suffix, samples, unit) in [
            ("compute", compute, "operations/s"),
            ("bandwidth", bandwidth, "bytes/s"),
        ] {
            let stats = statistics(&samples);
            metrics.push(Metric {
                name: format!("working_set_{bytes}.{suffix}"),
                value: stats.median,
                unit: unit.into(),
                statistics: stats,
            });
        }
    }
    progress(ProgressUpdate {
        fraction: 0.92,
        phase: "cpu_compute_reference".into(),
        message: "Checking current AVX2 FP32 compute reference".into(),
    });
    let check = compute::run(
        REFERENCE,
        &ref_config,
        cancellation,
        &mut quiet,
        topology.representative,
        device,
        topology,
    )?;
    let mut reference = check.metrics[0].clone();
    reference.name = "measured_compute_ceiling".into();
    let reference_value = reference.value;
    let drift = ((reference_value / initial.metrics[0].value - 1.0) * 100.0).abs();
    let compute_tiers: Vec<_> = metrics
        .iter()
        .filter(|m| m.name.ends_with(".compute"))
        .collect();
    let baseline = compute_tiers
        .iter()
        .take(3)
        .map(|m| m.value)
        .fold(0.0_f64, f64::max);
    let transition = transition_index(&compute_tiers, baseline);
    let largest = compute_tiers.last().unwrap().value;
    let mut metadata = BTreeMap::from([
        ("compute_reference_benchmark".into(), REFERENCE.into()),
        ("compute_reference_selection".into(), "latest_post_sweep".into()),
        ("compute_reference_kind".into(), "measured_register_resident; not_theoretical_peak".into()),
        ("compute_reference_drift_percent".into(), format!("{drift:.2}")),
        ("kernel_revision".into(), "avx2-streaming-1".into()),
        ("arithmetic_iterations".into(), REUSE.to_string()),
        ("data_type".into(), "fp32".into()),
        ("operation_definition".into(), "one floating-point multiply or add per lane; each FMA counts as two".into()),
        ("profile_sample_count_per_tier".into(), config.samples.to_string()),
        ("target_duration_ms_per_tier".into(), config.target_duration_ms.to_string()),
        ("warmup_ms_per_tier".into(), "100".into()),
        ("arithmetic_intensity_operations_per_byte".into(), format!("{:.4}", (2 * REUSE) as f64 / BYTES_PER_ELEMENT as f64)),
        ("thread_count".into(), locations.len().to_string()),
        ("thread_mode".into(), if crate::topology::explicit_core_limit(config) || config.options.get("thread_mode").is_some_and(|s| s == "physical_cores") { "physical_cores" } else { "logical_processors" }.into()),
        ("cpu_worker_percent".into(), gluj_bench_core::workload_percent(config, "cpu_worker_percent")?.to_string()),
        ("affinity".into(), "one_worker_per_selected_logical_processor".into()),
        ("buffer_ownership".into(), "pinned_thread_first_touch; disjoint arrays".into()),
        ("instruction_path".into(), "x86_avx2_fma".into()),
        ("byte_definition".into(), "two_fp32_reads_plus_one_fp32_write; effective payload, excludes write allocation and cache-line writeback".into()),
        ("working_set_definition".into(), "aggregate of two input arrays and one output array across selected workers".into()),
        ("ram_budget_mib".into(), requested_mib.to_string()),
        ("available_memory_bytes_at_run".into(), available.to_string()),
        ("allocation_budget_bytes".into(), maximum.to_string()),
        ("allocated_test_buffer_bytes".into(), tiers.last().unwrap().to_string()),
        ("per_thread_working_set_bytes".into(), (tiers.last().unwrap() / locations.len() as u64).to_string()),
        ("allocation_limit_note".into(), "Percentage requests use installed RAM, capped at 80% of available RAM with at least 2 GiB left available. Legacy requests use ram_budget_mib and one eighth of available RAM.".into()),
        ("tested_tier_count".into(), tiers.len().to_string()),
        ("baseline_operations_per_second".into(), baseline.to_string()),
        ("largest_compute_delta_percent".into(), format!("{:.2}", (largest / reference_value - 1.0) * 100.0)),
        ("bandwidth_transition_status".into(), if transition.is_some() { "observed" } else { "not_observed_within_tested_range" }.into()),
        ("bound_classification".into(), if transition.is_some() { "memory_bandwidth_bound" } else { "not_established" }.into()),
        ("classification_is_inference".into(), "true".into()),
        ("transition_threshold".into(), "two consecutive tiers below 80% of small-data baseline, each with <=10% variation, with no later two-tier recovery".into()),
        ("tuning_guidance".into(), "Compare raw TOPS and effective GB/s across dataset sizes. A sustained slowdown suggests memory pressure for this kernel; it is not a direct measurement of memory-stall time. Repeat noisy runs using the same CPU worker allocation. No clock setting was changed or recommended.".into()),
        ("checksum".into(), checksum.to_string()),
    ]);
    if let Some(percent) = ram_percent {
        metadata.remove("ram_budget_mib");
        metadata.insert("ram_budget_percent".into(), percent.to_string());
    }
    if crate::topology::explicit_core_limit(config) {
        metadata.insert(
            "allocated_physical_cores".into(),
            locations.len().to_string(),
        );
    }
    // Memory-pressure evidence motivates a trial; it cannot predict optimal parallelism.
    let stable = |m: &Metric| {
        m.value.is_finite()
            && m.value > 0.0
            && m.statistics.sample_count >= 2
            && m.statistics.standard_deviation.is_finite()
            && m.statistics.standard_deviation.abs() / m.value <= 0.10
    };
    if transition.is_some()
        && largest < baseline * 0.85
        && stable(&reference)
        && compute_tiers
            .iter()
            .take(3)
            .max_by(|a, b| a.value.total_cmp(&b.value))
            .is_some_and(|m| stable(m))
        && locations.len() > 1
        && drift <= 10.0
        && compute_tiers.last().is_some_and(|m| {
            m.statistics.sample_count >= 2 && m.statistics.standard_deviation / m.value <= 0.10
        })
        && crate::topology::explicit_core_limit(config)
    {
        let trial = (locations.len() * 3 / 4).max(1);
        metadata.insert("suggested_cpu_core_count_trial".into(), trial.to_string());
        metadata.insert(
            "cpu_core_trial_baseline_operations_per_second".into(),
            largest.to_string(),
        );
        metadata.insert("tuning_guidance".into(), format!("Try {trial} physical cores instead of {} and retest the same largest aggregate dataset. Keep the change only if throughput stays within 0–5% of this run or improves; restore more cores if loss exceeds 5%. Memory pressure motivates this exploratory trial, but fewer cores have not been measured. Power and temperatures are not measured. For another process, set its worker count or affinity manually and verify its own workload; this benchmark cannot predict its benefit. Matrix dimensions per worker change with the worker count, so inspect the recorded shapes when comparing. No process settings or clocks were changed.", locations.len()));
    } else {
        metadata.insert("tuning_guidance".into(), "No measured optimal core count. Select an exact physical-core allocation and compare repeated runs at the same aggregate dataset size. Fewer workers may reduce contention, power use or heat, but can also lower throughput. Power and temperatures are not measured; application benefits require testing that application's workload. No process affinity or clock setting was changed.".into());
    }
    if let Some(index) = transition {
        metadata.insert(
            "bandwidth_transition_working_set_bytes".into(),
            tiers[index].to_string(),
        );
        metadata.insert(
            "bandwidth_transition_retained_ratio".into(),
            format!("{:.4}", compute_tiers[index].value / baseline),
        );
    }
    if matrix {
        let n = crate::matrix_scaling::dimension_for_bytes(
            *tiers.last().unwrap() / locations.len() as u64,
        );
        metadata.remove("arithmetic_iterations");
        metadata.insert("kernel_revision".into(), "avx2-matrix-1".into());
        metadata.insert("matrix_m".into(), crate::matrix_scaling::ROWS.to_string());
        metadata.insert("matrix_n".into(), n.to_string());
        metadata.insert("matrix_k".into(), n.to_string());
        metadata.insert(
            "matrix_blocking".into(),
            "8x8 AVX2 microtiles; 64 columns and 128 K values per cache block".into(),
        );
        metadata.insert(
            "operation_definition".into(),
            "2*M*N*K FP32 operations per complete matrix product; FMA counts as two per lane"
                .into(),
        );
        metadata.insert("byte_definition".into(), "effective kernel accesses: scalar A reads, vector B reads, C writes per K block and C reads after the first K block; cache reuse means this is not physical RAM traffic".into());
        metadata.insert(
            "arithmetic_intensity_operations_per_byte".into(),
            format!(
                "{:.4}",
                crate::matrix_scaling::operations(n) as f64
                    / crate::matrix_scaling::traffic(n) as f64
            ),
        );
        metadata.insert("working_set_definition".into(), "aggregate A[32,K], B[K,N], C[32,N] across selected workers; K=N increases with dataset size".into());
        metadata.insert(
            "matrix_profile_shapes".into(),
            tiers
                .iter()
                .map(|bytes| {
                    let n =
                        crate::matrix_scaling::dimension_for_bytes(*bytes / locations.len() as u64);
                    format!("{bytes}:32x{n}x{n}")
                })
                .collect::<Vec<_>>()
                .join(","),
        );
        metadata.get_mut("tuning_guidance").unwrap().push_str(" Matrix GB/s counts effective accesses inside the blocked kernel, including cache reuse; it is not physical RAM-bus utilization. Compare this workload with its own small-data baseline.");
    }
    metrics.insert(0, reference);
    progress(ProgressUpdate {
        fraction: 1.0,
        phase: "complete".into(),
        message: if matrix {
            "AVX2 FP32 matrix scaling complete"
        } else {
            "AVX2 FP32 vector scaling complete"
        }
        .into(),
    });
    Ok(BenchmarkResult {
        benchmark_id: id.into(),
        device_id: device.id.clone(),
        elapsed_ns: elapsed,
        metrics,
        workload_metadata: metadata,
        device_metadata: BTreeMap::from([("name".into(), device.name.clone())]),
    })
}

fn transition_index(tiers: &[&Metric], baseline: f64) -> Option<usize> {
    let stable = |m: &Metric| {
        m.value.is_finite()
            && m.value > 0.0
            && m.statistics.sample_count >= 2
            && m.statistics.standard_deviation.is_finite()
            && m.statistics.standard_deviation.abs() / m.value <= 0.10
    };
    tiers.windows(2).enumerate().find_map(|(index, pair)| {
        let low = pair.iter().all(|m| stable(m) && m.value < baseline * 0.80);
        let recovery = tiers[index + 2..]
            .windows(2)
            .any(|pair| pair.iter().all(|m| stable(m) && m.value >= baseline * 0.90));
        (low && !recovery).then_some(index)
    })
}

fn allocation_budget(total: u64, available: u64, percent: Option<u32>, legacy_mib: u64) -> u64 {
    match percent {
        Some(p) => (total / 100 * p as u64)
            .min(available / 5 * 4)
            .min(available.saturating_sub(2 * 1024 * MIB)),
        None => (legacy_mib * MIB).min(available / 8),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentage_ram_budget_uses_installed_memory_and_preserves_available_headroom() {
        let gib = 1024 * MIB;
        assert_eq!(
            allocation_budget(64 * gib, 60 * gib, Some(20), 1024),
            64 * gib / 100 * 20
        );
        assert_eq!(
            allocation_budget(64 * gib, 10 * gib, Some(80), 1024),
            8 * gib
        );
        assert_eq!(allocation_budget(64 * gib, 3 * gib, Some(80), 1024), gib);
        assert_eq!(allocation_budget(64 * gib, gib, Some(80), 1024), 0);
        assert_eq!(allocation_budget(64 * gib, 60 * gib, None, 1024), gib);
        for matrix in [false, true] {
            let budget = allocation_budget(64 * gib, 60 * gib, Some(20), 1024);
            let tiers = if matrix {
                crate::matrix_scaling::sizes(budget, 4)
            } else {
                sizes(budget, 4)
            };
            assert!(
                *tiers.last().unwrap() > 4 * gib,
                "percentage budget must bypass the legacy 4 GiB limit"
            );
            assert!(*tiers.last().unwrap() <= budget);
        }
    }
    #[test]
    fn sweep_sizes_are_aligned_cover_ram_and_respect_limits() {
        for workers in [1, 12, 32, 64] {
            let tiers = sizes(1024 * MIB, workers);
            assert!(tiers.windows(2).all(|p| p[0] < p[1]));
            assert_eq!(tiers[0] / workers as u64, 12 * 1024);
            assert!(*tiers.last().unwrap() <= 1024 * MIB);
            assert!(*tiers.last().unwrap() > 1000 * MIB);
            assert!(
                tiers
                    .iter()
                    .all(|bytes| bytes % (workers as u64 * BLOCK as u64 * BYTES_PER_ELEMENT) == 0)
            );
        }
    }
    #[test]
    fn vector_kernel_matches_scalar_fma_and_operation_accounting() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut data = Buffers::new(BLOCK * 3, 0).unwrap();
            for (i, v) in data.a.iter_mut().enumerate() {
                *v += i as f32 * 0.001;
            }
            data.sweep();
            for (index, value) in data.output.iter().enumerate() {
                let mut expected = data.a[index];
                for _ in 0..REUSE {
                    // These FP32 operands have an exact product and sum in FP64.
                    // Round once to FP32 instead of relying on MinGW's fmaf runtime.
                    expected = (f64::from(expected) * f64::from(data.b[index])
                        + f64::from(0.000_001_f32)) as f32;
                }
                assert_eq!(*value, expected);
            }
            assert_eq!(
                (BLOCK as u64 * 2 * REUSE) as f64 / (BLOCK as u64 * BYTES_PER_ELEMENT) as f64,
                32.0 / 12.0
            );
        }
    }
    #[test]
    fn transition_requires_consistent_sustained_drop_without_recovery() {
        let make = |value| Metric {
            name: "compute".into(),
            value,
            unit: "operations/s".into(),
            statistics: gluj_bench_core::SampleStatistics {
                sample_count: 5,
                standard_deviation: value * 0.01,
                ..Default::default()
            },
        };
        let mut tiers = vec![make(100.0), make(100.0), make(60.0), make(55.0)];
        assert_eq!(
            transition_index(&tiers.iter().collect::<Vec<_>>(), 100.0),
            Some(2)
        );
        tiers[2].statistics.standard_deviation = 20.0;
        assert_eq!(
            transition_index(&tiers.iter().collect::<Vec<_>>(), 100.0),
            None
        );
        tiers[2] = make(60.0);
        tiers.extend([make(100.0), make(100.0)]);
        assert_eq!(
            transition_index(&tiers.iter().collect::<Vec<_>>(), 100.0),
            None
        );
    }
}
