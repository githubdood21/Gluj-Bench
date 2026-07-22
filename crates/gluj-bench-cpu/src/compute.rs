use crate::{
    kernels::statistics,
    topology::{AffinityGuard, CpuTopology, ProcessorLocation},
};
use flate2::{Compress, Compression, Decompress, FlushCompress, FlushDecompress};
use gluj_bench_core::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult,
    CancellationToken, DeviceDescriptor, Metric, ProgressCallback, ProgressUpdate,
};
use miniz_oxide::{deflate::compress_to_vec, inflate::decompress_to_vec};
use std::{
    collections::BTreeMap,
    hint::black_box,
    sync::{Arc, Barrier},
    thread,
    time::Instant,
};

const SUITE_ID: &str = "cpu.performance";
const PROBE_NS: u64 = 250_000_000;
const PROBE_SAMPLES: u32 = 3;
const WARMUP_NS: u64 = 200_000_000;
const MEMORY_BOUND_RATIO: f64 = 0.80;
const STRING_BYTES: usize = 64;
const PRIME_SEARCH_LIMIT: usize = 1_000_000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Workload {
    Integer,
    Float32,
    Float64,
    String,
    Prime,
    Compress,
    Decompress,
    Popcnt,
    Aes,
    Avx2Fma,
    Avx2Fma64,
    SingleThread,
}

struct Definition {
    id: &'static str,
    name: &'static str,
    workload: Workload,
    data_type: &'static str,
    operation: &'static str,
    order: u32,
}

const DEFINITIONS: &[Definition] = &[
    Definition {
        id: "cpu.performance.integer.i64",
        name: "64-bit integer multiply-add",
        workload: Workload::Integer,
        data_type: "i64",
        operation: "one 64-bit integer multiply or add",
        order: 0,
    },
    Definition {
        id: "cpu.performance.float.f32",
        name: "32-bit floating point",
        workload: Workload::Float32,
        data_type: "f32",
        operation: "scalar multiply and add",
        order: 1,
    },
    Definition {
        id: "cpu.performance.float.f64",
        name: "64-bit floating point",
        workload: Workload::Float64,
        data_type: "f64",
        operation: "scalar multiply and add",
        order: 2,
    },
    Definition {
        id: "cpu.performance.string.ascii",
        name: "ASCII string scan",
        workload: Workload::String,
        data_type: "64-byte ASCII record",
        operation: "one fixed 64-byte ASCII string scanned",
        order: 3,
    },
    Definition {
        id: "cpu.performance.prime.sieve",
        name: "Prime number sieve",
        workload: Workload::Prime,
        data_type: "integers 2..=1,000,000",
        operation: "one prime number found by the Sieve of Eratosthenes",
        order: 4,
    },
    Definition {
        id: "cpu.performance.compression.deflate",
        name: "DEFLATE compression",
        workload: Workload::Compress,
        data_type: "u8",
        operation: "one uncompressed input byte processed",
        order: 5,
    },
    Definition {
        id: "cpu.performance.decompression.deflate",
        name: "DEFLATE decompression",
        workload: Workload::Decompress,
        data_type: "u8",
        operation: "one decompressed output byte produced",
        order: 6,
    },
    Definition {
        id: "cpu.performance.extended.popcnt",
        name: "POPCNT throughput",
        workload: Workload::Popcnt,
        data_type: "u64",
        operation: "one 64-bit population-count instruction",
        order: 7,
    },
    Definition {
        id: "cpu.performance.extended.aes",
        name: "AES round throughput",
        workload: Workload::Aes,
        data_type: "128-bit block",
        operation: "one hardware AES round instruction",
        order: 8,
    },
    Definition {
        id: "cpu.performance.avx2.f32_fma",
        name: "AVX2/FMA floating point",
        workload: Workload::Avx2Fma,
        data_type: "8xf32",
        operation: "one floating-point add or multiply (FMA counts as two)",
        order: 9,
    },
    Definition {
        id: "cpu.performance.avx2.f64_fma",
        name: "AVX2/FMA 64-bit floating point",
        workload: Workload::Avx2Fma64,
        data_type: "4xf64",
        operation: "one floating-point add or multiply (FMA counts as two)",
        order: 10,
    },
    Definition {
        id: "cpu.performance.single_thread.integer.i64",
        name: "Single-thread 64-bit integer multiply-add",
        workload: Workload::SingleThread,
        data_type: "i64",
        operation: "one 64-bit integer multiply or add",
        order: 11,
    },
];

fn available(workload: Workload) -> (bool, &'static str) {
    #[cfg(target_arch = "x86_64")]
    {
        match workload {
            Workload::Popcnt if !std::is_x86_feature_detected!("popcnt") => {
                (false, "popcnt_not_supported")
            }
            Workload::Aes if !std::is_x86_feature_detected!("aes") => (false, "aes_not_supported"),
            Workload::Avx2Fma | Workload::Avx2Fma64
                if !(std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("fma")) =>
            {
                (false, "avx2_fma_not_supported")
            }
            _ => (true, ""),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        match workload {
            Workload::Popcnt | Workload::Aes | Workload::Avx2Fma | Workload::Avx2Fma64 => {
                (false, "x86_64_instruction_set_required")
            }
            _ => (true, ""),
        }
    }
}

pub fn descriptors(
    device: &DeviceDescriptor,
    topology_available: bool,
) -> Vec<BenchmarkDescriptor> {
    DEFINITIONS
        .iter()
        .map(|definition| {
            let (instruction_available, instruction_reason) = available(definition.workload);
            let available = topology_available && instruction_available;
            let reason = if !topology_available {
                "cpu_topology_unavailable"
            } else {
                instruction_reason
            };
            let mut metadata = BTreeMap::new();
            metadata.insert("operation_definition".into(), definition.operation.into());
            metadata.insert(
                "scope".into(),
                if definition.workload == Workload::SingleThread {
                    "single_pinned_thread"
                } else {
                    "aggregate_logical_processors"
                }
                .into(),
            );
            metadata.insert(
                "bound_diagnosis".into(),
                "working_set_sensitivity_proxy".into(),
            );
            if let Some(group) = comparison_group(definition.workload) {
                metadata.insert("comparison_group".into(), group.into());
                metadata.insert(
                    "comparison_basis".into(),
                    "lane-level multiply and add operations".into(),
                );
            }
            metadata.insert("kernel_revision".into(), "4".into());
            metadata.insert("harness_revision".into(), "2".into());
            BenchmarkDescriptor {
                id: definition.id.into(),
                name: definition.name.into(),
                category: BenchmarkCategory::Cpu,
                workload: definition.operation.into(),
                data_type: definition.data_type.into(),
                unit: result_unit(definition.workload).into(),
                supported_device_ids: if available {
                    vec![device.id.clone()]
                } else {
                    Vec::new()
                },
                available,
                unavailable_reason: reason.into(),
                suite_id: SUITE_ID.into(),
                display_order: definition.order,
                metadata,
            }
        })
        .collect()
}

struct Measurement {
    values: Vec<f64>,
    elapsed_ns: u64,
    cpu_ns: u64,
    checksum: u64,
}

fn measure<F>(
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
    label: &str,
    mut chunk: F,
) -> Result<Measurement, BenchmarkError>
where
    F: FnMut() -> (u64, u64),
{
    black_box(chunk());
    let target_ns = config.target_duration_ms.saturating_mul(1_000_000) / u64::from(config.samples);
    let mut values = Vec::with_capacity(config.samples as usize);
    let mut elapsed_total = 0u64;
    let mut cpu_total = 0u64;
    let mut checksum = 0u64;
    for sample in 0..config.samples {
        if cancellation.is_cancelled() {
            return Err(BenchmarkError::new(
                "cancelled",
                "The benchmark was cancelled.",
            ));
        }
        progress(ProgressUpdate {
            fraction: f64::from(sample) / f64::from(config.samples),
            phase: "measure".into(),
            message: format!("{label}: sample {}/{}", sample + 1, config.samples),
        });
        let cpu_start = thread_cpu_time_ns();
        let start = Instant::now();
        let mut operations = 0u64;
        loop {
            let (completed, value) = chunk();
            operations = operations.saturating_add(completed);
            checksum ^= value.rotate_left((operations & 63) as u32);
            if start.elapsed().as_nanos() >= u128::from(target_ns) {
                break;
            }
            if cancellation.is_cancelled() {
                return Err(BenchmarkError::new(
                    "cancelled",
                    "The benchmark was cancelled.",
                ));
            }
        }
        let elapsed = start.elapsed().as_nanos().max(1) as u64;
        let cpu = thread_cpu_time_ns().saturating_sub(cpu_start);
        values.push(operations as f64 * 1e9 / elapsed as f64);
        elapsed_total = elapsed_total.saturating_add(elapsed);
        cpu_total = cpu_total.saturating_add(cpu);
    }
    black_box(checksum);
    Ok(Measurement {
        values,
        elapsed_ns: elapsed_total,
        cpu_ns: cpu_total,
        checksum,
    })
}

fn quick_rate<F>(mut chunk: F) -> f64
where
    F: FnMut() -> (u64, u64),
{
    black_box(chunk());
    let start = Instant::now();
    let mut operations = 0u64;
    let mut checksum = 0u64;
    while start.elapsed().as_nanos() < u128::from(PROBE_NS) {
        let (count, value) = chunk();
        operations = operations.saturating_add(count);
        checksum ^= value;
    }
    black_box(checksum);
    operations as f64 * 1e9 / start.elapsed().as_nanos().max(1) as f64
}

fn patterned_bytes(length: usize) -> Vec<u8> {
    (0..length)
        .map(|index| ((index / 64 + index % 17) % 251) as u8)
        .collect()
}

fn scan_scalar(data: &[u8]) -> (u64, u64) {
    let hits = data
        .iter()
        .fold(0u64, |sum, byte| sum.wrapping_add(u64::from(*byte == b'e')));
    (data.len() as u64, hits)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(data: &[u8]) -> (u64, u64) {
    use std::arch::x86_64::{
        _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi8,
    };
    let needle = _mm256_set1_epi8(b'e' as i8);
    let vector_bytes = data.len() / 32 * 32;
    let mut hits = 0u64;
    for offset in (0..vector_bytes).step_by(32) {
        let value = unsafe { _mm256_loadu_si256(data.as_ptr().add(offset).cast()) };
        hits += u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(value, needle)).count_ones());
    }
    hits += data[vector_bytes..]
        .iter()
        .filter(|byte| **byte == b'e')
        .count() as u64;
    (data.len() as u64, hits)
}

fn scan(data: &[u8]) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        let (_, hits) = unsafe { scan_avx2(data) };
        return ((data.len() / STRING_BYTES) as u64, hits);
    }
    let (_, hits) = scan_scalar(data);
    ((data.len() / STRING_BYTES) as u64, hits)
}

fn classify(ratio: f64) -> &'static str {
    if ratio < MEMORY_BOUND_RATIO {
        "memory_bandwidth_bound"
    } else {
        "compute_bound"
    }
}

struct CodecCompressor {
    input: Vec<u8>,
    codec: Compress,
    output: Vec<u8>,
}

impl CodecCompressor {
    fn new(input: Vec<u8>) -> Self {
        let output_capacity = input
            .len()
            .saturating_add(input.len() / 16)
            .saturating_add(1024);
        Self {
            input,
            codec: Compress::new(Compression::new(6), false),
            output: Vec::with_capacity(output_capacity),
        }
    }

    fn chunk(&mut self) -> (u64, u64) {
        self.output.clear();
        self.codec.reset();
        if self
            .codec
            .compress_vec(&self.input, &mut self.output, FlushCompress::Finish)
            .is_err()
        {
            return (0, 0);
        }
        (
            self.input.len() as u64,
            self.output.len() as u64 ^ u64::from(self.output.first().copied().unwrap_or(0)),
        )
    }
}

struct CodecDecompressor {
    input: Vec<u8>,
    output_len: usize,
    codec: Decompress,
    output: Vec<u8>,
}

impl CodecDecompressor {
    fn new(input: Vec<u8>, output_len: usize) -> Self {
        Self {
            input,
            output_len,
            codec: Decompress::new(false),
            output: Vec::with_capacity(output_len),
        }
    }

    fn chunk(&mut self) -> (u64, u64) {
        self.output.clear();
        self.codec.reset(false);
        if self
            .codec
            .decompress_vec(&self.input, &mut self.output, FlushDecompress::Finish)
            .is_err()
        {
            return (0, 0);
        }
        (
            self.output_len as u64,
            self.output.len() as u64 ^ u64::from(self.output.first().copied().unwrap_or(0)),
        )
    }
}

struct PrimeFinder {
    marks: Vec<u8>,
}

impl PrimeFinder {
    fn new() -> Self {
        Self {
            marks: vec![1; PRIME_SEARCH_LIMIT + 1],
        }
    }

    fn chunk(&mut self) -> (u64, u64) {
        self.marks.fill(1);
        self.marks[0] = 0;
        self.marks[1] = 0;
        let mut prime = 2usize;
        while prime * prime <= PRIME_SEARCH_LIMIT {
            if self.marks[prime] != 0 {
                for composite in (prime * prime..=PRIME_SEARCH_LIMIT).step_by(prime) {
                    self.marks[composite] = 0;
                }
            }
            prime += 1;
        }
        let found = self.marks.iter().map(|mark| u64::from(*mark)).sum::<u64>();
        (found, found ^ PRIME_SEARCH_LIMIT as u64)
    }
}

enum ThreadWorkload {
    Integer(u64),
    Float32(f32),
    Float64(f64),
    String(Vec<u8>),
    Prime(PrimeFinder),
    Compress(CodecCompressor),
    Decompress(CodecDecompressor),
    Popcnt(u64),
    Aes(u64),
    Avx2Fma(f32),
    Avx2Fma64(f64),
}

impl ThreadWorkload {
    fn new(workload: Workload, thread_index: usize, large: bool, thread_count: usize) -> Self {
        let seed = thread_index as u64 + 1;
        match workload {
            Workload::Integer => Self::Integer(0x9e37_79b9_7f4a_7c15 ^ seed),
            Workload::Float32 => Self::Float32(1.000_001 + thread_index as f32 * 0.000_001),
            Workload::Float64 => Self::Float64(1.000_000_001 + thread_index as f64 * 0.000_000_001),
            Workload::String => {
                Self::String(patterned_bytes(data_bytes(workload, large, thread_count)))
            }
            Workload::Prime => Self::Prime(PrimeFinder::new()),
            Workload::Compress => Self::Compress(CodecCompressor::new(patterned_bytes(
                data_bytes(workload, large, thread_count),
            ))),
            Workload::Decompress => {
                let source = patterned_bytes(data_bytes(workload, large, thread_count));
                Self::Decompress(CodecDecompressor::new(
                    compress_to_vec(&source, 6),
                    source.len(),
                ))
            }
            Workload::Popcnt => Self::Popcnt(0x0123_4567_89ab_cdef ^ seed),
            Workload::Aes => Self::Aes(seed),
            Workload::Avx2Fma => Self::Avx2Fma(1.0 + thread_index as f32 * 0.001),
            Workload::Avx2Fma64 => Self::Avx2Fma64(1.0 + thread_index as f64 * 0.001),
            Workload::SingleThread => unreachable!("single-thread workload is measured separately"),
        }
    }

    fn chunk(&mut self) -> (u64, u64) {
        match self {
            Self::Integer(seed) => integer_chunk(seed),
            Self::Float32(seed) => float32_chunk(seed),
            Self::Float64(seed) => float64_chunk(seed),
            Self::String(data) => scan(data),
            Self::Prime(finder) => finder.chunk(),
            Self::Compress(codec) => codec.chunk(),
            Self::Decompress(codec) => codec.chunk(),
            Self::Popcnt(seed) => popcnt_chunk(seed),
            Self::Aes(seed) => aes_chunk(seed),
            Self::Avx2Fma(seed) => avx2_fma_chunk(seed),
            Self::Avx2Fma64(seed) => avx2_fma64_chunk(seed),
        }
    }
}

fn data_bytes(workload: Workload, large: bool, thread_count: usize) -> usize {
    match (workload, large) {
        (Workload::String, true) => (256 * 1024 * 1024 / thread_count.max(1)).max(1024 * 1024),
        (Workload::String, false) => 64 * 1024,
        (Workload::Compress | Workload::Decompress, true) => 2 * 1024 * 1024,
        (Workload::Compress | Workload::Decompress, false) => 128 * 1024,
        (Workload::Prime, _) => PRIME_SEARCH_LIMIT + 1,
        _ => 0,
    }
}

fn has_data_profile(workload: Workload) -> bool {
    matches!(
        workload,
        Workload::String | Workload::Compress | Workload::Decompress
    )
}

fn comparison_group(workload: Workload) -> Option<&'static str> {
    matches!(
        workload,
        Workload::Integer
            | Workload::Float32
            | Workload::Float64
            | Workload::Avx2Fma
            | Workload::Avx2Fma64
            | Workload::SingleThread
    )
    .then_some("arithmetic_multiply_add")
}

fn instruction_path(workload: Workload) -> &'static str {
    match workload {
        Workload::Integer => "scalar_i64",
        Workload::Float32 => "scalar_f32",
        Workload::Float64 => "scalar_f64",
        Workload::String => {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                return "avx2_ascii_scan";
            }
            "scalar_ascii_scan"
        }
        Workload::Prime => "sieve_of_eratosthenes_reused_marks",
        Workload::Compress | Workload::Decompress => "flate2_miniz_oxide_reusable_deflate_level_6",
        Workload::Popcnt => "x86_popcnt",
        Workload::Aes => "x86_aesni",
        Workload::Avx2Fma => "x86_avx2_fma",
        Workload::Avx2Fma64 => "x86_avx2_fma_f64",
        Workload::SingleThread => "scalar_i64",
    }
}

fn result_unit(workload: Workload) -> &'static str {
    match workload {
        Workload::Compress | Workload::Decompress => "MB/s",
        Workload::String => "strings/s",
        Workload::Prime => "primes/s",
        _ => "operations/s",
    }
}

fn reported_rate(workload: Workload, raw_rate: f64) -> f64 {
    if matches!(workload, Workload::Compress | Workload::Decompress) {
        raw_rate / 1_000_000.0
    } else {
        raw_rate
    }
}

#[derive(Clone, Copy)]
struct ThreadSample {
    operations: u64,
    elapsed_ns: u64,
    cpu_ns: u64,
    checksum: u64,
}

fn measure_parallel(
    workload: Workload,
    locations: &[ProcessorLocation],
    sample_count: u32,
    target_ns: u64,
    cancellation: &CancellationToken,
    large: bool,
) -> Result<Measurement, BenchmarkError> {
    for location in locations.iter().copied() {
        drop(AffinityGuard::pin(location)?);
    }
    let barrier = Arc::new(Barrier::new(locations.len()));
    let thread_results = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(locations.len());
        for (thread_index, location) in locations.iter().copied().enumerate() {
            let barrier = barrier.clone();
            handles.push(
                scope.spawn(move || -> Result<Vec<ThreadSample>, BenchmarkError> {
                    let _affinity = AffinityGuard::pin(location)?;
                    let mut state =
                        ThreadWorkload::new(workload, thread_index, large, locations.len());
                    let warmup_start = Instant::now();
                    while warmup_start.elapsed().as_nanos() < u128::from(WARMUP_NS) {
                        black_box(state.chunk());
                        if cancellation.is_cancelled() {
                            return Err(BenchmarkError::new(
                                "cancelled",
                                "The benchmark was cancelled.",
                            ));
                        }
                    }
                    let mut samples = Vec::with_capacity(sample_count as usize);
                    for _ in 0..sample_count {
                        barrier.wait();
                        let cpu_start = thread_cpu_time_ns();
                        let start = Instant::now();
                        let mut operations = 0u64;
                        let mut checksum = 0u64;
                        let mut cancelled = false;
                        loop {
                            let (count, value) = state.chunk();
                            operations = operations.saturating_add(count);
                            checksum ^= value.rotate_left((operations & 63) as u32);
                            if start.elapsed().as_nanos() >= u128::from(target_ns) {
                                break;
                            }
                            if cancellation.is_cancelled() {
                                cancelled = true;
                                break;
                            }
                        }
                        let elapsed_ns = start.elapsed().as_nanos().max(1) as u64;
                        let cpu_ns = thread_cpu_time_ns().saturating_sub(cpu_start);
                        barrier.wait();
                        if cancelled {
                            return Err(BenchmarkError::new(
                                "cancelled",
                                "The benchmark was cancelled.",
                            ));
                        }
                        samples.push(ThreadSample {
                            operations,
                            elapsed_ns,
                            cpu_ns,
                            checksum,
                        });
                    }
                    Ok(samples)
                }),
            );
        }
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    BenchmarkError::new(
                        "worker_failed",
                        "A CPU performance worker thread panicked.",
                    )
                })?
            })
            .collect::<Result<Vec<_>, _>>()
    })?;

    let mut values = Vec::with_capacity(sample_count as usize);
    let mut elapsed_total = 0u64;
    let mut cpu_total = 0u64;
    let mut checksum = 0u64;
    for sample_index in 0..sample_count as usize {
        let elapsed = thread_results
            .iter()
            .map(|samples| samples[sample_index].elapsed_ns)
            .max()
            .unwrap_or(1);
        let operations = thread_results
            .iter()
            .map(|samples| samples[sample_index].operations)
            .sum::<u64>();
        values.push(operations as f64 * 1e9 / elapsed as f64);
        elapsed_total = elapsed_total.saturating_add(elapsed);
        cpu_total = cpu_total.saturating_add(
            thread_results
                .iter()
                .map(|samples| samples[sample_index].cpu_ns)
                .sum::<u64>(),
        );
        checksum ^= thread_results
            .iter()
            .map(|samples| samples[sample_index].checksum)
            .fold(0, u64::wrapping_add);
    }
    Ok(Measurement {
        values,
        elapsed_ns: elapsed_total,
        cpu_ns: cpu_total,
        checksum,
    })
}

fn run_parallel(
    definition: &Definition,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
    locations: &[ProcessorLocation],
    device: &DeviceDescriptor,
    topology: &CpuTopology,
) -> Result<BenchmarkResult, BenchmarkError> {
    progress(ProgressUpdate {
        fraction: 0.0,
        phase: "prepare".into(),
        message: format!("Preparing {} logical-processor workers", locations.len()),
    });
    let per_sample_ns =
        config.target_duration_ms.saturating_mul(1_000_000) / u64::from(config.samples);
    let mut memory_ratio = 1.0;
    if has_data_profile(definition.workload) {
        progress(ProgressUpdate {
            fraction: 0.05,
            phase: "diagnose".into(),
            message: "Comparing cache-resident and large working sets".into(),
        });
        let cached = measure_parallel(
            definition.workload,
            locations,
            PROBE_SAMPLES,
            PROBE_NS,
            cancellation,
            false,
        )?;
        let large = measure_parallel(
            definition.workload,
            locations,
            PROBE_SAMPLES,
            PROBE_NS,
            cancellation,
            true,
        )?;
        memory_ratio =
            (statistics(&large.values).median / statistics(&cached.values).median).clamp(0.0, 2.0);
    }
    progress(ProgressUpdate {
        fraction: 0.1,
        phase: "measure".into(),
        message: format!(
            "Measuring {} across {} logical processors",
            definition.name,
            locations.len()
        ),
    });
    let measurement = measure_parallel(
        definition.workload,
        locations,
        config.samples,
        per_sample_ns,
        cancellation,
        true,
    )?;
    progress(ProgressUpdate {
        fraction: 1.0,
        phase: "complete".into(),
        message: format!("{} complete", definition.name),
    });

    let reported_values: Vec<_> = measurement
        .values
        .iter()
        .map(|value| reported_rate(definition.workload, *value))
        .collect();
    let sample_statistics = statistics(&reported_values);
    let metric = Metric {
        name: "throughput".into(),
        value: sample_statistics.median,
        unit: result_unit(definition.workload).into(),
        statistics: sample_statistics,
    };
    let denominator = measurement.elapsed_ns.max(1) as f64 * locations.len() as f64;
    let busy = (measurement.cpu_ns as f64 * 100.0 / denominator).clamp(0.0, 100.0);
    let per_thread_bytes = data_bytes(definition.workload, true, locations.len());
    let aggregate_bytes = per_thread_bytes.saturating_mul(locations.len());
    let mut workload_metadata = BTreeMap::new();
    workload_metadata.insert("operation_definition".into(), definition.operation.into());
    workload_metadata.insert("kernel_revision".into(), "4".into());
    workload_metadata.insert("harness_revision".into(), "2".into());
    if let Some(group) = comparison_group(definition.workload) {
        workload_metadata.insert("comparison_group".into(), group.into());
        workload_metadata.insert(
            "comparison_basis".into(),
            "lane-level multiply and add operations".into(),
        );
    }
    if matches!(
        definition.workload,
        Workload::Compress | Workload::Decompress
    ) {
        workload_metadata.insert(
            "rate_definition".into(),
            "decimal megabytes per second (1 MB = 1,000,000 bytes)".into(),
        );
    }
    if definition.workload == Workload::Prime {
        workload_metadata.insert("algorithm".into(), "sieve_of_eratosthenes".into());
        workload_metadata.insert("search_start".into(), "2".into());
        workload_metadata.insert(
            "search_limit_inclusive".into(),
            PRIME_SEARCH_LIMIT.to_string(),
        );
        workload_metadata.insert("known_primes_per_sieve".into(), "78498".into());
        workload_metadata.insert("marks_reused_between_sieves".into(), "true".into());
    }
    workload_metadata.insert(
        "instruction_path".into(),
        instruction_path(definition.workload).into(),
    );
    workload_metadata.insert("thread_count".into(), locations.len().to_string());
    workload_metadata.insert("thread_mode".into(), "logical_processors".into());
    workload_metadata.insert("warmup_ms".into(), (WARMUP_NS / 1_000_000).to_string());
    workload_metadata.insert("probe_samples".into(), PROBE_SAMPLES.to_string());
    workload_metadata.insert(
        "probe_duration_ms_per_sample".into(),
        (PROBE_NS / 1_000_000).to_string(),
    );
    workload_metadata.insert("scope".into(), "aggregate_logical_processors".into());
    workload_metadata.insert("affinity".into(), "one_worker_per_logical_processor".into());
    workload_metadata.insert(
        "buffer_ownership".into(),
        "pinned_thread_first_touch".into(),
    );
    workload_metadata.insert(
        "per_thread_working_set_bytes".into(),
        per_thread_bytes.to_string(),
    );
    workload_metadata.insert(
        "aggregate_working_set_bytes".into(),
        aggregate_bytes.to_string(),
    );
    workload_metadata.insert("cpu_runnable_percent".into(), format!("{busy:.2}"));
    workload_metadata.insert(
        "cpu_activity_interpretation".into(),
        "aggregate OS-scheduled thread time; memory stalls still count as running".into(),
    );
    workload_metadata.insert(
        "memory_sensitivity_ratio".into(),
        format!("{memory_ratio:.4}"),
    );
    workload_metadata.insert(
        "large_data_slowdown_percent".into(),
        format!("{:.2}", (1.0 - memory_ratio.min(1.0)) * 100.0),
    );
    workload_metadata.insert("bound_classification".into(), classify(memory_ratio).into());
    workload_metadata.insert(
        "classification_method".into(),
        if has_data_profile(definition.workload) {
            "aggregate large-working-set throughput divided by aggregate cache-resident throughput; ratio below 0.80 is memory bandwidth bound"
        } else if definition.workload == Workload::Prime {
            "fixed one-million-entry sieve per thread; no alternate working-set probe"
        } else {
            "register/compute workload with no streamed working set"
        }
        .into(),
    );
    workload_metadata.insert("classification_is_inference".into(), "true".into());
    workload_metadata.insert("checksum".into(), measurement.checksum.to_string());
    let mut device_metadata = BTreeMap::new();
    device_metadata.insert("logical_processors".into(), locations.len().to_string());
    device_metadata.insert(
        "physical_cores".into(),
        topology.physical_cores.len().to_string(),
    );
    Ok(BenchmarkResult {
        benchmark_id: definition.id.into(),
        device_id: device.id.clone(),
        elapsed_ns: measurement.elapsed_ns,
        metrics: vec![metric],
        workload_metadata,
        device_metadata,
    })
}

pub fn run(
    id: &str,
    config: &BenchmarkConfig,
    cancellation: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
    processor: ProcessorLocation,
    device: &DeviceDescriptor,
    topology: &CpuTopology,
) -> Result<BenchmarkResult, BenchmarkError> {
    let definition = DEFINITIONS
        .iter()
        .find(|item| item.id == id)
        .ok_or_else(|| {
            BenchmarkError::new(
                "unsupported_benchmark",
                format!("Unknown CPU performance benchmark '{id}'."),
            )
        })?;
    let (is_available, reason) = available(definition.workload);
    if !is_available {
        return Err(BenchmarkError::new("benchmark_unavailable", reason));
    }
    if definition.workload != Workload::SingleThread {
        let mut locations: Vec<_> = topology.core_threads.values().flatten().copied().collect();
        locations.sort_unstable();
        locations.dedup();
        if locations.is_empty() {
            return Err(BenchmarkError::new(
                "topology_unavailable",
                "No logical processors are available for the CPU performance benchmark.",
            ));
        }
        return run_parallel(
            definition,
            config,
            cancellation,
            progress,
            &locations,
            device,
            topology,
        );
    }
    let _affinity = AffinityGuard::pin(processor)?;
    let mut memory_ratio = 1.0;
    let mut working_set_bytes = 0usize;
    let instruction_path: &str;

    let measurement = match definition.workload {
        Workload::Integer => {
            instruction_path = "scalar_i64";
            let mut seed = 0x9e37_79b9_7f4a_7c15u64;
            measure(config, cancellation, progress, definition.name, || {
                integer_chunk(&mut seed)
            })?
        }
        Workload::Float32 => {
            instruction_path = "scalar_f32";
            let mut seed = 1.000_001f32;
            measure(config, cancellation, progress, definition.name, || {
                float32_chunk(&mut seed)
            })?
        }
        Workload::Float64 => {
            instruction_path = "scalar_f64";
            let mut seed = 1.000_000_001f64;
            measure(config, cancellation, progress, definition.name, || {
                float64_chunk(&mut seed)
            })?
        }
        Workload::String => {
            instruction_path = "scalar_ascii_scan";
            let small = patterned_bytes(256 * 1024);
            let large = patterned_bytes(64 * 1024 * 1024);
            working_set_bytes = large.len();
            let cached = quick_rate(|| scan(&small));
            let streaming = quick_rate(|| scan(&large));
            memory_ratio = (streaming / cached).clamp(0.0, 2.0);
            measure(config, cancellation, progress, definition.name, || {
                scan(&large)
            })?
        }
        Workload::Prime => {
            instruction_path = "sieve_of_eratosthenes_reused_marks";
            let mut finder = PrimeFinder::new();
            measure(config, cancellation, progress, definition.name, || {
                finder.chunk()
            })?
        }
        Workload::Compress => {
            instruction_path = "miniz_oxide_deflate_level_6";
            let small = patterned_bytes(256 * 1024);
            let large = patterned_bytes(8 * 1024 * 1024);
            working_set_bytes = large.len();
            let cached = quick_rate(|| compress_chunk(&small));
            let streaming = quick_rate(|| compress_chunk(&large));
            memory_ratio = (streaming / cached).clamp(0.0, 2.0);
            measure(config, cancellation, progress, definition.name, || {
                compress_chunk(&large)
            })?
        }
        Workload::Decompress => {
            instruction_path = "miniz_oxide_deflate_level_6";
            let small_source = patterned_bytes(256 * 1024);
            let large_source = patterned_bytes(8 * 1024 * 1024);
            let small = compress_to_vec(&small_source, 6);
            let large = compress_to_vec(&large_source, 6);
            working_set_bytes = large_source.len();
            let cached = quick_rate(|| decompress_chunk(&small, small_source.len()));
            let streaming = quick_rate(|| decompress_chunk(&large, large_source.len()));
            memory_ratio = (streaming / cached).clamp(0.0, 2.0);
            measure(config, cancellation, progress, definition.name, || {
                decompress_chunk(&large, large_source.len())
            })?
        }
        Workload::Popcnt => {
            instruction_path = "x86_popcnt";
            let mut seed = 0x0123_4567_89ab_cdefu64;
            measure(config, cancellation, progress, definition.name, || {
                popcnt_chunk(&mut seed)
            })?
        }
        Workload::Aes => {
            instruction_path = "x86_aesni";
            let mut seed = 1u64;
            measure(config, cancellation, progress, definition.name, || {
                aes_chunk(&mut seed)
            })?
        }
        Workload::Avx2Fma => {
            instruction_path = "x86_avx2_fma";
            let mut seed = 1.0f32;
            measure(config, cancellation, progress, definition.name, || {
                avx2_fma_chunk(&mut seed)
            })?
        }
        Workload::Avx2Fma64 => {
            instruction_path = "x86_avx2_fma_f64";
            let mut seed = 1.0f64;
            measure(config, cancellation, progress, definition.name, || {
                avx2_fma64_chunk(&mut seed)
            })?
        }
        Workload::SingleThread => {
            instruction_path = "scalar_i64";
            let mut seed = 0x9e37_79b9_7f4a_7c15u64;
            measure(config, cancellation, progress, definition.name, || {
                integer_chunk(&mut seed)
            })?
        }
    };

    progress(ProgressUpdate {
        fraction: 1.0,
        phase: "complete".into(),
        message: format!("{} complete", definition.name),
    });
    let sample_statistics = statistics(&measurement.values);
    let metric = Metric {
        name: "throughput".into(),
        value: sample_statistics.median,
        unit: "operations/s".into(),
        statistics: sample_statistics,
    };
    let busy = (measurement.cpu_ns as f64 * 100.0 / measurement.elapsed_ns.max(1) as f64)
        .clamp(0.0, 100.0);
    let mut workload_metadata = BTreeMap::new();
    workload_metadata.insert("operation_definition".into(), definition.operation.into());
    workload_metadata.insert("instruction_path".into(), instruction_path.into());
    workload_metadata.insert("kernel_revision".into(), "4".into());
    workload_metadata.insert("harness_revision".into(), "2".into());
    if let Some(group) = comparison_group(definition.workload) {
        workload_metadata.insert("comparison_group".into(), group.into());
        workload_metadata.insert(
            "comparison_basis".into(),
            "lane-level multiply and add operations".into(),
        );
    }
    workload_metadata.insert("thread_count".into(), "1".into());
    workload_metadata.insert("scope".into(), "single_pinned_thread".into());
    workload_metadata.insert("working_set_bytes".into(), working_set_bytes.to_string());
    workload_metadata.insert("cpu_runnable_percent".into(), format!("{busy:.2}"));
    workload_metadata.insert(
        "cpu_activity_interpretation".into(),
        "OS-scheduled thread time; memory stalls still count as running".into(),
    );
    workload_metadata.insert(
        "memory_sensitivity_ratio".into(),
        format!("{memory_ratio:.4}"),
    );
    workload_metadata.insert(
        "large_data_slowdown_percent".into(),
        format!("{:.2}", (1.0 - memory_ratio.min(1.0)) * 100.0),
    );
    workload_metadata.insert("bound_classification".into(), classify(memory_ratio).into());
    workload_metadata.insert(
        "classification_method".into(),
        if working_set_bytes == 0 {
            "register/compute workload with no streamed working set"
        } else {
            "large-working-set throughput divided by cache-resident throughput; ratio below 0.80 is memory bandwidth bound"
        }
        .into(),
    );
    workload_metadata.insert("classification_is_inference".into(), "true".into());
    workload_metadata.insert("checksum".into(), measurement.checksum.to_string());
    let mut device_metadata = BTreeMap::new();
    device_metadata.insert("processor_group".into(), processor.group.to_string());
    device_metadata.insert("processor_index".into(), processor.index.to_string());
    device_metadata.insert(
        "physical_cores".into(),
        topology.physical_cores.len().to_string(),
    );
    Ok(BenchmarkResult {
        benchmark_id: id.into(),
        device_id: device.id.clone(),
        elapsed_ns: measurement.elapsed_ns,
        metrics: vec![metric],
        workload_metadata,
        device_metadata,
    })
}

#[inline(never)]
fn integer_chunk(seed: &mut u64) -> (u64, u64) {
    let mut a = black_box(*seed);
    let mut b = a ^ 0xd1b5_4a32_d192_ed03;
    let mut c = a ^ 0x94d0_49bb_1331_11eb;
    let mut d = a ^ 0xbf58_476d_1ce4_e5b9;
    let mut e = a ^ 0x2545_f491_4f6c_dd1d;
    let mut f = a ^ 0x369d_ea0f_31a5_3f85;
    let mut g = a ^ 0xdb4f_0b91_75ae_2165;
    let mut h = a ^ 0xbb67_ae85_84ca_a73b;
    for _ in 0..4096 {
        a = a.wrapping_mul(0x9e37_79b1).wrapping_add(0x85eb_ca6b);
        b = b.wrapping_mul(0x85eb_ca77).wrapping_add(0xc2b2_ae3d);
        c = c.wrapping_mul(0xc2b2_ae3d).wrapping_add(0x27d4_eb2f);
        d = d.wrapping_mul(0x27d4_eb2d).wrapping_add(0x1656_67b1);
        e = e.wrapping_mul(0x1656_67b1).wrapping_add(0x9e37_79b9);
        f = f.wrapping_mul(0x9e37_79b9).wrapping_add(0x85eb_ca77);
        g = g.wrapping_mul(0x85eb_ca77).wrapping_add(0xc2b2_ae3d);
        h = h.wrapping_mul(0xc2b2_ae3d).wrapping_add(0x27d4_eb2d);
    }
    *seed = a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
    (4096 * 8 * 2, *seed)
}

#[inline(never)]
fn float32_chunk(seed: &mut f32) -> (u64, u64) {
    let mut a = black_box(*seed);
    let mut b = a + 0.25;
    let mut c = a + 0.5;
    let mut d = a + 0.75;
    let mut e = a + 1.0;
    let mut f = a + 1.25;
    let mut g = a + 1.5;
    let mut h = a + 1.75;
    for _ in 0..4096 {
        a = a * 1.000_000_1 + 0.000_001;
        b = b * 0.999_999_9 + 0.000_002;
        c = c * 1.000_000_1 + 0.000_003;
        d = d * 0.999_999_9 + 0.000_004;
        e = e * 1.000_000_1 + 0.000_005;
        f = f * 0.999_999_9 + 0.000_006;
        g = g * 1.000_000_1 + 0.000_007;
        h = h * 0.999_999_9 + 0.000_008;
    }
    *seed = (a + b + c + d + e + f + g + h) * 0.125;
    (4096 * 16, u64::from(seed.to_bits()))
}

#[inline(never)]
fn float64_chunk(seed: &mut f64) -> (u64, u64) {
    let mut a = black_box(*seed);
    let mut b = a + 0.25;
    let mut c = a + 0.5;
    let mut d = a + 0.75;
    let mut e = a + 1.0;
    let mut f = a + 1.25;
    let mut g = a + 1.5;
    let mut h = a + 1.75;
    for _ in 0..4096 {
        a = a * 1.000_000_000_1 + 0.000_000_001;
        b = b * 0.999_999_999_9 + 0.000_000_002;
        c = c * 1.000_000_000_1 + 0.000_000_003;
        d = d * 0.999_999_999_9 + 0.000_000_004;
        e = e * 1.000_000_000_1 + 0.000_000_005;
        f = f * 0.999_999_999_9 + 0.000_000_006;
        g = g * 1.000_000_000_1 + 0.000_000_007;
        h = h * 0.999_999_999_9 + 0.000_000_008;
    }
    *seed = (a + b + c + d + e + f + g + h) * 0.125;
    (4096 * 16, seed.to_bits())
}

fn compress_chunk(data: &[u8]) -> (u64, u64) {
    let output = compress_to_vec(data, 6);
    (
        data.len() as u64,
        output.len() as u64 ^ u64::from(output.first().copied().unwrap_or(0)),
    )
}

fn decompress_chunk(data: &[u8], output_length: usize) -> (u64, u64) {
    match decompress_to_vec(data) {
        Ok(output) => (
            output_length as u64,
            output.len() as u64 ^ u64::from(output.first().copied().unwrap_or(0)),
        ),
        Err(_) => (0, 0),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn popcnt_inner(seed: &mut u64) -> (u64, u64) {
    use std::arch::x86_64::_popcnt64;
    let mut value = *seed;
    let mut total = 0i64;
    for _ in 0..4096 {
        total += i64::from(_popcnt64(value as i64));
        total += i64::from(_popcnt64(value.rotate_left(7) as i64));
        total += i64::from(_popcnt64(value.rotate_left(17) as i64));
        total += i64::from(_popcnt64(value.rotate_left(29) as i64));
        value = value.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
    }
    *seed = value ^ total as u64;
    (4096 * 4, *seed)
}

fn popcnt_chunk(seed: &mut u64) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        popcnt_inner(seed)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = seed;
        (0, 0)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "aes")]
unsafe fn aes_inner(seed: &mut u64) -> (u64, u64) {
    use std::arch::x86_64::{_mm_aesenc_si128, _mm_cvtsi128_si64, _mm_set1_epi64x, _mm_xor_si128};
    let key = _mm_set1_epi64x(0x1bd1_1bda_a9fc_1a22u64 as i64);
    let mut a = _mm_set1_epi64x(*seed as i64);
    let mut b = _mm_set1_epi64x(seed.rotate_left(17) as i64);
    let mut c = _mm_set1_epi64x(seed.rotate_left(23) as i64);
    let mut d = _mm_set1_epi64x(seed.rotate_left(31) as i64);
    let mut e = _mm_set1_epi64x(seed.rotate_left(37) as i64);
    let mut f = _mm_set1_epi64x(seed.rotate_left(43) as i64);
    let mut g = _mm_set1_epi64x(seed.rotate_left(53) as i64);
    let mut h = _mm_set1_epi64x(seed.rotate_left(59) as i64);
    for _ in 0..4096 {
        a = _mm_aesenc_si128(a, key);
        b = _mm_aesenc_si128(b, key);
        c = _mm_aesenc_si128(c, key);
        d = _mm_aesenc_si128(d, key);
        e = _mm_aesenc_si128(e, key);
        f = _mm_aesenc_si128(f, key);
        g = _mm_aesenc_si128(g, key);
        h = _mm_aesenc_si128(h, key);
    }
    let result = _mm_xor_si128(
        _mm_xor_si128(_mm_xor_si128(a, b), _mm_xor_si128(c, d)),
        _mm_xor_si128(_mm_xor_si128(e, f), _mm_xor_si128(g, h)),
    );
    *seed = _mm_cvtsi128_si64(result) as u64;
    (4096 * 8, *seed)
}

fn aes_chunk(seed: &mut u64) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        aes_inner(seed)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = seed;
        (0, 0)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fma_inner(seed: &mut f32) -> (u64, u64) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_cvtss_f32, _mm256_fmadd_ps, _mm256_set1_ps};
    let multiplier = _mm256_set1_ps(0.999_999_94);
    let addend = _mm256_set1_ps(0.000_001);
    let mut a = _mm256_set1_ps(*seed);
    let mut b = _mm256_set1_ps(*seed + 0.25);
    let mut c = _mm256_set1_ps(*seed + 0.5);
    let mut d = _mm256_set1_ps(*seed + 0.75);
    let mut e = _mm256_set1_ps(*seed + 1.0);
    let mut f = _mm256_set1_ps(*seed + 1.25);
    let mut g = _mm256_set1_ps(*seed + 1.5);
    let mut h = _mm256_set1_ps(*seed + 1.75);
    for _ in 0..4096 {
        a = _mm256_fmadd_ps(a, multiplier, addend);
        b = _mm256_fmadd_ps(b, multiplier, addend);
        c = _mm256_fmadd_ps(c, multiplier, addend);
        d = _mm256_fmadd_ps(d, multiplier, addend);
        e = _mm256_fmadd_ps(e, multiplier, addend);
        f = _mm256_fmadd_ps(f, multiplier, addend);
        g = _mm256_fmadd_ps(g, multiplier, addend);
        h = _mm256_fmadd_ps(h, multiplier, addend);
    }
    *seed = _mm256_cvtss_f32(_mm256_add_ps(
        _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d)),
        _mm256_add_ps(_mm256_add_ps(e, f), _mm256_add_ps(g, h)),
    ));
    (4096 * 8 * 8 * 2, u64::from(seed.to_bits()))
}

fn avx2_fma_chunk(seed: &mut f32) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2_fma_inner(seed)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        float32_chunk(seed)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fma64_inner(seed: &mut f64) -> (u64, u64) {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_cvtsd_f64, _mm256_fmadd_pd, _mm256_set1_pd};
    let multiplier = _mm256_set1_pd(0.999_999_999_999);
    let addend = _mm256_set1_pd(0.000_000_001);
    let mut a = _mm256_set1_pd(*seed);
    let mut b = _mm256_set1_pd(*seed + 0.25);
    let mut c = _mm256_set1_pd(*seed + 0.5);
    let mut d = _mm256_set1_pd(*seed + 0.75);
    let mut e = _mm256_set1_pd(*seed + 1.0);
    let mut f = _mm256_set1_pd(*seed + 1.25);
    let mut g = _mm256_set1_pd(*seed + 1.5);
    let mut h = _mm256_set1_pd(*seed + 1.75);
    for _ in 0..4096 {
        a = _mm256_fmadd_pd(a, multiplier, addend);
        b = _mm256_fmadd_pd(b, multiplier, addend);
        c = _mm256_fmadd_pd(c, multiplier, addend);
        d = _mm256_fmadd_pd(d, multiplier, addend);
        e = _mm256_fmadd_pd(e, multiplier, addend);
        f = _mm256_fmadd_pd(f, multiplier, addend);
        g = _mm256_fmadd_pd(g, multiplier, addend);
        h = _mm256_fmadd_pd(h, multiplier, addend);
    }
    *seed = _mm256_cvtsd_f64(_mm256_add_pd(
        _mm256_add_pd(_mm256_add_pd(a, b), _mm256_add_pd(c, d)),
        _mm256_add_pd(_mm256_add_pd(e, f), _mm256_add_pd(g, h)),
    ));
    (4096 * 8 * 4 * 2, seed.to_bits())
}

fn avx2_fma64_chunk(seed: &mut f64) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avx2_fma64_inner(seed)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        float64_chunk(seed)
    }
}

#[cfg(windows)]
fn thread_cpu_time_ns() -> u64 {
    use windows_sys::Win32::{
        Foundation::FILETIME,
        System::Threading::{GetCurrentThread, GetThreadTimes},
    };
    let mut creation = FILETIME::default();
    let mut exit = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();
    let ok = unsafe {
        GetThreadTimes(
            GetCurrentThread(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    };
    if ok == 0 {
        return 0;
    }
    let ticks =
        |time: FILETIME| (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
    ticks(kernel)
        .saturating_add(ticks(user))
        .saturating_mul(100)
}

#[cfg(not(windows))]
fn thread_cpu_time_ns() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_use_workload_appropriate_units() {
        let device = DeviceDescriptor {
            id: "cpu:test".into(),
            name: "test".into(),
            category: gluj_bench_core::DeviceCategory::Cpu,
            available: true,
            status: String::new(),
            properties: BTreeMap::new(),
            caches: Vec::new(),
        };
        let items = descriptors(&device, true);
        assert!(items.iter().all(|item| {
            if item.id.contains("compression") {
                item.unit == "MB/s"
            } else if item.id.contains("string") {
                item.unit == "strings/s"
            } else if item.id.contains("prime") {
                item.unit == "primes/s"
            } else {
                item.unit == "operations/s"
            }
        }));
        assert!(
            items
                .iter()
                .filter(|item| item.id != "cpu.performance.single_thread.integer.i64")
                .all(|item| item.metadata["scope"] == "aggregate_logical_processors")
        );
        assert_eq!(
            items
                .iter()
                .find(|item| item.id == "cpu.performance.single_thread.integer.i64")
                .unwrap()
                .metadata["scope"],
            "single_pinned_thread"
        );
    }

    #[test]
    fn prime_sieve_finds_the_known_population_below_one_million() {
        let mut finder = PrimeFinder::new();
        assert_eq!(finder.chunk().0, 78_498);
        assert_eq!(finder.chunk().0, 78_498);
    }

    #[test]
    fn classification_threshold_is_stable() {
        assert_eq!(classify(0.79), "memory_bandwidth_bound");
        assert_eq!(classify(0.80), "compute_bound");
        assert_eq!(classify(0.90), "compute_bound");
    }

    #[test]
    fn codec_rates_convert_raw_bytes_to_decimal_megabytes() {
        assert_eq!(
            reported_rate(Workload::Compress, 3_744_186_882.15),
            3_744.186_882_15
        );
        assert_eq!(reported_rate(Workload::Decompress, 1_000_000.0), 1.0);
        assert_eq!(reported_rate(Workload::Integer, 1_000_000.0), 1_000_000.0);
    }

    #[test]
    fn deflate_workloads_round_trip_and_count_input_operations() {
        let source = patterned_bytes(64 * 1024);
        let compressed = compress_to_vec(&source, 6);
        let (operations, checksum) = decompress_chunk(&compressed, source.len());
        assert_eq!(operations, source.len() as u64);
        assert_ne!(checksum, 0);
        assert_eq!(decompress_to_vec(&compressed).unwrap(), source);
    }

    #[test]
    fn reusable_codec_workers_do_not_reallocate_between_chunks() {
        let source = patterned_bytes(256 * 1024);
        let mut compressor = CodecCompressor::new(source.clone());
        let output_capacity = compressor.output.capacity();
        let first = compressor.chunk();
        let compressed = compressor.output.clone();
        let second = compressor.chunk();
        assert_eq!(first.0, source.len() as u64);
        assert_eq!(second.0, source.len() as u64);
        assert_eq!(compressor.output.capacity(), output_capacity);

        let mut decompressor = CodecDecompressor::new(compressed, source.len());
        let decode_capacity = decompressor.output.capacity();
        assert_eq!(decompressor.chunk().0, source.len() as u64);
        assert_eq!(decompressor.chunk().0, source.len() as u64);
        assert_eq!(decompressor.output.capacity(), decode_capacity);
        assert_eq!(decompressor.output, source);
    }

    #[test]
    fn optimized_ascii_scan_matches_scalar_result() {
        let source = patterned_bytes(1024 * 1024);
        let optimized = scan(&source);
        let scalar = scan_scalar(&source);
        assert_eq!(optimized.0, (source.len() / STRING_BYTES) as u64);
        assert_eq!(optimized.1, scalar.1);
    }

    #[test]
    fn normalized_arithmetic_chunks_count_multiply_and_add_operations() {
        let mut integer = 1u64;
        let mut float32 = 1.0f32;
        let mut float64 = 1.0f64;
        assert_eq!(integer_chunk(&mut integer).0, 4096 * 8 * 2);
        assert_eq!(float32_chunk(&mut float32).0, 4096 * 8 * 2);
        assert_eq!(float64_chunk(&mut float64).0, 4096 * 8 * 2);
    }
}
