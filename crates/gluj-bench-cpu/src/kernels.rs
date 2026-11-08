use crate::topology::{AffinityGuard, ProcessorLocation};
use gluj_bench_core::{BenchmarkError, CancellationToken, SampleStatistics};
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    hint::black_box,
    ptr::NonNull,
    sync::{Arc, Barrier, atomic::Ordering},
    thread,
    time::Instant,
};

const ALIGNMENT: usize = 64;
const MAX_PASSES: usize = 1_000_000_000;

pub struct AlignedBuffer {
    pointer: NonNull<u8>,
    length: usize,
    layout: Layout,
}
unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    pub fn new(length: usize) -> Result<Self, BenchmarkError> {
        let length = length.max(ALIGNMENT).next_multiple_of(ALIGNMENT);
        let layout = Layout::from_size_align(length, ALIGNMENT).map_err(|_| {
            BenchmarkError::new("allocation_failed", "Invalid aligned buffer layout.")
        })?;
        let pointer = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or_else(|| {
            BenchmarkError::new(
                "allocation_failed",
                format!("Unable to allocate {length} benchmark bytes."),
            )
        })?;
        Ok(Self {
            pointer,
            length,
            layout,
        })
    }
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn as_ptr(&self) -> *const u8 {
        self.pointer.as_ptr()
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pointer.as_ptr()
    }
    pub fn initialize(&mut self, seed: u8) {
        unsafe { std::ptr::write_bytes(self.as_mut_ptr(), seed, self.length) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Read,
    Write,
    Copy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePreparation {
    Hot,
    Cold,
}

#[derive(Debug, Clone, Copy)]
pub struct MeasurementOptions {
    pub access: gluj_bench_core::AccessPolicy,
    pub sample_count: u32,
    pub target_ns: u64,
    pub streaming: bool,
    pub preparation: CachePreparation,
    pub byte_multiplier: u8,
}

#[derive(Debug, Clone, Copy)]
struct SweepOptions {
    access: gluj_bench_core::AccessPolicy,
    realization: u32,
    rotation: usize,
    passes: usize,
    chunk_passes: usize,
    streaming: bool,
    preparation: CachePreparation,
    first_touch: bool,
}

pub fn simd_path() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            "avx2"
        } else {
            "sse2"
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "scalar"
    }
}

pub fn cold_cache_method() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        "clflush_mfence"
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "best_effort_cache_thrash"
    }
}

#[inline(never)]
pub fn execute(
    operation: Operation,
    source: &mut AlignedBuffer,
    destination: &mut AlignedBuffer,
    passes: usize,
    streaming: bool,
) {
    execute_rotated(operation, source, destination, passes, streaming, 0);
}

#[inline(never)]
fn execute_rotated(
    operation: Operation,
    source: &mut AlignedBuffer,
    destination: &mut AlignedBuffer,
    passes: usize,
    streaming: bool,
    start_offset: usize,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if std::is_x86_feature_detected!("avx2") {
            return avx2(
                operation,
                source,
                destination,
                passes,
                streaming,
                start_offset,
            );
        }
        sse2(
            operation,
            source,
            destination,
            passes,
            streaming,
            start_offset,
        )
    }
    #[cfg(not(target_arch = "x86_64"))]
    scalar(operation, source, destination, passes, start_offset);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(
    operation: Operation,
    source: &mut AlignedBuffer,
    destination: &mut AlignedBuffer,
    passes: usize,
    streaming: bool,
    start_offset: usize,
) {
    use std::arch::x86_64::*;
    let length = source.len().min(destination.len());
    let start_offset = start_offset.min(length).next_multiple_of(32).min(length);
    let source_pointer = source.as_ptr();
    let source_mut_pointer = source.as_mut_ptr();
    let destination_pointer = destination.as_mut_ptr();
    let mut checksum = _mm256_setzero_si256();
    match operation {
        Operation::Read => {
            let mut accumulators = [_mm256_setzero_si256(); 8];
            macro_rules! read_range {
                ($begin:expr, $end:expr) => {{
                    let end = $end;
                    let mut offset = $begin;
                    while offset + 256 <= end {
                        accumulators[0] = _mm256_xor_si256(accumulators[0], unsafe {
                            _mm256_load_si256(source_pointer.add(offset).cast())
                        });
                        accumulators[1] = _mm256_xor_si256(accumulators[1], unsafe {
                            _mm256_load_si256(source_pointer.add(offset + 32).cast())
                        });
                        accumulators[2] = _mm256_xor_si256(accumulators[2], unsafe {
                            _mm256_load_si256(source_pointer.add(offset + 64).cast())
                        });
                        accumulators[3] = _mm256_xor_si256(accumulators[3], unsafe {
                            _mm256_load_si256(source_pointer.add(offset + 96).cast())
                        });
                        accumulators[4] = _mm256_xor_si256(accumulators[4], unsafe {
                            _mm256_load_si256(source_pointer.add(offset + 128).cast())
                        });
                        accumulators[5] = _mm256_xor_si256(accumulators[5], unsafe {
                            _mm256_load_si256(source_pointer.add(offset + 160).cast())
                        });
                        accumulators[6] = _mm256_xor_si256(accumulators[6], unsafe {
                            _mm256_load_si256(source_pointer.add(offset + 192).cast())
                        });
                        accumulators[7] = _mm256_xor_si256(accumulators[7], unsafe {
                            _mm256_load_si256(source_pointer.add(offset + 224).cast())
                        });
                        offset += 256;
                    }
                    while offset < end {
                        accumulators[0] = _mm256_xor_si256(accumulators[0], unsafe {
                            _mm256_load_si256(source_pointer.add(offset).cast())
                        });
                        offset += 32;
                    }
                }};
            }
            for _ in 0..passes {
                read_range!(start_offset, length);
                read_range!(0, start_offset);
            }
            checksum = _mm256_xor_si256(accumulators[0], accumulators[1]);
            checksum = _mm256_xor_si256(checksum, accumulators[2]);
            checksum = _mm256_xor_si256(checksum, accumulators[3]);
            checksum = _mm256_xor_si256(checksum, accumulators[4]);
            checksum = _mm256_xor_si256(checksum, accumulators[5]);
            checksum = _mm256_xor_si256(checksum, accumulators[6]);
            checksum = _mm256_xor_si256(checksum, accumulators[7]);
        }
        Operation::Write => {
            macro_rules! write_range {
                ($begin:expr, $end:expr, $store:ident, $value:expr) => {{
                    let end = $end;
                    let mut offset = $begin;
                    while offset + 256 <= end {
                        unsafe {
                            $store(source_mut_pointer.add(offset).cast(), $value);
                            $store(source_mut_pointer.add(offset + 32).cast(), $value);
                            $store(source_mut_pointer.add(offset + 64).cast(), $value);
                            $store(source_mut_pointer.add(offset + 96).cast(), $value);
                            $store(source_mut_pointer.add(offset + 128).cast(), $value);
                            $store(source_mut_pointer.add(offset + 160).cast(), $value);
                            $store(source_mut_pointer.add(offset + 192).cast(), $value);
                            $store(source_mut_pointer.add(offset + 224).cast(), $value);
                        }
                        offset += 256;
                    }
                    while offset < end {
                        unsafe { $store(source_mut_pointer.add(offset).cast(), $value) };
                        offset += 32;
                    }
                }};
            }
            for pass in 0..passes {
                let value = _mm256_set1_epi64x(pass as i64 ^ 0x5a5a5a5a);
                if streaming {
                    write_range!(start_offset, length, _mm256_stream_si256, value);
                    write_range!(0, start_offset, _mm256_stream_si256, value);
                } else {
                    write_range!(start_offset, length, _mm256_store_si256, value);
                    write_range!(0, start_offset, _mm256_store_si256, value);
                }
            }
        }
        Operation::Copy => {
            macro_rules! copy_range {
                ($begin:expr, $end:expr, $store:ident) => {{
                    let end = $end;
                    let mut offset = $begin;
                    while offset + 256 <= end {
                        let values = unsafe {
                            [
                                _mm256_load_si256(source_pointer.add(offset).cast()),
                                _mm256_load_si256(source_pointer.add(offset + 32).cast()),
                                _mm256_load_si256(source_pointer.add(offset + 64).cast()),
                                _mm256_load_si256(source_pointer.add(offset + 96).cast()),
                                _mm256_load_si256(source_pointer.add(offset + 128).cast()),
                                _mm256_load_si256(source_pointer.add(offset + 160).cast()),
                                _mm256_load_si256(source_pointer.add(offset + 192).cast()),
                                _mm256_load_si256(source_pointer.add(offset + 224).cast()),
                            ]
                        };
                        unsafe {
                            $store(destination_pointer.add(offset).cast(), values[0]);
                            $store(destination_pointer.add(offset + 32).cast(), values[1]);
                            $store(destination_pointer.add(offset + 64).cast(), values[2]);
                            $store(destination_pointer.add(offset + 96).cast(), values[3]);
                            $store(destination_pointer.add(offset + 128).cast(), values[4]);
                            $store(destination_pointer.add(offset + 160).cast(), values[5]);
                            $store(destination_pointer.add(offset + 192).cast(), values[6]);
                            $store(destination_pointer.add(offset + 224).cast(), values[7]);
                        }
                        offset += 256;
                    }
                    while offset < end {
                        let value = unsafe { _mm256_load_si256(source_pointer.add(offset).cast()) };
                        unsafe { $store(destination_pointer.add(offset).cast(), value) };
                        offset += 32;
                    }
                }};
            }
            for _ in 0..passes {
                if streaming {
                    copy_range!(start_offset, length, _mm256_stream_si256);
                    copy_range!(0, start_offset, _mm256_stream_si256);
                } else {
                    copy_range!(start_offset, length, _mm256_store_si256);
                    copy_range!(0, start_offset, _mm256_store_si256);
                }
            }
        }
    }
    if streaming && !matches!(operation, Operation::Read) {
        _mm_sfence();
    }
    black_box(checksum);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2(
    operation: Operation,
    source: &mut AlignedBuffer,
    destination: &mut AlignedBuffer,
    passes: usize,
    streaming: bool,
    start_offset: usize,
) {
    use std::arch::x86_64::*;
    let length = source.len().min(destination.len());
    let start_offset = start_offset.min(length).next_multiple_of(16).min(length);
    let source_pointer = source.as_ptr();
    let source_mut_pointer = source.as_mut_ptr();
    let destination_pointer = destination.as_mut_ptr();
    let mut checksum = _mm_setzero_si128();
    match operation {
        Operation::Read => {
            let mut accumulators = [_mm_setzero_si128(); 8];
            macro_rules! read_range {
                ($begin:expr, $end:expr) => {{
                    let end = $end;
                    let mut offset = $begin;
                    while offset + 128 <= end {
                        accumulators[0] = _mm_xor_si128(accumulators[0], unsafe {
                            _mm_load_si128(source_pointer.add(offset).cast())
                        });
                        accumulators[1] = _mm_xor_si128(accumulators[1], unsafe {
                            _mm_load_si128(source_pointer.add(offset + 16).cast())
                        });
                        accumulators[2] = _mm_xor_si128(accumulators[2], unsafe {
                            _mm_load_si128(source_pointer.add(offset + 32).cast())
                        });
                        accumulators[3] = _mm_xor_si128(accumulators[3], unsafe {
                            _mm_load_si128(source_pointer.add(offset + 48).cast())
                        });
                        accumulators[4] = _mm_xor_si128(accumulators[4], unsafe {
                            _mm_load_si128(source_pointer.add(offset + 64).cast())
                        });
                        accumulators[5] = _mm_xor_si128(accumulators[5], unsafe {
                            _mm_load_si128(source_pointer.add(offset + 80).cast())
                        });
                        accumulators[6] = _mm_xor_si128(accumulators[6], unsafe {
                            _mm_load_si128(source_pointer.add(offset + 96).cast())
                        });
                        accumulators[7] = _mm_xor_si128(accumulators[7], unsafe {
                            _mm_load_si128(source_pointer.add(offset + 112).cast())
                        });
                        offset += 128;
                    }
                    while offset < end {
                        accumulators[0] = _mm_xor_si128(accumulators[0], unsafe {
                            _mm_load_si128(source_pointer.add(offset).cast())
                        });
                        offset += 16;
                    }
                }};
            }
            for _ in 0..passes {
                read_range!(start_offset, length);
                read_range!(0, start_offset);
            }
            checksum = _mm_xor_si128(accumulators[0], accumulators[1]);
            for accumulator in &accumulators[2..] {
                checksum = _mm_xor_si128(checksum, *accumulator);
            }
        }
        Operation::Write => {
            macro_rules! write_range {
                ($begin:expr, $end:expr, $store:ident, $value:expr) => {{
                    let end = $end;
                    let mut offset = $begin;
                    while offset + 128 <= end {
                        unsafe {
                            $store(source_mut_pointer.add(offset).cast(), $value);
                            $store(source_mut_pointer.add(offset + 16).cast(), $value);
                            $store(source_mut_pointer.add(offset + 32).cast(), $value);
                            $store(source_mut_pointer.add(offset + 48).cast(), $value);
                            $store(source_mut_pointer.add(offset + 64).cast(), $value);
                            $store(source_mut_pointer.add(offset + 80).cast(), $value);
                            $store(source_mut_pointer.add(offset + 96).cast(), $value);
                            $store(source_mut_pointer.add(offset + 112).cast(), $value);
                        }
                        offset += 128;
                    }
                    while offset < end {
                        unsafe { $store(source_mut_pointer.add(offset).cast(), $value) };
                        offset += 16;
                    }
                }};
            }
            for pass in 0..passes {
                let value = _mm_set1_epi64x(pass as i64 ^ 0x5a5a5a5a);
                if streaming {
                    write_range!(start_offset, length, _mm_stream_si128, value);
                    write_range!(0, start_offset, _mm_stream_si128, value);
                } else {
                    write_range!(start_offset, length, _mm_store_si128, value);
                    write_range!(0, start_offset, _mm_store_si128, value);
                }
            }
        }
        Operation::Copy => {
            macro_rules! copy_range {
                ($begin:expr, $end:expr, $store:ident) => {{
                    let end = $end;
                    let mut offset = $begin;
                    while offset + 128 <= end {
                        let values = unsafe {
                            [
                                _mm_load_si128(source_pointer.add(offset).cast()),
                                _mm_load_si128(source_pointer.add(offset + 16).cast()),
                                _mm_load_si128(source_pointer.add(offset + 32).cast()),
                                _mm_load_si128(source_pointer.add(offset + 48).cast()),
                                _mm_load_si128(source_pointer.add(offset + 64).cast()),
                                _mm_load_si128(source_pointer.add(offset + 80).cast()),
                                _mm_load_si128(source_pointer.add(offset + 96).cast()),
                                _mm_load_si128(source_pointer.add(offset + 112).cast()),
                            ]
                        };
                        unsafe {
                            $store(destination_pointer.add(offset).cast(), values[0]);
                            $store(destination_pointer.add(offset + 16).cast(), values[1]);
                            $store(destination_pointer.add(offset + 32).cast(), values[2]);
                            $store(destination_pointer.add(offset + 48).cast(), values[3]);
                            $store(destination_pointer.add(offset + 64).cast(), values[4]);
                            $store(destination_pointer.add(offset + 80).cast(), values[5]);
                            $store(destination_pointer.add(offset + 96).cast(), values[6]);
                            $store(destination_pointer.add(offset + 112).cast(), values[7]);
                        }
                        offset += 128;
                    }
                    while offset < end {
                        let value = unsafe { _mm_load_si128(source_pointer.add(offset).cast()) };
                        unsafe { $store(destination_pointer.add(offset).cast(), value) };
                        offset += 16;
                    }
                }};
            }
            for _ in 0..passes {
                if streaming {
                    copy_range!(start_offset, length, _mm_stream_si128);
                    copy_range!(0, start_offset, _mm_stream_si128);
                } else {
                    copy_range!(start_offset, length, _mm_store_si128);
                    copy_range!(0, start_offset, _mm_store_si128);
                }
            }
        }
    }
    if streaming && !matches!(operation, Operation::Read) {
        _mm_sfence();
    }
    black_box(checksum);
}

#[cfg(not(target_arch = "x86_64"))]
fn scalar(
    operation: Operation,
    source: &mut AlignedBuffer,
    destination: &mut AlignedBuffer,
    passes: usize,
    start_offset: usize,
) {
    let mut checksum = 0u8;
    for pass in 0..passes {
        let length = source.len().min(destination.len());
        for step in 0..length {
            let offset = (start_offset + step) % length;
            unsafe {
                match operation {
                    Operation::Read => checksum ^= source.as_ptr().add(offset).read(),
                    Operation::Write => source.as_mut_ptr().add(offset).write(pass as u8),
                    Operation::Copy => destination
                        .as_mut_ptr()
                        .add(offset)
                        .write(source.as_ptr().add(offset).read()),
                }
            }
        }
    }
    black_box(checksum);
}

pub fn measure_parallel(
    operation: Operation,
    locations: &[ProcessorLocation],
    total_payload_bytes: usize,
    options: MeasurementOptions,
    cancellation: &CancellationToken,
) -> Result<(SampleStatistics, u64), BenchmarkError> {
    let thread_count = locations.len().max(1);
    let per_thread = (total_payload_bytes / thread_count)
        .max(ALIGNMENT)
        .next_multiple_of(ALIGNMENT);
    let payload_sizes = vec![per_thread; thread_count];
    measure_parallel_sizes(operation, locations, &payload_sizes, options, cancellation)
}

pub fn measure_parallel_sizes(
    operation: Operation,
    locations: &[ProcessorLocation],
    payload_sizes: &[usize],
    options: MeasurementOptions,
    cancellation: &CancellationToken,
) -> Result<(SampleStatistics, u64), BenchmarkError> {
    if locations.is_empty() || locations.len() != payload_sizes.len() {
        return Err(BenchmarkError::new(
            "invalid_topology",
            "Parallel benchmark locations and payload sizes must be non-empty and equal in length.",
        ));
    }
    let total_payload_bytes = payload_sizes.iter().try_fold(0usize, |total, size| {
        total.checked_add(*size).ok_or_else(|| {
            BenchmarkError::new(
                "working_set_too_large",
                "Aggregate cache payload overflowed.",
            )
        })
    })?;
    let (visited_payload_bytes, auxiliary_bytes) = {
        let _phase = gluj_bench_core::phase_timer("preparation");
        let mut visited = 0;
        let mut auxiliary = 0;
        for (worker, bytes) in payload_sizes.iter().copied().enumerate() {
            let schedule = options
                .access
                .schedule(bytes, 64, worker, 0, cancellation)?;
            visited += schedule.visited_bytes();
            auxiliary += schedule.auxiliary_bytes();
        }
        (visited, auxiliary)
    };
    let thread_count = locations.len();
    let preparation_timer = gluj_bench_core::phase_timer("preparation");
    let mut buffers = Vec::with_capacity(thread_count);
    for payload_size in payload_sizes.iter().copied() {
        buffers.push((
            AlignedBuffer::new(payload_size)?,
            AlignedBuffer::new(payload_size)?,
        ));
    }
    drop(preparation_timer);
    let calibration_ns = parallel_once(
        operation,
        locations,
        &mut buffers,
        SweepOptions {
            access: options.access,
            realization: 0,
            rotation: 0,
            passes: 1,
            chunk_passes: 1,
            streaming: options.streaming,
            preparation: options.preparation,
            first_touch: true,
        },
        cancellation,
    )?;
    let mut passes = ((options.target_ns / calibration_ns).max(1) as usize).min(MAX_PASSES);
    if options.preparation == CachePreparation::Hot {
        let probe_passes = ((100_000_000 / calibration_ns).max(1) as usize).min(MAX_PASSES);
        let probe_ns = parallel_once(
            operation,
            locations,
            &mut buffers,
            SweepOptions {
                access: options.access,
                realization: 0,
                rotation: 0,
                passes: probe_passes,
                chunk_passes: probe_passes,
                streaming: options.streaming,
                preparation: options.preparation,
                first_touch: false,
            },
            cancellation,
        )?;
        let scaled_passes = options.target_ns as u128 * probe_passes as u128 / probe_ns as u128;
        passes = scaled_passes.clamp(1, MAX_PASSES as u128) as usize;
    }
    let chunk_passes = ((50_000_000 / calibration_ns).max(1) as usize).min(passes);
    let mut values = Vec::with_capacity(options.sample_count as usize);
    let mut elapsed_total = 0u64;
    let mut sample_passes = passes;
    let mut cold_pacer = crate::pacing::CpuPacer::current(cancellation);
    for sample in 0..options.sample_count * options.access.realizations {
        let realization = sample % options.access.realizations;
        let observation_started = std::time::Instant::now();
        if cancellation.is_cancelled() {
            return Err(BenchmarkError::new(
                "cancelled",
                "The benchmark was cancelled.",
            ));
        }
        let mut window = gluj_bench_core::SampleWindow::new(options.target_ns);
        let mut completed_passes = 0usize;
        while window.needs_batch() {
            let wall_ns = if options.preparation == CachePreparation::Cold {
                let mut elapsed = 0u64;
                for pass in 0..sample_passes {
                    let active_ns = parallel_once(
                        operation,
                        locations,
                        &mut buffers,
                        SweepOptions {
                            access: options.access,
                            realization,
                            rotation: completed_passes + pass,
                            passes: 1,
                            chunk_passes: 1,
                            streaming: options.streaming,
                            preparation: options.preparation,
                            first_touch: false,
                        },
                        cancellation,
                    )?;
                    elapsed = elapsed.saturating_add(active_ns);
                    cold_pacer.account(std::time::Duration::from_nanos(active_ns))?;
                }
                elapsed
            } else {
                parallel_once(
                    operation,
                    locations,
                    &mut buffers,
                    SweepOptions {
                        access: options.access,
                        realization,
                        rotation: completed_passes,
                        passes: sample_passes,
                        chunk_passes,
                        streaming: options.streaming,
                        preparation: options.preparation,
                        first_touch: false,
                    },
                    cancellation,
                )?
            };
            window.add(wall_ns)?;
            completed_passes = completed_passes.saturating_add(sample_passes);
        }
        let wall_ns = window.finish()?;
        elapsed_total = elapsed_total.saturating_add(wall_ns);
        values.push(bandwidth_value(
            visited_payload_bytes,
            options.byte_multiplier,
            completed_passes,
            wall_ns,
        ));
        if let Some(recorder) = gluj_bench_core::observation_recorder() {
            recorder
                .with_realization(
                    options
                        .access
                        .derived_seed(0, total_payload_bytes, realization),
                    std::collections::BTreeMap::from([
                        ("root_seed".into(), options.access.seed.to_string()),
                        ("realization".into(), realization.to_string()),
                        ("access_pattern".into(), options.access.name().into()),
                        ("worker".into(),"aggregate".into()),("dataset_bytes".into(),total_payload_bytes.to_string()),("schedule_auxiliary_bytes".into(),auxiliary_bytes.to_string()),
                        ("worker_seed_derivation".into(),"derive(root,worker,aligned_worker_payload,realization); rotation begins at zero per observation and advances per complete pass".into()),
                    ]),
                )
                .record(
                    &format!(
                        "{operation:?}:payload-{total_payload_bytes}:streaming-{}",
                        options.streaming
                    ),
                    "bandwidth",
                    *values.last().unwrap(),
                    "bytes/s",
                    (visited_payload_bytes as u64)
                        .saturating_mul(u64::from(options.byte_multiplier))
                        .saturating_mul(completed_passes as u64),
                    "accounted_bytes",
                    wall_ns,
                    observation_started.elapsed().as_nanos() as u64,
                    "not_recorded",
                );
        }
        let adjusted_passes =
            options.target_ns as u128 * completed_passes as u128 / wall_ns as u128;
        sample_passes = adjusted_passes.clamp(1, MAX_PASSES as u128) as usize;
    }
    let _validation_timer = gluj_bench_core::phase_timer("validation");
    for (source, destination) in &buffers {
        let first = unsafe { source.as_ptr().read() };
        for index in gluj_bench_core::validation_indices(source.len() / 8) {
            let source_word = unsafe {
                source
                    .as_ptr()
                    .add(index * 8)
                    .cast::<u64>()
                    .read_unaligned()
            };
            let valid = match operation {
                Operation::Read => first & 1 != 0 && source_word == u64::from_ne_bytes([first; 8]),
                Operation::Copy => {
                    source_word
                        == unsafe {
                            destination
                                .as_ptr()
                                .add(index * 8)
                                .cast::<u64>()
                                .read_unaligned()
                        }
                }
                Operation::Write => {
                    let first_word = unsafe { source.as_ptr().cast::<u64>().read_unaligned() };
                    source_word == first_word
                        && (!cfg!(target_arch = "x86_64") || first_word >> 32 == 0)
                }
            };
            if !valid {
                return Err(BenchmarkError::new(
                    "result_validation_failed",
                    "Bandwidth read-integrity/write/copy boundary or interior check failed.",
                ));
            }
        }
    }
    Ok((statistics(&values), elapsed_total))
}

fn bandwidth_value(payload_bytes: usize, multiplier: u8, passes: usize, elapsed_ns: u64) -> f64 {
    payload_bytes as f64 * multiplier as f64 * passes as f64 * 1_000_000_000.0
        / elapsed_ns.max(1) as f64
}

fn parallel_once(
    operation: Operation,
    locations: &[ProcessorLocation],
    buffers: &mut [(AlignedBuffer, AlignedBuffer)],
    options: SweepOptions,
    cancellation: &CancellationToken,
) -> Result<u64, BenchmarkError> {
    let barrier = Arc::new(Barrier::new(locations.len()));
    let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let intensity = gluj_bench_core::cpu_activity_percent();
    let recorder = gluj_bench_core::observation_recorder();
    let elapsed = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(locations.len());
        for (thread_index, ((source, destination), location)) in buffers
            .iter_mut()
            .zip(locations.iter().copied())
            .enumerate()
        {
            let recorder = recorder.clone();
            let barrier = barrier.clone();
            let failed = failed.clone();
            handles.push(scope.spawn(move || -> Result<u64, BenchmarkError> {
                let prepared = (|| {
                    let _phase = recorder.as_ref().map(|r| r.phase_timer("preparation"));
                    let affinity = AffinityGuard::pin(location)?;
                    let schedule = options.access.schedule(
                        source.len().min(destination.len()),
                        64,
                        thread_index,
                        options.realization,
                        cancellation,
                    )?;
                    Ok::<_, BenchmarkError>((affinity, schedule))
                })();
                if prepared.is_err() {
                    failed.store(true, Ordering::SeqCst);
                }
                barrier.wait();
                if failed.load(Ordering::SeqCst) {
                    return Err(prepared.err().unwrap_or_else(|| {
                        BenchmarkError::new(
                            "worker_prepare_failed",
                            "Another bandwidth worker could not set processor placement.",
                        )
                    }));
                }
                let (_affinity, mut schedule) = prepared?;
                schedule.reset_pass(options.rotation);
                let preparation_timer = recorder
                    .as_ref()
                    .map(|recorder| recorder.phase_timer("preparation"));
                if options.first_touch {
                    source.initialize(
                        options.access.derived_seed(thread_index, source.len(), 0) as u8 | 1,
                    );
                    destination.initialize(0);
                }
                drop(preparation_timer);
                let cache_timer = recorder.as_ref().map(|recorder| {
                    recorder.phase_timer(if options.preparation == CachePreparation::Hot {
                        "warmup"
                    } else {
                        "preparation"
                    })
                });
                prepare(
                    operation,
                    source,
                    destination,
                    options.preparation,
                    cancellation,
                );
                let start_offset = if options.preparation == CachePreparation::Cold {
                    rotated_offset(
                        options.access.derived_seed(
                            thread_index,
                            source.len(),
                            options.realization,
                        ) ^ options.rotation as u64,
                        source.len().min(destination.len()),
                    )
                } else {
                    0
                };
                drop(cache_timer);
                barrier.wait();
                let mut pacer = crate::pacing::CpuPacer::new(
                    if options.preparation == CachePreparation::Hot {
                        intensity
                    } else {
                        100
                    },
                    cancellation,
                );
                let start = Instant::now();
                let idle_before = pacer.idle();
                let mut remaining = options.passes;
                while remaining != 0 {
                    if cancellation.is_cancelled() {
                        return Err(BenchmarkError::new(
                            "cancelled",
                            "The benchmark was cancelled.",
                        ));
                    }
                    let current = remaining.min(options.chunk_passes);
                    let batch_start = Instant::now();
                    if options.access.pattern == gluj_bench_core::AccessPattern::Sequential {
                        execute_rotated(
                            operation,
                            source,
                            destination,
                            current,
                            options.streaming,
                            start_offset,
                        );
                    } else {
                        for _ in 0..current {
                            if !schedule.visit(cancellation, |range| {
                                execute_range(
                                    operation,
                                    source,
                                    destination,
                                    range,
                                    options.streaming,
                                )
                            }) {
                                return Err(BenchmarkError::new(
                                    "cancelled",
                                    "Bandwidth traversal cancelled.",
                                ));
                            }
                        }
                    }
                    remaining -= current;
                    pacer.account(batch_start.elapsed())?;
                }
                Ok(pacer.active_elapsed(start, idle_before).as_nanos().max(1) as u64)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    BenchmarkError::new("worker_failed", "A RAM benchmark thread panicked.")
                })?
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    Ok(elapsed.into_iter().max().unwrap_or(1))
}

fn rotated_offset(mut value: u64, length: usize) -> usize {
    let lines = length / ALIGNMENT;
    if lines <= 1 {
        return 0;
    }
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    value as usize % lines * ALIGNMENT
}

fn prepare(
    operation: Operation,
    source: &mut AlignedBuffer,
    destination: &mut AlignedBuffer,
    preparation: CachePreparation,
    cancellation: &CancellationToken,
) {
    if preparation == CachePreparation::Hot {
        execute(operation, source, destination, 1, false);
        return;
    }
    flush_from_cache(source, cancellation);
    if matches!(operation, Operation::Copy) {
        flush_from_cache(destination, cancellation);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_mfence();
    }
}

#[cfg(target_arch = "x86_64")]
fn flush_from_cache(buffer: &AlignedBuffer, cancellation: &CancellationToken) {
    use std::arch::x86_64::_mm_clflush;
    for offset in (0..buffer.len()).step_by(ALIGNMENT) {
        if offset.is_multiple_of(256 * 1024) && cancellation.is_cancelled() {
            break;
        }
        unsafe { _mm_clflush(buffer.as_ptr().add(offset)) };
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn flush_from_cache(buffer: &AlignedBuffer, cancellation: &CancellationToken) {
    for offset in (0..buffer.len()).step_by(ALIGNMENT) {
        if offset.is_multiple_of(256 * 1024) && cancellation.is_cancelled() {
            break;
        }
        black_box(unsafe { buffer.as_ptr().add(offset).read_volatile() });
    }
}

pub fn statistics(values: &[f64]) -> SampleStatistics {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / sorted.len() as f64;
    SampleStatistics {
        sample_count: sorted.len() as u32,
        minimum: sorted[0],
        median,
        maximum: *sorted.last().unwrap(),
        standard_deviation: variance.sqrt(),
    }
}

fn execute_range(
    operation: Operation,
    source: &mut AlignedBuffer,
    destination: &mut AlignedBuffer,
    range: std::ops::Range<usize>,
    streaming: bool,
) {
    assert!(
        range.start.is_multiple_of(64)
            && range.len().is_multiple_of(64)
            && range.end <= source.len().min(destination.len())
    );
    // SAFETY: schedules give disjoint aligned ranges within both live exclusive buffers.
    // These non-owning views must not deallocate the original allocations.
    let mut a = std::mem::ManuallyDrop::new(AlignedBuffer {
        pointer: unsafe { NonNull::new_unchecked(source.as_mut_ptr().add(range.start)) },
        length: range.len(),
        layout: source.layout,
    });
    let mut b = std::mem::ManuallyDrop::new(AlignedBuffer {
        pointer: unsafe { NonNull::new_unchecked(destination.as_mut_ptr().add(range.start)) },
        length: range.len(),
        layout: destination.layout,
    });
    execute_rotated(operation, &mut a, &mut b, 1, streaming, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn statistics_are_per_metric() {
        let s = statistics(&[1.0, 3.0, 2.0]);
        assert_eq!(s.median, 2.0);
        assert_eq!(s.sample_count, 3);
    }
    #[test]
    fn copy_traffic_accounting_counts_read_and_write() {
        let useful = bandwidth_value(4096, 1, 10, 1_000);
        let read_plus_write = bandwidth_value(4096, 2, 10, 1_000);
        assert_eq!(read_plus_write, useful * 2.0);
    }
    #[test]
    fn kernels_modify_and_copy_buffers() {
        let mut a = AlignedBuffer::new(4096).unwrap();
        let mut b = AlignedBuffer::new(4096).unwrap();
        a.initialize(7);
        execute(Operation::Copy, &mut a, &mut b, 1, false);
        assert_eq!(unsafe { b.as_ptr().read() }, 7);
        execute(Operation::Write, &mut a, &mut b, 1, false);
        assert_ne!(unsafe { a.as_ptr().read() }, 7);
    }

    #[test]
    fn rotated_copy_still_covers_the_complete_payload() {
        let mut source = AlignedBuffer::new(4096).unwrap();
        let mut destination = AlignedBuffer::new(4096).unwrap();
        source.initialize(11);
        execute_rotated(
            Operation::Copy,
            &mut source,
            &mut destination,
            1,
            false,
            17 * ALIGNMENT,
        );
        assert_eq!(unsafe { destination.as_ptr().read() }, 11);
        assert_eq!(unsafe { destination.as_ptr().add(4095).read() }, 11);
    }
}

#[cfg(test)]
mod access_regressions {
    use super::*;
    #[test]
    fn scattered_and_hot_cold_bandwidth_ranges_copy_the_full_payload() {
        let token = CancellationToken::default();
        for pattern in [
            "shuffled_blocks",
            "rotating_regions",
            "scattered",
            "hot_cold",
        ] {
            let config = gluj_bench_core::BenchmarkConfig {
                options: std::collections::BTreeMap::from([
                    ("access_pattern".into(), pattern.into()),
                    ("access_block_bytes".into(), "192".into()),
                ]),
                ..Default::default()
            };
            let mut schedule = gluj_bench_core::AccessPolicy::from_config(&config)
                .unwrap()
                .schedule(64 * 17, 64, 0, 0, &token)
                .unwrap();
            let mut source = AlignedBuffer::new(64 * 17).unwrap();
            let mut destination = AlignedBuffer::new(64 * 17).unwrap();
            for i in 0..source.len() {
                unsafe {
                    source.as_mut_ptr().add(i).write((i % 251) as u8);
                }
            }
            assert!(schedule.visit(&token, |range| execute_range(
                Operation::Copy,
                &mut source,
                &mut destination,
                range,
                false
            )));
            for i in 0..source.len() {
                assert_eq!(unsafe { source.as_ptr().add(i).read() }, unsafe {
                    destination.as_ptr().add(i).read()
                });
            }
        }
    }
}
