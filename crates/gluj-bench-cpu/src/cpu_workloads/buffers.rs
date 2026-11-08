use super::*;
use flate2::{Compress, Compression, Decompress, FlushCompress, FlushDecompress, Status};
use std::hint::black_box;

pub(super) fn vector<T: Clone>(
    len: usize,
    value: T,
    token: &CancellationToken,
) -> Result<Vec<T>, BenchmarkError> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|e| error("allocation_failed", e.to_string()))?;
    for start in (0..len).step_by(65536) {
        check(token)?;
        data.resize(len.min(start + 65536), value.clone());
    }
    Ok(data)
}
fn value(index: usize) -> f64 {
    1.0 + (index % 251) as f64 / 4096.0
}
pub(super) fn pop_reference(mut value: u64) -> u64 {
    let mut count = 0;
    while value != 0 {
        value &= value - 1;
        count += 1;
    }
    count
}
fn word(seed: u64, index: usize) -> u64 {
    gluj_bench_core::derive_seed(seed, 0, index as u64, 0)
}
pub(super) struct Block {
    source: Vec<u8>,
    compressed: Vec<u8>,
    output: Vec<u8>,
}
pub(super) enum State {
    Float {
        a: Vec<f64>,
        b: Vec<f64>,
        out: Vec<f64>,
        simd: bool,
    },
    Pop {
        input: Vec<u64>,
        expected: u64,
        found: u64,
    },
    Codec {
        blocks: Vec<Block>,
        compress: Compress,
        decompress: Decompress,
        decode: bool,
        expected_bytes: u64,
    },
    Sieve {
        marks: Vec<u8>,
        expected: u64,
        found: u64,
    },
    Matrix {
        a: Vec<f32>,
        b: Arc<Vec<f32>>,
        out: Vec<f32>,
        rows: usize,
        n: usize,
        k: usize,
        reuse: u64,
        block: usize,
    },
}
fn partition(total: usize, worker: usize, workers: usize) -> (usize, usize) {
    let start = total * worker / workers;
    let end = total * (worker + 1) / workers;
    (start, end - start)
}
impl State {
    pub(super) fn new(
        spec: Spec,
        config: &BenchmarkConfig,
        worker: usize,
        workers: usize,
        payload: u64,
        shared: &Shared,
        token: &CancellationToken,
    ) -> Result<Self, BenchmarkError> {
        let seed = gluj_bench_core::AccessPolicy::from_config(config)?.seed;
        match spec {
            Spec::Float(simd) => {
                let (start, blocks) = partition(payload as usize / 24 / 64, worker, workers);
                let elements = blocks * 64;
                let mut a = vector(elements, 0.0, token)?;
                let mut b = vector(elements, 0.0, token)?;
                for i in 0..elements {
                    if i % 65536 == 0 {
                        check(token)?;
                    }
                    a[i] = value(start * 64 + i);
                    b[i] = 0.999_999_999;
                }
                Ok(Self::Float {
                    a,
                    b,
                    out: vector(elements, 0.0, token)?,
                    simd,
                })
            }
            Spec::Pop => {
                let (start, blocks) = partition(payload as usize / 8 / 64, worker, workers);
                let mut input = vector(blocks * 64, 0, token)?;
                let mut expected = 0;
                for (i, destination) in input.iter_mut().enumerate() {
                    if i % 65536 == 0 {
                        check(token)?;
                    }
                    *destination = word(seed, start * 64 + i);
                    expected += pop_reference(*destination);
                }
                Ok(Self::Pop {
                    input,
                    expected,
                    found: 0,
                })
            }
            Spec::Codec(decode) => {
                let bytes = number(config, "codec_corpus_bytes", 16777216) as usize;
                let block = number(config, "codec_block_bytes", 65536) as usize;
                let (start, count) = partition(bytes.div_ceil(block), worker, workers);
                let level = number(config, "codec_level", 6) as u32;
                let corpus = config
                    .options
                    .get("codec_corpus")
                    .map(String::as_str)
                    .unwrap_or("mixed");
                let mut blocks = Vec::new();
                blocks
                    .try_reserve_exact(count)
                    .map_err(|e| error("allocation_failed", e.to_string()))?;
                let mut compressor = Compress::new(Compression::new(level), false);
                let mut expected_bytes = 0;
                for index in start..start + count {
                    check(token)?;
                    let length = block.min(bytes - index * block);
                    let mut source = vector(length, 0, token)?;
                    for (i, byte) in source.iter_mut().enumerate() {
                        if i % 65536 == 0 {
                            check(token)?;
                        }
                        let global = index * block + i;
                        *byte = match corpus {
                            "compressible" => b'A' + (global % 7) as u8,
                            "incompressible" => word(seed, global) as u8,
                            _ => {
                                if (global / 4096).is_multiple_of(2) {
                                    b'A' + (global % 7) as u8
                                } else {
                                    word(seed, global) as u8
                                }
                            }
                        };
                    }
                    let mut compressed = Vec::new();
                    compressed
                        .try_reserve_exact(length + length / 16 + 1024)
                        .map_err(|e| error("allocation_failed", e.to_string()))?;
                    compressor.reset();
                    let status = compressor
                        .compress_vec(&source, &mut compressed, FlushCompress::Finish)
                        .map_err(|e| error("codec_failed", e.to_string()))?;
                    if status != Status::StreamEnd || compressor.total_in() != length as u64 {
                        return Err(error("codec_failed", "Incomplete prepared DEFLATE stream."));
                    }
                    if miniz_oxide::inflate::decompress_to_vec(&compressed)
                        .map_err(|e| error("validation_failed", format!("{e:?}")))?
                        != source
                    {
                        return Err(error(
                            "validation_failed",
                            "Independent DEFLATE round trip differs.",
                        ));
                    }
                    let mut output = Vec::new();
                    output
                        .try_reserve_exact(if decode {
                            length
                        } else {
                            length + length / 16 + 1024
                        })
                        .map_err(|e| error("allocation_failed", e.to_string()))?;
                    expected_bytes += length as u64;
                    blocks.push(Block {
                        source,
                        compressed,
                        output,
                    });
                }
                Ok(Self::Codec {
                    blocks,
                    compress: compressor,
                    decompress: Decompress::new(false),
                    decode,
                    expected_bytes,
                })
            }
            Spec::Sieve => Ok(Self::Sieve {
                marks: vector(
                    number(config, "sieve_limit", 1000000) as usize + 1,
                    1,
                    token,
                )?,
                expected: shared.primes,
                found: 0,
            }),
            Spec::Matrix => {
                let [m, n, k] = shared.shape;
                let (start, groups) = partition(m / 8, worker, workers);
                let rows = groups * 8;
                let mut a = vector(rows * k, 0.0, token)?;
                for (i, v) in a.iter_mut().enumerate() {
                    *v = (((start * 8 * k + i) % 17) as f32 - 8.0) / 32.0;
                }
                Ok(Self::Matrix {
                    a,
                    b: shared.matrix_b.clone(),
                    out: vector(rows * n, 0.0, token)?,
                    rows,
                    n,
                    k,
                    reuse: number(config, "matrix_reuse", 1),
                    block: number(config, "matrix_k_block", 128) as usize,
                })
            }
        }
    }
    pub(super) fn allocated_buffers(&self) -> u64 {
        match self {
            Self::Float { a, b, out, .. } => {
                (a.capacity() + b.capacity() + out.capacity()) as u64 * 8
            }
            Self::Pop { input, .. } => input.capacity() as u64 * 8,
            Self::Codec { blocks, .. } => blocks
                .iter()
                .map(|b| {
                    (b.source.capacity() + b.compressed.capacity() + b.output.capacity()) as u64
                })
                .sum(),
            Self::Sieve { marks, .. } => marks.capacity() as u64,
            Self::Matrix { a, out, .. } => (a.capacity() + out.capacity()) as u64 * 4,
        }
    }
    pub(super) fn expected_population(&self) -> u64 {
        match self {
            Self::Pop { expected, .. } => *expected,
            _ => 0,
        }
    }
    pub(super) fn prepare_reference(
        &mut self,
        schedule: &mut gluj_bench_core::AccessSchedule,
        token: &CancellationToken,
    ) -> Result<(), BenchmarkError> {
        if let Self::Pop {
            input, expected, ..
        } = self
        {
            let mut sum = 0;
            let complete = schedule.visit(token, |range| {
                for i in range {
                    sum += pop_reference(input[i]);
                }
            });
            if !complete {
                check(token)?;
            }
            *expected = sum;
            schedule.reset_pass(0);
        }
        Ok(())
    }
    pub(super) fn span(&self) -> usize {
        match self {
            Self::Float { a, .. } => a.len(),
            Self::Pop { input, .. } => input.len(),
            Self::Codec { blocks, .. } => blocks.len() * 64,
            _ => 64,
        }
    }
    pub(super) fn work(
        &mut self,
        schedule: &mut gluj_bench_core::AccessSchedule,
        token: &CancellationToken,
    ) -> Result<(u64, u64), BenchmarkError> {
        match self {
            Self::Float { a, b, out, simd } => {
                let mut elements = 0;
                let complete = schedule.visit(token, |range| {
                    elements += range.len() as u64;
                    if *simd {
                        unsafe {
                            fp64_fma(&a[range.clone()], &b[range.clone()], &mut out[range]);
                        }
                    } else {
                        for i in range {
                            let mut v = a[i];
                            for _ in 0..16 {
                                v = black_box(v) * b[i] + 0.000_001;
                            }
                            out[i] = v;
                        }
                    }
                });
                if !complete {
                    check(token)?;
                }
                Ok((elements * 32, out[0].to_bits()))
            }
            Self::Pop { input, found, .. } => {
                let mut count = 0;
                let mut sum = 0;
                let complete = schedule.visit(token, |range| {
                    for i in range {
                        sum += unsafe { population(input[i]) };
                        count += 1;
                    }
                });
                if !complete {
                    check(token)?;
                }
                *found = sum;
                Ok((count, sum))
            }
            Self::Codec {
                blocks,
                compress,
                decompress,
                decode,
                ..
            } => {
                let mut bytes = 0;
                let mut checksum = 0;
                let mut failed = None;
                let complete = schedule.visit(token, |range| {
                    for index in (range.start..range.end).step_by(64) {
                        if failed.is_some() {
                            continue;
                        }
                        let block = &mut blocks[index / 64];
                        block.output.clear();
                        let valid = if *decode {
                            decompress.reset(false);
                            decompress
                                .decompress_vec(
                                    &block.compressed,
                                    &mut block.output,
                                    FlushDecompress::Finish,
                                )
                                .is_ok_and(|s| s == Status::StreamEnd)
                                && decompress.total_out() == block.source.len() as u64
                        } else {
                            compress.reset();
                            compress
                                .compress_vec(
                                    &block.source,
                                    &mut block.output,
                                    FlushCompress::Finish,
                                )
                                .is_ok_and(|s| s == Status::StreamEnd)
                                && compress.total_in() == block.source.len() as u64
                        };
                        if !valid {
                            failed = Some(error(
                                "codec_failed",
                                "DEFLATE did not finish a complete independent block.",
                            ));
                        } else {
                            bytes += block.source.len() as u64;
                            checksum ^= block.output.len() as u64;
                        }
                    }
                });
                if !complete {
                    check(token)?;
                }
                if let Some(e) = failed {
                    return Err(e);
                }
                Ok((bytes, checksum))
            }
            Self::Sieve {
                marks,
                expected,
                found,
            } => {
                let limit = marks.len() - 1;
                for chunk in marks.chunks_mut(65536) {
                    check(token)?;
                    chunk.fill(1);
                }
                marks[0] = 0;
                marks[1] = 0;
                for prime in 2.. {
                    if prime > limit / prime {
                        break;
                    }
                    if marks[prime] != 0 {
                        for (step, composite) in (prime * prime..=limit).step_by(prime).enumerate()
                        {
                            if step % 4096 == 0 {
                                check(token)?;
                            }
                            marks[composite] = 0;
                        }
                    }
                }
                *found = 0;
                for chunk in marks.chunks(65536) {
                    check(token)?;
                    *found += chunk.iter().map(|b| u64::from(*b)).sum::<u64>();
                }
                if *found != *expected {
                    return Err(error(
                        "validation_failed",
                        "Complete sieve count differs from its independent reference.",
                    ));
                }
                Ok((*found, *found))
            }
            Self::Matrix {
                a,
                b,
                out,
                rows,
                n,
                k,
                reuse,
                block,
            } => {
                for _ in 0..*reuse {
                    unsafe {
                        matrix(*rows, *n, *k, *block, a, b, out, token)?;
                    }
                    black_box(&*out);
                }
                Ok((
                    2 * (*rows as u64) * (*n as u64) * (*k as u64) * *reuse,
                    u64::from(out[0].to_bits()),
                ))
            }
        }
    }
    pub(super) fn validate(&self) -> Result<(), BenchmarkError> {
        let valid = match self {
            Self::Float { a, b, out, simd } => gluj_bench_core::validation_indices(a.len())
                .into_iter()
                .all(|i| {
                    let mut v = a[i];
                    for _ in 0..16 {
                        v = if *simd {
                            v.mul_add(b[i], 0.000_001)
                        } else {
                            v * b[i] + 0.000_001
                        };
                    }
                    out[i].is_finite() && (out[i] - v).abs() < v.abs().max(1.0) * 1e-12
                }),
            Self::Pop {
                expected, found, ..
            } => found == expected,
            Self::Codec { blocks, decode, .. } => blocks.iter().all(|block| {
                if *decode {
                    block.output == block.source
                } else {
                    miniz_oxide::inflate::decompress_to_vec(&block.output)
                        .is_ok_and(|v| v == block.source)
                }
            }),
            Self::Sieve {
                marks, expected, ..
            } => marks.iter().map(|v| u64::from(*v)).sum::<u64>() == *expected,
            Self::Matrix {
                a, b, out, n, k, ..
            } => gluj_bench_core::validation_indices(out.len())
                .into_iter()
                .all(|i| {
                    let row = i / *n;
                    let col = i % *n;
                    let expected = (0..*k)
                        .map(|t| f64::from(a[row * *k + t]) * f64::from(b[t * *n + col]))
                        .sum::<f64>();
                    out[i].is_finite()
                        && (f64::from(out[i]) - expected).abs() <= expected.abs().max(1.0) * 1e-5
                }),
        };
        if valid {
            Ok(())
        } else {
            Err(error(
                "validation_failed",
                "CPU workload output differs from its independent reference.",
            ))
        }
    }
    pub(super) fn codec_sizes(&self) -> (u64, u64) {
        match self {Self::Codec {blocks,decode,expected_bytes,..}=>(*expected_bytes,blocks.iter().map(|b|if *decode {b.compressed.len()} else {b.output.len()} as u64).sum()),_=>(0,0)}
    }
    pub(super) fn work_bytes(&self, count: u64) -> u64 {
        match self {
            Self::Float { .. } => count / 32 * 24,
            Self::Pop { .. } => count * 8,
            Self::Codec { .. } => count,
            Self::Sieve {
                marks, expected, ..
            } => count / *expected * marks.len() as u64,
            Self::Matrix {
                rows,
                n,
                k,
                reuse,
                block,
                ..
            } => {
                let per = 2 * (*rows as u64) * (*n as u64) * (*k as u64) * *reuse;
                count / per
                    * (((*rows * *n * *k) as u64
                        + 4 * (*rows * *n) as u64 * (2 * k.div_ceil(*block) as u64 - 1))
                        * *reuse)
            }
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn population(v: u64) -> u64 {
    std::arch::x86_64::_popcnt64(black_box(v) as i64) as u64
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn population(v: u64) -> u64 {
    pop_reference(v)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fp64_fma(a: &[f64], b: &[f64], out: &mut [f64]) {
    use std::arch::x86_64::*;
    for base in (0..a.len()).step_by(32) {
        let mut values = [_mm256_setzero_pd(); 8];
        let mut factors = values;
        for lane in 0..8 {
            values[lane] = unsafe { _mm256_loadu_pd(a.as_ptr().add(base + lane * 4)) };
            factors[lane] = unsafe { _mm256_loadu_pd(b.as_ptr().add(base + lane * 4)) };
        }
        for _ in 0..16 {
            for lane in 0..8 {
                values[lane] =
                    _mm256_fmadd_pd(values[lane], factors[lane], _mm256_set1_pd(0.000_001));
            }
        }
        for (lane, v) in values.iter().enumerate() {
            unsafe {
                _mm256_storeu_pd(out.as_mut_ptr().add(base + lane * 4), *v);
            }
        }
    }
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn fp64_fma(_: &[f64], _: &[f64], _: &mut [f64]) {
    unreachable!()
}
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn matrix(
    m: usize,
    n: usize,
    k: usize,
    block: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    token: &CancellationToken,
) -> Result<(), BenchmarkError> {
    use std::arch::x86_64::*;
    for row in (0..m).step_by(8) {
        for col in (0..n).step_by(8) {
            for begin in (0..k).step_by(block) {
                check(token)?;
                let mut accum = [_mm256_setzero_ps(); 8];
                if begin != 0 {
                    for (lane, v) in accum.iter_mut().enumerate() {
                        *v = unsafe { _mm256_loadu_ps(c.as_ptr().add((row + lane) * n + col)) };
                    }
                }
                for t in begin..k.min(begin + block) {
                    let weights = unsafe { _mm256_loadu_ps(b.as_ptr().add(t * n + col)) };
                    for (lane, v) in accum.iter_mut().enumerate() {
                        *v = _mm256_fmadd_ps(_mm256_set1_ps(a[(row + lane) * k + t]), weights, *v);
                    }
                }
                for (lane, v) in accum.iter().enumerate() {
                    unsafe {
                        _mm256_storeu_ps(c.as_mut_ptr().add((row + lane) * n + col), *v);
                    }
                }
            }
        }
    }
    Ok(())
}
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn matrix(
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: &[f32],
    _: &[f32],
    _: &mut [f32],
    _: &CancellationToken,
) -> Result<(), BenchmarkError> {
    Err(error("unavailable", "Matrix requires AVX2/FMA."))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(options: &[(&str, &str)]) -> BenchmarkConfig {
        BenchmarkConfig {
            options: options
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..Default::default()
        }
    }
    fn schedule(
        state: &State,
        pattern: &str,
        token: &CancellationToken,
    ) -> gluj_bench_core::AccessSchedule {
        let c = config(&[("access_pattern", pattern), ("access_block_bytes", "64")]);
        gluj_bench_core::AccessPolicy::from_config(&c)
            .unwrap()
            .schedule(state.span(), 64, 0, 1, token)
            .unwrap()
    }
    #[test]
    fn data_kernels_validate_for_every_pattern_and_keep_inputs_fixed_across_cores() {
        let token = CancellationToken::default();
        let shared = Shared {
            matrix_b: Arc::new(Vec::new()),
            shape: [0, 0, 0],
            primes: 0,
        };
        for spec in [Spec::Float(false), Spec::Float(true), Spec::Pop] {
            if matches!(spec, Spec::Float(true))
                && !(std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx2"))
            {
                continue;
            }
            let payload = 64 * 17 * spec.unit();
            let mut baseline = Vec::new();
            for workers in [1, 2, 4] {
                let mut inputs = Vec::new();
                let mut work = 0;
                for worker in 0..workers {
                    let mut state = State::new(
                        spec,
                        &Default::default(),
                        worker,
                        workers,
                        payload,
                        &shared,
                        &token,
                    )
                    .unwrap();
                    match &state {
                        State::Float { a, .. } => inputs.extend(a.iter().map(|v| v.to_bits())),
                        State::Pop { input, .. } => inputs.extend(input),
                        _ => unreachable!(),
                    }
                    for pattern in [
                        "sequential",
                        "shuffled_blocks",
                        "rotating_regions",
                        "scattered",
                        "hot_cold",
                    ] {
                        let mut order = schedule(&state, pattern, &token);
                        state.prepare_reference(&mut order, &token).unwrap();
                        let (count, _) = state.work(&mut order, &token).unwrap();
                        state.validate().unwrap();
                        if pattern == "sequential" {
                            work += count;
                        }
                    }
                }
                assert_eq!(
                    work,
                    payload / spec.unit() * if matches!(spec, Spec::Pop) { 1 } else { 32 }
                );
                if baseline.is_empty() {
                    baseline = inputs;
                } else {
                    assert_eq!(baseline, inputs);
                }
            }
        }
    }
    #[test]
    fn codec_levels_corpora_and_shuffled_blocks_round_trip_with_reused_outputs() {
        let token = CancellationToken::default();
        let shared = Shared {
            matrix_b: Arc::new(Vec::new()),
            shape: [0, 0, 0],
            primes: 0,
        };
        for level in ["0", "1", "6", "9"] {
            for corpus in ["compressible", "mixed", "incompressible"] {
                for decode in [false, true] {
                    let c = config(&[
                        ("codec_corpus_bytes", "65792"),
                        ("codec_block_bytes", "4096"),
                        ("codec_level", level),
                        ("codec_corpus", corpus),
                    ]);
                    let mut state =
                        State::new(Spec::Codec(decode), &c, 0, 1, 65792, &shared, &token).unwrap();
                    let mut order = schedule(&state, "shuffled_blocks", &token);
                    let before = match &state {
                        State::Codec { blocks, .. } => {
                            blocks.iter().map(|b| b.output.as_ptr()).collect::<Vec<_>>()
                        }
                        _ => unreachable!(),
                    };
                    for _ in 0..2 {
                        let (bytes, _) = state.work(&mut order, &token).unwrap();
                        assert_eq!(bytes, 65792);
                        state.validate().unwrap();
                    }
                    if let State::Codec { blocks, .. } = &state {
                        assert_eq!(
                            before,
                            blocks.iter().map(|b| b.output.as_ptr()).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn explicit_matrix_geometry_and_useful_work_stay_fixed_across_workers_and_reuse() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let token = CancellationToken::default();
        for shape in ["square", "tall", "wide"] {
            for reuse in ["1", "4"] {
                let c = config(&[
                    ("matrix_shape", shape),
                    ("matrix_reuse", reuse),
                    ("matrix_k_block", "32"),
                ]);
                let shared = prepare(Spec::Matrix, &c, &token).unwrap();
                let [m, n, k] = shared.shape;
                let payload = 4 * (m * k + k * n + m * n) as u64;
                let mut baseline = Vec::new();
                for workers in [1, 4] {
                    let mut output = Vec::new();
                    let mut work = 0;
                    for worker in 0..workers {
                        let mut state =
                            State::new(Spec::Matrix, &c, worker, workers, payload, &shared, &token)
                                .unwrap();
                        let mut order = schedule(&state, "sequential", &token);
                        work += state.work(&mut order, &token).unwrap().0;
                        state.validate().unwrap();
                        if let State::Matrix { out, b, .. } = state {
                            assert!(Arc::ptr_eq(&b, &shared.matrix_b));
                            output.extend(out);
                        }
                    }
                    assert_eq!(work, 2 * (m * n * k) as u64 * reuse.parse::<u64>().unwrap());
                    if baseline.is_empty() {
                        baseline = output;
                    } else {
                        assert_eq!(baseline, output);
                    }
                }
            }
        }
    }
}
