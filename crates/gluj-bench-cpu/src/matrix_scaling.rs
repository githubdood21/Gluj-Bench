use gluj_bench_core::{BenchmarkError, CancellationToken};
use std::hint::black_box;

pub const ID: &str = "cpu.performance.matrix.fp32.scaling";
pub const ROWS: usize = 32;
const BLOCK_K: usize = 128;
const BLOCK_N: usize = 64;

pub fn bytes(dimension: usize) -> u64 {
    4 * (dimension as u64 * dimension as u64 + 2 * ROWS as u64 * dimension as u64)
}
pub fn dimension_for_bytes(budget: u64) -> usize {
    // A: 32×K, B: K×N, C: 32×N, with K=N and eight-column vector alignment.
    (((ROWS as f64 * ROWS as f64 + budget as f64 / 4.0).sqrt() - ROWS as f64) as usize) / 8 * 8
}
pub fn sizes(maximum: u64, workers: usize) -> Vec<u64> {
    let maximum_dimension = dimension_for_bytes(maximum / workers as u64);
    let mut result = Vec::new();
    let mut target = bytes(32) * workers as u64;
    while target < maximum {
        let dimension = dimension_for_bytes(target / workers as u64).min(maximum_dimension);
        if dimension >= 32 {
            let actual = bytes(dimension) * workers as u64;
            if result.last() != Some(&actual) {
                result.push(actual);
            }
        }
        target = target.saturating_mul(2);
    }
    if maximum_dimension >= 32 {
        let actual = bytes(maximum_dimension) * workers as u64;
        if result.last() != Some(&actual) {
            result.push(actual);
        }
    }
    result
}
pub fn operations(dimension: usize) -> u64 {
    2 * ROWS as u64 * dimension as u64 * dimension as u64
}
pub fn traffic(dimension: usize) -> u64 {
    // Per 8×8 microtile and K step: eight scalar A reads + one B vector read.
    // C is written for every K block, and loaded again after the first block.
    let n = dimension as u64;
    ROWS as u64 * n * n + 4 * ROWS as u64 * n * (2 * dimension.div_ceil(BLOCK_K) as u64 - 1)
}

pub struct Buffers {
    pub dimension: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
}
impl Buffers {
    pub fn new(dimension: usize, worker: usize) -> Result<Self, BenchmarkError> {
        fn array(size: usize, phase: usize) -> Result<Vec<f32>, BenchmarkError> {
            let mut data = Vec::new();
            data.try_reserve_exact(size)
                .map_err(|error| BenchmarkError::new("allocation_failed", error.to_string()))?;
            data.extend((0..size).map(|i| ((i + phase) % 17) as f32 / 32.0 - 0.25));
            Ok(data)
        }
        Ok(Self {
            dimension,
            a: array(ROWS * dimension, worker)?,
            b: array(dimension * dimension, worker + 3)?,
            c: array(ROWS * dimension, 0)?,
        })
    }
    pub fn sweep(&mut self, cancellation: &CancellationToken) -> bool {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the provider gates AVX2/FMA; dimensions are multiples of eight;
        // arrays are disjoint and sized to all matrix accesses.
        unsafe {
            let complete = multiply(self.dimension, &self.a, &self.b, &mut self.c, cancellation);
            black_box(&self.c);
            complete
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = cancellation;
            false
        }
    }
    pub fn checksum(&self) -> u64 {
        let hash = self
            .c
            .iter()
            .step_by(self.dimension)
            .fold(0_u64, |hash, value| {
                hash.rotate_left(7) ^ u64::from(value.to_bits())
            });
        black_box(&self.c);
        black_box(hash)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn multiply(
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    cancellation: &CancellationToken,
) -> bool {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};
    for column_block in (0..n).step_by(BLOCK_N) {
        for k_block in (0..n).step_by(BLOCK_K) {
            if cancellation.is_cancelled() {
                return false;
            }
            for column in (column_block..(column_block + BLOCK_N).min(n)).step_by(8) {
                for row in (0..ROWS).step_by(8) {
                    let mut accumulators = [_mm256_set1_ps(0.0); 8];
                    if k_block > 0 {
                        for lane in 0..8 {
                            accumulators[lane] = unsafe {
                                _mm256_loadu_ps(c.as_ptr().add((row + lane) * n + column))
                            };
                        }
                    }
                    for k in k_block..(k_block + BLOCK_K).min(n) {
                        let weights = unsafe { _mm256_loadu_ps(b.as_ptr().add(k * n + column)) };
                        for lane in 0..8 {
                            let input =
                                _mm256_set1_ps(unsafe { *a.as_ptr().add((row + lane) * n + k) });
                            accumulators[lane] =
                                _mm256_fmadd_ps(input, weights, accumulators[lane]);
                        }
                    }
                    for lane in 0..8 {
                        unsafe {
                            _mm256_storeu_ps(
                                c.as_mut_ptr().add((row + lane) * n + column),
                                accumulators[lane],
                            );
                        }
                    }
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocked_matrix_matches_scalar_including_partial_blocks_and_repeated_products() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            for n in [32, 72, 136] {
                let mut buffers = Buffers::new(n, 3).unwrap();
                for _ in 0..2 {
                    assert!(buffers.sweep(&CancellationToken::default()));
                    for row in 0..ROWS {
                        for column in 0..n {
                            let expected = (0..n).fold(0.0_f32, |sum, k| {
                                buffers.a[row * n + k].mul_add(buffers.b[k * n + column], sum)
                            });
                            assert_eq!(buffers.c[row * n + column], expected);
                        }
                    }
                }
                let cancellation = CancellationToken::default();
                cancellation.cancel();
                assert!(!buffers.sweep(&cancellation));
            }
        }
    }
    #[test]
    fn matrix_sizes_and_counts_match_actual_allocations() {
        for workers in [1, 16, 24, 64] {
            let budget = 1024 * 1024 * 1024;
            let sizes = sizes(budget, workers);
            assert!(sizes.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(*sizes.last().unwrap() <= budget);
            assert!(*sizes.last().unwrap() > budget * 98 / 100);
            for size in sizes {
                let n = dimension_for_bytes(size / workers as u64);
                assert_eq!(n % 8, 0);
                assert_eq!(bytes(n) * workers as u64, size);
                assert_eq!(operations(n), 2 * ROWS as u64 * n as u64 * n as u64);
                let reads = (ROWS / 8 * (n / 8) * n * (8 * 4 + 8 * 4)) as u64;
                let output = (4 * ROWS * n * (2 * n.div_ceil(BLOCK_K) - 1)) as u64;
                assert_eq!(traffic(n), reads + output);
            }
        }
    }
}
