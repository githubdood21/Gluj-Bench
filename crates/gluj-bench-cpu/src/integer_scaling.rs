use gluj_bench_core::{BenchmarkError, CancellationToken};
use std::hint::black_box;

pub const ID: &str = "cpu.performance.integer.i64";
pub const BYTES_PER_ELEMENT: u64 = 16; // one input and one output u64
pub const BLOCK_ELEMENTS: u64 = 64;
pub const REUSE: u64 = 16;
// The same eight multiply/add pairs used by the register INT64 workload.
const MULTIPLIERS: [u64; 8] = [
    0x9e37_79b1,
    0x85eb_ca77,
    0xc2b2_ae3d,
    0x27d4_eb2d,
    0x1656_67b1,
    0x9e37_79b9,
    0x85eb_ca77,
    0xc2b2_ae3d,
];
const ADDENDS: [u64; 8] = [
    0x85eb_ca6b,
    0xc2b2_ae3d,
    0x27d4_eb2f,
    0x1656_67b1,
    0x9e37_79b9,
    0x85eb_ca77,
    0xc2b2_ae3d,
    0x27d4_eb2d,
];

pub(crate) struct Buffers {
    input: Vec<u64>,
    output: Vec<u64>,
}
impl Buffers {
    pub(crate) fn new(
        elements: usize,
        worker: usize,
        cancellation: &CancellationToken,
    ) -> Result<Self, BenchmarkError> {
        let mut input = Vec::new();
        let mut output = Vec::new();
        for array in [&mut input, &mut output] {
            array
                .try_reserve_exact(elements)
                .map_err(|error| BenchmarkError::new("allocation_failed", error.to_string()))?;
        }
        for start in (0..elements).step_by(65536) {
            if cancellation.is_cancelled() {
                return Err(BenchmarkError::new(
                    "cancelled",
                    "INT64 buffer preparation cancelled.",
                ));
            }
            let end = elements.min(start + 65536);
            input.extend(
                (start..end).map(|index| {
                    (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ worker as u64
                }),
            );
            output.resize(end, 0);
        }
        Ok(Self { input, output })
    }
    pub(crate) fn sweep(&mut self, cancellation: &CancellationToken) -> bool {
        let input = black_box(self.input.as_slice());
        for (input, output) in input.chunks(65536).zip(self.output.chunks_mut(65536)) {
            if cancellation.is_cancelled() {
                return false;
            }
            for (source, destination) in input.chunks_exact(8).zip(output.chunks_exact_mut(8)) {
                let mut values: [u64; 8] = source.try_into().unwrap();
                for _ in 0..REUSE {
                    for lane in 0..8 {
                        // Preserve each multiply/add round: modulo arithmetic can
                        // otherwise collapse repeated constant affine transforms.
                        values[lane] = black_box(values[lane])
                            .wrapping_mul(MULTIPLIERS[lane])
                            .wrapping_add(ADDENDS[lane]);
                    }
                }
                destination.copy_from_slice(&values);
            }
        }
        true
    }
    pub(crate) fn checksum(&self) -> u64 {
        black_box(self.output[0]) ^ black_box(*self.output.last().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_integer_matches_wrapping_reference_and_reuses_buffers() {
        let cancellation = CancellationToken::default();
        for elements in [64, 128, 65536 + 64] {
            let mut buffers = Buffers::new(elements, 3, &cancellation).unwrap();
            let pointers = (buffers.input.as_ptr(), buffers.output.as_ptr());
            assert_eq!(
                (buffers.input.len() + buffers.output.len()) * 8,
                elements * 16
            );
            assert!(buffers.sweep(&cancellation));
            for (index, (&input, &actual)) in buffers.input.iter().zip(&buffers.output).enumerate()
            {
                let lane = index % 8;
                // Independently apply the recurrence in a wider integer and keep
                // its low 64 bits, including overflowing inputs and products.
                let mut expected = input as u128;
                for _ in 0..REUSE {
                    expected = (expected * MULTIPLIERS[lane] as u128 + ADDENDS[lane] as u128)
                        & u64::MAX as u128;
                }
                assert_eq!(actual, expected as u64);
            }
            let hash = buffers.checksum();
            assert!(buffers.sweep(&cancellation));
            assert_eq!(buffers.checksum(), hash);
            assert_eq!((buffers.input.as_ptr(), buffers.output.as_ptr()), pointers);
        }
        let mut buffers = Buffers::new(64, 0, &cancellation).unwrap();
        cancellation.cancel();
        assert!(!buffers.sweep(&cancellation));
        assert!(Buffers::new(64, 0, &cancellation).is_err());
    }
}
