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
    #[cfg(test)]
    pub(crate) fn sweep(&mut self, cancellation: &CancellationToken) -> bool {
        if cancellation.is_cancelled() {
            return false;
        }
        let mut schedule = gluj_bench_core::AccessPolicy::from_config(&Default::default())
            .unwrap()
            .schedule(self.input.len(), 64, 0, 0, cancellation)
            .unwrap();
        self.sweep_scheduled(cancellation, &mut schedule)
    }
    pub(crate) fn sweep_scheduled(
        &mut self,
        cancellation: &CancellationToken,
        schedule: &mut gluj_bench_core::AccessSchedule,
    ) -> bool {
        let input = black_box(self.input.as_slice());
        schedule.visit(cancellation, |range| {
            let input = &input[range.clone()];
            let output = &mut self.output[range];
            for (source, destination) in input
                .as_chunks::<8>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<8>().0.iter_mut())
            {
                let mut values = *source;
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
        })
    }
    pub(crate) fn checksum(&self) -> u64 {
        black_box(self.output[0]) ^ black_box(*self.output.last().unwrap())
    }
    pub(crate) fn validate(&self) -> Result<(), BenchmarkError> {
        for index in gluj_bench_core::validation_indices(self.output.len()) {
            let mut expected = self.input[index];
            for _ in 0..REUSE {
                expected = expected
                    .wrapping_mul(MULTIPLIERS[index % 8])
                    .wrapping_add(ADDENDS[index % 8]);
            }
            if self.output[index] != expected {
                return Err(BenchmarkError::new(
                    "result_validation_failed",
                    "INT64 output differs from its wrapping arithmetic reference.",
                ));
            }
        }
        Ok(())
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
