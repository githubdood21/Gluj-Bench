//! Reproducible access schedules. Order variation is not a cache-miss measurement.
use crate::{BenchmarkConfig, BenchmarkError, CancellationToken};
use std::collections::BTreeMap;

pub const ACCESS_REVISION: &str = "splitmix64-fisher-yates-v1";
pub const DEFAULT_ACCESS_SEED: u64 = 0x6a09_e667_f3bc_c909;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessPattern {
    Sequential,
    Shuffled,
    Rotating,
    Scattered,
    HotCold,
}
#[derive(Clone, Copy, Debug)]
pub struct AccessPolicy {
    pub pattern: AccessPattern,
    pub seed: u64,
    pub block_bytes: usize,
    pub realizations: u32,
}
pub fn access_supported(id: &str) -> bool {
    matches!(
        id,
        "cpu.performance.avx2.f32_fma.scaling"
            | "cpu.performance.integer.i64"
            | "cpu.performance.string.ascii"
    ) || id.starts_with("cpu.bandwidth.")
        || crate::cpu_data_mode(id)
}
impl AccessPolicy {
    pub fn from_config(config: &BenchmarkConfig) -> Result<Self, BenchmarkError> {
        let invalid = || {
            BenchmarkError::new(
                "invalid_config",
                "Invalid access pattern, seed, ensemble or block size.",
            )
        };
        let pattern = match config
            .options
            .get("access_pattern")
            .map(String::as_str)
            .unwrap_or("sequential")
        {
            "sequential" => AccessPattern::Sequential,
            "shuffled_blocks" => AccessPattern::Shuffled,
            "rotating_regions" => AccessPattern::Rotating,
            "scattered" => AccessPattern::Scattered,
            "hot_cold" => AccessPattern::HotCold,
            _ => return Err(invalid()),
        };
        let seed = config
            .options
            .get("access_seed")
            .map(|s| s.parse())
            .transpose()
            .map_err(|_| invalid())?
            .unwrap_or(DEFAULT_ACCESS_SEED);
        let block_bytes = config
            .options
            .get("access_block_bytes")
            .map(|s| s.parse())
            .transpose()
            .map_err(|_| invalid())?
            .unwrap_or(65536);
        let realizations = config
            .options
            .get("access_realizations")
            .map(|s| s.parse())
            .transpose()
            .map_err(|_| invalid())?
            .unwrap_or(1);
        if !(64..=1048576).contains(&block_bytes)
            || block_bytes % 64 != 0
            || ![1, 3].contains(&realizations)
        {
            return Err(invalid());
        }
        Ok(Self {
            pattern,
            seed,
            block_bytes,
            realizations,
        })
    }
    /// Reserve schedule storage as well as the requested data. Numeric kernels
    /// visit complete 64-element blocks; byte kernels use 64-byte blocks.
    pub fn payload_budget(self, budget: u64, workers: usize, bytes_per_element: u64) -> u64 {
        let Some(minimum) = 64u64.checked_mul(bytes_per_element).filter(|n| *n > 0) else {
            return 0;
        };
        if workers == 0 {
            return 0;
        }
        let block = if self.pattern == AccessPattern::Scattered {
            minimum
        } else {
            (self.block_bytes as u64 / minimum).max(1) * minimum
        };
        let overhead = 8 * u64::from(self.realizations);
        let reserve = overhead.saturating_mul(workers as u64);
        ((u128::from(budget.saturating_sub(reserve)) * u128::from(block))
            / (u128::from(block) + u128::from(overhead))) as u64
    }
    pub fn name(self) -> &'static str {
        match self.pattern {
            AccessPattern::Sequential => "sequential",
            AccessPattern::Shuffled => "shuffled_blocks",
            AccessPattern::Rotating => "rotating_regions",
            AccessPattern::Scattered => "scattered",
            AccessPattern::HotCold => "hot_cold",
        }
    }
    pub fn derived_seed(self, worker: usize, bytes: usize, realization: u32) -> u64 {
        derive_seed(self.seed, worker as u64, bytes as u64, realization as u64)
    }
    pub fn metadata(self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("access_pattern".into(), self.name().into()), ("access_seed".into(), self.seed.to_string()),
            ("access_generator_revision".into(), ACCESS_REVISION.into()),
            ("access_block_bytes".into(), self.block_bytes.to_string()),
            ("access_realizations".into(), self.realizations.to_string()),
            ("access_schedule_timing".into(), "schedule setup excluded; schedule lookup and traversal included".into()),
            ("access_summary_rule".into(), "equal observation counts per realization; pooled descriptive median/spread; realization summaries separate in native records; sequential compute repeats identical traversal as equal-coverage cohorts".into()),
            ("access_cache_claim".into(), "locality/turnover workload; cache misses and residency not measured".into()),
            ("access_hot_cold_definition".into(), "full cold traversal plus one repeated first-quarter hot block after each cold block; useful work counts all visits".into()),
            ("access_rotation_definition".into(), "full coverage per pass; starting block advances by one per pass".into()),
        ])
    }
    pub fn schedule(
        self,
        bytes: usize,
        alignment: usize,
        worker: usize,
        realization: u32,
        cancellation: &CancellationToken,
    ) -> Result<AccessSchedule, BenchmarkError> {
        if bytes == 0 || alignment == 0 || !bytes.is_multiple_of(alignment) {
            return Err(BenchmarkError::new(
                "invalid_config",
                "Access span must contain aligned complete elements.",
            ));
        }
        let block = if self.pattern == AccessPattern::Scattered {
            alignment
        } else {
            self.block_bytes.max(alignment).div_ceil(alignment) * alignment
        };
        let count = bytes.div_ceil(block);
        let mut order = Vec::new();
        order
            .try_reserve_exact(count)
            .map_err(|e| BenchmarkError::new("allocation_failed", e.to_string()))?;
        for i in 0..count {
            if i % 4096 == 0 {
                check(cancellation)?;
            }
            order.push(i);
        }
        if matches!(
            self.pattern,
            AccessPattern::Shuffled | AccessPattern::Scattered | AccessPattern::HotCold
        ) {
            let mut state = self.derived_seed(worker, bytes, realization);
            for i in (1..count).rev() {
                if i % 4096 == 0 {
                    check(cancellation)?;
                }
                let j = random_below(&mut state, i + 1);
                order.swap(i, j);
            }
        }
        Ok(AccessSchedule {
            order,
            block,
            bytes,
            pattern: self.pattern,
            pass: 0,
        })
    }
}
pub fn derive_seed(root: u64, worker: u64, bytes: u64, realization: u64) -> u64 {
    let mut state = root
        ^ worker.wrapping_mul(0x9e3779b97f4a7c15)
        ^ bytes.rotate_left(23)
        ^ realization.wrapping_mul(0xd1b54a32d192ed03);
    random(&mut state)
}
pub fn fresh_access_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    derive_seed(
        time,
        std::process::id().into(),
        NONCE.fetch_add(1, Ordering::Relaxed),
        0,
    )
}
fn check(c: &CancellationToken) -> Result<(), BenchmarkError> {
    if c.is_cancelled() {
        Err(BenchmarkError::new(
            "cancelled",
            "Access schedule preparation cancelled.",
        ))
    } else {
        Ok(())
    }
}
fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn random_below(state: &mut u64, bound: usize) -> usize {
    let n = bound as u64;
    let threshold = n.wrapping_neg() % n;
    loop {
        let value = random(state);
        if value >= threshold {
            return (value % n) as usize;
        }
    }
}
pub struct AccessSchedule {
    order: Vec<usize>,
    block: usize,
    bytes: usize,
    pattern: AccessPattern,
    pass: usize,
}
impl AccessSchedule {
    pub fn reset_pass(&mut self, pass: usize) {
        self.pass = pass;
    }
    pub fn auxiliary_bytes(&self) -> usize {
        self.order.capacity() * std::mem::size_of::<usize>()
    }
    pub fn visited_bytes(&self) -> usize {
        if self.pattern == AccessPattern::HotCold {
            let count = self.order.len();
            let hot = count.div_ceil(4);
            let hot_bytes = self.bytes.min(hot * self.block);
            self.bytes + (count / hot) * hot_bytes + self.bytes.min((count % hot) * self.block)
        } else {
            self.bytes
        }
    }
    fn range(&self, index: usize) -> std::ops::Range<usize> {
        let begin = index * self.block;
        begin..self.bytes.min(begin + self.block)
    }
    pub fn visit(
        &mut self,
        cancellation: &CancellationToken,
        mut work: impl FnMut(std::ops::Range<usize>),
    ) -> bool {
        let count = self.order.len();
        for i in 0..count {
            if i % 256 == 0 && cancellation.is_cancelled() {
                return false;
            }
            let index = self.order[(i + if self.pattern == AccessPattern::Rotating {
                self.pass % count
            } else {
                0
            }) % count];
            work(self.range(index));
            if self.pattern == AccessPattern::HotCold {
                work(self.range(i % count.div_ceil(4)));
            }
        }
        self.pass = self.pass.wrapping_add(1);
        true
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedules_cover_non_power_of_two_spans_and_reproduce() {
        for pattern in [
            "sequential",
            "shuffled_blocks",
            "rotating_regions",
            "scattered",
            "hot_cold",
        ] {
            let config = BenchmarkConfig {
                options: BTreeMap::from([
                    ("access_pattern".into(), pattern.into()),
                    ("access_block_bytes".into(), "192".into()),
                ]),
                ..Default::default()
            };
            let policy = AccessPolicy::from_config(&config).unwrap();
            let token = CancellationToken::default();
            let collect = || {
                let mut s = policy.schedule(64 * 17, 64, 2, 1, &token).unwrap();
                let bytes = s.visited_bytes();
                let mut ranges = Vec::new();
                assert!(s.visit(&token, |r| ranges.push(r)));
                assert_eq!(ranges.iter().map(|r| r.len()).sum::<usize>(), bytes);
                ranges
            };
            let ranges = collect();
            assert_eq!(ranges, collect());
            let mut seen = [0; 17];
            for r in ranges {
                for i in (r.start..r.end).step_by(64) {
                    seen[i / 64] += 1;
                }
            }
            assert!(seen.iter().all(|n| *n >= 1));
            if pattern != "hot_cold" {
                assert!(seen.iter().all(|n| *n == 1));
            }
        }
    }
    #[test]
    fn seed_derivation_and_cancellation_are_explicit() {
        assert_ne!(derive_seed(1, 2, 3, 4), derive_seed(1, 2, 3, 5));
        let token = CancellationToken::default();
        token.cancel();
        assert!(
            AccessPolicy::from_config(&BenchmarkConfig::default())
                .unwrap()
                .schedule(64, 64, 0, 0, &token)
                .is_err()
        );
    }
}

#[cfg(test)]
mod phase_four_regressions {
    use super::*;
    use crate::{BenchmarkResult, SampleCoverage};
    #[test]
    fn rotating_order_changes_without_losing_coverage_and_auxiliary_is_explicit() {
        let config = BenchmarkConfig {
            options: BTreeMap::from([
                ("access_pattern".into(), "rotating_regions".into()),
                ("access_block_bytes".into(), "128".into()),
            ]),
            ..Default::default()
        };
        let token = CancellationToken::default();
        let mut schedule = AccessPolicy::from_config(&config)
            .unwrap()
            .schedule(64 * 7, 64, 0, 0, &token)
            .unwrap();
        assert_eq!(schedule.auxiliary_bytes(), 4 * std::mem::size_of::<usize>());
        let mut first = Vec::new();
        let mut second = Vec::new();
        schedule.visit(&token, |r| first.push(r));
        schedule.visit(&token, |r| second.push(r));
        assert_ne!(first, second);
        first.sort_by_key(|r| r.start);
        second.sort_by_key(|r| r.start);
        assert_eq!(first, second);
    }
    #[test]
    fn ensemble_records_require_each_declared_seed_and_keep_within_seed_statistics() {
        let config = BenchmarkConfig {
            samples: 6,
            target_duration_ms: 600,
            options: BTreeMap::from([
                ("measurement_policy".into(), "shorter".into()),
                ("access_realizations".into(), "3".into()),
            ]),
        };
        let run = |seeds| {
            crate::capture_measurement("cpu.performance.string.ascii", &config, || {
                let recorder = crate::observation_recorder().unwrap();
                for seed in 0..seeds {
                    let recorder = recorder.with_realization(seed, BTreeMap::new());
                    for sample in 0..6 {
                        recorder.record(
                            "worker-0",
                            "compute",
                            10.0 + seed as f64 + sample as f64,
                            "operations/s",
                            1,
                            "operations",
                            100_000_000,
                            100_000_000,
                            "validated",
                        );
                    }
                }
                Ok(BenchmarkResult {
                    benchmark_id: "cpu.performance.string.ascii".into(),
                    device_id: "cpu".into(),
                    elapsed_ns: 600_000_000,
                    metrics: Vec::new(),
                    workload_metadata: BTreeMap::new(),
                    device_metadata: BTreeMap::new(),
                })
            })
        };
        let complete = run(3);
        assert!(complete.result.is_ok());
        assert_eq!(complete.sample_coverage, SampleCoverage::Complete);
        assert_eq!(complete.realizations.len(), 3);
        assert_eq!(complete.samples.len(), 18);
        assert!(
            complete
                .realizations
                .iter()
                .all(|r| r.parameters.values().any(|v| v.contains("count=6")))
        );
        let partial = run(2);
        assert!(partial.result.is_err());
        assert_eq!(partial.samples.len(), 12);
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    #[test]
    fn payload_budget_reserves_all_realization_schedule_storage() {
        let token = CancellationToken::default();
        let budget = 1048576;
        for unit in [1u64, 12, 16] {
            for workers in [1, 8] {
                for pattern in ["sequential", "scattered", "hot_cold"] {
                    let config = BenchmarkConfig {
                        options: BTreeMap::from([
                            ("access_pattern".into(), pattern.into()),
                            ("access_block_bytes".into(), "192".into()),
                            ("access_realizations".into(), "3".into()),
                        ]),
                        ..Default::default()
                    };
                    let policy = AccessPolicy::from_config(&config).unwrap();
                    let alignment = 64 * unit * workers as u64;
                    let payload =
                        policy.payload_budget(budget, workers, unit) / alignment * alignment;
                    let mut schedule_policy = policy;
                    schedule_policy.block_bytes =
                        (policy.block_bytes / unit as usize / 64).max(1) * 64;
                    let elements = payload / workers as u64 / unit;
                    let schedule = schedule_policy
                        .schedule(elements as usize, 64, 0, 0, &token)
                        .unwrap();
                    assert!(
                        payload + schedule.auxiliary_bytes() as u64 * 3 * workers as u64 <= budget
                    );
                }
            }
        }
    }
}
