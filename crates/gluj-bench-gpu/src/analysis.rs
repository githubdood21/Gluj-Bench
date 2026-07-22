use gluj_bench_core::SampleStatistics;

pub const KIB: u64 = 1024;
pub const MIB: u64 = 1024 * KIB;
pub const CACHE_SWEEP_MIN: u64 = 256 * KIB;
pub const CACHE_SWEEP_MAX: u64 = 512 * MIB;
pub const VRAM_MIN: u64 = 256 * MIB;
pub const VRAM_MAX_PER_BUFFER: u64 = 512 * MIB;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepPoint {
    pub size_bytes: u64,
    pub bandwidth_bytes_per_second: f64,
    pub coefficient_of_variation: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectiveCacheTier {
    pub last_cached_size: u64,
    pub first_uncached_size: u64,
    pub estimated_capacity: u64,
    pub confidence: f64,
}

pub fn sweep_sizes(limit: u64) -> Vec<u64> {
    let limit = limit.min(CACHE_SWEEP_MAX);
    let mut sizes = Vec::new();
    let mut size = CACHE_SWEEP_MIN;
    while size <= limit {
        sizes.push(size);
        let Some(next) = size.checked_mul(2) else {
            break;
        };
        size = next;
    }
    sizes
}

pub fn detect_effective_cache_tiers(points: &[SweepPoint]) -> Vec<EffectiveCacheTier> {
    if points.len() < 4 {
        return Vec::new();
    }
    let mut tiers = Vec::new();
    let mut previous_boundary = None;
    for boundary in 2..points.len() - 1 {
        let before = &points[boundary - 2..boundary];
        if before.iter().any(|point| {
            !point.bandwidth_bytes_per_second.is_finite()
                || point.bandwidth_bytes_per_second <= 0.0
                || point.coefficient_of_variation > 0.10
        }) {
            continue;
        }
        let plateau =
            (before[0].bandwidth_bytes_per_second + before[1].bandwidth_bytes_per_second) / 2.0;
        let plateau_variation =
            (before[0].bandwidth_bytes_per_second - before[1].bandwidth_bytes_per_second).abs()
                / (plateau * 2.0);
        if plateau_variation > 0.10 {
            continue;
        }
        let first = points[boundary].bandwidth_bytes_per_second;
        let second = points[boundary + 1].bandwidth_bytes_per_second;
        if first > plateau * 0.80 || second > plateau * 0.80 {
            continue;
        }
        if previous_boundary.is_some_and(|prior| boundary <= prior + 1) {
            continue;
        }
        let last_cached_size = points[boundary - 1].size_bytes;
        let first_uncached_size = points[boundary].size_bytes;
        let estimated_capacity = geometric_mean(last_cached_size, first_uncached_size);
        let sustained = first.max(second) / plateau;
        tiers.push(EffectiveCacheTier {
            last_cached_size,
            first_uncached_size,
            estimated_capacity,
            confidence: (1.0 - sustained).clamp(0.0, 1.0),
        });
        previous_boundary = Some(boundary);
    }
    let outer_cliff = (1..points.len().saturating_sub(1))
        .rev()
        .find_map(|boundary| {
            let before = points[boundary - 1].bandwidth_bytes_per_second;
            let first = points[boundary].bandwidth_bytes_per_second;
            let second = points[boundary + 1].bandwidth_bytes_per_second;
            (before.is_finite()
                && before > 0.0
                && first <= before * 0.50
                && second <= before * 0.50)
                .then(|| EffectiveCacheTier {
                    last_cached_size: points[boundary - 1].size_bytes,
                    first_uncached_size: points[boundary].size_bytes,
                    estimated_capacity: geometric_mean(
                        points[boundary - 1].size_bytes,
                        points[boundary].size_bytes,
                    ),
                    confidence: (1.0 - first.max(second) / before).clamp(0.0, 1.0),
                })
        });
    if let Some(outer) = outer_cliff
        && tiers
            .last()
            .is_none_or(|tier| tier.first_uncached_size < outer.first_uncached_size)
    {
        tiers.push(outer);
    }
    if tiers.len() > 2 {
        vec![tiers[0], tiers[tiers.len() - 1]]
    } else {
        tiers
    }
}

pub fn cache_working_set(
    tier: EffectiveCacheTier,
    lower: Option<EffectiveCacheTier>,
) -> Option<u64> {
    let desired = tier.estimated_capacity.saturating_mul(40) / 100;
    let minimum = lower
        .map(|previous| previous.first_uncached_size.saturating_mul(2))
        .unwrap_or(CACHE_SWEEP_MIN);
    let mut selected = floor_power_of_two(desired.max(CACHE_SWEEP_MIN));
    while selected < minimum {
        selected = selected.checked_mul(2)?;
    }
    (selected >= CACHE_SWEEP_MIN && selected <= tier.last_cached_size).then_some(selected)
}

pub fn vram_working_set(outer: Option<EffectiveCacheTier>, adapter_limit: u64) -> Option<u64> {
    let separation = match outer {
        Some(tier) => tier.first_uncached_size.checked_mul(4)?,
        None => VRAM_MIN,
    };
    let desired = VRAM_MIN.max(separation);
    let cap = adapter_limit.min(VRAM_MAX_PER_BUFFER);
    (desired <= cap).then_some(align_down(desired, 256))
}

pub fn statistics(values: &[f64]) -> SampleStatistics {
    if values.is_empty() {
        return SampleStatistics::default();
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let median = if ordered.len().is_multiple_of(2) {
        let middle = ordered.len() / 2;
        (ordered[middle - 1] + ordered[middle]) / 2.0
    } else {
        ordered[ordered.len() / 2]
    };
    let mean = ordered.iter().sum::<f64>() / ordered.len() as f64;
    let variance = ordered
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / ordered.len() as f64;
    SampleStatistics {
        sample_count: ordered.len() as u32,
        minimum: ordered[0],
        median,
        maximum: ordered[ordered.len() - 1],
        standard_deviation: variance.sqrt(),
    }
}

pub fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean <= 0.0 {
        return f64::INFINITY;
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

fn geometric_mean(left: u64, right: u64) -> u64 {
    ((left as f64 * right as f64).sqrt() as u64).min(right)
}

fn align_down(value: u64, alignment: u64) -> u64 {
    value / alignment * alignment
}

fn floor_power_of_two(value: u64) -> u64 {
    if value == 0 {
        0
    } else {
        1_u64 << (63 - value.leading_zeros())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(size_mib: u64, bandwidth: f64) -> SweepPoint {
        SweepPoint {
            size_bytes: size_mib * MIB,
            bandwidth_bytes_per_second: bandwidth,
            coefficient_of_variation: 0.02,
        }
    }

    #[test]
    fn detects_one_sustained_tier_boundary() {
        let points = [
            point(1, 100.0),
            point(2, 101.0),
            point(4, 60.0),
            point(8, 59.0),
            point(16, 58.0),
        ];
        let tiers = detect_effective_cache_tiers(&points);
        assert_eq!(tiers.len(), 1);
        assert_eq!(tiers[0].last_cached_size, 2 * MIB);
        assert_eq!(tiers[0].first_uncached_size, 4 * MIB);
    }

    #[test]
    fn detects_two_separated_boundaries() {
        let points = [
            point(1, 200.0),
            point(2, 198.0),
            point(4, 130.0),
            point(8, 128.0),
            point(16, 70.0),
            point(32, 69.0),
        ];
        assert_eq!(detect_effective_cache_tiers(&points).len(), 2);
    }

    #[test]
    fn keeps_the_outermost_sustained_cliff() {
        let points = [
            point(1, 200.0),
            point(2, 198.0),
            point(4, 130.0),
            point(8, 128.0),
            point(16, 90.0),
            point(32, 55.0),
            point(64, 12.0),
            point(128, 11.0),
        ];
        let tiers = detect_effective_cache_tiers(&points);
        assert_eq!(tiers.len(), 2);
        assert_eq!(tiers[1].last_cached_size, 32 * MIB);
        assert_eq!(tiers[1].first_uncached_size, 64 * MIB);
    }

    #[test]
    fn rejects_noise_and_gradual_slopes() {
        let noisy = [
            point(1, 100.0),
            SweepPoint {
                coefficient_of_variation: 0.2,
                ..point(2, 100.0)
            },
            point(4, 70.0),
            point(8, 70.0),
        ];
        assert!(detect_effective_cache_tiers(&noisy).is_empty());
        let gradual = [
            point(1, 100.0),
            point(2, 92.0),
            point(4, 85.0),
            point(8, 78.0),
        ];
        assert!(detect_effective_cache_tiers(&gradual).is_empty());
    }

    #[test]
    fn working_sets_enforce_limits_and_separation() {
        let tier = EffectiveCacheTier {
            last_cached_size: 16 * MIB,
            first_uncached_size: 32 * MIB,
            estimated_capacity: 22 * MIB,
            confidence: 0.4,
        };
        assert_eq!(cache_working_set(tier, None), Some(8 * MIB));
        assert_eq!(vram_working_set(Some(tier), 512 * MIB), Some(256 * MIB));
        assert_eq!(vram_working_set(Some(tier), 128 * MIB), None);
    }

    #[test]
    fn statistics_are_population_statistics() {
        let stats = statistics(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(stats.median, 2.5);
        assert!((stats.standard_deviation - 1.118_033_988_7).abs() < 1e-9);
    }
}
