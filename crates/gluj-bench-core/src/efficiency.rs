//! Manual search candidates, not a model of measured core/clock scaling or energy use.
//! The throughput ratio selects how broadly to explore. Every candidate must be retested
//! against the unchanged starting workload; passing candidates are not known in advance.

pub fn cpu_efficiency_trials(current: u32, retained: f64) -> Vec<u32> {
    if current < 2 || !retained.is_finite() || retained <= 0.0 || retained >= 0.85 {
        return Vec::new();
    }
    let candidates = if retained <= 0.15 {
        [current.div_ceil(4), current.div_ceil(8)]
    } else if retained <= 0.65 {
        [current.div_ceil(2), current.div_ceil(4)]
    } else {
        [((u64::from(current) * 3) / 4) as u32, current.div_ceil(2)]
    };
    let mut trials = Vec::new();
    for count in candidates {
        let count = count.clamp(1, current - 1);
        if trials.last().is_none_or(|previous| count < *previous) {
            trials.push(count);
        }
    }
    trials
}

// Reductions are each relative to the limit BEFORE starting the search, not compounded.
pub fn gpu_efficiency_trials(retained: f64) -> Vec<u32> {
    if !retained.is_finite() || retained <= 0.0 || retained >= 0.85 {
        return Vec::new();
    }
    if retained <= 0.4 {
        vec![50, 75]
    } else if retained <= 0.65 {
        vec![35, 50]
    } else {
        vec![10, 25]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn severe_gaps_explore_fewer_cores_and_lower_clock_limits() {
        assert_eq!(cpu_efficiency_trials(16, 0.05), [4, 2]);
        assert_eq!(cpu_efficiency_trials(16, 0.5), [8, 4]);
        assert_eq!(cpu_efficiency_trials(16, 0.8), [12, 8]);
        assert_eq!(gpu_efficiency_trials(0.05), [50, 75]);
        assert_eq!(gpu_efficiency_trials(0.5), [35, 50]);
        assert_eq!(gpu_efficiency_trials(0.8), [10, 25]);
    }
    #[test]
    fn trials_are_bounded_distinct_and_do_not_claim_headroom_for_invalid_ratios() {
        for count in 1..=128 {
            for retained in [0.01, 0.15, 0.4, 0.65, 0.84] {
                let trials = cpu_efficiency_trials(count, retained);
                assert!(trials.iter().all(|n| *n > 0 && *n < count));
                assert!(trials.windows(2).all(|pair| pair[1] < pair[0]));
            }
        }
        for ratio in [f64::NAN, f64::INFINITY, -1.0, 0.0, 0.85, 1.0, 2.0] {
            assert!(cpu_efficiency_trials(16, ratio).is_empty());
            assert!(gpu_efficiency_trials(ratio).is_empty());
        }
    }
}
