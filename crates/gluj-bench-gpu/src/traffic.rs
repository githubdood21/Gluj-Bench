//! Fixed arrival-rate targets for offered work; nominal accesses are not PCIe counters.
use gluj_bench_core::{BenchmarkConfig, BenchmarkError};

#[derive(Clone, Copy)]
pub(super) enum TrafficLimit {
    Static(u32),
    Dynamic(u64),
}
#[derive(Clone, Copy)]
pub(super) struct PassRate(pub f64);
impl TrafficLimit {
    pub fn from_config(config: &BenchmarkConfig) -> Result<Option<Self>, BenchmarkError> {
        let rate = config.options.get("offload_passes_per_second");
        let bytes = config.options.get("offload_traffic_bytes_per_second");
        let invalid = || {
            BenchmarkError::new(
                "invalid_config",
                "Choose either static traffic (1–1000 passes/s) or dynamic traffic (1 MiB/s–64 GiB/s), not both.",
            )
        };
        match (rate, bytes) {
            (None, None) => Ok(None),
            (Some(value), None) => value
                .parse::<u32>()
                .ok()
                .filter(|n| (1..=1000).contains(n))
                .map(|n| Some(Self::Static(n)))
                .ok_or_else(invalid),
            (None, Some(value)) => value
                .parse::<u64>()
                .ok()
                .filter(|n| (1_048_576..=68_719_476_736).contains(n))
                .map(|n| Some(Self::Dynamic(n)))
                .ok_or_else(invalid),
            _ => Err(invalid()),
        }
    }
    pub fn for_dataset(self, bytes: u64) -> PassRate {
        PassRate(match self {
            Self::Static(rate) => f64::from(rate),
            Self::Dynamic(rate) => {
                gluj_bench_core::dynamic_offload_traffic(rate as f64, bytes) / bytes.max(1) as f64
            }
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn traffic_is_optional_validated_and_retains_the_requested_rate() {
        let mut config = BenchmarkConfig::default();
        assert!(TrafficLimit::from_config(&config).unwrap().is_none());
        for value in ["0", "1001", "-1", "bad"] {
            config
                .options
                .insert("offload_passes_per_second".into(), value.into());
            assert!(TrafficLimit::from_config(&config).is_err());
        }
        config
            .options
            .insert("offload_passes_per_second".into(), "100".into());
        let limit = TrafficLimit::from_config(&config)
            .unwrap()
            .unwrap()
            .for_dataset(2 * 1024 * 1024);
        assert_eq!(limit.0, 100.0);
    }
    #[test]
    fn dynamic_grows_traffic_and_reduces_pass_rate_as_size_grows() {
        let mib = 1024_u64.pow(2);
        let limit = TrafficLimit::Dynamic(8 * mib);
        assert_eq!(limit.for_dataset(mib).0, 8.0);
        assert_eq!(limit.for_dataset(4 * mib).0, 4.0);
        assert_eq!(limit.for_dataset(1024 * mib).0, 0.25);
        let mut previous_rate = f64::INFINITY;
        let mut previous_traffic = 0.0;
        for size in [mib, 2 * mib, 4 * mib, 16 * mib, 1024 * mib] {
            let rate = limit.for_dataset(size).0;
            assert!(rate < previous_rate);
            assert!(rate * size as f64 > previous_traffic);
            previous_rate = rate;
            previous_traffic = rate * size as f64;
        }
        assert_eq!(TrafficLimit::Static(100).for_dataset(2 * mib).0, 100.0);
        assert_eq!(TrafficLimit::Static(100).for_dataset(32 * mib).0, 100.0);
        let mut config = BenchmarkConfig::default();
        config.options.insert(
            "offload_traffic_bytes_per_second".into(),
            (8 * mib).to_string(),
        );
        assert!(TrafficLimit::from_config(&config).unwrap().is_some());
        config
            .options
            .insert("offload_passes_per_second".into(), "100".into());
        assert!(TrafficLimit::from_config(&config).is_err());
        config.options.remove("offload_passes_per_second");
        config
            .options
            .insert("offload_traffic_bytes_per_second".into(), "0".into());
        assert!(TrafficLimit::from_config(&config).is_err());
    }
}
