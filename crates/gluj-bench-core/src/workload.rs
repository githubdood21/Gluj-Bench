use crate::{BenchmarkConfig, BenchmarkError, CancellationToken};
use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

pub fn workload_percent(config: &BenchmarkConfig, key: &str) -> Result<u32, BenchmarkError> {
    let percent = config
        .options
        .get(key)
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|_| {
            BenchmarkError::new(
                "invalid_config",
                format!("{key} must be 25, 50, 75, or 100."),
            )
        })?
        .unwrap_or(100);
    if ![25, 50, 75, 100].contains(&percent) {
        return Err(BenchmarkError::new(
            "invalid_config",
            format!("{key} must be 25, 50, 75, or 100."),
        ));
    }
    Ok(percent)
}

pub fn worker_budget(available: usize, percent: u32) -> usize {
    if available == 0 {
        0
    } else {
        (available * percent as usize / 100).max(1).min(available)
    }
}

pub fn vram_budget_percent(config: &BenchmarkConfig) -> Result<u32, BenchmarkError> {
    let percent = config
        .options
        .get("vram_budget_percent")
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|_| {
            BenchmarkError::new(
                "invalid_config",
                "VRAM budget must be between 20 and 80 percent.",
            )
        })?
        .unwrap_or(25);
    if !(20..=80).contains(&percent) {
        return Err(BenchmarkError::new(
            "invalid_config",
            "VRAM budget must be between 20 and 80 percent.",
        ));
    }
    Ok(percent)
}

pub fn vram_budget_bytes(total: u64, percent: u32) -> u64 {
    (total / 100).saturating_mul(u64::from(percent))
}

/// Resolve an explicit scaling dataset before allocation. A request must fit the budget;
/// kernel alignment/shape rounding is reported separately in the result's actual tier size.
pub fn configured_dataset(
    config: &BenchmarkConfig,
    limit: u64,
) -> Result<(u64, bool), BenchmarkError> {
    let mode = config
        .options
        .get("dataset_mode")
        .map(String::as_str)
        .unwrap_or("automatic");
    if mode == "automatic" {
        return Ok((limit, false));
    }
    if !["sweep", "single"].contains(&mode) {
        return Err(BenchmarkError::new(
            "invalid_config",
            "Dataset mode must be automatic, sweep, or single.",
        ));
    }
    let bytes = config
        .options
        .get("dataset_bytes")
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|bytes| *bytes >= 256 * 1024)
        .ok_or_else(|| {
            BenchmarkError::new("invalid_config", "Choose a dataset of at least 256 KiB.")
        })?;
    if bytes > limit {
        return Err(BenchmarkError::new(
            "dataset_exceeds_budget",
            format!(
                "Requested dataset is {:.2} MiB, but the current allocation budgets/device limits allow {:.2} MiB. Reduce the dataset or increase the relevant budget.",
                bytes as f64 / 1048576.0,
                limit as f64 / 1048576.0
            ),
        ));
    }
    Ok((bytes, mode == "single"))
}

thread_local! {
    static GPU_PACING: RefCell<Option<(u32, CancellationToken)>> = const { RefCell::new(None) };
}

// A worker runs one benchmark at a time. Pacing is scoped to its submission thread.
pub struct WorkloadGuard(Option<(u32, CancellationToken)>);

impl WorkloadGuard {
    pub fn enter(
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
    ) -> Result<Self, BenchmarkError> {
        workload_percent(config, "cpu_worker_percent")?;
        vram_budget_percent(config)?;
        let percent = workload_percent(config, "gpu_activity_percent")?;
        let old = GPU_PACING.with(|state| state.replace(Some((percent, cancellation.clone()))));
        Ok(Self(old))
    }
}

impl Drop for WorkloadGuard {
    fn drop(&mut self) {
        GPU_PACING.with(|state| {
            state.replace(self.0.take());
        });
    }
}

pub fn gpu_activity_percent() -> u32 {
    GPU_PACING.with(|state| {
        state
            .borrow()
            .as_ref()
            .map(|(percent, _)| *percent)
            .unwrap_or(100)
    })
}

pub fn gpu_burst_duration(duration: Duration) -> Duration {
    if gpu_activity_percent() < 100 {
        duration.min(Duration::from_millis(50))
    } else {
        duration
    }
}

fn idle_duration(active: Duration, percent: u32) -> Duration {
    active.mul_f64(f64::from(100 - percent) / f64::from(percent))
}

// Call only after completed GPU work. The pause is outside GPU timestamp scores.
pub fn pace_gpu(active: Duration) -> Result<(), BenchmarkError> {
    let policy = GPU_PACING.with(|state| state.borrow().clone());
    let Some((percent, cancellation)) = policy else {
        return Ok(());
    };
    let idle = idle_duration(active, percent);
    let started = Instant::now();
    loop {
        if cancellation.is_cancelled() {
            return Err(BenchmarkError::new(
                "cancelled",
                "The benchmark was cancelled.",
            ));
        }
        let remaining = idle.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Ok(());
        }
        std::thread::sleep(remaining.min(Duration::from_millis(10)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_datasets_are_single_or_sweeps_and_never_silently_clamped() {
        let mut config = BenchmarkConfig::default();
        assert_eq!(
            configured_dataset(&config, 1024 * 1024).unwrap(),
            (1024 * 1024, false)
        );
        config
            .options
            .insert("dataset_mode".into(), "single".into());
        assert!(configured_dataset(&config, 1024 * 1024).is_err());
        config
            .options
            .insert("dataset_bytes".into(), "524288".into());
        assert_eq!(
            configured_dataset(&config, 1024 * 1024).unwrap(),
            (524288, true)
        );
        config.options.insert("dataset_mode".into(), "sweep".into());
        assert_eq!(
            configured_dataset(&config, 1024 * 1024).unwrap(),
            (524288, false)
        );
        assert_eq!(
            configured_dataset(&config, 262144).unwrap_err().code,
            "dataset_exceeds_budget"
        );
        config
            .options
            .insert("dataset_mode".into(), "unknown".into());
        assert!(configured_dataset(&config, 1024 * 1024).is_err());
    }
    #[test]
    fn reduced_modes_leave_workers_free_and_add_proportional_idle_time() {
        assert_eq!(worker_budget(16, 75), 12);
        assert_eq!(worker_budget(1, 25), 1);
        assert_eq!(worker_budget(0, 75), 0);
        assert_eq!(
            idle_duration(Duration::from_millis(30), 75),
            Duration::from_millis(10)
        );
        assert_eq!(
            idle_duration(Duration::from_millis(30), 50),
            Duration::from_millis(30)
        );
        assert_eq!(
            idle_duration(Duration::from_millis(30), 100),
            Duration::ZERO
        );
    }
    #[test]
    fn vram_budget_accepts_only_twenty_to_eighty_percent() {
        let mut config = BenchmarkConfig::default();
        assert_eq!(vram_budget_percent(&config).unwrap(), 25);
        for percent in [20, 25, 80] {
            config
                .options
                .insert("vram_budget_percent".into(), percent.to_string());
            assert_eq!(vram_budget_percent(&config).unwrap(), percent);
            assert!(vram_budget_bytes(24 * 1024 * 1024 * 1024, percent) <= 24 * 1024 * 1024 * 1024);
        }
        for value in ["19", "81", "invalid"] {
            config
                .options
                .insert("vram_budget_percent".into(), value.into());
            assert!(vram_budget_percent(&config).is_err());
        }
    }
    #[test]
    fn scoped_gpu_pacing_restores_defaults_and_cancellation_remains_responsive() {
        let mut config = BenchmarkConfig::default();
        config
            .options
            .insert("gpu_activity_percent".into(), "50".into());
        let cancellation = CancellationToken::default();
        let guard = WorkloadGuard::enter(&config, &cancellation).unwrap();
        assert_eq!(gpu_activity_percent(), 50);
        assert_eq!(
            gpu_burst_duration(Duration::from_secs(1)),
            Duration::from_millis(50)
        );
        cancellation.cancel();
        assert_eq!(
            pace_gpu(Duration::from_secs(1)).unwrap_err().code,
            "cancelled"
        );
        drop(guard);
        assert_eq!(gpu_activity_percent(), 100);
        assert_eq!(
            gpu_burst_duration(Duration::from_secs(1)),
            Duration::from_secs(1)
        );
        config
            .options
            .insert("gpu_activity_percent".into(), "0".into());
        assert!(WorkloadGuard::enter(&config, &CancellationToken::default()).is_err());
    }
}
