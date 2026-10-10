use gluj_bench_core::{BenchmarkError, CancellationToken, cpu_activity_percent};
use std::time::{Duration, Instant};

/// Account kernel work in batches, then pause outside the reported measurement.
/// Each worker owns its pacer; the selected core count is unchanged.
pub(crate) struct CpuPacer {
    percent: u32,
    active: Duration,
    idle: Duration,
    cancellation: CancellationToken,
}

impl CpuPacer {
    pub fn current(cancellation: &CancellationToken) -> Self {
        Self::new(cpu_activity_percent(), cancellation)
    }

    pub fn new(percent: u32, cancellation: &CancellationToken) -> Self {
        Self {
            percent,
            active: Duration::ZERO,
            idle: Duration::ZERO,
            cancellation: cancellation.clone(),
        }
    }

    pub fn idle(&self) -> Duration {
        self.idle
    }

    pub fn active_elapsed(&self, start: Instant, idle_before: Duration) -> Duration {
        start
            .elapsed()
            .saturating_sub(self.idle.saturating_sub(idle_before))
    }

    pub fn account(&mut self, active: Duration) -> Result<(), BenchmarkError> {
        if self.cancellation.is_cancelled() {
            return Err(BenchmarkError::new("cancelled", "CPU workload cancelled."));
        }
        if self.percent == 100 {
            return Ok(());
        }
        self.active = self.active.saturating_add(active);
        if self.active < Duration::from_millis(50) {
            return Ok(());
        }
        let pause = self
            .active
            .mul_f64(f64::from(100 - self.percent) / f64::from(self.percent));
        self.active = Duration::ZERO;
        let start = Instant::now();
        loop {
            if self.cancellation.is_cancelled() {
                return Err(BenchmarkError::new("cancelled", "CPU workload cancelled."));
            }
            let remaining = pause.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                break;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(1)));
        }
        self.idle = self.idle.saturating_add(start.elapsed());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intensity_pauses_work_without_changing_worker_count_and_full_has_no_idle() {
        let token = CancellationToken::default();
        let mut paced = CpuPacer::new(95, &token);
        paced.account(Duration::from_millis(95)).unwrap();
        assert!(paced.idle() >= Duration::from_millis(5));
        let mut full = CpuPacer::new(100, &token);
        full.account(Duration::from_secs(1)).unwrap();
        assert_eq!(full.idle(), Duration::ZERO);
        token.cancel();
        assert_eq!(
            paced.account(Duration::from_millis(50)).unwrap_err().code,
            "cancelled"
        );
    }
}
