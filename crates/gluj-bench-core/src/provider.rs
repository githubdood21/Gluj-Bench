use crate::{
    BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult, CancellationToken,
    DeviceDescriptor, ProgressUpdate,
};

pub type ProgressCallback<'a> = dyn FnMut(ProgressUpdate) + Send + 'a;

pub struct ProviderMeasurement {
    pub result: Result<BenchmarkResult, BenchmarkError>,
    pub samples: Vec<crate::RawSample>,
    pub sample_coverage: crate::SampleCoverage,
    pub realizations: Vec<crate::WorkloadRealization>,
    pub blocks: Vec<crate::SampleBlock>,
    pub reference_anchors: Vec<crate::ReferenceAnchor>,
    pub phase_timings: Vec<crate::PhaseTiming>,
}

pub trait BenchmarkProvider: Send {
    fn devices(&self) -> Vec<DeviceDescriptor>;
    fn benchmarks(&self) -> Vec<BenchmarkDescriptor>;

    fn conditions(&self) -> Vec<crate::ConditionDescriptor> {
        self.benchmarks()
            .iter()
            .map(crate::legacy_condition)
            .collect()
    }

    /// Existing runners supply summaries only. Phase-2 runners override this to retain
    /// real chronological observations; summary statistics must never be expanded into samples.
    fn run_with_samples(
        &mut self,
        benchmark_id: &str,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> ProviderMeasurement {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let handle = scope.spawn(move || {
                crate::capture_measurement_with_progress(
                    benchmark_id,
                    config,
                    Some(sender.clone()),
                    || {
                        // Pacing is thread-local. Install it on the actual execution thread too.
                        let _workload = crate::WorkloadGuard::enter(config, cancellation)?;
                        let mut report = |update| {
                            let _ = sender.send(update);
                        };
                        self.run(benchmark_id, config, cancellation, &mut report)
                    },
                )
            });
            let mut latest_fraction = 0.0;
            for mut update in receiver {
                if update.phase == "sample_coverage" {
                    update.fraction = latest_fraction;
                } else {
                    latest_fraction = update.fraction;
                }
                progress(update);
            }
            handle.join().unwrap_or_else(|_| ProviderMeasurement {
                result: Err(BenchmarkError::new(
                    "worker_failed",
                    "The measurement thread failed.",
                )),
                samples: vec![],
                sample_coverage: crate::SampleCoverage::SummaryOnly,
                realizations: vec![],
                blocks: vec![],
                reference_anchors: vec![],
                phase_timings: vec![],
            })
        })
    }

    fn run(
        &mut self,
        benchmark_id: &str,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError>;
}
