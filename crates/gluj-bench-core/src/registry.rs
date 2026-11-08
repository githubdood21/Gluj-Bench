use crate::{
    BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkProvider, BenchmarkResult,
    CancellationToken, DeviceDescriptor, ProgressCallback,
};

#[derive(Default)]
pub struct BenchmarkRegistry {
    providers: Vec<Box<dyn BenchmarkProvider>>,
}

impl BenchmarkRegistry {
    pub fn conditions(&self) -> Vec<crate::ConditionDescriptor> {
        self.providers
            .iter()
            .flat_map(|provider| provider.conditions())
            .collect()
    }

    pub fn families(&self) -> Vec<crate::FamilyDescriptor> {
        crate::family_catalogue(self.conditions())
    }

    pub fn resolve(
        &self,
        id: &str,
        config: &BenchmarkConfig,
    ) -> Result<crate::ResolvedCondition, BenchmarkError> {
        let conditions = self.conditions();
        let descriptor = conditions
            .iter()
            .find(|condition| condition.benchmark_id == id)
            .ok_or_else(|| {
                BenchmarkError::new(
                    "unsupported_benchmark",
                    format!("No benchmark is registered with id '{id}'."),
                )
            })?;
        let devices = self.devices();
        let resolved = descriptor.resolve(config, &devices)?;
        if id == "cpu.performance.single_thread.integer.i64" {
            let mut canonical = resolved.config();
            canonical
                .options
                .insert("cpu_core_limit".into(), "1".into());
            canonical
                .options
                .insert("thread_mode".into(), "physical_cores".into());
            canonical
                .options
                .insert("int64_preset".into(), "register".into());
            canonical
                .options
                .insert("dataset_mode".into(), "register".into());
            return conditions
                .iter()
                .find(|condition| condition.benchmark_id == "cpu.performance.integer.i64")
                .ok_or_else(|| {
                    BenchmarkError::new(
                        "unsupported_benchmark",
                        "Canonical INT64 runner is unavailable.",
                    )
                })?
                .resolve(&canonical, &devices);
        }
        Ok(resolved)
    }

    pub fn run_recorded(
        &mut self,
        id: &str,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> crate::RunRecord {
        let mut record = crate::RunRecord::start(id, config, self.devices());
        let outcome = (|| {
            let actual = freeze_access_seed(config);
            let resolved = self.resolve(id, &actual)?;
            let config = resolved.config();
            let canonical_id = resolved.benchmark_id.clone();
            record.resolved = Some(resolved);
            let measurement = self.execute(&canonical_id, &config, cancellation, progress)?;
            record.samples = measurement.samples;
            record.realizations = measurement.realizations;
            record.blocks = measurement.blocks;
            record.reference_anchors = measurement.reference_anchors;
            record.phase_timings = measurement.phase_timings;
            record.sample_coverage = measurement.sample_coverage;
            measurement.result
        })();
        record.finish(outcome);
        if let Err(error) = record.validate() {
            record.samples.clear();
            record.realizations.clear();
            record.blocks.clear();
            record.reference_anchors.clear();
            record.sample_coverage = crate::SampleCoverage::SummaryOnly;
            record.finish(Err(error));
        }
        record
    }
    pub fn add(&mut self, provider: Box<dyn BenchmarkProvider>) {
        self.providers.push(provider);
    }

    pub fn devices(&self) -> Vec<DeviceDescriptor> {
        self.providers
            .iter()
            .flat_map(|provider| provider.devices())
            .collect()
    }

    pub fn benchmarks(&self) -> Vec<BenchmarkDescriptor> {
        self.providers
            .iter()
            .flat_map(|provider| provider.benchmarks())
            .collect()
    }

    pub fn run(
        &mut self,
        benchmark_id: &str,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let actual = freeze_access_seed(config);
        let resolved = self.resolve(benchmark_id, &actual)?;
        self.execute(
            &resolved.benchmark_id,
            &resolved.config(),
            cancellation,
            progress,
        )
        .and_then(|measurement| measurement.result)
    }

    fn execute(
        &mut self,
        benchmark_id: &str,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<crate::ProviderMeasurement, BenchmarkError> {
        let _workload = crate::WorkloadGuard::enter(config, cancellation)?;
        if cancellation.is_cancelled() {
            return Err(BenchmarkError::new(
                "cancelled",
                "The request was cancelled.",
            ));
        }

        for provider in &mut self.providers {
            if provider
                .benchmarks()
                .iter()
                .any(|benchmark| benchmark.id == benchmark_id)
            {
                let mut measurement =
                    provider.run_with_samples(benchmark_id, config, cancellation, progress);
                if cancellation.is_cancelled() && measurement.result.is_ok() {
                    measurement.result = Err(BenchmarkError::new(
                        "cancelled",
                        "The condition was cancelled; completed observations are retained without a successful score.",
                    ));
                    measurement.sample_coverage = if measurement.samples.is_empty() {
                        crate::SampleCoverage::SummaryOnly
                    } else {
                        crate::SampleCoverage::Partial
                    };
                }
                let Ok(result) = &mut measurement.result else {
                    return Ok(measurement);
                };
                for key in ["cpu_activity_percent", "gpu_activity_percent"] {
                    result.workload_metadata.insert(
                        key.into(),
                        crate::workload_percent(config, key)?.to_string(),
                    );
                }
                if benchmark_id.starts_with("cpu.") || benchmark_id.starts_with("memory.") {
                    result
                        .workload_metadata
                        .insert("cpu_pacing_revision".into(), "1".into());
                    result.workload_metadata.insert(
                        "cpu_pacing_timing".into(),
                        "intentional idle intervals excluded from measured rates and latency"
                            .into(),
                    );
                }
                if benchmark_id.starts_with("gpu.") && benchmark_id.ends_with(".scaling") {
                    result.workload_metadata.insert(
                        "vram_budget_percent".into(),
                        crate::vram_budget_percent(config)?.to_string(),
                    );
                }
                if benchmark_id.ends_with(".scaling") {
                    result
                        .workload_metadata
                        .entry("dataset_mode".into())
                        .or_insert_with(|| {
                            config
                                .options
                                .get("dataset_mode")
                                .cloned()
                                .unwrap_or_else(|| "automatic".into())
                        });
                    if result
                        .workload_metadata
                        .get("dataset_mode")
                        .map(String::as_str)
                        != Some("register")
                        && let Some(bytes) = config.options.get("dataset_bytes")
                    {
                        result
                            .workload_metadata
                            .entry("requested_dataset_bytes".into())
                            .or_insert_with(|| bytes.clone());
                    }
                }
                if (benchmark_id.starts_with("cpu.") || benchmark_id.starts_with("memory."))
                    && let Some(limit) = config.options.get("cpu_core_limit")
                {
                    result
                        .workload_metadata
                        .insert("cpu_core_limit".into(), limit.clone());
                }
                return Ok(measurement);
            }
        }

        Err(BenchmarkError::new(
            "unsupported_benchmark",
            format!("No benchmark is registered with id '{benchmark_id}'."),
        ))
    }
}

fn freeze_access_seed(config: &BenchmarkConfig) -> BenchmarkConfig {
    let mut actual = config.clone();
    if actual
        .options
        .get("access_seed_policy")
        .is_some_and(|s| s == "fresh")
    {
        actual
            .options
            .insert("access_seed".into(), crate::fresh_access_seed().to_string());
        actual
            .options
            .insert("access_seed_policy".into(), "fixed".into());
    }
    actual
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProgressUpdate;

    #[test]
    fn unsupported_ids_are_distinct_from_placeholders() {
        let mut registry = BenchmarkRegistry::default();
        let cancellation = CancellationToken::default();
        let mut progress = |_update: ProgressUpdate| {};
        let problem = registry
            .run(
                "missing",
                &BenchmarkConfig::default(),
                &cancellation,
                &mut progress,
            )
            .expect_err("unknown benchmark must fail");
        assert_eq!(problem.code, "unsupported_benchmark");
    }
}

#[cfg(test)]
mod access_seed_tests {
    use super::*;
    #[test]
    fn fresh_seed_freezes_once_and_preserves_requested_provenance() {
        let mut requested = BenchmarkConfig::default();
        requested
            .options
            .insert("access_seed_policy".into(), "fresh".into());
        requested.options.insert("access_seed".into(), "7".into());
        let a = freeze_access_seed(&requested);
        let b = freeze_access_seed(&requested);
        assert_eq!(requested.options["access_seed"], "7");
        assert_eq!(requested.options["access_seed_policy"], "fresh");
        assert_eq!(a.options["access_seed_policy"], "fixed");
        assert_ne!(a.options["access_seed"], b.options["access_seed"]);
        assert_eq!(freeze_access_seed(&a), a);
    }
}
