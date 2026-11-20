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
                let has_dataset_controls = benchmark_id.ends_with(".scaling")
                    || crate::gpu_buffered_vector(benchmark_id)
                    || crate::gpu_memory(benchmark_id)
                    || crate::gpu_transfer(benchmark_id)
                    || crate::cpu_data_mode(benchmark_id)
                    || matches!(
                        benchmark_id,
                        "cpu.performance.integer.i64" | "cpu.performance.string.ascii"
                    );
                if benchmark_id.starts_with("gpu.") && has_dataset_controls {
                    result.workload_metadata.insert(
                        "vram_budget_percent".into(),
                        crate::vram_budget_percent(config)?.to_string(),
                    );
                }
                if has_dataset_controls {
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

    #[test]
    fn vram_budget_and_dataset_metadata_cover_non_scaling_gpu_conditions() {
        use crate::{
            BenchmarkCategory, DeviceCategory, Metric, ProgressCallback, SampleStatistics,
        };
        use std::collections::BTreeMap;

        struct StubGpuProvider;
        impl BenchmarkProvider for StubGpuProvider {
            fn devices(&self) -> Vec<DeviceDescriptor> {
                vec![DeviceDescriptor {
                    id: "gpu-0".into(),
                    name: "Stub GPU".into(),
                    category: DeviceCategory::Gpu,
                    available: true,
                    status: "ready".into(),
                    properties: BTreeMap::new(),
                    caches: Vec::new(),
                }]
            }
            fn benchmarks(&self) -> Vec<BenchmarkDescriptor> {
                [
                    "gpu.bandwidth.vram",
                    "gpu.performance.fp16",
                    "gpu.performance.fp32",
                ]
                .into_iter()
                .map(|id| BenchmarkDescriptor {
                    id: id.into(),
                    name: id.into(),
                    category: BenchmarkCategory::Gpu,
                    workload: "stub".into(),
                    data_type: "fp32".into(),
                    unit: "GB/s".into(),
                    supported_device_ids: vec!["gpu-0".into()],
                    available: true,
                    unavailable_reason: String::new(),
                    suite_id: "gpu".into(),
                    display_order: 0,
                    metadata: BTreeMap::new(),
                })
                .collect()
            }
            fn run(
                &mut self,
                benchmark_id: &str,
                config: &BenchmarkConfig,
                _cancellation: &CancellationToken,
                _progress: &mut ProgressCallback<'_>,
            ) -> Result<BenchmarkResult, BenchmarkError> {
                let sample_ns = (config.target_duration_ms.saturating_mul(1_000_000)
                    / u64::from(config.samples.max(1)))
                .max(250_000_000);
                for _ in 0..config.samples.max(6) {
                    crate::record_observation(
                        "stub",
                        "throughput",
                        10.0,
                        "GB/s",
                        10,
                        "bytes",
                        sample_ns,
                        sample_ns,
                    );
                }
                Ok(BenchmarkResult {
                    benchmark_id: benchmark_id.into(),
                    device_id: "gpu-0".into(),
                    elapsed_ns: 1,
                    metrics: vec![Metric {
                        name: "throughput".into(),
                        value: 10.0,
                        unit: "GB/s".into(),
                        statistics: SampleStatistics {
                            sample_count: config.samples.max(6),
                            minimum: 10.0,
                            median: 10.0,
                            maximum: 10.0,
                            standard_deviation: 0.0,
                        },
                    }],
                    workload_metadata: BTreeMap::new(),
                    device_metadata: BTreeMap::new(),
                })
            }
        }

        let mut registry = BenchmarkRegistry::default();
        registry.add(Box::new(StubGpuProvider));
        let token = CancellationToken::default();
        let mut progress = |_u: ProgressUpdate| {};

        let vram_bw = registry
            .run(
                "gpu.bandwidth.vram",
                &BenchmarkConfig {
                    options: BTreeMap::from([
                        ("vram_budget_percent".into(), "40".into()),
                        ("dataset_mode".into(), "single".into()),
                        ("dataset_bytes".into(), "1048576".into()),
                    ]),
                    ..Default::default()
                },
                &token,
                &mut progress,
            )
            .unwrap();
        assert_eq!(vram_bw.workload_metadata["vram_budget_percent"], "40");
        assert_eq!(vram_bw.workload_metadata["dataset_mode"], "single");
        assert_eq!(
            vram_bw.workload_metadata["requested_dataset_bytes"],
            "1048576"
        );

        let fp16 = registry
            .run(
                "gpu.performance.fp16",
                &BenchmarkConfig::default(),
                &token,
                &mut progress,
            )
            .unwrap();
        assert_eq!(fp16.workload_metadata["vram_budget_percent"], "25");

        let fp32 = registry
            .run(
                "gpu.performance.fp32",
                &BenchmarkConfig::default(),
                &token,
                &mut progress,
            )
            .unwrap();
        assert!(!fp32.workload_metadata.contains_key("vram_budget_percent"));
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
