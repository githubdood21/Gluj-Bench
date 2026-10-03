use crate::{
    BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkProvider, BenchmarkResult,
    CancellationToken, DeviceDescriptor, ProgressCallback,
};

#[derive(Default)]
pub struct BenchmarkRegistry {
    providers: Vec<Box<dyn BenchmarkProvider>>,
}

impl BenchmarkRegistry {
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
                let mut result = provider.run(benchmark_id, config, cancellation, progress)?;
                for key in ["cpu_worker_percent", "gpu_activity_percent"] {
                    result.workload_metadata.insert(
                        key.into(),
                        crate::workload_percent(config, key)?.to_string(),
                    );
                }
                if benchmark_id.starts_with("gpu.") && benchmark_id.ends_with(".scaling") {
                    result.workload_metadata.insert(
                        "vram_budget_percent".into(),
                        crate::vram_budget_percent(config)?.to_string(),
                    );
                }
                if benchmark_id.starts_with("cpu.") || benchmark_id.starts_with("memory.") {
                    if let Some(limit) = config.options.get("cpu_core_limit") {
                        result
                            .workload_metadata
                            .insert("cpu_core_limit".into(), limit.clone());
                    }
                }
                return Ok(result);
            }
        }

        Err(BenchmarkError::new(
            "unsupported_benchmark",
            format!("No benchmark is registered with id '{benchmark_id}'."),
        ))
    }
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
