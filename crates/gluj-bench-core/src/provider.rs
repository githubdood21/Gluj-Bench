use crate::{
    BenchmarkConfig, BenchmarkDescriptor, BenchmarkError, BenchmarkResult, CancellationToken,
    DeviceDescriptor, ProgressUpdate,
};

pub type ProgressCallback<'a> = dyn FnMut(ProgressUpdate) + Send + 'a;

pub trait BenchmarkProvider: Send {
    fn devices(&self) -> Vec<DeviceDescriptor>;
    fn benchmarks(&self) -> Vec<BenchmarkDescriptor>;

    fn run(
        &mut self,
        benchmark_id: &str,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError>;
}
