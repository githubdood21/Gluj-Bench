mod cancellation;
mod model;
mod protocol;
mod provider;
mod registry;
mod workload;

pub use cancellation::CancellationToken;
pub use model::*;
pub use protocol::{
    ProtocolCommand, ProtocolRequest, benchmark_error_response, error_response, parse_request,
    progress_response, result_response, success_response,
};
pub use provider::{BenchmarkProvider, ProgressCallback};
pub use registry::BenchmarkRegistry;
pub use workload::{
    WorkloadGuard, gpu_activity_percent, gpu_burst_duration, pace_gpu, vram_budget_bytes,
    vram_budget_percent, worker_budget, workload_percent,
};

pub const PROTOCOL_VERSION: u32 = 2;
