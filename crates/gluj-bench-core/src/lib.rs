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
    WorkloadGuard, configured_dataset, gpu_activity_percent, gpu_burst_duration, pace_gpu,
    vram_budget_bytes, vram_budget_percent, worker_budget, workload_percent,
};

pub const PROTOCOL_VERSION: u32 = 2;

pub const GPU_TIMING_EXPLANATION: &str = "Lower time is faster for the same dataset and settings. Larger datasets contain more work; \
     use the throughput graph to compare processing rates. GPU execution measures the GPU timestamp interval, including computation \
     and memory stalls. End-to-end measures the full sample on the host, including command recording, \
     submission, completion, result retrieval and intentional GPU activity pauses. Setup, warmup \
     and compute-reference runs are excluded. Both times are divided by the number of full dataset \
     passes in each sample batch. Batch lengths can vary across sizes; inspect the selected dataset \
     for the recorded batch size. These timings do not separate memory stalls from computation.";

pub const GPU_OFFLOAD_TUNING_TAKEAWAY: &str =
    "Try a lower GPU core-frequency limit or smaller GPU work batches, then retest throughput.";

pub const GPU_OFFLOAD_TUNING_GUIDANCE: &str = "Suggested tuning: reduce GPU demand by lowering the GPU core-frequency limit in small steps, \
     or by sending smaller batches or fewer concurrent GPU jobs in your application. Change one \
     setting at a time. Keep the dataset, RAM offload share and arithmetic settings fixed when \
     checking a frequency-limit change; aim to retain at least 95% of the original throughput, \
     and restore the previous limit if the loss is larger. Judge smaller batches or reduced \
     concurrency using the application's end-to-end throughput and responsiveness. In this \
     benchmark, a lower GPU activity setting adds idle time between submissions; TOPS excludes \
     those pauses, so an unchanged TOPS value does not prove unchanged completion time. These \
     are exploratory trials: this run does not measure performance after a frequency change, \
     power savings or temperature changes.";
