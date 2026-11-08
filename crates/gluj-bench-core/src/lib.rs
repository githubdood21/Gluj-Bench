mod phase_five;
pub use phase_five::*;
mod access;
pub use access::*;
mod cancellation;
mod catalogue;
mod contracts;
mod efficiency;
mod history;
mod history_actions;
mod measurement;
mod model;
mod protocol;
mod provider;
mod registry;
mod workload;

/// Nominal dynamic traffic at a dataset size, anchored at 1 MiB.
/// Four times the dataset requests twice the traffic and half the passes/s.
pub fn dynamic_offload_traffic(base_bytes_per_second: f64, dataset_bytes: u64) -> f64 {
    base_bytes_per_second * (dataset_bytes.max(1) as f64 / 1_048_576.0).sqrt()
}

pub use cancellation::CancellationToken;
pub use catalogue::{family_catalogue, family_id, legacy_condition};
pub use contracts::*;
pub use efficiency::{cpu_efficiency_trials, gpu_efficiency_trials};
pub use history::*;
pub use history_actions::*;
pub use measurement::*;
pub use model::*;
pub use protocol::{
    ProtocolCommand, ProtocolRequest, benchmark_error_response, error_response, parse_request,
    progress_response, result_response, success_response,
};
pub use provider::{BenchmarkProvider, ProgressCallback, ProviderMeasurement};
pub use registry::BenchmarkRegistry;
pub use workload::{
    WorkloadGuard, configured_dataset, cpu_activity_percent, cpu_matrix_bytes,
    cpu_matrix_dimension, gpu_activity_percent, gpu_burst_duration, gpu_fp16_matrix_dataset_range,
    gpu_fp32_dataset_range, gpu_fp32_offload_dataset_range, gpu_int8_matrix_dataset_range,
    gpu_mixed_allocation_limit, gpu_offload_ram_budget, pace_gpu, ram_allocation_budget,
    vram_budget_bytes, vram_budget_percent, worker_budget, workload_percent,
};

pub const PROTOCOL_VERSION: u32 = 2;

pub const GPU_TIMING_EXPLANATION: &str = "Lower time is faster for the same dataset and settings. Larger datasets contain more work; \
     use the throughput graph to compare processing rates. GPU execution measures the GPU timestamp interval, including computation \
     and memory stalls. End-to-end measures the full sample on the host, including command recording, \
     submission, completion, result retrieval and intentional GPU activity pauses. Setup, warmup \
     and compute-reference runs are excluded. Both times are divided by the number of full dataset \
     passes in each sample batch. Batch lengths can vary across samples and sizes; inspect the recorded raw work and the selected dataset \
     for the recorded batch size. These timings do not separate memory stalls from computation.";

pub const COMPUTE_MEMORY_BALANCE_GUIDANCE: &str = "When dataset scaling indicates memory pressure, test a small supported RAM/VRAM clock increase with compute settings fixed. Higher compute clocks are not the suggested response to that result; identify a compute limit before targeting compute clocks. Keep only stable, error-free throughput improvements on the same workload. Conversely, if a resource has spare capacity, test reducing its clock or CPU core allocation while retaining at least 95% of original throughput. Balance means supplying data as fast as this workload can process it, not equal clock speeds or a literal 1:1 ratio of TOPS to GB/s. A throughput gap is not a memory-clock multiplier; effective traffic can include cache reuse and kernel overhead. Verify application completion time, power use and temperatures separately.";

pub const GPU_OFFLOAD_TUNING_TAKEAWAY: &str =
    "Keep more of the workload in VRAM, or reduce the offloaded portion if possible.";

pub const GPU_OFFLOAD_TUNING_GUIDANCE: &str = "Keep more of the workload in VRAM, or reduce the offloaded portion if possible. Compare end-to-end completion time for the same total work. Smaller batches alone do not necessarily reduce total host traffic. Static size tests one dataset; dynamic size sweeps up to the chosen size. Static traffic is the default and holds passes per second fixed. Dynamic traffic doubles nominal bytes per second and halves passes per second for every fourfold dataset increase, relative to the configured traffic at 1 MiB. Nominal host traffic counts accesses to the RAM region, including cache hits; this is not a PCIe bus counter or proof that the link is fully saturated. Core-clock changes are not suggested for this host-memory path. GPU power use and savings while waiting are not measured. With configured traffic, requests arrive independently of completions. Average throughput counts completed work over the observation window; request completion delay and pending work expose overload. Requested traffic is not achieved bandwidth. Active GPU service throughput is recorded separately.";

pub use catalogue::gpu_buffered_vector;

pub use catalogue::{gpu_memory, gpu_transfer};

/// Optional portable/qualification data root; executables remain beside the UI.
/// Child workers inherit the same environment, so records and preferences agree.
pub fn app_data_path(name: &str) -> std::path::PathBuf {
    let root = std::env::var_os("GLUJ_BENCH_DATA_DIR")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(std::path::Path::to_path_buf))
                .unwrap_or_else(|| ".".into())
        });
    root.join(name)
}
