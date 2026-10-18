use crate::{BenchmarkConfig, BenchmarkError, CancellationToken};
use std::{
    cell::{Cell, RefCell},
    time::{Duration, Instant},
};

pub fn workload_percent(config: &BenchmarkConfig, key: &str) -> Result<u32, BenchmarkError> {
    let percent = config
        .options
        .get(key)
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|_| {
            BenchmarkError::new("invalid_config", format!("{key} must be 95, 99, or 100."))
        })?
        .unwrap_or(100);
    if ![95, 99, 100].contains(&percent) {
        return Err(BenchmarkError::new(
            "invalid_config",
            format!("{key} must be 95, 99, or 100."),
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

/// Shared CPU buffer limit for sizing previews and allocation at run time.
pub fn ram_allocation_budget(total: u64, available: u64, percent: u32) -> u64 {
    (total / 100 * u64::from(percent))
        .min(available / 5 * 4)
        .min(available.saturating_sub(2 * 1024 * 1024 * 1024))
}

/// FP32 GPU profile: three vec4 arrays, aligned to a workgroup and descriptor offsets.
pub fn gpu_fp32_dataset_range(
    local_memory: u64,
    storage_range: u64,
    offset_alignment: u64,
    percent: u32,
    shared_available: Option<u64>,
) -> (u64, u64, u64) {
    let alignment = 48 * 256_u64.max(offset_alignment / 16);
    let minimum = (256 * 1024_u64).div_ceil(alignment) * alignment;
    let mut maximum = vram_budget_bytes(local_memory, percent)
        .saturating_sub(16 * 1024 * 1024)
        .min(storage_range.saturating_mul(3))
        .min(u64::from(u32::MAX) / (alignment / 48) * alignment);
    if let Some(available) = shared_available {
        maximum =
            maximum.min((available / 5 * 4).min(available.saturating_sub(2 * 1024 * 1024 * 1024)));
    }
    (minimum, maximum / alignment * alignment, alignment)
}

/// Host allocation policy for direct GPU access to system RAM.
pub fn gpu_offload_ram_budget(total: u64, available: u64, heap: u64, percent: u32) -> u64 {
    (total / 100 * u64::from(percent))
        .min(available / 100 * 80)
        .min(available.saturating_sub(2 * 1024 * 1024 * 1024))
        .min(heap)
        .saturating_sub(16 * 1024 * 1024)
}

/// Total bytes allowed by the independent budgets for a fixed RAM/VRAM split.
pub fn gpu_mixed_allocation_limit(vram: u64, ram: u64, offload: u32) -> u64 {
    if offload > 100 {
        return 0;
    }
    let host = if offload == 0 {
        u64::MAX
    } else {
        ram.saturating_mul(100) / u64::from(offload)
    };
    let local = if offload == 100 {
        u64::MAX
    } else {
        vram.saturating_mul(100) / u64::from(100 - offload)
    };
    host.min(local)
}

/// Three vec4 arrays with exact 50/75/100% host shares and complete workgroups.
pub fn gpu_fp32_offload_dataset_range(
    vram_budget: u64,
    ram_budget: u64,
    storage_range: u64,
    offset_alignment: u64,
    offload: u32,
) -> (u64, u64, u64) {
    let alignment = 48 * 256_u64.max(offset_alignment / 16) * 4;
    let minimum = (256 * 1024_u64).div_ceil(alignment) * alignment;
    if ![50, 75, 100].contains(&offload) {
        return (minimum, 0, alignment);
    }
    let maximum = gpu_mixed_allocation_limit(vram_budget, ram_budget, offload)
        .min(storage_range.saturating_mul(3))
        .min(u64::from(u32::MAX) / (alignment / 48) * alignment);
    (minimum, maximum / alignment * alignment, alignment)
}

/// FP16 matrix operands; reserve FP32 output for every supported saturation width.
/// The shared-memory limit also leaves room for the OS and other applications.
pub fn gpu_fp16_matrix_dataset_range(
    local_memory: u64,
    storage_range: u64,
    shape: [u64; 3],
    subgroup_size: u64,
    percent: u32,
    shared_available: Option<u64>,
) -> (u64, u64, u64) {
    gpu_matrix_dataset_range(
        local_memory,
        storage_range,
        shape,
        subgroup_size,
        percent,
        shared_available,
        2,
    )
}

/// INT8 operands with 32-bit output; the same tile layout and memory policy as FP16.
pub fn gpu_int8_matrix_dataset_range(
    local_memory: u64,
    storage_range: u64,
    shape: [u64; 3],
    subgroup_size: u64,
    percent: u32,
    shared_available: Option<u64>,
) -> (u64, u64, u64) {
    gpu_matrix_dataset_range(
        local_memory,
        storage_range,
        shape,
        subgroup_size,
        percent,
        shared_available,
        1,
    )
}

fn gpu_matrix_dataset_range(
    local_memory: u64,
    storage_range: u64,
    shape: [u64; 3],
    subgroup_size: u64,
    percent: u32,
    shared_available: Option<u64>,
    operand_bytes: u64,
) -> (u64, u64, u64) {
    let [m, n, k] = shape;
    if m == 0 || n == 0 || k == 0 || subgroup_size == 0 || 64 % subgroup_size != 0 {
        return (256 * 1024, 0, 1);
    }
    let a = m.saturating_mul(k);
    let b = k.saturating_mul(n);
    let alignment = a.saturating_add(b).saturating_mul(operand_bytes);
    let minimum = (256 * 1024_u64).div_ceil(alignment) * alignment;
    let output_reserve = (8192 * 64 / subgroup_size) * m * n * 4;
    let reserve = output_reserve.saturating_add(16 * 1024 * 1024);
    let tiles = (storage_range / a.saturating_mul(operand_bytes))
        .min(storage_range / b.saturating_mul(operand_bytes))
        .min(u64::from(u32::MAX) / a.max(b));
    let mut maximum = vram_budget_bytes(local_memory, percent)
        .saturating_sub(reserve)
        .min(tiles.saturating_mul(alignment));
    if let Some(available) = shared_available {
        maximum = maximum.min(local_memory / 8).min(
            (available / 5 * 4)
                .min(available.saturating_sub(2 * 1024 * 1024 * 1024))
                .saturating_sub(reserve),
        );
    }
    (minimum, maximum / alignment * alignment, alignment)
}

/// Fixed 32-row CPU matrix layout: A[32,K], B[K,N], C[32,N], K=N.
pub fn cpu_matrix_bytes(dimension: u64) -> u64 {
    4_u64.saturating_mul(
        dimension
            .saturating_mul(dimension)
            .saturating_add(64_u64.saturating_mul(dimension)),
    )
}
pub fn cpu_matrix_dimension(budget: u64) -> u64 {
    let mut dimension = (((1024.0 + budget as f64 / 4.0).sqrt() - 32.0) as u64) / 8 * 8;
    while dimension > 0
        && 4 * (u128::from(dimension) * u128::from(dimension) + 64 * u128::from(dimension))
            > u128::from(budget)
    {
        dimension -= 8;
    }
    dimension
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
    static CPU_ACTIVITY: Cell<u32> = const { Cell::new(100) };
}

// A worker runs one benchmark at a time. Pacing is scoped to its submission thread.
pub struct WorkloadGuard {
    gpu: Option<(u32, CancellationToken)>,
    cpu: u32,
}

impl WorkloadGuard {
    pub fn enter(
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
    ) -> Result<Self, BenchmarkError> {
        let cpu = workload_percent(config, "cpu_activity_percent")?;
        vram_budget_percent(config)?;
        let percent = workload_percent(config, "gpu_activity_percent")?;
        let old = GPU_PACING.with(|state| state.replace(Some((percent, cancellation.clone()))));
        let old_cpu = CPU_ACTIVITY.with(|state| state.replace(cpu));
        Ok(Self {
            gpu: old,
            cpu: old_cpu,
        })
    }
}

impl Drop for WorkloadGuard {
    fn drop(&mut self) {
        GPU_PACING.with(|state| {
            state.replace(self.gpu.take());
        });
        CPU_ACTIVITY.with(|state| state.set(self.cpu));
    }
}

pub fn cpu_activity_percent() -> u32 {
    CPU_ACTIVITY.with(Cell::get)
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
    fn offload_ranges_preserve_exact_shares_and_both_memory_budgets() {
        let gib = 1024_u64.pow(3);
        for share in [50, 75, 100] {
            let (minimum, maximum, alignment) =
                gpu_fp32_offload_dataset_range(2 * gib, 3 * gib, gib, 256, share);
            assert_eq!(minimum, 294912);
            assert_eq!(alignment, 49152);
            assert_eq!(maximum % alignment, 0);
            assert!(maximum * u64::from(share) / 100 <= 3 * gib);
            assert!(maximum * u64::from(100 - share) / 100 <= 2 * gib);
            assert_eq!((maximum / 48 * u64::from(share) / 100) % 256, 0);
            assert!(maximum <= 3 * gib);
        }
        assert_eq!(
            gpu_fp32_offload_dataset_range(0, 3 * gib, gib, 4096, 100).1,
            3 * gib
        );
        assert_eq!(
            gpu_fp32_offload_dataset_range(0, 3 * gib, gib, 256, 50).1,
            0
        );
        assert_eq!(gpu_fp32_offload_dataset_range(gib, 0, gib, 256, 100).1, 0);
        assert_eq!(gpu_fp32_offload_dataset_range(gib, gib, gib, 256, 60).1, 0);
    }

    #[test]
    fn int8_range_uses_one_byte_operands_and_reserves_32_bit_outputs() {
        let gib = 1024_u64.pow(3);
        let (minimum, maximum, alignment) =
            gpu_int8_matrix_dataset_range(8 * gib, 128 * 1024 * 1024, [16, 8, 32], 32, 25, None);
        assert_eq!(minimum, 262656);
        assert_eq!(alignment, 768);
        assert_eq!(maximum, 192 * 1024 * 1024);
        let (_, fp16_maximum, fp16_alignment) =
            gpu_fp16_matrix_dataset_range(8 * gib, gib, [16, 16, 32], 32, 25, None);
        let (_, int8_maximum, int8_alignment) =
            gpu_int8_matrix_dataset_range(8 * gib, gib, [16, 16, 32], 32, 25, None);
        assert_eq!(fp16_alignment, 2 * int8_alignment);
        assert_eq!(int8_maximum / fp16_alignment * fp16_alignment, fp16_maximum);
    }

    #[test]
    fn fp16_operand_range_reserves_output_and_respects_shader_and_shared_limits() {
        let gib = 1024_u64.pow(3);
        let (minimum, maximum, alignment) =
            gpu_fp16_matrix_dataset_range(8 * gib, gib, [16, 16, 16], 32, 25, None);
        assert_eq!((minimum, alignment), (262144, 1024));
        assert_eq!(
            maximum,
            (vram_budget_bytes(8 * gib, 25) - 32 * 1024 * 1024) / 1024 * 1024
        );
        let (_, binding_limited, _) =
            gpu_fp16_matrix_dataset_range(80 * gib, 128 * 1024 * 1024, [16, 8, 32], 32, 80, None);
        assert_eq!(binding_limited, 192 * 1024 * 1024);
        let (_, shared, _) =
            gpu_fp16_matrix_dataset_range(8 * gib, gib, [16, 16, 16], 32, 80, Some(3 * gib));
        assert_eq!(shared, gib - 32 * 1024 * 1024);
        assert_eq!(
            gpu_fp16_matrix_dataset_range(8 * gib, gib, [16, 16, 16], 32, 80, Some(gib)).1,
            0
        );
        assert_eq!(
            gpu_fp16_matrix_dataset_range(8 * gib, gib, [0, 16, 16], 32, 25, None).1,
            0
        );
        assert_eq!(
            gpu_fp16_matrix_dataset_range(8 * gib, gib, [16, 16, 16], 0, 25, None).1,
            0
        );
    }

    #[test]
    fn cpu_matrix_layout_rounds_down_and_handles_large_budgets() {
        for budget in [0, 12287, 12288, 65536, 1024_u64.pow(4), u64::MAX] {
            let dimension = cpu_matrix_dimension(budget);
            assert_eq!(dimension % 8, 0);
            assert!(cpu_matrix_bytes(dimension) <= budget);
            let next = u128::from(dimension + 8);
            assert!(4 * (next * next + 64 * next) > u128::from(budget));
        }
        assert_eq!(cpu_matrix_bytes(32), 12288);
        assert_eq!(cpu_matrix_dimension(12288), 32);
    }
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
        assert_eq!(worker_budget(16, 95), 15);
        assert_eq!(worker_budget(16, 99), 15);
        assert_eq!(worker_budget(16, 100), 16);
        assert_eq!(worker_budget(1, 95), 1);
        assert_eq!(worker_budget(0, 99), 0);
        assert_eq!(
            idle_duration(Duration::from_millis(95), 95),
            Duration::from_millis(5)
        );
        assert_eq!(
            idle_duration(Duration::from_millis(99), 99),
            Duration::from_millis(1)
        );
        assert_eq!(
            idle_duration(Duration::from_millis(30), 100),
            Duration::ZERO
        );
    }

    #[test]
    fn allocation_options_accept_near_full_presets_and_reject_old_values() {
        for key in ["cpu_activity_percent", "gpu_activity_percent"] {
            let mut config = BenchmarkConfig::default();
            assert_eq!(workload_percent(&config, key).unwrap(), 100);
            for percent in [95, 99, 100] {
                config.options.insert(key.into(), percent.to_string());
                assert_eq!(workload_percent(&config, key).unwrap(), percent);
            }
            for value in ["0", "25", "50", "75", "94", "96", "101", "invalid"] {
                config.options.insert(key.into(), value.into());
                assert!(workload_percent(&config, key).is_err());
            }
        }
    }

    #[test]
    fn gpu_fp32_sizing_respects_buffers_alignment_and_shared_memory() {
        let gib = 1024_u64.pow(3);
        for offset in [0, 256, 8192] {
            let (minimum, maximum, alignment) =
                gpu_fp32_dataset_range(16 * gib, gib, offset, 25, None);
            assert!(minimum >= 256 * 1024);
            assert_eq!(minimum % alignment, 0);
            assert_eq!(maximum % alignment, 0);
            assert!(maximum <= 3 * gib);
            assert!(maximum > gib);
            let (_, shared, _) = gpu_fp32_dataset_range(16 * gib, gib, offset, 25, Some(3 * gib));
            assert!(shared <= gib);
        }
        assert_eq!(gpu_fp32_dataset_range(0, 0, 0, 25, None).1, 0);
        assert_eq!(
            gpu_fp32_dataset_range(16 * gib, gib, 256, 25, Some(gib)).1,
            0
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
            .insert("gpu_activity_percent".into(), "99".into());
        config
            .options
            .insert("cpu_activity_percent".into(), "95".into());
        let cancellation = CancellationToken::default();
        let guard = WorkloadGuard::enter(&config, &cancellation).unwrap();
        assert_eq!(gpu_activity_percent(), 99);
        assert_eq!(cpu_activity_percent(), 95);
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
        assert_eq!(cpu_activity_percent(), 100);
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
