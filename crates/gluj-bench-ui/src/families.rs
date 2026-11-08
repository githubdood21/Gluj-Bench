//! Family presentation and explicit, capability-checked execution plans.
use super::*;
use gluj_bench_core::{CompletionStatus, ConditionDescriptor, SuiteCondition, SuitePlan};

pub(super) fn title(id: &str) -> &str {
    match id {
        "cpu.memory_bandwidth" => "CPU memory bandwidth",
        "cpu.dependent_latency" => "CPU dependent-read latency",
        "cpu.fp32" => "CPU FP32 arithmetic",
        "cpu.fp64" => "CPU FP64 arithmetic",
        "cpu.matrix" => "CPU FP32 matrix compute",
        "cpu.int64" => "CPU INT64 arithmetic",
        "cpu.ascii" => "CPU ASCII scan",
        "cpu.deflate" => "CPU DEFLATE",
        "cpu.sieve" => "CPU prime sieve",
        "cpu.popcnt" => "CPU population count",
        "cpu.aes_rounds" => "CPU AES rounds",
        "gpu.local_bandwidth" => "GPU local-memory bandwidth",
        "gpu.host_transfers" => "GPU host-device transfers",
        "gpu.vector" => "GPU vector compute",
        "gpu.matrix" => "GPU matrix compute",
        _ => id,
    }
}
fn retired(id: &str) -> bool {
    id == "cpu.performance.single_thread.integer.i64"
}
fn implementation_detail(id: &str) -> bool {
    id.contains(".sparse.") || id.ends_with(".fp8")
}
pub(super) fn mode_name(b: &BenchmarkDescriptor) -> String {
    match b.id.as_str() {
        "gpu.bandwidth.cache" => "Cache-oriented shader bandwidth".into(),
        "gpu.bandwidth.vram" => "Streaming / VRAM bandwidth".into(),
        CPU_VECTOR_ID => "FP32 AVX2/FMA | registers or buffered".into(),
        "cpu.performance.float.f32" => "FP32 scalar | registers".into(),
        "cpu.performance.float.f64" => "FP64 scalar | registers or buffered".into(),
        "cpu.performance.avx2.f64_fma" => "FP64 AVX2/FMA | registers or buffered".into(),
        GPU_VECTOR_ID => "FP32 | GPU-local registers or buffered".into(),
        GPU_OFFLOAD_ID => "FP32 | direct system-RAM access".into(),
        "gpu.performance.fp16" => "FP16 | registers or buffered".into(),
        "gpu.performance.fp64" => "FP64 | registers or buffered".into(),
        "cpu.performance.compression.deflate" => "Compression".into(),
        "cpu.performance.decompression.deflate" => "Decompression".into(),
        "cpu.latency.memory" => "RAM | scattered dependent reads".into(),
        "cpu.latency.memory.localized" => "RAM | localized dependent reads".into(),
        _ => b.name.clone(),
    }
}

#[derive(Clone)]
pub(super) struct PreparedPlan {
    pub runs: VecDeque<QueuedRun>,
    pub suite: SuitePlan,
}
impl PreparedPlan {
    #[cfg(test)]
    pub fn details(&self) -> String {
        self.suite
            .conditions
            .iter()
            .map(|c| {
                let options = &c.condition.options;
                let workers = if c.condition.benchmark_id.starts_with("gpu.") {
                    "GPU".into()
                } else if c.condition.benchmark_id.starts_with("cpu.latency.") {
                    "1 pinned CPU worker".into()
                } else {
                    format!(
                        "{} physical workers",
                        options
                            .get("cpu_core_limit")
                            .map(String::as_str)
                            .unwrap_or("runtime-selected")
                    )
                };
                let dataset = options
                    .get("dataset_bytes")
                    .and_then(|n| n.parse::<u64>().ok())
                    .map(|n| {
                        let per_worker = options
                            .get("cpu_core_limit")
                            .and_then(|n| n.parse::<u64>().ok())
                            .filter(|n| *n > 0);
                        format!(
                            "{} aggregate target{}",
                            format_binary_size(n as f64),
                            per_worker
                                .map(|cores| format!(
                                    "; about {} / worker before alignment",
                                    format_binary_size((n / cores) as f64)
                                ))
                                .unwrap_or_default()
                        )
                    })
                    .unwrap_or_else(|| {
                        if let Some(shape)=options.get("matrix_shape").and_then(|v|gluj_bench_core::cpu_matrix_shape(v)) {
                            return format!("global {} × {} × {} matrix; {} aggregate A/B/C; immutable B shared",shape[0],shape[1],shape[2],format_binary_size((4*(shape[0]*shape[2]+shape[2]*shape[1]+shape[0]*shape[1])) as f64));
                        }
                        if options.get("dataset_mode").is_some_and(|s| s == "register") {
                            "registers; no streaming dataset".into()
                        } else {
                            "fixed or runtime-selected dataset".into()
                        }
                    });
                format!(
                    "{} | {} | {workers} | {dataset} | {} access | {} realization(s){}",
                    c.intended_coverage
                        .get(1)
                        .map(String::as_str)
                        .unwrap_or(&c.condition.benchmark_id),
                    c.condition.device_id,
                    options
                        .get("access_pattern")
                        .or_else(|| options.get("matrix_traversal"))
                        .map(String::as_str)
                        .unwrap_or("runner-defined"),
                    options
                        .get("access_realizations")
                        .map(String::as_str)
                        .unwrap_or("1"),
                    ["dataset_mode","memory_operation","memory_copy_path","transfer_direction","staging_buffers","transfer_batch","transfer_submission","matrix_tile_shape","matrix_tile_reuse"].iter().filter_map(|key|options.get(*key).map(|value|format!(" | {}: {value}",key.replace('_'," ")))).collect::<String>()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
impl App {
    pub(super) fn gpu_tile_names(&self, id: &str) -> Vec<String> {
        let Some(b) = self.benchmarks.iter().find(|b| b.id == id) else {
            return vec!["automatic".into()];
        };
        let c = self.condition_for(b);
        let Some(option) = c.options.iter().find(|o| o.key == "matrix_tile_shape") else {
            return vec!["automatic".into()];
        };
        let gluj_bench_core::OptionDomain::Choice { values } = &option.domain else {
            return vec!["automatic".into()];
        };
        values
            .iter()
            .filter(|v| {
                c.capabilities
                    .iter()
                    .filter(|cap| cap.when.get("matrix_tile_shape") == Some(*v))
                    .all(|cap| {
                        self.selected_gpu_id
                            .as_ref()
                            .is_some_and(|id| cap.supported_device_ids.contains(id))
                    })
            })
            .cloned()
            .collect()
    }

    pub(super) fn finish_plan_condition(&mut self, status: CompletionStatus) {
        if let Some(plan) = &mut self.active_plan {
            if let Some(c) = plan
                .conditions
                .iter_mut()
                .find(|c| c.status == CompletionStatus::Running)
            {
                c.status = status.clone();
            }
            if status != CompletionStatus::Completed
                && !(status == CompletionStatus::Cancelled && self.continue_after_cancel)
            {
                for c in &mut plan.conditions {
                    if c.status == CompletionStatus::Pending {
                        c.status = CompletionStatus::Skipped;
                    }
                }
            }
        }
    }
    pub(super) fn supported_presets(
        &self,
        id: &str,
        settings: &AppSettings,
    ) -> Vec<(usize, String)> {
        let Some(b) = self.benchmarks.iter().find(|b| b.id == id) else {
            return vec![];
        };
        let descriptor = self.condition_for(b);
        let range = self.compute_slider_range(id, settings);
        let indices: &[usize] = if gpu_offload(id) {
            &[0, 2, 5, 6]
        } else if configurable_gpu_compute(id) {
            &[0, 1, 2, 5, 6]
        } else {
            &[0, 1, 2, 3, 4, 5, 6]
        };
        indices
            .iter()
            .filter_map(|preset| {
                if *preset != 1 && !range.available() {
                    return None;
                }
                if !configurable_gpu_compute(id)
                    && (2..=4).contains(preset)
                    && !self.devices.iter().any(|d| {
                        d.category == DeviceCategory::Cpu
                            && d.caches
                                .iter()
                                .any(|c| usize::from(c.level) == *preset - 1 && c.size_bytes > 0)
                    })
                {
                    return None;
                }
                let mut candidate = settings.clone();
                candidate.vector_preset = *preset;
                candidate.vector_size = format!("{} B", range.minimum);
                let run = QueuedRun {
                    benchmark_id: id.into(),
                    settings: candidate,
                    gpu_id: self.selected_gpu_id.clone(),
                    physical_cores: self.physical_core_count(),
                };
                if resolved_ui_config(&descriptor, &run, &self.devices).is_err() {
                    return None;
                }
                let label = if configurable_gpu_compute(id) {
                    match preset {
                        0 => "Use global dataset settings",
                        1 => "Register compute",
                        2 if gpu_offload(id) || id == GPU_VECTOR_ID => "Automatic peak start",
                        2 => "Small dataset",
                        5 if gpu_offload(id) => "Automatic RAM/VRAM budget",
                        5 => "Automatic VRAM budget",
                        _ => "Custom size",
                    }
                } else if buffered_cpu_compute(id) && *preset == 1 {
                    if id == CPU_ASCII_ID {
                        "Automatic scan-sized input"
                    } else {
                        "Automatic compute-sized data"
                    }
                } else {
                    VECTOR_NAMES[*preset]
                };
                Some((*preset, label.into()))
            })
            .collect()
    }
    pub(super) fn condition_for(&self, b: &BenchmarkDescriptor) -> ConditionDescriptor {
        self.condition_descriptors
            .iter()
            .find(|c| c.benchmark_id == b.id)
            .cloned()
            .unwrap_or_else(|| gluj_bench_core::legacy_condition(b))
    }
    pub(super) fn family_of<'a>(&'a self, b: &'a BenchmarkDescriptor) -> &'a str {
        self.condition_descriptors
            .iter()
            .find(|c| c.benchmark_id == b.id)
            .map(|c| c.family_id.as_str())
            .unwrap_or_else(|| gluj_bench_core::family_id(&b.id))
    }
    pub(super) fn family_members(&self, family: &str) -> Vec<&BenchmarkDescriptor> {
        let mut members: Vec<_> = self
            .benchmarks
            .iter()
            .filter(|b| {
                self.family_of(b) == family && !retired(&b.id) && !implementation_detail(&b.id)
            })
            .collect();
        members.sort_by_key(|b| b.display_order);
        members
    }
    pub(super) fn runnable_modes(&self, family: &str) -> Vec<&BenchmarkDescriptor> {
        self.family_members(family)
            .into_iter()
            .filter(|b| self.benchmark_available(b))
            .collect()
    }
    pub(super) fn visible_families(&self) -> Vec<String> {
        let mut families = Vec::new();
        let mut descriptors: Vec<_> = self
            .benchmarks
            .iter()
            .filter(|b| {
                b.suite_id == self.selected_suite
                    && !retired(&b.id)
                    && !implementation_detail(&b.id)
            })
            .collect();
        descriptors.sort_by_key(|b| b.display_order);
        for b in descriptors {
            let family = self.family_of(b).to_owned();
            if !families.contains(&family) {
                families.push(family);
            }
        }
        families
    }
    pub(super) fn selected_family(&self) -> Option<&str> {
        self.selected
            .as_deref()
            .and_then(|id| self.benchmarks.iter().find(|b| b.id == id))
            .map(|b| self.family_of(b))
    }
    pub(super) fn select_family(&mut self, family: &str) {
        if self.active_request.is_some() {
            return;
        }
        let modes = self.runnable_modes(family);
        self.selected = self
            .family_modes
            .get(family)
            .filter(|id| modes.iter().any(|b| &b.id == *id))
            .cloned()
            .or_else(|| modes.first().map(|b| b.id.clone()));
        self.ui_dirty = true;
    }
    pub(super) fn select_family_mode(&mut self, index: i32) {
        if self.active_request.is_some() {
            return;
        }
        let Some(family) = self.selected_family().map(str::to_owned) else {
            return;
        };
        let id = usize::try_from(index)
            .ok()
            .and_then(|i| self.runnable_modes(&family).get(i).map(|b| b.id.clone()));
        if let Some(id) = id {
            self.selected = Some(id.clone());
            self.family_modes.insert(family, id);
            self.persist_benchmark_preferences();
        }
    }
    pub(super) fn capability_details(&self) -> String {
        self.benchmarks
            .iter()
            .filter(|b| {
                b.suite_id == self.selected_suite
                    && (implementation_detail(&b.id) || !self.benchmark_available(b))
            })
            .map(|b| {
                format!(
                    "{}: {}",
                    b.name,
                    if b.unavailable_reason.is_empty() {
                        "unsupported on selected device"
                    } else {
                        &b.unavailable_reason
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
    pub(super) fn core_experiment_plan(&self) -> Result<PreparedPlan, String> {
        let id = self.selected.as_deref().ok_or("Select a CPU mode.")?;
        let mut run = self.queued_run(id)?;
        if gluj_bench_core::cpu_data_mode(id) {
            if run.settings.vector_preset != 6 {
                return Err(
                    "Choose a custom buffered dataset to hold aggregate input fixed across cores."
                        .into(),
                );
            }
            run.settings.vector_sweep = false;
        } else if id == CPU_MATRIX_ID {
            if run.settings.matrix_shape == 0 {
                return Err("Select an explicit global matrix shape for a fixed-work core experiment. The legacy geometry changes with worker count.".into());
            }
        } else if !gluj_bench_core::cpu_codec(id) {
            return Err("Fixed-work core experiments support buffered FP64, data-backed POPCNT, independent DEFLATE blocks and explicit global matrix shapes.".into());
        }
        let mut maximum = if run.settings.cpu_core_limit == 0 {
            self.physical_core_count()
        } else {
            run.settings.cpu_core_limit.min(self.physical_core_count())
        };
        if id == CPU_MATRIX_ID {
            maximum = maximum.min(
                (gluj_bench_core::cpu_matrix_shape(CPU_MATRIX_SHAPES[run.settings.matrix_shape])
                    .unwrap()[0]
                    / 8) as u32,
            );
        }
        if gluj_bench_core::cpu_codec(id) {
            maximum = maximum.min(
                CODEC_SIZES[run.settings.codec_size]
                    .div_ceil(CODEC_BLOCKS[run.settings.codec_block]) as u32,
            );
        }
        if maximum < 2 {
            return Err(
                "At least two supported physical cores are needed for this experiment.".into(),
            );
        }
        run.settings.core_experiment = true;
        run.settings.access_fresh = false;
        let mut counts = vec![1];
        let mut next = 2;
        while next < maximum {
            counts.push(next);
            next *= 2;
        }
        counts.push(maximum);
        self.prepare_plan(
            counts
                .into_iter()
                .map(|cores| {
                    let mut trial = run.clone();
                    trial.settings.cpu_core_limit = cores;
                    trial
                })
                .collect(),
            "fixed-work-core-counts",
        )
    }
    pub(super) fn configured_plan(&self, id: &str) -> Result<PreparedPlan, String> {
        self.prepare_plan(vec![self.queued_run(id)?], "configured-mode")
    }
    pub(super) fn prepare_plan(
        &self,
        candidates: Vec<QueuedRun>,
        label: &str,
    ) -> Result<PreparedPlan, String> {
        let mut runs = VecDeque::new();
        let mut conditions: Vec<SuiteCondition> = Vec::new();
        for mut run in candidates {
            run.settings.shorter_run = self.settings.shorter_run;
            let b = self
                .benchmarks
                .iter()
                .find(|b| b.id == run.benchmark_id)
                .ok_or("Plan contains an unknown mode")?;
            if !benchmark_available_for(b, run.gpu_id.as_deref()) {
                return Err(format!("{} is unavailable on the plan's device", b.name));
            }
            let descriptor = self.condition_for(b);
            let config =
                resolved_ui_config(&descriptor, &run, &self.devices).map_err(|e| e.to_string())?;
            let resolved = descriptor
                .resolve(&config, &self.devices)
                .map_err(|e| e.to_string())?;
            if conditions.iter().any(|c| c.condition == resolved) {
                continue;
            }
            let peak = self.plan_peak(&run);
            conditions.push(SuiteCondition {
                id: format!("condition-{}", conditions.len() + 1),
                intended_coverage: vec![
                    self.family_of(b).into(),
                    mode_name(b),
                    format!(
                        "{} physical-core limit; {} access; {} realization(s)",
                        run.settings.cpu_core_limit,
                        config
                            .options
                            .get("access_pattern")
                            .map(String::as_str)
                            .unwrap_or("runner-defined"),
                        config
                            .options
                            .get("access_realizations")
                            .map(String::as_str)
                            .unwrap_or("1")
                    ),
                ],
                estimated_duration_ms: Some(
                    config.target_duration_ms
                        * config
                            .options
                            .get("access_realizations")
                            .and_then(|v| v.parse::<u64>().ok())
                            .unwrap_or(1),
                ),
                estimated_peak_bytes: peak,
                condition: resolved,
                status: CompletionStatus::Pending,
            });
            runs.push_back(run);
        }
        if runs.is_empty() {
            return Err("No supported conditions in this plan.".into());
        }
        let suite = SuitePlan {
            version: gluj_bench_core::CONTRACT_VERSION,
            id: label.into(),
            order_seed: 0,
            conditions,
        };
        suite.validate().map_err(|e| e.to_string())?;
        Ok(PreparedPlan { runs, suite })
    }
    fn plan_peak(&self, run: &QueuedRun) -> Option<u64> {
        let s = &run.settings;
        let b = self.benchmarks.iter().find(|b| b.id == run.benchmark_id)?;
        if s.dataset_mode != 0
            && (gluj_bench_core::gpu_memory(&run.benchmark_id)
                || gluj_bench_core::gpu_transfer(&run.benchmark_id))
        {
            let payload = s.dataset_mib.checked_mul(1024 * 1024)?;
            // Transfer buffers include every host staging slot and the GPU buffer.
            // Local-memory shader paths reserve two arrays and a checksum word.
            return if gluj_bench_core::gpu_transfer(&run.benchmark_id) {
                payload.checked_mul([1, 3, 8][s.staging_buffers] + 1)
            } else {
                payload.checked_mul(2)?.checked_add(4)
            };
        }
        if run.benchmark_id == CPU_MATRIX_ID && s.matrix_shape > 0 {
            let [m, n, k] = gluj_bench_core::cpu_matrix_shape(CPU_MATRIX_SHAPES[s.matrix_shape])?;
            return Some((4 * (m * k + k * n + m * n)) as u64 + 4 * 1024 * 1024);
        }
        if gluj_bench_core::cpu_codec(&run.benchmark_id) {
            return Some(
                CODEC_SIZES[s.codec_size] * 4
                    + u64::from(if s.cpu_core_limit == 0 {
                        run.physical_cores
                    } else {
                        s.cpu_core_limit
                    }) * 10
                        * 1024
                        * 1024,
            );
        }
        if run.benchmark_id == "cpu.performance.prime.sieve" {
            return Some(
                (SIEVE_LIMITS[s.sieve_size] + 1)
                    * u64::from(if s.cpu_core_limit == 0 {
                        run.physical_cores
                    } else {
                        s.cpu_core_limit
                    })
                    + SIEVE_LIMITS[s.sieve_size],
            );
        }
        if let Some(bytes) = b
            .metadata
            .get("estimated_peak_test_buffer_bytes")
            .and_then(|n| n.parse::<u64>().ok())
        {
            let extra = if gluj_bench_core::access_supported(&run.benchmark_id) {
                bytes / 8
            } else {
                0
            };
            return bytes.checked_add(extra);
        }
        if let Some(bytes) = b
            .metadata
            .get("estimated_peak_test_buffer_bytes_per_worker")
            .and_then(|n| n.parse::<u64>().ok())
        {
            let cores = if s.cpu_core_limit == 0 {
                run.physical_cores
            } else {
                s.cpu_core_limit.min(run.physical_cores)
            };
            // Buffered INT64/ASCII override fixed-register/algorithm estimates below.
            if !gluj_bench_core::access_supported(&run.benchmark_id) || s.vector_preset == 1 {
                return bytes.checked_mul(u64::from(cores));
            }
        }
        if configurable_compute(&run.benchmark_id) && s.vector_preset == 6 {
            // Includes conservative auxiliary/reference space; matrix outputs are additional.
            let target = vector_size_bytes(&s.vector_size).ok()?;
            return target
                .checked_mul(if gpu_matrix_compute(&run.benchmark_id) {
                    4
                } else {
                    1
                })?
                .checked_add(if gluj_bench_core::access_supported(&run.benchmark_id) {
                    target / 8 * if s.access_ensemble { 3 } else { 1 }
                } else {
                    0
                })?
                .checked_add(512 * 1024 * 1024);
        }
        None
    }
    pub(super) fn curated_plan(&self, family: Option<&str>) -> Result<PreparedPlan, String> {
        let mut candidates = Vec::new();
        let mut descriptors: Vec<_> = self
            .benchmarks
            .iter()
            .filter(|b| {
                b.suite_id == self.selected_suite
                    && self.benchmark_available(b)
                    && !retired(&b.id)
                    && !implementation_detail(&b.id)
                    && family.is_none_or(|f| self.family_of(b) == f)
            })
            .collect();
        descriptors.sort_by_key(|b| b.display_order);
        for b in descriptors {
            let descriptor = self.condition_for(b);
            let threaded = descriptor.options.iter().any(|o| o.key == "cpu_core_limit")
                && b.suite_id == "cpu.performance";
            let counts = if threaded && self.physical_core_count() > 1 {
                vec![1, self.physical_core_count()]
            } else {
                vec![self.physical_core_count()]
            };
            let has_register = descriptor.options.iter().any(|o| match &o.domain {
                gluj_bench_core::OptionDomain::Choice { values } => {
                    o.key.ends_with("_preset") && values.iter().any(|v| v == "register")
                }
                _ => false,
            });
            for cores in counts {
                let mut settings = self.settings.clone();
                settings.cpu_core_limit = cores;
                settings.codec_corpus = 1;
                settings.codec_size = 1;
                settings.codec_block = 1;
                settings.codec_level = 2;
                settings.sieve_size = 2;
                settings.matrix_shape = 0;
                settings.matrix_reuse = 0;
                settings.matrix_block = 1;
                settings.sustained_mode = 0;
                settings.gpu_matrix_tile = "automatic".into();
                settings.gpu_matrix_reuse = 2;
                settings.gpu_matrix_traversal = 0;
                settings.memory_operation = 0;
                settings.memory_copy_path = 1;
                if gluj_bench_core::gpu_memory(&b.id) || gluj_bench_core::gpu_transfer(&b.id) {
                    settings.dataset_mode = 1;
                    settings.dataset_mib = 64;
                }
                settings.transfer_direction = 0;
                settings.staging_buffers = 1;
                settings.transfer_batch = 1;
                settings.transfer_submission = 0;
                settings.core_experiment = false;
                settings.access_pattern = 0;
                settings.access_fresh = false;
                settings.access_seed = gluj_bench_core::DEFAULT_ACCESS_SEED.to_string();
                settings.access_ensemble = false;
                settings.access_block = if b.id.starts_with("cpu.bandwidth.cache.") {
                    0
                } else {
                    2
                };
                settings.access_dataset.clear();
                settings.ascii_content = 0;
                if b.id == CPU_RAM_BANDWIDTH_ID {
                    settings.dataset_mode = 0;
                }

                if configurable_compute(&b.id) {
                    if has_register {
                        let mut register = settings.clone();
                        register.vector_preset = 1;
                        register.vector_sweep = false;
                        candidates.push(QueuedRun {
                            benchmark_id: b.id.clone(),
                            settings: register,
                            gpu_id: self.selected_gpu_id.clone(),
                            physical_cores: self.physical_core_count(),
                        });
                    }
                    settings.vector_preset = 6;
                    settings.vector_sweep = true;
                    if gpu_offload(&b.id) {
                        settings.ram_offload_percent = 75;
                        settings.offload_traffic_mode = Some(2);
                        settings.offload_passes_per_second = 100;
                    }
                    let range = self.compute_slider_range(&b.id, &settings);
                    if !range.available() {
                        continue;
                    }
                    settings.vector_size = format!("{} B", range.clamp(CURATED_DATASET_BYTES));
                }
                candidates.push(QueuedRun {
                    benchmark_id: b.id.clone(),
                    settings,
                    gpu_id: self.selected_gpu_id.clone(),
                    physical_cores: self.physical_core_count(),
                });
            }
        }
        let mut shaped = Vec::new();
        for run in candidates {
            if run.benchmark_id == CPU_MATRIX_ID {
                for (shape, shape_name) in CPU_MATRIX_SHAPES.iter().enumerate().skip(1) {
                    if run.settings.cpu_core_limit
                        <= gluj_bench_core::cpu_matrix_shape(shape_name).unwrap()[0] as u32 / 8
                    {
                        let mut fixed = run.clone();
                        fixed.settings.matrix_shape = shape;
                        fixed.settings.matrix_reuse = 0;
                        fixed.settings.matrix_block = 1;
                        shaped.push(fixed);
                    }
                }
            }
            shaped.push(run);
        }
        let expanded = shaped;
        let candidates = expanded;
        // A mode may support registers without storage. Resolve each generated condition;
        // omit only advertised capability failures, never hide a malformed configuration.
        let mut supported = Vec::new();
        for run in candidates {
            let b = self
                .benchmarks
                .iter()
                .find(|b| b.id == run.benchmark_id)
                .unwrap();
            match resolved_ui_config(&self.condition_for(b), &run, &self.devices) {
                Ok(_) => supported.push(run),
                Err(e) if e.code == "mode_unavailable" || e.code == "device_unavailable" => {}
                Err(e) => return Err(e.to_string()),
            }
        }
        self.prepare_plan(supported, family.unwrap_or(&self.selected_suite))
    }
    pub(super) fn begin_plan(&mut self, plan: Result<PreparedPlan, String>) {
        if !self.worker.online() {
            self.status = "Reconnect the benchmark engine before starting a plan.".into();
            self.ui_dirty = true;
            return;
        }
        if self.history_busy() {
            self.status = "Wait for history loading/export to finish before benchmarking.".into();
            self.ui_dirty = true;
            return;
        }
        if self.active_request.is_some() {
            return;
        }
        match plan {
            Ok(plan) => {
                self.presentation.begin_result_batch();
                self.queue = plan.runs;
                self.active_plan = Some(plan.suite);
                self.continue_after_cancel = false;
                self.persist_plan_status();
                self.start_next();
            }
            Err(error) => {
                self.status = error;
                self.ui_dirty = true;
            }
        }
    }
    pub(super) fn add_custom_condition(&mut self) {
        let Some(id) = &self.selected else {
            return;
        };
        let plan = self.configured_plan(id).and_then(|configured| {
            let mut runs = self.custom_plan.clone();
            runs.extend(configured.runs);
            self.prepare_plan(runs, "custom")
        });
        match plan {
            Ok(plan) => {
                self.custom_plan = plan.runs.into();
                self.presentation.scope = 1;
                self.status = format!("Custom list: {} unique tests", self.custom_plan.len());
            }
            Err(error) => self.status = error,
        }
        self.ui_dirty = true;
    }
}

#[cfg(test)]
mod tests;
