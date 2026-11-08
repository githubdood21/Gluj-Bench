//! Configurable CPU workloads with fixed input geometry and bounded observations.
mod buffers;
use crate::topology::{AffinityGuard, CpuTopology};
use buffers::State;
use gluj_bench_core::{
    BenchmarkConfig, BenchmarkError, BenchmarkResult, CancellationToken, DeviceDescriptor, Metric,
    ProgressCallback, ProgressUpdate,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
};
#[derive(Clone, Copy, Debug)]
enum Spec {
    Float(bool),
    Pop,
    Codec(bool),
    Sieve,
    Matrix,
}
impl Spec {
    fn from_id(id: &str, config: &BenchmarkConfig) -> Option<Self> {
        match id {
            "cpu.performance.float.f64" | "cpu.performance.avx2.f64_fma"
                if config
                    .options
                    .get("fp64_preset")
                    .is_some_and(|s| s != "register") =>
            {
                Some(Self::Float(id.contains("avx2")))
            }
            "cpu.performance.extended.popcnt"
                if config
                    .options
                    .get("popcnt_preset")
                    .is_some_and(|s| s != "register") =>
            {
                Some(Self::Pop)
            }
            "cpu.performance.compression.deflate" => Some(Self::Codec(false)),
            "cpu.performance.decompression.deflate" => Some(Self::Codec(true)),
            "cpu.performance.prime.sieve" => Some(Self::Sieve),
            "cpu.performance.matrix.fp32.scaling"
                if config
                    .options
                    .get("matrix_shape")
                    .is_some_and(|s| s != "legacy") =>
            {
                Some(Self::Matrix)
            }
            _ => None,
        }
    }
    fn unit(self) -> u64 {
        match self {
            Self::Float(_) => 24,
            Self::Pop => 8,
            _ => 1,
        }
    }
    fn metric_unit(self) -> &'static str {
        match self {
            Self::Codec(_) => "bytes/s",
            Self::Sieve => "primes/s",
            _ => "operations/s",
        }
    }
    fn revision(self) -> &'static str {
        match self {
            Self::Float(false) => "fp64-scalar-buffered-1",
            Self::Float(true) => "fp64-avx2-fma-buffered-1",
            Self::Pop => "popcnt-data-1",
            Self::Codec(_) => "deflate-independent-blocks-1",
            Self::Sieve => "sieve-configurable-2",
            Self::Matrix => "fp32-global-shape-matrix-2",
        }
    }
}
struct Shared {
    matrix_b: Arc<Vec<f32>>,
    shape: [usize; 3],
    primes: u64,
}
fn error(code: &str, message: impl Into<String>) -> BenchmarkError {
    BenchmarkError::new(code, message)
}
fn check(token: &CancellationToken) -> Result<(), BenchmarkError> {
    if token.is_cancelled() {
        Err(error("cancelled", "CPU workload cancelled."))
    } else {
        Ok(())
    }
}
fn number(config: &BenchmarkConfig, key: &str, default: u64) -> u64 {
    config
        .options
        .get(key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
pub(super) fn handles(id: &str, config: &BenchmarkConfig) -> bool {
    Spec::from_id(id, config).is_some()
}
fn reference_primes(limit: usize, token: &CancellationToken) -> Result<u64, BenchmarkError> {
    if let Some(count) = match limit {
        10000 => Some(1229),
        100000 => Some(9592),
        1000000 => Some(78498),
        10000000 => Some(664579),
        100000000 => Some(5761455),
        _ => None,
    } {
        return Ok(count);
    }
    let mut odd = buffers::vector(limit.div_ceil(2), true, token)?;
    odd[0] = false;
    let mut p = 3;
    while p <= limit / p {
        check(token)?;
        if odd[p / 2] {
            for (step, index) in (p * p / 2..odd.len()).step_by(p).enumerate() {
                if step % 4096 == 0 {
                    check(token)?;
                }
                odd[index] = false;
            }
        }
        p += 2;
    }
    let mut count = 1;
    for chunk in odd.chunks(65536) {
        check(token)?;
        count += chunk.iter().filter(|v| **v).count() as u64;
    }
    Ok(count)
}
fn prepare(
    spec: Spec,
    config: &BenchmarkConfig,
    token: &CancellationToken,
) -> Result<Shared, BenchmarkError> {
    let _phase = gluj_bench_core::phase_timer("preparation-and-independent-reference");
    let shape = gluj_bench_core::cpu_matrix_shape(
        config
            .options
            .get("matrix_shape")
            .map(String::as_str)
            .unwrap_or("legacy"),
    )
    .unwrap_or([0, 0, 0]);
    let mut b = Vec::new();
    if matches!(spec, Spec::Matrix) {
        b = buffers::vector(shape[1] * shape[2], 0.0, token)?;
        for (i, v) in b.iter_mut().enumerate() {
            *v = ((i % 19) as f32 - 9.0) / 32.0;
        }
    }
    let primes = if matches!(spec, Spec::Sieve) {
        reference_primes(number(config, "sieve_limit", 1000000) as usize, token)?
    } else {
        0
    };
    Ok(Shared {
        matrix_b: Arc::new(b),
        shape,
        primes,
    })
}
struct Point {
    rates: Vec<f64>,
    traffic: Vec<f64>,
    elapsed: u64,
    source: u64,
    compressed: u64,
    allocated: u64,
    worker_buffers: Vec<u64>,
}
struct WorkerSample {
    work: u64,
    active_ns: u64,
    traffic: u64,
}
struct WorkerResult {
    samples: Vec<WorkerSample>,
    source_bytes: u64,
    compressed_bytes: u64,
    allocated_bytes: u64,
}
struct WorkerContext<'a> {
    spec: Spec,
    config: &'a BenchmarkConfig,
    payload: u64,
    locations: &'a [crate::topology::ProcessorLocation],
    shared: &'a Shared,
    token: &'a CancellationToken,
    recorder: Option<gluj_bench_core::ObservationRecorder>,
    access: gluj_bench_core::AccessPolicy,
    ensembles: u32,
    intensity: u32,
    barrier: Barrier,
    failed: AtomicBool,
}
impl WorkerContext<'_> {
    fn measure(
        &self,
        worker: usize,
        location: crate::topology::ProcessorLocation,
    ) -> Result<WorkerResult, BenchmarkError> {
        let prepared = (|| {
            let _phase = self.recorder.as_ref().map(|r| r.phase_timer("preparation"));
            let affinity = AffinityGuard::pin(location)?;
            let mut state = State::new(
                self.spec,
                self.config,
                worker,
                self.locations.len(),
                self.payload,
                self.shared,
                self.token,
            )?;
            let mut policy = self.access;
            policy.seed = gluj_bench_core::derive_seed(
                self.access.seed,
                0,
                self.payload,
                self.recorder
                    .as_ref()
                    .map(|r| r.block_index())
                    .unwrap_or(0)
                    .into(),
            );
            policy.block_bytes = if matches!(self.spec, Spec::Codec(_)) {
                64
            } else {
                (self.access.block_bytes / self.spec.unit() as usize / 64).max(1) * 64
            };
            let mut schedules = (0..self.ensembles)
                .map(|real| policy.schedule(state.span(), 64, worker, real, self.token))
                .collect::<Result<Vec<_>, _>>()?;
            state.prepare_reference(&mut schedules[0], self.token)?;
            Ok::<_, BenchmarkError>((affinity, state, schedules))
        })();
        if prepared.is_err() {
            self.failed.store(true, Ordering::SeqCst);
        }
        self.barrier.wait();
        if self.failed.load(Ordering::SeqCst) {
            return Err(prepared.err().unwrap_or_else(|| {
                error(
                    "worker_prepare_failed",
                    "Another worker could not prepare its input or processor placement.",
                )
            }));
        }
        let (_affinity, mut state, mut schedules) = prepared?;
        let mut pacer = crate::pacing::CpuPacer::new(self.intensity, self.token);
        let mut failure = None;
        {
            let _phase = self.recorder.as_ref().map(|r| r.phase_timer("warmup"));
            let started = Instant::now();
            while started.elapsed().as_nanos()
                < u128::from(gluj_bench_core::measurement_budget(self.config).warmup_ns)
            {
                let batch = Instant::now();
                if let Err(e) = state
                    .work(&mut schedules[0], self.token)
                    .and_then(|_| pacer.account(batch.elapsed()))
                {
                    failure = Some(e);
                    self.failed.store(true, Ordering::SeqCst);
                    break;
                }
            }
        }
        for schedule in &mut schedules {
            schedule.reset_pass(0);
        }
        self.barrier.wait();
        if self.failed.load(Ordering::SeqCst) {
            return Err(failure.unwrap_or_else(|| error("cancelled", "CPU warmup interrupted.")));
        }
        let mut samples = Vec::new();
        for sample in 0..self.config.samples * self.ensembles {
            let real = sample % self.ensembles;
            let schedule = &mut schedules[real as usize];
            let observation = if matches!(self.spec, Spec::Matrix | Spec::Sieve) {
                self.recorder.clone()
            } else {
                self.recorder.as_ref().map(|r|r.with_realization(self.access.derived_seed(worker,self.payload as usize,real),BTreeMap::from([
                    ("root_seed".into(),self.access.seed.to_string()),
                    ("worker".into(),worker.to_string()),
                    ("worker_count".into(),self.locations.len().to_string()),
                    ("processor_group".into(),location.group.to_string()),("processor_index".into(),location.index.to_string()),
                    ("schedule_span_units".into(),state.span().to_string()),
                    ("input_generator_revision".into(),self.spec.revision().into()),
                    ("dataset_bytes".into(),self.payload.to_string()),
                    ("realization".into(),real.to_string()),
                    ("schedule_auxiliary_bytes".into(),(schedule.auxiliary_bytes()*self.ensembles as usize).to_string()),
                    ("partition".into(),format!("worker {worker}/{}; contiguous complete blocks, global input indices",self.locations.len())),
                    ("expected_population_per_pass".into(),if matches!(self.spec,Spec::Pop) {state.expected_population().to_string()} else {"not_applicable".into()}),
                    (format!("block-{}-root",r.block_index()),gluj_bench_core::derive_seed(self.access.seed,0,self.payload,r.block_index().into()).to_string()),
                ])))
            };
            if sample < self.ensembles {
                let _phase = self.recorder.as_ref().map(|r| r.phase_timer("warmup"));
                if let Err(e) = state.work(schedule, self.token) {
                    failure = Some(e);
                    self.failed.store(true, Ordering::SeqCst);
                }
                schedule.reset_pass(0);
            }
            self.barrier.wait();
            if self.failed.load(Ordering::SeqCst) {
                return Err(failure
                    .unwrap_or_else(|| error("cancelled", "CPU workload warmup interrupted.")));
            }
            let started = Instant::now();
            let idle = pacer.idle();
            let mut work = 0u64;
            let mut traffic = 0u64;
            while pacer.active_elapsed(started, idle).as_nanos()
                < u128::from(gluj_bench_core::sample_target_ns(self.config))
            {
                let batch = Instant::now();
                match state.work(schedule, self.token) {
                    Ok((count, checksum)) => {
                        work = work.saturating_add(count);
                        traffic = traffic.saturating_add(state.work_bytes(count));
                        std::hint::black_box(checksum);
                    }
                    Err(e) => {
                        failure = Some(e);
                        self.failed.store(true, Ordering::SeqCst);
                        break;
                    }
                }
                if let Err(e) = pacer.account(batch.elapsed()) {
                    failure = Some(e);
                    self.failed.store(true, Ordering::SeqCst);
                    break;
                }
            }
            if work == 0 {
                failure = Some(error(
                    "invalid_work_count",
                    "CPU condition completed no useful work.",
                ));
                self.failed.store(true, Ordering::SeqCst);
            }
            let active_ns = pacer.active_elapsed(started, idle).as_nanos().max(1) as u64;
            self.barrier.wait();
            if self.failed.load(Ordering::SeqCst) {
                return Err(failure.unwrap_or_else(|| {
                    error(
                        "worker_failed",
                        "Another CPU worker interrupted the condition.",
                    )
                }));
            }
            if work == 0 {
                return Err(error(
                    "invalid_work_count",
                    "CPU condition completed no useful work.",
                ));
            }
            samples.push(WorkerSample {
                work,
                active_ns,
                traffic,
            });
            if let Some(recorder) = observation {
                recorder.record(
                    &format!(
                        "bytes-{}:worker-{worker}:group-{}:processor-{}",
                        self.payload, location.group, location.index
                    ),
                    "compute",
                    work as f64 * 1e9 / active_ns as f64,
                    self.spec.metric_unit(),
                    work,
                    match self.spec {
                        Spec::Codec(_) => "uncompressed_bytes",
                        Spec::Sieve => "primes",
                        _ => "operations",
                    },
                    active_ns,
                    started.elapsed().as_nanos() as u64,
                    "not_recorded",
                );
            }
        }
        {
            let _phase = self.recorder.as_ref().map(|r| r.phase_timer("validation"));
            state.validate()?;
        }
        let (source_bytes, compressed_bytes) = state.codec_sizes();
        Ok(WorkerResult {
            samples,
            source_bytes,
            compressed_bytes,
            allocated_bytes: state.allocated_buffers(),
        })
    }
}
fn point(
    spec: Spec,
    config: &BenchmarkConfig,
    payload: u64,
    locations: &[crate::topology::ProcessorLocation],
    shared: &Shared,
    token: &CancellationToken,
) -> Result<Point, BenchmarkError> {
    let access = gluj_bench_core::AccessPolicy::from_config(config)?;
    let ensembles = if matches!(spec, Spec::Sieve | Spec::Matrix) {
        1
    } else {
        access.realizations
    };
    let context = WorkerContext {
        spec,
        config,
        payload,
        locations,
        shared,
        token,
        recorder: gluj_bench_core::observation_recorder(),
        access,
        ensembles,
        intensity: gluj_bench_core::workload_percent(config, "cpu_activity_percent")?,
        barrier: Barrier::new(locations.len()),
        failed: AtomicBool::new(false),
    };
    let workers = thread::scope(|scope| {
        let handles = locations
            .iter()
            .copied()
            .enumerate()
            .map(|(worker, location)| {
                let context = &context;
                scope.spawn(move || context.measure(worker, location))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| error("worker_failed", "CPU workload thread panicked."))?
            })
            .collect::<Result<Vec<_>, BenchmarkError>>()
    })?;
    check(token)?;
    let mut rates = Vec::new();
    let mut traffic = Vec::new();
    let mut elapsed = 0;
    for sample in 0..config.samples * ensembles {
        let active = workers
            .iter()
            .map(|w| w.samples[sample as usize].active_ns)
            .max()
            .unwrap_or(1);
        let work = workers
            .iter()
            .map(|w| w.samples[sample as usize].work)
            .sum::<u64>();
        let bytes = workers
            .iter()
            .map(|w| w.samples[sample as usize].traffic)
            .sum::<u64>();
        rates.push(work as f64 * 1e9 / active as f64);
        traffic.push(bytes as f64 * 1e9 / active as f64);
        elapsed += active;
    }
    Ok(Point {
        rates,
        traffic,
        elapsed,
        source: workers.iter().map(|w| w.source_bytes).sum(),
        compressed: workers.iter().map(|w| w.compressed_bytes).sum(),
        worker_buffers: workers.iter().map(|w| w.allocated_bytes).collect(),
        allocated: workers.iter().map(|w| w.allocated_bytes).sum::<u64>()
            + if matches!(spec, Spec::Matrix) {
                shared.matrix_b.capacity() as u64 * 4
            } else {
                0
            }
            + if matches!(spec, Spec::Matrix | Spec::Sieve) {
                locations.len() as u64 * 8
            } else {
                0
            },
    })
}

fn metric(name: String, values: &[f64], unit: &str) -> Metric {
    let statistics = crate::kernels::statistics(values);
    Metric {
        name,
        value: statistics.median,
        unit: unit.into(),
        statistics,
    }
}
fn sizes(
    spec: Spec,
    config: &BenchmarkConfig,
    locations: &[crate::topology::ProcessorLocation],
    budget: u64,
    shape: [usize; 3],
) -> Result<Vec<u64>, BenchmarkError> {
    match spec {
        Spec::Codec(_) => Ok(vec![number(config, "codec_corpus_bytes", 16777216)]),
        Spec::Sieve => Ok(vec![number(config, "sieve_limit", 1000000) + 1]),
        Spec::Matrix => Ok(vec![
            4 * (shape[0] * shape[2] + shape[2] * shape[1] + shape[0] * shape[1]) as u64,
        ]),
        _ => {
            let access = gluj_bench_core::AccessPolicy::from_config(config)?;
            let limit = access.payload_budget(budget, locations.len(), spec.unit());
            let preset = config
                .options
                .get(gluj_bench_core::cpu_data_preset_key(
                    if matches!(spec, Spec::Pop) {
                        "cpu.performance.extended.popcnt"
                    } else {
                        "cpu.performance.float.f64"
                    },
                ))
                .map(String::as_str)
                .unwrap_or("register");
            let alignment = 64 * spec.unit();
            let requested = match preset {
                "l1" | "l2" | "l3" => 0,
                "ram" => limit,
                _ => number(config, "dataset_bytes", 256 * 1024),
            };
            let requested = if requested == 0 {
                return Err(error(
                    "invalid_config",
                    "Cache targets must be resolved before measuring the CPU workload.",
                ));
            } else {
                requested
            };
            let actual = requested / alignment * alignment;
            if actual < alignment * locations.len() as u64 || requested > limit {
                return Err(error(
                    "allocation_limit",
                    "Dataset must fit its allocation budget and contain one complete kernel block per worker.",
                ));
            }
            let sweep = config
                .options
                .get("dataset_mode")
                .is_some_and(|s| s == "sweep");
            let mut sizes = Vec::new();
            if sweep {
                let mut size = alignment * locations.len() as u64;
                while size < actual {
                    sizes.push(size);
                    size = size.saturating_mul(2);
                }
            }
            sizes.push(actual);
            Ok(sizes)
        }
    }
}
pub(super) fn run(
    id: &str,
    config: &BenchmarkConfig,
    token: &CancellationToken,
    progress: &mut ProgressCallback<'_>,
    device: &DeviceDescriptor,
    topology: &CpuTopology,
) -> Result<BenchmarkResult, BenchmarkError> {
    let spec = Spec::from_id(id, config).ok_or_else(|| {
        error(
            "unsupported_benchmark",
            "No configurable CPU runner matches this mode.",
        )
    })?;
    gluj_bench_core::validate_cpu_workload_options(id, config)?;
    #[cfg(target_arch = "x86_64")]
    let supported = match spec {
        Spec::Float(true) | Spec::Matrix => {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        }
        Spec::Pop => std::is_x86_feature_detected!("popcnt"),
        _ => true,
    };
    #[cfg(not(target_arch = "x86_64"))]
    let supported = !matches!(spec, Spec::Float(true) | Spec::Matrix | Spec::Pop);
    if !supported {
        return Err(error(
            "benchmark_unavailable",
            "The selected CPU instruction path is not available.",
        ));
    }
    let locations = crate::topology::selected_workers(topology, config, false)?;
    let workers = locations.len();
    if workers == 0 {
        return Err(error(
            "topology_unavailable",
            "No selected CPU workers are available.",
        ));
    }
    let mut memory = sysinfo::System::new();
    memory.refresh_memory();
    let budget = gluj_bench_core::ram_allocation_budget(
        memory.total_memory(),
        memory.available_memory(),
        number(config, "ram_budget_percent", 20) as u32,
    );
    let mut actual = config.clone();
    if matches!(spec, Spec::Float(_) | Spec::Pop) {
        let key = gluj_bench_core::cpu_data_preset_key(id);
        if let Some(preset) = config.options.get(key)
            && matches!(preset.as_str(), "l1" | "l2" | "l3")
        {
            let bytes = crate::compute_scaling::cache_dataset_aligned(
                topology,
                &locations,
                preset.as_bytes()[1] - b'0',
                64 * spec.unit(),
            )?;
            actual.options.insert(key.into(), "custom".into());
            actual
                .options
                .insert("dataset_bytes".into(), bytes.to_string());
        }
    }
    let planned = match spec {
        Spec::Codec(_) => number(config, "codec_corpus_bytes", 16777216)
            .saturating_mul(4)
            .saturating_add(workers as u64 * 10 * 1024 * 1024),
        Spec::Sieve => {
            (number(config, "sieve_limit", 1000000) + 1).saturating_mul(workers as u64 + 1)
        }
        Spec::Matrix => {
            let shape = gluj_bench_core::cpu_matrix_shape(
                config
                    .options
                    .get("matrix_shape")
                    .map(String::as_str)
                    .unwrap_or("legacy"),
            )
            .unwrap();
            4 * (shape[0] * shape[2] + shape[2] * shape[1] + shape[0] * shape[1]) as u64
        }
        _ => 0,
    };
    if planned > budget {
        return Err(error(
            "allocation_limit",
            "The complete CPU workload and its auxiliary buffers exceed the available-memory budget.",
        ));
    }
    if matches!(spec, Spec::Codec(_))
        && workers as u64
            > number(config, "codec_corpus_bytes", 16777216).div_ceil(number(
                config,
                "codec_block_bytes",
                65536,
            ))
    {
        return Err(error(
            "mode_unavailable",
            "Choose at least one independent codec block per worker.",
        ));
    }
    if matches!(spec, Spec::Matrix)
        && workers
            > gluj_bench_core::cpu_matrix_shape(
                config
                    .options
                    .get("matrix_shape")
                    .map(String::as_str)
                    .unwrap_or("legacy"),
            )
            .unwrap()[0]
                / 8
    {
        return Err(error(
            "mode_unavailable",
            "This global matrix shape needs at least eight rows per worker.",
        ));
    }
    let shared = prepare(spec, config, token)?;
    let points = sizes(spec, &actual, &locations, budget, shared.shape)?;
    let mut values: BTreeMap<u64, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut elapsed = 0;
    let mut ratio = None;
    let mut allocated_peak = 0;
    let mut worker_buffers = Vec::new();
    let visits = gluj_bench_core::condition_visits(&points, config);
    for (index, (payload, block_config, block)) in visits.iter().enumerate() {
        check(token)?;
        progress(ProgressUpdate {
            fraction: index as f64 / visits.len() as f64,
            phase: "cpu_workload".into(),
            message: format!(
                "{}: {} bytes, measurement block {block}",
                spec.revision(),
                payload
            ),
        });
        let _scope = gluj_bench_core::MeasurementScope::block(block_config, id, *block);
        let measured = point(spec, block_config, *payload, &locations, &shared, token)?;
        elapsed += measured.elapsed;
        allocated_peak = allocated_peak.max(measured.allocated);
        worker_buffers = measured.worker_buffers;
        if measured.source > 0 {
            ratio = Some(measured.compressed as f64 / measured.source as f64);
        }
        let entry = values.entry(*payload).or_default();
        entry.0.extend(measured.rates);
        entry.1.extend(measured.traffic);
    }
    let profile = matches!(spec, Spec::Float(_) | Spec::Pop | Spec::Matrix);
    let mut metrics = Vec::new();
    for (payload, (rates, traffic)) in &values {
        metrics.push(metric(
            if profile {
                format!("working_set_{payload}.compute")
            } else {
                "throughput".into()
            },
            rates,
            spec.metric_unit(),
        ));
        if profile {
            metrics.push(metric(
                format!("working_set_{payload}.bandwidth"),
                traffic,
                "bytes/s",
            ));
        }
    }
    if number(config, "sustained_seconds", 0) > 0 {
        for (payload, (rates, _)) in &values {
            let width = (rates.len() / 3).max(1);
            let prefix = if profile {
                format!("working_set_{payload}.")
            } else {
                String::new()
            };
            metrics.push(metric(
                format!("{prefix}early_throughput"),
                &rates[..width],
                spec.metric_unit(),
            ));
            metrics.push(metric(
                format!("{prefix}late_throughput"),
                &rates[rates.len() - width..],
                spec.metric_unit(),
            ));
        }
    }
    if let Some(ratio) = ratio {
        metrics.push(metric("compression_ratio".into(), &[ratio], "ratio"));
    }
    let payload = *points.last().unwrap();
    let mut metadata=BTreeMap::from([
        ("kernel_revision".into(),spec.revision().into()),("harness_revision".into(),"cpu-configurable-1".into()),("result_validation".into(),"independent output/round-trip/prime-count references; preparation and final validation outside timing".into()),
        ("thread_count".into(),workers.to_string()),("thread_mode".into(),config.options.get("thread_mode").cloned().unwrap_or_else(||"physical_cores".into())),("cpu_core_limit".into(),workers.to_string()),
        ("requested_dataset_bytes".into(),config.options.get("dataset_bytes").cloned().unwrap_or_else(||payload.to_string())),("resolved_dataset_bytes".into(),payload.to_string()),
        ("dataset_mode".into(),if points.len()>1 {"sweep"} else {"single"}.into()),("allocation_budget_bytes".into(),budget.to_string()),
        ("allocated_test_buffer_bytes".into(),allocated_peak.to_string()),("core_experiment_work".into(),if matches!(spec,Spec::Sieve) {"one complete sieve per worker; aggregate work changes with worker count"} else {"fixed aggregate input/shape and useful work per full pass; partition/placement changes with workers"}.into()),
        ("byte_definition".into(),if matches!(spec,Spec::Matrix) {"logical blocked-kernel A/B reads and C loads/stores including reuse; not bus traffic"} else {"all scheduled payload visits; effective bytes, not bus traffic"}.into()),
        ("profile_sample_count_per_tier".into(),(config.samples*number(config,"access_realizations",1) as u32).to_string()),
    ]);
    if matches!(spec, Spec::Float(_) | Spec::Pop) {
        metadata.extend(gluj_bench_core::AccessPolicy::from_config(config)?.metadata());
        metadata.insert(
            gluj_bench_core::cpu_data_preset_key(id).into(),
            config.options[gluj_bench_core::cpu_data_preset_key(id)].clone(),
        );
        metadata.insert(
            "data_type".into(),
            if matches!(spec, Spec::Pop) {
                "u64"
            } else {
                "fp64"
            }
            .into(),
        );
        metadata.insert("operation_definition".into(),match spec {Spec::Float(false)=>"one scalar non-fused FP64 multiply or add; 16 pairs per value; compiler barriers included",Spec::Float(true)=>"one FP64 multiply or add; AVX2 FMA counts as two; 16 FMAs per value",_=>"one hardware POPCNT per 64-bit input; load and reduction overhead included"}.into());
    }
    if matches!(spec, Spec::Codec(_)) {
        for key in [
            "codec_corpus",
            "codec_corpus_bytes",
            "codec_block_bytes",
            "codec_level",
            "access_pattern",
            "access_seed",
            "access_realizations",
        ] {
            if let Some(v) = config.options.get(key) {
                metadata.insert(key.into(), v.clone());
            }
        }
        metadata.insert("codec_state_policy".into(),"independent raw DEFLATE streams; codec reset and output clear per block are timed; buffers reused".into());
        metadata.insert(
            "compression_ratio_definition".into(),
            "compressed bytes / uncompressed corpus bytes; actual completed independent blocks"
                .into(),
        );
    }
    if matches!(spec, Spec::Sieve) {
        metadata.insert(
            "sieve_limit".into(),
            number(config, "sieve_limit", 1000000).to_string(),
        );
        metadata.insert("known_primes_per_sieve".into(), shared.primes.to_string());
        metadata.insert(
            "operation_definition".into(),
            "prime found during complete reset, marking and count over 2..=recorded limit".into(),
        );
    }
    if matches!(spec, Spec::Matrix) {
        let [m, n, k] = shared.shape;
        for (key, value) in [("matrix_m", m), ("matrix_n", n), ("matrix_k", k)] {
            metadata.insert(key.into(), value.to_string());
        }
        for key in ["matrix_shape", "matrix_reuse", "matrix_k_block"] {
            metadata.insert(key.into(), config.options[key].clone());
        }
        metadata.insert("matrix_shared_b_bytes".into(), (n * k * 4).to_string());
        metadata.insert("matrix_shape_definition".into(),"global M,N,K fixed across worker counts; disjoint 8-row output shards, shared immutable B; each reuse overwrites a full product".into());
    }
    let blocks = match spec {
        Spec::Float(_) | Spec::Pop => payload / spec.unit() / 64,
        Spec::Codec(_) => number(config, "codec_corpus_bytes", 16777216).div_ceil(number(
            config,
            "codec_block_bytes",
            65536,
        )),
        Spec::Matrix => shared.shape[0] as u64 / 8,
        Spec::Sieve => 1,
    };
    let worker_layout:Vec<_>=locations.iter().enumerate().map(|(worker,location)| {
        let first=blocks*worker as u64/workers as u64;let end=blocks*(worker as u64+1)/workers as u64;
        let (offset,count,unit)=match spec {Spec::Float(_)|Spec::Pop=>(first*64,(end-first)*64,"numeric_elements"),Spec::Matrix=>(first*8,(end-first)*8,"output_rows"),Spec::Codec(_)=>(first,end-first,"independent_streams"),Spec::Sieve=>(0,number(config,"sieve_limit",1000000)+1,"complete_sieve_marks")};
        serde_json::json!({"worker":worker,"processor_group":location.group,"processor_index":location.index,"input_offset":offset,"input_count":count,"input_unit":unit,"allocated_vector_bytes":worker_buffers.get(worker),"shared_immutable_b_bytes":if matches!(spec,Spec::Matrix) {shared.matrix_b.capacity()*4} else {0}})
    }).collect();
    metadata.insert(
        "worker_layout".into(),
        serde_json::to_string(&worker_layout).expect("worker layout"),
    );
    metadata.insert("input_generator_revision".into(), spec.revision().into());
    metadata.insert("aggregate_rate_definition".into(),"sum of completed worker work / slowest worker active duration for each synchronized observation; preparation, warmup and barrier waits excluded".into());
    metadata.insert("allocation_reserve_bytes".into(), planned.to_string());
    if matches!(spec, Spec::Codec(_)) {
        metadata.insert("codec_internal_allocation".into(),"library-managed state excluded from enumerated test-buffer bytes; budget reserves 10 MiB per worker".into());
    }
    metadata.insert(
        "core_experiment".into(),
        config
            .options
            .get("core_experiment")
            .cloned()
            .unwrap_or_else(|| "none".into()),
    );
    Ok(BenchmarkResult {
        benchmark_id: id.into(),
        device_id: device.id.clone(),
        elapsed_ns: elapsed,
        metrics,
        workload_metadata: metadata,
        device_metadata: BTreeMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sieve_reference_and_complete_work_agree_for_custom_limits_and_cancel() {
        let token = CancellationToken::default();
        for limit in [10000usize, 10013, 12345] {
            let expected = (2..=limit)
                .filter(|n| {
                    let mut d = 2;
                    while d <= n / d {
                        if n % d == 0 {
                            return false;
                        }
                        d += 1;
                    }
                    true
                })
                .count() as u64;
            assert_eq!(reference_primes(limit, &token).unwrap(), expected);
            let c = BenchmarkConfig {
                options: BTreeMap::from([("sieve_limit".into(), limit.to_string())]),
                ..Default::default()
            };
            let shared = prepare(Spec::Sieve, &c, &token).unwrap();
            let mut state =
                State::new(Spec::Sieve, &c, 0, 1, limit as u64 + 1, &shared, &token).unwrap();
            let mut schedule = gluj_bench_core::AccessPolicy::from_config(&c)
                .unwrap()
                .schedule(64, 64, 0, 0, &token)
                .unwrap();
            assert_eq!(state.work(&mut schedule, &token).unwrap().0, expected);
            state.validate().unwrap();
        }
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(reference_primes(12345, &cancelled).is_err());
    }
}
