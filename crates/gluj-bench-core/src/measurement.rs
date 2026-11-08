//! Bounded descriptive measurement, without performance-dependent stopping or discarded outliers.
use crate::*;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementBudget {
    pub minimum_samples: u32,
    pub minimum_sample_ns: u64,
    pub warmup_ns: u64,
    pub maximum_batches: u32,
}
pub fn measurement_budget(config: &BenchmarkConfig) -> MeasurementBudget {
    let shorter = config
        .options
        .get("measurement_policy")
        .is_some_and(|v| v == "shorter");
    let reference = config.options.contains_key("measurement_role");
    MeasurementBudget {
        minimum_samples: if reference || shorter { 3 } else { 6 },
        minimum_sample_ns: if reference || shorter {
            100_000_000
        } else {
            250_000_000
        },
        warmup_ns: if shorter { 100_000_000 } else { 250_000_000 },
        maximum_batches: 1024,
    }
}
pub fn resolve_measurement_config(
    config: &BenchmarkConfig,
) -> Result<BenchmarkConfig, BenchmarkError> {
    let name = config
        .options
        .get("measurement_policy")
        .map(String::as_str)
        .unwrap_or("extensive");
    if !["extensive", "shorter"].contains(&name)
        || config.samples == 0
        || config.samples > 64
        || config.target_duration_ms == 0
        || config.target_duration_ms > 120_000
    {
        return Err(BenchmarkError::new(
            "invalid_config",
            "Choose extensive or shorter measurement, 1–64 samples and a positive duration up to 120 seconds per condition.",
        ));
    }
    let sustained = config
        .options
        .get("sustained_seconds")
        .map(|s| s.parse::<u64>())
        .transpose()
        .map_err(|_| {
            BenchmarkError::new(
                "invalid_config",
                "Sustained duration must be a whole number of seconds.",
            )
        })?
        .unwrap_or(0);
    let sustained = if config.options.contains_key("measurement_role") {
        0
    } else {
        sustained
    };
    if sustained > 120 {
        return Err(BenchmarkError::new(
            "invalid_config",
            "Sustained coverage is bounded to 120 seconds per realization/condition.",
        ));
    }
    let budget = measurement_budget(config);
    let mut resolved = config.clone();
    resolved.samples = config.samples.max(budget.minimum_samples);
    resolved.target_duration_ms = config
        .target_duration_ms
        .max(budget.minimum_sample_ns / 1_000_000 * u64::from(resolved.samples))
        .max(sustained * 1000);
    resolved
        .options
        .insert("measurement_policy".into(), name.into());
    Ok(resolved)
}
pub fn sample_target_ns(config: &BenchmarkConfig) -> u64 {
    (config.target_duration_ms.saturating_mul(1_000_000) / u64::from(config.samples.max(1)))
        .max(measurement_budget(config).minimum_sample_ns)
}
pub fn reference_config(config: &BenchmarkConfig, role: &str) -> BenchmarkConfig {
    let mut reference = config.clone();
    reference
        .options
        .insert("measurement_role".into(), role.into());
    let budget = measurement_budget(&reference);
    reference.samples = budget.minimum_samples;
    reference.target_duration_ms =
        budget.minimum_sample_ns / 1_000_000 * u64::from(reference.samples);
    reference
}

/// Extensions repeat bounded batches until the duration floor is covered. Variation
/// and observed speed never decide whether an observation is retained or extended.
#[derive(Debug)]
pub struct SampleWindow {
    target_ns: u64,
    active_ns: u64,
    batches: u32,
    maximum_batches: u32,
}
impl SampleWindow {
    pub fn new(target_ns: u64) -> Self {
        Self {
            target_ns: target_ns.max(1),
            active_ns: 0,
            batches: 0,
            maximum_batches: 1024,
        }
    }
    pub fn needs_batch(&self) -> bool {
        self.active_ns < self.target_ns && self.batches < self.maximum_batches
    }
    pub fn add(&mut self, active_ns: u64) -> Result<(), BenchmarkError> {
        if active_ns == 0 {
            return Err(BenchmarkError::new(
                "invalid_timing",
                "A measurement batch reported no active time.",
            ));
        }
        self.active_ns = self.active_ns.saturating_add(active_ns);
        self.batches += 1;
        Ok(())
    }
    pub fn finish(&self) -> Result<u64, BenchmarkError> {
        if self.active_ns < self.target_ns {
            Err(BenchmarkError::new(
                "measurement_budget_exhausted",
                "The bounded batch limit was reached before the requested active duration. Partial observations are retained.",
            ))
        } else {
            Ok(self.active_ns)
        }
    }
    pub fn batches(&self) -> u32 {
        self.batches
    }
}

/// Alternate forward/reverse traversals. Each call prepares its documented state;
/// no randomized workload or seed ensemble is implied by measurement order.
pub fn measurement_order<T: Clone>(values: &[T]) -> Vec<T> {
    CONTEXT.with(|context| {
        let context = context.borrow();
        let reverse = context.as_ref().is_some_and(|c| c.pass % 2 == 1);
        if reverse {
            values.iter().rev().cloned().collect()
        } else {
            values.to_vec()
        }
    })
}
pub fn condition_visits<T: Clone>(
    values: &[T],
    config: &BenchmarkConfig,
) -> Vec<(T, BenchmarkConfig, u32)> {
    let blocks = if config.samples >= 6
        && !config
            .options
            .get("sustained_seconds")
            .is_some_and(|s| s != "0")
    {
        3
    } else {
        1
    };
    let mut visits = Vec::new();
    for block in 0..blocks {
        let mut part = config.clone();
        part.samples = config.samples / blocks + u32::from(block < config.samples % blocks);
        part.target_duration_ms = sample_target_ns(config)
            .div_ceil(1_000_000)
            .saturating_mul(u64::from(part.samples));
        let ordered: Vec<_> = if block % 2 == 1 {
            values.iter().rev().cloned().collect()
        } else {
            values.to_vec()
        };
        visits.extend(
            ordered
                .into_iter()
                .map(|value| (value, part.clone(), block)),
        );
    }
    visits
}

#[derive(Clone)]
struct Context {
    realization_id: String,
    recorder: Recorder,
    scope: String,
    role: String,
    pass: u32,
    config: BenchmarkConfig,
    gpu: bool,
}
thread_local! { static CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) }; }
#[derive(Clone)]
pub struct Recorder {
    inner: Arc<Mutex<Capture>>,
    origin: Instant,
    progress: Option<std::sync::mpsc::Sender<ProgressUpdate>>,
    minimum_samples: u32,
}
#[derive(Default)]
struct Capture {
    samples: Vec<RawSample>,
    error: Option<BenchmarkError>,
    counts: BTreeMap<(String, String), u32>,
    last_progress: Option<Instant>,
    phases: Vec<PhaseTiming>,
    realizations: BTreeMap<String, WorkloadRealization>,
}

pub struct MeasurementScope {
    previous: Option<Context>,
    _timer: Option<PhaseTimer>,
}
impl MeasurementScope {
    /// Associate this thread's observations with one reproducible workload realization.
    pub fn realization(seed: u64, parameters: BTreeMap<String, String>) -> Self {
        let (config, scope) = CONTEXT.with(|c| {
            c.borrow()
                .as_ref()
                .map(|c| (c.config.clone(), c.scope.clone()))
                .unwrap_or_default()
        });
        let guard = Self::enter(&config, scope);
        if let Some(recorder) = observation_recorder() {
            let recorder = recorder.with_realization(seed, parameters);
            CONTEXT.with(|c| {
                if let Some(c) = c.borrow_mut().as_mut() {
                    c.realization_id = recorder.realization_id;
                }
            });
        }
        guard
    }
    pub fn condition(label: &str) -> Self {
        let (config, scope) = CONTEXT.with(|context| {
            context
                .borrow()
                .as_ref()
                .map(|c| (c.config.clone(), format!("{}:{label}", c.scope)))
                .unwrap_or_else(|| (BenchmarkConfig::default(), label.into()))
        });
        Self::enter(&config, scope)
    }
    pub fn block(config: &BenchmarkConfig, scope: impl Into<String>, block: u32) -> Self {
        let guard = Self::enter(config, scope);
        CONTEXT.with(|context| {
            if let Some(context) = context.borrow_mut().as_mut() {
                context.pass = block;
            }
        });
        guard
    }
    pub fn enter(config: &BenchmarkConfig, scope: impl Into<String>) -> Self {
        let previous = CONTEXT.with(|context| {
            let previous = context.borrow().clone();
            if let Some(mut next) = previous.clone() {
                next.scope = scope.into();
                next.config = config.clone();
                if let Some(role) = config.options.get("measurement_role") {
                    next.role = role.clone();
                }
                *context.borrow_mut() = Some(next);
            }
            previous
        });
        let timer = config
            .options
            .get("measurement_role")
            .map(|role| phase_timer(role));
        Self {
            previous,
            _timer: timer,
        }
    }
}
impl Drop for MeasurementScope {
    fn drop(&mut self) {
        CONTEXT.with(|context| *context.borrow_mut() = self.previous.take());
    }
}
pub fn current_reference_budget() -> MeasurementBudget {
    CONTEXT.with(|context| {
        let config = context
            .borrow()
            .as_ref()
            .map(|c| c.config.clone())
            .unwrap_or_default();
        measurement_budget(&reference_config(&config, "calibration"))
    })
}
pub fn current_measurement_budget() -> MeasurementBudget {
    CONTEXT.with(|context| {
        measurement_budget(
            &context
                .borrow()
                .as_ref()
                .map(|c| c.config.clone())
                .unwrap_or_default(),
        )
    })
}
impl MeasurementScope {
    pub fn phase(role: &str) -> Self {
        let config = CONTEXT.with(|context| {
            context
                .borrow()
                .as_ref()
                .map(|c| c.config.clone())
                .unwrap_or_default()
        });
        let config = reference_config(&config, role);
        let scope = CONTEXT.with(|context| {
            context
                .borrow()
                .as_ref()
                .map(|c| c.scope.clone())
                .unwrap_or_else(|| "calibration".into())
        });
        Self::enter(&config, scope)
    }
}

/// Shared across CPU workers; each raw record remains a batch, not individual loop iterations.
#[derive(Clone)]
pub struct ObservationRecorder {
    recorder: Recorder,
    scope: String,
    role: String,
    gpu: bool,
    pass: u32,
    minimum_samples: u32,
    realization_id: String,
}

pub struct PhaseTimer {
    recorder: Option<ObservationRecorder>,
    phase: String,
    offset_ns: u64,
}
pub fn phase_timer(phase: &str) -> PhaseTimer {
    match observation_recorder() {
        Some(recorder) => recorder.phase_timer(phase),
        None => PhaseTimer {
            recorder: None,
            phase: phase.into(),
            offset_ns: 0,
        },
    }
}
impl ObservationRecorder {
    pub fn phase_timer(&self, phase: &str) -> PhaseTimer {
        PhaseTimer {
            recorder: Some(self.clone()),
            phase: phase.into(),
            offset_ns: self
                .recorder
                .origin
                .elapsed()
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64,
        }
    }
}
impl Drop for PhaseTimer {
    fn drop(&mut self) {
        if let Some(recorder) = &self.recorder {
            // Both endpoints use the same recorder origin so nested inclusive intervals remain ordered.
            let elapsed_ns = (recorder
                .recorder
                .origin
                .elapsed()
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64)
                .saturating_sub(self.offset_ns);
            recorder
                .recorder
                .inner
                .lock()
                .expect("phase recorder")
                .phases
                .push(PhaseTiming {
                    phase: self.phase.clone(),
                    condition: recorder.scope.clone(),
                    offset_ns: self.offset_ns,
                    elapsed_ns,
                    inclusive: true,
                });
        }
    }
}
pub fn observation_recorder() -> Option<ObservationRecorder> {
    CONTEXT.with(|context| {
        context.borrow().as_ref().map(|c| ObservationRecorder {
            recorder: c.recorder.clone(),
            scope: c.scope.clone(),
            role: c.role.clone(),
            gpu: c.gpu,
            pass: c.pass,
            realization_id: c.realization_id.clone(),
            minimum_samples: if c.role == "measurement" {
                c.recorder.minimum_samples
            } else {
                c.config.samples
            },
        })
    })
}
impl ObservationRecorder {
    pub fn block_index(&self) -> u32 {
        self.pass
    }
    pub fn with_realization(&self, seed: u64, parameters: BTreeMap<String, String>) -> Self {
        let id = format!("seed-{seed:016x}");
        self.recorder
            .inner
            .lock()
            .expect("sample recorder")
            .realizations
            .entry(id.clone())
            .or_insert(WorkloadRealization {
                id: id.clone(),
                seed: Some(seed),
                generator_revision: Some(
                    parameters
                        .get("generator_revision")
                        .cloned()
                        .unwrap_or_else(|| crate::ACCESS_REVISION.into()),
                ),
                parameters: BTreeMap::new(),
            })
            .parameters
            .extend(parameters);
        let mut next = self.clone();
        next.realization_id = id;
        next
    }
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        condition: &str,
        metric: &str,
        value: f64,
        unit: &str,
        useful_work: u64,
        work_unit: &str,
        active_ns: u64,
        elapsed_ns: u64,
        validation: &str,
    ) {
        let mut capture = self.recorder.inner.lock().expect("sample recorder");
        if !value.is_finite() {
            capture.error = Some(BenchmarkError::new(
                "invalid_observation",
                "A measured observation was non-finite; no successful score is produced.",
            ));
            return;
        }
        let sequence = capture.samples.len() as u64;
        capture.samples.push(RawSample {
            version: CONTRACT_VERSION,
            sequence,
            timing_scope: if metric == "completed_compute_over_observation_window"
                || metric.starts_with("offered_")
            {
                SampleTimingScope::OfferedWindowWithAggregateService
            } else if metric == "host_transfer" {
                SampleTimingScope::HostCopyElapsed
            } else if self.gpu {
                SampleTimingScope::GpuTimestampAndHostElapsed
            } else {
                SampleTimingScope::ActiveExecution
            },
            block_id: format!(
                "{}:{}:block-{}:{}",
                self.role, self.scope, self.pass, self.realization_id
            ),
            realization_id: self.realization_id.clone(),
            condition_id: format!("{}:{condition}", self.scope),
            metric: metric.into(),
            value,
            unit: unit.into(),
            useful_work,
            useful_work_unit: work_unit.into(),
            active_ns,
            elapsed_ns,
            offset_ns: self
                .recorder
                .origin
                .elapsed()
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64,
            reference_anchor: None,
            validation: ValidationStatus {
                status: validation.into(),
                scope: "per-kernel validation scope recorded in result metadata".into(),
            },
            status: CompletionStatus::Completed,
        });
        let key = (format!("{}:{condition}", self.scope), metric.to_owned());
        let count = capture.counts.entry(key).or_default();
        *count += 1;
        let count = *count;
        if capture
            .last_progress
            .is_none_or(|instant| instant.elapsed() >= Duration::from_millis(200))
        {
            capture.last_progress = Some(Instant::now());
            if let Some(sender) = &self.recorder.progress {
                let phase = if self.role.starts_with("calibration") {
                    "Calibration"
                } else if self.role.starts_with("reference") {
                    "Reference"
                } else {
                    "Measurement"
                };
                let _=sender.send(ProgressUpdate { fraction:0.0,phase:"sample_coverage".into(),message:format!("{phase}: {count} completed observations for the active condition (minimum {}).",self.minimum_samples) });
            }
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub fn record_observation(
    condition: &str,
    metric: &str,
    value: f64,
    unit: &str,
    useful_work: u64,
    work_unit: &str,
    active_ns: u64,
    elapsed_ns: u64,
) {
    if let Some(recorder) = observation_recorder() {
        recorder.record(
            condition,
            metric,
            value,
            unit,
            useful_work,
            work_unit,
            active_ns,
            elapsed_ns,
            "not_recorded",
        );
    }
}

pub fn capture_measurement<F>(id: &str, config: &BenchmarkConfig, run: F) -> ProviderMeasurement
where
    F: FnOnce() -> Result<BenchmarkResult, BenchmarkError>,
{
    capture_measurement_with_progress(id, config, None, run)
}

pub fn capture_measurement_with_progress<F>(
    id: &str,
    config: &BenchmarkConfig,
    progress: Option<std::sync::mpsc::Sender<ProgressUpdate>>,
    run: F,
) -> ProviderMeasurement
where
    F: FnOnce() -> Result<BenchmarkResult, BenchmarkError>,
{
    let recorder = Recorder {
        inner: Arc::new(Mutex::new(Capture::default())),
        origin: Instant::now(),
        progress,
        minimum_samples: config.samples,
    };
    let previous = CONTEXT.with(|context| {
        context.replace(Some(Context {
            realization_id: "fixed-workload".into(),
            recorder: recorder.clone(),
            scope: id.into(),
            role: "measurement".into(),
            pass: 0,
            config: config.clone(),
            gpu: id.starts_with("gpu."),
        }))
    });
    let _restore = MeasurementScope {
        previous,
        _timer: None,
    };
    let mut result = run();
    let mut capture = recorder.inner.lock().expect("sample recorder");
    let mut samples = std::mem::take(&mut capture.samples);
    let mut phase_timings = std::mem::take(&mut capture.phases);
    let mut realizations = std::mem::take(&mut capture.realizations);
    phase_timings.sort_by_key(|phase| phase.offset_ns);
    if let Some(error) = capture.error.take() {
        result = Err(error);
    }
    drop(capture);
    if result.is_ok() && samples.is_empty() {
        result = Err(BenchmarkError::new(
            "measurement_unavailable",
            "This runner has not supplied the required raw observations.",
        ));
    }
    if result.is_ok() {
        let mut coverage: BTreeMap<(&str, &str, &str), (u32, u128)> = BTreeMap::new();
        for sample in &samples {
            if sample.block_id.starts_with("measurement:")
                && !matches!(
                    sample.metric.as_str(),
                    "offered_mean_completion_delay"
                        | "offered_queue_wait"
                        | "offered_demand_fulfillment"
                )
            {
                let entry = coverage
                    .entry((&sample.condition_id, &sample.metric, &sample.realization_id))
                    .or_default();
                entry.0 += 1;
                entry.1 += u128::from(
                    if sample.timing_scope == SampleTimingScope::OfferedWindowWithAggregateService {
                        sample.elapsed_ns
                    } else {
                        sample.active_ns
                    },
                );
            }
        }
        let expected_realizations = config
            .options
            .get("access_realizations")
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or(1);
        let mut realization_coverage: BTreeMap<(&str, &str), std::collections::BTreeSet<&str>> =
            BTreeMap::new();
        for (condition, metric, realization) in coverage.keys() {
            realization_coverage
                .entry((condition, metric))
                .or_default()
                .insert(realization);
        }
        let missing_realization = realization_coverage
            .values()
            .any(|ids| ids.len() != expected_realizations);
        if coverage.is_empty()
            || missing_realization
            || coverage.values().any(|(count, active)| {
                *count < config.samples
                    || *active + u128::from(*count)
                        < u128::from(sample_target_ns(config)) * u128::from(config.samples)
            })
        {
            result = Err(BenchmarkError::new(
                "incomplete_measurement_coverage",
                "A condition did not meet its declared sample count and active/observation-duration floor. Completed observations are preserved.",
            ));
        }
    }
    let mut groups = BTreeMap::new();
    for sample in &samples {
        groups
            .entry(sample.block_id.clone())
            .or_insert_with(|| SampleBlock {
                id: sample.block_id.clone(),
                realization_id: sample.realization_id.clone(),
                order: sample.sequence,
                preparation: "runner preparation/warmup outside measured active execution".into(),
            });
    }
    let reference_anchors: Vec<_> = groups
        .values()
        .filter(|block| block.id.starts_with("reference"))
        .map(|block| ReferenceAnchor {
            id: block.id.clone(),
            sample_sequences: samples
                .iter()
                .filter(|sample| sample.block_id == block.id)
                .map(|sample| sample.sequence)
                .collect(),
            scope: block.id.clone(),
        })
        .collect();
    for sample in &mut samples {
        if sample.block_id.starts_with("measurement") {
            sample.reference_anchor = reference_anchors
                .iter()
                .filter_map(|anchor| {
                    anchor
                        .sample_sequences
                        .iter()
                        .max()
                        .filter(|sequence| **sequence < sample.sequence)
                        .map(|sequence| (*sequence, anchor.id.clone()))
                })
                .max_by_key(|(sequence, _)| *sequence)
                .map(|(_, id)| id);
        }
    }
    let mut auxiliary_by_worker: BTreeMap<(String, String), u64> = BTreeMap::new();
    for realization in realizations.values() {
        if let (Some(bytes), Some(worker), Some(aux)) = (
            realization.parameters.get("dataset_bytes"),
            realization.parameters.get("worker"),
            realization
                .parameters
                .get("schedule_auxiliary_bytes")
                .and_then(|s| s.parse::<u64>().ok()),
        ) {
            auxiliary_by_worker
                .entry((bytes.clone(), worker.clone()))
                .and_modify(|n| *n = (*n).max(aux))
                .or_insert(aux);
        }
    }
    let mut auxiliary_by_size: BTreeMap<String, u64> = BTreeMap::new();
    for ((bytes, _), aux) in auxiliary_by_worker {
        *auxiliary_by_size.entry(bytes).or_default() += aux;
    }
    let peak_auxiliary = auxiliary_by_size.values().max().copied().unwrap_or(0);
    if let Ok(summary) = &mut result {
        if peak_auxiliary > 0 {
            summary.workload_metadata.insert(
                "access_schedule_auxiliary_bytes".into(),
                peak_auxiliary.to_string(),
            );
            if let Some(buffers) = summary
                .workload_metadata
                .get("allocated_test_buffer_bytes")
                .and_then(|s| s.parse::<u64>().ok())
            {
                summary.workload_metadata.insert(
                    "allocated_test_buffer_bytes".into(),
                    buffers.saturating_add(peak_auxiliary).to_string(),
                );
            }
        }
        summary.workload_metadata.insert(
            "measurement_runner_wall_ns".into(),
            recorder.origin.elapsed().as_nanos().to_string(),
        );
        summary.workload_metadata.insert("raw_clock_origin".into(),"measurement capture start; preparation/reference/validation phase intervals are inclusive and may overlap across workers".into());
        summary
            .workload_metadata
            .insert("measurement_policy_revision".into(), "bounded-v2".into());
        summary.workload_metadata.insert(
            "measurement_policy".into(),
            config
                .options
                .get("measurement_policy")
                .cloned()
                .unwrap_or_else(|| "extensive".into()),
        );
        summary
            .workload_metadata
            .insert("raw_observation_count".into(), samples.len().to_string());
        summary.workload_metadata.insert("measurement_stopping_rule".into(), "fixed minimum count and active-duration floor; bounded batch extensions; all completed observations retained; no variability-based rerun advice".into());
        summary.workload_metadata.insert("statistical_summary".into(), if config.options.get("access_realizations").is_some_and(|n|n!="1") {"pooled descriptive median/minimum/maximum/population standard deviation across equally covered realizations; within-realization summaries in native records; not confidence intervals"} else {"descriptive median/minimum/maximum/population standard deviation; not confidence intervals"}.into());
    }
    if let Ok(summary) = &mut result
        && config
            .options
            .get("sustained_seconds")
            .is_some_and(|s| s != "0")
    {
        let mut series: BTreeMap<String, Vec<&RawSample>> = BTreeMap::new();
        for sample in samples
            .iter()
            .filter(|s| s.block_id.starts_with("measurement:"))
        {
            series
                .entry(format!(
                    "{}:{}:{}",
                    sample.condition_id, sample.metric, sample.realization_id
                ))
                .or_default()
                .push(sample);
        }
        let mut summaries = BTreeMap::new();
        for (key, values) in series {
            let width = (values.len() / 3).max(1);
            let mean = |slice: &[&RawSample]| {
                slice.iter().map(|s| s.value).sum::<f64>() / slice.len() as f64
            };
            summaries.insert(key,serde_json::json!({"early_mean":mean(&values[..width]),"late_mean":mean(&values[values.len()-width..]),"observations_each":width,"unit":values[0].unit,"first_offset_ns":values[0].offset_ns,"last_offset_ns":values.last().unwrap().offset_ns}));
        }
        summary.workload_metadata.insert(
            "sustained_early_late".into(),
            serde_json::to_string(&summaries).expect("finite observations"),
        );
        summary.workload_metadata.insert(
            "sustained_seconds".into(),
            config.options["sustained_seconds"].clone(),
        );
        summary.workload_metadata.insert("sustained_interpretation".into(),"chronological early/late sample means per worker/condition/realization; raw offsets retain the time series; no thermal cause inferred".into());
    }
    let coverage = if samples.is_empty() {
        SampleCoverage::SummaryOnly
    } else if result.is_ok() {
        SampleCoverage::Complete
    } else {
        SampleCoverage::Partial
    };
    if samples.iter().any(|s| s.realization_id == "fixed-workload") || realizations.is_empty() {
        let seed = result
            .as_ref()
            .ok()
            .and_then(|r| r.workload_metadata.get("random_seed"))
            .and_then(|s| s.parse().ok());
        realizations.insert(
            "fixed-workload".into(),
            WorkloadRealization {
                id: "fixed-workload".into(),
                seed,
                generator_revision: None,
                parameters: BTreeMap::new(),
            },
        );
    }
    // Keep fixed-realization dispersion inspectable instead of presenting the ensemble
    // spread as repeatability of a single workload.
    for realization in realizations.values_mut() {
        let mut metrics: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for sample in samples.iter().filter(|s| {
            s.realization_id == realization.id && s.block_id.starts_with("measurement:")
        }) {
            metrics
                .entry(format!("{}:{}", sample.condition_id, sample.metric))
                .or_default()
                .push(sample.value);
        }
        for (key, mut values) in metrics {
            values.sort_by(f64::total_cmp);
            let n = values.len();
            let median = if n % 2 == 0 {
                (values[n / 2 - 1] + values[n / 2]) / 2.0
            } else {
                values[n / 2]
            };
            let mean = values.iter().sum::<f64>() / n as f64;
            let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            realization.parameters.insert(
                format!("statistics:{key}"),
                format!(
                    "count={n};median={median};min={};max={};sd={sd}",
                    values[0],
                    values[n - 1]
                ),
            );
        }
    }
    ProviderMeasurement {
        result,
        sample_coverage: coverage,
        samples,
        realizations: realizations.into_values().collect(),
        blocks: groups.into_values().collect(),
        reference_anchors,
        phase_timings,
    }
}

pub fn sample_duration(config: &BenchmarkConfig) -> Duration {
    Duration::from_nanos(sample_target_ns(config))
}

pub fn measurement_estimate(config: &BenchmarkConfig, conditions: usize) -> String {
    let seconds =
        sample_target_ns(config) as f64 * f64::from(config.samples) * conditions as f64 / 1e9;
    let scope = if config.options.contains_key("offload_passes_per_second")
        || config
            .options
            .contains_key("offload_traffic_bytes_per_second")
    {
        "arrival-observation"
    } else {
        "active measurement"
    };
    format!(
        "{conditions} conditions, {} observations each; at least {seconds:.1} seconds of {scope} time. Preparation, references and validation add time.",
        config.samples
    )
}

pub fn validation_indices(length: usize) -> Vec<usize> {
    if length == 0 {
        return vec![];
    }
    let mut indices = vec![
        0,
        length - 1,
        length / 2,
        length / 3,
        length * 2 / 3,
        (length / 2).saturating_sub(1),
        (length / 2 + 1).min(length - 1),
    ];
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Independent wider-precision reference for the bounded, finite FP32 test inputs.
/// This deliberately avoids depending on the platform's scalar `fmaf` implementation.
/// Consumers still apply their documented tolerance; this is not a general FMA API.
pub fn reference_fma_f32(a: f32, b: f32, c: f32) -> f32 {
    (f64::from(a) * f64::from(b) + f64::from(c)) as f32
}

pub struct BatchTiming {
    pub active_ns: f64,
    pub elapsed_ns: u64,
    pub batches: u32,
}
pub fn measure_batches<F>(
    target_ns: u64,
    cancellation: &CancellationToken,
    mut measure: F,
) -> Result<BatchTiming, BenchmarkError>
where
    F: FnMut() -> Result<f64, BenchmarkError>,
{
    let started = Instant::now();
    let mut window = SampleWindow::new(target_ns);
    let mut active_ns = 0.0;
    while window.needs_batch() {
        if cancellation.is_cancelled() {
            return Err(BenchmarkError::new(
                "cancelled",
                "Measurement cancelled; completed observations are retained.",
            ));
        }
        let batch = measure()?;
        if !batch.is_finite() || batch <= 0.0 {
            return Err(BenchmarkError::new(
                "invalid_timing",
                "The device returned invalid timing.",
            ));
        }
        active_ns += batch;
        window.add(batch.ceil().min(u64::MAX as f64) as u64)?;
    }
    window.finish()?;
    Ok(BatchTiming {
        active_ns,
        elapsed_ns: started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
        batches: window.batches(),
    })
}

#[cfg(test)]
mod qualification_diagnostics {
    use super::*;
    #[test]
    #[ignore = "diagnostic recording cost; synthetic observations are never benchmark scores"]
    fn observation_recording_cost() {
        for progress_enabled in [false, true] {
            let (sender, _receiver) = std::sync::mpsc::channel();
            let config = BenchmarkConfig::default();
            let mut elapsed = Duration::ZERO;
            let measured = capture_measurement_with_progress(
                "qualification.diagnostic",
                &config,
                progress_enabled.then_some(sender),
                || {
                    let start = Instant::now();
                    for _ in 0..20000 {
                        record_observation(
                            "synthetic",
                            "synthetic",
                            1.,
                            "units/s",
                            1,
                            "units",
                            1,
                            1,
                        );
                    }
                    elapsed = start.elapsed();
                    Err(BenchmarkError::new(
                        "diagnostic_only",
                        "Synthetic recorder cost; no benchmark score.",
                    ))
                },
            );
            assert_eq!(measured.samples.len(), 20000);
            println!(
                "recorder_diagnostic progress={progress_enabled} observations=20000 nanoseconds_per_observation={:.0}",
                elapsed.as_nanos() as f64 / 20000.
            );
        }
    }
}
