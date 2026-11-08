use gluj_bench_core::*;
use std::collections::BTreeMap;

fn config(shorter: bool) -> BenchmarkConfig {
    BenchmarkConfig {
        samples: if shorter { 3 } else { 6 },
        target_duration_ms: if shorter { 300 } else { 1500 },
        options: BTreeMap::from([(
            "measurement_policy".into(),
            if shorter { "shorter" } else { "extensive" }.into(),
        )]),
    }
}
fn result() -> BenchmarkResult {
    BenchmarkResult {
        benchmark_id: "fixture".into(),
        device_id: "cpu:test".into(),
        elapsed_ns: 100,
        metrics: vec![],
        workload_metadata: BTreeMap::new(),
        device_metadata: BTreeMap::new(),
    }
}
fn observation(value: f64) {
    record_observation(
        "fixed-input",
        "compute",
        value,
        "operations/s",
        10,
        "operations",
        100_000_000,
        120_000_000,
    );
}

#[test]
fn extensive_and_shorter_floors_are_independent_of_sample_count() {
    let default = resolve_measurement_config(&BenchmarkConfig::default()).unwrap();
    assert_eq!(default.samples, 6);
    assert!(default.target_duration_ms >= 1500);
    let mut more = config(false);
    more.samples = 24;
    let more = resolve_measurement_config(&more).unwrap();
    assert!(more.target_duration_ms >= 6000);
    assert_eq!(sample_target_ns(&more), 250_000_000);
    let short = resolve_measurement_config(&config(true)).unwrap();
    assert_eq!(short.samples, 3);
    assert_eq!(short.target_duration_ms, 300);
    for measured in [config(false), config(true)] {
        let reference = reference_config(&measured, "reference-before");
        assert_eq!(reference.samples, 3);
        assert_eq!(reference.target_duration_ms, 300);
    }
    for samples in [0, 65, u32::MAX] {
        let mut invalid = config(false);
        invalid.samples = samples;
        assert!(resolve_measurement_config(&invalid).is_err());
    }
}

#[test]
fn batch_extensions_have_an_exact_count_limit_and_preserve_overshoot() {
    let mut window = SampleWindow::new(20);
    window.add(3).unwrap();
    assert!(window.needs_batch());
    window.add(19).unwrap();
    assert_eq!(window.finish().unwrap(), 22);
    assert!(!window.needs_batch());
    let mut exhausted = SampleWindow::new(10_000);
    while exhausted.needs_batch() {
        exhausted.add(1).unwrap();
    }
    assert_eq!(exhausted.batches(), 1024);
    assert_eq!(
        exhausted.finish().unwrap_err().code,
        "measurement_budget_exhausted"
    );
}

#[test]
fn cancellation_is_checked_between_device_batches() {
    let token = CancellationToken::default();
    let mut calls = 0;
    let error = measure_batches(100, &token, || {
        calls += 1;
        token.cancel();
        Ok(10.)
    })
    .err()
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(error.code, "cancelled");
    assert!(measure_batches(100, &CancellationToken::default(), || Ok(f64::NAN)).is_err());
}

#[test]
fn slow_and_variable_observations_are_neither_filtered_nor_replaced() {
    let values = [100., 1., 103., 2., 99., 0.];
    let measurement = capture_measurement("fixture", &config(true), || {
        for value in values {
            observation(value);
        }
        Ok(result())
    });
    assert_eq!(
        measurement
            .samples
            .iter()
            .map(|s| s.value)
            .collect::<Vec<_>>(),
        values
    );
    assert_eq!(measurement.sample_coverage, SampleCoverage::Complete);
    assert!(
        measurement.samples.iter().all(|s| s.useful_work == 10
            && s.active_ns == 100_000_000
            && s.elapsed_ns == 120_000_000)
    );
}

#[test]
fn failed_and_cancelled_runs_keep_completed_raw_observations() {
    for code in ["cancelled", "result_validation_failed"] {
        let measurement = capture_measurement("fixture", &config(true), || {
            observation(10.);
            Err(BenchmarkError::new(code, "fixture"))
        });
        assert_eq!(measurement.samples.len(), 1);
        assert_eq!(measurement.sample_coverage, SampleCoverage::Partial);
        assert_eq!(measurement.result.unwrap_err().code, code);
    }
}

#[test]
fn invalid_observations_cannot_produce_success_or_erase_prior_observations() {
    let measurement = capture_measurement("fixture", &config(true), || {
        observation(10.);
        observation(f64::INFINITY);
        Ok(result())
    });
    assert_eq!(measurement.result.unwrap_err().code, "invalid_observation");
    assert_eq!(measurement.samples.len(), 1);
    assert_eq!(
        capture_measurement("fixture", &config(true), || Ok(result()))
            .result
            .unwrap_err()
            .code,
        "measurement_unavailable"
    );
}

#[test]
fn incomplete_count_or_duration_cannot_become_success() {
    let budget = config(true);
    let count_limited = capture_measurement("fixture", &budget, || {
        for _ in 0..budget.samples - 1 {
            observation(10.);
        }
        Ok(result())
    });
    assert_eq!(
        count_limited.result.unwrap_err().code,
        "incomplete_measurement_coverage"
    );
    assert_eq!(count_limited.samples.len(), (budget.samples - 1) as usize);
    let duration_limited = capture_measurement("fixture", &config(true), || {
        for _ in 0..6 {
            record_observation(
                "fixed-input",
                "compute",
                10.,
                "operations/s",
                10,
                "operations",
                1,
                2,
            );
        }
        Ok(result())
    });
    assert_eq!(
        duration_limited.result.unwrap_err().code,
        "incomplete_measurement_coverage"
    );
    assert_eq!(duration_limited.samples.len(), 6);
}

#[test]
fn explicit_reference_anchors_do_not_pool_with_dataset_measurements() {
    let cfg = config(true);
    let measured = capture_measurement("fixture", &cfg, || {
        {
            let _scope =
                MeasurementScope::enter(&reference_config(&cfg, "reference-before"), "register");
            observation(100.);
        }
        observation(10.);
        {
            let _scope =
                MeasurementScope::enter(&reference_config(&cfg, "reference-after"), "register");
            observation(90.);
        }
        Ok(result())
    });
    assert_eq!(measured.reference_anchors.len(), 2);
    assert!(
        measured.samples[1]
            .reference_anchor
            .as_ref()
            .unwrap()
            .starts_with("reference-before")
    );
    assert!(measured.samples[0].reference_anchor.is_none());
    assert_eq!(measured.realizations.len(), 1);
    assert!(measured.realizations[0].seed.is_none());
}

#[test]
fn shared_worker_capture_has_unique_chronological_sequence_numbers() {
    let measurement = capture_measurement("fixture", &config(true), || {
        let recorder = observation_recorder().unwrap();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let recorder = recorder.clone();
                scope.spawn(move || {
                    for _ in 0..25 {
                        recorder.record(
                            "worker",
                            "compute",
                            10.,
                            "operations/s",
                            10,
                            "operations",
                            100_000_000,
                            120_000_000,
                            "passed",
                        );
                    }
                });
            }
        });
        Ok(result())
    });
    assert_eq!(measurement.samples.len(), 100);
    for (index, sample) in measurement.samples.iter().enumerate() {
        assert_eq!(sample.sequence, index as u64);
    }
    assert!(
        measurement
            .samples
            .windows(2)
            .all(|samples| samples[0].offset_ns <= samples[1].offset_ns)
    );
}

#[test]
fn counterbalanced_visits_preserve_each_declared_condition_and_total_sample_count() {
    let visits = condition_visits(&[1, 2, 3], &config(false));
    assert_eq!(
        visits.iter().map(|(size, _, _)| *size).collect::<Vec<_>>(),
        [1, 2, 3, 3, 2, 1, 1, 2, 3]
    );
    for size in [1, 2, 3] {
        let matching: Vec<_> = visits
            .iter()
            .filter(|(candidate, _, _)| *candidate == size)
            .collect();
        assert_eq!(
            matching.iter().map(|(_, cfg, _)| cfg.samples).sum::<u32>(),
            6
        );
        assert_eq!(
            matching
                .iter()
                .map(|(_, cfg, _)| cfg.target_duration_ms)
                .sum::<u64>(),
            1500
        );
    }
    for length in 1..100 {
        let indices = validation_indices(length);
        assert!(indices.contains(&0) && indices.contains(&(length - 1)));
        assert!(indices.iter().all(|index| *index < length));
    }
}

#[test]
fn gpu_service_and_observation_window_keep_their_distinct_clock_semantics() {
    let measured = capture_measurement("gpu.fixture", &config(true), || {
        for _ in 0..6 {
            record_observation(
                "overload",
                "completed_compute_over_observation_window",
                0.,
                "operations/s",
                0,
                "operations",
                150_000_000,
                100_000_000,
            );
        }
        Ok(result())
    });
    assert_eq!(
        measured.samples[0].timing_scope,
        SampleTimingScope::OfferedWindowWithAggregateService
    );
    assert_eq!(measured.samples[0].active_ns, 150_000_000);
    assert_eq!(measured.samples[0].elapsed_ns, 100_000_000);
}

struct ReportingProvider;
impl BenchmarkProvider for ReportingProvider {
    fn devices(&self) -> Vec<DeviceDescriptor> {
        vec![DeviceDescriptor {
            id: "cpu:test".into(),
            name: "fixture".into(),
            category: DeviceCategory::Cpu,
            available: true,
            status: String::new(),
            properties: BTreeMap::new(),
            caches: vec![],
        }]
    }
    fn benchmarks(&self) -> Vec<BenchmarkDescriptor> {
        vec![BenchmarkDescriptor {
            id: "cpu.fixture".into(),
            name: "fixture".into(),
            category: BenchmarkCategory::Cpu,
            workload: "fixture".into(),
            data_type: "fixture".into(),
            unit: "operations/s".into(),
            supported_device_ids: vec!["cpu:test".into()],
            available: true,
            unavailable_reason: String::new(),
            suite_id: String::new(),
            display_order: 0,
            metadata: BTreeMap::new(),
        }]
    }
    fn run(
        &mut self,
        id: &str,
        config: &BenchmarkConfig,
        _: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        assert_eq!(cpu_activity_percent(), 99);
        progress(ProgressUpdate {
            fraction: 0.5,
            phase: "measure".into(),
            message: "fixture".into(),
        });
        for _ in 0..config.samples {
            observation(10.);
        }
        let mut result = result();
        result.benchmark_id = id.into();
        Ok(result)
    }
}

#[test]
fn execution_thread_preserves_pacing_and_forwards_actual_sample_progress() {
    let mut registry = BenchmarkRegistry::default();
    registry.add(Box::new(ReportingProvider));
    let mut config = config(true);
    config
        .options
        .insert("cpu_activity_percent".into(), "99".into());
    let mut updates = Vec::new();
    let record = registry.run_recorded(
        "cpu.fixture",
        &config,
        &CancellationToken::default(),
        &mut |update| updates.push(update),
    );
    assert_eq!(record.status, CompletionStatus::Completed);
    assert_eq!(record.samples.len(), config.samples as usize);
    let sample_progress = updates
        .iter()
        .find(|update| update.phase == "sample_coverage")
        .expect("live sample progress");
    assert_eq!(sample_progress.fraction, 0.5);
    assert!(sample_progress.message.contains("completed observations"));
    record.validate().unwrap();
}

#[test]
fn phase_intervals_preserve_labels_and_inclusive_clock_offsets() {
    let measured = capture_measurement("cpu.fixture", &config(true), || {
        {
            let _preparation = phase_timer("preparation");
            let _validation = phase_timer("validation");
        }
        for _ in 0..6 {
            observation(10.);
        }
        Ok(result())
    });
    assert!(measured.result.is_ok());
    assert_eq!(measured.phase_timings.len(), 2);
    let preparation = &measured.phase_timings[0];
    let validation = &measured.phase_timings[1];
    assert_eq!(preparation.phase, "preparation");
    assert_eq!(validation.phase, "validation");
    assert!(preparation.inclusive && validation.inclusive);
    assert!(preparation.offset_ns <= validation.offset_ns);
    assert!(
        preparation.offset_ns + preparation.elapsed_ns
            >= validation.offset_ns + validation.elapsed_ns
    );
}

struct LateCancellationProvider;
impl BenchmarkProvider for LateCancellationProvider {
    fn devices(&self) -> Vec<DeviceDescriptor> {
        ReportingProvider.devices()
    }
    fn benchmarks(&self) -> Vec<BenchmarkDescriptor> {
        ReportingProvider.benchmarks()
    }
    fn run(
        &mut self,
        id: &str,
        config: &BenchmarkConfig,
        cancellation: &CancellationToken,
        progress: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let result = ReportingProvider.run(id, config, cancellation, progress);
        cancellation.cancel();
        result
    }
}
#[test]
fn cancellation_after_last_observation_preserves_raw_data_without_a_score() {
    let mut registry = BenchmarkRegistry::default();
    registry.add(Box::new(LateCancellationProvider));
    let mut config = config(true);
    config
        .options
        .insert("cpu_activity_percent".into(), "99".into());
    let record = registry.run_recorded(
        "cpu.fixture",
        &config,
        &CancellationToken::default(),
        &mut |_| {},
    );
    assert_eq!(record.status, CompletionStatus::Cancelled);
    assert_eq!(record.sample_coverage, SampleCoverage::Partial);
    assert_eq!(record.samples.len(), config.samples as usize);
    assert!(record.result.is_none());
    record.validate().unwrap();
}

#[test]
fn sustained_coverage_keeps_a_contiguous_condition_and_short_reference_budget() {
    let mut uneven = config(true);
    uneven.target_duration_ms = 1000;
    uneven
        .options
        .insert("sustained_seconds".into(), "1".into());
    let uneven = resolve_measurement_config(&uneven).unwrap();
    for (_, part, _) in condition_visits(&[1], &uneven) {
        assert!(sample_target_ns(&part) >= sample_target_ns(&uneven));
        assert!(part.target_duration_ms >= uneven.target_duration_ms);
    }
    let mut requested = config(true);
    requested
        .options
        .insert("sustained_seconds".into(), "15".into());
    let sustained = resolve_measurement_config(&requested).unwrap();
    assert_eq!(sustained.target_duration_ms, 15000);
    assert_eq!(sustained.samples, 3);
    let visits = condition_visits(&[1, 2], &sustained);
    assert_eq!(visits.len(), 2);
    assert!(
        visits
            .iter()
            .all(|(_, c, _)| c.samples == 3 && c.target_duration_ms == 15000)
    );
    let reference = reference_config(&sustained, "reference-before");
    assert_eq!(
        resolve_measurement_config(&reference)
            .unwrap()
            .target_duration_ms,
        300
    );
    let measured = capture_measurement("cpu.fixture", &sustained, || {
        for value in [10., 11., 12., 13., 14., 15.] {
            record_observation(
                "input",
                "compute",
                value,
                "operations/s",
                10,
                "operations",
                2500000000,
                2500000000,
            );
        }
        Ok(result())
    });
    assert!(measured.result.is_ok());
    let summary = measured.result.unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&summary.workload_metadata["sustained_early_late"]).unwrap();
    let series = json.as_object().unwrap().values().next().unwrap();
    assert_eq!(series["early_mean"], 10.5);
    assert_eq!(series["late_mean"], 14.5);
    assert_eq!(measured.samples.len(), 6);
}
