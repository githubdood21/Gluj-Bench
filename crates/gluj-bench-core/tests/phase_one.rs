use gluj_bench_core::*;
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn device(gpu: bool) -> DeviceDescriptor {
    DeviceDescriptor {
        id: if gpu { "gpu:fixture" } else { "cpu:system" }.into(),
        name: "Fixture".into(),
        category: if gpu {
            DeviceCategory::Gpu
        } else {
            DeviceCategory::Cpu
        },
        available: true,
        status: "ready".into(),
        properties: BTreeMap::new(),
        caches: vec![],
    }
}
fn descriptor(id: &str) -> BenchmarkDescriptor {
    let gpu = id.starts_with("gpu.");
    BenchmarkDescriptor {
        id: id.into(),
        name: id.into(),
        category: if gpu {
            BenchmarkCategory::Gpu
        } else {
            BenchmarkCategory::Cpu
        },
        workload: "fixture".into(),
        data_type: "fixture".into(),
        unit: "operations/s".into(),
        supported_device_ids: vec![device(gpu).id],
        available: true,
        unavailable_reason: String::new(),
        suite_id: "fixture".into(),
        display_order: 0,
        metadata: BTreeMap::from([("kernel_revision".into(), "fixture-v1".into())]),
    }
}
fn config(options: &[(&str, &str)]) -> BenchmarkConfig {
    BenchmarkConfig {
        options: options
            .iter()
            .map(|(k, v)| ((*k).into(), (*v).into()))
            .collect(),
        ..Default::default()
    }
}
fn summary() -> BenchmarkResult {
    BenchmarkResult {
        benchmark_id: "cpu.performance.integer.i64".into(),
        device_id: device(false).id,
        elapsed_ns: 10,
        metrics: vec![],
        workload_metadata: BTreeMap::new(),
        device_metadata: BTreeMap::new(),
    }
}
struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
                "gluj-contract-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )))
    }
    fn store(&self) -> RunStore {
        RunStore::new(&self.0)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn equivalent_explicit_conditions_resolve_identically_and_round_trip() {
    let schema = legacy_condition(&descriptor("cpu.performance.integer.i64"));
    let defaults = schema
        .resolve(&BenchmarkConfig::default(), &[device(false)])
        .unwrap();
    assert_eq!(defaults.options["dataset_mode"], "register");
    assert!(!defaults.options.contains_key("ram_budget_percent"));
    let explicit = schema
        .resolve(&defaults.config(), &[device(false)])
        .unwrap();
    assert_eq!(defaults, explicit);
    assert_eq!(
        serde_json::from_value::<ResolvedCondition>(json!(defaults)).unwrap(),
        defaults
    );
    let different_format = config(&[
        ("cpu_core_limit", "0004"),
        ("thread_mode", "physical_cores"),
    ]);
    assert_eq!(
        schema
            .resolve(&different_format, &[device(false)])
            .unwrap()
            .options["cpu_core_limit"],
        "4"
    );
}

#[test]
fn rejects_unknown_inert_conflicting_and_invalid_options() {
    let schema = legacy_condition(&descriptor("cpu.performance.avx2.f32_fma.scaling"));
    for options in [
        vec![("made_up", "1")],
        vec![("dataset_mode", "single")],
        vec![("vector_preset", "custom")],
        vec![("vector_preset", "ram"), ("dataset_bytes", "500000")],
        vec![("vector_preset", "register"), ("dataset_mode", "sweep")],
        vec![
            ("vector_preset", "register"),
            ("arithmetic_iterations", "64"),
        ],
        vec![
            ("cpu_core_limit", "2"),
            ("thread_mode", "logical_processors"),
        ],
        vec![("ram_budget_mib", "100"), ("ram_budget_percent", "20")],
    ] {
        assert!(
            schema.resolve(&config(&options), &[device(false)]).is_err(),
            "{options:?}"
        );
    }
    let gpu = legacy_condition(&descriptor("gpu.performance.fp32.offload.scaling"));
    assert!(
        gpu.resolve(
            &config(&[
                ("offload_passes_per_second", "1"),
                ("offload_traffic_bytes_per_second", "1048576")
            ]),
            &[device(true)]
        )
        .is_err()
    );
    assert!(
        gpu.resolve(&config(&[("device_id", "missing")]), &[device(true)])
            .is_err()
    );
    let mut unavailable = device(true);
    unavailable.available = false;
    assert!(
        gpu.resolve(&BenchmarkConfig::default(), &[unavailable])
            .is_err()
    );
}

#[test]
fn every_existing_descriptor_has_one_of_fifteen_family_destinations() {
    let ids = [
        "cpu.bandwidth.cache.l1",
        "cpu.bandwidth.cache.l2",
        "cpu.bandwidth.cache.l3",
        "cpu.bandwidth.memory",
        "cpu.latency.cache.l1",
        "cpu.latency.cache.l2",
        "cpu.latency.cache.l3",
        "cpu.latency.memory",
        "cpu.latency.memory.localized",
        "cpu.performance.integer.i64",
        "cpu.performance.single_thread.integer.i64",
        "cpu.performance.float.f32",
        "cpu.performance.float.f64",
        "cpu.performance.avx2.f32_fma.scaling",
        "cpu.performance.avx2.f64_fma",
        "cpu.performance.matrix.fp32.scaling",
        "cpu.performance.string.ascii",
        "cpu.performance.compression.deflate",
        "cpu.performance.decompression.deflate",
        "cpu.performance.prime.sieve",
        "cpu.performance.extended.popcnt",
        "cpu.performance.extended.aes",
        "gpu.bandwidth.cache",
        "gpu.bandwidth.vram",
        "gpu.bandwidth.host_link",
        "gpu.performance.fp16",
        "gpu.performance.fp64",
        "gpu.performance.fp32.scaling",
        "gpu.performance.fp32.offload.scaling",
        "gpu.performance.matrix.fp16.scaling",
        "gpu.performance.matrix.int8.scaling",
        "gpu.performance.matrix.fp8",
        "gpu.performance.matrix.sparse.fp16",
        "gpu.performance.matrix.sparse.int8",
        "gpu.performance.matrix.sparse.fp8",
    ];
    let families = family_catalogue(
        ids.iter()
            .map(|id| legacy_condition(&descriptor(id)))
            .collect(),
    );
    assert_eq!(families.len(), 15);
    assert_eq!(
        families.iter().map(|f| f.conditions.len()).sum::<usize>(),
        35
    );
    assert!(
        families
            .iter()
            .all(|family| ids.iter().all(|id| family.id != *id))
    );
    for family in &families {
        for condition in &family.conditions {
            let resolved = condition
                .resolve(
                    &BenchmarkConfig::default(),
                    &[device(condition.benchmark_id.starts_with("gpu."))],
                )
                .unwrap();
            assert_eq!(
                condition
                    .resolve(
                        &resolved.config(),
                        &[device(condition.benchmark_id.starts_with("gpu."))]
                    )
                    .unwrap(),
                resolved
            );
        }
    }
}

#[test]
fn policy_changes_are_descriptive_and_kernel_changes_are_incompatible() {
    let resolved = legacy_condition(&descriptor("cpu.performance.integer.i64"))
        .resolve(&BenchmarkConfig::default(), &[device(false)])
        .unwrap();
    let mut left = ConditionIdentity::from_result(resolved, &summary());
    assert!(matches!(
        compare_conditions(&left, &left),
        Comparability::DescriptiveOnly(_)
    ));
    left.complete = true;
    assert_eq!(compare_conditions(&left, &left), Comparability::Comparable);
    let mut right = left.clone();
    right.resolved.policy.samples += 1;
    assert!(matches!(
        compare_conditions(&left, &right),
        Comparability::DescriptiveOnly(_)
    ));
    right.resolved.kernel_revision = Some("v2".into());
    assert!(
        matches!(compare_conditions(&left, &right), Comparability::Incompatible(reasons) if reasons.contains(&"kernel revision differs".into()))
    );
}

#[test]
fn suite_order_coverage_and_duplicate_validation_survive_serialization() {
    let resolved = legacy_condition(&descriptor("cpu.performance.integer.i64"))
        .resolve(&BenchmarkConfig::default(), &[device(false)])
        .unwrap();
    let mut plan = SuitePlan {
        version: CONTRACT_VERSION,
        id: "cpu-check".into(),
        order_seed: 42,
        conditions: vec![SuiteCondition {
            id: "integer-register".into(),
            condition: resolved,
            intended_coverage: vec!["integer arithmetic".into()],
            estimated_duration_ms: None,
            estimated_peak_bytes: None,
            status: CompletionStatus::Pending,
        }],
    };
    plan.validate().unwrap();
    assert_eq!(
        serde_json::from_value::<SuitePlan>(json!(plan)).unwrap(),
        plan
    );
    let mut duplicate = plan.conditions[0].clone();
    duplicate.id = "other".into();
    plan.conditions.push(duplicate);
    assert_eq!(plan.validate().unwrap_err().code, "duplicate_condition");
}

#[test]
fn durable_history_is_immutable_and_recovers_a_lost_index() {
    let temp = Temporary::new();
    let store = temp.store();
    let mut record = RunRecord::start(
        "cpu.performance.integer.i64",
        &BenchmarkConfig::default(),
        vec![device(false)],
    );
    record.finish(Ok(summary()));
    store.save(&record).unwrap();
    assert_eq!(store.load(&record.run_id).unwrap(), record);
    assert!(store.save(&record).is_err());
    fs::write(temp.0.join("index.json"), b"interrupted index").unwrap();
    let index = store.rebuild_index().unwrap();
    assert_eq!(index.runs.len(), 1);
    let mut replacement = record.clone();
    replacement.result.as_mut().unwrap().elapsed_ns = 999;
    assert!(store.save(&replacement).is_err());
    assert_eq!(store.load(&record.run_id).unwrap(), record);
    assert!(store.load("../escape").is_err());
    let mut future = json!(record);
    future["version"] = json!(999);
    fs::write(
        temp.0.join("runs").join(format!("{}.json", record.run_id)),
        serde_json::to_vec(&future).unwrap(),
    )
    .unwrap();
    assert!(store.rebuild_index().is_err());
}

#[test]
fn migration_preserves_original_ids_revisions_sources_and_unknowns() {
    let temp = Temporary::new();
    let store = temp.store();
    let bytes = include_bytes!("fixtures/results-v1.json");
    assert_eq!(
        store
            .import_legacy_results(bytes, "results.json", RunSource::LegacyLocal)
            .unwrap(),
        5
    );
    assert_eq!(
        store
            .import_legacy_results(bytes, "results.json", RunSource::LegacyLocal)
            .unwrap(),
        0
    );
    let index = store.rebuild_index().unwrap();
    let records: Vec<_> = index
        .runs
        .iter()
        .map(|entry| store.load(&entry.run_id).unwrap())
        .collect();
    assert!(records.iter().all(|r| r.samples.is_empty()
        && r.sample_coverage == SampleCoverage::SummaryOnly
        && r.resolved.is_none()
        && r.application_version.is_none()));
    let scalar = records
        .iter()
        .find(|r| r.requested_benchmark_id == "cpu.performance.float.f32")
        .unwrap();
    assert_eq!(scalar.started_unix_ms, Some(1_700_000_000_000));
    let offload = records
        .iter()
        .find(|r| r.requested_benchmark_id == "gpu.performance.fp32.offload75.scaling")
        .unwrap();
    assert_eq!(offload.started_unix_ms, None);
    assert_eq!(
        offload.result.as_ref().unwrap().workload_metadata["offload_profile_revision"],
        "old"
    );
    assert_eq!(
        store
            .import_legacy_results(bytes, "snapshot.png", RunSource::ImportedComparison)
            .unwrap(),
        5
    );
    assert_eq!(
        store
            .rebuild_index()
            .unwrap()
            .runs
            .iter()
            .filter(|r| r.source == RunSource::ImportedComparison)
            .count(),
        5
    );
}

#[test]
fn raw_samples_and_cancelled_or_failed_runs_have_explicit_semantics() {
    let temp = Temporary::new();
    let store = temp.store();
    let mut record = RunRecord::start("test", &BenchmarkConfig::default(), vec![]);
    record.finish(Err(BenchmarkError::new("cancelled", "User cancelled")));
    store.save(&record).unwrap();
    assert!(record.result.is_none());
    assert_eq!(record.status, CompletionStatus::Cancelled);
    let sample = RawSample {
        timing_scope: SampleTimingScope::ActiveExecution,
        version: CONTRACT_VERSION,
        sequence: 0,
        block_id: "b1".into(),
        realization_id: "seed-1".into(),
        condition_id: "size-1024".into(),
        metric: "throughput".into(),
        value: 1.0,
        unit: "bytes/s".into(),
        useful_work: 1024,
        useful_work_unit: "bytes".into(),
        active_ns: 10,
        elapsed_ns: 12,
        offset_ns: 0,
        reference_anchor: Some("before".into()),
        validation: ValidationStatus {
            status: "passed".into(),
            scope: "full output".into(),
        },
        status: CompletionStatus::Completed,
    };
    record.samples.push(sample.clone());
    record.realizations.push(WorkloadRealization {
        id: "seed-1".into(),
        seed: Some(42),
        generator_revision: Some("fixture-v1".into()),
        parameters: BTreeMap::new(),
    });
    record.blocks.push(SampleBlock {
        id: "b1".into(),
        realization_id: "seed-1".into(),
        order: 0,
        preparation: "warm".into(),
    });
    record.reference_anchors.push(ReferenceAnchor {
        id: "before".into(),
        sample_sequences: vec![0],
        scope: "fixture reference".into(),
    });
    assert!(record.validate().is_err());
    record.sample_coverage = SampleCoverage::Partial;
    record.validate().unwrap();
    assert_eq!(
        serde_json::from_value::<RunRecord>(json!(record)).unwrap(),
        record
    );
    record.samples.push(sample);
    assert!(record.validate().is_err());
    record.samples.pop();
    record.sample_coverage = SampleCoverage::Complete;
    assert!(record.validate().is_err());
    record.sample_coverage = SampleCoverage::Partial;
    record.result = Some(summary());
    assert!(record.validate().is_err());
}

#[test]
fn additive_protocol_commands_share_run_parsing_and_reject_bad_types() {
    let request = parse_request(&json!({"protocol":PROTOCOL_VERSION,"id":"resolve","command":"resolve","arguments":{"benchmark_id":"cpu.performance.integer.i64","samples":7}})).unwrap();
    assert!(matches!(
        request.command,
        ProtocolCommand::Resolve {
            config: BenchmarkConfig { samples: 7, .. },
            ..
        }
    ));
    assert_eq!(
        parse_request(&json!({"protocol":PROTOCOL_VERSION,"id":"families","command":"families"}))
            .unwrap()
            .command,
        ProtocolCommand::Families
    );
    for bad in [json!(null), json!("5"), json!(-1), json!(1.5), json!(0)] {
        assert!(parse_request(&json!({"protocol":PROTOCOL_VERSION,"id":"bad","command":"run","arguments":{"benchmark_id":"test","samples":bad}})).is_err());
    }
}

#[test]
fn mode_capabilities_do_not_require_buffer_support_for_register_work() {
    let mut condition = legacy_condition(&descriptor("gpu.performance.matrix.fp16.scaling"));
    condition.capabilities = vec![
        ModeCapability {
            when: BTreeMap::from([("gpu_matrix_preset".into(), "register".into())]),
            supported_device_ids: vec![device(true).id],
            unavailable_reason: "register unsupported".into(),
        },
        ModeCapability {
            when: BTreeMap::from([("gpu_matrix_preset".into(), "legacy".into())]),
            supported_device_ids: vec![],
            unavailable_reason: "16-bit buffer storage unsupported".into(),
        },
    ];
    assert!(
        condition
            .resolve(
                &config(&[("gpu_matrix_preset", "register")]),
                &[device(true)]
            )
            .is_ok()
    );
    assert!(
        condition
            .resolve(&BenchmarkConfig::default(), &[device(true)])
            .is_err()
    );
}

struct PartialProvider;
impl BenchmarkProvider for PartialProvider {
    fn devices(&self) -> Vec<DeviceDescriptor> {
        vec![device(false)]
    }
    fn benchmarks(&self) -> Vec<BenchmarkDescriptor> {
        vec![descriptor("cpu.performance.integer.i64")]
    }
    fn run(
        &mut self,
        _: &str,
        _: &BenchmarkConfig,
        _: &CancellationToken,
        _: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        Ok(summary())
    }
    fn run_with_samples(
        &mut self,
        _: &str,
        _: &BenchmarkConfig,
        _: &CancellationToken,
        _: &mut ProgressCallback<'_>,
    ) -> ProviderMeasurement {
        ProviderMeasurement {
            result: Err(BenchmarkError::new(
                "cancelled",
                "user cancelled after first observation",
            )),
            samples: vec![RawSample {
                timing_scope: SampleTimingScope::ActiveExecution,
                version: CONTRACT_VERSION,
                sequence: 0,
                block_id: "block".into(),
                realization_id: "fixed".into(),
                condition_id: "register".into(),
                metric: "work".into(),
                value: 1.,
                unit: "operations/s".into(),
                useful_work: 1,
                useful_work_unit: "operations".into(),
                active_ns: 10,
                elapsed_ns: 12,
                offset_ns: 1,
                reference_anchor: None,
                validation: ValidationStatus {
                    status: "passed".into(),
                    scope: "fixture".into(),
                },
                status: CompletionStatus::Completed,
            }],
            sample_coverage: SampleCoverage::Partial,
            realizations: vec![WorkloadRealization {
                id: "fixed".into(),
                seed: None,
                generator_revision: None,
                parameters: BTreeMap::new(),
            }],
            blocks: vec![SampleBlock {
                id: "block".into(),
                realization_id: "fixed".into(),
                order: 0,
                preparation: "warm".into(),
            }],
            reference_anchors: vec![],
            phase_timings: vec![],
        }
    }
}

#[test]
fn registry_preserves_real_partial_samples_when_a_provider_is_cancelled() {
    let mut registry = BenchmarkRegistry::default();
    registry.add(Box::new(PartialProvider));
    let record = registry.run_recorded(
        "cpu.performance.integer.i64",
        &BenchmarkConfig::default(),
        &CancellationToken::default(),
        &mut |_| {},
    );
    assert_eq!(record.status, CompletionStatus::Cancelled);
    assert_eq!(record.samples.len(), 1);
    assert!(record.result.is_none());
    record.validate().unwrap();
    let temp = Temporary::new();
    temp.store().save(&record).unwrap();
    assert_eq!(temp.store().load(&record.run_id).unwrap(), record);
    assert!(matches!(
        compare_runs(&record, &record),
        Comparability::Incompatible(_)
    ));
}

#[test]
fn full_sample_history_round_trips_without_expanding_the_compact_index() {
    let mut registry = BenchmarkRegistry::default();
    registry.add(Box::new(PartialProvider));
    let partial = registry.run_recorded(
        "cpu.performance.integer.i64",
        &BenchmarkConfig::default(),
        &CancellationToken::default(),
        &mut |_| {},
    );
    let mut record = RunRecord::start(
        "cpu.performance.integer.i64",
        &BenchmarkConfig::default(),
        vec![device(false)],
    );
    record.resolved = partial.resolved;
    record.realizations = partial.realizations;
    record.blocks = partial.blocks;
    record.finish(Ok(summary()));
    record.sample_coverage = SampleCoverage::Complete;
    let template = partial.samples[0].clone();
    record.samples = (0..1000)
        .map(|sequence| RawSample {
            sequence,
            offset_ns: sequence * 20,
            value: (sequence % 7) as f64,
            ..template.clone()
        })
        .collect();
    let temp = Temporary::new();
    let store = temp.store();
    store.save(&record).unwrap();
    assert_eq!(store.load(&record.run_id).unwrap(), record);
    let index_size = fs::metadata(temp.0.join("index.json")).unwrap().len();
    let full_size = fs::metadata(temp.0.join("runs").join(format!("{}.json", record.run_id)))
        .unwrap()
        .len();
    assert!(index_size < 2048 && index_size * 100 < full_size);
    assert_eq!(store.rebuild_index().unwrap().runs[0].sample_count, 1000);
}

#[test]
fn latest_result_cache_does_not_duplicate_an_existing_native_measurement() {
    let temp = Temporary::new();
    let store = temp.store();
    let mut native = RunRecord::start(
        "cpu.performance.integer.i64",
        &BenchmarkConfig::default(),
        vec![device(false)],
    );
    native.finish(Ok(summary()));
    store.save(&native).unwrap();
    let mut cached = native.result.clone().unwrap();
    cached
        .device_metadata
        .insert("native_run_id".into(), native.run_id);
    cached
        .device_metadata
        .insert("saved_at".into(), "1700000000".into());
    let bytes = serde_json::to_vec(&json!({"version":1,"results":[cached]})).unwrap();
    assert_eq!(
        store
            .import_legacy_results(&bytes, "results.json", RunSource::LegacyLocal)
            .unwrap(),
        0
    );
    assert_eq!(
        store
            .import_legacy_results(&bytes, "snapshot.png", RunSource::ImportedComparison)
            .unwrap(),
        1
    );
}

struct Int64AliasProvider;
impl BenchmarkProvider for Int64AliasProvider {
    fn devices(&self) -> Vec<DeviceDescriptor> {
        vec![device(false)]
    }
    fn benchmarks(&self) -> Vec<BenchmarkDescriptor> {
        vec![
            descriptor("cpu.performance.integer.i64"),
            descriptor("cpu.performance.single_thread.integer.i64"),
        ]
    }
    fn run(
        &mut self,
        _: &str,
        _: &BenchmarkConfig,
        _: &CancellationToken,
        _: &mut ProgressCallback<'_>,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        unreachable!("resolution test never executes work")
    }
}
#[test]
fn legacy_single_thread_int64_resolves_to_the_canonical_one_core_register_condition() {
    let mut registry = BenchmarkRegistry::default();
    registry.add(Box::new(Int64AliasProvider));
    let alias = registry
        .resolve(
            "cpu.performance.single_thread.integer.i64",
            &BenchmarkConfig::default(),
        )
        .unwrap();
    let canonical = registry
        .resolve(
            "cpu.performance.integer.i64",
            &config(&[
                ("cpu_core_limit", "1"),
                ("thread_mode", "physical_cores"),
                ("int64_preset", "register"),
                ("dataset_mode", "register"),
            ]),
        )
        .unwrap();
    assert_eq!(alias, canonical);
    assert_eq!(alias.benchmark_id, "cpu.performance.integer.i64");
    let invalid = config(&[("dataset_bytes", "1234")]);
    assert!(
        registry
            .resolve("cpu.performance.single_thread.integer.i64", &invalid)
            .is_err()
    );
}

#[test]
fn recorded_alias_keeps_requested_provenance_and_accepts_only_the_matching_canonical_result() {
    let mut registry = BenchmarkRegistry::default();
    registry.add(Box::new(Int64AliasProvider));
    let id = "cpu.performance.single_thread.integer.i64";
    let mut record = RunRecord::start(id, &BenchmarkConfig::default(), vec![device(false)]);
    record.resolved = Some(registry.resolve(id, &BenchmarkConfig::default()).unwrap());
    record.finish(Ok(summary()));
    record.validate().unwrap();
    assert_eq!(record.requested_benchmark_id, id);
    record.result.as_mut().unwrap().benchmark_id = "cpu.performance.float.f32".into();
    assert!(record.validate().is_err());
}

#[test]
fn phase_four_access_options_resolve_only_for_supported_buffered_modes() {
    let ascii = legacy_condition(&descriptor("cpu.performance.string.ascii"));
    let resolved = ascii
        .resolve(&BenchmarkConfig::default(), &[device(false)])
        .unwrap();
    assert_eq!(resolved.options["access_pattern"], "sequential");
    assert_eq!(resolved.options["access_realizations"], "1");
    let integer = legacy_condition(&descriptor("cpu.performance.integer.i64"));
    let register = integer
        .resolve(&BenchmarkConfig::default(), &[device(false)])
        .unwrap();
    assert!(!register.options.contains_key("access_seed"));
    assert!(
        integer
            .resolve(
                &config(&[("access_pattern", "shuffled_blocks")]),
                &[device(false)]
            )
            .is_err()
    );
    assert!(
        ascii
            .resolve(&config(&[("access_block_bytes", "65")]), &[device(false)])
            .is_err()
    );
    let ram = legacy_condition(&descriptor("cpu.bandwidth.memory"));
    assert!(
        ram.resolve(&config(&[("dataset_mode", "sweep")]), &[device(false)])
            .is_err()
    );
    let selected = ram
        .resolve(
            &config(&[
                ("dataset_mode", "single"),
                ("dataset_bytes", "1048576"),
                ("access_pattern", "shuffled_blocks"),
                ("access_realizations", "3"),
            ]),
            &[device(false)],
        )
        .unwrap();
    assert_eq!(selected.options["dataset_bytes"], "1048576");
    assert!(
        ram.resolve(&config(&[("ram_bandwidth_bytes", "1")]), &[device(false)])
            .is_err()
    );
}

#[test]
fn phase_five_modes_reject_inert_controls_and_record_actual_experiment_geometry() {
    for id in [
        "cpu.performance.float.f64",
        "cpu.performance.avx2.f64_fma",
        "cpu.performance.extended.popcnt",
    ] {
        let schema = legacy_condition(&descriptor(id));
        let key = cpu_data_preset_key(id);
        let register = schema
            .resolve(&BenchmarkConfig::default(), &[device(false)])
            .unwrap();
        assert_eq!(register.options["dataset_mode"], "register");
        let data = schema
            .resolve(
                &config(&[
                    (key, "custom"),
                    ("dataset_mode", "single"),
                    ("dataset_bytes", "98304"),
                    ("access_pattern", "shuffled_blocks"),
                    ("core_experiment", "fixed_aggregate"),
                ]),
                &[device(false)],
            )
            .unwrap();
        assert_eq!(data.options["dataset_bytes"], "98304");
        assert!(
            schema
                .resolve(&config(&[("ram_budget_mib", "64")]), &[device(false)])
                .is_err()
        );
        assert!(
            schema
                .resolve(
                    &config(&[("core_experiment", "fixed_aggregate")]),
                    &[device(false)]
                )
                .is_err()
        );
    }
    let matrix = legacy_condition(&descriptor("cpu.performance.matrix.fp32.scaling"));
    assert!(
        matrix
            .resolve(
                &config(&[("matrix_shape", "legacy"), ("matrix_reuse", "4")]),
                &[device(false)]
            )
            .is_err()
    );
    assert!(
        matrix
            .resolve(
                &config(&[("matrix_shape", "square"), ("dataset_bytes", "12288")]),
                &[device(false)]
            )
            .is_err()
    );
    assert!(
        matrix
            .resolve(
                &config(&[
                    ("matrix_shape", "square"),
                    ("matrix_reuse", "4"),
                    ("cpu_core_limit", "4")
                ]),
                &[device(false)]
            )
            .is_ok()
    );
    let codec = legacy_condition(&descriptor("cpu.performance.compression.deflate"));
    assert!(
        codec
            .resolve(&config(&[("access_pattern", "hot_cold")]), &[device(false)])
            .is_err()
    );
    assert!(
        codec
            .resolve(&config(&[("codec_level", "10")]), &[device(false)])
            .is_err()
    );
    assert!(
        codec
            .resolve(
                &config(&[("access_pattern", "shuffled_blocks"), ("codec_level", "9")]),
                &[device(false)]
            )
            .is_ok()
    );
}

#[test]
fn gpu_vector_precision_domains_preserve_registers_and_reject_unsupported_placement() {
    for id in [
        "gpu.performance.fp16",
        "gpu.performance.fp64",
        "gpu.performance.fp32.scaling",
    ] {
        let c = legacy_condition(&descriptor(id));
        for pattern in [
            "sequential",
            "shuffled_blocks",
            "rotating_regions",
            "lane_scattered",
        ] {
            let r = c
                .resolve(
                    &config(&[
                        ("gpu_preset", "custom"),
                        ("dataset_mode", "single"),
                        ("dataset_bytes", "983040"),
                        ("access_pattern", pattern),
                        ("access_realizations", "3"),
                        ("sustained_seconds", "1"),
                    ]),
                    &[device(true)],
                )
                .unwrap();
            assert_eq!(r.options["access_pattern"], pattern);
        }
        assert!(
            c.resolve(
                &config(&[
                    ("gpu_preset", "register"),
                    ("access_pattern", "lane_scattered")
                ]),
                &[device(true)]
            )
            .is_err()
        );
        assert!(
            c.resolve(&config(&[("ram_offload_percent", "75")]), &[device(true)])
                .is_err()
        );
        assert!(
            c.resolve(
                &config(&[("offload_passes_per_second", "100")]),
                &[device(true)]
            )
            .is_err()
        );
        if id != "gpu.performance.fp32.scaling" {
            assert_eq!(
                c.resolve(&config(&[]), &[device(true)]).unwrap().options["dataset_mode"],
                "register"
            );
        }
    }
}

#[test]
fn phase_seven_memory_transfer_and_matrix_controls_reject_inert_combinations() {
    let devices = [device(true)];
    for id in ["gpu.bandwidth.cache", "gpu.bandwidth.vram"] {
        let c = legacy_condition(&descriptor(id));
        for pattern in [
            "sequential",
            "shuffled_blocks",
            "rotating_regions",
            "lane_scattered",
        ] {
            let r = c
                .resolve(
                    &config(&[
                        ("dataset_mode", "sweep"),
                        ("dataset_bytes", "1048576"),
                        ("access_pattern", pattern),
                        ("memory_copy_path", "shader"),
                        ("access_realizations", "3"),
                        ("sustained_seconds", "15"),
                    ]),
                    &devices,
                )
                .unwrap();
            assert_eq!(r.options["access_pattern"], pattern);
        }
        assert!(
            c.resolve(
                &config(&[
                    ("memory_copy_path", "native"),
                    ("access_pattern", "lane_scattered")
                ]),
                &devices
            )
            .is_err()
        );
        assert!(
            c.resolve(&config(&[("dataset_mode", "single")]), &devices)
                .is_err()
        );
    }
    let c = legacy_condition(&descriptor("gpu.bandwidth.host_link"));
    for direction in ["both", "host_to_device", "device_to_host"] {
        let r = c
            .resolve(
                &config(&[
                    ("dataset_mode", "single"),
                    ("dataset_bytes", "1048576"),
                    ("transfer_direction", direction),
                    ("staging_buffers", "8"),
                    ("transfer_submission", "serial"),
                    ("transfer_batch", "1"),
                ]),
                &devices,
            )
            .unwrap();
        assert_eq!(r.options["transfer_direction"], direction);
    }
    assert!(
        c.resolve(
            &config(&[("transfer_submission", "serial"), ("transfer_batch", "8")]),
            &devices
        )
        .is_err()
    );
    assert!(
        c.resolve(&config(&[("staging_buffers", "2")]), &devices)
            .is_err()
    );
    let c = legacy_condition(&descriptor("gpu.performance.matrix.fp16.scaling"));
    assert!(
        c.resolve(
            &config(&[
                ("gpu_matrix_preset", "register"),
                ("matrix_traversal", "shuffled_tiles")
            ]),
            &devices
        )
        .is_err()
    );
    assert!(
        c.resolve(
            &config(&[
                ("gpu_matrix_preset", "custom"),
                ("dataset_bytes", "1048576"),
                ("matrix_traversal", "shuffled_tiles"),
                ("matrix_tile_reuse", "16"),
                ("access_realizations", "3")
            ]),
            &devices
        )
        .is_ok()
    );
}
