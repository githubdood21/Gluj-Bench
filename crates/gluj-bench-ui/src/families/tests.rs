use super::*;
use crate::configuration_tests::fixture;

pub(crate) fn catalogue() -> App {
    let mut app = fixture();
    let cpu = DeviceDescriptor {
        id: "cpu:system".into(),
        name: "CPU fixture".into(),
        category: DeviceCategory::Cpu,
        available: true,
        status: "ready".into(),
        properties: BTreeMap::from([("physical_cores".into(), "8".into())]),
        caches: vec![],
    };
    app.devices.push(cpu);
    let template = app.benchmarks[0].clone();
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
        CPU_INT64_ID,
        "cpu.performance.single_thread.integer.i64",
        "cpu.performance.float.f32",
        "cpu.performance.float.f64",
        CPU_VECTOR_ID,
        "cpu.performance.avx2.f64_fma",
        CPU_MATRIX_ID,
        CPU_ASCII_ID,
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
        GPU_VECTOR_ID,
        GPU_OFFLOAD_ID,
        GPU_MATRIX_ID,
        GPU_INT8_MATRIX_ID,
        "gpu.performance.matrix.fp8",
        "gpu.performance.matrix.sparse.fp16",
        "gpu.performance.matrix.sparse.int8",
        "gpu.performance.matrix.sparse.fp8",
    ];
    app.benchmarks = ids
        .iter()
        .enumerate()
        .map(|(order, id)| {
            let mut b = template.clone();
            b.id = (*id).into();
            b.name = (*id).into();
            b.display_order = order as u32;
            b.suite_id = if id.starts_with("cpu.performance.") {
                "cpu.performance"
            } else if id.starts_with("cpu.") {
                "cpu.bandwidth"
            } else if id.starts_with("gpu.performance.") {
                "gpu.performance"
            } else {
                "gpu.bandwidth"
            }
            .into();
            let gpu = id.starts_with("gpu.");
            b.category = if gpu {
                BenchmarkCategory::Gpu
            } else {
                BenchmarkCategory::Cpu
            };
            b.supported_device_ids = vec![if gpu { "gpu:fixture" } else { "cpu:system" }.into()];
            b.available = !implementation_detail(id);
            b.unavailable_reason = if b.available { "" } else { "runner pending" }.into();
            b
        })
        .collect();
    app.condition_descriptors = app
        .benchmarks
        .iter()
        .map(gluj_bench_core::legacy_condition)
        .collect();
    app
}
#[test]
fn compute_templates_follow_the_selected_cpu_or_gpu_category() {
    let mut app = catalogue();
    for (suite, prefix) in [
        ("gpu.performance", "gpu."),
        ("gpu.bandwidth", "gpu."),
        ("cpu.performance", "cpu."),
        ("cpu.bandwidth", "cpu."),
    ] {
        app.selected_suite = suite.into();
        app.run_task("compute");
        assert!(!app.custom_plan.is_empty(), "{}", app.status);
        assert!(
            app.custom_plan
                .iter()
                .all(|run| run.benchmark_id.starts_with(prefix)),
            "{suite} prepared a template for another processor"
        );
        assert_eq!(app.selected_suite, suite);
        assert_eq!(app.presentation.scope, 1);
    }
}

#[test]
fn unsupported_operations_exclude_supported_modes_and_other_categories() {
    let mut app = catalogue();
    app.selected_suite = "gpu.performance".into();
    let fp8 = app
        .benchmarks
        .iter_mut()
        .find(|benchmark| benchmark.id == "gpu.performance.matrix.fp8")
        .unwrap();
    fp8.available = true;
    fp8.unavailable_reason.clear();
    let details = app.capability_details();
    assert!(
        !details
            .lines()
            .any(|line| line.starts_with("gpu.performance.matrix.fp8:"))
    );
    assert!(details.contains("runner pending"));
    assert!(!details.contains("cpu."));
    app.selected_gpu_id = Some("gpu:unsupported".into());
    assert!(
        app.capability_details()
            .contains("unsupported on selected device")
    );
    app.selected_suite = "cpu.performance".into();
    assert!(app.capability_details().is_empty());
}

#[test]
fn fifteen_family_cards_preserve_all_implemented_catalogue_destinations() {
    let mut app = catalogue();
    let mut families = Vec::new();
    let mut covered = std::collections::BTreeSet::new();
    for suite in [
        "cpu.bandwidth",
        "cpu.performance",
        "gpu.bandwidth",
        "gpu.performance",
    ] {
        app.selected_suite = suite.into();
        families.extend(app.visible_families());
        let plan = app.curated_plan(None).unwrap();
        plan.suite.validate().unwrap();
        for c in plan.suite.conditions {
            covered.insert(c.condition.benchmark_id);
        }
    }
    families.sort();
    families.dedup();
    assert_eq!(families.len(), 15);
    for b in &app.benchmarks {
        if b.available && !retired(&b.id) {
            assert!(covered.contains(&b.id), "missing {}", b.id);
        }
    }
    assert_eq!(covered.len(), 30);
    assert!(
        !covered
            .iter()
            .any(|id| implementation_detail(id) || retired(id))
    );
    assert!(app.capability_details().contains("runner pending"));
}
#[test]
fn curated_plans_preserve_one_and_all_core_scalar_simd_and_codec_directions() {
    let mut app = catalogue();
    app.selected_suite = "cpu.performance".into();
    let plan = app.curated_plan(None).unwrap();
    for id in [
        CPU_INT64_ID,
        CPU_VECTOR_ID,
        "cpu.performance.float.f32",
        "cpu.performance.float.f64",
        "cpu.performance.avx2.f64_fma",
        "cpu.performance.compression.deflate",
        "cpu.performance.decompression.deflate",
    ] {
        let counts: std::collections::BTreeSet<_> = plan
            .suite
            .conditions
            .iter()
            .filter(|c| c.condition.benchmark_id == id)
            .map(|c| c.condition.options["cpu_core_limit"].as_str())
            .collect();
        assert_eq!(counts, std::collections::BTreeSet::from(["1", "8"]), "{id}");
    }
    assert_eq!(
        plan.suite
            .conditions
            .iter()
            .filter(|c| c.condition.benchmark_id == CPU_INT64_ID
                && c.condition.options["cpu_core_limit"] == "1"
                && c.condition.options["dataset_mode"] == "register")
            .count(),
        1
    );
    assert!(plan.details().contains("aggregate target"));
    assert!(plan.details().contains("/ worker"));
}
#[test]
fn custom_plans_deduplicate_equivalent_settings_without_merging_distinct_workloads() {
    let mut app = catalogue();
    app.selected_suite = "cpu.performance".into();
    let mut run = app.queued_run(CPU_INT64_ID).unwrap();
    run.settings.vector_preset = 1;
    run.settings.cpu_core_limit = 1;
    let mut other = run.clone();
    other.settings.cpu_core_limit = 8;
    let plan = app
        .prepare_plan(vec![run.clone(), run, other], "custom")
        .unwrap();
    assert_eq!(plan.runs.len(), 2);
    app.settings.shorter_run = true;
    let short = app.prepare_plan(plan.runs.into(), "custom").unwrap();
    assert!(
        short
            .suite
            .conditions
            .iter()
            .all(|c| c.condition.policy.samples == 3)
    );
}
#[test]
fn register_only_devices_hide_buffered_choices_and_keep_matrix_precision_conditions() {
    let mut app = catalogue();
    app.selected_suite = "gpu.performance".into();
    assert_eq!(
        app.supported_presets(GPU_MATRIX_ID, &app.settings)
            .iter()
            .map(|(i, _)| *i)
            .collect::<Vec<_>>(),
        vec![1]
    );
    let plan = app.curated_plan(Some("gpu.matrix")).unwrap();
    assert_eq!(plan.runs.len(), 2);
    assert!(
        plan.suite
            .conditions
            .iter()
            .all(|c| c.condition.options["dataset_mode"] == "register")
    );
    let b = app
        .benchmarks
        .iter()
        .find(|b| b.id == GPU_VECTOR_ID)
        .unwrap();
    let mut condition = app.condition_for(b);
    condition.capabilities = vec![gluj_bench_core::ModeCapability {
        when: BTreeMap::from([("dataset_mode".into(), "register".into())]),
        supported_device_ids: vec![],
        unavailable_reason: "register disabled".into(),
    }];
    app.condition_descriptors
        .retain(|c| c.benchmark_id != GPU_VECTOR_ID);
    app.condition_descriptors.push(condition);
    assert!(
        !app.supported_presets(GPU_VECTOR_ID, &app.settings)
            .iter()
            .any(|(i, _)| *i == 1)
    );
}
#[test]
fn version_one_single_thread_preferences_migrate_without_overriding_a_canonical_preference() {
    let mut app = catalogue();
    let mut custom = TestOverride {
        settings: AppSettings::default(),
        dataset_text: "256".into(),
    };
    custom.settings.cpu_intensity = 0;
    let legacy = "cpu.performance.single_thread.integer.i64";
    let mut preferences = BenchmarkPreferences::default();
    preferences.tests.insert(
        legacy.into(),
        SavedBenchmarkPreference {
            selected_option: None,
            custom: Some(custom.clone()),
        },
    );
    app.restore_benchmark_preferences(preferences.clone());
    assert_eq!(app.compute_choice(CPU_INT64_ID), 1);
    let migrated = app.queued_run(CPU_INT64_ID).unwrap();
    assert_eq!(migrated.settings.cpu_core_limit, 1);
    assert_eq!(migrated.settings.vector_preset, 1);
    assert_eq!(migrated.settings.cpu_intensity, 0);
    custom.settings.cpu_core_limit = 4;
    preferences.tests.insert(
        CPU_INT64_ID.into(),
        SavedBenchmarkPreference {
            selected_option: Some("custom".into()),
            custom: Some(custom.clone()),
        },
    );
    app.restore_benchmark_preferences(preferences);
    assert_eq!(app.run_overrides[CPU_INT64_ID], custom);
    let old: BenchmarkPreferences = serde_json::from_str(r#"{"version":1,"tests":{}}"#).unwrap();
    assert!(old.family_modes.is_empty());
}
#[test]
fn family_mode_preferences_round_trip_and_unavailable_modes_fall_back_safely() {
    let mut app = catalogue();
    app.select_family("cpu.fp64");
    app.select_family_mode(1);
    let saved = app.snapshot_benchmark_preferences();
    assert_eq!(
        saved.family_modes["cpu.fp64"],
        "cpu.performance.avx2.f64_fma"
    );
    let mut restored = catalogue();
    restored.restore_benchmark_preferences(
        serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap(),
    );
    restored.select_family("cpu.fp64");
    assert_eq!(
        restored.selected.as_deref(),
        Some("cpu.performance.avx2.f64_fma")
    );
    restored
        .benchmarks
        .iter_mut()
        .find(|b| b.id == "cpu.performance.avx2.f64_fma")
        .unwrap()
        .available = false;
    restored.select_family("cpu.fp64");
    assert_eq!(
        restored.selected.as_deref(),
        Some("cpu.performance.float.f64")
    );
}
#[test]
fn cancelling_one_condition_keeps_unstarted_conditions_pending_until_global_stop() {
    let mut app = catalogue();
    app.selected_suite = "cpu.performance".into();
    let mut suite = app.curated_plan(Some("cpu.fp64")).unwrap().suite;
    assert!(suite.conditions.len() > 1);
    suite.conditions[0].status = CompletionStatus::Running;
    app.active_plan = Some(suite);
    app.continue_after_cancel = true;
    app.finish_plan_condition(CompletionStatus::Cancelled);
    let plan = app.active_plan.as_ref().unwrap();
    assert_eq!(plan.conditions[0].status, CompletionStatus::Cancelled);
    assert!(
        plan.conditions[1..]
            .iter()
            .all(|c| c.status == CompletionStatus::Pending)
    );

    app.continue_after_cancel = false;
    app.active_plan.as_mut().unwrap().conditions[1].status = CompletionStatus::Running;
    app.finish_plan_condition(CompletionStatus::Cancelled);
    let plan = app.active_plan.as_ref().unwrap();
    assert_eq!(plan.conditions[1].status, CompletionStatus::Cancelled);
    assert!(
        plan.conditions[2..]
            .iter()
            .all(|c| c.status == CompletionStatus::Skipped)
    );
}

#[test]
fn a_failed_condition_marks_unstarted_plan_entries_skipped() {
    let mut app = catalogue();
    app.selected_suite = "cpu.performance".into();
    let mut suite = app.curated_plan(Some("cpu.fp64")).unwrap().suite;
    suite.conditions[0].status = CompletionStatus::Running;
    app.active_plan = Some(suite);
    app.finish_plan_condition(CompletionStatus::Failed);
    let plan = app.active_plan.as_ref().unwrap();
    assert_eq!(plan.conditions[0].status, CompletionStatus::Failed);
    assert!(
        plan.conditions[1..]
            .iter()
            .all(|c| c.status == CompletionStatus::Skipped)
    );
}

#[test]
fn scalar_one_and_all_core_results_are_distinct_visible_rows() {
    let make = |cores: &str| BenchmarkResult {
        benchmark_id: "cpu.performance.float.f64".into(),
        device_id: "cpu:system".into(),
        elapsed_ns: 1,
        metrics: vec![],
        workload_metadata: BTreeMap::from([
            ("cpu_core_limit".into(), cores.into()),
            ("thread_mode".into(), "physical_cores".into()),
        ]),
        device_metadata: BTreeMap::new(),
    };
    let results = vec![make("1"), make("8")];
    assert_eq!(component_results(&results, "cpu:system").len(), 2);
    assert!(result_configuration_label(&results[0]).contains("1 cores"));
    assert!(
        configuration_summary(&AppSettings::default(), "cpu.latency.memory")
            .contains("1 pinned CPU worker")
    );
}

#[test]
fn curated_access_is_standard_and_custom_defaults_stay_sequential() {
    let mut app = catalogue();
    app.selected_suite = "cpu.performance".into();
    let plan = app.curated_plan(Some("cpu.ascii")).unwrap();
    assert_eq!(plan.suite.conditions.len(), 2);
    for cores in ["1", "8"] {
        let rows: Vec<_> = plan
            .suite
            .conditions
            .iter()
            .filter(|c| c.condition.options["cpu_core_limit"] == cores)
            .collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].condition.options["access_pattern"], "sequential");
        assert_eq!(rows[0].condition.options["access_realizations"], "1");
    }
    let run = app.queued_run(CPU_ASCII_ID).unwrap();
    let b = app
        .benchmarks
        .iter()
        .find(|b| b.id == CPU_ASCII_ID)
        .unwrap();
    let config = resolved_ui_config(&app.condition_for(b), &run, &app.devices).unwrap();
    assert_eq!(config.options["access_pattern"], "sequential");
    assert_eq!(config.options["access_realizations"], "1");
}

#[test]
fn result_groups_preserve_access_seed_and_accounting_identity() {
    let mut a = BenchmarkResult {
        benchmark_id: CPU_VECTOR_ID.into(),
        device_id: "cpu:system".into(),
        elapsed_ns: 1,
        metrics: Vec::new(),
        workload_metadata: BTreeMap::new(),
        device_metadata: BTreeMap::new(),
    };
    a.workload_metadata
        .insert("access_pattern".into(), "sequential".into());
    let mut b = a.clone();
    b.workload_metadata
        .insert("access_pattern".into(), "shuffled_blocks".into());
    assert!(!same_run_configuration(&a, &b));
    assert!(result_configuration_label(&b).contains("shuffled blocks"));
    let mut cache = vec![a.clone()];
    remember_result(&mut cache, &b);
    assert_eq!(cache.len(), 2);
    let mut bandwidth = a;
    bandwidth.benchmark_id = CPU_RAM_BANDWIDTH_ID.into();
    let mut other = bandwidth.clone();
    other
        .workload_metadata
        .insert("access_pattern".into(), "shuffled_blocks".into());
    let mut cache = vec![bandwidth];
    remember_result(&mut cache, &other);
    assert_eq!(cache.len(), 2);
}

#[test]
fn curated_cache_access_uses_cache_line_blocks_to_avoid_identical_single_block_conditions() {
    let mut app = catalogue();
    app.selected_suite = "cpu.bandwidth".into();
    let plan = app.curated_plan(Some("cpu.memory_bandwidth")).unwrap();
    for c in plan
        .suite
        .conditions
        .iter()
        .filter(|c| c.condition.benchmark_id.starts_with("cpu.bandwidth.cache."))
    {
        assert_eq!(c.condition.options["access_block_bytes"], "64");
    }
}

#[test]
fn default_configuration_is_standard_and_custom_keeps_randomized_seed_coverage() {
    let mut app = catalogue();
    app.selected_suite = "cpu.performance".into();
    let curated = app.configured_plan(CPU_ASCII_ID).unwrap();
    assert_eq!(curated.suite.conditions.len(), 1);
    assert_eq!(
        curated.suite.conditions[0].condition.options["access_pattern"],
        "sequential"
    );
    assert_eq!(
        curated.suite.conditions[0].condition.options["access_realizations"],
        "1"
    );
    app.vector_choices.insert(CPU_ASCII_ID.into(), 1);
    app.run_overrides.insert(
        CPU_ASCII_ID.into(),
        TestOverride {
            settings: AppSettings {
                access_pattern: 1,
                access_ensemble: true,
                access_seed: "1234".into(),
                vector_preset: 6,
                vector_size: "64 KiB".into(),
                ..Default::default()
            },
            dataset_text: "256".into(),
        },
    );
    let custom = app.configured_plan(CPU_ASCII_ID).unwrap();
    assert_eq!(custom.suite.conditions.len(), 1);
    assert_eq!(
        custom.suite.conditions[0].condition.options["access_pattern"],
        "shuffled_blocks"
    );
    assert_eq!(
        custom.suite.conditions[0].condition.options["access_realizations"],
        "3"
    );
    assert_eq!(
        custom.suite.conditions[0].condition.options["access_seed"],
        "1234"
    );
}

#[test]
fn phase_five_curated_coverage_and_fixed_work_plans_preserve_supported_modes() {
    let mut app = catalogue();
    app.selected_suite = "cpu.performance".into();
    for id in [CPU_FP64_ID, CPU_FP64_SIMD_ID, CPU_POPCNT_ID] {
        let plan = app
            .curated_plan(Some(
                app.family_of(app.benchmarks.iter().find(|b| b.id == id).unwrap()),
            ))
            .unwrap();
        for mode in ["register", "single", "sweep"] {
            if mode == "single" {
                continue;
            }
            assert!(
                plan.suite
                    .conditions
                    .iter()
                    .any(|c| c.condition.benchmark_id == id
                        && c.condition.options["dataset_mode"] == mode)
            );
        }
        app.selected = Some(id.into());
        app.vector_choices.insert(id.into(), 1);
        app.run_overrides.insert(
            id.into(),
            TestOverride {
                settings: AppSettings {
                    vector_preset: 6,
                    vector_sweep: false,
                    vector_size: "96 KiB".into(),
                    ..Default::default()
                },
                dataset_text: "256".into(),
            },
        );
        let plan = app.core_experiment_plan().unwrap();
        assert_eq!(plan.suite.conditions.len(), 4);
        assert!(
            plan.suite
                .conditions
                .iter()
                .all(|c| c.condition.options["dataset_bytes"] == "98304"
                    && c.condition.options["core_experiment"] == "fixed_aggregate")
        );
    }
    app.selected = Some(CPU_MATRIX_ID.into());
    app.vector_choices.insert(CPU_MATRIX_ID.into(), 1);
    app.run_overrides.insert(
        CPU_MATRIX_ID.into(),
        TestOverride {
            settings: AppSettings {
                matrix_shape: 1,
                matrix_reuse: 1,
                vector_preset: 6,
                ..Default::default()
            },
            dataset_text: "256".into(),
        },
    );
    let plan = app.core_experiment_plan().unwrap();
    assert!(
        plan.suite
            .conditions
            .iter()
            .all(|c| c.condition.options["matrix_shape"] == "square"
                && !c.condition.options.contains_key("dataset_bytes"))
    );
    app.selected = Some("cpu.performance.prime.sieve".into());
    assert!(app.core_experiment_plan().is_err());
}

#[test]
fn legacy_fp64_and_popcnt_preferences_stay_register_and_explicit_shapes_ignore_hidden_sizes() {
    let mut app = catalogue();
    let old:BenchmarkPreferences=serde_json::from_value(json!({"version":1,"tests":{"cpu.performance.float.f64":{"custom":{"settings":{"cpu_core_limit":1,"cpu_intensity":0},"dataset_text":"256"}},"cpu.performance.extended.popcnt":{"custom":{"settings":{"cpu_core_limit":2},"dataset_text":"256"}}}})).unwrap();
    app.restore_benchmark_preferences(old);
    for id in [CPU_FP64_ID, CPU_POPCNT_ID] {
        let run = app.queued_run(id).unwrap();
        assert_eq!(run.settings.vector_preset, 1);
        let b = app.benchmarks.iter().find(|b| b.id == id).unwrap();
        let c = resolved_ui_config(&app.condition_for(b), &run, &app.devices).unwrap();
        assert_eq!(c.options["dataset_mode"], "register");
    }
    app.selected = Some(CPU_MATRIX_ID.into());
    app.vector_choices.insert(CPU_MATRIX_ID.into(), 1);
    app.run_overrides.insert(
        CPU_MATRIX_ID.into(),
        TestOverride {
            settings: AppSettings {
                matrix_shape: 1,
                vector_preset: 6,
                vector_size: "12 KiB".into(),
                ..Default::default()
            },
            dataset_text: "256".into(),
        },
    );
    assert!(app.configured_plan(CPU_MATRIX_ID).is_ok());
    let local = app.run_overrides.get_mut(CPU_MATRIX_ID).unwrap();
    local.settings.matrix_shape = 0;
    local.settings.matrix_reuse = 2;
    local.settings.matrix_block = 0;
    local.settings.vector_size = "256 MiB".into();
    assert!(app.configured_plan(CPU_MATRIX_ID).is_ok());
}

#[test]
fn phase_six_gpu_curated_conditions_use_standard_access_and_keep_precisions() {
    let mut app = catalogue();
    app.selected_suite = "gpu.performance".into();
    app.devices
        .iter_mut()
        .find(|d| d.id == "gpu:fixture")
        .unwrap()
        .properties
        .insert("storage_buffer_16bit_access".into(), "true".into());
    let plan = app.curated_plan(Some("gpu.vector")).unwrap();
    for id in [
        "gpu.performance.fp16",
        "gpu.performance.fp64",
        GPU_VECTOR_ID,
        GPU_OFFLOAD_ID,
    ] {
        let conditions = plan
            .suite
            .conditions
            .iter()
            .filter(|c| {
                c.condition.benchmark_id == id && c.condition.options["dataset_mode"] != "register"
            })
            .collect::<Vec<_>>();
        assert_eq!(conditions.len(), 1, "{id}");
        assert_eq!(
            app.configured_plan(id).unwrap().suite.conditions.len(),
            1,
            "configured curated {id}"
        );
        assert_eq!(
            conditions[0].condition.options["access_pattern"],
            "sequential"
        );
        for c in conditions {
            assert_eq!(c.condition.options["access_realizations"], "1");
        }
    }
    assert_eq!(
        plan.suite
            .conditions
            .iter()
            .filter(|c| c.condition.options["dataset_mode"] == "register")
            .count(),
        3
    );
    app.devices
        .iter_mut()
        .find(|d| d.id == "gpu:fixture")
        .unwrap()
        .properties
        .insert("storage_buffer_16bit_access".into(), "false".into());
    assert_eq!(
        app.supported_presets("gpu.performance.fp16", &app.settings)
            .iter()
            .map(|(i, _)| *i)
            .collect::<Vec<_>>(),
        vec![1]
    );
}
#[test]
fn phase_six_legacy_precision_preferences_remain_register_and_scattering_is_explicit() {
    let mut app = catalogue();
    let old:BenchmarkPreferences=serde_json::from_value(json!({"version":1,"tests":{"gpu.performance.fp16":{"custom":{"settings":{"gpu_intensity":0},"dataset_text":"256"}},"gpu.performance.fp64":{"custom":{"settings":{},"dataset_text":"256"}}}})).unwrap();
    app.restore_benchmark_preferences(old);
    for id in ["gpu.performance.fp16", "gpu.performance.fp64"] {
        let run = app.queued_run(id).unwrap();
        assert_eq!(run.settings.vector_preset, 1);
        let b = app.benchmarks.iter().find(|b| b.id == id).unwrap();
        assert_eq!(
            resolved_ui_config(&app.condition_for(b), &run, &app.devices)
                .unwrap()
                .options["dataset_mode"],
            "register"
        );
    }
    let run = QueuedRun {
        benchmark_id: GPU_VECTOR_ID.into(),
        settings: AppSettings {
            vector_preset: 6,
            vector_size: "960 KiB".into(),
            vector_sweep: false,
            access_pattern: 3,
            ..Default::default()
        },
        gpu_id: app.selected_gpu_id.clone(),
        physical_cores: 8,
    };
    assert_eq!(run_options(&run)["access_pattern"], "lane_scattered");
    let mut standard = run;
    standard.settings.access_pattern = 0;
    assert_eq!(run_options(&standard)["access_pattern"], "sequential");
}

#[test]
fn phase_seven_curated_memory_and_matrix_conditions_use_standard_traversals() {
    let mut app = catalogue();
    app.selected_suite = "gpu.bandwidth".into();
    let plan = app.curated_plan(Some("gpu.local_bandwidth")).unwrap();
    for id in ["gpu.bandwidth.cache", "gpu.bandwidth.vram"] {
        let rows = plan
            .suite
            .conditions
            .iter()
            .filter(|r| r.condition.benchmark_id == id)
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].condition.options["access_pattern"], "sequential");
        assert!(
            rows.iter()
                .all(|r| r.condition.options["access_realizations"] == "1"
                    && r.condition.options["memory_copy_path"] == "shader")
        );
    }
    let run = QueuedRun {
        benchmark_id: "gpu.bandwidth.host_link".into(),
        settings: AppSettings {
            dataset_mode: 2,
            dataset_mib: 4,
            transfer_direction: 1,
            staging_buffers: 2,
            transfer_submission: 1,
            transfer_batch: 2,
            ..Default::default()
        },
        gpu_id: app.selected_gpu_id.clone(),
        physical_cores: 8,
    };
    let b = app
        .benchmarks
        .iter()
        .find(|b| b.id == run.benchmark_id)
        .unwrap();
    let c = resolved_ui_config(&app.condition_for(b), &run, &app.devices).unwrap();
    assert_eq!(c.options["transfer_batch"], "1");
    assert_eq!(c.options["transfer_direction"], "host_to_device");
    assert_eq!(c.options["staging_buffers"], "8");
}

#[test]
fn curated_coverage_ignores_custom_randomness_across_categories() {
    let mut app = catalogue();
    app.settings.access_pattern = 1;
    app.settings.access_ensemble = true;
    app.settings.access_fresh = true;
    app.settings.access_seed = "1234".into();
    app.settings.gpu_matrix_traversal = 1;
    for suite in [
        "cpu.bandwidth",
        "cpu.performance",
        "gpu.bandwidth",
        "gpu.performance",
    ] {
        app.selected_suite = suite.into();
        let plan = app.curated_plan(None).unwrap();
        assert!(!plan.suite.conditions.is_empty());
        for row in &plan.suite.conditions {
            if let Some(pattern) = row.condition.options.get("access_pattern") {
                assert_eq!(pattern, "sequential", "{}", row.condition.benchmark_id);
            }
            if let Some(realizations) = row.condition.options.get("access_realizations") {
                assert_eq!(realizations, "1", "{}", row.condition.benchmark_id);
            }
            if let Some(traversal) = row.condition.options.get("matrix_traversal") {
                assert_eq!(traversal, "sequential");
            }
        }
    }
}
