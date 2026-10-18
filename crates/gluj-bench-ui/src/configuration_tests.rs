use super::*;

pub(super) fn fixture() -> App {
    let (_, events) = mpsc::channel();
    App {
        page: 1,
        worker: WorkerClient {
            child: None,
            input: None,
            events,
        },
        status: "Ready".into(),
        devices: vec![DeviceDescriptor {
            id: "gpu:fixture".into(),
            name: "Discrete GPU".into(),
            category: DeviceCategory::Gpu,
            available: true,
            status: "Ready".into(),
            properties: BTreeMap::from([
                ("device_type".into(), "DiscreteGpu".into()),
                (
                    "device_local_memory_bytes".into(),
                    (8 * 1024_u64.pow(3)).to_string(),
                ),
                (
                    "host_memory_heap_bytes".into(),
                    (64 * 1024_u64.pow(3)).to_string(),
                ),
                (
                    "max_storage_buffer_binding_size".into(),
                    (1024_u64.pow(3)).to_string(),
                ),
                ("min_storage_buffer_offset_alignment".into(), "256".into()),
            ]),
            caches: vec![],
        }],
        benchmarks: vec![BenchmarkDescriptor {
            id: "gpu.performance.fp32.offload.scaling".into(),
            name: "FP32 system-RAM offload scaling".into(),
            category: BenchmarkCategory::Gpu,
            workload:
                "GPU calculation with a configurable share of input/output data in system RAM"
                    .into(),
            data_type: "fp32".into(),
            unit: "operations/s".into(),
            supported_device_ids: vec!["gpu:fixture".into()],
            available: true,
            unavailable_reason: String::new(),
            suite_id: "gpu.performance".into(),
            display_order: 104,
            metadata: BTreeMap::new(),
        }],
        selected_suite: "gpu.performance".into(),
        selected_gpu_id: Some("gpu:fixture".into()),
        selected: Some("gpu.performance.fp32.offload.scaling".into()),
        expanded_device_id: None,
        result_details_expanded: false,
        snapshot: snapshot::SnapshotState::default(),
        result_presentations: RefCell::default(),
        results: vec![],
        saved_results: vec![],
        baseline_results: vec![],
        comparison_selection: 0,
        results_file_status: String::new(),
        results_file_writable: false,
        settings: AppSettings::default(),
        settings_status: "Global settings apply unless a test override is enabled.".into(),
        settings_writable: false,
        result_component: 0,
        result_selection: 0,
        queue: VecDeque::new(),
        run_overrides: BTreeMap::new(),
        vector_choices: BTreeMap::new(),
        benchmark_preferences: BenchmarkPreferences::default(),
        benchmark_preferences_path: std::env::temp_dir().join("unused-benchmark-settings.json"),
        benchmark_preferences_writable: false,
        benchmark_preferences_status: String::new(),
        dataset_text: "256".into(),
        active_request: None,
        progress: 0.,
        progress_message: String::new(),
        request_counter: 0,
        fingerprint: None,
        metadata_scan_pending: false,
        metadata_devices_received: false,
        metadata_benchmarks_received: false,
        ui_dirty: true,
    }
}

#[test]
fn saved_scaling_configurations_restore_the_latest_three_rows() {
    let path = std::env::temp_dir().join(format!(
        "scaling-results-{}.json",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut saved = vec![];
    for i in 1..=5 {
        let result = BenchmarkResult {
            benchmark_id: GPU_VECTOR_ID.into(),
            device_id: "gpu:fixture".into(),
            elapsed_ns: i,
            metrics: vec![],
            workload_metadata: BTreeMap::from([
                ("gpu_preset".into(), "custom".into()),
                ("dataset_mode".into(), "single".into()),
                ("requested_dataset_bytes".into(), (i * 1048576).to_string()),
            ]),
            device_metadata: BTreeMap::new(),
        };
        remember_result(&mut saved, &result);
    }
    save_results(&path, &saved).unwrap();
    let restored = load_results(&path).unwrap();
    fs::remove_file(&path).unwrap();
    let groups = component_results(&restored, "gpu:fixture");
    assert_eq!(groups.len(), 3);
    assert_eq!(
        groups.iter().map(|g| g[0].elapsed_ns).collect::<Vec<_>>(),
        [5, 4, 3]
    );
    assert!(result_configuration_label(groups[0][0]).contains("Custom"));
    assert!(result_configuration_label(groups[0][0]).contains("5.00 MiB"));
}

#[test]
fn configurable_results_keep_three_latest_configurations_and_repeat_statistics() {
    let make = |id: &str, device: &str, size: u64, speed: f64| BenchmarkResult {
        benchmark_id: id.into(),
        device_id: device.into(),
        elapsed_ns: 1,
        metrics: vec![Metric {
            name: format!("working_set_{size}.compute"),
            value: speed,
            unit: "operations/s".into(),
            statistics: Default::default(),
        }],
        workload_metadata: BTreeMap::from([
            ("dataset_mode".into(), "single".into()),
            ("requested_dataset_bytes".into(), size.to_string()),
        ]),
        device_metadata: BTreeMap::new(),
    };
    for id in [
        CPU_VECTOR_ID,
        CPU_MATRIX_ID,
        CPU_ASCII_ID,
        CPU_INT64_ID,
        GPU_VECTOR_ID,
        GPU_MATRIX_ID,
        GPU_INT8_MATRIX_ID,
        GPU_OFFLOAD_ID,
    ] {
        let mut results = (1..=4)
            .map(|i| make(id, "gpu:fixture", i * 1048576, i as f64))
            .collect::<Vec<_>>();
        let groups = component_results(&results, "gpu:fixture");
        assert_eq!(groups.len(), 3, "{id}");
        assert_eq!(
            groups
                .iter()
                .map(|g| g[0].workload_metadata["requested_dataset_bytes"].as_str())
                .collect::<Vec<_>>(),
            ["4194304", "3145728", "2097152"]
        );
        results.push(make(id, "gpu:fixture", 2 * 1048576, 0.5));
        let groups = component_results(&results, "gpu:fixture");
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[0][0].metrics[0].value, 0.5);
        assert_eq!(best_primary(&groups[0]).unwrap().value, 2.0);
        results.push(make(id, "gpu:fixture", 1048576, 5.0));
        results.push(make(
            "cpu.performance.extended.popcnt",
            "gpu:fixture",
            1,
            1.0,
        ));
        results.push(make(id, "gpu:other", 1048576, 1.0));
        let groups = component_results(&results, "gpu:fixture");
        let configured = groups
            .iter()
            .filter(|g| g[0].benchmark_id == id)
            .collect::<Vec<_>>();
        assert_eq!(configured.len(), 3);
        assert_eq!(configured[0][0].metrics[0].value, 5.0);
        assert_eq!(configured[1][0].metrics[0].value, 0.5);
        assert_eq!(configured[2][0].metrics[0].value, 4.0);
        let mut saved = vec![];
        for result in &results {
            remember_result(&mut saved, result);
        }
        assert_eq!(component_results(&saved, "gpu:fixture").len(), 4);
    }
}

#[test]
fn configuration_rows_distinguish_modes_workers_activity_and_ram_shares() {
    let mut base = BenchmarkResult {
        benchmark_id: GPU_OFFLOAD_ID.into(),
        device_id: "gpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![],
        workload_metadata: BTreeMap::from([
            ("dataset_mode".into(), "single".into()),
            ("requested_dataset_bytes".into(), "1048576".into()),
            ("ram_offload_percent".into(), "50".into()),
        ]),
        device_metadata: BTreeMap::new(),
    };
    let mut sweep = base.clone();
    sweep
        .workload_metadata
        .insert("dataset_mode".into(), "sweep".into());
    let mut full_ram = base.clone();
    full_ram
        .workload_metadata
        .insert("ram_offload_percent".into(), "100".into());
    let mut activity = base.clone();
    activity
        .workload_metadata
        .insert("gpu_activity_percent".into(), "95".into());
    assert_eq!(
        component_results(&[base.clone(), sweep, full_ram, activity], "gpu:fixture").len(),
        3
    );
    assert!(result_configuration_label(&base).contains("One size: 1.00 MiB"));
    base.benchmark_id = CPU_VECTOR_ID.into();
    base.device_id = "cpu:system".into();
    base.workload_metadata
        .insert("vector_preset".into(), "register".into());
    base.workload_metadata
        .insert("cpu_core_limit".into(), "4".into());
    assert!(result_configuration_label(&base).contains("Register compute"));
    assert!(result_configuration_label(&base).contains("4 cores"));
    let mut other = base.clone();
    other
        .workload_metadata
        .insert("cpu_core_limit".into(), "8".into());
    assert_eq!(component_results(&[base, other], "cpu:system").len(), 2);
}

#[test]
fn offload_legacy_preferences_keep_payload_share_and_global_fallback() {
    let path = std::env::temp_dir().join(format!(
        "offload-prefs-{}.json",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let data = json!({"version":1,"tests":{"gpu.performance.fp32.offload50.scaling":{"custom":{"settings":{"vector_preset":1,"dataset_mode":2,"dataset_mib":128,"gpu_intensity":0},"dataset_text":"128"}}}});
    fs::write(&path, serde_json::to_vec(&data).unwrap()).unwrap();
    let prefs = load_benchmark_preferences(&path).unwrap();
    fs::remove_file(&path).unwrap();
    let mut app = fixture();
    app.restore_benchmark_preferences(prefs);
    assert_eq!(app.compute_choice(GPU_OFFLOAD_ID), 2);
    let options = run_options(&app.queued_run(GPU_OFFLOAD_ID).unwrap());
    assert_eq!(options["ram_offload_percent"], "50");
    assert_eq!(options["dataset_mode"], "single");
    assert_eq!(options["dataset_bytes"], (128 * 1048576_u64).to_string());
    assert_eq!(options["gpu_activity_percent"], "95");
    assert!(!options.contains_key("gpu_offload_preset"));
    app.select_vector_choice(0);
    let selected = app.snapshot_benchmark_preferences();
    assert_eq!(
        selected.tests[GPU_OFFLOAD_ID].selected_option.as_deref(),
        Some("single_dataset")
    );
    write_json(&path, &selected).unwrap();
    let round_trip = load_benchmark_preferences(&path).unwrap();
    fs::remove_file(&path).unwrap();
    app.restore_benchmark_preferences(round_trip);
    assert_eq!(app.compute_choice(GPU_OFFLOAD_ID), 0);
}

#[test]
fn offload_choices_default_to_sweep_and_restore_custom_split_and_slider() {
    let mut app = fixture();
    assert_eq!(app.compute_choice(GPU_OFFLOAD_ID), 1);
    let sweep = app.queued_run(GPU_OFFLOAD_ID).unwrap();
    let options = run_options(&sweep);
    assert_eq!(options["gpu_offload_preset"], "budget");
    assert_eq!(options["dataset_mode"], "sweep");
    assert_eq!(options["ram_offload_percent"], "75");
    assert!(!options.contains_key("gpu_preset"));
    app.select_vector_choice(0);
    assert_eq!(
        run_options(&app.queued_run(GPU_OFFLOAD_ID).unwrap())["dataset_mode"],
        "single"
    );
    app.select_vector_choice(2);
    app.edit_configuration(true, "offload", 2);
    app.edit_vector_slider(0.0);
    let custom = app.queued_run(GPU_OFFLOAD_ID).unwrap();
    assert_eq!(run_options(&custom)["dataset_bytes"], "294912");
    assert_eq!(run_options(&custom)["ram_offload_percent"], "100");
    app.edit_configuration(true, "offload-preset", 1);
    assert_eq!(
        run_options(&app.queued_run(GPU_OFFLOAD_ID).unwrap())["gpu_offload_preset"],
        "small"
    );
    app.edit_configuration(true, "offload-preset", 3);
    app.select_vector_choice(1);
    let prefs = app.snapshot_benchmark_preferences();
    assert_eq!(
        prefs.tests[GPU_OFFLOAD_ID].selected_option.as_deref(),
        Some("data_sizes")
    );
    let mut reopened = fixture();
    reopened.restore_benchmark_preferences(prefs);
    assert_eq!(reopened.compute_choice(GPU_OFFLOAD_ID), 1);
    assert_eq!(reopened.compute_choice(GPU_VECTOR_ID), 0);
    reopened.select_vector_choice(2);
    assert_eq!(
        reopened.queued_run(GPU_OFFLOAD_ID).unwrap().settings,
        custom.settings
    );
    reopened.edit_vector_slider(1000.0);
    let settings = reopened.queued_run(GPU_OFFLOAD_ID).unwrap().settings;
    let range = reopened.compute_slider_range(GPU_OFFLOAD_ID, &settings);
    assert_eq!(
        vector_size_bytes(&settings.vector_size).unwrap(),
        range.maximum
    );
    assert_eq!(range.maximum % range.alignment, 0);
    reopened.reset_vector_configuration();
    assert_eq!(reopened.compute_choice(GPU_OFFLOAD_ID), 1);
    assert_eq!(run_options(&sweep)["dataset_mode"], "sweep");
    let invalid = AppSettings {
        vector_preset: 1,
        ..Default::default()
    };
    assert!(validated_run_settings(&invalid, "256", GPU_OFFLOAD_ID).is_err());
}

#[test]
fn vector_presets_units_and_queue_snapshots_are_isolated() {
    assert_eq!(vector_size_bytes("64 KiB").unwrap(), 65536);
    assert_eq!(vector_size_bytes("8 GiB").unwrap(), 8 * 1024_u64.pow(3));
    assert_eq!(vector_size_bytes("256").unwrap(), 256 * 1048576);
    for invalid in [
        "",
        "0 GiB",
        "-1 MiB",
        "1.5 GiB",
        "8 GB",
        "1 MiB extra",
        "18446744073709551615 GiB",
    ] {
        assert!(vector_size_bytes(invalid).is_err(), "{invalid}");
    }
    let mut app = fixture();
    app.selected = Some(CPU_VECTOR_ID.into());
    let mut settings = app.settings.clone();
    settings.vector_preset = 6;
    settings.vector_size = "8 GiB".into();
    settings.vector_sweep = false;
    // Unused global dataset text must not block a selected vector preset.
    settings.dataset_mode = 2;
    app.run_overrides.insert(
        CPU_VECTOR_ID.into(),
        TestOverride {
            settings,
            dataset_text: String::new(),
        },
    );
    let queued = app.queued_run(CPU_VECTOR_ID).unwrap();
    let selected_bytes = vector_size_bytes(&queued.settings.vector_size).unwrap();
    assert!(selected_bytes <= 8 * 1024_u64.pow(3));
    app.edit_configuration(true, "vector-preset", 1);
    assert_eq!(
        run_options(&queued)["dataset_bytes"],
        selected_bytes.to_string()
    );
    assert_eq!(run_options(&queued)["dataset_mode"], "single");
    assert_eq!(
        run_options(&app.queued_run(CPU_VECTOR_ID).unwrap())["dataset_mode"],
        "register"
    );
    assert!(!run_options(&app.queued_run(CPU_VECTOR_ID).unwrap()).contains_key("dataset_bytes"));
    assert!(
        !run_options(
            &app.queued_run("cpu.performance.matrix.fp32.scaling")
                .unwrap()
        )
        .contains_key("vector_preset")
    );
    let config = app.run_overrides.get_mut(CPU_VECTOR_ID).unwrap();
    config.settings.vector_preset = 6;
    config.settings.vector_size = "0 MiB".into();
    assert!(app.queued_run(CPU_VECTOR_ID).is_err());
}

#[test]
fn automatic_choices_use_full_cpu_allocation_and_preserve_custom_settings() {
    let mut app = fixture();
    app.selected = Some(CPU_VECTOR_ID.into());
    app.devices[0]
        .properties
        .insert("physical_cores".into(), "8".into());
    app.settings.cpu_core_limit = 2;
    app.settings.dataset_mode = 2;
    app.dataset_text = "invalid".into();
    assert_eq!(app.vector_choice(), 0);
    assert_eq!(
        run_options(&app.queued_run(CPU_VECTOR_ID).unwrap())["vector_preset"],
        "register"
    );
    app.select_vector_choice(1);
    let sweep = app.queued_run(CPU_VECTOR_ID).unwrap();
    assert_eq!(run_options(&sweep)["vector_preset"], "ram");
    assert_eq!(run_options(&sweep)["dataset_mode"], "sweep");
    assert_eq!(run_options(&sweep)["cpu_core_limit"], "8");
    app.select_vector_choice(0);
    let register = app.queued_run(CPU_VECTOR_ID).unwrap();
    assert_eq!(run_options(&register)["vector_preset"], "register");
    assert!(!run_options(&register).contains_key("dataset_bytes"));
    assert_eq!(run_options(&register)["cpu_core_limit"], "8");
    app.dataset_text = "256".into();
    app.select_vector_choice(2);
    app.edit_configuration(true, "vector-preset", 6);
    app.edit_configuration(true, "vector-mode", 0);
    app.edit_configuration(true, "cores", 3);
    app.edit_vector_slider(300.0);
    let custom = app.queued_run(CPU_VECTOR_ID).unwrap();
    app.select_vector_choice(1);
    assert_eq!(
        app.queued_run(CPU_VECTOR_ID)
            .unwrap()
            .settings
            .cpu_core_limit,
        0
    );
    app.select_vector_choice(2);
    let restored = app.queued_run(CPU_VECTOR_ID).unwrap();
    assert_eq!(restored.settings.cpu_core_limit, 3);
    assert_eq!(restored.settings.vector_size, custom.settings.vector_size);
    assert_eq!(run_options(&restored)["dataset_mode"], "single");
    assert_eq!(run_options(&sweep)["dataset_mode"], "sweep");
    app.active_request = Some("running".into());
    app.select_vector_choice(0);
    assert_eq!(app.vector_choice(), 2);
}

#[test]
fn benchmark_preferences_restore_choices_and_custom_settings_across_launches() {
    let directory = std::env::temp_dir().join(format!(
        "gluj-benchmark-preferences-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&directory).unwrap();
    let path = directory.join("benchmark-settings.json");
    assert_eq!(
        load_benchmark_preferences(&path).unwrap(),
        BenchmarkPreferences::default()
    );
    let mut app = fixture();
    app.benchmark_preferences_path = path.clone();
    app.benchmark_preferences_writable = true;
    app.selected = Some(CPU_VECTOR_ID.into());
    app.devices[0]
        .properties
        .insert("physical_cores".into(), "8".into());
    assert_eq!(app.vector_choice(), 0);
    app.select_vector_choice(2);
    app.edit_configuration(true, "cores", 3);
    app.edit_configuration(true, "vector-mode", 0);
    app.edit_vector_slider(350.0);
    let custom = app.run_overrides[CPU_VECTOR_ID].clone();
    let queued = app.queued_run(CPU_VECTOR_ID).unwrap();
    app.select_vector_choice(1);
    let gpu = "gpu.performance.fp32.offload.scaling";
    app.selected = Some(gpu.into());
    let settings = AppSettings {
        dataset_mode: 2,
        dataset_mib: 128,
        ..Default::default()
    };
    app.run_overrides.insert(
        gpu.into(),
        TestOverride {
            settings,
            dataset_text: "128".into(),
        },
    );
    app.edit_configuration(true, "offload", 2);
    let disk = load_benchmark_preferences(&path).unwrap();
    assert_eq!(
        disk.tests[CPU_VECTOR_ID].selected_option.as_deref(),
        Some("data_sizes")
    );
    assert_eq!(disk.tests[CPU_VECTOR_ID].custom.as_ref().unwrap(), &custom);
    assert_eq!(
        disk.tests[gpu]
            .custom
            .as_ref()
            .unwrap()
            .settings
            .ram_offload_percent,
        100
    );
    let mut reopened = fixture();
    reopened.benchmark_preferences_path = path.clone();
    reopened.benchmark_preferences_writable = true;
    reopened.devices[0]
        .properties
        .insert("physical_cores".into(), "8".into());
    reopened.restore_benchmark_preferences(disk);
    assert_eq!(reopened.vector_choice(), 1);
    assert_eq!(reopened.run_overrides[CPU_VECTOR_ID], custom);
    assert_eq!(reopened.effective_settings(gpu).unwrap().dataset_mib, 128);
    reopened.selected = Some(CPU_VECTOR_ID.into());
    reopened.select_vector_choice(2);
    assert_eq!(
        reopened.queued_run(CPU_VECTOR_ID).unwrap().settings,
        queued.settings
    );
    reopened.select_vector_choice(0);
    assert_eq!(
        load_benchmark_preferences(&path).unwrap().tests[CPU_VECTOR_ID]
            .selected_option
            .as_deref(),
        Some("compute_speed")
    );
    reopened.reset_vector_configuration();
    let reset = load_benchmark_preferences(&path).unwrap();
    assert!(!reset.tests.contains_key(CPU_VECTOR_ID));
    assert!(reset.tests.contains_key(gpu));
    assert_eq!(reopened.vector_choice(), 0);
    assert_eq!(queued.settings.cpu_core_limit, 3);
    fs::remove_file(path).unwrap();
    fs::remove_dir(directory).unwrap();
}

#[test]
fn invalid_edits_keep_the_last_valid_saved_configuration() {
    let directory = std::env::temp_dir().join(format!(
        "gluj-benchmark-valid-edit-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&directory).unwrap();
    let path = directory.join("benchmark-settings.json");
    let mut app = fixture();
    app.benchmark_preferences_path = path.clone();
    app.benchmark_preferences_writable = true;
    let id = app.selected.clone().unwrap();
    let settings = AppSettings {
        dataset_mode: 2,
        dataset_mib: 128,
        ..Default::default()
    };
    app.run_overrides.insert(
        id.clone(),
        TestOverride {
            settings,
            dataset_text: "128".into(),
        },
    );
    app.persist_benchmark_preferences();
    let before = fs::read(&path).unwrap();
    app.edit_dataset(true, String::new());
    assert!(app.queued_run(&id).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    app.edit_dataset(true, "64".into());
    assert_eq!(
        load_benchmark_preferences(&path).unwrap().tests[&id]
            .custom
            .as_ref()
            .unwrap()
            .settings
            .dataset_mib,
        64
    );
    app.run_overrides.remove(&id);
    app.persist_benchmark_preferences();
    assert!(load_benchmark_preferences(&path).unwrap().tests.is_empty());
    fs::remove_file(path).unwrap();
    fs::remove_dir(directory).unwrap();
}

#[test]
fn invalid_preferences_are_preserved_and_write_failures_keep_session_choices() {
    let directory = std::env::temp_dir().join(format!(
        "gluj-benchmark-invalid-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&directory).unwrap();
    let path = directory.join("benchmark-settings.json");
    for invalid in [
        "broken json",
        "{\"version\":99,\"tests\":{}}",
        "{\"version\":1,\"tests\":{\"cpu.performance.avx2.f32_fma.scaling\":{\"selected_option\":\"unknown\"}}}",
    ] {
        fs::write(&path, invalid).unwrap();
        assert!(load_benchmark_preferences(&path).is_err());
        let mut app = fixture();
        app.benchmark_preferences_path = path.clone();
        app.selected = Some(CPU_VECTOR_ID.into());
        app.select_vector_choice(1);
        assert_eq!(fs::read_to_string(&path).unwrap(), invalid);
        assert_eq!(app.vector_choice(), 1);
    }
    let mut app = fixture();
    app.selected = Some(CPU_VECTOR_ID.into());
    app.benchmark_preferences_path = directory
        .join("missing-parent")
        .join("benchmark-settings.json");
    app.benchmark_preferences_writable = true;
    app.select_vector_choice(1);
    assert_eq!(app.vector_choice(), 1);
    assert!(
        app.benchmark_preferences_status
            .contains("could not be saved")
    );
    fs::remove_file(path).unwrap();
    fs::remove_dir(directory).unwrap();
}

#[test]
fn different_ram_bandwidth_payloads_remain_separate_in_history() {
    let mut result = BenchmarkResult {
        benchmark_id: CPU_RAM_BANDWIDTH_ID.into(),
        device_id: "memory:system".into(),
        elapsed_ns: 1,
        metrics: vec![],
        workload_metadata: BTreeMap::from([("payload_bytes".into(), "268435456".into())]),
        device_metadata: BTreeMap::new(),
    };
    let old = result.clone();
    result
        .workload_metadata
        .insert("actual_payload_bytes".into(), "268435456".into());
    assert!(same_run_configuration(&old, &result));
    result
        .workload_metadata
        .insert("actual_payload_bytes".into(), "536870912".into());
    assert!(!same_run_configuration(&old, &result));
    let mut history = vec![old];
    remember_result(&mut history, &result);
    assert_eq!(history.len(), 2);
}

#[test]
fn retired_ram_size_preferences_do_not_override_the_default_payload() {
    let preferences: BenchmarkPreferences = serde_json::from_value(json!({
        "version": 1,
        "tests": { "cpu.bandwidth.memory": { "custom": {
            "settings": { "ram_bandwidth_custom": true, "ram_bandwidth_bytes": 1, "cpu_intensity": 0 },
            "dataset_text": "256"
        } } }
    })).unwrap();
    let mut app = fixture();
    app.restore_benchmark_preferences(preferences);
    let queued = app.queued_run(CPU_RAM_BANDWIDTH_ID).unwrap();
    let options = run_options(&queued);
    assert_eq!(options["cpu_activity_percent"], "95");
    for field in [
        "ram_bandwidth_mode",
        "ram_bandwidth_bytes",
        "dataset_mode",
        "dataset_bytes",
    ] {
        assert!(!options.contains_key(field));
    }
    let saved = serde_json::to_value(app.snapshot_benchmark_preferences()).unwrap();
    let settings = &saved["tests"][CPU_RAM_BANDWIDTH_ID]["custom"]["settings"];
    assert!(settings.get("ram_bandwidth_custom").is_none());
    assert!(settings.get("ram_bandwidth_bytes").is_none());
}

#[test]
fn gpu_fp32_choices_sizes_and_adapter_snapshots_are_independent() {
    let mut app = fixture();
    app.selected = Some(GPU_VECTOR_ID.into());
    app.devices[0].properties = BTreeMap::from([
        (
            "device_local_memory_bytes".into(),
            (8 * 1024_u64.pow(3)).to_string(),
        ),
        (
            "max_storage_buffer_binding_size".into(),
            (1024_u64.pow(3)).to_string(),
        ),
        ("min_storage_buffer_offset_alignment".into(), "256".into()),
    ]);
    let defaults = run_options(&app.queued_run(GPU_VECTOR_ID).unwrap());
    assert_eq!(defaults["gpu_preset"], "register");
    assert_eq!(defaults["dataset_mode"], "register");
    assert!(!defaults.contains_key("dataset_bytes"));
    assert!(!defaults.contains_key("vector_preset"));
    app.select_vector_choice(1);
    assert_eq!(
        run_options(&app.queued_run(GPU_VECTOR_ID).unwrap())["gpu_preset"],
        "vram"
    );
    app.select_vector_choice(2);
    app.edit_vector_slider(0.0);
    let queued = app.queued_run(GPU_VECTOR_ID).unwrap();
    assert_eq!(run_options(&queued)["dataset_bytes"], "270336");
    app.edit_configuration(true, "gpu-preset", 2);
    assert_eq!(
        run_options(&app.queued_run(GPU_VECTOR_ID).unwrap())["gpu_preset"],
        "small"
    );
    app.edit_configuration(true, "gpu-preset", 4);
    app.edit_vector_slider(1000.0);
    let range = app.compute_slider_range(
        GPU_VECTOR_ID,
        &app.queued_run(GPU_VECTOR_ID).unwrap().settings,
    );
    assert_eq!(range.bytes(1000.0), range.maximum);
    assert_eq!(range.maximum % range.alignment, 0);
    let custom = app.queued_run(GPU_VECTOR_ID).unwrap();
    app.select_vector_choice(0);
    let mut reopened = fixture();
    reopened.devices = app.devices.clone();
    reopened.restore_benchmark_preferences(app.snapshot_benchmark_preferences());
    assert_eq!(reopened.compute_choice(GPU_VECTOR_ID), 0);
    assert_eq!(reopened.compute_choice(CPU_VECTOR_ID), 0);
    reopened.selected = Some(GPU_VECTOR_ID.into());
    reopened.select_vector_choice(2);
    assert_eq!(
        reopened.queued_run(GPU_VECTOR_ID).unwrap().settings,
        custom.settings
    );
    reopened.selected_gpu_id = Some("gpu:other".into());
    assert!(reopened.queued_run(GPU_VECTOR_ID).is_err());
    assert_eq!(run_options(&custom)["device_id"], "gpu:fixture");
    assert_eq!(run_options(&queued)["dataset_bytes"], "270336");
    reopened.reset_vector_configuration();
    assert_eq!(reopened.compute_choice(GPU_VECTOR_ID), 0);
}

#[test]
fn gpu_matrix_choices_use_operand_limits_and_save_independently() {
    let mut app = fixture();
    app.selected = Some(GPU_MATRIX_ID.into());
    app.devices[0].properties = BTreeMap::from([
        (
            "device_local_memory_bytes".into(),
            (8 * 1024_u64.pow(3)).to_string(),
        ),
        (
            "max_storage_buffer_binding_size".into(),
            (1024_u64.pow(3)).to_string(),
        ),
        ("cooperative_matrix_fp16_shape".into(), "16x16x16".into()),
        ("vulkan_subgroup_size".into(), "32".into()),
        ("storage_buffer_16bit_access".into(), "true".into()),
    ]);
    let defaults = run_options(&app.queued_run(GPU_MATRIX_ID).unwrap());
    assert_eq!(defaults["gpu_matrix_preset"], "register");
    assert!(!defaults.contains_key("gpu_preset"));
    app.select_vector_choice(1);
    let sweep = run_options(&app.queued_run(GPU_MATRIX_ID).unwrap());
    assert_eq!(sweep["gpu_matrix_preset"], "vram");
    assert_eq!(sweep["dataset_mode"], "sweep");
    app.select_vector_choice(2);
    app.edit_vector_slider(0.0);
    let custom = app.queued_run(GPU_MATRIX_ID).unwrap();
    assert_eq!(run_options(&custom)["dataset_bytes"], "262144");
    let range = app.compute_slider_range(GPU_MATRIX_ID, &custom.settings);
    assert_eq!(range.alignment, 1024);
    assert_eq!(range.bytes(1000.0), range.maximum);
    assert_eq!(
        range.maximum,
        (gluj_bench_core::vram_budget_bytes(8 * 1024_u64.pow(3), 25) - 32 * 1024 * 1024) / 1024
            * 1024
    );
    let mut reopened = fixture();
    reopened.devices = app.devices.clone();
    reopened.restore_benchmark_preferences(app.snapshot_benchmark_preferences());
    assert_eq!(reopened.compute_choice(GPU_MATRIX_ID), 2);
    assert_eq!(reopened.compute_choice(GPU_VECTOR_ID), 0);
    assert_eq!(
        reopened.queued_run(GPU_MATRIX_ID).unwrap().settings,
        custom.settings
    );
    reopened.devices[0]
        .properties
        .remove("storage_buffer_16bit_access");
    assert!(
        !reopened
            .compute_slider_range(GPU_MATRIX_ID, &custom.settings)
            .available()
    );
    assert!(reopened.queued_run(GPU_MATRIX_ID).is_err());
    reopened
        .run_overrides
        .get_mut(GPU_MATRIX_ID)
        .unwrap()
        .settings
        .vector_preset = 2;
    assert!(reopened.queued_run(GPU_MATRIX_ID).is_err());

    reopened.selected = Some(GPU_MATRIX_ID.into());
    reopened.select_vector_choice(0);
    assert_eq!(
        run_options(&reopened.queued_run(GPU_MATRIX_ID).unwrap())["gpu_matrix_preset"],
        "register"
    );
}

#[test]
fn gpu_int8_matrix_choices_use_operand_limits_and_save_independently() {
    let mut app = fixture();
    app.selected = Some(GPU_INT8_MATRIX_ID.into());
    app.devices[0].properties = BTreeMap::from([
        (
            "device_local_memory_bytes".into(),
            (8 * 1024_u64.pow(3)).to_string(),
        ),
        (
            "max_storage_buffer_binding_size".into(),
            (1024_u64.pow(3)).to_string(),
        ),
        ("cooperative_matrix_int8_shape".into(), "16x16x32".into()),
        ("vulkan_subgroup_size".into(), "32".into()),
        ("storage_buffer_8bit_access".into(), "true".into()),
    ]);
    let defaults = run_options(&app.queued_run(GPU_INT8_MATRIX_ID).unwrap());
    assert_eq!(defaults["gpu_int8_matrix_preset"], "register");
    assert!(!defaults.contains_key("gpu_preset"));
    app.select_vector_choice(1);
    let sweep = run_options(&app.queued_run(GPU_INT8_MATRIX_ID).unwrap());
    assert_eq!(sweep["gpu_int8_matrix_preset"], "vram");
    assert_eq!(sweep["dataset_mode"], "sweep");
    app.select_vector_choice(2);
    app.edit_vector_slider(0.0);
    let custom = app.queued_run(GPU_INT8_MATRIX_ID).unwrap();
    assert_eq!(run_options(&custom)["dataset_bytes"], "262144");
    let range = app.compute_slider_range(GPU_INT8_MATRIX_ID, &custom.settings);
    assert_eq!(range.alignment, 1024);
    assert_eq!(range.bytes(1000.0), range.maximum);
    assert_eq!(
        range.maximum,
        (gluj_bench_core::vram_budget_bytes(8 * 1024_u64.pow(3), 25) - 32 * 1024 * 1024) / 1024
            * 1024
    );
    let mut reopened = fixture();
    reopened.devices = app.devices.clone();
    reopened.restore_benchmark_preferences(app.snapshot_benchmark_preferences());
    assert_eq!(reopened.compute_choice(GPU_INT8_MATRIX_ID), 2);
    assert_eq!(reopened.compute_choice(GPU_VECTOR_ID), 0);
    assert_eq!(
        reopened.queued_run(GPU_INT8_MATRIX_ID).unwrap().settings,
        custom.settings
    );
    reopened.devices[0]
        .properties
        .remove("storage_buffer_8bit_access");
    assert!(
        !reopened
            .compute_slider_range(GPU_INT8_MATRIX_ID, &custom.settings)
            .available()
    );
    assert!(reopened.queued_run(GPU_INT8_MATRIX_ID).is_err());
    reopened
        .run_overrides
        .get_mut(GPU_INT8_MATRIX_ID)
        .unwrap()
        .settings
        .vector_preset = 2;
    assert!(reopened.queued_run(GPU_INT8_MATRIX_ID).is_err());

    reopened.selected = Some(GPU_INT8_MATRIX_ID.into());
    reopened.select_vector_choice(0);
    assert_eq!(
        run_options(&reopened.queued_run(GPU_INT8_MATRIX_ID).unwrap())["gpu_int8_matrix_preset"],
        "register"
    );
}

#[test]
fn old_gpu_register_history_and_preferences_migrate_without_scaling_graphs() {
    let path = std::env::temp_dir().join(format!(
        "gluj-gpu-register-{}.json",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let result = BenchmarkResult {
        benchmark_id: "gpu.performance.fp32".into(),
        device_id: "gpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![Metric {
            name: "throughput".into(),
            value: 3e12,
            unit: "operations/s".into(),
            statistics: Default::default(),
        }],
        workload_metadata: BTreeMap::new(),
        device_metadata: BTreeMap::new(),
    };
    write_json(
        &path,
        &ResultsFile {
            version: 1,
            results: vec![result],
        },
    )
    .unwrap();
    let loaded = load_results(&path).unwrap();
    fs::remove_file(path).unwrap();
    assert_eq!(loaded[0].benchmark_id, GPU_VECTOR_ID);
    assert!(!scaling_result(&loaded[0]));
    let row = result_row(&loaded[0], &[], &[]);
    assert!(!row.gpu_timing_visible);
    assert!(row.summary.contains("Register-resident GPU FP32"));
    assert!(chart_tiers(&loaded[0]).is_empty());
    assert_eq!(featured_metric(&loaded[0]).unwrap().value, 3e12);
    let prefs: BenchmarkPreferences = serde_json::from_value(json!({"version":1,"tests":{"gpu.performance.fp32":{"custom":{"settings":{"gpu_intensity":0},"dataset_text":"256"}}}})).unwrap();
    let mut app = fixture();
    app.restore_benchmark_preferences(prefs);
    let migrated = run_options(&app.queued_run(GPU_VECTOR_ID).unwrap());
    assert_eq!(migrated["gpu_preset"], "register");
    assert_eq!(migrated["gpu_activity_percent"], "95");
}

#[test]
fn old_gpu_matrix_history_and_preferences_migrate_without_scaling_graphs() {
    let path = std::env::temp_dir().join(format!(
        "gluj-gpu-register-{}.json",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let result = BenchmarkResult {
        benchmark_id: "gpu.performance.matrix.fp16".into(),
        device_id: "gpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![Metric {
            name: "throughput".into(),
            value: 3e12,
            unit: "operations/s".into(),
            statistics: Default::default(),
        }],
        workload_metadata: BTreeMap::new(),
        device_metadata: BTreeMap::new(),
    };
    write_json(
        &path,
        &ResultsFile {
            version: 1,
            results: vec![result],
        },
    )
    .unwrap();
    let loaded = load_results(&path).unwrap();
    fs::remove_file(path).unwrap();
    assert_eq!(loaded[0].benchmark_id, GPU_MATRIX_ID);
    assert!(!scaling_result(&loaded[0]));
    let row = result_row(&loaded[0], &[], &[]);
    assert!(!row.gpu_timing_visible);
    assert!(row.summary.contains("Register-resident FP16 matrix"));
    assert!(chart_tiers(&loaded[0]).is_empty());
    assert_eq!(featured_metric(&loaded[0]).unwrap().value, 3e12);
    let prefs: BenchmarkPreferences = serde_json::from_value(json!({"version":1,"tests":{"gpu.performance.matrix.fp16":{"custom":{"settings":{"gpu_intensity":0},"dataset_text":"256"}}}})).unwrap();
    let mut app = fixture();
    app.restore_benchmark_preferences(prefs);
    let migrated = run_options(&app.queued_run(GPU_MATRIX_ID).unwrap());
    assert_eq!(migrated["gpu_matrix_preset"], "register");
    assert_eq!(migrated["gpu_activity_percent"], "95");
}

#[test]
fn int8_scaling_results_keep_native_reference_and_configuration_separate() {
    let mut result = BenchmarkResult {
        benchmark_id: GPU_INT8_MATRIX_ID.into(),
        device_id: "gpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![Metric {
            name: "working_set_262144.compute".into(),
            value: 3e12,
            unit: "operations/s".into(),
            statistics: Default::default(),
        }],
        workload_metadata: BTreeMap::from([
            ("gpu_int8_matrix_preset".into(), "small".into()),
            ("dataset_mode".into(), "single".into()),
        ]),
        device_metadata: BTreeMap::new(),
    };
    assert!(scaling_result(&result));
    assert!(result_summary(&result).contains("INT8 matrix reference"));
    assert!(!result_summary(&result).contains("FP16"));
    let mut other = result.clone();
    other
        .workload_metadata
        .insert("gpu_int8_matrix_preset".into(), "custom".into());
    assert!(!same_run_configuration(&result, &other));
    result
        .workload_metadata
        .insert("gpu_int8_matrix_preset".into(), "register".into());
    assert!(!scaling_result(&result));
}

#[test]
fn old_gpu_int8_matrix_history_and_preferences_migrate_without_scaling_graphs() {
    let path = std::env::temp_dir().join(format!(
        "gluj-gpu-register-{}.json",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let result = BenchmarkResult {
        benchmark_id: "gpu.performance.matrix.int8".into(),
        device_id: "gpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![Metric {
            name: "throughput".into(),
            value: 3e12,
            unit: "operations/s".into(),
            statistics: Default::default(),
        }],
        workload_metadata: BTreeMap::new(),
        device_metadata: BTreeMap::new(),
    };
    write_json(
        &path,
        &ResultsFile {
            version: 1,
            results: vec![result],
        },
    )
    .unwrap();
    let loaded = load_results(&path).unwrap();
    fs::remove_file(path).unwrap();
    assert_eq!(loaded[0].benchmark_id, GPU_INT8_MATRIX_ID);
    assert!(!scaling_result(&loaded[0]));
    let row = result_row(&loaded[0], &[], &[]);
    assert!(!row.gpu_timing_visible);
    assert!(row.summary.contains("Register-resident INT8 matrix"));
    assert!(chart_tiers(&loaded[0]).is_empty());
    assert_eq!(featured_metric(&loaded[0]).unwrap().value, 3e12);
    let prefs: BenchmarkPreferences = serde_json::from_value(json!({"version":1,"tests":{"gpu.performance.matrix.int8":{"custom":{"settings":{"gpu_intensity":0},"dataset_text":"256"}}}})).unwrap();
    let mut app = fixture();
    app.restore_benchmark_preferences(prefs);
    let migrated = run_options(&app.queued_run(GPU_INT8_MATRIX_ID).unwrap());
    assert_eq!(migrated["gpu_int8_matrix_preset"], "register");
    assert_eq!(migrated["gpu_activity_percent"], "95");
}

#[test]
fn integer_choices_preserve_register_default_and_independent_custom_settings() {
    let mut app = fixture();
    app.selected = Some(CPU_INT64_ID.into());
    app.devices[0]
        .properties
        .insert("physical_cores".into(), "4".into());
    let default = run_options(&app.queued_run(CPU_INT64_ID).unwrap());
    assert_eq!(default["int64_preset"], "register");
    assert_eq!(default["dataset_mode"], "register");
    assert!(!default.contains_key("dataset_bytes"));
    assert!(!default.contains_key("vector_preset"));
    app.select_vector_choice(2);
    app.edit_configuration(true, "cores", 2);
    app.edit_vector_slider(0.0);
    let custom = app.queued_run(CPU_INT64_ID).unwrap();
    assert_eq!(run_options(&custom)["dataset_bytes"], "2048");
    assert_eq!(run_options(&custom)["dataset_mode"], "single");
    app.select_vector_choice(1);
    let sweep = run_options(&app.queued_run(CPU_INT64_ID).unwrap());
    assert_eq!(sweep["int64_preset"], "ram");
    assert_eq!(sweep["dataset_mode"], "sweep");
    assert_eq!(sweep["cpu_core_limit"], "4");
    let mut reopened = fixture();
    reopened.devices = app.devices.clone();
    reopened.restore_benchmark_preferences(app.snapshot_benchmark_preferences());
    assert_eq!(reopened.compute_choice(CPU_INT64_ID), 1);
    for id in [CPU_VECTOR_ID, CPU_MATRIX_ID, CPU_ASCII_ID] {
        assert_eq!(reopened.compute_choice(id), 0);
    }
    reopened.selected = Some(CPU_INT64_ID.into());
    reopened.select_vector_choice(2);
    assert_eq!(
        reopened.queued_run(CPU_INT64_ID).unwrap().settings,
        custom.settings
    );
    reopened.reset_vector_configuration();
    assert_eq!(
        run_options(&reopened.queued_run(CPU_INT64_ID).unwrap())["int64_preset"],
        "register"
    );
    let range = DatasetSliderRange::for_integer(3, 65537);
    assert_eq!(range.minimum, 3072);
    assert_eq!(range.bytes(1000.0), 64512);
    assert_eq!(range.clamp(4097), 3072);
}

#[test]
fn integer_results_keep_register_and_streaming_history_and_labels_separate() {
    let mut result = BenchmarkResult {
        benchmark_id: CPU_INT64_ID.into(),
        device_id: "cpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![Metric {
            name: "throughput".into(),
            value: 10e9,
            unit: "operations/s".into(),
            statistics: Default::default(),
        }],
        workload_metadata: BTreeMap::from([("kernel_revision".into(), "4".into())]),
        device_metadata: BTreeMap::new(),
    };
    let old = result.clone();
    assert!(!scaling_result(&old));
    assert!(result_summary(&old).contains("Register-resident INT64"));
    result
        .workload_metadata
        .insert("kernel_revision".into(), "int64-streaming-1".into());
    result
        .workload_metadata
        .insert("int64_preset".into(), "custom".into());
    result.metrics[0].name = "measured_compute_ceiling".into();
    result.metrics.push(Metric {
        name: "working_set_65536.compute".into(),
        value: 5e9,
        unit: "operations/s".into(),
        statistics: Default::default(),
    });
    result.metrics.push(Metric {
        name: "working_set_65536.bandwidth".into(),
        value: 2.5e9,
        unit: "bytes/s".into(),
        statistics: Default::default(),
    });
    let row = result_row(&result, &[], &[]);
    assert_eq!(row.fact_one_label, "INT64 REGISTER REFERENCE");
    assert!(row.summary.contains("INT64 register reference"));
    assert!(!row.summary.contains("FP32"));
    assert_eq!(featured_metric(&result).unwrap().value, 5e9);
    assert_eq!(chart_tiers(&result).len(), 1);
    assert!(!same_run_configuration(&old, &result));
    let mut history = vec![old];
    remember_result(&mut history, &result);
    let mut register = history[0].clone();
    register
        .workload_metadata
        .insert("int64_preset".into(), "register".into());
    remember_result(&mut history, &register);
    assert_eq!(history.len(), 3);
    let settings = AppSettings {
        vector_preset: 1,
        ..Default::default()
    };
    assert!(!configuration_summary(&settings, CPU_INT64_ID).contains("RAM budget"));
}

#[test]
fn ascii_choices_preserve_custom_settings_and_remain_independent() {
    let mut app = fixture();
    app.selected = Some(CPU_ASCII_ID.into());
    app.devices[0]
        .properties
        .insert("physical_cores".into(), "4".into());
    let options = run_options(&app.queued_run(CPU_ASCII_ID).unwrap());
    assert_eq!(options["ascii_preset"], "compute");
    assert_eq!(options["dataset_mode"], "single");
    assert_eq!(options["cpu_core_limit"], "4");
    assert!(!options.contains_key("vector_preset"));
    app.select_vector_choice(2);
    app.edit_configuration(true, "cores", 2);
    app.edit_vector_slider(0.0);
    let custom = app.queued_run(CPU_ASCII_ID).unwrap();
    assert_eq!(run_options(&custom)["dataset_bytes"], "128");
    app.select_vector_choice(1);
    assert_eq!(
        run_options(&app.queued_run(CPU_ASCII_ID).unwrap())["dataset_mode"],
        "sweep"
    );
    let saved = app.snapshot_benchmark_preferences();
    let mut reopened = fixture();
    reopened.devices[0]
        .properties
        .insert("physical_cores".into(), "4".into());
    reopened.restore_benchmark_preferences(saved);
    assert_eq!(reopened.compute_choice(CPU_ASCII_ID), 1);
    assert_eq!(reopened.compute_choice(CPU_MATRIX_ID), 0);
    assert_eq!(reopened.compute_choice(CPU_VECTOR_ID), 0);
    reopened.selected = Some(CPU_ASCII_ID.into());
    reopened.select_vector_choice(2);
    assert_eq!(
        reopened.queued_run(CPU_ASCII_ID).unwrap().settings,
        custom.settings
    );
    reopened.edit_configuration(true, "vector-preset", 0);
    reopened.edit_configuration(true, "dataset", 2);
    reopened.edit_dataset(true, "32".into());
    let legacy = run_options(&reopened.queued_run(CPU_ASCII_ID).unwrap());
    assert_eq!(legacy["ascii_preset"], "legacy");
    assert_eq!(legacy["dataset_bytes"], (32 * 1048576_u64).to_string());
    reopened.reset_vector_configuration();
    assert_eq!(reopened.compute_choice(CPU_ASCII_ID), 0);
    let range = DatasetSliderRange::for_ascii(3, 65537);
    assert_eq!(range.minimum, 192);
    assert_eq!(range.bytes(1000.0), 65472);
    assert_eq!(range.clamp(193), 192);
}

#[test]
fn ascii_profiles_show_scan_units_and_preserve_older_history() {
    let mut result = BenchmarkResult {
        benchmark_id: CPU_ASCII_ID.into(),
        device_id: "cpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![],
        workload_metadata: BTreeMap::from([
            ("kernel_revision".into(), "ascii-scan-1".into()),
            ("thread_count".into(), "2".into()),
            ("ascii_preset".into(), "compute".into()),
        ]),
        device_metadata: BTreeMap::new(),
    };
    for (bytes, rate) in [(4096, 20e6), (65536, 10e6)] {
        for (suffix, value, unit) in [
            ("compute", rate, "strings/s"),
            ("bandwidth", rate * 64.0, "bytes/s"),
        ] {
            result.metrics.push(Metric {
                name: format!("working_set_{bytes}.{suffix}"),
                value,
                unit: unit.into(),
                statistics: gluj_bench_core::SampleStatistics {
                    sample_count: 3,
                    median: value,
                    minimum: value * 0.99,
                    maximum: value * 1.01,
                    ..Default::default()
                },
            });
        }
    }
    let row = result_row(&result, &[], &[]);
    assert!(row.primary_name.contains("Scanning"));
    assert!(row.primary_value.contains("strings/s"));
    assert!(!row.summary.contains("FP32"));
    assert!(!row.summary.contains("reference"));
    assert_eq!(row.fact_one_label, "SMALL-INPUT BASELINE");
    let (_, labels) = scaling_chart_data(&result).unwrap();
    assert!(
        labels
            .iter()
            .any(|label| label.text.contains("M records/s"))
    );
    assert!(!labels.iter().any(|label| label.text.contains("TOPS")));
    let mut older = result.clone();
    older.workload_metadata.clear();
    older.metrics = vec![Metric {
        name: "throughput".into(),
        value: 15e6,
        unit: "strings/s".into(),
        statistics: Default::default(),
    }];
    assert!(!scaling_result(&older));
    assert_eq!(featured_metric(&older).unwrap().value, 15e6);
    assert!(!same_run_configuration(&older, &result));
    let mut history = vec![older];
    remember_result(&mut history, &result);
    let mut cache = result.clone();
    cache
        .workload_metadata
        .insert("ascii_preset".into(), "l2".into());
    remember_result(&mut history, &cache);
    assert_eq!(history.len(), 3);
}

#[test]
fn matrix_choices_and_saved_settings_are_independent_of_vector_choices() {
    let mut app = fixture();
    app.devices[0]
        .properties
        .insert("physical_cores".into(), "4".into());
    app.selected = Some(CPU_MATRIX_ID.into());
    let automatic = app.queued_run(CPU_MATRIX_ID).unwrap();
    assert_eq!(run_options(&automatic)["matrix_preset"], "compute");
    assert_eq!(run_options(&automatic)["dataset_mode"], "single");
    assert!(!run_options(&automatic).contains_key("vector_preset"));
    app.select_vector_choice(2);
    app.edit_configuration(true, "cores", 2);
    app.edit_vector_slider(0.0);
    let custom = app.queued_run(CPU_MATRIX_ID).unwrap();
    assert_eq!(run_options(&custom)["dataset_bytes"], "24576");
    assert_eq!(run_options(&custom)["matrix_preset"], "custom");
    app.select_vector_choice(1);
    assert_eq!(
        run_options(&app.queued_run(CPU_MATRIX_ID).unwrap())["dataset_mode"],
        "sweep"
    );
    assert_eq!(app.compute_choice(CPU_VECTOR_ID), 0);
    let path = std::env::temp_dir().join(format!(
        "gluj-matrix-preferences-{}.json",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    write_json(&path, &app.snapshot_benchmark_preferences()).unwrap();
    let mut restored = fixture();
    restored.devices[0]
        .properties
        .insert("physical_cores".into(), "4".into());
    restored.restore_benchmark_preferences(load_benchmark_preferences(&path).unwrap());
    assert_eq!(restored.compute_choice(CPU_MATRIX_ID), 1);
    restored.selected = Some(CPU_MATRIX_ID.into());
    restored.select_vector_choice(2);
    assert_eq!(
        restored.queued_run(CPU_MATRIX_ID).unwrap().settings,
        custom.settings
    );
    restored.reset_vector_configuration();
    assert_eq!(restored.compute_choice(CPU_MATRIX_ID), 0);
    assert_eq!(custom.settings.cpu_core_limit, 2);
    fs::remove_file(path).unwrap();
}

#[test]
fn matrix_slider_rounds_to_supported_dimensions_and_reaches_its_budget() {
    let range = DatasetSliderRange::for_matrix(3, 1024 * 1024);
    assert_eq!(range.minimum, 12288 * 3);
    assert!(range.maximum <= 1024 * 1024);
    assert_eq!(range.bytes(1000.0), range.maximum);
    let values = (0..=1000)
        .map(|position| range.bytes(position as f32))
        .collect::<Vec<_>>();
    assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
    for bytes in values {
        let dimension = gluj_bench_core::cpu_matrix_dimension(bytes / 3);
        assert!(dimension >= 32 && dimension.is_multiple_of(8));
        assert_eq!(bytes, gluj_bench_core::cpu_matrix_bytes(dimension) * 3);
    }
    assert!(!DatasetSliderRange::for_matrix(3, 1024).available());
}

#[test]
fn size_slider_is_aligned_monotonic_and_reaches_the_available_limit() {
    let budget =
        gluj_bench_core::ram_allocation_budget(64 * 1024_u64.pow(3), 10 * 1024_u64.pow(3), 80);
    let range = DatasetSliderRange::new(12, budget);
    assert!(range.available());
    assert_eq!(range.minimum, 12 * 768);
    assert!(range.maximum <= budget && budget - range.maximum < range.minimum);
    assert_eq!(range.bytes(0.0), range.minimum);
    assert_eq!(range.bytes(1000.0), range.maximum);
    assert_eq!(range.position(range.minimum), 0.0);
    assert_eq!(range.position(range.maximum), 1000.0);
    assert_eq!(range.clamp(u64::MAX), range.maximum);
    let values = (0..=1000)
        .map(|step| range.bytes(step as f32))
        .collect::<Vec<_>>();
    assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!(values.iter().all(|bytes| bytes % range.minimum == 0));
    let unavailable = DatasetSliderRange::new(12, 1024);
    assert!(!unavailable.available());
    assert_eq!(unavailable.bytes(1000.0), 0);
}

#[test]
fn sliders_preserve_per_test_values_and_queued_settings() {
    let mut app = fixture();
    app.devices[0]
        .properties
        .insert("physical_cores".into(), "8".into());
    app.selected = Some(CPU_VECTOR_ID.into());
    let settings = AppSettings {
        vector_preset: 6,
        ..Default::default()
    };
    app.run_overrides.insert(
        CPU_VECTOR_ID.into(),
        TestOverride {
            settings,
            dataset_text: "256".into(),
        },
    );
    app.edit_configuration(true, "cores", 3);
    assert_eq!(app.run_overrides[CPU_VECTOR_ID].settings.cpu_core_limit, 3);
    app.edit_vector_slider(1000.0);
    let queued = app.queued_run(CPU_VECTOR_ID).unwrap();
    app.edit_vector_slider(0.0);
    assert_eq!(
        vector_size_bytes(&app.run_overrides[CPU_VECTOR_ID].settings.vector_size).unwrap(),
        3 * 768
    );
    assert!(vector_size_bytes(&queued.settings.vector_size).unwrap() > 3 * 768);
    app.edit_configuration(true, "cores", 8);
    assert_eq!(app.run_overrides[CPU_VECTOR_ID].settings.cpu_core_limit, 0);
    app.selected = Some("cpu.performance.integer.i64".into());
    app.edit_vector_slider(500.0);
    assert_eq!(
        vector_size_bytes(&app.run_overrides[CPU_VECTOR_ID].settings.vector_size).unwrap(),
        3 * 768
    );
}

#[test]
fn register_vector_results_have_throughput_and_no_scaling_chart() {
    let mut result = BenchmarkResult {
        benchmark_id: CPU_VECTOR_ID.into(),
        device_id: "cpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![Metric {
            name: "compute".into(),
            value: 1e12,
            unit: "operations/s".into(),
            statistics: Default::default(),
        }],
        workload_metadata: BTreeMap::from([
            ("vector_preset".into(), "register".into()),
            ("dataset_mode".into(), "register".into()),
        ]),
        device_metadata: BTreeMap::new(),
    };
    assert_eq!(featured_metric(&result).unwrap().value, 1e12);
    assert!(scaling_chart_data(&result).is_none());
    assert!(result_summary(&result).contains("Register-resident"));
    let register = result.clone();
    result
        .workload_metadata
        .insert("vector_preset".into(), "l1".into());
    result
        .workload_metadata
        .insert("dataset_mode".into(), "single".into());
    assert!(!same_run_configuration(&register, &result));
    let mut history = vec![];
    remember_result(&mut history, &register);
    remember_result(&mut history, &result);
    assert_eq!(history.len(), 2);
}

#[test]
fn legacy_register_history_loads_into_the_unified_vector_family() {
    let directory = std::env::temp_dir().join(format!(
        "gluj-vector-history-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&directory).unwrap();
    let path = directory.join("results.json");
    let old = BenchmarkResult {
        benchmark_id: "cpu.performance.avx2.f32_fma".into(),
        device_id: "cpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![],
        workload_metadata: BTreeMap::new(),
        device_metadata: BTreeMap::new(),
    };
    save_results(&path, &[old]).unwrap();
    let results = load_results(&path).unwrap();
    assert_eq!(results[0].benchmark_id, CPU_VECTOR_ID);
    assert_eq!(results[0].workload_metadata["vector_preset"], "register");
    assert_eq!(results[0].workload_metadata["dataset_mode"], "register");
    fs::remove_file(path).unwrap();
    fs::remove_dir(directory).unwrap();
}

#[test]
fn global_defaults_overrides_and_queue_snapshots_are_independent() {
    let mut app = fixture();
    let id = app.selected.clone().unwrap();
    app.settings.dataset_mode = 2;
    app.dataset_text = "384".into();
    assert_eq!(app.effective_settings(&id).unwrap().dataset_mode, 0);
    app.run_overrides.insert(
        id.clone(),
        TestOverride {
            settings: app.settings.clone(),
            dataset_text: "128".into(),
        },
    );
    app.edit_configuration(true, "offload", 2);
    let queued = app.queued_run(&id).unwrap();
    app.edit_dataset(true, "64".into());
    app.edit_configuration(false, "offload", 0);
    app.selected_gpu_id = Some("gpu:other".into());
    let options = run_options(&queued);
    assert_eq!(options["dataset_mode"], "single");
    assert_eq!(options["dataset_bytes"], (128_u64 * 1048576).to_string());
    assert_eq!(options["ram_offload_percent"], "100");
    assert_eq!(options["device_id"], "gpu:fixture");
    assert_eq!(app.settings.ram_offload_percent, 50);
    assert_eq!(app.effective_settings(&id).unwrap().dataset_mib, 64);
    app.edit_dataset(true, "".into());
    assert!(app.queued_run(&id).is_err());
    app.run_overrides.remove(&id);
    assert!(app.effective_settings(&id).is_err());
    app.selected_gpu_id = Some("gpu:fixture".into());
    assert_eq!(app.effective_settings(&id).unwrap().dataset_mode, 0);
    assert_eq!(app.effective_settings(&id).unwrap().ram_offload_percent, 50);
    let ordinary = app.queued_run("gpu.performance.fp32").unwrap();
    assert!(!run_options(&ordinary).contains_key("dataset_bytes"));
}

#[test]
fn saved_preset_indices_map_to_intensity_independently_of_core_count() {
    for (index, percent) in [(0, 95), (1, 99), (2, 100)] {
        let settings: AppSettings = serde_json::from_str(&format!(
            "{{\"cpu_intensity\":{index},\"gpu_intensity\":{index}}}"
        ))
        .unwrap();
        let mut run = QueuedRun {
            benchmark_id: "cpu.performance.integer.i64".into(),
            settings,
            gpu_id: None,
            physical_cores: 16,
        };
        let options = run_options(&run);
        assert_eq!(options["cpu_activity_percent"], percent.to_string());
        assert_eq!(options["gpu_activity_percent"], percent.to_string());
        assert_eq!(options["cpu_core_limit"], "16");
        run.settings.cpu_core_limit = 8;
        let options = run_options(&run);
        assert_eq!(options["cpu_activity_percent"], percent.to_string());
        assert_eq!(options["cpu_core_limit"], "8");
        run.benchmark_id = "gpu.performance.fp32.offload.scaling".into();
        assert_eq!(run_options(&run)["ram_offload_percent"], "75");
    }
    assert_eq!(INTENSITY_PERCENT[AppSettings::default().cpu_intensity], 99);
    assert_eq!(INTENSITY_PERCENT[AppSettings::default().gpu_intensity], 99);
}

#[test]
fn result_history_keeps_offload_shares_and_datasets_separate() {
    use gluj_bench_core::SampleStatistics;
    let result = |percent: &str, bytes: &str| BenchmarkResult {
        benchmark_id: "gpu.performance.fp32.offload.scaling".into(),
        device_id: "gpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![Metric {
            name: "working_set_134217728.compute".into(),
            value: 1e12,
            unit: "operations/s".into(),
            statistics: SampleStatistics::default(),
        }],
        workload_metadata: BTreeMap::from([
            ("ram_offload_percent".into(), percent.into()),
            ("dataset_mode".into(), "single".into()),
            ("requested_dataset_bytes".into(), bytes.into()),
        ]),
        device_metadata: BTreeMap::new(),
    };
    let mut saved = vec![];
    for sample in [
        result("50", "134217728"),
        result("100", "134217728"),
        result("100", "268435456"),
    ] {
        remember_result(&mut saved, &sample);
    }
    assert_eq!(saved.len(), 3);
    assert_eq!(component_results(&saved, "gpu:fixture").len(), 3);
    assert!(!same_run_configuration(&saved[0], &saved[1]));
    assert!(!same_run_configuration(&saved[1], &saved[2]));
    let previous = saved[0].clone();
    remember_result(&mut saved, &previous);
    assert_eq!(saved.len(), 3);
}

#[test]
fn older_settings_and_offload_result_ids_load_with_defaults_and_preserved_shares() {
    let path = std::env::temp_dir().join(format!(
        "gluj-config-migration-{}.json",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&path, br#"{"cpu_intensity":1,"gpu_intensity":1,"ram_budget_percent":20,"vram_budget_percent":25}"#).unwrap();
    assert_eq!(load_settings(&path).unwrap(), AppSettings::default());
    let result = BenchmarkResult {
        benchmark_id: "gpu.performance.fp32.offload50.scaling".into(),
        device_id: "gpu:fixture".into(),
        elapsed_ns: 1,
        metrics: vec![
            Metric {
                name: "working_set_1.gpu_busy_estimate".into(),
                value: 25.0,
                unit: "%".into(),
                statistics: Default::default(),
            },
            Metric {
                name: "working_set_1.compute".into(),
                value: 1e12,
                unit: "operations/s".into(),
                statistics: Default::default(),
            },
        ],
        workload_metadata: BTreeMap::from([(
            "gpu_busy_wait_method".into(),
            "throughput_equivalent_time_v1".into(),
        )]),
        device_metadata: BTreeMap::new(),
    };
    write_json(
        &path,
        &ResultsFile {
            version: 1,
            results: vec![result],
        },
    )
    .unwrap();
    let results = load_results(&path).unwrap();
    assert_eq!(
        results[0].benchmark_id,
        "gpu.performance.fp32.offload.scaling"
    );
    assert_eq!(results[0].workload_metadata["ram_offload_percent"], "50");
    assert_eq!(results[0].metrics.len(), 1);
    assert!(results[0].metrics[0].name.ends_with(".compute"));
    assert!(
        !results[0]
            .workload_metadata
            .contains_key("gpu_busy_wait_method")
    );
    fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "renders the configurable CPU vector benchmark to target/ui-preview"]
fn render_vector_configuration_previews() {
    use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
    struct Headless(Rc<MinimalSoftwareWindow>);
    impl slint::platform::Platform for Headless {
        fn create_window_adapter(
            &self,
        ) -> Result<Rc<dyn slint::platform::WindowAdapter>, slint::PlatformError> {
            Ok(self.0.clone())
        }
    }
    let adapter = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    slint::platform::set_platform(Box::new(Headless(adapter.clone()))).unwrap();
    let window = MainWindow::new().unwrap();
    let mut app = fixture();
    app.devices[0].id = "cpu:fixture".into();
    app.devices[0].name = "CPU preview".into();
    app.devices[0].category = DeviceCategory::Cpu;
    app.devices[0]
        .properties
        .insert("physical_cores".into(), "4".into());
    app.benchmarks[0].id = CPU_VECTOR_ID.into();
    app.benchmarks[0].name = "AVX2 FP32 vector compute".into();
    app.benchmarks[0].workload =
        "Configurable AVX2/FMA throughput from registers through cache to RAM".into();
    app.benchmarks[0].suite_id = "cpu.performance".into();
    app.benchmarks[0].category = BenchmarkCategory::Cpu;
    app.benchmarks[0].supported_device_ids = vec!["cpu:fixture".into()];
    app.benchmarks[0].metadata.insert(
        "cache_l1_dataset_sizes".into(),
        "24576,49152,73728,98304".into(),
    );
    app.selected_suite = "cpu.performance".into();
    app.selected = Some(CPU_VECTOR_ID.into());
    app.run_overrides.clear();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/ui-preview");
    fs::create_dir_all(&directory).unwrap();
    for (preset, choice, width, height, name) in [
        (1, 2, 1280, 820, "vector-register"),
        (2, 2, 1280, 820, "vector-cache"),
        (6, 2, 980, 680, "vector-custom-min"),
        (0, 0, 1280, 820, "vector-simple-compute"),
        (0, 1, 1280, 820, "vector-simple-sweep"),
        (0, 1, 980, 680, "vector-simple-sweep-min"),
        (1, 0, 1280, 820, "matrix-default"),
        (5, 1, 980, 680, "matrix-sweep-min"),
        (6, 2, 980, 680, "matrix-custom-min"),
        (1, 0, 1280, 820, "ascii-default"),
        (5, 1, 980, 680, "ascii-sweep-min"),
        (6, 2, 980, 680, "ascii-custom-min"),
        (1, 0, 1280, 820, "integer-default"),
        (5, 1, 980, 680, "integer-sweep-min"),
        (2, 2, 1280, 820, "integer-cache"),
        (6, 2, 980, 680, "integer-custom-min"),
    ] {
        let id = if name.starts_with("integer") {
            CPU_INT64_ID
        } else if name.starts_with("ascii") {
            CPU_ASCII_ID
        } else if name.starts_with("matrix") {
            CPU_MATRIX_ID
        } else {
            CPU_VECTOR_ID
        };
        app.selected = Some(id.into());
        app.benchmarks[0].id = id.into();
        app.benchmarks[0].name = if id == CPU_INT64_ID {
            "64-bit integer multiply-add"
        } else if id == CPU_ASCII_ID {
            "ASCII string scan"
        } else if id == CPU_MATRIX_ID {
            "AVX2 FP32 matrix compute"
        } else {
            "AVX2 FP32 vector compute"
        }
        .into();
        app.benchmarks[0].workload = if id == CPU_ASCII_ID {
            "ASCII character counting with configurable input sizes"
        } else {
            "CPU calculations with configurable data sizes"
        }
        .into();
        app.benchmarks[0].unit = if id == CPU_ASCII_ID {
            "strings/s"
        } else {
            "operations/s"
        }
        .into();
        app.benchmarks[0].data_type = if id == CPU_ASCII_ID {
            "64-byte ASCII record"
        } else if id == CPU_INT64_ID {
            "i64"
        } else {
            "fp32"
        }
        .into();
        app.benchmarks[0].metadata.insert(
            "automatic_compute_dataset_sizes".into(),
            "262144,524288,786432,1048576".into(),
        );
        let settings = AppSettings {
            vector_preset: preset,
            vector_size: "8 GiB".into(),
            ..Default::default()
        };
        app.run_overrides.insert(
            id.into(),
            TestOverride {
                settings,
                dataset_text: "256".into(),
            },
        );
        app.vector_choices.insert(id.into(), choice);
        if choice != 2 {
            app.run_overrides.clear();
        }
        App::refresh(&window, &app);
        adapter.set_size(slint::PhysicalSize::new(width, height));
        window.show().unwrap();
        if name == "vector-custom-min"
            || name == "matrix-custom-min"
            || name == "ascii-custom-min"
            || name == "integer-custom-min"
        {
            window
                .window()
                .dispatch_event(slint::platform::WindowEvent::PointerScrolled {
                    position: slint::LogicalPosition::new(850., 450.),
                    delta_x: 0.,
                    delta_y: -330.,
                });
        }
        let mut pixels = vec![slint::Rgb8Pixel::default(); width as usize * height as usize];
        assert!(adapter.draw_if_needed(|renderer| {
            renderer.render(&mut pixels, width as usize);
        }));
        let mut image = format!("P6\n{width} {height}\n255\n").into_bytes();
        for pixel in pixels {
            image.extend_from_slice(&[pixel.r, pixel.g, pixel.b]);
        }
        fs::write(directory.join(format!("{name}.ppm")), image).unwrap();
    }

    // Switching between automatic modes and full customization must preserve
    // custom controls without requiring the old override checkbox.
    app.run_overrides.clear();
    app.selected = Some(CPU_VECTOR_ID.into());
    app.benchmarks[0].id = CPU_VECTOR_ID.into();
    app.benchmarks[0].name = "AVX2 FP32 vector compute".into();
    app.settings.vector_preset = 1;
    let mut second = app.benchmarks[0].clone();
    second.id = "cpu.performance.integer.i64".into();
    second.name = "Integer compute".into();
    app.benchmarks.push(second);
    let app = Rc::new(RefCell::new(app));
    App::wire(&window, &app);
    adapter.set_size(slint::PhysicalSize::new(1280, 820));
    App::refresh(&window, &app.borrow());
    window.set_configuration_scroll_offset(0.0);
    window.show().unwrap();
    let mut pixels = vec![slint::Rgb8Pixel::default(); 1280 * 820];
    adapter.draw_if_needed(|renderer| {
        renderer.render(&mut pixels, 1280);
    });
    let mut image = b"P6\n1280 820\n255\n".to_vec();
    for pixel in &pixels {
        image.extend_from_slice(&[pixel.r, pixel.g, pixel.b]);
    }
    fs::write(directory.join("vector-interaction-start.ppm"), image).unwrap();
    assert!(!window.get_test_override_enabled());
    assert_eq!(window.get_test_vector_choice_index(), 1);
    window.invoke_select_vector_choice(2);
    App::refresh(&window, &app.borrow());
    assert!(app.borrow().run_overrides.contains_key(CPU_VECTOR_ID));
    assert!(window.get_test_override_enabled());
    App::refresh(&window, &app.borrow());
    window.set_test_vector_preset_index(6);
    window.invoke_select_test_configuration("vector-preset".into(), 6);
    window.invoke_select_vector_size(500.0);
    window.invoke_select_test_configuration("cores".into(), 2);
    App::refresh(&window, &app.borrow());
    let saved_position = window.get_test_vector_size_position();
    window.invoke_select_vector_choice(0);
    App::refresh(&window, &app.borrow());
    assert_eq!(window.get_test_vector_choice_index(), 0);
    assert_eq!(window.get_test_vector_preset_index(), 1);
    assert_eq!(window.get_test_core_value(), 4.0);
    window.invoke_select_vector_choice(2);
    App::refresh(&window, &app.borrow());
    assert_eq!(window.get_test_vector_preset_index(), 6);
    window.set_configuration_scroll_offset(-100.0);
    window.invoke_select_benchmark("cpu.performance.integer.i64".into());
    App::refresh(&window, &app.borrow());
    assert!(!window.get_test_override_enabled());
    assert_eq!(window.get_test_vector_preset_index(), 1);
    assert_eq!(window.get_test_core_value(), 4.0);
    window.invoke_select_benchmark(CPU_VECTOR_ID.into());
    App::refresh(&window, &app.borrow());
    assert!(window.get_test_override_enabled());
    assert_eq!(window.get_test_vector_preset_index(), 6);
    assert_eq!(window.get_test_core_value(), 2.0);
    assert!((window.get_test_vector_size_position() - saved_position).abs() < 1.0);
    assert_eq!(window.get_configuration_scroll_offset(), 0.0);
    window.invoke_reset_vector_configuration();
    App::refresh(&window, &app.borrow());
    assert!(!window.get_test_override_enabled());
    assert_eq!(window.get_test_vector_choice_index(), 0);
    assert_eq!(window.get_test_vector_preset_index(), 1);
}

#[test]
#[ignore = "renders the GPU FP32 configuration to target/ui-preview"]
fn render_gpu_configuration_previews() {
    use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
    struct Headless(Rc<MinimalSoftwareWindow>);
    impl slint::platform::Platform for Headless {
        fn create_window_adapter(
            &self,
        ) -> Result<Rc<dyn slint::platform::WindowAdapter>, slint::PlatformError> {
            Ok(self.0.clone())
        }
    }
    let adapter = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    slint::platform::set_platform(Box::new(Headless(adapter.clone()))).unwrap();
    let window = MainWindow::new().unwrap();
    let mut app = fixture();
    app.devices[0].name = "GPU preview".into();
    app.devices[0].properties = BTreeMap::from([
        (
            "device_local_memory_bytes".into(),
            (8 * 1024_u64.pow(3)).to_string(),
        ),
        (
            "max_storage_buffer_binding_size".into(),
            (1024_u64.pow(3)).to_string(),
        ),
        ("min_storage_buffer_offset_alignment".into(), "256".into()),
    ]);
    app.devices[0].properties.extend(BTreeMap::from([
        ("cooperative_matrix_fp16_shape".into(), "16x16x16".into()),
        ("vulkan_subgroup_size".into(), "32".into()),
        ("storage_buffer_16bit_access".into(), "true".into()),
    ]));
    app.benchmarks[0].id = GPU_VECTOR_ID.into();
    app.benchmarks[0].name = "FP32 vector compute".into();
    app.benchmarks[0].workload =
        "GPU FP32 calculations with register, small-data, VRAM or custom working sets".into();
    app.selected = Some(GPU_VECTOR_ID.into());
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/ui-preview");
    fs::create_dir_all(&directory).unwrap();
    for id in [
        GPU_VECTOR_ID,
        GPU_MATRIX_ID,
        GPU_INT8_MATRIX_ID,
        GPU_OFFLOAD_ID,
    ] {
        app.selected = Some(id.into());
        app.benchmarks[0].id = id.into();
        if id == GPU_MATRIX_ID {
            app.benchmarks[0].name = "FP16 matrix compute".into();
            app.benchmarks[0].data_type = "fp16".into();
            app.benchmarks[0].workload =
                "FP16 matrix multiply-accumulate with FP32 accumulation".into();
        }
        if id == GPU_INT8_MATRIX_ID {
            app.devices[0]
                .properties
                .insert("cooperative_matrix_int8_shape".into(), "16x16x32".into());
            app.devices[0]
                .properties
                .insert("storage_buffer_8bit_access".into(), "true".into());
            app.benchmarks[0].name = "INT8 matrix compute".into();
            app.benchmarks[0].data_type = "int8".into();
            app.benchmarks[0].workload =
                "INT8 matrix multiply-accumulate with 32-bit accumulation".into();
        }
        if id == GPU_OFFLOAD_ID {
            app.devices[0]
                .properties
                .insert("device_type".into(), "DiscreteGpu".into());
            app.devices[0].properties.insert(
                "host_memory_heap_bytes".into(),
                (64 * 1024_u64.pow(3)).to_string(),
            );
            app.benchmarks[0].name = "FP32 system-RAM offload scaling".into();
            app.benchmarks[0].data_type = "fp32".into();
            app.benchmarks[0].workload =
                "GPU FP32 compute with data split between system RAM and VRAM".into();
        }
        for (choice, preset, width, height, name) in [
            (0, 1, 1280, 820, "gpu-fp32-default"),
            (1, 5, 980, 680, "gpu-fp32-sweep-min"),
            (2, 2, 1280, 820, "gpu-fp32-small"),
            (2, 6, 980, 680, "gpu-fp32-custom-min"),
        ] {
            app.vector_choices.insert(id.into(), choice);
            app.run_overrides.clear();
            if choice == 2 {
                app.run_overrides.insert(
                    id.into(),
                    TestOverride {
                        settings: AppSettings {
                            vector_preset: preset,
                            ram_offload_percent: if id == GPU_OFFLOAD_ID && preset == 6 {
                                100
                            } else {
                                75
                            },
                            vector_size: "8 GiB".into(),
                            vector_sweep: false,
                            ..Default::default()
                        },
                        dataset_text: "256".into(),
                    },
                );
            }
            window.set_configuration_scroll_offset(0.0);
            App::refresh(&window, &app);
            adapter.set_size(slint::PhysicalSize::new(width, height));
            window.show().unwrap();
            if name == "gpu-fp32-custom-min" {
                window
                    .window()
                    .dispatch_event(slint::platform::WindowEvent::PointerScrolled {
                        position: slint::LogicalPosition::new(850., 450.),
                        delta_x: 0.,
                        delta_y: if id == GPU_OFFLOAD_ID { -430. } else { -230. },
                    });
            }
            let mut pixels = vec![slint::Rgb8Pixel::default(); width as usize * height as usize];
            assert!(adapter.draw_if_needed(|renderer| {
                renderer.render(&mut pixels, width as usize);
            }));
            let mut image = format!("P6\n{width} {height}\n255\n").into_bytes();
            for pixel in pixels {
                image.extend_from_slice(&[pixel.r, pixel.g, pixel.b]);
            }
            fs::write(
                directory.join(format!(
                    "{name}-{}.ppm",
                    if id == GPU_OFFLOAD_ID {
                        "offload"
                    } else if id == GPU_INT8_MATRIX_ID {
                        "int8-matrix"
                    } else if id == GPU_MATRIX_ID {
                        "matrix"
                    } else {
                        "vector"
                    }
                )),
                image,
            )
            .unwrap();
            assert!(!window.get_test_is_cpu());
            assert_eq!(
                window.get_test_vector_register(),
                choice == 0 && id != GPU_OFFLOAD_ID
            );
        }
    }
}

/// Manual visual QA without opening a native window or running benchmarks.
#[test]
#[ignore = "renders configuration previews to target/ui-preview"]
fn render_configuration_previews() {
    use slint::Model;
    use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
    struct Headless(Rc<MinimalSoftwareWindow>);
    impl slint::platform::Platform for Headless {
        fn create_window_adapter(
            &self,
        ) -> Result<Rc<dyn slint::platform::WindowAdapter>, slint::PlatformError> {
            Ok(self.0.clone())
        }
    }
    let adapter = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    slint::platform::set_platform(Box::new(Headless(adapter.clone()))).unwrap();
    let window = MainWindow::new().unwrap();
    window.on_chart_point_at(super::chart_hover::nearest);
    let mut app = fixture();
    let id = app.selected.clone().unwrap();
    let mut settings = app.settings.clone();
    settings.dataset_mode = 2;
    app.run_overrides.insert(
        id,
        TestOverride {
            settings,
            dataset_text: "256".into(),
        },
    );
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/ui-preview");
    fs::create_dir_all(&directory).unwrap();
    for (page, width, height, name) in [
        (1, 1280, 820, "test-override"),
        (1, 980, 680, "test-override-min"),
        (1, 1280, 820, "test-override-options"),
        (3, 1280, 820, "global-settings"),
        (2, 1280, 820, "snapshot-picker"),
        (2, 980, 680, "snapshot-picker-min"),
        (2, 1280, 820, "scaling-history"),
        (2, 980, 680, "scaling-history-min"),
        (2, 1280, 820, "gpu-timings"),
        (2, 980, 680, "gpu-timings-min"),
        (2, 1280, 820, "tuning-visuals"),
        (2, 980, 680, "tuning-visuals-min"),
        (2, 1280, 820, "tuning-visuals-charts"),
        (2, 980, 680, "tuning-visuals-charts-min"),
        (2, 1280, 820, "latest-details"),
        (2, 980, 680, "latest-details-min"),
        (2, 1280, 820, "chart-hover"),
        (2, 980, 680, "chart-hover-min"),
        (2, 1280, 820, "chart-guide"),
        (2, 980, 680, "chart-guide-min"),
        (2, 1280, 820, "timing-hover"),
        (2, 1280, 820, "ram-latency"),
        (2, 980, 680, "ram-latency-min"),
        (2, 1280, 820, "cache-latency"),
        (2, 980, 680, "cache-latency-min"),
    ] {
        app.page = page;
        if page == 2 {
            use gluj_bench_core::SampleStatistics;
            let metric = |name: &str, value: f64, unit: &str| Metric {
                name: name.into(),
                value,
                unit: unit.into(),
                statistics: SampleStatistics {
                    sample_count: 5,
                    minimum: value * 0.99,
                    median: value,
                    maximum: value * 1.01,
                    standard_deviation: value * 0.01,
                },
            };
            let mut metrics = vec![metric("measured_compute_ceiling", 100e12, "operations/s")];
            for (bytes, throughput) in [
                (262144_u64, 80e12),
                (1048576, 75e12),
                (67108864, 25e12),
                (268435456, 20e12),
            ] {
                metrics.push(metric(
                    &format!("working_set_{bytes}.compute"),
                    throughput,
                    "operations/s",
                ));
                metrics.push(metric(
                    &format!("working_set_{bytes}.host_bandwidth"),
                    throughput / 1000.0,
                    "bytes/s",
                ));
                let gpu_ns = bytes as f64 / 262144.0 * 2000.0;
                metrics.push(metric(
                    &format!("working_set_{bytes}.gpu_execution_time"),
                    gpu_ns,
                    "ns",
                ));
                metrics.push(metric(
                    &format!("working_set_{bytes}.end_to_end_time"),
                    gpu_ns * 1.4 + 1000.0,
                    "ns",
                ));
            }
            app.results = vec![BenchmarkResult {
                benchmark_id: "gpu.performance.fp32.offload.scaling".into(),
                device_id: "gpu:fixture".into(),
                elapsed_ns: 1_000_000_000,
                metrics,
                workload_metadata: BTreeMap::from([
                    ("ram_offload_percent".into(), "75".into()),
                    ("compute_reference_drift_percent".into(), "1.0".into()),
                    (
                        "gpu_timing_batch_passes".into(),
                        "262144:64,1048576:32,67108864:8,268435456:2".into(),
                    ),
                ]),
                device_metadata: BTreeMap::new(),
            }];
            if name.starts_with("tuning-visuals") {
                let result = &mut app.results[0];
                result.metrics = vec![
                    metric("measured_compute_ceiling", 38.18e12, "operations/s"),
                    metric("working_set_1024000.compute", 18.08e12, "operations/s"),
                    metric("working_set_12884901888.compute", 3.16e12, "operations/s"),
                    metric("working_set_12884901888.bandwidth", 36.98e9, "bytes/s"),
                ];
                result.metrics[2].statistics.standard_deviation = 3.16e12 * 0.003;
                for (key, value) in [
                    ("compute_reference_drift_percent", "8.87"),
                    ("allocated_test_buffer_bytes", "12884901888"),
                    ("allocation_budget_bytes", "22817013760"),
                    ("host_allocated_test_buffer_bytes", "9663676416"),
                    ("device_allocated_test_buffer_bytes", "3221225472"),
                ] {
                    result.workload_metadata.insert(key.into(), value.into());
                }
            }
            if name.starts_with("latest-details") {
                let result = &mut app.results[0];
                result.benchmark_id = "gpu.performance.fp16".into();
                result.metrics = vec![metric("throughput", 38.36e12, "operations/s")];
                result.metrics[0].statistics.minimum = 34.55e12;
                result.metrics[0].statistics.maximum = 48.20e12;
                result.metrics[0].statistics.standard_deviation = 38.36e12 * 0.152;
                result.elapsed_ns = 1_198_940_000;
                result.workload_metadata = BTreeMap::from([
                    ("data_type".into(), "fp16".into()),
                    ("api".into(), "Vulkan".into()),
                    ("bound_classification".into(), "compute_bound".into()),
                ]);
            }
            if name.starts_with("scaling-history") {
                let template = app.results[0].clone();
                app.results.clear();
                for (share, bytes) in [
                    (50, 262144),
                    (75, 1048576),
                    (100, 67108864),
                    (75, 268435456),
                ] {
                    let mut result = template.clone();
                    result
                        .workload_metadata
                        .insert("ram_offload_percent".into(), share.to_string());
                    result
                        .workload_metadata
                        .insert("gpu_offload_preset".into(), "custom".into());
                    result
                        .workload_metadata
                        .insert("dataset_mode".into(), "single".into());
                    result
                        .workload_metadata
                        .insert("requested_dataset_bytes".into(), bytes.to_string());
                    result
                        .workload_metadata
                        .insert("resolved_dataset_bytes".into(), bytes.to_string());
                    result.metrics.retain(|m| {
                        !m.name.starts_with("working_set_")
                            || m.name.starts_with(&format!("working_set_{bytes}."))
                    });
                    app.results.push(result);
                }
                assert_eq!(component_results(&app.results, "gpu:fixture").len(), 3);
                app.result_selection = 0;
            }
            app.result_details_expanded = !name.starts_with("tuning-visuals");
            if name.starts_with("ram-latency") || name.starts_with("cache-latency") {
                app.devices = vec![DeviceDescriptor {
                    id: "cpu:system".into(),
                    name: "Example CPU / system RAM".into(),
                    category: DeviceCategory::Cpu,
                    available: true,
                    status: "Ready".into(),
                    properties: BTreeMap::new(),
                    caches: vec![],
                }];
                let benchmark = &mut app.benchmarks[0];
                benchmark.id = "cpu.latency.memory".into();
                benchmark.name = "RAM random-object latency".into();
                benchmark.category = BenchmarkCategory::Memory;
                benchmark.unit = "ns".into();
                benchmark.suite_id = "cpu.bandwidth".into();
                benchmark.supported_device_ids = vec!["memory:system".into()];
                let result = &mut app.results[0];
                result.benchmark_id = "cpu.latency.memory".into();
                result.device_id = "memory:system".into();
                result.metrics = vec![metric("read_latency", 116.0, "ns")];
                result.workload_metadata = BTreeMap::from([
                    ("working_set_bytes".into(), "268435456".into()),
                    ("thread_count".into(), "1".into()),
                ]);
                let mut localized = result.clone();
                localized.benchmark_id = "cpu.latency.memory.localized".into();
                localized.metrics[0] = metric("read_latency", 84.0, "ns");
                localized
                    .workload_metadata
                    .insert("locality_block_bytes".into(), "65536".into());
                let mut localized_descriptor = benchmark.clone();
                localized_descriptor.id = "cpu.latency.memory.localized".into();
                localized_descriptor.name = "RAM read latency (localized)".into();
                app.benchmarks.push(localized_descriptor);
                app.results.push(localized);
                if name.starts_with("cache-latency") {
                    let template = app.results[0].clone();
                    let descriptor = app.benchmarks[0].clone();
                    app.results.clear();
                    app.benchmarks.clear();
                    for (level, value, bytes, capacity) in [
                        (1, 1.1, 24576, 32768),
                        (2, 3.3, 131072, 524288),
                        (3, 11.4, 2097152, 33554432),
                    ] {
                        let mut result = template.clone();
                        result.benchmark_id = format!("cpu.latency.cache.l{level}");
                        result.device_id = "cpu:system".into();
                        result.metrics[0] = metric("read_latency", value, "ns");
                        result
                            .workload_metadata
                            .insert("working_set_bytes".into(), bytes.to_string());
                        result
                            .workload_metadata
                            .insert("target_cache_bytes".into(), capacity.to_string());
                        let mut benchmark = descriptor.clone();
                        benchmark.id = result.benchmark_id.clone();
                        benchmark.name = format!("L{level} cache read latency");
                        benchmark.category = BenchmarkCategory::Cpu;
                        app.results.push(result);
                        app.benchmarks.push(benchmark);
                    }
                    app.result_selection = 2;
                }
            }
        }
        app.snapshot = snapshot::SnapshotState::default();
        if name.starts_with("snapshot-picker") {
            app.open_snapshot();
            app.finish_snapshot_for_test();
            app.snapshot.choices = (0..6).map(|index| crate::snapshot::Choice {
                result_index: 0,
                selected: index == 0,
                title: "AVX2 FP32 vector compute".into(),
                detail: format!("AMD Ryzen 9 5950X 16-Core Processor - {} - 16 cores - 99% activity - 16 iterations", ["Sweep to 15.95 GiB", "L1 target - Sweep to 384.00 KiB", "Register compute"][index % 3]),
            }).collect();
        }
        App::refresh(&window, &app);
        if name.starts_with("snapshot-picker") {
            let summaries = window.get_result_summaries();
            let details = window.get_selected_result();
            let choices = window.get_snapshot_choices();
            app.ui_dirty = false;
            app.snapshot.choices[1].selected = true;
            app.update_snapshot();
            App::refresh_snapshot(&window, &app);
            assert!(
                window.get_snapshot_choices() == choices,
                "Checkbox model must be retained"
            );
            assert!(window.get_snapshot_choices().row_data(1).unwrap().selected);
            assert!(
                window.get_result_summaries() == summaries,
                "Snapshot clicks must preserve result/chart models"
            );
            assert!(window.get_selected_result().profile_points == details.profile_points);
            assert!(!window.get_snapshot_ready());
            assert!(!app.ui_dirty);
            app.finish_snapshot_for_test();
            App::refresh_snapshot(&window, &app);
            assert!(window.get_snapshot_ready());
            assert!(window.get_result_summaries() == summaries);
        }
        window.set_tuning_details_expanded(name.starts_with("tuning-visuals"));
        adapter.set_size(slint::PhysicalSize::new(width, height));
        window.show().unwrap();
        if name == "test-override-options" {
            window
                .window()
                .dispatch_event(slint::platform::WindowEvent::PointerScrolled {
                    position: slint::LogicalPosition::new(1100., 500.),
                    delta_x: 0.,
                    delta_y: -450.,
                });
        }
        if page == 2 {
            window
                .window()
                .dispatch_event(slint::platform::WindowEvent::PointerScrolled {
                    position: slint::LogicalPosition::new(width as f32 - 100.0, 500.0),
                    delta_x: 0.0,
                    delta_y: 10000.0,
                });
            window
                .window()
                .dispatch_event(slint::platform::WindowEvent::PointerScrolled {
                    position: slint::LogicalPosition::new(width as f32 - 100.0, 500.0),
                    delta_x: 0.0,
                    delta_y: if name.starts_with("scaling-history")
                        || name.starts_with("snapshot-picker")
                    {
                        0.0
                    } else if name.starts_with("ram-latency") || name.starts_with("cache-latency") {
                        -280.0
                    } else if name.starts_with("chart-") {
                        -850.0
                    } else if name == "timing-hover" {
                        -1350.0
                    } else if name.starts_with("latest-details") {
                        -480.0
                    } else if name.starts_with("tuning-visuals") {
                        if name.contains("charts") {
                            -1040.0
                        } else {
                            -520.0
                        }
                    } else if width > 1000 {
                        -1050.0
                    } else {
                        -850.0
                    },
                });
        }
        if name.starts_with("chart-") || name == "timing-hover" {
            let position = if name.starts_with("chart-guide") {
                if width > 1000 {
                    slint::LogicalPosition::new(800.0, 330.0)
                } else {
                    slint::LogicalPosition::new(650.0, 315.0)
                }
            } else if name == "timing-hover" {
                slint::LogicalPosition::new(1042.0, 350.0)
            } else if width > 1000 {
                slint::LogicalPosition::new(564.0, 302.0)
            } else {
                slint::LogicalPosition::new(477.0, 286.0)
            };
            window
                .window()
                .dispatch_event(slint::platform::WindowEvent::PointerMoved { position });
        }
        let mut pixels = vec![slint::Rgb8Pixel::default(); width as usize * height as usize];
        assert!(adapter.draw_if_needed(|renderer| {
            renderer.render(&mut pixels, width as usize);
        }));
        let mut image = format!("P6\n{width} {height}\n255\n").into_bytes();
        for pixel in &pixels {
            image.extend_from_slice(&[pixel.r, pixel.g, pixel.b]);
        }
        fs::write(directory.join(format!("{name}.ppm")), image).unwrap();
        if name.starts_with("chart-") || name == "timing-hover" {
            window
                .window()
                .dispatch_event(slint::platform::WindowEvent::PointerExited);
            let mut cleared = vec![slint::Rgb8Pixel::default(); pixels.len()];
            assert!(adapter.draw_if_needed(|renderer| {
                renderer.render(&mut cleared, width as usize);
            }));
            assert!(
                pixels
                    .iter()
                    .zip(cleared.iter())
                    .any(|(a, b)| (a.r, a.g, a.b) != (b.r, b.g, b.b)),
                "Hover guides must disappear when the pointer leaves"
            );
        }
    }
}

/// Exercise the actual navigation callbacks without polling timers or benchmark execution.
#[test]
#[ignore = "headless navigation and presentation-cache regression check"]
fn navigation_updates_immediately_and_reuses_result_presentations() {
    use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
    struct Headless(Rc<MinimalSoftwareWindow>);
    impl slint::platform::Platform for Headless {
        fn create_window_adapter(
            &self,
        ) -> Result<Rc<dyn slint::platform::WindowAdapter>, slint::PlatformError> {
            Ok(self.0.clone())
        }
    }
    let adapter = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    slint::platform::set_platform(Box::new(Headless(adapter.clone()))).unwrap();
    let window = MainWindow::new().unwrap();
    let mut app = fixture();
    let mut second = app.benchmarks[0].clone();
    second.id = GPU_VECTOR_ID.into();
    second.name = "FP32 vector compute".into();
    app.benchmarks.push(second);
    for id in [GPU_VECTOR_ID, GPU_OFFLOAD_ID] {
        app.results.push(BenchmarkResult {
            benchmark_id: id.into(),
            device_id: "gpu:fixture".into(),
            elapsed_ns: 123456,
            metrics: vec![Metric {
                name: "working_set_1048576.compute".into(),
                value: 10e12,
                unit: "operations/s".into(),
                statistics: Default::default(),
            }],
            workload_metadata: BTreeMap::from([
                ("dataset_mode".into(), "single".into()),
                ("gpu_preset".into(), "custom".into()),
                ("ram_offload_percent".into(), "75".into()),
            ]),
            device_metadata: BTreeMap::new(),
        });
    }
    let app = Rc::new(RefCell::new(app));
    App::wire(&window, &app);
    App::refresh(&window, &app.borrow());
    app.borrow_mut().ui_dirty = false;
    let devices = window.get_devices();
    let benchmarks = window.get_benchmarks();
    let summaries = window.get_result_summaries();
    let detail = window.get_selected_result();
    let started = std::time::Instant::now();
    for page in [0, 1, 2, 3, 2] {
        window.invoke_select_page(page);
        assert_eq!(
            window.get_page(),
            page,
            "Page must change before a polling tick"
        );
        assert_eq!(app.borrow().page, page);
        assert!(!app.borrow().ui_dirty);
        assert!(window.get_devices() == devices);
        assert!(window.get_benchmarks() == benchmarks);
        assert!(window.get_result_summaries() == summaries);
        assert!(window.get_selected_result().measurement_facts == detail.measurement_facts);
    }
    println!(
        "Five page switches: {:?} (callback time, excludes painting)",
        started.elapsed()
    );
    window.invoke_toggle_device_details("gpu:fixture".into());
    assert_eq!(window.get_expanded_device_id().as_str(), "gpu:fixture");
    window.invoke_toggle_device_details("gpu:fixture".into());
    assert!(window.get_expanded_device_id().is_empty());
    window.invoke_select_benchmark(GPU_VECTOR_ID.into());
    assert_eq!(
        window.get_selected_benchmark_name().as_str(),
        "FP32 vector compute"
    );
    assert_eq!(window.get_configuration_test_id().as_str(), GPU_VECTOR_ID);
    assert!(window.get_devices() == devices);
    assert!(window.get_result_summaries() == summaries);
    window.invoke_select_suite("cpu.bandwidth".into());
    assert!(window.get_suite_description().contains("Cache/RAM"));
    window.invoke_select_suite("gpu.performance".into());
    window.invoke_select_gpu(0);
    assert!(!app.borrow().ui_dirty);
    assert!(window.get_result_summaries() == summaries);
    window.invoke_select_result_row(1);
    assert_eq!(window.get_result_selection(), 1);
    assert!(!window.get_result_details_expanded());
    window.invoke_select_result_row(0);
    assert!(window.get_selected_result().measurement_facts == detail.measurement_facts);
    assert!(window.get_selected_result().profile_points == detail.profile_points);
    assert_eq!(app.borrow().result_presentations.borrow().len(), 2);
    window.invoke_toggle_result_details();
    assert!(window.get_result_details_expanded());
    window.invoke_select_comparison(0);
    assert!(window.get_selected_result().measurement_facts == detail.measurement_facts);
    assert!(!app.borrow().ui_dirty);
    app.borrow_mut().results[1].metrics[0].value = 20e12;
    App::refresh(&window, &app.borrow());
    assert!(window.get_selected_result().measurement_facts != detail.measurement_facts);
    assert_eq!(
        window.get_selected_result().primary_value.as_str(),
        "20.00 TOPS"
    );
    adapter.set_size(slint::PhysicalSize::new(980, 680));
    window.show().unwrap();
    let mut pixels = vec![slint::Rgb8Pixel::default(); 980 * 680];
    assert!(adapter.draw_if_needed(|renderer| {
        renderer.render(&mut pixels, 980);
    }));
}
