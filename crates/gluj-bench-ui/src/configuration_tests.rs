use super::*;

fn fixture() -> App {
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
            properties: BTreeMap::new(),
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
fn global_defaults_overrides_and_queue_snapshots_are_independent() {
    let mut app = fixture();
    let id = app.selected.clone().unwrap();
    app.settings.dataset_mode = 2;
    app.dataset_text = "384".into();
    assert_eq!(app.effective_settings(&id).unwrap().dataset_mib, 384);
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
    assert_eq!(app.effective_settings(&id).unwrap().dataset_mib, 384);
    assert_eq!(app.effective_settings(&id).unwrap().ram_offload_percent, 50);
    let ordinary = app.queued_run("gpu.performance.fp32").unwrap();
    assert!(!run_options(&ordinary).contains_key("dataset_bytes"));
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
    assert_eq!(component_results(&saved, "gpu:fixture").len(), 2);
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

/// Manual visual QA without opening a native window or running benchmarks.
#[test]
#[ignore = "renders configuration previews to target/ui-preview"]
fn render_configuration_previews() {
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
            app.result_details_expanded = !name.starts_with("tuning-visuals");
        }
        App::refresh(&window, &app);
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
                    delta_y: if name.starts_with("chart-") {
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
