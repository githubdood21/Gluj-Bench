#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use gluj_bench_core::{
    BenchmarkCategory, BenchmarkDescriptor, BenchmarkResult, DeviceCategory, DeviceDescriptor,
    PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use slint::{Color, ComponentHandle, ModelRc, SharedString, Timer, TimerMode, VecModel};
use std::{
    cell::RefCell,
    collections::VecDeque,
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    rc::Rc,
    sync::mpsc::{self, Receiver},
    thread,
    time::Duration,
};

slint::include_modules!();

const HARDWARE_METADATA_CACHE_VERSION: u32 = 7;
const BENCHMARK_TARGET_DURATION_MS: u64 = 2_000;
const BENCHMARK_SAMPLES: u32 = 5;

fn main() -> Result<(), slint::PlatformError> {
    let window = MainWindow::new()?;
    let app = Rc::new(RefCell::new(App::new()));
    App::wire(&window, &app);
    app.borrow_mut().request_startup_check();
    App::refresh(&window, &app.borrow());
    app.borrow_mut().ui_dirty = false;
    let weak_window = window.as_weak();
    let weak_app = Rc::downgrade(&app);
    let timer = Timer::default();
    timer.start(TimerMode::Repeated, Duration::from_millis(100), move || {
        if let (Some(window), Some(app)) = (weak_window.upgrade(), weak_app.upgrade()) {
            let should_refresh = {
                let mut app = app.borrow_mut();
                app.receive_worker_events();
                std::mem::take(&mut app.ui_dirty)
            };
            if should_refresh {
                App::refresh(&window, &app.borrow());
            }
        }
    });
    window.run()
}

enum WorkerEvent {
    Response(Value),
    Status(String),
}
struct WorkerClient {
    child: Option<Child>,
    input: Option<ChildStdin>,
    events: Receiver<WorkerEvent>,
}
impl WorkerClient {
    fn start() -> (Self, String) {
        let (sender, events) = mpsc::channel();
        let path = worker_path();
        let mut command = Command::new(&path);
        command
            .arg("--stdio")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        match command.spawn() {
            Ok(mut child) => {
                let input = child.stdin.take();
                if let Some(stdout) = child.stdout.take() {
                    let sender = sender.clone();
                    thread::spawn(move || {
                        for line in BufReader::new(stdout).lines() {
                            match line {
                                Ok(line) => match serde_json::from_str(&line) {
                                    Ok(response) => {
                                        let _ = sender.send(WorkerEvent::Response(response));
                                    }
                                    Err(problem) => {
                                        let _ = sender.send(WorkerEvent::Status(format!(
                                            "Worker returned invalid JSON: {problem}"
                                        )));
                                    }
                                },
                                Err(problem) => {
                                    let _ = sender.send(WorkerEvent::Status(format!(
                                        "Worker output failed: {problem}"
                                    )));
                                    break;
                                }
                            }
                        }
                    });
                }
                if let Some(stderr) = child.stderr.take() {
                    thread::spawn(move || {
                        for message in BufReader::new(stderr).lines().map_while(Result::ok) {
                            let _ = sender.send(WorkerEvent::Status(message));
                        }
                    });
                }
                (
                    Self {
                        child: Some(child),
                        input,
                        events,
                    },
                    "Benchmark worker connected; loading capabilities…".into(),
                )
            }
            Err(problem) => (
                Self {
                    child: None,
                    input: None,
                    events,
                },
                format!("Unable to start {}: {problem}", path.display()),
            ),
        }
    }
    fn send(&mut self, request: Value) -> Result<(), String> {
        let input = self
            .input
            .as_mut()
            .ok_or_else(|| "The benchmark worker is not running.".to_owned())?;
        writeln!(input, "{request}").map_err(|problem| problem.to_string())?;
        input.flush().map_err(|problem| problem.to_string())
    }
    fn online(&self) -> bool {
        self.child.is_some()
    }
}
impl Drop for WorkerClient {
    fn drop(&mut self) {
        self.input.take();
        if let Some(mut child) = self.child.take()
            && !matches!(child.try_wait(), Ok(Some(_)))
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct App {
    page: i32,
    worker: WorkerClient,
    status: String,
    devices: Vec<DeviceDescriptor>,
    benchmarks: Vec<BenchmarkDescriptor>,
    selected_suite: String,
    selected_gpu_id: Option<String>,
    selected: Option<String>,
    expanded_device_id: Option<String>,
    result_details_expanded: bool,
    results: Vec<BenchmarkResult>,
    queue: VecDeque<String>,
    active_request: Option<String>,
    progress: f32,
    progress_message: String,
    request_counter: u64,
    fingerprint: Option<String>,
    metadata_scan_pending: bool,
    metadata_devices_received: bool,
    metadata_benchmarks_received: bool,
    ui_dirty: bool,
}

#[derive(Serialize, Deserialize)]
struct HardwareMetadataCache {
    version: u32,
    fingerprint: String,
    devices: Vec<DeviceDescriptor>,
    benchmarks: Vec<BenchmarkDescriptor>,
}
impl App {
    fn new() -> Self {
        let (worker, status) = WorkerClient::start();
        let mut app = Self {
            page: 0,
            worker,
            status,
            devices: vec![],
            benchmarks: vec![],
            selected_suite: "cpu.bandwidth".into(),
            selected_gpu_id: None,
            selected: None,
            expanded_device_id: None,
            result_details_expanded: false,
            results: vec![],
            queue: VecDeque::new(),
            active_request: None,
            progress: 0.,
            progress_message: String::new(),
            request_counter: 0,
            fingerprint: None,
            metadata_scan_pending: false,
            metadata_devices_received: false,
            metadata_benchmarks_received: false,
            ui_dirty: true,
        };
        if let Some(cache) = load_metadata_cache() {
            app.devices = cache.devices;
            app.benchmarks = cache.benchmarks;
            app.fingerprint = Some(cache.fingerprint);
            app.ensure_selected_gpu();
            app.select_first_available();
            app.status = "Loaded saved hardware profile; verifying this system…".into();
        }
        app
    }
    fn wire(window: &MainWindow, app: &Rc<RefCell<Self>>) {
        let app_weak = Rc::downgrade(app);
        window.on_select_page(move |page| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.page = page;
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_select_suite(move |suite| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.select_suite(suite.as_str());
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_select_gpu(move |index| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.select_gpu(index);
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_select_benchmark(move |id| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.selected = Some(id.to_string());
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_toggle_device_details(move |id| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.expanded_device_id = (app.expanded_device_id.as_deref() != Some(id.as_str()))
                    .then(|| id.to_string());
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_toggle_result_details(move || {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.result_details_expanded = !app.result_details_expanded;
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_run_selected(move || {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.run_selected();
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_run_all(move || {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.run_all();
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_cancel(move || {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.cancel();
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_clear_results(move || {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.results.clear();
                app.result_details_expanded = false;
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_rescan_hardware(move || {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.rescan_hardware();
                app.ui_dirty = true;
            }
        });
    }
    fn next_id(&mut self, prefix: &str) -> String {
        self.request_counter += 1;
        format!("ui-{prefix}-{}", self.request_counter)
    }
    fn request_startup_check(&mut self) {
        let id = self.next_id("fingerprint");
        if let Err(problem) = self
            .worker
            .send(json!({"protocol": PROTOCOL_VERSION, "id": id, "command": "fingerprint"}))
        {
            self.status = problem;
        }
    }
    fn rescan_hardware(&mut self) {
        self.status = "Rescanning hardware capabilities…".into();
        self.request_capabilities();
    }
    fn request_capabilities(&mut self) {
        self.metadata_scan_pending = true;
        self.metadata_devices_received = false;
        self.metadata_benchmarks_received = false;
        for command in ["devices", "benchmarks"] {
            let id = self.next_id(command);
            if let Err(problem) = self
                .worker
                .send(json!({"protocol": PROTOCOL_VERSION, "id": id, "command": command}))
            {
                self.status = problem;
                break;
            }
        }
    }
    fn select_suite(&mut self, suite: &str) {
        self.selected_suite = suite.into();
        self.ensure_selected_gpu();
        self.select_first_available();
    }
    fn ensure_selected_gpu(&mut self) {
        self.selected_gpu_id = preferred_gpu_id(&self.devices, self.selected_gpu_id.as_deref());
    }
    fn select_gpu(&mut self, index: i32) {
        let ids: Vec<_> = self
            .devices
            .iter()
            .filter(|d| d.category == DeviceCategory::Gpu && d.available)
            .map(|d| d.id.clone())
            .collect();
        self.selected_gpu_id = usize::try_from(index)
            .ok()
            .and_then(|index| ids.get(index).cloned());
        self.select_first_available();
    }
    fn select_first_available(&mut self) {
        self.selected = self
            .benchmarks
            .iter()
            .find(|b| b.suite_id == self.selected_suite && self.benchmark_available(b))
            .map(|b| b.id.clone());
    }
    fn benchmark_available(&self, benchmark: &BenchmarkDescriptor) -> bool {
        benchmark_available_for(benchmark, self.selected_gpu_id.as_deref())
    }
    fn start_next(&mut self) {
        if self.active_request.is_some() {
            return;
        }
        let Some(benchmark_id) = self.queue.pop_front() else {
            return;
        };
        let id = self.next_id("run");
        let mut options = serde_json::Map::new();
        if benchmark_id.starts_with("gpu.")
            && let Some(device_id) = &self.selected_gpu_id
        {
            options.insert("device_id".into(), Value::String(device_id.clone()));
        }
        match self.worker.send(json!({"protocol": PROTOCOL_VERSION, "id": id, "command": "run", "arguments": {"benchmark_id": benchmark_id, "target_duration_ms": BENCHMARK_TARGET_DURATION_MS, "samples": BENCHMARK_SAMPLES, "options": options}})) { Ok(()) => { self.active_request = Some(id); self.progress = 0.; self.progress_message = format!("Starting {benchmark_id}"); self.status = "Benchmark running…".into(); }, Err(problem) => { self.status = problem; self.queue.clear(); } }
    }
    fn run_selected(&mut self) {
        if let Some(id) = self.selected.clone() {
            self.queue.clear();
            self.queue.push_back(id);
            self.start_next();
        }
    }
    fn run_all(&mut self) {
        let mut items: Vec<_> = self
            .benchmarks
            .iter()
            .filter(|b| b.suite_id == self.selected_suite && self.benchmark_available(b))
            .cloned()
            .collect();
        items.sort_by_key(|b| b.display_order);
        self.queue = items.into_iter().map(|b| b.id).collect();
        self.start_next();
    }
    fn cancel(&mut self) {
        self.queue.clear();
        if let Some(request_id) = self.active_request.clone() {
            let id = self.next_id("cancel");
            let _ = self.worker.send(json!({"protocol": PROTOCOL_VERSION, "id": id, "command": "cancel", "arguments": {"request_id": request_id}}));
            self.status = "Cancelling benchmark…".into();
        }
    }
    fn receive_worker_events(&mut self) {
        let mut continue_queue = false;
        while let Ok(event) = self.worker.events.try_recv() {
            self.ui_dirty = true;
            match event {
                WorkerEvent::Status(message) => self.status = message,
                WorkerEvent::Response(response) => {
                    let response_id = response["id"].as_str().unwrap_or_default();
                    match response["type"].as_str().unwrap_or_default() {
                        "progress" if self.active_request.as_deref() == Some(response_id) => {
                            self.progress =
                                response["data"]["fraction"].as_f64().unwrap_or(0.) as f32;
                            self.progress_message = response["data"]["message"]
                                .as_str()
                                .unwrap_or_default()
                                .into();
                        }
                        "result" if self.active_request.as_deref() == Some(response_id) => {
                            match serde_json::from_value(response["data"].clone()) {
                                Ok(result) => {
                                    self.results.push(result);
                                    self.result_details_expanded = false;
                                    self.status = "Benchmark completed.".into();
                                    self.page = 2;
                                }
                                Err(problem) => self.status = problem.to_string(),
                            }
                            self.active_request = None;
                            self.progress = 1.;
                            continue_queue = true;
                        }
                        "error" => {
                            let message = response["error"]["message"]
                                .as_str()
                                .unwrap_or("Worker request failed.")
                                .to_owned();
                            if self.active_request.as_deref() == Some(response_id) {
                                self.active_request = None;
                                self.queue.clear();
                                self.progress = 0.;
                            }
                            self.status = message;
                        }
                        _ => {
                            if let Some(fingerprint) = response["data"]["fingerprint"].as_str() {
                                if self.fingerprint.as_deref() == Some(fingerprint)
                                    && !self.devices.is_empty()
                                    && !self.benchmarks.is_empty()
                                {
                                    self.status = "Saved hardware profile matches this system. Select Rescan hardware to refresh it.".into();
                                } else {
                                    self.fingerprint = Some(fingerprint.to_owned());
                                    self.status = "Hardware changed or no saved profile found; scanning capabilities…".into();
                                    self.request_capabilities();
                                }
                            }
                            if let Some(devices) = response["data"].get("devices") {
                                match serde_json::from_value(devices.clone()) {
                                    Ok(items) => {
                                        self.devices = items;
                                        self.metadata_devices_received = true;
                                        self.ensure_selected_gpu();
                                        self.status = "Hardware capabilities loaded.".into();
                                    }
                                    Err(problem) => self.status = problem.to_string(),
                                }
                            }
                            if let Some(benchmarks) = response["data"].get("benchmarks") {
                                match serde_json::from_value(benchmarks.clone()) {
                                    Ok(items) => {
                                        self.benchmarks = items;
                                        self.metadata_benchmarks_received = true;
                                        self.select_first_available();
                                    }
                                    Err(problem) => self.status = problem.to_string(),
                                }
                            }
                            if self.metadata_scan_pending
                                && self.metadata_devices_received
                                && self.metadata_benchmarks_received
                            {
                                self.metadata_scan_pending = false;
                                if let Some(fingerprint) = &self.fingerprint {
                                    match save_metadata_cache(fingerprint, &self.devices, &self.benchmarks) {
                                        Ok(()) => self.status = "Hardware capabilities scanned and saved for future launches.".into(),
                                        Err(problem) => self.status = format!("Hardware scan completed, but metadata could not be saved: {problem}"),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if continue_queue {
            self.start_next();
        }
    }
    fn refresh(window: &MainWindow, app: &Self) {
        window.set_page(app.page);
        window.set_status(app.status.clone().into());
        window.set_worker_online(app.worker.online());
        window.set_running(app.active_request.is_some());
        window.set_progress(app.progress);
        window.set_progress_message(app.progress_message.clone().into());
        window.set_queued_count(app.queue.len() as i32);
        window.set_suite_description(suite_description(&app.selected_suite).into());
        window.set_show_gpu_selector(app.selected_suite.starts_with("gpu."));
        window.set_expanded_device_id(app.expanded_device_id.clone().unwrap_or_default().into());
        window.set_result_details_expanded(app.result_details_expanded);
        window.set_device_count(app.devices.len() as i32);
        window
            .set_available_device_count(app.devices.iter().filter(|d| d.available).count() as i32);
        window.set_benchmark_count(app.benchmarks.len() as i32);
        let gpu_devices: Vec<_> = app
            .devices
            .iter()
            .filter(|d| d.category == DeviceCategory::Gpu && d.available)
            .collect();
        let gpu_names: Vec<SharedString> = gpu_devices
            .iter()
            .map(|d| SharedString::from(d.name.as_str()))
            .collect();
        window.set_gpu_adapters(ModelRc::new(VecModel::from(gpu_names)));
        window.set_selected_gpu(
            app.selected_gpu_id
                .as_ref()
                .and_then(|id| gpu_devices.iter().position(|d| &d.id == id))
                .map(|i| i as i32)
                .unwrap_or(-1),
        );
        window.set_selected_gpu_name(
            app.selected_gpu_id
                .as_ref()
                .and_then(|id| gpu_devices.iter().find(|device| &device.id == id))
                .map(|device| device.name.as_str())
                .unwrap_or("No compatible GPU selected")
                .into(),
        );
        let devices: Vec<DeviceRow> = app
            .devices
            .iter()
            .map(|d| DeviceRow {
                id: d.id.as_str().into(),
                name: d.name.as_str().into(),
                category: category_name(d.category).into(),
                availability: if d.available {
                    "AVAILABLE"
                } else {
                    "UNAVAILABLE"
                }
                .into(),
                status: d.status.as_str().into(),
                details: d
                    .properties
                    .iter()
                    .take(4)
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<_>>()
                    .join(" | ")
                    .into(),
                cache_details: cache_details(d).into(),
                accent: device_accent(d.category),
                available: d.available,
            })
            .collect();
        window.set_devices(ModelRc::new(VecModel::from(devices)));
        let benchmarks: Vec<BenchmarkRow> = app
            .benchmarks
            .iter()
            .filter(|b| b.suite_id == app.selected_suite)
            .map(|b| {
                let hardware_ready =
                    benchmark_hardware_supported_for(b, app.selected_gpu_id.as_deref())
                        && !app.benchmark_available(b);
                BenchmarkRow {
                    id: b.id.as_str().into(),
                    name: b.name.as_str().into(),
                    workload: b.workload.as_str().into(),
                    availability: if hardware_ready {
                        "Hardware supported | Vulkan runner pending"
                    } else if b.available {
                        "unsupported on selected adapter"
                    } else {
                        b.unavailable_reason.as_str()
                    }
                    .into(),
                    data_type: b.data_type.as_str().into(),
                    unit: b.unit.as_str().into(),
                    enabled: app.benchmark_available(b),
                    selected: app.selected.as_deref() == Some(b.id.as_str()),
                    hardware_ready,
                }
            })
            .collect();
        window.set_benchmarks(ModelRc::new(VecModel::from(benchmarks)));
        let selected = app
            .selected
            .as_ref()
            .and_then(|id| app.benchmarks.iter().find(|benchmark| &benchmark.id == id));
        window.set_selected_benchmark_name(
            selected
                .map(|benchmark| benchmark.name.as_str())
                .unwrap_or("No workload selected")
                .into(),
        );
        window.set_selected_benchmark_description(
            selected
                .map(|benchmark| benchmark.workload.as_str())
                .unwrap_or("Select an available workload to inspect and run it.")
                .into(),
        );
        window.set_selected_benchmark_availability(
            selected
                .map(|benchmark| {
                    if app.benchmark_available(benchmark) {
                        format!("Available | {} | {}", benchmark.data_type, benchmark.unit)
                    } else if benchmark_hardware_supported_for(
                        benchmark,
                        app.selected_gpu_id.as_deref(),
                    ) {
                        format!(
                            "Hardware supported | Vulkan runner pending | {}",
                            benchmark.data_type
                        )
                    } else {
                        format!("Unavailable: {}", benchmark.unavailable_reason)
                    }
                })
                .unwrap_or_default()
                .into(),
        );
        window.set_selected_benchmark_enabled(
            selected.is_some_and(|benchmark| app.benchmark_available(benchmark)),
        );
        let target_device = if app.selected_suite.starts_with("gpu.") {
            app.selected_gpu_id
                .as_ref()
                .and_then(|id| app.devices.iter().find(|device| &device.id == id))
        } else {
            let category = selected.map(|benchmark| benchmark.category);
            category
                .and_then(|category| {
                    app.devices.iter().find(|device| {
                        matches!(
                            (category, device.category),
                            (BenchmarkCategory::Cpu, DeviceCategory::Cpu)
                                | (BenchmarkCategory::Memory, DeviceCategory::Memory)
                        )
                    })
                })
                .or_else(|| {
                    app.devices
                        .iter()
                        .find(|device| device.category == DeviceCategory::Cpu)
                })
        };
        window.set_target_hardware_name(
            target_device
                .map(|device| device.name.as_str())
                .unwrap_or("No compatible target detected")
                .into(),
        );
        window.set_target_hardware_description(
            if app.selected_suite.starts_with("gpu.") {
                if target_device.is_some() {
                    "GPU workloads run on the selected adapter above."
                } else {
                    "No compatible GPU adapter is available for this suite."
                }
            } else if selected
                .is_some_and(|benchmark| benchmark.category == BenchmarkCategory::Memory)
            {
                "This workload measures CPU-visible system memory."
            } else {
                "CPU workloads run on the discovered processor."
            }
            .into(),
        );
        let results: Vec<ResultRow> = app
            .results
            .iter()
            .rev()
            .map(|result| result_row(result, &app.devices))
            .collect();
        window.set_results(ModelRc::new(VecModel::from(results)));
    }
}

fn benchmark_available_for(benchmark: &BenchmarkDescriptor, selected_gpu_id: Option<&str>) -> bool {
    benchmark.available
        && (benchmark.category != BenchmarkCategory::Gpu
            || selected_gpu_id.is_some_and(|id| {
                benchmark
                    .supported_device_ids
                    .iter()
                    .any(|supported| supported == id)
            }))
}

fn benchmark_hardware_supported_for(
    benchmark: &BenchmarkDescriptor,
    selected_gpu_id: Option<&str>,
) -> bool {
    let Some(selected_gpu_id) = selected_gpu_id else {
        return false;
    };
    benchmark
        .metadata
        .get("hardware_supported_device_ids")
        .is_some_and(|ids| ids.split(',').any(|id| id == selected_gpu_id))
}

fn preferred_gpu_id(devices: &[DeviceDescriptor], current: Option<&str>) -> Option<String> {
    current
        .and_then(|id| {
            devices
                .iter()
                .find(|device| {
                    device.category == DeviceCategory::Gpu && device.available && device.id == id
                })
                .map(|device| device.id.clone())
        })
        .or_else(|| {
            devices
                .iter()
                .find(|device| device.category == DeviceCategory::Gpu && device.available)
                .map(|device| device.id.clone())
        })
}

fn category_name(category: DeviceCategory) -> &'static str {
    match category {
        DeviceCategory::Cpu => "CPU",
        DeviceCategory::Memory => "MEMORY",
        DeviceCategory::Gpu => "GPU",
    }
}
fn device_accent(category: DeviceCategory) -> Color {
    match category {
        DeviceCategory::Cpu => Color::from_rgb_u8(108, 158, 255),
        DeviceCategory::Memory => Color::from_rgb_u8(85, 220, 174),
        DeviceCategory::Gpu => Color::from_rgb_u8(209, 155, 255),
    }
}
fn cache_details(device: &DeviceDescriptor) -> String {
    if device.caches.is_empty() {
        return "Cache topology is not exposed by this device; benchmarking remains available where supported.".into();
    }
    device
        .caches
        .iter()
        .map(|cache| {
            format!(
                "L{} {:?}: {} KiB, {} instance(s), {} B lines",
                cache.level,
                cache.kind,
                cache.size_bytes / 1024,
                cache.instances,
                cache.line_size_bytes
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}
fn suite_description(suite: &str) -> &'static str {
    match suite {
        "cpu.bandwidth" => "Read, write, and copy bandwidth with per-operation sample statistics.",
        "cpu.performance" => {
            "Aggregate processor throughput, single-thread INT64, and compute-vs-memory diagnosis."
        }
        "gpu.bandwidth" => "Estimated cache, GPU-local memory, and host-device bandwidth.",
        "gpu.performance" => {
            "Vector shader and cooperative-matrix throughput, capability-gated per format."
        }
        _ => "Select a benchmark suite.",
    }
}
fn result_row(result: &BenchmarkResult, devices: &[DeviceDescriptor]) -> ResultRow {
    let primary = result.metrics.first();
    let secondary_metrics = result
        .metrics
        .iter()
        .skip(1)
        .map(|metric| {
            format!(
                "{}: {}",
                metric.name,
                format_metric(metric.value, &metric.unit)
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let diagnosis = result
        .workload_metadata
        .get("bound_classification")
        .map(|s| format!("Bound diagnosis: {}", s.replace('_', " ")))
        .unwrap_or_else(|| "Five-sample statistics recorded.".into());
    let details = result
        .workload_metadata
        .iter()
        .chain(result.device_metadata.iter())
        .take(5)
        .map(|(key, value)| format!("{key}: {value}"))
        .collect::<Vec<_>>()
        .join(" | ");
    let device = devices
        .iter()
        .find(|device| device.id == result.device_id)
        .map(|device| device.name.as_str())
        .unwrap_or(result.device_id.as_str());
    ResultRow {
        title: result.benchmark_id.as_str().into(),
        device: device.into(),
        elapsed: format!("{:.2} ms", result.elapsed_ns as f64 / 1e6).into(),
        primary_name: primary
            .map(|metric| metric.name.as_str())
            .unwrap_or("No metric reported")
            .into(),
        primary_value: primary
            .map(|metric| format_metric(metric.value, &metric.unit))
            .unwrap_or_else(|| "N/A".into())
            .into(),
        statistics: primary
            .map(|metric| {
                format!(
                    "{} samples | min {} | median {} | max {} | variation {:.2}",
                    metric.statistics.sample_count,
                    format_metric(metric.statistics.minimum, &metric.unit),
                    format_metric(metric.statistics.median, &metric.unit),
                    format_metric(metric.statistics.maximum, &metric.unit),
                    metric.statistics.standard_deviation
                )
            })
            .unwrap_or_else(|| "No sample statistics reported.".into())
            .into(),
        secondary_metrics: if secondary_metrics.is_empty() {
            "No secondary metrics reported."
        } else {
            &secondary_metrics
        }
        .into(),
        diagnosis: diagnosis.into(),
        details: if details.is_empty() {
            "No additional workload metadata reported."
        } else {
            &details
        }
        .into(),
    }
}
fn worker_path() -> PathBuf {
    let mut path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    path.set_file_name(if cfg!(windows) {
        "gluj-bench-worker.exe"
    } else {
        "gluj-bench-worker"
    });
    path
}

fn metadata_cache_path() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(|path| {
        PathBuf::from(path)
            .join("Gluj-Bench")
            .join("hardware-metadata.json")
    })
}

fn load_metadata_cache() -> Option<HardwareMetadataCache> {
    let path = metadata_cache_path()?;
    let content = fs::read_to_string(path).ok()?;
    let cache: HardwareMetadataCache = serde_json::from_str(&content).ok()?;
    (cache.version == HARDWARE_METADATA_CACHE_VERSION).then_some(cache)
}

fn save_metadata_cache(
    fingerprint: &str,
    devices: &[DeviceDescriptor],
    benchmarks: &[BenchmarkDescriptor],
) -> Result<(), String> {
    let path = metadata_cache_path().ok_or_else(|| "APPDATA is unavailable".to_owned())?;
    let directory = path
        .parent()
        .ok_or_else(|| "cache directory is unavailable".to_owned())?;
    fs::create_dir_all(directory).map_err(|problem| problem.to_string())?;
    let content = serde_json::to_vec_pretty(&HardwareMetadataCache {
        version: HARDWARE_METADATA_CACHE_VERSION,
        fingerprint: fingerprint.to_owned(),
        devices: devices.to_vec(),
        benchmarks: benchmarks.to_vec(),
    })
    .map_err(|problem| problem.to_string())?;
    fs::write(path, content).map_err(|problem| problem.to_string())
}
fn format_metric(value: f64, unit: &str) -> String {
    match unit {
        "bytes/s" => format!("{:.2} GB/s", value / 1e9),
        "operations/s" => readable_rate(value, "OPS"),
        "MB/s" => format!("{value:.2} MB/s"),
        "strings/s" => readable_rate(value, "strings/s"),
        "primes/s" => readable_rate(value, "primes/s"),
        _ => format!("{value:.2} {unit}"),
    }
}
fn readable_rate(value: f64, suffix: &str) -> String {
    let (scale, prefix) = if value >= 1e12 {
        (1e12, "T")
    } else if value >= 1e6 {
        (1e6, "M")
    } else if value >= 1e3 {
        (1e3, "K")
    } else {
        (1., "")
    };
    format!("{:.2} {prefix}{suffix}", value / scale)
}

#[cfg(test)]
mod tests {
    use super::{
        benchmark_available_for, benchmark_hardware_supported_for, cache_details, format_metric,
        preferred_gpu_id, result_row,
    };
    use gluj_bench_core::{
        BenchmarkCategory, BenchmarkDescriptor, BenchmarkResult, CacheDescriptor, CacheKind,
        DeviceCategory, DeviceDescriptor, Metric, SampleStatistics,
    };
    use std::collections::BTreeMap;
    #[test]
    fn operation_rates_use_readable_si_prefixes() {
        assert_eq!(format_metric(12_500.0, "operations/s"), "12.50 KOPS");
        assert_eq!(
            format_metric(3_200_000_000_000.0, "operations/s"),
            "3.20 TOPS"
        );
    }

    #[test]
    fn gpu_workload_requires_the_selected_supported_adapter() {
        let benchmark = BenchmarkDescriptor {
            id: "gpu.vector.fp32".into(),
            name: "FP32".into(),
            category: BenchmarkCategory::Gpu,
            workload: "test".into(),
            data_type: "FP32".into(),
            unit: "operations/s".into(),
            supported_device_ids: vec!["gpu:one".into()],
            available: true,
            unavailable_reason: String::new(),
            suite_id: "gpu.performance".into(),
            display_order: 0,
            metadata: BTreeMap::new(),
        };
        assert!(benchmark_available_for(&benchmark, Some("gpu:one")));
        assert!(!benchmark_available_for(&benchmark, Some("gpu:two")));
        assert!(!benchmark_available_for(&benchmark, None));
    }

    #[test]
    fn capability_metadata_does_not_hide_hardware_support() {
        let benchmark = BenchmarkDescriptor {
            id: "gpu.performance.matrix.int8".into(),
            name: "Dense INT8 matrix performance".into(),
            category: BenchmarkCategory::Gpu,
            workload: "test".into(),
            data_type: "int8".into(),
            unit: "operations/s".into(),
            supported_device_ids: Vec::new(),
            available: false,
            unavailable_reason: "capability_gated_runner_unavailable".into(),
            suite_id: "gpu.performance".into(),
            display_order: 0,
            metadata: BTreeMap::from([(
                "hardware_supported_device_ids".into(),
                "gpu:vulkan:xtx".into(),
            )]),
        };
        assert!(benchmark_hardware_supported_for(
            &benchmark,
            Some("gpu:vulkan:xtx")
        ));
        assert!(!benchmark_hardware_supported_for(
            &benchmark,
            Some("gpu:vulkan:other")
        ));
        assert!(!benchmark_available_for(&benchmark, Some("gpu:vulkan:xtx")));
    }

    #[test]
    fn cached_gpu_metadata_selects_the_first_available_adapter() {
        let devices = [DeviceDescriptor {
            id: "gpu:one".into(),
            name: "Graphics adapter".into(),
            category: DeviceCategory::Gpu,
            available: true,
            status: "Ready".into(),
            properties: BTreeMap::new(),
            caches: vec![],
        }];
        assert_eq!(preferred_gpu_id(&devices, None).as_deref(), Some("gpu:one"));
        assert_eq!(
            preferred_gpu_id(&devices, Some("missing")).as_deref(),
            Some("gpu:one")
        );
    }

    #[test]
    fn result_rows_preserve_statistics_and_cache_discovery_context() {
        let result = BenchmarkResult {
            benchmark_id: "gpu.bandwidth.cache".into(),
            device_id: "gpu:one".into(),
            elapsed_ns: 2_000_000,
            metrics: vec![Metric {
                name: "Bandwidth".into(),
                value: 1_000_000_000.0,
                unit: "bytes/s".into(),
                statistics: SampleStatistics {
                    sample_count: 5,
                    minimum: 900_000_000.0,
                    median: 1_000_000_000.0,
                    maximum: 1_100_000_000.0,
                    standard_deviation: 10.0,
                },
            }],
            workload_metadata: BTreeMap::from([(
                "cache_discovery_status".into(),
                "not_detected".into(),
            )]),
            device_metadata: BTreeMap::new(),
        };
        let device = DeviceDescriptor {
            id: "gpu:one".into(),
            name: "Integrated GPU".into(),
            category: DeviceCategory::Gpu,
            available: true,
            status: "Ready".into(),
            properties: BTreeMap::new(),
            caches: vec![],
        };
        let row = result_row(&result, &[device]);
        assert_eq!(row.device.as_str(), "Integrated GPU");
        assert!(row.statistics.as_str().contains("5 samples"));
        assert!(
            row.details
                .as_str()
                .contains("cache_discovery_status: not_detected")
        );
    }

    #[test]
    fn empty_cache_topology_is_an_explanatory_state() {
        let mut device = DeviceDescriptor {
            id: "gpu:one".into(),
            name: "Integrated GPU".into(),
            category: DeviceCategory::Gpu,
            available: true,
            status: "Ready".into(),
            properties: BTreeMap::new(),
            caches: vec![],
        };
        assert!(cache_details(&device).contains("not exposed"));
        let cache = CacheDescriptor {
            level: 3,
            kind: CacheKind::Unified,
            size_bytes: 32 * 1024 * 1024,
            line_size_bytes: 64,
            sharing_logical_processors: 16,
            instances: 1,
        };
        device.caches.push(cache);
        assert!(cache_details(&device).contains("L3 Unified"));
    }
}
