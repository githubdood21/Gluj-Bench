#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use gluj_bench_core::{
    BenchmarkCategory, BenchmarkDescriptor, BenchmarkResult, DeviceCategory, DeviceDescriptor,
    Metric, PROTOCOL_VERSION,
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
    time::{Duration, SystemTime, UNIX_EPOCH},
};

slint::include_modules!();

const HARDWARE_METADATA_CACHE_VERSION: u32 = 10;
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
    saved_results: Vec<BenchmarkResult>,
    baseline_results: Vec<BenchmarkResult>,
    comparison_selection: usize,
    results_file_status: String,
    results_file_writable: bool,
    settings: AppSettings,
    settings_status: String,
    settings_writable: bool,
    result_component: usize,
    result_page: usize,
    result_selection: usize,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct AppSettings {
    cpu_intensity: usize,
    gpu_intensity: usize,
    vram_budget_percent: u32,
}
impl Default for AppSettings {
    fn default() -> Self {
        Self {
            cpu_intensity: 1,
            gpu_intensity: 1,
            vram_budget_percent: 25,
        }
    }
}
const INTENSITY_PERCENT: [u32; 3] = [50, 75, 100];
const INTENSITY_NAMES: [&str; 3] = ["Gentle · 50%", "Balanced · 75%", "Full · 100%"];
const VRAM_BUDGET_OPTIONS: [u32; 8] = [20, 25, 30, 40, 50, 60, 70, 80];

fn settings_path() -> PathBuf {
    results_path().with_file_name("settings.json")
}

fn load_settings(path: &std::path::Path) -> Result<AppSettings, String> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(AppSettings::default());
        }
        Err(error) => return Err(error.to_string()),
    };
    let settings: AppSettings = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if settings.cpu_intensity >= 3
        || settings.gpu_intensity >= 3
        || !(20..=80).contains(&settings.vram_budget_percent)
    {
        return Err("invalid intensity setting".into());
    }
    Ok(settings)
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
            saved_results: vec![],
            baseline_results: vec![],
            comparison_selection: 0,
            results_file_status: "Results save automatically to results.json beside the app."
                .into(),
            results_file_writable: true,
            settings: AppSettings::default(),
            settings_status: "Settings save to settings.json beside the app.".into(),
            settings_writable: true,
            result_component: 0,
            result_page: 0,
            result_selection: 0,
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
        match load_settings(&settings_path()) {
            Ok(settings) => app.settings = settings,
            Err(error) => {
                app.settings_writable = false;
                app.settings_status = format!(
                    "Using Balanced defaults. Could not load settings.json: {error}. Existing file preserved."
                );
            }
        }
        match load_results(&results_path()) {
            Ok(results) => {
                app.results = results.clone();
                app.baseline_results = results.clone();
                app.saved_results = results;
            }
            Err(error) => {
                app.results_file_writable = false;
                app.results_file_status = format!(
                    "Could not load results.json: {error}. Existing file has been preserved."
                );
            }
        }
        app
    }
    fn wire(window: &MainWindow, app: &Rc<RefCell<Self>>) {
        let app_weak = Rc::downgrade(app);
        window.on_select_cpu_intensity(move |index| {
            if let Some(app) = app_weak.upgrade() {
                app.borrow_mut().set_intensity(index, false);
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_select_gpu_intensity(move |index| {
            if let Some(app) = app_weak.upgrade() {
                app.borrow_mut().set_intensity(index, true);
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_select_vram_budget(move |index| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                if app.active_request.is_some() || !(0..8).contains(&index) {
                    return;
                }
                app.settings.vram_budget_percent = VRAM_BUDGET_OPTIONS[index as usize];
                app.persist_settings();
            }
        });
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
        window.on_select_result_component(move |index| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.result_component = index.max(0) as usize;
                app.result_page = 0;
                app.result_selection = 0;
                app.comparison_selection = 0;
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_select_comparison(move |index| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.comparison_selection = index.max(0) as usize;
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_select_result_row(move |index| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.result_selection = app.result_page * RESULT_PAGE_SIZE + index.max(0) as usize;
                app.result_details_expanded = false;
                app.ui_dirty = true;
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_change_result_page(move |delta| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                app.result_page = app.result_page.saturating_add_signed(delta as isize);
                app.result_selection = app.result_page * RESULT_PAGE_SIZE;
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
                app.result_component = 0;
                app.result_page = 0;
                app.result_selection = 0;
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
    fn set_intensity(&mut self, index: i32, gpu: bool) {
        if self.active_request.is_some() || !(0..3).contains(&index) {
            return;
        }
        if gpu {
            self.settings.gpu_intensity = index as usize;
        } else {
            self.settings.cpu_intensity = index as usize;
        }
        self.persist_settings();
    }
    fn persist_settings(&mut self) {
        if self.settings_writable {
            self.settings_status = match write_json(&settings_path(), &self.settings) {
                Ok(()) => "Settings saved. They apply to the next test.".into(),
                Err(error) => {
                    format!("Settings apply for this session, but could not be saved: {error}")
                }
            };
        }
        self.ui_dirty = true;
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
        options.insert(
            "vram_budget_percent".into(),
            self.settings.vram_budget_percent.to_string().into(),
        );
        options.insert(
            "cpu_worker_percent".into(),
            INTENSITY_PERCENT[self.settings.cpu_intensity]
                .to_string()
                .into(),
        );
        options.insert(
            "gpu_activity_percent".into(),
            INTENSITY_PERCENT[self.settings.gpu_intensity]
                .to_string()
                .into(),
        );
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
                                Ok(mut result) => {
                                    stamp_result(
                                        &mut result,
                                        &self.devices,
                                        self.fingerprint.as_deref(),
                                    );
                                    remember_result(&mut self.saved_results, &result);
                                    if self.results_file_writable {
                                        self.results_file_status = match save_results(&results_path(), &self.saved_results) {
                                            Ok(()) => "Saved to results.json beside the app. Changes compare against results loaded at startup.".into(),
                                            Err(error) => format!("Results are available here, but could not be saved: {error}"),
                                        };
                                    }
                                    self.results.push(result);
                                    let latest = self.results.last().unwrap();
                                    let key = result_component_key(latest);
                                    let benchmark = latest.benchmark_id.clone();
                                    let mut keys = Vec::new();
                                    for result in &self.results {
                                        let key = result_component_key(result);
                                        if !keys.contains(&key) {
                                            keys.push(key);
                                        }
                                    }
                                    self.result_component =
                                        keys.iter().position(|id| *id == key).unwrap_or(0);
                                    let groups = component_results(&self.results, &key);
                                    self.result_selection = groups
                                        .iter()
                                        .position(|g| g[0].benchmark_id == benchmark)
                                        .unwrap_or(0);
                                    self.result_page = self.result_selection / RESULT_PAGE_SIZE;
                                    self.comparison_selection = 0;
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
                                    self.status =
                                        "Checking hardware for saved result comparisons…".into();
                                    self.request_capabilities();
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
        window
            .set_vram_budget_name(format!("{}% of VRAM", app.settings.vram_budget_percent).into());
        window.set_cpu_intensity_name(INTENSITY_NAMES[app.settings.cpu_intensity].into());
        window.set_gpu_intensity_name(INTENSITY_NAMES[app.settings.gpu_intensity].into());
        window.set_settings_status(app.settings_status.clone().into());
        window.set_load_description(
            format!(
                "CPU workers: {}% · GPU activity target: {}%",
                INTENSITY_PERCENT[app.settings.cpu_intensity],
                INTENSITY_PERCENT[app.settings.gpu_intensity]
            )
            .into(),
        );
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
        let mut components: Vec<String> = Vec::new();
        for result in &app.results {
            let component = result_component_key(result);
            if !components.contains(&component) {
                components.push(component);
            }
        }
        let component_index = app.result_component.min(components.len().saturating_sub(1));
        let names: Vec<SharedString> = components
            .iter()
            .enumerate()
            .map(|(index, id)| {
                let result = app
                    .results
                    .iter()
                    .find(|r| result_component_key(r) == *id)
                    .unwrap();
                let label = component_label(result, &app.devices);
                if components
                    .iter()
                    .filter(|key| {
                        app.results
                            .iter()
                            .find(|r| result_component_key(r) == **key)
                            .is_some_and(|r| component_label(r, &app.devices) == label)
                    })
                    .count()
                    > 1
                {
                    format!("{label} · setup {}", index + 1).into()
                } else {
                    label.into()
                }
            })
            .collect();
        window.set_result_components(ModelRc::new(VecModel::from(names.clone())));
        window.set_result_component_name(names.get(component_index).cloned().unwrap_or_default());
        window.set_result_count(app.results.len() as i32);
        window.set_results_file_status(app.results_file_status.clone().into());
        let groups = component_results(
            &app.results,
            components
                .get(component_index)
                .map(String::as_str)
                .unwrap_or(""),
        );
        let category = groups.first().map(|g| component_family(g[0]));
        let mut baseline_keys = Vec::new();
        let mut comparison_names: Vec<SharedString> = vec!["Last saved score for each test".into()];
        for result in app
            .baseline_results
            .iter()
            .rev()
            .filter(|r| Some(component_family(r)) == category)
        {
            let key = result_component_key(result);
            if !baseline_keys.contains(&key) {
                baseline_keys.push(key);
                comparison_names.push(
                    format!(
                        "{} · saved setup {}",
                        component_label(result, &app.devices),
                        baseline_keys.len()
                    )
                    .into(),
                );
            }
        }
        let comparison_index = app.comparison_selection.min(comparison_names.len() - 1);
        let comparison_key = comparison_index
            .checked_sub(1)
            .and_then(|i| baseline_keys.get(i))
            .map(String::as_str);
        window.set_comparison_name(comparison_names[comparison_index].clone());
        window.set_comparison_options(ModelRc::new(VecModel::from(comparison_names)));
        let page_count = groups.len().div_ceil(RESULT_PAGE_SIZE).max(1);
        let page = app.result_page.min(page_count - 1);
        let selection = app.result_selection.min(groups.len().saturating_sub(1));
        let rows = groups
            .iter()
            .skip(page * RESULT_PAGE_SIZE)
            .take(RESULT_PAGE_SIZE)
            .map(|group| {
                let latest = result_row(group[0], &app.devices, &app.benchmarks);
                SummaryRow {
                    section: result_section(group[0]),
                    bandwidth: ["read", "write", "copy"].iter().all(|name| {
                        group[0]
                            .metrics
                            .iter()
                            .any(|m| m.name == *name && m.unit == "bytes/s")
                    }),
                    read_latest: bandwidth_score(group, "read", false).into(),
                    write_latest: bandwidth_score(group, "write", false).into(),
                    copy_latest: bandwidth_score(group, "copy", false).into(),
                    read_best: bandwidth_score(group, "read", true).into(),
                    write_best: bandwidth_score(group, "write", true).into(),
                    copy_best: bandwidth_score(group, "copy", true).into(),
                    delta: primary_delta(group[0], &app.baseline_results, comparison_key).into(),
                    read_delta: metric_delta(
                        group[0],
                        "read",
                        &app.baseline_results,
                        comparison_key,
                    )
                    .into(),
                    write_delta: metric_delta(
                        group[0],
                        "write",
                        &app.baseline_results,
                        comparison_key,
                    )
                    .into(),
                    copy_delta: metric_delta(
                        group[0],
                        "copy",
                        &app.baseline_results,
                        comparison_key,
                    )
                    .into(),
                    title: if group[0].benchmark_id.ends_with(".scaling") {
                        format!(
                            "{} · {}",
                            latest.title,
                            scaling_compute_tiers(group[0])
                                .last()
                                .map(|(bytes, _)| format_binary_size(*bytes as f64))
                                .unwrap_or_default()
                        )
                        .into()
                    } else {
                        latest.title
                    },
                    metric: latest.primary_name,
                    latest: latest.primary_value,
                    best: best_primary(group)
                        .map(|m| format_metric(m.value, &m.unit))
                        .unwrap_or_else(|| "—".into())
                        .into(),
                    runs: group.len().to_string().into(),
                }
            })
            .collect::<Vec<_>>();
        let headings = rows
            .iter()
            .enumerate()
            .filter(|(i, row)| *i == 0 || rows[*i - 1].section != row.section)
            .count();
        window.set_result_header_count(headings as i32);
        window.set_result_has_bandwidth(rows.iter().any(|row| row.bandwidth));
        window.set_result_summaries(ModelRc::new(VecModel::from(rows)));
        window.set_result_page(page as i32);
        window.set_result_page_count(page_count as i32);
        window.set_result_selection(selection as i32 - (page * RESULT_PAGE_SIZE) as i32);
        window.set_comparison_note(groups.get(selection).map(|group| {
            let current = group[0];
            let Some(metric) = featured_metric(current) else { return "No metric available for comparison.".into(); };
            let Some((old, previous)) = comparison_metric(current, &metric.name, &app.baseline_results, comparison_key) else {
                return "Run this test again after restarting the app to compare it with a saved result. — means no matching saved measurement or compatible settings.".into();
            };
            format!("Compared with saved {}: {} → {} ({}).", component_label(old, &app.devices),
                format_metric(previous.value, &previous.unit), format_metric(metric.value, &metric.unit),
                metric_delta(current, &metric.name, &app.baseline_results, comparison_key)).into()
        }).unwrap_or_default());
        window.set_selected_result(
            groups
                .get(selection)
                .map(|group| result_row(group[0], &app.devices, &app.benchmarks))
                .unwrap_or_default(),
        );
    }
}

const RESULT_PAGE_SIZE: usize = 16;

#[derive(Serialize, Deserialize)]
struct ResultsFile {
    version: u32,
    results: Vec<BenchmarkResult>,
}

fn results_path() -> PathBuf {
    let mut path = worker_path();
    path.set_file_name("results.json");
    path
}

fn load_results(path: &std::path::Path) -> Result<Vec<BenchmarkResult>, String> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(error) => return Err(error.to_string()),
    };
    let file: ResultsFile = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if file.version != 1 {
        return Err("unsupported results file version".into());
    }
    Ok(file.results)
}

fn save_results(path: &std::path::Path, results: &[BenchmarkResult]) -> Result<(), String> {
    write_json(
        path,
        &ResultsFile {
            version: 1,
            results: results.to_vec(),
        },
    )
}

fn write_json<T: Serialize>(path: &std::path::Path, value: &T) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
    fs::write(&temporary, bytes).map_err(|e| e.to_string())?;
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error.to_string());
    }
    Ok(())
}

fn stamp_result(
    result: &mut BenchmarkResult,
    devices: &[DeviceDescriptor],
    fingerprint: Option<&str>,
) {
    let name = component_label(result, devices);
    let device_name = devices
        .iter()
        .find(|d| d.id == result.device_id)
        .map(|d| d.name.clone())
        .unwrap_or_else(|| result.device_id.clone());
    // Exclude changing free-memory and health fields from hardware identity.
    let hardware: Vec<_> = devices
        .iter()
        .filter(|device| {
            if component_family(result) == "cpu" {
                matches!(
                    device.category,
                    DeviceCategory::Cpu | DeviceCategory::Memory
                )
            } else {
                device.id == result.device_id
            }
        })
        .map(|device| {
            let properties: std::collections::BTreeMap<_, _> = device
                .properties
                .iter()
                .filter(|(key, _)| key.as_str() != "available_bytes_at_discovery")
                .collect();
            (&device.id, &device.name, properties, &device.caches)
        })
        .collect();
    let identity =
        serde_json::to_vec(&(fingerprint.unwrap_or_default(), hardware)).unwrap_or_default();
    let hash = identity.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    let setup = format!("{hash:016x}");
    result
        .device_metadata
        .insert("saved_component_name".into(), name);
    result
        .device_metadata
        .insert("saved_device_name".into(), device_name);
    result.device_metadata.insert("saved_setup".into(), setup);
    result.device_metadata.insert(
        "saved_at".into(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .to_string(),
    );
}

fn remember_result(saved: &mut Vec<BenchmarkResult>, result: &BenchmarkResult) {
    saved.retain(|old| {
        old.benchmark_id != result.benchmark_id
            || result_component_key(old) != result_component_key(result)
            || result_load_signature(old) != result_load_signature(result)
    });
    saved.push(result.clone());
    // Keep one latest result per workload/setup, with a bounded portable archive.
    if saved.len() > 128 {
        saved.drain(..saved.len() - 128);
    }
}

fn comparison_metric<'a>(
    current: &BenchmarkResult,
    name: &str,
    saved: &'a [BenchmarkResult],
    key: Option<&str>,
) -> Option<(&'a BenchmarkResult, &'a Metric)> {
    let metric = current.metrics.iter().find(|m| m.name == name)?;
    saved
        .iter()
        .rev()
        .filter(|old| {
            old.benchmark_id == current.benchmark_id
                && component_family(old) == component_family(current)
                && result_load_signature(old) == result_load_signature(current)
                && key.is_none_or(|key| result_component_key(old) == key)
                && [
                    "data_type",
                    "thread_mode",
                    "matrix_m",
                    "matrix_n",
                    "matrix_k",
                    "copy_byte_definition",
                    "clock_analysis_revision",
                    "arithmetic_iterations",
                    "matrix_tile_reuse",
                ]
                .iter()
                .all(|key| old.workload_metadata.get(*key) == current.workload_metadata.get(*key))
        })
        .find_map(|old| {
            old.metrics
                .iter()
                .find(|m| {
                    m.name == name && m.unit == metric.unit && m.value.is_finite() && m.value > 0.0
                })
                .map(|metric| (old, metric))
        })
}

fn metric_delta(
    current: &BenchmarkResult,
    name: &str,
    saved: &[BenchmarkResult],
    key: Option<&str>,
) -> String {
    let Some(metric) = current
        .metrics
        .iter()
        .find(|m| m.name == name && m.value.is_finite())
    else {
        return "—".into();
    };
    let Some((_, previous)) = comparison_metric(current, name, saved, key) else {
        return "—".into();
    };
    let change = (metric.value / previous.value - 1.0) * 100.0;
    if change.is_finite() {
        format!("{change:+.1}%")
    } else {
        "—".into()
    }
}

fn primary_delta(
    current: &BenchmarkResult,
    saved: &[BenchmarkResult],
    key: Option<&str>,
) -> String {
    featured_metric(current)
        .filter(|m| m.unit.ends_with("/s"))
        .map(|m| metric_delta(current, &m.name, saved, key))
        .unwrap_or_else(|| "—".into())
}

fn result_component_id(device_id: &str) -> &str {
    match device_id {
        "memory:system" => "cpu:system",
        _ => device_id,
    }
}

fn result_component_key(result: &BenchmarkResult) -> String {
    let id = result_component_id(&result.device_id);
    result
        .device_metadata
        .get("saved_setup")
        .map(|setup| format!("{id}|{setup}"))
        .unwrap_or_else(|| id.to_owned())
}

fn component_family(result: &BenchmarkResult) -> &str {
    if result.device_id.starts_with("gpu:") {
        "gpu"
    } else {
        "cpu"
    }
}

fn result_load_percent(result: &BenchmarkResult) -> u32 {
    let key = if component_family(result) == "gpu" {
        "gpu_activity_percent"
    } else {
        "cpu_worker_percent"
    };
    result
        .workload_metadata
        .get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(100)
}

fn result_load_signature(result: &BenchmarkResult) -> (u32, u32) {
    let vram = if component_family(result) == "gpu" && result.benchmark_id.ends_with(".scaling") {
        result
            .workload_metadata
            .get("vram_budget_percent")
            .and_then(|value| value.parse().ok())
            .unwrap_or(25)
    } else {
        0
    };
    (result_load_percent(result), vram)
}

fn result_settings_note(result: &BenchmarkResult) -> String {
    let (load, vram) = result_load_signature(result);
    let target = if component_family(result) == "gpu" {
        "GPU activity target"
    } else {
        "CPU worker allocation"
    };
    let mut note = format!(
        "Recorded settings: {target} {load}%. Reduced intensity can lower scores. Compare runs with matching settings."
    );
    if vram != 0 {
        note.push_str(&format!(" VRAM ceiling: {vram}%."));
        if let Some(bytes) = metadata_size(result, "allocated_test_buffer_bytes") {
            note.push_str(&format!(" Actual test buffers: {bytes}."));
        }
        if let Some(limit) = result.workload_metadata.get("allocation_limit_note") {
            note.push_str(&format!(" {limit}"));
        }
    }
    note
}

fn component_label(result: &BenchmarkResult, devices: &[DeviceDescriptor]) -> String {
    result
        .device_metadata
        .get("saved_component_name")
        .cloned()
        .unwrap_or_else(|| {
            devices
                .iter()
                .find(|d| d.id == result_component_id(&result.device_id))
                .map(|d| d.name.clone())
                .unwrap_or_else(|| result_component_id(&result.device_id).into())
        })
}

// Show RAM with its CPU, preserving the actual measurement device and workload.
// Newest completed run is always first.
fn component_results<'a>(
    results: &'a [BenchmarkResult],
    device_id: &str,
) -> Vec<Vec<&'a BenchmarkResult>> {
    let mut groups: Vec<Vec<&BenchmarkResult>> = Vec::new();
    for result in results
        .iter()
        .rev()
        .filter(|r| result_component_key(r) == device_id)
    {
        if let Some(group) = groups.iter_mut().find(|g| {
            g[0].benchmark_id == result.benchmark_id && g[0].device_id == result.device_id
        }) {
            group.push(result);
        } else {
            groups.push(vec![result]);
        }
    }
    groups.sort_by(|a, b| {
        (result_section(a[0]), &a[0].benchmark_id).cmp(&(result_section(b[0]), &b[0].benchmark_id))
    });
    groups
}

fn result_section(result: &BenchmarkResult) -> i32 {
    if ["read", "write", "copy"].iter().all(|name| {
        result
            .metrics
            .iter()
            .any(|m| m.name == *name && m.unit == "bytes/s")
    }) {
        0 // Read/write/copy columns.
    } else if result.benchmark_id.contains(".bandwidth.") {
        1 // Cache estimates and transfers with their own featured metrics.
    } else {
        2 // Compute throughput.
    }
}

fn best_primary<'a>(group: &[&'a BenchmarkResult]) -> Option<&'a Metric> {
    let latest = featured_metric(group.first()?)?;
    // Capacity and inferred transition sizes are context, not performance scores.
    if !latest.unit.ends_with("/s") {
        return None;
    }
    group
        .iter()
        .filter(|r| result_load_signature(r) == result_load_signature(group[0]))
        .filter_map(|r| featured_metric(r))
        .filter(|m| m.name == latest.name && m.unit == latest.unit && m.value.is_finite())
        .max_by(|a, b| a.value.total_cmp(&b.value))
}

fn bandwidth_score(group: &[&BenchmarkResult], operation: &str, best: bool) -> String {
    let matching = |metric: &&Metric| {
        metric.name == operation && metric.unit == "bytes/s" && metric.value.is_finite()
    };
    let metric = if best {
        group
            .iter()
            .filter(|r| result_load_signature(r) == result_load_signature(group[0]))
            .flat_map(|result| result.metrics.iter())
            .filter(matching)
            .max_by(|a, b| a.value.total_cmp(&b.value))
    } else {
        group
            .first()
            .and_then(|result| result.metrics.iter().find(matching))
    };
    metric
        .map(|m| format!("{:.2} GB/s", m.value / 1e9))
        .unwrap_or_else(|| "—".into())
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
            "Register throughput, working-set scaling, and capability-gated matrix performance."
        }
        _ => "Select a benchmark suite.",
    }
}
fn result_row(
    result: &BenchmarkResult,
    devices: &[DeviceDescriptor],
    benchmarks: &[BenchmarkDescriptor],
) -> ResultRow {
    let primary = featured_metric(result);
    let title = benchmarks
        .iter()
        .find(|benchmark| benchmark.id == result.benchmark_id)
        .map(|benchmark| benchmark.name.as_str())
        .unwrap_or(result.benchmark_id.as_str());
    let diagnosis = if result.benchmark_id.ends_with(".scaling") {
        match (
            result
                .workload_metadata
                .get("bandwidth_transition_status")
                .map(String::as_str),
            result
                .workload_metadata
                .get("bandwidth_transition_working_set_bytes")
                .and_then(|value| value.parse::<f64>().ok()),
            result
                .workload_metadata
                .get("bandwidth_transition_retained_ratio")
                .and_then(|value| value.parse::<f64>().ok()),
        ) {
            (Some("observed"), Some(bytes), Some(ratio)) => format!(
                "Bandwidth influence begins near {}; compute retained {:.1}% of the small-working-set baseline.",
                format_metric(bytes, "bytes"),
                ratio * 100.0
            ),
            _ => "No sustained bandwidth transition was observed within the tested working-set range."
                .into(),
        }
    } else {
        result
            .workload_metadata
            .get("bound_classification")
            .map(|s| format!("Bound diagnosis: {}", s.replace('_', " ")))
            .unwrap_or_else(|| "Sample statistics recorded.".into())
    };
    let metadata_details = result
        .workload_metadata
        .iter()
        .chain(result.device_metadata.iter())
        .take(5)
        .map(|(key, value)| format!("{key}: {value}"))
        .collect::<Vec<_>>()
        .join(" | ");
    let metric_details = result
        .metrics
        .iter()
        .map(|metric| {
            format!(
                "{}: {} (min {}, max {}, {} samples)",
                display_metric_name(&metric.name),
                format_metric(metric.value, &metric.unit),
                format_metric(metric.statistics.minimum, &metric.unit),
                format_metric(metric.statistics.maximum, &metric.unit),
                metric.statistics.sample_count
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let details = format!("{metric_details}\n{metadata_details}");
    let device = result
        .device_metadata
        .get("saved_device_name")
        .map(String::as_str)
        .unwrap_or_else(|| {
            devices
                .iter()
                .find(|device| device.id == result.device_id)
                .map(|device| device.name.as_str())
                .unwrap_or(result.device_id.as_str())
        });
    let facts = result_facts(result, primary);
    ResultRow {
        tuning_guidance: result.workload_metadata.get("tuning_guidance").map(String::as_str).unwrap_or("").into(),
        title: title.into(),
        description: test_description(&result.benchmark_id).into(),
        everyday_use: test_usage(&result.benchmark_id).into(),
        score_guide: format!("{} {} {}",
            if result.metrics.iter().any(|m| m.name == "read") {
                "Read retrieves data; write stores new data; copy moves data between locations. Cache and GPU copy scores count both reading and writing; RAM copy scores count the amount moved, so compare the same test across runs."
            } else { "" }, match primary.map(|m| m.unit.as_str()) {
            Some("operations/s") => "The score counts calculations per second. TOPS means trillions of operations per second; higher is faster within this test.",
            Some("bytes/s" | "MB/s") => "The score shows how much data is processed per second; higher is faster.",
            Some("strings/s" | "primes/s") => "The score counts items processed per second; higher is faster.",
            _ => "Compare scores from the same test to see changes in performance.",
        }, result_settings_note(result)).into(),
        device: device.into(),
        elapsed: format!("{:.2} ms", result.elapsed_ns as f64 / 1e6).into(),
        primary_name: if result.benchmark_id.ends_with(".scaling") { "Largest tested data set compute".into() } else { primary
            .map(|metric| display_metric_name(&metric.name))
            .unwrap_or_else(|| "No metric reported".into())
            .into() },
        primary_value: primary
            .map(|metric| format_metric(metric.value, &metric.unit))
            .unwrap_or_else(|| "N/A".into())
            .into(),
        sample_count: primary
            .map(|metric| format!("{} samples", metric.statistics.sample_count))
            .unwrap_or_else(|| "No samples".into())
            .into(),
        fact_one_label: facts[0].0.as_str().into(),
        fact_one_value: facts[0].1.as_str().into(),
        fact_two_label: facts[1].0.as_str().into(),
        fact_two_value: facts[1].1.as_str().into(),
        fact_three_label: facts[2].0.as_str().into(),
        fact_three_value: facts[2].1.as_str().into(),
        fact_four_label: facts[3].0.as_str().into(),
        fact_four_value: facts[3].1.as_str().into(),
        summary: result_summary(result).into(),
        diagnosis: diagnosis.into(),
        details: if details.is_empty() {
            "No additional workload metadata reported."
        } else {
            &details
        }
        .into(),
    }
}

fn test_description(id: &str) -> &'static str {
    match id {
        "cpu.performance.integer.i64" => {
            "Tests how quickly your processor multiplies and adds whole numbers using all its available cores."
        }
        "cpu.performance.single_thread.integer.i64" => {
            "Runs whole-number calculations on one processor thread, showing performance for tasks that cannot use every core at once."
        }
        "cpu.performance.float.f32" => {
            "Measures how quickly your processor calculates with decimal numbers, using standard precision."
        }
        "cpu.performance.float.f64" => {
            "Measures decimal-number calculations with extra precision, as used in scientific and technical work."
        }
        "cpu.performance.avx2.f32_fma" | "cpu.performance.avx2.f64_fma" => {
            "Tests processor instructions that calculate several decimal numbers at once, helping speed up math-heavy tasks."
        }
        "cpu.performance.string.ascii" => {
            "Measures how quickly your processor scans short pieces of text, a common part of searching and processing data."
        }
        "cpu.performance.prime.sieve" => {
            "Finds prime numbers in a large range to test your processor's ability to work through a numerical problem."
        }
        "cpu.performance.compression.deflate" => {
            "Tests how quickly your processor compresses data into a smaller size, like creating a ZIP archive."
        }
        "cpu.performance.decompression.deflate" => {
            "Tests how quickly your processor restores compressed data, like extracting files from a ZIP archive."
        }
        "cpu.performance.extended.popcnt" => {
            "Counts the set bits inside numbers using a dedicated processor instruction, useful when searching and analyzing data."
        }
        "cpu.performance.extended.aes" => {
            "Tests a processor instruction used in AES encryption, which helps protect data. This measures part of encryption, rather than a complete file transfer."
        }
        "cpu.bandwidth.memory" => {
            "RAM is your computer's main working space for open apps and their data. It holds much more than the CPU caches, but takes longer to access. Results depend on both your RAM and the processor's memory controller."
        }
        "cpu.bandwidth.cache.l1" => {
            "L1 is the smallest and fastest data cache, closest to each processor core. It keeps the data that core needs right away. This test uses small amounts of data across the cores to measure how quickly they can access it."
        }
        "cpu.bandwidth.cache.l2" => {
            "L2 is a larger cache that backs up L1. It holds more data nearby, usually with a little more delay. It helps the processor keep working when the needed data no longer fits in L1."
        }
        "cpu.bandwidth.cache.l3" => {
            "L3 is a larger cache shared by groups of processor cores. It helps those cores reuse data before reaching out to slower system memory (RAM). It usually holds more than L1 or L2, but takes longer to access. These scores measure data-transfer speed, rather than access delay."
        }
        "gpu.bandwidth.cache" => {
            "The graphics card keeps frequently reused data in a small, fast cache near its computing units. This test grows the amount of data and looks for speed changes that suggest cache boundaries. Its effective cache layers are estimates from measured behavior."
        }
        "gpu.bandwidth.vram" => {
            "Measures read, write, and copy speeds in the graphics card's memory, often called VRAM. VRAM holds textures, image buffers, and AI model data. Bandwidth tells you how quickly that data moves; memory capacity tells you how much fits."
        }
        "gpu.bandwidth.host_link" => {
            "Measures data transfers between system RAM and the graphics card, usually across PCI Express. Upload sends data to the GPU; download brings results back. Both directions are recorded, with the featured score showing the first transfer metric."
        }
        "gpu.performance.fp16" => {
            "FP16 uses 16-bit decimal numbers, which trade some precision for smaller values. This test measures many independent calculations on the GPU's general shader units. Matrix tests exercise a separate calculation path."
        }
        "gpu.performance.fp32" => {
            "FP32 uses 32-bit decimal numbers, a common format for graphics calculations. This test measures how quickly the GPU's shader units multiply and add many numbers in parallel."
        }
        "gpu.performance.fp64" => {
            "FP64 uses 64-bit decimal numbers for greater precision. This test measures parallel calculations that need more numerical accuracy. Hardware support and speed vary widely between graphics cards."
        }
        "gpu.performance.fp32.scaling" => {
            "First measures FP32 calculation speed with most work kept in GPU registers, creating a measured compute ceiling rather than a theoretical peak. It then repeats calculations with increasingly large data sets up to your VRAM budget, showing the performance gap as memory traffic grows."
        }
        "gpu.performance.matrix.fp16.scaling" => {
            "First measures FP16 matrix calculation speed with operands kept in GPU registers. It then multiplies grids of 16-bit numbers while increasing input data from cache-sized sets toward your VRAM budget. Each size reports TOPS, effective input traffic in GB/s, and a delta from the measured compute ceiling. This is a measured reference, not a theoretical peak."
        }
        _ if id.starts_with("gpu.performance.matrix.sparse.") => {
            "Tests matrix calculations designed to skip selected zero values. This requires a supported sparse calculation path and specially structured input data; its speed applies to workloads with the matching structure."
        }
        "gpu.performance.matrix.fp16" => {
            "Multiplies grids of 16-bit decimal numbers using the GPU's supported matrix instructions. These calculations are a building block of neural networks. Dense means the full grids are processed, without skipping selected zero entries."
        }
        "gpu.performance.matrix.int8" => {
            "Multiplies grids of compact 8-bit whole numbers. AI models can be converted to use smaller numbers, a process called quantization. This test measures the supported INT8 matrix calculation path."
        }
        "gpu.performance.matrix.fp8" => {
            "Multiplies grids of 8-bit decimal numbers. This compact format can reduce data size and accelerate suitable neural-network calculations, with less numerical precision than FP16 or FP32."
        }
        _ if id.starts_with("cpu.bandwidth.cache.") => {
            "Measures how quickly your processor reads, writes, and copies data in its small, fast built-in cache."
        }
        _ if id.starts_with("gpu.performance.") && id.ends_with(".scaling") => {
            "Repeats graphics-card calculations with increasingly large amounts of data to show when memory access starts to slow them down."
        }
        _ if id.starts_with("gpu.performance.matrix.") => {
            "Tests how quickly the graphics card multiplies grids of numbers, a building block of AI and other math-heavy tasks."
        }
        _ if id.starts_with("gpu.performance.") => {
            "Tests how quickly the graphics card performs many decimal-number calculations in parallel. This is a calculation test, not a game frame-rate estimate."
        }
        _ => "Measures how quickly this component completes the selected workload.",
    }
}

fn test_usage(id: &str) -> &'static str {
    match id {
        "cpu.performance.integer.i64" => {
            "Whole-number calculations support data processing and many program tasks. This all-core test is useful for workloads that can split their work across processor cores; app speed also depends on memory and software."
        }
        "cpu.performance.single_thread.integer.i64" => {
            "Useful context for app responsiveness and parts of game logic that run on one thread. Games and apps combine many kinds of work, so their overall speed also depends on graphics, memory, and other processor tasks."
        }
        "cpu.performance.float.f32" => {
            "Standard-precision math is used in simulations, image processing, and some game physics. This score shows one part of the processor's calculation ability; the app's choice of instructions and use of multiple cores also matter."
        }
        "cpu.performance.float.f64" => {
            "Useful for scientific calculations, engineering, and simulations that need extra precision. Higher precision improves numerical detail, while usually requiring more computation and memory."
        }
        "cpu.performance.avx2.f32_fma" | "cpu.performance.avx2.f64_fma" => {
            "Software that uses these vector instructions can accelerate image processing, simulations, and numerical calculations. The gain depends on whether the application is written to use this instruction path."
        }
        "cpu.performance.string.ascii" => {
            "Text scanning is part of searching documents, reading logs, and processing text data. This test scans fixed-size ASCII records; real text searches also depend on text format, search rules, and storage speed."
        }
        "cpu.performance.prime.sieve" => {
            "A repeatable example of a numerical problem that combines calculation and memory access. It helps compare this algorithm across runs; each application has its own mix of work."
        }
        "cpu.performance.compression.deflate" => {
            "Relevant to creating ZIP archives and compressing data for storage or transfer. The type of input, compression settings, and disk speed affect the time needed for a real archive."
        }
        "cpu.performance.decompression.deflate" => {
            "Relevant to extracting ZIP archives and restoring compressed assets. Installers and apps may use other compression formats, and storage can affect how fast the restored data is saved."
        }
        "cpu.performance.extended.popcnt" => {
            "Counting bits can help databases filter records, search indexes match data, and programs compare compact sets of flags. The result measures a specialized instruction used within those larger tasks."
        }
        "cpu.performance.extended.aes" => {
            "AES is used to protect files and network traffic. Faster AES instructions can help encryption-heavy work; a complete encryption task also includes other operations, memory access, and often storage or network traffic."
        }
        "cpu.bandwidth.memory" => {
            "RAM bandwidth can matter when apps work through large data sets or several cores need data at once. App performance also depends on how long each access takes and whether enough RAM is available."
        }
        _ if id.starts_with("cpu.bandwidth.cache.") => {
            "Fast cache access helps repeated calculations reuse nearby data in games and everyday apps. A program benefits most when its active data fits in the relevant cache. Your score aggregates work across the tested cores."
        }
        "gpu.bandwidth.cache" => {
            "Data reuse can help graphics shaders, image processing, and GPU calculations avoid repeated trips to VRAM. The benefit depends on how the application organizes and reuses its data."
        }
        "gpu.bandwidth.vram" => {
            "VRAM bandwidth can matter for high-resolution graphics, large images, and AI workloads that repeatedly read model data. The amount of VRAM available also determines which textures or models can fit on the card."
        }
        "gpu.bandwidth.host_link" => {
            "Transfer speed can affect uploading graphics resources, moving AI model data onto the GPU, or bringing computed results back. Once data stays on the GPU, its own memory and computing units handle most of the ongoing work."
        }
        "gpu.performance.fp32" => {
            "FP32 shader math is used in gaming graphics, lighting, visual effects, and rendering. A stronger score can help when shader calculations limit performance. Frame rate also depends on memory, the CPU, and other graphics hardware."
        }
        "gpu.performance.fp16" => {
            "Lower-precision shader math can help suitable graphics effects and image-processing tasks do more work with smaller numbers. The application needs to use FP16 where its precision is sufficient."
        }
        "gpu.performance.fp64" => {
            "Useful for scientific computing and simulations needing extra numerical precision. Most gaming graphics use FP32 or lower precision, so FP64 matters most for specialist applications."
        }
        "gpu.performance.matrix.fp16.scaling" => {
            "Useful for understanding neural-network calculations, including processing an AI prompt and working with larger model weights. Small sets benefit from cache; larger sets depend more on VRAM. This measures one building block of AI performance; model design, batch size, and software also affect how quickly an AI app responds."
        }
        "gpu.performance.fp32.scaling" => {
            "Useful context for graphics or compute tasks whose data grows beyond fast cache. A drop in this profile shows how this workload responds to more memory traffic; other applications can reuse data differently."
        }
        _ if id.starts_with("gpu.performance.matrix.sparse.") => {
            "Relevant to neural networks prepared to use the supported sparsity pattern. Ordinary dense models need a compatible transformation and software path to benefit from these instructions."
        }
        "gpu.performance.matrix.fp16" => {
            "Relevant to neural networks, image-generation models, and the prompt-processing stage of language models (prefill), where many input tokens are processed together. Actual speed also depends on the model, batch size, software, and memory traffic."
        }
        "gpu.performance.matrix.int8" => {
            "Relevant to running quantized neural networks, including some image-recognition and language models. The model and software must support this INT8 path; model accuracy and memory needs also depend on how it was quantized."
        }
        "gpu.performance.matrix.fp8" => {
            "Relevant to neural networks prepared for FP8 training or inference. The model, numerical scaling, and software support determine whether this compact format provides useful speed with acceptable accuracy."
        }
        _ if id.starts_with("gpu.performance.matrix.") => {
            "Matrix calculations underpin neural networks and many numerical workloads. Application performance depends on using the matching data format and supported hardware path."
        }
        _ => {
            "Use this result to compare repeated runs of the same workload. Application performance depends on its mix of calculations, memory access, and software."
        }
    }
}

fn result_facts(result: &BenchmarkResult, primary: Option<&Metric>) -> [(String, String); 4] {
    if result.benchmark_id.ends_with(".scaling") {
        return scaling_facts(result);
    }
    let consistency = primary.map_or_else(
        || "Not available".into(),
        |metric| {
            if metric.statistics.sample_count < 2 || metric.statistics.median == 0.0 {
                return "Single sample".into();
            }
            let variation =
                metric.statistics.standard_deviation.abs() / metric.statistics.median.abs() * 100.0;
            let rating = if variation < 1.0 {
                "Excellent"
            } else if variation < 3.0 {
                "Good"
            } else if variation < 7.0 {
                "Fair"
            } else {
                "Variable"
            };
            format!("{rating} / {variation:.1}%")
        },
    );
    let (workload_label, workload_value) = matrix_shape(result)
        .map(|shape| ("MATRIX TILE".into(), shape))
        .or_else(|| {
            metadata_size(result, "working_set_bytes").map(|size| ("WORKING SET".into(), size))
        })
        .unwrap_or_else(|| {
            (
                "DATA TYPE".into(),
                result
                    .workload_metadata
                    .get("data_type")
                    .cloned()
                    .unwrap_or_else(|| "Workload-specific".into())
                    .to_uppercase(),
            )
        });
    let (analysis_label, analysis_value) =
        if let Some(value) = result.workload_metadata.get("bound_classification") {
            (
                "LIMITING FACTOR".into(),
                match value.as_str() {
                    "compute_bound" => "Compute".into(),
                    "memory_bandwidth_bound" => "Memory bandwidth".into(),
                    _ => value.replace('_', " "),
                },
            )
        } else if let Some(status) = result.workload_metadata.get("cache_discovery_status") {
            ("CACHE PROFILE".into(), status.replace('_', " "))
        } else {
            let count = result.metrics.len();
            (
                "MEASUREMENTS".into(),
                format!("{count} {}", if count == 1 { "output" } else { "outputs" }),
            )
        };
    let execution = result
        .workload_metadata
        .get("api")
        .or_else(|| result.workload_metadata.get("execution_backend"))
        .map(|value| match value.as_str() {
            "VK_KHR_cooperative_matrix" => "Vulkan matrix".into(),
            "raw-vulkan" => "Raw Vulkan".into(),
            _ => value.replace('_', " "),
        })
        .unwrap_or_else(|| "Native".into());
    [
        ("CONSISTENCY".into(), consistency),
        (workload_label, workload_value),
        (analysis_label, analysis_value),
        ("EXECUTION".into(), execution),
    ]
}

fn scaling_facts(result: &BenchmarkResult) -> [(String, String); 4] {
    let tiers = scaling_compute_tiers(result);
    let transition_size = result
        .workload_metadata
        .get("bandwidth_transition_working_set_bytes")
        .and_then(|value| value.parse::<f64>().ok());
    let largest = tiers.last().copied();
    let reference = result
        .metrics
        .iter()
        .find(|m| m.name == "measured_compute_ceiling");
    let peak_value = reference.map(|m| m.value).unwrap_or(0.0);
    let largest_retained = largest.and_then(|(_, metric)| {
        (peak_value > 0.0).then_some((metric.value / peak_value * 100.0).clamp(0.0, 999.0))
    });
    [
        (
            if result.benchmark_id == "gpu.performance.fp32.scaling" {
                "FP32 VECTOR REFERENCE".into()
            } else {
                "FP16 MATRIX REFERENCE".into()
            },
            reference
                .map(|metric| format_metric(metric.value, &metric.unit))
                .unwrap_or_else(|| "Not recorded; rerun".into()),
        ),
        (
            "LARGEST TESTED".into(),
            largest
                .map(|(bytes, _)| format_binary_size(bytes as f64))
                .unwrap_or_else(|| "N/A".into()),
        ),
        (
            "DEGRADATION BEGINS".into(),
            transition_size
                .map(format_binary_size)
                .unwrap_or_else(|| "Not observed".into()),
        ),
        (
            largest_retained
                .map(|retained| {
                    format!(
                        "AT LARGEST / {retained:.1}% {}",
                        if reference.is_some() {
                            "REFERENCE"
                        } else {
                            "PEAK"
                        }
                    )
                })
                .unwrap_or_else(|| "AT LARGEST".into()),
            largest
                .map(|(_, metric)| format_metric(metric.value, &metric.unit))
                .unwrap_or_else(|| "N/A".into()),
        ),
    ]
}

fn scaling_compute_tiers(result: &BenchmarkResult) -> Vec<(u64, &Metric)> {
    let mut tiers = result
        .metrics
        .iter()
        .filter_map(|metric| {
            metric
                .name
                .strip_prefix("working_set_")
                .and_then(|rest| rest.strip_suffix(".compute"))
                .and_then(|bytes| bytes.parse::<u64>().ok())
                .map(|bytes| (bytes, metric))
        })
        .collect::<Vec<_>>();
    tiers.sort_by_key(|(bytes, _)| *bytes);
    tiers
}

fn featured_metric(result: &BenchmarkResult) -> Option<&Metric> {
    if result.benchmark_id.ends_with(".scaling") {
        scaling_compute_tiers(result)
            .last()
            .map(|(_, metric)| *metric)
    } else {
        result.metrics.first()
    }
}

fn matrix_shape(result: &BenchmarkResult) -> Option<String> {
    Some(format!(
        "{}x{}x{}",
        result.workload_metadata.get("matrix_m")?,
        result.workload_metadata.get("matrix_n")?,
        result.workload_metadata.get("matrix_k")?
    ))
}

fn metadata_size(result: &BenchmarkResult, key: &str) -> Option<String> {
    result
        .workload_metadata
        .get(key)?
        .parse::<f64>()
        .ok()
        .map(format_binary_size)
}

fn result_summary(result: &BenchmarkResult) -> String {
    if result.benchmark_id.ends_with(".scaling") {
        let compute_tiers = scaling_compute_tiers(result);
        if let (Some(first), Some(last)) = (compute_tiers.first(), compute_tiers.last()) {
            let tier_label = if compute_tiers.len() == 1 {
                "tier"
            } else {
                "tiers"
            };
            let reference = result.metrics.iter().find(|m| {
                m.name == "measured_compute_ceiling" && m.value.is_finite() && m.value > 0.0
            });
            let reference_label = if result.benchmark_id == "gpu.performance.fp32.scaling" {
                "FP32 vector"
            } else {
                "FP16 matrix"
            };
            let comparison = reference
                .map(|metric| {
                    let delta = (last.1.value / metric.value - 1.0) * 100.0;
                    format!(
                        "{reference_label} reference: {} | {:.1}% {} than reference",
                        format_metric(metric.value, &metric.unit),
                        delta.abs(),
                        if delta <= 0.0 { "lower" } else { "higher" }
                    )
                })
                .unwrap_or_else(|| {
                    format!(
                        "{reference_label} reference was not recorded; rerun to measure the loss"
                    )
                });
            return format!(
                "Largest tested: {} at {} | {comparison} | {} {tier_label} from {}",
                format_binary_size(last.0 as f64),
                format_metric(last.1.value, &last.1.unit),
                compute_tiers.len(),
                format_binary_size(first.0 as f64)
            );
        }
    }
    let secondary = result
        .metrics
        .iter()
        .skip(1)
        .take(3)
        .map(|metric| {
            format!(
                "{} {}",
                display_metric_name(&metric.name),
                format_metric(metric.value, &metric.unit)
            )
        })
        .collect::<Vec<_>>();
    if secondary.is_empty() {
        "Compare this score with earlier runs of the same test. More information is available under Details."
            .into()
    } else {
        let remaining = result.metrics.len().saturating_sub(1 + secondary.len());
        format!(
            "Also measured: {}{}",
            secondary.join(" | "),
            if remaining > 0 {
                format!(" | +{remaining} more")
            } else {
                String::new()
            }
        )
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
        "bytes" => format_binary_size(value),
        "bytes/s" => format!("{:.2} GB/s", value / 1e9),
        "operations/s" if value.abs() >= 1e9 => {
            let precision = if value.abs() >= 1e12 {
                2
            } else if value.abs() >= 1e11 {
                3
            } else if value.abs() >= 1e10 {
                4
            } else {
                5
            };
            format!("{:.*} TOPS", precision, value / 1e12)
        }
        "operations/s" => readable_rate(value, "OPS"),
        "MB/s" if value.abs() >= 1_000.0 => format!("{:.2} GB/s", value / 1_000.0),
        "MB/s" => format!("{value:.2} MB/s"),
        "strings/s" => readable_rate(value, "strings/s"),
        "primes/s" => readable_rate(value, "primes/s"),
        _ => format!("{value:.2} {unit}"),
    }
}

fn format_binary_size(bytes: f64) -> String {
    if bytes >= 1024.0 * 1024.0 * 1024.0 {
        format!("{:.2} GiB", bytes / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024.0 * 1024.0 {
        format!("{:.2} MiB", bytes / (1024.0 * 1024.0))
    } else if bytes >= 1024.0 {
        format!("{:.2} KiB", bytes / 1024.0)
    } else {
        format!("{bytes:.0} B")
    }
}

fn display_metric_name(name: &str) -> String {
    if name == "measured_compute_ceiling" {
        return "Measured compute ceiling".into();
    }
    if name == "cache_resident_compute" {
        return "Small-working-set compute baseline".into();
    }
    if name == "bandwidth_transition_working_set" {
        return "Detected bandwidth transition".into();
    }
    if let Some(rest) = name.strip_prefix("working_set_")
        && let Some((bytes, kind)) = rest.split_once('.')
        && let Ok(bytes) = bytes.parse::<f64>()
    {
        let label = match kind {
            "compute" => "compute",
            "reference_delta" => "delta vs compute reference",
            _ => "effective traffic",
        };
        return format!("{} {label}", format_binary_size(bytes));
    }
    name.replace('_', " ")
}
fn readable_rate(value: f64, suffix: &str) -> String {
    let (scale, prefix) = if value >= 1e12 {
        (1e12, "T")
    } else if value >= 1e9 {
        (1e9, "G")
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
    fn saved_test_result(value: f64) -> BenchmarkResult {
        BenchmarkResult {
            benchmark_id: "cpu.performance.float.f32".into(),
            device_id: "cpu:system".into(),
            elapsed_ns: 1,
            metrics: vec![Metric {
                name: "throughput".into(),
                value,
                unit: "operations/s".into(),
                statistics: SampleStatistics::default(),
            }],
            workload_metadata: BTreeMap::new(),
            device_metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn saved_results_round_trip_replace_and_report_bad_files() {
        let directory = std::env::temp_dir().join(format!(
            "gluj-results-test-{}-{}",
            std::process::id(),
            super::SystemTime::now()
                .duration_since(super::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("results.json");
        assert!(super::load_results(&path).unwrap().is_empty());
        let result = saved_test_result(100.0);
        super::save_results(&path, std::slice::from_ref(&result)).unwrap();
        assert_eq!(super::load_results(&path).unwrap(), [result]);
        let replacement = saved_test_result(120.0);
        super::save_results(&path, std::slice::from_ref(&replacement)).unwrap();
        assert_eq!(super::load_results(&path).unwrap(), [replacement]);
        std::fs::write(&path, b"invalid json").unwrap();
        assert!(super::load_results(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"invalid json");
        std::fs::write(&path, br#"{"version":99,"results":[]}"#).unwrap();
        assert!(super::load_results(&path).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn hardware_snapshots_keep_cpu_changes_separate_and_ignore_free_ram() {
        let cpu = DeviceDescriptor {
            id: "cpu:system".into(),
            name: "Old CPU".into(),
            category: DeviceCategory::Cpu,
            available: true,
            status: "Ready".into(),
            properties: BTreeMap::new(),
            caches: vec![],
        };
        let memory = DeviceDescriptor {
            id: "memory:system".into(),
            name: "System memory".into(),
            category: DeviceCategory::Memory,
            available: true,
            status: "Ready".into(),
            properties: BTreeMap::from([
                ("total_bytes".into(), "16000".into()),
                ("available_bytes_at_discovery".into(), "8000".into()),
            ]),
            caches: vec![],
        };
        let mut old = saved_test_result(100.0);
        super::stamp_result(
            &mut old,
            &[cpu.clone(), memory.clone()],
            Some("same-platform"),
        );
        let mut devices = [cpu, memory];
        devices[1]
            .properties
            .insert("available_bytes_at_discovery".into(), "4000".into());
        let mut same = saved_test_result(110.0);
        super::stamp_result(&mut same, &devices, Some("same-platform"));
        assert_eq!(
            super::result_component_key(&old),
            super::result_component_key(&same)
        );
        devices[0].name = "New CPU".into();
        let mut new = saved_test_result(120.0);
        super::stamp_result(&mut new, &devices, Some("same-platform"));
        assert_ne!(
            super::result_component_key(&old),
            super::result_component_key(&new)
        );
        assert_eq!(
            super::result_row(&old, &devices, &[]).device.as_str(),
            "Old CPU"
        );
        let mut archive = vec![];
        super::remember_result(&mut archive, &old);
        super::remember_result(&mut archive, &same);
        super::remember_result(&mut archive, &new);
        assert_eq!(archive.len(), 2);
        assert_eq!(archive[0].metrics[0].value, 110.0);
        assert_eq!(
            super::component_results(&archive, &super::result_component_key(&old)).len(),
            1
        );
    }

    #[test]
    fn saved_comparison_selects_baseline_and_rejects_incompatible_metrics() {
        let mut old = saved_test_result(100.0);
        old.device_metadata
            .insert("saved_setup".into(), "old".into());
        let mut later = saved_test_result(110.0);
        later
            .device_metadata
            .insert("saved_setup".into(), "later".into());
        let current = saved_test_result(120.0);
        let key = super::result_component_key(&old);
        let mut archive = vec![old, later];
        assert_eq!(
            super::primary_delta(&current, &archive, Some(&key)),
            "+20.0%"
        );
        assert_eq!(super::primary_delta(&current, &archive, None), "+9.1%");
        archive[0].metrics[0].unit = "bytes/s".into();
        assert_eq!(super::primary_delta(&current, &archive, Some(&key)), "—");
        archive[0].metrics[0].unit = "operations/s".into();
        archive[0].metrics[0].value = 0.0;
        assert_eq!(super::primary_delta(&current, &archive, Some(&key)), "—");
        archive[0].metrics[0].value = 100.0;
        archive[0]
            .workload_metadata
            .insert("thread_mode".into(), "logical_processors".into());
        assert_eq!(super::primary_delta(&current, &archive, Some(&key)), "—");
    }
    #[test]
    fn settings_round_trip_and_reduced_results_do_not_use_full_load_baselines() {
        let directory = std::env::temp_dir().join(format!(
            "gluj-settings-test-{}-{}",
            std::process::id(),
            super::SystemTime::now()
                .duration_since(super::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("settings.json");
        assert_eq!(
            super::load_settings(&path).unwrap(),
            super::AppSettings::default()
        );
        let settings = super::AppSettings {
            cpu_intensity: 0,
            gpu_intensity: 2,
            vram_budget_percent: 80,
        };
        super::write_json(&path, &settings).unwrap();
        assert_eq!(super::load_settings(&path).unwrap(), settings);
        std::fs::write(&path, br#"{"cpu_intensity":3,"vram_budget_percent":90}"#).unwrap();
        assert!(super::load_settings(&path).is_err());
        std::fs::remove_dir_all(directory).unwrap();

        let full = saved_test_result(100.0); // Legacy records use full intensity.
        let mut reduced = saved_test_result(60.0);
        reduced
            .workload_metadata
            .insert("cpu_worker_percent".into(), "50".into());
        assert_eq!(
            super::primary_delta(&reduced, std::slice::from_ref(&full), None),
            "—"
        );
        assert_eq!(super::best_primary(&[&reduced, &full]).unwrap().value, 60.0);
        let mut archive = vec![full];
        super::remember_result(&mut archive, &reduced);
        assert_eq!(archive.len(), 2);

        let mut gpu_old = saved_test_result(100.0);
        gpu_old.benchmark_id = "gpu.performance.fp32.scaling".into();
        gpu_old.device_id = "gpu:one".into();
        gpu_old
            .workload_metadata
            .insert("vram_budget_percent".into(), "20".into());
        let mut gpu_new = gpu_old.clone();
        gpu_new
            .workload_metadata
            .insert("vram_budget_percent".into(), "80".into());
        assert_eq!(super::primary_delta(&gpu_new, &[gpu_old], None), "—");
    }
    #[test]
    fn component_summary_keeps_latest_and_best_separate() {
        let make = |device: &str, workload: &str, value: f64, unit: &str| BenchmarkResult {
            benchmark_id: workload.into(),
            device_id: device.into(),
            elapsed_ns: 1,
            metrics: vec![Metric {
                name: "Throughput".into(),
                value,
                unit: unit.into(),
                statistics: SampleStatistics::default(),
            }],
            workload_metadata: BTreeMap::new(),
            device_metadata: BTreeMap::new(),
        };
        let results = vec![
            make("gpu:one", "fp32", 100.0, "operations/s"),
            make("gpu:two", "fp32", 900.0, "operations/s"),
            make("gpu:one", "fp32", 500.0, "bytes/s"),
            make("gpu:one", "fp32", 80.0, "operations/s"),
            make("gpu:one", "capacity", 1024.0, "bytes"),
        ];
        let groups = super::component_results(&results, "gpu:one");
        assert_eq!(groups.len(), 2);
        assert!(super::best_primary(&groups[0]).is_none());
        assert_eq!(groups[1].len(), 3);
        assert_eq!(groups[1][0].metrics[0].value, 80.0);
        assert_eq!(super::best_primary(&groups[1]).unwrap().value, 100.0);
        assert!(super::component_results(&results, "missing").is_empty());
    }
    #[test]
    fn system_memory_results_share_the_cpu_component() {
        let make = |device: &str, benchmark: &str| BenchmarkResult {
            benchmark_id: benchmark.into(),
            device_id: device.into(),
            elapsed_ns: 1,
            metrics: vec![],
            workload_metadata: BTreeMap::new(),
            device_metadata: BTreeMap::new(),
        };
        let memory = make("memory:system", "cpu.bandwidth.memory");
        // A RAM-only session still creates the CPU component.
        assert_eq!(super::result_component_id(&memory.device_id), "cpu:system");
        assert_eq!(
            super::component_results(std::slice::from_ref(&memory), "cpu:system").len(),
            1
        );
        let results = vec![
            make("cpu:system", "cpu.performance.float.f32"),
            memory,
            make("gpu:one", "gpu.bandwidth.vram"),
            make("memory:system", "cpu.bandwidth.memory"),
        ];
        let groups = super::component_results(&results, "cpu:system");
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[0][0].device_id, "memory:system");
        assert_eq!(super::component_results(&results, "gpu:one").len(), 1);
        assert!(super::component_results(&results, "memory:system").is_empty());
    }
    #[test]
    fn gpu_sections_stay_together_without_repeated_compute_headers() {
        let make = |id: &str, operations: bool| BenchmarkResult {
            benchmark_id: id.into(),
            device_id: "gpu:one".into(),
            elapsed_ns: 1,
            metrics: if operations {
                ["read", "write", "copy"]
                    .into_iter()
                    .map(|name| Metric {
                        name: name.into(),
                        value: 1e9,
                        unit: "bytes/s".into(),
                        statistics: SampleStatistics::default(),
                    })
                    .collect()
            } else {
                vec![]
            },
            workload_metadata: BTreeMap::new(),
            device_metadata: BTreeMap::new(),
        };
        let results = [
            make("gpu.performance.fp32", false),
            make("gpu.bandwidth.cache", false),
            make("gpu.bandwidth.vram", true),
            make("gpu.bandwidth.host_link", false),
        ];
        let groups = super::component_results(&results, "gpu:one");
        let sections: Vec<_> = groups.iter().map(|g| super::result_section(g[0])).collect();
        assert_eq!(sections, [0, 1, 1, 2]);
        assert_eq!(groups[0][0].benchmark_id, "gpu.bandwidth.vram");
        assert_eq!(groups[3][0].benchmark_id, "gpu.performance.fp32");
    }
    #[test]
    fn bandwidth_columns_use_each_operations_latest_and_best_score() {
        let make = |values: [f64; 3]| BenchmarkResult {
            benchmark_id: "cpu.bandwidth.memory".into(),
            device_id: "memory:system".into(),
            elapsed_ns: 1,
            metrics: ["read", "write", "copy"]
                .into_iter()
                .zip(values)
                .map(|(name, value)| Metric {
                    name: name.into(),
                    value: value * 1e9,
                    unit: "bytes/s".into(),
                    statistics: SampleStatistics::default(),
                })
                .collect(),
            workload_metadata: BTreeMap::new(),
            device_metadata: BTreeMap::new(),
        };
        let old = make([90.0, 30.0, 50.0]);
        let mut latest = make([80.0, 40.0, 45.0]);
        latest.metrics.reverse();
        let group = [&latest, &old];
        assert_eq!(super::bandwidth_score(&group, "read", false), "80.00 GB/s");
        assert_eq!(super::bandwidth_score(&group, "read", true), "90.00 GB/s");
        assert_eq!(super::bandwidth_score(&group, "write", true), "40.00 GB/s");
        assert_eq!(super::bandwidth_score(&group, "copy", true), "50.00 GB/s");
        assert_eq!(super::bandwidth_score(&group, "missing", false), "—");
    }
    #[test]
    fn operation_rates_use_readable_si_prefixes() {
        assert_eq!(format_metric(12_500.0, "operations/s"), "12.50 KOPS");
        assert_eq!(format_metric(133_375.55e6, "operations/s"), "0.133 TOPS");
        assert_eq!(format_metric(8_452.27e6, "operations/s"), "0.00845 TOPS");
        assert_eq!(format_metric(1_807.42, "MB/s"), "1.81 GB/s");
        assert_eq!(format_metric(1_117.99e6, "strings/s"), "1.12 Gstrings/s");
        assert_eq!(
            format_metric(3_200_000_000_000.0, "operations/s"),
            "3.20 TOPS"
        );
        assert_eq!(format_metric(8.0 * 1024.0 * 1024.0, "bytes"), "8.00 MiB");
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
        let row = result_row(&result, &[device], &[]);
        assert_eq!(row.device.as_str(), "Integrated GPU");
        assert_eq!(row.sample_count.as_str(), "5 samples");
        assert_eq!(row.fact_one_label.as_str(), "CONSISTENCY");
        assert_eq!(row.fact_one_value.as_str(), "Excellent / 0.0%");
        assert!(
            row.details
                .as_str()
                .contains("cache_discovery_status: not_detected")
        );
    }

    #[test]
    fn fp16_matrix_scaling_has_its_own_summary_and_largest_tier_result() {
        let mut matrix = saved_test_result(80e12);
        matrix.benchmark_id = "gpu.performance.matrix.fp16.scaling".into();
        matrix.device_id = "gpu:one".into();
        matrix.metrics[0].name = "cache_resident_compute".into();
        for (bytes, compute) in [(262144, 80e12), (4294967296_u64, 40e12)] {
            matrix.metrics.push(Metric {
                name: format!("working_set_{bytes}.compute"),
                value: compute,
                unit: "operations/s".into(),
                statistics: SampleStatistics::default(),
            });
            matrix.metrics.push(Metric {
                name: format!("working_set_{bytes}.bandwidth"),
                value: 900e9,
                unit: "bytes/s".into(),
                statistics: SampleStatistics::default(),
            });
        }
        matrix
            .workload_metadata
            .insert("vram_budget_percent".into(), "20".into());
        let mut vector = matrix.clone();
        vector.benchmark_id = "gpu.performance.fp32.scaling".into();
        let results = [vector, matrix.clone()];
        let groups = super::component_results(&results, "gpu:one");
        assert_eq!(groups.len(), 2);
        let row = result_row(&matrix, &[], &[]);
        assert_eq!(row.primary_value.as_str(), "40.00 TOPS");
        assert_eq!(row.fact_two_value.as_str(), "4.00 GiB");
        assert_eq!(row.fact_four_value.as_str(), "40.00 TOPS");
        assert!(row.summary.as_str().contains("reference was not recorded"));
        assert!(row.description.as_str().contains("GB/s"));
        assert!(row.everyday_use.as_str().contains("AI prompt"));
        assert!(row.score_guide.as_str().contains("20%"));
        let mut small_only = matrix.clone();
        small_only.metrics = vec![Metric {
            name: "working_set_8388608.compute".into(),
            value: 100e12,
            unit: "operations/s".into(),
            statistics: SampleStatistics::default(),
        }];
        assert_eq!(
            super::best_primary(&[&matrix, &small_only]).unwrap().value,
            40e12
        );
        assert_eq!(super::primary_delta(&matrix, &[small_only], None), "—");
        let mut reference = matrix.metrics[0].clone();
        reference.name = "measured_compute_ceiling".into();
        reference.value = 100e12;
        matrix.metrics.insert(0, reference);
        matrix
            .workload_metadata
            .insert("largest_compute_delta_percent".into(), "-60.00".into());
        matrix.workload_metadata.insert(
            "tuning_guidance".into(),
            "Try a small 5% core-clock reduction and rerun.".into(),
        );
        let row = result_row(&matrix, &[], &[]);
        assert_eq!(row.fact_one_label.as_str(), "FP16 MATRIX REFERENCE");
        assert_eq!(row.fact_one_value.as_str(), "100.00 TOPS");
        assert_eq!(row.fact_four_label.as_str(), "AT LARGEST / 40.0% REFERENCE");
        assert!(row.summary.as_str().contains("60.0% lower than reference"));
        assert!(row.tuning_guidance.as_str().contains("5%"));
    }

    #[test]
    fn compute_profile_result_explains_the_detected_transition() {
        let stats = SampleStatistics {
            sample_count: 3,
            minimum: 49e12,
            median: 50e12,
            maximum: 51e12,
            standard_deviation: 1e12,
        };
        let result = BenchmarkResult {
            benchmark_id: "gpu.performance.fp32.scaling".into(),
            device_id: "gpu:one".into(),
            elapsed_ns: 1,
            metrics: vec![
                Metric {
                    name: "cache_resident_compute".into(),
                    value: 50e12,
                    unit: "operations/s".into(),
                    statistics: stats.clone(),
                },
                Metric {
                    name: "working_set_4194304.compute".into(),
                    value: 52e12,
                    unit: "operations/s".into(),
                    statistics: stats.clone(),
                },
                Metric {
                    name: "working_set_8388608.compute".into(),
                    value: 43e12,
                    unit: "operations/s".into(),
                    statistics: stats.clone(),
                },
                Metric {
                    name: "working_set_536870912.compute".into(),
                    value: 39e12,
                    unit: "operations/s".into(),
                    statistics: stats,
                },
            ],
            workload_metadata: BTreeMap::from([
                ("bandwidth_transition_status".into(), "observed".into()),
                (
                    "bandwidth_transition_working_set_bytes".into(),
                    "8388608".into(),
                ),
                ("bandwidth_transition_retained_ratio".into(), "0.86".into()),
            ]),
            device_metadata: BTreeMap::new(),
        };
        let row = result_row(&result, &[], &[]);
        assert_eq!(row.primary_name.as_str(), "Largest tested data set compute");
        assert!(row.diagnosis.as_str().contains("8.00 MiB"));
        assert!(row.diagnosis.as_str().contains("86.0%"));
        assert!(row.summary.as_str().contains("3 tiers"));
        assert!(row.summary.as_str().contains("512.00 MiB"));
        assert_eq!(row.primary_value.as_str(), "39.00 TOPS");
        assert!(row.summary.as_str().contains("reference was not recorded"));
        assert_eq!(row.fact_one_label.as_str(), "FP32 VECTOR REFERENCE");
        assert_eq!(row.fact_one_value.as_str(), "Not recorded; rerun");
        assert_eq!(row.fact_two_label.as_str(), "LARGEST TESTED");
        assert_eq!(row.fact_two_value.as_str(), "512.00 MiB");
        assert_eq!(row.fact_three_value.as_str(), "8.00 MiB");
        assert_eq!(row.fact_four_label.as_str(), "AT LARGEST");
        assert_eq!(row.fact_four_value.as_str(), "39.00 TOPS");
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
