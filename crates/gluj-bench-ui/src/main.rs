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
    collections::{BTreeMap, VecDeque},
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

const HARDWARE_METADATA_CACHE_VERSION: u32 = 17;
const BENCHMARK_TARGET_DURATION_MS: u64 = 2_000;
const BENCHMARK_SAMPLES: u32 = 5;

mod chart_hover;
#[cfg(test)]
mod configuration_tests;
mod gpu_timings;
mod render_pacing;
mod result_visuals;
mod tuning_visuals;

fn main() -> Result<(), slint::PlatformError> {
    // The app-wide redraw filter requires the desktop winit backend.
    // Slint still honors renderer selection (for example winit-software).
    slint::BackendSelector::new()
        .backend_name("winit".into())
        .select()?;
    let window = MainWindow::new()?;
    let app = Rc::new(RefCell::new(App::new()));
    let pacing_app = Rc::downgrade(&app);
    render_pacing::install(&window, move || {
        pacing_app
            .upgrade()
            .is_some_and(|app| app.borrow().active_request.is_some())
    });
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
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // Keep the CLI-capable worker hidden when started by the desktop app.
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
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
    result_selection: usize,
    queue: VecDeque<QueuedRun>,
    run_overrides: BTreeMap<String, TestOverride>,
    dataset_text: String,
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
    ram_budget_percent: u32,
    cpu_core_limit: u32,
    dataset_mode: usize,
    dataset_mib: u64,
    ram_offload_percent: u32,
}
impl Default for AppSettings {
    fn default() -> Self {
        Self {
            cpu_intensity: 1,
            gpu_intensity: 1,
            vram_budget_percent: 25,
            ram_budget_percent: 20,
            cpu_core_limit: 0,
            dataset_mode: 0,
            dataset_mib: 256,
            ram_offload_percent: 75,
        }
    }
}
#[derive(Clone)]
struct TestOverride {
    settings: AppSettings,
    dataset_text: String,
}

#[derive(Clone)]
struct QueuedRun {
    benchmark_id: String,
    settings: AppSettings,
    gpu_id: Option<String>,
    physical_cores: u32,
}

const DATASET_NAMES: [&str; 3] = [
    "Automatic (allocation budget)",
    "Sweep up to chosen size",
    "Test one chosen size",
];
const DATASET_MODES: [&str; 3] = ["automatic", "sweep", "single"];
const OFFLOAD_PERCENT: [u32; 3] = [50, 75, 100];

fn validated_run_settings(
    settings: &AppSettings,
    text: &str,
    id: &str,
) -> Result<AppSettings, String> {
    let mut settings = settings.clone();
    if id.ends_with(".scaling") && settings.dataset_mode != 0 {
        settings.dataset_mib = text
            .trim()
            .parse::<u64>()
            .ok()
            .filter(|n| (1..=1_048_576).contains(n))
            .ok_or_else(|| {
                "Dataset size must be a whole number from 1 to 1,048,576 MiB.".to_owned()
            })?;
    }
    Ok(settings)
}

fn run_options(run: &QueuedRun) -> serde_json::Map<String, Value> {
    let settings = &run.settings;
    let id = &run.benchmark_id;
    let mut options = serde_json::Map::new();
    options.insert(
        "vram_budget_percent".into(),
        settings.vram_budget_percent.to_string().into(),
    );
    options.insert(
        "cpu_worker_percent".into(),
        INTENSITY_PERCENT[settings.cpu_intensity].to_string().into(),
    );
    options.insert(
        "gpu_activity_percent".into(),
        INTENSITY_PERCENT[settings.gpu_intensity].to_string().into(),
    );
    if id.starts_with("cpu.") || id.starts_with("memory.") {
        let cores = if settings.cpu_core_limit == 0 {
            (run.physical_cores * INTENSITY_PERCENT[settings.cpu_intensity] / 100).max(1)
        } else {
            settings.cpu_core_limit.min(run.physical_cores).max(1)
        };
        options.insert("cpu_core_limit".into(), cores.to_string().into());
        options.insert("thread_mode".into(), "physical_cores".into());
        if settings.cpu_core_limit > 0 {
            options.insert("cpu_worker_percent".into(), "100".into());
        }
    }
    if cpu_scaling(id) || gpu_offload(id) {
        options.insert(
            "ram_budget_percent".into(),
            settings.ram_budget_percent.to_string().into(),
        );
    }
    if gpu_offload(id) {
        options.insert(
            "ram_offload_percent".into(),
            settings.ram_offload_percent.to_string().into(),
        );
    }
    if id.ends_with(".scaling") {
        options.insert(
            "dataset_mode".into(),
            DATASET_MODES[settings.dataset_mode].into(),
        );
        if settings.dataset_mode != 0 {
            options.insert(
                "dataset_bytes".into(),
                (settings.dataset_mib * 1024 * 1024).to_string().into(),
            );
        }
    }
    if id.starts_with("gpu.")
        && let Some(device) = &run.gpu_id
    {
        options.insert("device_id".into(), device.clone().into());
    }
    options
}

fn configuration_summary(settings: &AppSettings, id: &str) -> String {
    let mut parts = Vec::new();
    if id.starts_with("gpu.") {
        parts.push(format!(
            "GPU activity {}%",
            INTENSITY_PERCENT[settings.gpu_intensity]
        ));
        if id.ends_with(".scaling") {
            parts.push(format!("VRAM budget {}%", settings.vram_budget_percent));
        }
    } else {
        parts.push(if settings.cpu_core_limit == 0 {
            format!(
                "CPU workers {}% (automatic cores)",
                INTENSITY_PERCENT[settings.cpu_intensity]
            )
        } else {
            format!("{} physical cores", settings.cpu_core_limit)
        });
    }
    if cpu_scaling(id) || gpu_offload(id) {
        parts.push(format!("RAM budget {}%", settings.ram_budget_percent));
    }
    if gpu_offload(id) {
        parts.push(format!("{}% RAM offload", settings.ram_offload_percent));
    }
    if id.ends_with(".scaling") {
        parts.push(match settings.dataset_mode {
            1 => format!("Sweep up to {} MiB", settings.dataset_mib),
            2 => format!("One dataset: {} MiB", settings.dataset_mib),
            _ => "Automatic dataset sweep".into(),
        });
    }
    parts.join(" · ")
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
        || !(20..=80).contains(&settings.ram_budget_percent)
        || settings.dataset_mode > 2
        || !(1..=1_048_576).contains(&settings.dataset_mib)
        || !OFFLOAD_PERCENT.contains(&settings.ram_offload_percent)
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
        app.dataset_text = app.settings.dataset_mib.to_string();
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
        window.on_chart_point_at(chart_hover::nearest);
        let weak = Rc::downgrade(app);
        window.on_select_default_configuration(move |field, index| {
            if let Some(app) = weak.upgrade() {
                app.borrow_mut()
                    .edit_configuration(false, field.as_str(), index);
            }
        });
        let weak = Rc::downgrade(app);
        window.on_edit_default_dataset(move |text| {
            if let Some(app) = weak.upgrade() {
                app.borrow_mut().edit_dataset(false, text.to_string());
            }
        });
        let weak = Rc::downgrade(app);
        window.on_toggle_test_override(move |enabled| {
            if let Some(app) = weak.upgrade() {
                let mut app = app.borrow_mut();
                if app.active_request.is_some() {
                    return;
                }
                if let Some(id) = app.selected.clone() {
                    if enabled {
                        let configuration = TestOverride {
                            settings: app.settings.clone(),
                            dataset_text: app.dataset_text.clone(),
                        };
                        app.run_overrides.entry(id).or_insert(configuration);
                    } else {
                        app.run_overrides.remove(&id);
                    }
                    app.ui_dirty = true;
                }
            }
        });
        let weak = Rc::downgrade(app);
        window.on_select_test_configuration(move |field, index| {
            if let Some(app) = weak.upgrade() {
                app.borrow_mut()
                    .edit_configuration(true, field.as_str(), index);
            }
        });
        let weak = Rc::downgrade(app);
        window.on_edit_test_dataset(move |text| {
            if let Some(app) = weak.upgrade() {
                app.borrow_mut().edit_dataset(true, text.to_string());
            }
        });
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
        window.on_select_ram_budget(move |index| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                if app.active_request.is_some() || !(0..8).contains(&index) {
                    return;
                }
                app.settings.ram_budget_percent = VRAM_BUDGET_OPTIONS[index as usize];
                app.persist_settings();
            }
        });
        let app_weak = Rc::downgrade(app);
        window.on_select_cpu_cores(move |index| {
            if let Some(app) = app_weak.upgrade() {
                let mut app = app.borrow_mut();
                if app.active_request.is_some()
                    || index < 0
                    || index as u32 > app.physical_core_count()
                {
                    return;
                }
                app.settings.cpu_core_limit = index as u32;
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
                app.result_selection = index.max(0) as usize;
                app.result_details_expanded = false;
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
    fn effective_settings(&self, id: &str) -> Result<AppSettings, String> {
        let (settings, text) = self
            .run_overrides
            .get(id)
            .map(|config| (&config.settings, config.dataset_text.as_str()))
            .unwrap_or((&self.settings, self.dataset_text.as_str()));
        validated_run_settings(settings, text, id)
    }
    fn edit_configuration(&mut self, local: bool, field: &str, index: i32) {
        if self.active_request.is_some() || index < 0 {
            return;
        }
        let cores = self.physical_core_count();
        let settings = if local {
            let Some(config) = self
                .selected
                .as_ref()
                .and_then(|id| self.run_overrides.get_mut(id))
            else {
                return;
            };
            &mut config.settings
        } else {
            &mut self.settings
        };
        match field {
            "cpu" if index < 3 => settings.cpu_intensity = index as usize,
            "gpu" if index < 3 => settings.gpu_intensity = index as usize,
            "cores" if index as u32 <= cores => settings.cpu_core_limit = index as u32,
            "ram" if index < 8 => settings.ram_budget_percent = VRAM_BUDGET_OPTIONS[index as usize],
            "vram" if index < 8 => {
                settings.vram_budget_percent = VRAM_BUDGET_OPTIONS[index as usize]
            }
            "dataset" if index < 3 => settings.dataset_mode = index as usize,
            "offload" if index < 3 => {
                settings.ram_offload_percent = OFFLOAD_PERCENT[index as usize]
            }
            _ => return,
        }
        if local {
            self.ui_dirty = true;
        } else {
            self.persist_settings();
        }
    }
    fn edit_dataset(&mut self, local: bool, text: String) {
        if self.active_request.is_some() {
            return;
        }
        let valid = text
            .trim()
            .parse::<u64>()
            .ok()
            .filter(|n| (1..=1_048_576).contains(n));
        if local {
            if let Some(config) = self
                .selected
                .as_ref()
                .and_then(|id| self.run_overrides.get_mut(id))
            {
                config.dataset_text = text;
                if let Some(mib) = valid {
                    config.settings.dataset_mib = mib;
                }
            }
        } else {
            self.dataset_text = text;
            if let Some(mib) = valid {
                self.settings.dataset_mib = mib;
                self.persist_settings();
            }
        }
        self.ui_dirty = true;
    }
    fn queued_run(&self, id: &str) -> Result<QueuedRun, String> {
        Ok(QueuedRun {
            benchmark_id: id.into(),
            settings: self.effective_settings(id)?,
            gpu_id: self.selected_gpu_id.clone(),
            physical_cores: self.physical_core_count(),
        })
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
    fn physical_core_count(&self) -> u32 {
        self.devices
            .iter()
            .find_map(|d| {
                d.properties
                    .get("physical_cores")
                    .and_then(|s| s.parse::<u32>().ok())
            })
            .unwrap_or(1)
            .max(1)
    }
    fn start_next(&mut self) {
        if self.active_request.is_some() {
            return;
        }
        let Some(run) = self.queue.pop_front() else {
            return;
        };
        let benchmark_id = &run.benchmark_id;
        let id = self.next_id("run");
        let options = run_options(&run);
        match self.worker.send(json!({"protocol": PROTOCOL_VERSION, "id": id, "command": "run", "arguments": {"benchmark_id": benchmark_id, "target_duration_ms": BENCHMARK_TARGET_DURATION_MS, "samples": BENCHMARK_SAMPLES, "options": options}})) { Ok(()) => { self.active_request = Some(id); self.progress = 0.; self.progress_message = format!("Starting {benchmark_id}"); self.status = "Benchmark running…".into(); }, Err(problem) => { self.status = problem; self.queue.clear(); } }
    }
    fn run_selected(&mut self) {
        if self.active_request.is_some() {
            return;
        }
        if let Some(id) = self.selected.clone() {
            self.queue.clear();
            match self.queued_run(&id) {
                Ok(run) => self.queue.push_back(run),
                Err(error) => {
                    self.status = error;
                    self.ui_dirty = true;
                    return;
                }
            }
            self.start_next();
        }
    }
    fn run_all(&mut self) {
        if self.active_request.is_some() {
            return;
        }
        let mut items: Vec<_> = self
            .benchmarks
            .iter()
            .filter(|b| b.suite_id == self.selected_suite && self.benchmark_available(b))
            .cloned()
            .collect();
        items.sort_by_key(|b| b.display_order);
        let runs = items
            .iter()
            .map(|b| self.queued_run(&b.id))
            .collect::<Result<VecDeque<_>, _>>();
        match runs {
            Ok(runs) => self.queue = runs,
            Err(error) => {
                self.status = error;
                self.ui_dirty = true;
                return;
            }
        }
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
        window.set_app_version(env!("CARGO_PKG_VERSION").into());
        window.set_system_ready(
            app.worker.online()
                && !app.metadata_scan_pending
                && app.metadata_devices_received
                && app.metadata_benchmarks_received
                && app.benchmarks.iter().any(|b| b.available),
        );
        window
            .set_vram_budget_name(format!("{}% of VRAM", app.settings.vram_budget_percent).into());
        window.set_ram_budget_name(format!("{}% of RAM", app.settings.ram_budget_percent).into());
        let mut core_options = vec![SharedString::from("Automatic · use allocation preset")];
        core_options.extend((1..=app.physical_core_count()).map(|n| {
            SharedString::from(format!(
                "{n} physical core{}",
                if n == 1 { "" } else { "s" }
            ))
        }));
        window.set_cpu_core_name(
            core_options[app.settings.cpu_core_limit.min(app.physical_core_count()) as usize]
                .clone(),
        );
        window.set_cpu_core_options(ModelRc::new(VecModel::from(core_options.clone())));
        window.set_default_dataset_mode(DATASET_NAMES[app.settings.dataset_mode].into());
        window.set_default_dataset_text(app.dataset_text.clone().into());
        window.set_default_dataset_custom(app.settings.dataset_mode != 0);
        window.set_default_offload_name(
            format!("{}% system RAM", app.settings.ram_offload_percent).into(),
        );
        window.set_default_dataset_error(
            validated_run_settings(&app.settings, &app.dataset_text, "global.scaling")
                .err()
                .unwrap_or_default()
                .into(),
        );
        let selected_id = app.selected.as_deref().unwrap_or("");
        let local = app.run_overrides.get(selected_id);
        let settings = local.map(|c| &c.settings).unwrap_or(&app.settings);
        window.set_test_override_enabled(local.is_some());
        window
            .set_test_is_cpu(selected_id.starts_with("cpu.") || selected_id.starts_with("memory."));
        window.set_test_is_gpu(selected_id.starts_with("gpu."));
        window.set_test_is_scaling(selected_id.ends_with(".scaling"));
        window.set_test_is_offload(gpu_offload(selected_id));
        window.set_test_has_ram(cpu_scaling(selected_id) || gpu_offload(selected_id));
        window.set_test_cpu_name(INTENSITY_NAMES[settings.cpu_intensity].into());
        window.set_test_gpu_name(INTENSITY_NAMES[settings.gpu_intensity].into());
        window.set_test_core_name(if settings.cpu_core_limit == 0 {
            core_options[0].clone()
        } else {
            core_options[settings.cpu_core_limit.min(app.physical_core_count()) as usize].clone()
        });
        window.set_test_ram_name(format!("{}% of RAM", settings.ram_budget_percent).into());
        window.set_test_vram_name(format!("{}% of VRAM", settings.vram_budget_percent).into());
        window
            .set_test_offload_name(format!("{}% system RAM", settings.ram_offload_percent).into());
        window.set_test_dataset_mode(DATASET_NAMES[settings.dataset_mode].into());
        window.set_test_dataset_custom(settings.dataset_mode != 0);
        window.set_test_dataset_text(
            local
                .map(|c| c.dataset_text.clone())
                .unwrap_or_else(|| app.dataset_text.clone())
                .into(),
        );
        let effective = app.effective_settings(selected_id);
        window.set_test_configuration_valid(effective.is_ok());
        window.set_test_configuration_summary(
            match effective {
                Ok(settings) => format!(
                    "{} · {}",
                    if local.is_some() {
                        "Test override"
                    } else {
                        "Global defaults"
                    },
                    configuration_summary(&settings, selected_id)
                ),
                Err(error) => error,
            }
            .into(),
        );
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
            .map(|d| {
                let (description, facts, badges) = overview_device(d);
                DeviceRow {
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
                        .map(|(k, v)| format!("{}: {v}", k.replace('_', " ")))
                        .collect::<Vec<_>>()
                        .join("\n")
                        .into(),
                    description: description.into(),
                    facts: ModelRc::new(VecModel::from(facts)),
                    badges: ModelRc::new(VecModel::from(
                        badges
                            .into_iter()
                            .map(SharedString::from)
                            .collect::<Vec<_>>(),
                    )),
                    cache_details: if d.category == DeviceCategory::Cpu {
                        cache_details(d).replace(" | ", "\n")
                    } else {
                        String::new()
                    }
                    .into(),
                    accent: device_accent(d.category),
                    available: d.available,
                }
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
                .unwrap_or("No benchmark selected")
                .into(),
        );
        window.set_selected_benchmark_description(
            selected
                .map(|benchmark| test_takeaway(&benchmark.id))
                .unwrap_or("Select a benchmark to view its details and run it.")
                .into(),
        );
        window.set_selected_benchmark_details(
            selected.map(test_lab_details).unwrap_or_default().into(),
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
                    "No compatible GPU adapter is available for this category."
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
        let mut comparison_names: Vec<SharedString> =
            vec!["Last saved measurement for each test".into()];
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
        let selection = app.result_selection.min(groups.len().saturating_sub(1));
        let rows = groups
            .iter()
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
        window.set_result_selection(selection as i32);
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
    let mut file: ResultsFile = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if file.version != 1 {
        return Err("unsupported results file version".into());
    }
    for result in &mut file.results {
        // Retire the old pseudo-utilization fields while preserving actual measurements.
        result.metrics.retain(|m| {
            !m.name.ends_with(".gpu_busy_estimate") && !m.name.ends_with(".gpu_wait_estimate")
        });
        result
            .workload_metadata
            .retain(|key, _| !key.starts_with("gpu_busy_wait_"));

        if let Some(percent) = match result.benchmark_id.as_str() {
            "gpu.performance.fp32.offload50.scaling" => Some(50),
            "gpu.performance.fp32.offload75.scaling" => Some(75),
            "gpu.performance.fp32.offload100.scaling" => Some(100),
            _ => None,
        } {
            result.benchmark_id = "gpu.performance.fp32.offload.scaling".into();
            result
                .workload_metadata
                .entry("ram_offload_percent".into())
                .or_insert_with(|| percent.to_string());
        }
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
            || old.workload_metadata.get("ram_offload_percent")
                != result.workload_metadata.get("ram_offload_percent")
            || old.workload_metadata.get("dataset_mode")
                != result.workload_metadata.get("dataset_mode")
            || old.workload_metadata.get("requested_dataset_bytes")
                != result.workload_metadata.get("requested_dataset_bytes")
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
                && same_run_configuration(old, current)
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
                    "compute_profile_revision",
                    "matrix_tile_reuse",
                    "ram_offload_percent",
                    "offload_profile_revision",
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
        .filter(|m| m.unit.ends_with("/s") || read_latency(current))
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

fn result_load_signature(result: &BenchmarkResult) -> (u32, u32, u32, u32) {
    let vram = if component_family(result) == "gpu" && result.benchmark_id.ends_with(".scaling") {
        result
            .workload_metadata
            .get("vram_budget_percent")
            .and_then(|value| value.parse().ok())
            .unwrap_or(25)
    } else {
        0
    };
    let cpu = component_family(result) != "gpu";
    let cores = if cpu {
        result
            .workload_metadata
            .get("cpu_core_limit")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    } else {
        0
    };
    let ram = if (cpu && cpu_scaling(&result.benchmark_id)) || gpu_offload(&result.benchmark_id) {
        result
            .workload_metadata
            .get("ram_budget_percent")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    } else {
        0
    };
    (result_load_percent(result), vram, cores, ram)
}

fn result_settings_note(result: &BenchmarkResult) -> String {
    if let Some(level) = cache_latency_level(&result.benchmark_id) {
        return format!(
            "One pinned CPU thread; allocation presets do not add workers. L{level} working set: {} / {} cache capacity. Warmed dependent reads; lower ns per read is better and a negative Change means decreased latency. Cache residency is inferred from working-set size, not confirmed by hardware counters.",
            metadata_size(result, "working_set_bytes").unwrap_or_else(|| "not recorded".into()),
            metadata_size(result, "target_cache_bytes").unwrap_or_else(|| "not recorded".into())
        );
    }
    if ram_latency(result) {
        return format!(
            "One pinned CPU thread; CPU core-allocation presets do not change this test. Working set: {}. {} Lower nanoseconds per read is better; a negative Change means latency decreased. Measures CPU-to-memory access, not DRAM CAS timing.",
            metadata_size(result, "working_set_bytes").unwrap_or_else(|| "not recorded".into()),
            if result.benchmark_id == "cpu.latency.memory.localized" {
                "Randomized within 64 KiB blocks to reduce translation overhead; not a PassMark-equivalent score."
            } else {
                "Scattered object-like reads across the full allocation include address-translation overhead."
            }
        );
    }
    let (load, vram, cores, ram) = result_load_signature(result);
    let target = if component_family(result) == "gpu" {
        "GPU activity target"
    } else {
        "CPU worker allocation"
    };
    let mut note = format!(
        "Recorded settings: {target} {load}%. Reduced intensity can lower measured throughput. Compare runs with matching settings."
    );
    if cores > 0 {
        note.push_str(&format!(" Physical-core limit: {cores}; one worker per core (single-thread tests still use one)."));
    }
    if ram > 0 {
        note.push_str(&format!(
            " RAM ceiling: {ram}% of installed RAM, limited by current availability."
        ));
    }
    if vram != 0 {
        note.push_str(&format!(" VRAM ceiling: {vram}%."));
        if let Some(bytes) = metadata_size(result, "allocated_test_buffer_bytes") {
            note.push_str(&format!(" Actual test buffers: {bytes}."));
        }
        if let Some(limit) = result.workload_metadata.get("allocation_limit_note") {
            note.push_str(&format!(" {limit}"));
        }
    }
    if gpu_offload(&result.benchmark_id) {
        if let Some(percent) = result.workload_metadata.get("ram_offload_percent") {
            note.push_str(&format!(" RAM offload: {percent}% of input/output data. Nominal host traffic includes cache-served accesses; it is not measured PCIe bus traffic."));
        }
        for (key, label) in [
            ("host_allocated_test_buffer_bytes", "RAM buffers"),
            ("device_allocated_test_buffer_bytes", "VRAM buffers"),
        ] {
            if let Some(bytes) = metadata_size(result, key) {
                note.push_str(&format!(" {label}: {bytes}."));
            }
        }
    }
    if let Some(mode) = result.workload_metadata.get("dataset_mode") {
        note.push_str(&format!(" Dataset mode: {mode}."));
        if let Some(bytes) = metadata_size(result, "requested_dataset_bytes") {
            note.push_str(&format!(
                " Requested dataset: {bytes}; actual tested sizes are aligned to the kernel shape."
            ));
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
            g[0].benchmark_id == result.benchmark_id
                && g[0].device_id == result.device_id
                && (!gpu_offload(&result.benchmark_id)
                    || g[0].workload_metadata.get("ram_offload_percent")
                        == result.workload_metadata.get("ram_offload_percent"))
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
    if read_latency(result) {
        return 3; // Latency has its own lower-is-better section.
    }
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

fn same_run_configuration(left: &BenchmarkResult, right: &BenchmarkResult) -> bool {
    result_load_signature(left) == result_load_signature(right)
        && [
            "ram_offload_percent",
            "requested_dataset_bytes",
            "arithmetic_iterations",
            "matrix_tile_reuse",
            "compute_profile_revision",
            "latency_profile_revision",
        ]
        .iter()
        .all(|key| left.workload_metadata.get(*key) == right.workload_metadata.get(*key))
        && left
            .workload_metadata
            .get("dataset_mode")
            .map(String::as_str)
            .unwrap_or("automatic")
            == right
                .workload_metadata
                .get("dataset_mode")
                .map(String::as_str)
                .unwrap_or("automatic")
        && (!read_latency(left)
            || [
                "working_set_bytes",
                "node_stride_bytes",
                "random_seed",
                "processor_group",
                "processor_index",
                "page_policy",
                "access_order",
                "locality_block_bytes",
                "target_cache_bytes",
                "cache_processor_group",
                "cache_processor_mask",
                "cache_sharing_logical_processors",
                "preceding_cache_bytes",
                "cache_state",
            ]
            .iter()
            .all(|key| left.workload_metadata.get(*key) == right.workload_metadata.get(*key)))
}

fn ram_latency(result: &BenchmarkResult) -> bool {
    matches!(
        result.benchmark_id.as_str(),
        "cpu.latency.memory" | "cpu.latency.memory.localized"
    )
}

fn cache_latency_level(id: &str) -> Option<u8> {
    match id {
        "cpu.latency.cache.l1" => Some(1),
        "cpu.latency.cache.l2" => Some(2),
        "cpu.latency.cache.l3" => Some(3),
        _ => None,
    }
}

fn read_latency(result: &BenchmarkResult) -> bool {
    ram_latency(result) || cache_latency_level(&result.benchmark_id).is_some()
}

fn best_primary<'a>(group: &[&'a BenchmarkResult]) -> Option<&'a Metric> {
    let latest = featured_metric(group.first()?)?;
    // Capacity and inferred transition sizes are context, not performance scores.
    if !latest.unit.ends_with("/s") && !read_latency(group[0]) {
        return None;
    }
    group
        .iter()
        .filter(|r| same_run_configuration(r, group[0]))
        .filter_map(|r| featured_metric(r))
        .filter(|m| {
            m.name == latest.name
                && m.unit == latest.unit
                && m.value.is_finite()
                && (!read_latency(group[0]) || m.value > 0.0)
        })
        .max_by(|a, b| {
            if read_latency(group[0]) {
                b.value.total_cmp(&a.value)
            } else {
                a.value.total_cmp(&b.value)
            }
        })
}

fn bandwidth_score(group: &[&BenchmarkResult], operation: &str, best: bool) -> String {
    let matching = |metric: &&Metric| {
        metric.name == operation && metric.unit == "bytes/s" && metric.value.is_finite()
    };
    let metric = if best {
        group
            .iter()
            .filter(|r| same_run_configuration(r, group[0]))
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
fn overview_device(device: &DeviceDescriptor) -> (String, Vec<DeviceFact>, Vec<String>) {
    let property = |key: &str| {
        device
            .properties
            .get(key)
            .cloned()
            .unwrap_or_else(|| "Not reported".into())
    };
    let bytes = |key: &str| {
        device
            .properties
            .get(key)
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|n| *n > 0)
            .map(|n| format_binary_size(n as f64))
            .unwrap_or_else(|| "Not reported".into())
    };
    let fact = |label: &str, value: String| DeviceFact {
        label: label.into(),
        value: value.into(),
    };
    let supported = |key: &str| device.properties.get(key).is_some_and(|s| s == "true");
    match device.category {
        DeviceCategory::Cpu => {
            let l3: u64 = device.caches.iter().filter(|c| c.level == 3).map(|c| c.size_bytes.saturating_mul(c.instances as u64)).sum();
            let architecture = match device.properties.get("architecture").map(String::as_str) {
                Some("x86_64") => "64-bit x86".into(), Some("aarch64") => "64-bit ARM".into(),
                Some(s) => s.to_owned(), None => "Architecture not reported".into(),
            };
            let mut badges = vec![architecture];
            if !device.caches.is_empty() { badges.push("Cache topology detected".into()); }
            ("Your system's calculation engine. Explore compute throughput and the path from cache to RAM.".into(),
                vec![fact("PHYSICAL CORES", property("physical_cores")), fact("LOGICAL THREADS", property("logical_processors")),
                    fact("TOTAL L3 CACHE", if l3 > 0 { format_binary_size(l3 as f64) } else { "Not reported".into() })], badges)
        }
        DeviceCategory::Memory => (
            "The processor's shared workspace for larger data sets. Available memory reflects the last hardware scan.".into(),
            vec![fact("INSTALLED MEMORY", bytes("total_bytes")), fact("AVAILABLE AT SCAN", bytes("available_bytes_at_discovery")), fact("CONNECTED TO", "Processor".into())],
            vec!["System RAM".into(), "Read / write / copy".into()]),
        DeviceCategory::Gpu => {
            let kind = match device.properties.get("device_type").map(String::as_str) {
                Some("DiscreteGpu" | "discrete") => "Discrete GPU".into(), Some("IntegratedGpu" | "integrated") => "Integrated GPU".into(),
                Some(s) => s.replace('_', " "), None => "Not reported".into(),
            };
            let mut badges = Vec::new();
            for (key, label) in [("shader_f16", "FP16 vectors"), ("shader_f64", "FP64 vectors"), ("cooperative_matrix_fp16", "FP16 matrix support"), ("cooperative_matrix_int8", "INT8 matrix support")] {
                if supported(key) { badges.push(label.into()); }
            }
            if badges.is_empty() { badges.push("See capability details".into()); }
            ("Parallel compute and memory throughput for graphics and AI workloads. Badges show detected hardware support; individual tests check runner availability.".into(),
                vec![fact(if device.properties.get("device_type").is_some_and(|s| s == "IntegratedGpu") { "SHARED GPU MEMORY" } else { "GRAPHICS MEMORY" }, bytes("device_local_memory_bytes")), fact("DEVICE TYPE", kind), fact("COMPUTE API", property("backend"))], badges)
        }
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
        "cpu.bandwidth" => {
            "Cache/RAM bandwidth and dependent RAM read latency with sample statistics."
        }
        "cpu.performance" => {
            "Aggregate processor throughput, single-thread INT64, and compute-vs-memory diagnosis."
        }
        "gpu.bandwidth" => "Estimated cache, GPU-local memory, and host-device bandwidth.",
        "gpu.performance" => {
            "Register throughput, working-set scaling, and capability-gated matrix performance."
        }
        _ => "Select a benchmark category.",
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
    let title = if gpu_offload(&result.benchmark_id) {
        result
            .workload_metadata
            .get("ram_offload_percent")
            .map(|percent| format!("{title} · {percent}% RAM"))
            .unwrap_or_else(|| title.to_owned())
    } else {
        title.to_owned()
    };
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
        .filter(|m| {
            !m.name.ends_with(".gpu_busy_estimate") && !m.name.ends_with(".gpu_wait_estimate")
        })
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
    let profile = chart_tiers(result);
    let (chart_svg, chart_labels, chart_points) =
        scaling_chart_interactive(result).unwrap_or_default();
    let (gpu_timing_svg, gpu_timing_labels, gpu_timing_points) =
        gpu_timings::interactive_chart(result).unwrap_or_default();
    let tuning = tuning_visuals::view(result);
    ResultRow {
        profile_points: ModelRc::new(VecModel::from(chart_points)),
        gpu_timing_points: ModelRc::new(VecModel::from(gpu_timing_points)),
        measurement_facts: ModelRc::new(VecModel::from(result_visuals::facts(result, primary))),
        measurement_bars: ModelRc::new(VecModel::from(result_visuals::bars(result, primary))),
        context_facts: ModelRc::new(VecModel::from(
            facts
                .iter()
                .map(|(label, value)| DeviceFact {
                    label: label.as_str().into(),
                    value: value.as_str().into(),
                })
                .collect::<Vec<_>>(),
        )),
        tuning_facts_top: ModelRc::new(VecModel::from(tuning.top)),
        tuning_facts_bottom: ModelRc::new(VecModel::from(tuning.bottom)),
        tuning_bars: ModelRc::new(VecModel::from(tuning.bars)),
        tuning_allocation: ModelRc::new(VecModel::from(tuning.allocation)),
        tuning_actions: ModelRc::new(VecModel::from(tuning.actions)),
        tuning_quality: tuning.quality.into(),
        gpu_timing_available: !gpu_timing_svg.is_empty(),
        gpu_timing_visible: result.benchmark_id.starts_with("gpu.")
            && result.benchmark_id.ends_with(".scaling"),
        gpu_timing_chart: slint::Image::load_from_svg_data(gpu_timing_svg.as_bytes())
            .ok()
            .unwrap_or_default(),
        gpu_timing_labels: ModelRc::new(VecModel::from(gpu_timing_labels)),
        gpu_timing_note: gpu_timings::note(result).into(),
        tuning_title: "TUNING SUGGESTIONS".into(),
        profile_chart: slint::Image::load_from_svg_data(chart_svg.as_bytes())
            .ok()
            .unwrap_or_default(),
        profile_labels: ModelRc::new(VecModel::from(chart_labels)),
        profile_sizes: ModelRc::new(VecModel::from(
            profile
                .iter()
                .map(|(bytes, _)| SharedString::from(format_binary_size(*bytes as f64)))
                .collect::<Vec<_>>(),
        )),
        profile_statistics: ModelRc::new(VecModel::from(
            profile
                .iter()
                .map(|(bytes, compute)| {
                    let mut text = format!(
                        "Compute: {} (min {}, max {}; {} samples)",
                        format_metric(compute.value, &compute.unit),
                        format_metric(compute.statistics.minimum, &compute.unit),
                        format_metric(compute.statistics.maximum, &compute.unit),
                        compute.statistics.sample_count
                    );
                    text.push_str(&gpu_timings::details(result, *bytes, compute));
                    if let Some(bandwidth) = chart_bandwidth(result, *bytes) {
                        text.push_str(&format!(
                            "\n{}: {} (min {}, max {}; {} samples)",
                            if gpu_offload(&result.benchmark_id) {
                                "Nominal host traffic (cache included)"
                            } else {
                                "Effective traffic"
                            },
                            format_metric(bandwidth.value, &bandwidth.unit),
                            format_metric(bandwidth.statistics.minimum, &bandwidth.unit),
                            format_metric(bandwidth.statistics.maximum, &bandwidth.unit),
                            bandwidth.statistics.sample_count
                        ));
                    }
                    if gpu_offload(&result.benchmark_id)
                        && let Some(percent) = result
                            .workload_metadata
                            .get("ram_offload_percent")
                            .and_then(|s| s.parse::<u64>().ok())
                            .filter(|p| *p <= 100)
                    {
                        text.push_str(&format!(
                            "\nPlacement: {} RAM + {} VRAM",
                            format_binary_size((*bytes * percent / 100) as f64),
                            format_binary_size((*bytes * (100 - percent) / 100) as f64)
                        ));
                    }
                    if let Some(shape) = cpu_matrix_tier_shape(result, *bytes) {
                        text.push_str(&format!(
                            "\nMatrix dimensions per worker (M × K × N): {shape}"
                        ));
                    }
                    SharedString::from(text)
                })
                .collect::<Vec<_>>(),
        )),
        description_tldr: test_takeaway(&result.benchmark_id).into(),
        tuning_tldr: tuning_takeaway(result).into(),
        tuning_stats: tuning_breakdown(result).into(),
        memory_pressure: memory_pressure_summary(result).into(),
        tuning_guidance: if gpu_offload(&result.benchmark_id) {
            // Apply current advice to older saved offload results as well.
            gluj_bench_core::GPU_OFFLOAD_TUNING_GUIDANCE
        } else {
            result
                .workload_metadata
                .get("tuning_guidance")
                .map(String::as_str)
                .unwrap_or("")
        }
        .into(),
        title: title.into(),
        description: test_description(&result.benchmark_id).into(),
        everyday_use: test_usage(&result.benchmark_id).into(),
        metric_guide: format!(
            "{} {}",
            metric_explanation(
                primary.map(|metric| metric.unit.as_str()).unwrap_or(""),
                result.metrics.iter().any(|metric| metric.name == "read")
            ),
            result_settings_note(result)
        )
        .into(),
        device: device.into(),
        elapsed: format!("{:.2} ms", result.elapsed_ns as f64 / 1e6).into(),
        primary_name: if result.benchmark_id.ends_with(".scaling") {
            "Largest tested data set compute".into()
        } else {
            primary
                .map(|metric| display_metric_name(&metric.name))
                .unwrap_or_else(|| "No metric reported".into())
                .into()
        },
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

fn metric_explanation(unit: &str, read_write_copy: bool) -> String {
    let units = match unit {
        "ns" => {
            "Nanoseconds measure time; lower is better. Read latency is the average time per dependent read, summarized across repeated samples."
        }
        "operations/s" => {
            "TOPS means trillions of calculations per second; higher means more compute throughput within this test."
        }
        "bytes/s" => {
            "GB/s means billions of bytes moved per second; higher means more bandwidth. This measures transfer speed, not the delay before data arrives."
        }
        "MB/s" => {
            "MB/s means millions of bytes processed per second; higher means more throughput."
        }
        "strings/s" | "primes/s" => {
            "The metric counts items processed per second; higher means more throughput."
        }
        _ => "Compare metrics from the same test to see changes in performance.",
    };
    if read_write_copy {
        format!(
            "Read retrieves data; write stores new data; copy moves data between locations. Cache and GPU copy bandwidth count both reading and writing; RAM copy bandwidth counts the amount moved, so compare the same test across runs. {units}"
        )
    } else {
        units.into()
    }
}

fn test_lab_details(benchmark: &BenchmarkDescriptor) -> String {
    let mut details = format!(
        "WHAT THIS TEST MEASURES\n{}\n\nWHERE IT CAN HELP\n{}\n\nHOW TO READ THE METRICS\n{}",
        test_description(&benchmark.id),
        test_usage(&benchmark.id),
        metric_explanation(
            &benchmark.unit,
            benchmark.id == "cpu.bandwidth.memory"
                || benchmark.id.starts_with("cpu.bandwidth.cache.")
                || benchmark.id == "gpu.bandwidth.vram"
        ),
    );
    if benchmark.id == "cpu.performance.matrix.fp32.scaling" {
        details.push_str("\n\nREADING THE MATRIX PROFILE\nEach CPU worker multiplies A[32,K] by B[K,N] to produce C[32,N], with K=N growing from 32 as the dataset increases. Dataset size includes all input and output matrices across selected workers. The blocked AVX2/FMA kernel reuses weight data across 32 rows. The RAM budget setting applies to total buffers and leaves available-memory headroom. Exact core allocation uses one worker per physical core. Changing workers also changes per-worker matrix dimensions; inspect shapes and aggregate dataset sizes when comparing.\n\nREADING THE METRICS\nTOPS counts 2×M×N×K floating-point operations per complete product. GB/s counts scalar input reads, vector weight reads, and output reads/writes within the cache blocks. Reused data can be served by cache, so effective GB/s can exceed physical RAM bandwidth. Arithmetic intensity here uses those kernel accesses, not just each matrix's unique bytes. Compare this matrix profile with its own small-data baseline; its reuse differs from the vector test.");
    } else if benchmark.id == "cpu.performance.avx2.f32_fma.scaling" {
        details.push_str("\n\nREADING THE SCALING PROFILE\nThe test compares increasingly large FP32 arrays with the current run's register-only AVX2/FMA reference. Dataset size is the total of two input arrays and one output array across all selected CPU workers; each worker handles an equal, separate share. The RAM budget setting limits total allocation and leaves available-memory headroom. Exact core allocation uses one worker per physical core. Compare at the same largest tested dataset size when changing cores.\n\nMEMORY PRESSURE\nTOPS counts multiply and add operations; GB/s counts two input reads and one output write. The kernel performs 16 FMAs per value to keep the compute-to-data ratio fixed. Effective traffic excludes write allocation and cache-line writeback, so it is not physical RAM-bus utilization. Sustained slowdown suggests cache or RAM pressure; exact cache boundaries and stall time are not measured.");
    } else if gpu_offload(&benchmark.id) {
        details.push_str("\n\nREADING RAM OFFLOAD\nThe percentage applies to all input and output bytes, with aligned RAM and VRAM regions. The GPU directly accesses system RAM; BAR-mapped VRAM is excluded. Each tier reports total effective traffic plus nominal host traffic in the expanded measurements. Cache reuse can reduce actual bus traffic. This does not force VRAM exhaustion or measure page migration. Compare identical datasets and arithmetic iterations using different RAM offload percentages.");
    } else if benchmark.id.ends_with(".scaling") {
        details.push_str("\n\nREADING THE SCALING PROFILE\nThe test measures a small compute reference and progressively larger data sets up to your VRAM budget, subject to available memory. The table shows throughput at the largest tested data set and its change against the current run's reference. Any core-limit suggestion is a trial: change the limit yourself, rerun with the same settings and data-set size, and aim for no more than 0–5% throughput loss from your original run.");
        details.push_str("\n\nMEMORY PRESSURE\nThe tuning box compares small and large data sets and shows effective test traffic in GB/s. A sustained slowdown suggests memory pressure, but it does not measure the exact percentage of time the GPU waits for memory. Cache reuse can make effective test traffic differ from physical VRAM traffic. Noisy samples require a repeat measurement before estimating a core-limit reduction.");
    }
    if matches!(
        benchmark.id.as_str(),
        "cpu.latency.memory" | "cpu.latency.memory.localized"
    ) || cache_latency_level(&benchmark.id).is_some()
    {
        details.push_str("\n\nBEFORE YOU RUN\nThis test always uses one pinned thread regardless of the CPU allocation preset. Close competing memory-heavy work and compare matching working sets. Large allocations require longer complete traversals; Stop all tests remains available during preparation and sampling.");
    } else {
        details.push_str("\n\nBEFORE YOU RUN\nLower test intensity leaves more room for other work but can lower measured throughput. Compare runs with matching intensity and workload settings. These are measurements of this test, rather than a direct prediction of game frame rates or whole-app performance.");
    }
    if !benchmark.workload.is_empty() {
        details.push_str(&format!("\n\nTEST WORKLOAD\n{}", benchmark.workload));
    }
    details
}

fn test_takeaway(id: &str) -> &'static str {
    match id {
        _ if cache_latency_level(id).is_some() => {
            "Measures warmed dependent reads in a cache-sized working set on one CPU core. Lower nanoseconds per read is better."
        }
        "cpu.latency.memory" => {
            "Measures dependent reads of scattered object-like nodes across RAM, including address translation. Lower is better."
        }
        "cpu.latency.memory.localized" => {
            "Measures dependent RAM reads with 64 KiB locality to reduce translation overhead. Lower nanoseconds per access is better."
        }
        "cpu.performance.matrix.fp32.scaling" => {
            "Shows how CPU FP32 matrix throughput changes as weight matrices grow beyond cache into RAM."
        }
        "cpu.performance.avx2.f32_fma.scaling" => {
            "Shows how CPU vector throughput changes as FP32 data grows beyond cache into RAM."
        }
        _ if gpu_offload(id) => {
            "Measures GPU calculation speed with your chosen share of data in system RAM. Choose the offload percentage and dataset in Settings or override them for this test."
        }
        "gpu.performance.fp32.scaling" => {
            "Shows how larger data sets affect graphics-card calculation throughput."
        }
        "gpu.performance.matrix.fp16.scaling" => {
            "Shows how larger AI data sets affect matrix calculation throughput."
        }
        "gpu.bandwidth.host_link" => {
            "Measures how quickly data moves between system RAM and the graphics card."
        }
        "gpu.bandwidth.vram" => "Measures GPU memory bandwidth for moving large amounts of data.",
        "gpu.bandwidth.cache" => "Measures bandwidth when GPU data fits in its fast cache.",
        "cpu.bandwidth.memory" => {
            "Measures RAM bandwidth used when applications move large amounts of data."
        }
        _ if id.starts_with("cpu.bandwidth.cache.") => {
            "Measures cache bandwidth that helps the processor reuse nearby data quickly."
        }
        _ if id.starts_with("gpu.performance.matrix.") => {
            "Measures matrix calculation throughput used by neural networks and AI applications."
        }
        "gpu.performance.fp32" => {
            "Measures shader calculation throughput used in gaming graphics and rendering."
        }
        "gpu.performance.fp16" => {
            "Measures throughput for compact calculations used in graphics and AI."
        }
        "gpu.performance.fp64" => {
            "Measures extra-precision calculation throughput used in scientific applications."
        }
        _ if id.contains("deflate") => {
            "Measures how quickly the processor packs or unpacks compressed data."
        }
        _ if id.contains("aes") => "Measures processor throughput for encrypting data.",
        _ if id.contains("string") => {
            "Measures how quickly the processor searches and processes text."
        }
        _ if id.contains("single_thread") => {
            "Measures calculation throughput for work that uses one processor thread."
        }
        _ if id.starts_with("cpu.performance.") => {
            "Measures processor calculation throughput for everyday and math-heavy applications."
        }
        _ => "Measures how quickly this component processes the selected workload.",
    }
}

fn gpu_offload(id: &str) -> bool {
    matches!(
        id,
        "gpu.performance.fp32.offload.scaling"
            | "gpu.performance.fp32.offload50.scaling"
            | "gpu.performance.fp32.offload75.scaling"
            | "gpu.performance.fp32.offload100.scaling"
    )
}

fn cpu_scaling(id: &str) -> bool {
    id.starts_with("cpu.performance.") && id.ends_with(".scaling")
}

fn cpu_matrix_tier_shape(result: &BenchmarkResult, bytes: u64) -> Option<String> {
    if result.benchmark_id != "cpu.performance.matrix.fp32.scaling" {
        return None;
    }
    result
        .workload_metadata
        .get("matrix_profile_shapes")?
        .split(',')
        .find_map(|entry| {
            let (size, shape) = entry.split_once(':')?;
            if size.parse::<u64>().ok()? != bytes {
                return None;
            }
            let dimensions = shape
                .split('x')
                .map(|n| n.parse::<u64>())
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            (dimensions.len() == 3 && dimensions.iter().all(|n| *n > 0)).then(|| {
                dimensions
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(" × ")
            })
        })
}

fn tuning_takeaway(result: &BenchmarkResult) -> String {
    if gpu_offload(&result.benchmark_id) {
        return gluj_bench_core::GPU_OFFLOAD_TUNING_TAKEAWAY.into();
    }
    if cpu_scaling(&result.benchmark_id) {
        if let Some(cores) = result
            .workload_metadata
            .get("suggested_cpu_core_count_trial")
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|n| *n > 0)
            && scaling_compute_tiers(result)
                .last()
                .is_some_and(|(_, m)| consistent_metric(m))
        {
            return format!(
                "Try {cores} physical cores, then retest the same dataset for at most 5% throughput loss."
            );
        }
        return if scaling_compute_tiers(result)
            .last()
            .is_some_and(|(_, metric)| !consistent_metric(metric))
        {
            "Repeat the run for consistent CPU scaling measurements.".into()
        } else {
            "Compare TOPS and GB/s across dataset sizes to inspect CPU memory pressure.".into()
        };
    }
    if let Some(reduction) = result
        .workload_metadata
        .get("suggested_core_frequency_limit_reduction_percent")
        .or_else(|| {
            result
                .workload_metadata
                .get("suggested_core_underclock_trial_percent")
        })
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0 && *value <= 40.0)
    {
        return format!(
            "Try a further {reduction:.0}% core-limit reduction, then retest for 0–5% throughput loss."
        );
    }
    if result
        .workload_metadata
        .get("tuning_guidance")
        .is_some_and(|guidance| guidance.starts_with("No "))
    {
        let tiers = scaling_compute_tiers(result);
        if let (Some((_, largest)), Some(reference)) = (
            tiers.last(),
            result
                .metrics
                .iter()
                .find(|m| m.name == "measured_compute_ceiling"),
        ) && reference.value.is_finite()
            && reference.value > 0.0
            && largest.value.is_finite()
            && largest.value > 0.0
            && largest.value / reference.value < 0.85
        {
            let noisy = !consistent_metric(reference)
                || !consistent_metric(largest)
                || tiers
                    .iter()
                    .take(3)
                    .max_by(|a, b| a.1.value.total_cmp(&b.1.value))
                    .is_some_and(|(_, small)| !consistent_metric(small));
            return if noisy {
                "Large throughput drop detected; rerun for consistent measurements before estimating a core-limit reduction.".into()
            } else {
                "Large throughput drop detected; more memory-pressure evidence is needed before estimating a core-limit reduction.".into()
            };
        }
        "No clear core-limit headroom measured; rerun to confirm the result.".into()
    } else {
        "Review the recorded guidance before changing the core-frequency limit.".into()
    }
}

fn consistent_metric(metric: &Metric) -> bool {
    metric.value.is_finite()
        && metric.value > 0.0
        && metric.statistics.sample_count >= 2
        && metric.statistics.standard_deviation.is_finite()
        && metric.statistics.standard_deviation.abs() / metric.value <= 0.10
}

fn memory_pressure_summary(result: &BenchmarkResult) -> String {
    let tiers = scaling_compute_tiers(result);
    let Some((largest_bytes, largest)) = tiers.last().copied() else {
        return String::new();
    };
    let Some((small_bytes, small)) = tiers
        .iter()
        .take(3)
        .filter(|(_, metric)| metric.value.is_finite() && metric.value > 0.0)
        .max_by(|a, b| a.1.value.total_cmp(&b.1.value))
        .copied()
    else {
        return String::new();
    };
    if small_bytes >= largest_bytes || !largest.value.is_finite() || largest.value <= 0.0 {
        return "Memory pressure: not enough comparable data-set measurements.".into();
    }
    let observed = result
        .workload_metadata
        .get("bandwidth_transition_status")
        .is_some_and(|status| status == "observed");
    let gap = (1.0 - largest.value / small.value) * 100.0;
    let status = if observed && gap > 10.0 && consistent_metric(small) && consistent_metric(largest)
    {
        "likely memory bottleneck"
    } else if observed && gap > 10.0 {
        "possible memory bottleneck"
    } else {
        "bottleneck not established"
    };
    let uncertainty = if consistent_metric(small) && consistent_metric(largest) {
        "estimate"
    } else {
        "estimate; noisy or insufficient samples"
    };
    let change = if gap >= 0.0 {
        format!("{gap:.1}% slower")
    } else {
        format!("{:.1}% faster", -gap)
    };
    let traffic = result
        .metrics
        .iter()
        .find(|m| {
            m.name == format!("working_set_{largest_bytes}.bandwidth")
                && m.unit == "bytes/s"
                && m.value.is_finite()
                && m.value > 0.0
        })
        .map(|m| {
            format!(
                " • {} effective test traffic",
                format_metric(m.value, &m.unit)
            )
        })
        .unwrap_or_default();
    format!("Memory pressure: {status} ({uncertainty}); {change} than the small data set{traffic}.")
}

fn tuning_breakdown(result: &BenchmarkResult) -> String {
    let Some((bytes, largest)) = scaling_compute_tiers(result).last().copied() else {
        return String::new();
    };
    let mut lines = vec![
        format!("Data set: {}", format_binary_size(bytes as f64)),
        format!(
            "Compute throughput: {}",
            format_metric(largest.value, &largest.unit)
        ),
    ];
    if cpu_scaling(&result.benchmark_id) {
        if let Some(workers) = result.workload_metadata.get("thread_count") {
            lines.push(format!(
                "Workers in latest run: {workers} ({})",
                result
                    .workload_metadata
                    .get("thread_mode")
                    .map(String::as_str)
                    .unwrap_or("not recorded")
            ));
        }
        if let Some(percent) = result.workload_metadata.get("ram_budget_percent") {
            lines.push(format!("Requested RAM budget: {percent}% of installed RAM"));
        }
        if let Some(budget) = metadata_size(result, "allocation_budget_bytes") {
            lines.push(format!("Available-memory-adjusted ceiling: {}", budget));
        }
        if let Some(trial) = result
            .workload_metadata
            .get("suggested_cpu_core_count_trial")
        {
            lines.push(format!("Exploratory core-count trial: {trial}. Fewer-core throughput, power and temperature benefits have not been measured."));
            lines.push(format!("Retest target at this dataset: at least {} (95% of latest throughput); retain an improvement too.", format_metric(largest.value * 0.95, &largest.unit)));
        }
    }
    if let Some(reference) = result.metrics.iter().find(|metric| {
        metric.name == "measured_compute_ceiling" && metric.value.is_finite() && metric.value > 0.0
    }) {
        lines.push(format!(
            "Compute reference: {}",
            format_metric(reference.value, &reference.unit)
        ));
        lines.push(format!(
            "Change against reference: {:+.1}%",
            (largest.value / reference.value - 1.0) * 100.0
        ));
    }
    if largest.value > 0.0 {
        lines.push(format!(
            "Sample variation: {:.1}% ({} samples)",
            largest.statistics.standard_deviation.abs() / largest.value * 100.0,
            largest.statistics.sample_count
        ));
    }
    if result
        .workload_metadata
        .get("tuning_guidance")
        .is_some_and(|s| s.starts_with("No "))
    {
        for (label, metric) in [
            ("largest data set", Some(largest)),
            (
                "compute reference",
                result
                    .metrics
                    .iter()
                    .find(|m| m.name == "measured_compute_ceiling"),
            ),
        ] {
            if let Some(metric) = metric.filter(|m| !consistent_metric(m)) {
                let reason = if metric.statistics.sample_count < 2 {
                    "fewer than two samples".into()
                } else if metric.value.is_finite()
                    && metric.value > 0.0
                    && metric.statistics.standard_deviation.is_finite()
                {
                    format!(
                        "{:.1}% sample variation exceeds the 10% consistency threshold",
                        metric.statistics.standard_deviation.abs() / metric.value * 100.0
                    )
                } else {
                    "invalid measurement statistics".into()
                };
                lines.push(format!(
                    "Tuning estimate unavailable: {label} has {reason}."
                ));
            }
        }
    }
    if let Some(anchor) = result
        .workload_metadata
        .get("frequency_trial_anchor_operations_per_second")
        .and_then(|value| value.parse::<f64>().ok())
    {
        lines.push(format!(
            "Trial compute baseline: {}",
            format_metric(anchor, "operations/s")
        ));
    }
    if let Some(drift) = result
        .workload_metadata
        .get("compute_reference_drift_percent")
    {
        lines.push(format!("Reference change during test: {drift}%"));
    }
    if let Some(confidence) = result.workload_metadata.get("tuning_confidence") {
        lines.push(format!("Confidence: {}", confidence.replace('_', " ")));
    }
    let tiers = scaling_compute_tiers(result);
    if let Some((small_bytes, small)) = tiers
        .iter()
        .take(3)
        .filter(|(_, m)| m.value.is_finite() && m.value > 0.0)
        .max_by(|a, b| a.1.value.total_cmp(&b.1.value))
        .copied()
    {
        lines.push(format!(
            "Small-data baseline: {} at {}",
            format_metric(small.value, &small.unit),
            format_binary_size(small_bytes as f64)
        ));
    }
    if let Some(bandwidth) = result.metrics.iter().find(|m| {
        m.name == format!("working_set_{bytes}.bandwidth")
            && m.unit == "bytes/s"
            && m.value.is_finite()
            && m.value > 0.0
    }) {
        lines.push(format!(
            "Effective test traffic at largest size: {}",
            format_metric(bandwidth.value, &bandwidth.unit)
        ));
        if largest.value.is_finite() && largest.value > 0.0 {
            lines.push(format!(
                "Work per byte of test traffic: {:.2} operations/byte",
                largest.value / bandwidth.value
            ));
        }
    }
    if let Some(transition) = result
        .workload_metadata
        .get("bandwidth_transition_working_set_bytes")
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
    {
        lines.push(format!(
            "Sustained memory-pressure slowdown first detected at: {}",
            format_binary_size(transition)
        ));
    }
    if let (Some(allocated), Some(budget)) = (
        result
            .workload_metadata
            .get("allocated_test_buffer_bytes")
            .and_then(|s| s.parse::<f64>().ok()),
        result
            .workload_metadata
            .get("allocation_budget_bytes")
            .and_then(|s| s.parse::<f64>().ok()),
    ) && allocated.is_finite()
        && allocated >= 0.0
        && budget.is_finite()
        && budget > 0.0
    {
        lines.push(format!(
            "Test memory allocation: {} / {} selected budget ({:.1}%)",
            format_binary_size(allocated),
            format_binary_size(budget),
            allocated / budget * 100.0
        ));
    }
    if result.benchmark_id.starts_with("cpu.") {
        if let Some(workers) = result.workload_metadata.get("thread_count") {
            lines.push(format!(
                "CPU workers: {workers}; dataset sizes include all three arrays across all workers."
            ));
        }
        if let Some(bytes) = result
            .workload_metadata
            .get("per_thread_working_set_bytes")
            .and_then(|s| s.parse::<f64>().ok())
        {
            lines.push(format!(
                "Largest dataset per worker: {}",
                format_binary_size(bytes)
            ));
        }
        if result.benchmark_id == "cpu.performance.matrix.fp32.scaling" {
            if let Some(shape) = cpu_matrix_tier_shape(result, bytes) {
                lines.push(format!(
                    "Largest matrix dimensions per worker (M × K × N): {shape}"
                ));
            }
            lines.push("Matrix traffic counts input accesses and output reads/writes inside the blocked kernel, including cache reuse. This differs from unique matrix bytes and physical RAM traffic. Shape is 32 rows with K=N growing with each dataset.".into());
        } else {
            lines.push("Effective traffic counts two input reads plus one output write; cache reuse, write allocation and writeback can change physical memory traffic.".into());
        }
        lines.push("Memory pressure is inferred from this workload's size sweep, not measured CPU stall time or RAM-bus utilization. Total system memory congestion is not measured.".into());
    } else {
        lines.push("Memory pressure is inferred from this workload's size sweep. The slowdown percentage is not memory-controller utilization or GPU stall time. Effective traffic counts the kernel's data accesses; cache reuse can make it differ from physical VRAM traffic. Total system memory congestion is not measured.".into());
    }
    lines.join("\n")
}

fn test_description(id: &str) -> &'static str {
    match id {
        _ if cache_latency_level(id).is_some() => {
            "Follows dependent pointers in a working set selected from the cache instance attached to the pinned core. L1 uses 75% of its capacity. L2 and L3 aim for four times the preceding level's capacity, capped at 75% of the target cache; the set must exceed twice the preceding level's capacity. This leaves shared-cache headroom. Missing or insufficiently separated levels remain disabled with an explanation. L1 uses a fully randomized chain; L2 and L3 randomize within 64 KiB blocks to reduce translation overhead. Two full traversals warm the allocation before timing. Batches contain at least 65,536 reads in complete cycles. Observed time includes loop/timer overhead, translation, prefetching and interference. Exact cache-hit rates and hardware hit-cycle counts are not measured. Shared-cache traffic and SMT siblings can affect results. One pinned thread regardless of allocation preset."
        }
        "cpu.latency.memory" => {
            "Follows a fully randomized pointer chain on one pinned CPU thread, modeling dependent reads of scattered object-like nodes. Each node stores an 8-byte pointer with cache-line spacing; its address supplies the next read. This measures access only, not object allocation or application logic. The working set is at least 256 MiB and four times the detected aggregate last-level cache. Allocation, initialization and one full warmup traversal are excluded from timing. Samples cover complete cycles and can exceed the requested duration. Results include address translation, cache effects, loop/timer overhead and system interference. Uses ordinary pages without explicit NUMA binding; this is not DRAM CAS timing. CPU allocation presets do not add workers."
        }
        "cpu.latency.memory.localized" => {
            "Follows dependent pointers randomized within 64 KiB blocks, advancing through the full allocation. Nearby reads reuse page translations, reducing TLB overhead compared with fully scattered reads. This follows the localized pointer-chasing approach described by conventional memory benchmarks, but does not reproduce PassMark's cache-subtest averaging or score. The working set remains at least 256 MiB and four times the detected aggregate last-level cache. Allocation, construction and full-cycle warmup are untimed; samples cover complete cycles. Cache effects, prefetching, block transitions, translation and OS interference can still affect the result. Ordinary pages, one pinned CPU thread, no explicit NUMA binding; lower ns per read is better."
        }
        "cpu.performance.matrix.fp32.scaling" => {
            "Multiplies 32 rows of FP32 inputs by progressively larger FP32 weight matrices, using cache-blocked AVX2/FMA instructions. Every selected CPU worker owns separate A[32,K], B[K,N], and C[32,N] matrices, with K=N. Eight rows share each loaded eight-value weight vector; the kernel uses 64-column and 128-K cache blocks. TOPS counts 2×M×N×K operations per complete product. GB/s reports effective reads and output reads/writes within the kernel, including cache-served reuse. The register-only AVX2 reference is measured before and after the sweep; the latest reference is used. These are raw matrix-workload measurements."
        }
        "cpu.performance.avx2.f32_fma.scaling" => {
            "Measures a register-only AVX2/FMA reference, then streams two FP32 input arrays and one output array through progressively larger datasets. Eight independent 256-bit vectors perform 16 fused multiply-adds per value. Small arrays can stay in L1/L2 cache; larger arrays pass through shared L3 and RAM. The graph reports aggregate TOPS and effective GB/s, with sample ranges. Dataset sizes include every selected worker's three arrays, so they are not literal sizes of one cache. These results are raw hardware measurements for this fixed kernel."
        }
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
            "L3 is a larger cache shared by groups of processor cores. It helps those cores reuse data before reaching out to slower system memory (RAM). It usually holds more than L1 or L2, but takes longer to access. These metrics measure data-transfer speed, rather than access delay."
        }
        "gpu.bandwidth.cache" => {
            "The graphics card keeps frequently reused data in a small, fast cache near its computing units. This test grows the amount of data and looks for speed changes that suggest cache boundaries. Its effective cache layers are estimates from measured behavior."
        }
        "gpu.bandwidth.vram" => {
            "Measures read, write, and copy speeds in the graphics card's memory, often called VRAM. VRAM holds textures, image buffers, and AI model data. Bandwidth tells you how quickly that data moves; memory capacity tells you how much fits."
        }
        "gpu.bandwidth.host_link" => {
            "Measures data transfers between system RAM and the graphics card, usually across PCI Express. Upload sends data to the GPU; download brings results back. Both directions are recorded, with the featured metric showing the first transfer metric."
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
        _ if gpu_offload(id) => {
            "The GPU reads two FP32 input arrays and writes an output array directly on a separate system-RAM heap for the selected offload share. The remaining share uses VRAM. Each dataset preserves the selected split. No CPU computation or timed RAM-to-VRAM staging copy is used. Small sets can benefit from GPU cache; nominal host traffic is not a PCIe bus-counter measurement. This models explicit offload rather than automatic driver eviction. The RAM budget limits the host region; the VRAM budget limits the remaining region. At 100%, all test arrays use system RAM."
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
        _ if cache_latency_level(id).is_some() => {
            "Useful for examining the delay of dependent lookups when a working set fits in the processor's cache hierarchy. Compare the same level with matching working-set size and pinned-core placement. Higher levels include traversal of the memory hierarchy; this is not the incremental delay added by that cache alone. Throughput, cache bandwidth and whole-application speed are separate measurements."
        }
        "cpu.latency.memory" => {
            "Useful when comparing the same machine before and after memory-setting changes, or investigating workloads with dependent lookups such as pointer-based structures. Compare matching working sets and processor placement. This test adds no competing memory-load workers; background activity can still affect results. It does not predict whole-application speed."
        }
        "cpu.latency.memory.localized" => {
            "Useful for comparing RAM settings with less address-translation pressure than widely scattered object reads. Compare this test with its own previous runs using matching working sets and processor placement. Its difference from random-object latency shows sensitivity to access locality, not a direct measurement of TLB-miss cost or a prediction of application speed."
        }
        "cpu.performance.matrix.fp32.scaling" => {
            "Measures the CPU's balance between FP32 matrix arithmetic, data reuse, and cache/RAM access. Matrix multiplication is a building block of neural networks, scientific computing and numerical processing. Compare matching matrix shapes, worker settings and dataset sizes; this is a fixed 32-row workload, not a complete application benchmark."
        }
        "cpu.performance.avx2.f32_fma.scaling" => {
            "Shows the CPU's balance between vector arithmetic and moving data for numerical array processing. Compare the same dataset sizes and worker settings across runs or hardware changes. The streaming kernel has a fixed 2.67 operations per byte of effective traffic; its cache and RAM throughput are specific to that workload."
        }
        "cpu.performance.integer.i64" => {
            "Whole-number calculations support data processing and many program tasks. This all-core test is useful for workloads that can split their work across processor cores; app speed also depends on memory and software."
        }
        "cpu.performance.single_thread.integer.i64" => {
            "Useful context for app responsiveness and parts of game logic that run on one thread. Games and apps combine many kinds of work, so their overall speed also depends on graphics, memory, and other processor tasks."
        }
        "cpu.performance.float.f32" => {
            "Standard-precision math is used in simulations, image processing, and some game physics. This metric shows one part of the processor's calculation ability; the app's choice of instructions and use of multiple cores also matter."
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
            "Fast cache access helps repeated calculations reuse nearby data in games and everyday apps. A program benefits most when its active data fits in the relevant cache. The metric aggregates work across the tested cores."
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
            "FP32 shader math is used in gaming graphics, lighting, visual effects, and rendering. Higher shader throughput can help when shader calculations limit performance. Frame rate also depends on memory, the CPU, and other graphics hardware."
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
        _ if gpu_offload(id) => {
            "Useful for exploring the cost of GPU access to offloaded data. Compare the same dataset across the three offload shares and inspect nominal host traffic in the expanded measurements. The CPU memory controller, interconnect, PCIe link, GPU caches and shader all affect this result; it does not isolate pure PCIe bandwidth or predict complete application speed."
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
    let (analysis_label, analysis_value) = if cache_latency_level(&result.benchmark_id).is_some() {
        (
            "CACHE INSTANCE".into(),
            metadata_size(result, "target_cache_bytes").unwrap_or_else(|| "Not recorded".into()),
        )
    } else if let Some(value) = result.workload_metadata.get("bound_classification") {
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
            if cpu_scaling(&result.benchmark_id) {
                "AVX2 FP32 REFERENCE".into()
            } else if result.benchmark_id == "gpu.performance.fp32.scaling"
                || gpu_offload(&result.benchmark_id)
            {
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

fn chart_tiers(result: &BenchmarkResult) -> Vec<(u64, &Metric)> {
    if !result.benchmark_id.ends_with(".scaling") {
        return Vec::new();
    }
    scaling_compute_tiers(result)
        .into_iter()
        .filter(|(bytes, metric)| {
            *bytes > 0
                && metric.unit == "operations/s"
                && metric.value.is_finite()
                && metric.value > 0.0
        })
        .collect()
}

fn chart_bandwidth(result: &BenchmarkResult, bytes: u64) -> Option<&Metric> {
    let name = if gpu_offload(&result.benchmark_id) {
        "host_bandwidth"
    } else {
        "bandwidth"
    };
    result.metrics.iter().find(|metric| {
        metric.name == format!("working_set_{bytes}.{name}")
            && metric.unit == "bytes/s"
            && metric.value.is_finite()
            && metric.value > 0.0
    })
}

fn chart_sample_range(metric: &Metric) -> (f64, f64) {
    let stats = &metric.statistics;
    if stats.sample_count > 0
        && stats.minimum.is_finite()
        && stats.maximum.is_finite()
        && stats.minimum >= 0.0
        && stats.minimum <= metric.value
        && stats.maximum >= metric.value
    {
        (stats.minimum, stats.maximum)
    } else {
        (metric.value, metric.value)
    }
}

#[cfg(test)]
fn scaling_chart_data(result: &BenchmarkResult) -> Option<(String, Vec<ChartLabel>)> {
    scaling_chart_interactive(result).map(|(svg, labels, _)| (svg, labels))
}

fn scaling_chart_interactive(
    result: &BenchmarkResult,
) -> Option<(String, Vec<ChartLabel>, Vec<ChartPoint>)> {
    use std::fmt::Write;
    let tiers = chart_tiers(result);
    let (first, last) = (tiers.first()?, tiers.last()?);
    let low = (first.0 as f64).log2();
    let span = ((last.0 as f64).log2() - low).max(1.0);
    let x = |bytes: u64| 110.0 + ((bytes as f64).log2() - low) / span * 760.0;
    let reference = result.metrics.iter().find(|metric| {
        metric.name == "measured_compute_ceiling"
            && metric.unit == "operations/s"
            && metric.value.is_finite()
            && metric.value > 0.0
    });
    let mut svg = String::from(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420" viewBox="0 0 900 420"><rect width="900" height="420" rx="10" fill="#101c29"/>"##,
    );
    let mut labels = Vec::new();
    let mut hover_points = Vec::new();
    let label =
        |x: f32, y: f32, width: f32, text: String, font_size: f32, color: &str, alignment: i32| {
            ChartLabel {
                x,
                y,
                width,
                text: text.into(),
                font_size,
                alignment,
                accent: match color {
                    "#73dcca" => Color::from_rgb_u8(115, 220, 202),
                    "#78a9ff" => Color::from_rgb_u8(120, 169, 255),
                    "#edc778" => Color::from_rgb_u8(237, 199, 120),
                    _ => Color::from_rgb_u8(184, 198, 215),
                },
            }
        };
    for (bandwidth, top, color, title, scale) in [
        (false, 42.0, "#73dcca", "Compute throughput · TOPS", 1e12),
        (
            true,
            232.0,
            "#78a9ff",
            if gpu_offload(&result.benchmark_id) {
                "Nominal host traffic · GB/s (cache included)"
            } else {
                "Effective test traffic · GB/s"
            },
            1e9,
        ),
    ] {
        let metric_at = |bytes, compute| {
            if bandwidth {
                chart_bandwidth(result, bytes)
            } else {
                Some(compute)
            }
        };
        let maximum = tiers
            .iter()
            .filter_map(|(bytes, compute)| metric_at(*bytes, *compute))
            .map(|metric| chart_sample_range(metric).1)
            .chain(reference.filter(|_| !bandwidth).map(|metric| metric.value))
            .fold(0.0_f64, f64::max);
        labels.push(label(
            80.0,
            (top - 34.0) as f32,
            600.0,
            title.into(),
            16.0,
            color,
            0,
        ));
        if maximum <= 0.0 {
            labels.push(label(
                80.0,
                (top + 35.0) as f32,
                600.0,
                "No traffic measurements recorded".into(),
                13.0,
                "",
                0,
            ));
            continue;
        }
        let maximum = maximum * 1.08;
        let y = |value: f64| top + 120.0 * (1.0 - value / maximum);
        for tick in 0..=4 {
            let position = 110.0 + 760.0 * tick as f64 / 4.0;
            writeln!(svg, r##"<line x1="{position:.2}" x2="{position:.2}" y1="{top:.2}" y2="{:.2}" stroke="#233346" stroke-dasharray="3 5"/>"##, top + 120.0).unwrap();
        }
        for tick in 0..=4 {
            let value = maximum * tick as f64 / 4.0;
            writeln!(
                svg,
                r##"<line x1="110" x2="870" y1="{0:.2}" y2="{0:.2}" stroke="#2b3d50"/>"##,
                y(value)
            )
            .unwrap();
            labels.push(label(
                0.0,
                (y(value) - 9.0) as f32,
                100.0,
                format!(
                    "{:.1} {}",
                    value / scale,
                    if bandwidth { "GB/s" } else { "TOPS" }
                ),
                13.0,
                "",
                2,
            ));
        }
        for pair in tiers.windows(2) {
            let ((a_bytes, a), (b_bytes, b)) = (pair[0], pair[1]);
            if let (Some(a), Some(b)) = (metric_at(a_bytes, a), metric_at(b_bytes, b)) {
                let (a_min, a_max) = chart_sample_range(a);
                let (b_min, b_max) = chart_sample_range(b);
                writeln!(svg, r#"<polygon points="{:.2},{:.2} {:.2},{:.2} {:.2},{:.2} {:.2},{:.2}" fill="{color}" fill-opacity="0.16"/><line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{color}" stroke-width="2.5"/>"#, x(a_bytes), y(a_max), x(b_bytes), y(b_max), x(b_bytes), y(b_min), x(a_bytes), y(a_min), x(a_bytes), y(a.value), x(b_bytes), y(b.value)).unwrap();
            }
        }
        for (bytes, compute) in &tiers {
            if let Some(metric) = metric_at(*bytes, *compute) {
                hover_points.push(chart_hover::point(
                    *bytes,
                    metric,
                    [x(*bytes), y(metric.value), top + 120.0],
                    (title, scale, if bandwidth { "GB/s" } else { "TOPS" }),
                    if bandwidth {
                        Color::from_rgb_u8(120, 169, 255)
                    } else {
                        Color::from_rgb_u8(115, 220, 202)
                    },
                ));
                writeln!(
                    svg,
                    r#"<circle cx="{:.2}" cy="{:.2}" r="3.5" fill="{color}"/>"#,
                    x(*bytes),
                    y(metric.value)
                )
                .unwrap();
            }
        }
        if let Some(reference) = reference.filter(|_| !bandwidth) {
            writeln!(svg, r##"<line x1="110" x2="870" y1="{0:.2}" y2="{0:.2}" stroke="#edc778" stroke-width="1.5" stroke-dasharray="6 5"/>"##, y(reference.value)).unwrap();
            labels.push(label(
                650.0,
                (y(reference.value) - 22.0) as f32,
                220.0,
                format!("Reference {:.2} TOPS", reference.value / scale),
                13.0,
                "#edc778",
                2,
            ));
        }
    }
    let mut previous_x = -100.0;
    let mut previous_index = usize::MAX;
    for tick in 0..=4 {
        let index = (tiers.len() - 1) * tick / 4;
        let bytes = tiers[index].0;
        let position = x(bytes);
        if index == previous_index
            || (tick != 4 && position - previous_x < 110.0)
            || (tick != 4 && x(last.0) - position < 110.0)
        {
            continue;
        }
        previous_x = position;
        previous_index = index;
        writeln!(
            svg,
            r##"<line x1="{position:.2}" x2="{position:.2}" y1="352" y2="358" stroke="#8f9caf"/>"##
        )
        .unwrap();
        let (left, alignment) = if tick == 0 {
            (position, 0)
        } else if tick == 4 {
            (position - 120.0, 2)
        } else {
            (position - 60.0, 1)
        };
        labels.push(label(
            left as f32,
            364.0,
            120.0,
            format_binary_size(bytes as f64),
            13.0,
            "",
            alignment,
        ));
    }
    labels.push(label(
        80.0,
        391.0,
        790.0,
        "Dataset size (logarithmic scale) →".into(),
        13.0,
        "",
        1,
    ));
    svg.push_str("</svg>");
    Some((svg, labels, hover_points))
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
            let reference_label = if cpu_scaling(&result.benchmark_id) {
                "AVX2 FP32"
            } else if result.benchmark_id == "gpu.performance.fp32.scaling"
                || gpu_offload(&result.benchmark_id)
            {
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
        "Compare this metric with earlier runs of the same test. More information is available under Details."
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
        "ns" => gpu_timings::format_time(value),
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
    if name == "cache_resident_compute" || name == "small_set_compute" {
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
            "host_bandwidth" => "nominal host traffic (cache included)",
            "gpu_execution_time" => "GPU execution time per pass",
            "end_to_end_time" => "End-to-end time per pass",
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
    #[test]
    fn cache_latency_levels_have_separate_lower_best_results_and_local_instance_comparisons() {
        let make = |level, value| BenchmarkResult {
            benchmark_id: format!("cpu.latency.cache.l{level}"),
            device_id: "cpu:system".into(),
            elapsed_ns: 1000,
            metrics: vec![Metric {
                name: "read_latency".into(),
                value,
                unit: "ns".into(),
                statistics: SampleStatistics::default(),
            }],
            workload_metadata: BTreeMap::from([
                ("working_set_bytes".into(), "2097152".into()),
                ("target_cache_bytes".into(), "33554432".into()),
                ("cache_processor_mask".into(), "ffff".into()),
            ]),
            device_metadata: BTreeMap::new(),
        };
        let current = make(3, 12.0);
        let previous = make(3, 15.0);
        assert_eq!(
            super::best_primary(&[&current, &previous]).unwrap().value,
            12.0
        );
        assert_eq!(
            super::primary_delta(&current, std::slice::from_ref(&previous), None),
            "-20.0%"
        );
        assert!(
            super::comparison_metric(&current, "read_latency", &[make(2, 3.0)], None).is_none()
        );
        let mut different_instance = previous.clone();
        different_instance
            .workload_metadata
            .insert("cache_processor_mask".into(), "ffff0000".into());
        assert!(
            super::comparison_metric(&current, "read_latency", &[different_instance], None)
                .is_none()
        );
        let samples = [make(1, 1.0), make(2, 3.0), current.clone()];
        assert_eq!(super::component_results(&samples, "cpu:system").len(), 3);
        assert_eq!(super::result_section(&current), 3);
        assert!(super::result_settings_note(&current).contains("L3 working set"));
        assert_eq!(
            super::result_visuals::facts(&current, current.metrics.first())[0].label,
            "MEASURED READ LATENCY"
        );
    }

    #[test]
    fn ram_latency_uses_lower_best_and_matching_working_sets() {
        use gluj_bench_core::{BenchmarkResult, Metric, SampleStatistics};
        let make = |value| BenchmarkResult {
            benchmark_id: "cpu.latency.memory".into(),
            device_id: "memory:system".into(),
            elapsed_ns: 1000,
            metrics: vec![Metric {
                name: "read_latency".into(),
                value,
                unit: "ns".into(),
                statistics: SampleStatistics::default(),
            }],
            workload_metadata: std::collections::BTreeMap::from([
                ("working_set_bytes".into(), "268435456".into()),
                ("latency_profile_revision".into(), "1".into()),
            ]),
            device_metadata: Default::default(),
        };
        let current = make(90.0);
        let previous = make(100.0);
        let fast = make(80.0);
        let mut incompatible = make(10.0);
        incompatible
            .workload_metadata
            .insert("working_set_bytes".into(), "536870912".into());
        assert_eq!(
            super::best_primary(&[&current, &previous, &fast, &incompatible])
                .unwrap()
                .value,
            80.0
        );
        assert_eq!(super::primary_delta(&current, &[previous], None), "-10.0%");
        assert!(
            super::comparison_metric(&current, "read_latency", &[incompatible], None).is_none()
        );
        assert_eq!(super::result_section(&current), 3);
        assert!(super::format_metric(90.0, "ns").contains("ns"));
        assert!(super::result_settings_note(&current).contains("negative Change"));
        assert_eq!(
            super::result_visuals::facts(&current, current.metrics.first())[0].label,
            "MEASURED READ LATENCY"
        );
        let mut localized = current.clone();
        localized.benchmark_id = "cpu.latency.memory.localized".into();
        localized
            .workload_metadata
            .insert("locality_block_bytes".into(), "65536".into());
        let mut localized_old = localized.clone();
        localized_old.metrics[0].value = 100.0;
        assert!(super::ram_latency(&localized));
        assert_eq!(
            super::best_primary(&[&localized, &localized_old])
                .unwrap()
                .value,
            90.0
        );
        assert_eq!(
            super::primary_delta(&localized, std::slice::from_ref(&localized_old), None),
            "-10.0%"
        );
        assert!(
            super::comparison_metric(
                &localized,
                "read_latency",
                std::slice::from_ref(&current),
                None
            )
            .is_none()
        );
        assert_eq!(
            super::component_results(&[current, localized.clone()], "cpu:system").len(),
            2
        );
        localized_old
            .workload_metadata
            .insert("locality_block_bytes".into(), "32768".into());
        assert!(
            super::comparison_metric(&localized, "read_latency", &[localized_old], None).is_none()
        );
    }

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
    fn cpu_allocation_comparisons_and_tuning_use_current_settings_and_stable_metrics() {
        let mut result = saved_test_result(1e12);
        result.benchmark_id = "cpu.performance.avx2.f32_fma.scaling".into();
        result.metrics[0].name = "working_set_1073741824.compute".into();
        result.metrics[0].statistics.sample_count = 5;
        result.metrics[0].statistics.standard_deviation = 1e10;
        result
            .workload_metadata
            .insert("cpu_core_limit".into(), "8".into());
        result
            .workload_metadata
            .insert("ram_budget_percent".into(), "20".into());
        result
            .workload_metadata
            .insert("suggested_cpu_core_count_trial".into(), "6".into());
        assert!(super::tuning_takeaway(&result).starts_with("Try 6 physical cores"));
        let mut other = result.clone();
        other
            .workload_metadata
            .insert("cpu_core_limit".into(), "6".into());
        assert_ne!(
            super::result_load_signature(&other),
            super::result_load_signature(&result)
        );
        other = result.clone();
        other
            .workload_metadata
            .insert("ram_budget_percent".into(), "80".into());
        assert_ne!(
            super::result_load_signature(&other),
            super::result_load_signature(&result)
        );
        result.metrics[0].statistics.standard_deviation = 2e11;
        assert!(super::tuning_takeaway(&result).starts_with("Repeat the run"));
        let defaults: super::AppSettings = serde_json::from_str(r#"{"cpu_intensity":0}"#).unwrap();
        assert_eq!(defaults.ram_budget_percent, 20);
        assert_eq!(defaults.cpu_core_limit, 0);
    }

    #[test]
    fn ram_offload_results_show_host_traffic_and_compare_matching_budgets() {
        use super::{
            chart_bandwidth, comparison_metric, result_summary, scaling_chart_data,
            test_description, tuning_takeaway,
        };
        use slint::Model;
        let mut result = saved_test_result(4e12);
        result.device_id = "gpu:one".into();
        result.benchmark_id = "gpu.performance.fp32.offload75.scaling".into();
        result.metrics[0].name = "working_set_1048576.compute".into();
        for (name, value) in [("bandwidth", 40e9), ("host_bandwidth", 30e9)] {
            result.metrics.push(Metric {
                name: format!("working_set_1048576.{name}"),
                value,
                unit: "bytes/s".into(),
                statistics: SampleStatistics::default(),
            });
        }
        result
            .workload_metadata
            .insert("ram_offload_percent".into(), "75".into());
        result
            .workload_metadata
            .insert("ram_budget_percent".into(), "20".into());
        let row = result_row(&result, &[], &[]);
        let tier = row.profile_statistics.row_data(0).unwrap();
        assert!(tier.contains("Nominal host traffic (cache included): 30.00 GB/s"));
        assert!(tier.contains("Placement: 768.00 KiB RAM + 256.00 KiB VRAM"));
        assert!(test_description(&result.benchmark_id).contains("system-RAM heap"));
        assert!(tuning_takeaway(&result).contains("GPU core-frequency limit"));
        assert!(row.tuning_guidance.contains("smaller batches"));
        assert!(
            !row.tuning_guidance
                .contains("Change the RAM offload percentage")
        );
        assert!(!result_summary(&result).contains("FP16 matrix"));
        assert!(
            scaling_chart_data(&result)
                .unwrap()
                .1
                .iter()
                .any(|label| label.text.contains("Nominal host traffic"))
        );
        assert_eq!(chart_bandwidth(&result, 1048576).unwrap().value, 30e9);
        let previous = result.clone();
        assert!(
            comparison_metric(
                &result,
                "working_set_1048576.compute",
                std::slice::from_ref(&previous),
                None
            )
            .is_some()
        );
        result
            .workload_metadata
            .insert("ram_budget_percent".into(), "80".into());
        assert!(
            comparison_metric(&result, "working_set_1048576.compute", &[previous], None).is_none()
        );
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
            ram_budget_percent: 80,
            cpu_core_limit: 4,
            ..super::AppSettings::default()
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
    fn tuning_takeaway_and_breakdown_use_recorded_metrics_and_keep_details_separate() {
        let mut result = saved_test_result(40e12);
        result.benchmark_id = "gpu.performance.fp32.scaling".into();
        result.metrics[0].name = "working_set_4294967296.compute".into();
        let mut reference = result.metrics[0].clone();
        reference.name = "measured_compute_ceiling".into();
        reference.value = 50e12;
        result.metrics.insert(0, reference);
        result.workload_metadata.insert(
            "suggested_core_frequency_limit_reduction_percent".into(),
            "10".into(),
        );
        result.workload_metadata.insert(
            "tuning_guidance".into(),
            "Detailed reasoning stays available.".into(),
        );
        let row = result_row(&result, &[], &[]);
        assert_eq!(
            row.tuning_tldr.as_str(),
            "Try a further 10% core-limit reduction, then retest for 0–5% throughput loss."
        );
        assert!(row.tuning_stats.as_str().contains("4.00 GiB"));
        assert!(row.tuning_stats.as_str().contains("40.00 TOPS"));
        assert!(row.tuning_stats.as_str().contains("50.00 TOPS"));
        assert!(row.tuning_stats.as_str().contains("-20.0%"));
        assert_eq!(
            row.tuning_guidance.as_str(),
            "Detailed reasoning stays available."
        );
        assert!(row.description_tldr.as_str().contains("larger data sets"));
        result
            .workload_metadata
            .remove("suggested_core_frequency_limit_reduction_percent");
        result.workload_metadata.insert(
            "tuning_guidance".into(),
            "No frequency-limit trial suggested.".into(),
        );
        assert!(super::tuning_takeaway(&result).starts_with("Large throughput drop detected"));
    }

    #[test]
    fn cpu_avx2_scaling_uses_cpu_reference_graph_and_memory_explanations() {
        let mut result = saved_test_result(2e12);
        result.benchmark_id = "cpu.performance.avx2.f32_fma.scaling".into();
        result.metrics[0].name = "measured_compute_ceiling".into();
        for (bytes, value) in [(196608_u64, 1.2e12), (393216, 1.1e12), (1073741824, 0.3e12)] {
            result.metrics.push(Metric {
                name: format!("working_set_{bytes}.compute"),
                value,
                unit: "operations/s".into(),
                statistics: SampleStatistics {
                    sample_count: 5,
                    minimum: value * 0.99,
                    maximum: value * 1.01,
                    median: value,
                    standard_deviation: value * 0.01,
                },
            });
            result.metrics.push(Metric {
                name: format!("working_set_{bytes}.bandwidth"),
                value: value * 12.0 / 32.0,
                unit: "bytes/s".into(),
                statistics: SampleStatistics::default(),
            });
        }
        result.workload_metadata.insert(
            "tuning_guidance".into(),
            "Raw CPU memory-pressure measurements.".into(),
        );
        result
            .workload_metadata
            .insert("thread_count".into(), "16".into());
        result
            .workload_metadata
            .insert("bandwidth_transition_status".into(), "observed".into());
        let row = result_row(&result, &[], &[]);
        assert_eq!(row.tuning_title.as_str(), "TUNING SUGGESTIONS");
        assert_eq!(row.fact_one_label.as_str(), "AVX2 FP32 REFERENCE");
        assert!(row.summary.as_str().contains("AVX2 FP32 reference"));
        assert!(row.tuning_tldr.as_str().contains("TOPS and GB/s"));
        assert!(!row.tuning_tldr.as_str().contains("core-limit"));
        assert!(row.tuning_stats.as_str().contains("RAM-bus utilization"));
        assert!(!row.tuning_stats.as_str().contains("VRAM"));
        use slint::Model;
        assert_eq!(row.profile_sizes.row_count(), 3);
        assert!(row.description.as_str().contains("RAM"));
        assert!(
            row.everyday_use
                .as_str()
                .contains("2.67 operations per byte")
        );
    }

    #[test]
    fn cpu_matrix_scaling_shows_per_tier_shapes_and_matrix_traffic_convention() {
        let mut result = saved_test_result(1e12);
        result.benchmark_id = "cpu.performance.matrix.fp32.scaling".into();
        result.metrics[0].name = "measured_compute_ceiling".into();
        let mut shapes = Vec::new();
        for n in [32_u64, 64, 256] {
            let bytes = 4 * (n * n + 64 * n) * 16;
            shapes.push(format!("{bytes}:32x{n}x{n}"));
            for (suffix, value, unit) in [
                ("compute", 0.5e12, "operations/s"),
                ("bandwidth", 250e9, "bytes/s"),
            ] {
                result.metrics.push(Metric {
                    name: format!("working_set_{bytes}.{suffix}"),
                    value,
                    unit: unit.into(),
                    statistics: SampleStatistics {
                        sample_count: 5,
                        median: value,
                        minimum: value * 0.99,
                        maximum: value * 1.01,
                        standard_deviation: value * 0.01,
                    },
                });
            }
        }
        result
            .workload_metadata
            .insert("matrix_profile_shapes".into(), shapes.join(","));
        result.workload_metadata.insert(
            "tuning_guidance".into(),
            "Compare matrix throughput.".into(),
        );
        let row = result_row(&result, &[], &[]);
        assert_eq!(row.tuning_title.as_str(), "TUNING SUGGESTIONS");
        assert!(row.summary.as_str().contains("AVX2 FP32 reference"));
        assert!(!row.summary.as_str().contains("FP16"));
        assert!(!row.tuning_tldr.as_str().contains("core-limit"));
        assert!(row.description.as_str().contains("cache-blocked"));
        assert!(row.tuning_stats.as_str().contains("32 × 256 × 256"));
        assert!(row.tuning_stats.as_str().contains("blocked kernel"));
        use slint::Model;
        assert!(
            row.profile_statistics
                .row_data(0)
                .unwrap()
                .as_str()
                .contains("32 × 32 × 32")
        );
        assert!(
            row.profile_statistics
                .row_data(2)
                .unwrap()
                .as_str()
                .contains("32 × 256 × 256")
        );
        result
            .workload_metadata
            .insert("matrix_profile_shapes".into(), "196608:32xNaNx32".into());
        assert!(super::cpu_matrix_tier_shape(&result, 196608).is_none());
    }

    #[test]
    fn scaling_chart_renders_sorted_tiers_with_reference_and_sample_ranges() {
        let mut result = saved_test_result(90e12);
        result.benchmark_id = "gpu.performance.matrix.fp16.scaling".into();
        result.metrics[0].name = "measured_compute_ceiling".into();
        for (bytes, compute, bandwidth) in [
            (8589934592_u64, 52e12, 810e9),
            (262144, 74e12, 2310e9),
            (1048576, 79e12, 2470e9),
            (4194304, 84e12, 1400e9),
            (16777216, 85e12, 1380e9),
            (67108864, 83e12, 1320e9),
            (268435456, 49e12, 760e9),
            (1073741824, 47e12, 730e9),
        ] {
            for (suffix, value, unit) in [
                ("compute", compute, "operations/s"),
                ("bandwidth", bandwidth, "bytes/s"),
            ] {
                result.metrics.push(Metric {
                    name: format!("working_set_{bytes}.{suffix}"),
                    value,
                    unit: unit.into(),
                    statistics: SampleStatistics {
                        sample_count: 5,
                        minimum: value * 0.85,
                        maximum: value * 1.05,
                        median: value,
                        standard_deviation: value * 0.06,
                    },
                });
            }
        }
        let (svg, labels) = super::scaling_chart_data(&result).unwrap();
        for expected in [
            "Reference 90.00 TOPS",
            "Effective test traffic · GB/s",
            "256.00 KiB",
            "8.00 GiB",
        ] {
            assert!(labels.iter().any(|label| label.text.as_str() == expected));
        }
        assert_eq!(svg.matches("<circle").count(), 16);
        assert_eq!(svg.matches("<polygon").count(), 14);
        let image = slint::Image::load_from_svg_data(svg.as_bytes()).unwrap();
        let pixels = image.to_rgba8().unwrap();
        assert_eq!((pixels.width(), pixels.height()), (900, 420));
        let row = result_row(&result, &[], &[]);
        use slint::Model;
        assert_eq!(row.profile_sizes.row_count(), 8);
        assert_eq!(row.profile_points.row_count(), 16);
        let hover = row.profile_points.row_data(15).unwrap();
        assert_eq!(hover.display_value, "810 GB/s");
        assert_eq!(hover.dataset_label, "8.00 GiB");
        assert_eq!(hover.bottom, 352.0);
        assert_eq!(
            row.profile_sizes.row_data(0).unwrap().as_str(),
            "256.00 KiB"
        );
        assert!(
            row.profile_statistics
                .row_data(7)
                .unwrap()
                .as_str()
                .contains("810.00 GB/s")
        );
        if let Ok(directory) = std::env::var("GLUJ_CHART_PREVIEW_DIR") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join("scaling-chart.svg"), svg).unwrap();
            std::fs::write(directory.join("labels.json"), serde_json::to_vec(&labels.iter().map(|label| serde_json::json!({"x":label.x,"y":label.y,"width":label.width,"text":label.text.as_str(),"font_size":label.font_size,"alignment":label.alignment})).collect::<Vec<_>>()).unwrap()).unwrap();
            let mut ppm = b"P6\n900 420\n255\n".to_vec();
            for pixel in pixels.as_slice() {
                ppm.extend_from_slice(&[pixel.r, pixel.g, pixel.b]);
            }
            std::fs::write(directory.join("scaling-chart.ppm"), ppm).unwrap();
        }
        // Invalid rates cannot create NaN coordinates, and missing traffic stays missing.
        result.metrics.retain(|m| !m.name.ends_with(".bandwidth"));
        result.metrics[1].value = f64::NAN;
        let (svg, labels) = super::scaling_chart_data(&result).unwrap();
        assert!(
            labels
                .iter()
                .any(|label| label.text.as_str() == "No traffic measurements recorded")
        );
        assert!(!svg.contains("NaN"));
        result.benchmark_id = "cpu.performance.float.f32".into();
        assert!(super::scaling_chart_data(&result).is_none());
    }

    #[test]
    fn noisy_matrix_drop_reports_memory_evidence_without_claiming_exact_utilization() {
        let mut result = saved_test_result(84.57e12);
        result.benchmark_id = "gpu.performance.matrix.fp16.scaling".into();
        result.metrics[0].name = "measured_compute_ceiling".into();
        result.metrics[0].statistics.sample_count = 5;
        result.metrics[0].statistics.standard_deviation = 8.31e12;
        for (name, value, deviation) in [
            ("working_set_262144.compute", 79.17e12, 0.23e12),
            ("working_set_524288.compute", 79.14e12, 0.3e12),
            ("working_set_1048576.compute", 79.09e12, 0.3e12),
            ("working_set_8589934592.compute", 51.85e12, 7.52e12),
            ("working_set_8589934592.bandwidth", 810.2e9, 117e9),
        ] {
            result.metrics.push(Metric {
                name: name.into(),
                value,
                unit: if name.ends_with(".bandwidth") {
                    "bytes/s"
                } else {
                    "operations/s"
                }
                .into(),
                statistics: SampleStatistics {
                    sample_count: 5,
                    standard_deviation: deviation,
                    ..Default::default()
                },
            });
        }
        result.workload_metadata.insert(
            "tuning_guidance".into(),
            "No frequency-limit trial suggested.".into(),
        );
        result
            .workload_metadata
            .insert("bandwidth_transition_status".into(), "observed".into());
        result
            .workload_metadata
            .insert("allocated_test_buffer_bytes".into(), "8589934592".into());
        result
            .workload_metadata
            .insert("allocation_budget_bytes".into(), "20602421200".into());
        let row = result_row(&result, &[], &[]);
        assert!(
            row.tuning_tldr
                .as_str()
                .contains("Large throughput drop detected")
        );
        assert!(row.tuning_tldr.as_str().contains("consistent measurements"));
        assert!(
            row.memory_pressure
                .as_str()
                .contains("possible memory bottleneck")
        );
        assert!(row.memory_pressure.as_str().contains("34.5% slower"));
        assert!(row.memory_pressure.as_str().contains("810.20 GB/s"));
        assert!(
            row.tuning_stats
                .as_str()
                .contains("14.5% sample variation exceeds the 10%")
        );
        assert!(
            row.tuning_stats
                .as_str()
                .contains("Test memory allocation: 8.00 GiB")
        );
        assert!(
            row.tuning_stats
                .as_str()
                .contains("not memory-controller utilization")
        );
        assert!(row.tuning_stats.as_str().contains("physical VRAM traffic"));

        // Consistent samples strengthen the inference; a gap without a sustained
        // transition still must not be labelled a demonstrated memory bottleneck.
        for metric in &mut result.metrics {
            metric.statistics.standard_deviation = metric.value * 0.01;
        }
        assert!(super::memory_pressure_summary(&result).contains("likely memory bottleneck"));
        result.workload_metadata.insert(
            "bandwidth_transition_status".into(),
            "not_observed_within_tested_range".into(),
        );
        assert!(super::memory_pressure_summary(&result).contains("bottleneck not established"));
        result
            .workload_metadata
            .insert("bandwidth_transition_status".into(), "observed".into());
        result
            .metrics
            .iter_mut()
            .find(|m| m.name == "working_set_8589934592.compute")
            .unwrap()
            .value = 80e12;
        assert!(super::memory_pressure_summary(&result).contains("bottleneck not established"));
    }

    #[test]
    fn tuning_box_uses_latest_run_even_when_an_older_run_has_the_best_score() {
        let mut older = saved_test_result(100e12);
        older.benchmark_id = "gpu.performance.fp32.scaling".into();
        older.device_id = "gpu:one".into();
        older.metrics[0].name = "working_set_4294967296.compute".into();
        older
            .workload_metadata
            .insert("tuning_guidance".into(), "Earlier suggestion".into());
        let mut latest = older.clone();
        latest.metrics[0].value = 40e12;
        latest
            .workload_metadata
            .insert("tuning_guidance".into(), "Latest suggestion".into());
        let results = [older, latest];
        let groups = super::component_results(&results, "gpu:one");
        assert_eq!(super::best_primary(&groups[0]).unwrap().value, 100e12);
        let row = result_row(groups[0][0], &[], &[]);
        assert_eq!(row.primary_value.as_str(), "40.00 TOPS");
        assert_eq!(row.tuning_guidance.as_str(), "Latest suggestion");
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
        assert!(row.metric_guide.as_str().contains("20%"));
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
