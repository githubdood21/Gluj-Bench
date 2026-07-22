#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use gluj_bench_core::{
    BenchmarkDescriptor, BenchmarkResult, DeviceCategory, DeviceDescriptor, PROTOCOL_VERSION,
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::Duration,
};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 780.0])
            .with_min_inner_size([900.0, 620.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Gluj-Bench",
        options,
        Box::new(|creation_context| {
            configure_style(&creation_context.egui_ctx);
            Ok(Box::new(GlujBenchApp::new()))
        }),
    )
}

const ACCENT: egui::Color32 = egui::Color32::from_rgb(61, 214, 198);
const ACCENT_BLUE: egui::Color32 = egui::Color32::from_rgb(93, 145, 255);
const SURFACE: egui::Color32 = egui::Color32::from_rgb(20, 24, 32);
const SURFACE_RAISED: egui::Color32 = egui::Color32::from_rgb(27, 32, 42);
const BORDER: egui::Color32 = egui::Color32::from_rgb(48, 57, 72);
const MUTED: egui::Color32 = egui::Color32::from_rgb(147, 158, 177);

fn configure_style(context: &egui::Context) {
    context.set_theme(egui::Theme::Dark);
    let mut style = (*context.style_of(egui::Theme::Dark)).clone();
    style.spacing.item_spacing = egui::vec2(10.0, 10.0);
    style.spacing.button_padding = egui::vec2(14.0, 8.0);
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = egui::Color32::from_rgb(13, 16, 22);
    style.visuals.window_fill = SURFACE;
    style.visuals.extreme_bg_color = egui::Color32::from_rgb(10, 13, 18);
    style.visuals.faint_bg_color = egui::Color32::from_rgb(24, 29, 38);
    style.visuals.selection.bg_fill = ACCENT.gamma_multiply(0.28);
    style.visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.noninteractive.bg_fill = SURFACE;
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.widgets.noninteractive.corner_radius = 7.into();
    style.visuals.widgets.inactive.bg_fill = SURFACE_RAISED;
    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    style.visuals.widgets.inactive.corner_radius = 7.into();
    style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(36, 44, 57);
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT_BLUE);
    style.visuals.widgets.hovered.corner_radius = 7.into();
    style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(33, 68, 72);
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    style.visuals.widgets.active.corner_radius = 7.into();
    context.set_style_of(egui::Theme::Dark, style);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Benchmarks,
    Results,
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
                    let output_sender = sender.clone();
                    thread::spawn(move || {
                        for line in BufReader::new(stdout).lines() {
                            match line {
                                Ok(line) => match serde_json::from_str(&line) {
                                    Ok(response) => {
                                        let _ = output_sender.send(WorkerEvent::Response(response));
                                    }
                                    Err(problem) => {
                                        let _ = output_sender.send(WorkerEvent::Status(format!(
                                            "Worker returned invalid JSON: {problem}"
                                        )));
                                    }
                                },
                                Err(problem) => {
                                    let _ = output_sender.send(WorkerEvent::Status(format!(
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
                    "Benchmark worker connected; loading capabilities...".into(),
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

struct GlujBenchApp {
    tab: Tab,
    worker: WorkerClient,
    status: String,
    devices: Vec<DeviceDescriptor>,
    benchmarks: Vec<BenchmarkDescriptor>,
    selected_suite: String,
    selected_gpu_id: Option<String>,
    selected: Option<String>,
    results: Vec<BenchmarkResult>,
    queue: VecDeque<String>,
    active_request: Option<String>,
    progress: f32,
    progress_message: String,
    request_counter: u64,
}

impl GlujBenchApp {
    fn new() -> Self {
        let (worker, status) = WorkerClient::start();
        let mut app = Self {
            tab: Tab::Overview,
            worker,
            status,
            devices: Vec::new(),
            benchmarks: Vec::new(),
            selected_suite: "cpu.bandwidth".into(),
            selected_gpu_id: None,
            selected: None,
            results: Vec::new(),
            queue: VecDeque::new(),
            active_request: None,
            progress: 0.0,
            progress_message: String::new(),
            request_counter: 0,
        };
        app.request_capabilities();
        app
    }
    fn next_id(&mut self, prefix: &str) -> String {
        self.request_counter += 1;
        format!("ui-{prefix}-{}", self.request_counter)
    }
    fn request_capabilities(&mut self) {
        for command in ["devices", "benchmarks"] {
            let id = self.next_id(command);
            if let Err(problem) = self
                .worker
                .send(json!({"protocol":PROTOCOL_VERSION,"id":id,"command":command}))
            {
                self.status = problem;
                break;
            }
        }
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
        match self.worker.send(json!({"protocol":PROTOCOL_VERSION,"id":id,"command":"run","arguments":{"benchmark_id":benchmark_id,"target_duration_ms":5000,"samples":5,"options":options}})) {
            Ok(()) => { self.active_request=Some(id); self.progress=0.0; self.progress_message=format!("Starting {benchmark_id}"); self.status="Benchmark running...".into(); }
            Err(problem) => { self.status=problem; self.queue.clear(); }
        }
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
            .filter(|item| item.suite_id == self.selected_suite && self.benchmark_available(item))
            .cloned()
            .collect();
        items.sort_by_key(|item| item.display_order);
        self.queue = items.into_iter().map(|item| item.id).collect();
        self.start_next();
    }
    fn cancel(&mut self) {
        self.queue.clear();
        if let Some(request_id) = self.active_request.clone() {
            let id = self.next_id("cancel");
            let _=self.worker.send(json!({"protocol":PROTOCOL_VERSION,"id":id,"command":"cancel","arguments":{"request_id":request_id}}));
            self.status = "Cancelling benchmark...".into();
        }
    }
    fn receive_worker_events(&mut self) {
        let mut continue_queue = false;
        while let Ok(event) = self.worker.events.try_recv() {
            match event {
                WorkerEvent::Status(message) => self.status = message,
                WorkerEvent::Response(response) => {
                    let response_id = response["id"].as_str().unwrap_or_default();
                    match response["type"].as_str().unwrap_or_default() {
                        "progress" if self.active_request.as_deref() == Some(response_id) => {
                            self.progress =
                                response["data"]["fraction"].as_f64().unwrap_or(0.0) as f32;
                            self.progress_message = response["data"]["message"]
                                .as_str()
                                .unwrap_or_default()
                                .to_owned();
                        }
                        "result" if self.active_request.as_deref() == Some(response_id) => {
                            match serde_json::from_value(response["data"].clone()) {
                                Ok(result) => {
                                    self.results.push(result);
                                    self.status = "Benchmark completed.".into();
                                    self.tab = Tab::Results;
                                }
                                Err(problem) => self.status = problem.to_string(),
                            }
                            self.active_request = None;
                            self.progress = 1.0;
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
                                self.progress = 0.0;
                            }
                            self.status = message;
                        }
                        _ => {
                            if let Some(devices) = response["data"].get("devices") {
                                match serde_json::from_value(devices.clone()) {
                                    Ok(items) => {
                                        self.devices = items;
                                        if self.selected_gpu_id.as_ref().is_none_or(|selected| {
                                            !self.devices.iter().any(|device| {
                                                device.category == DeviceCategory::Gpu
                                                    && device.available
                                                    && &device.id == selected
                                            })
                                        }) {
                                            self.selected_gpu_id = self
                                                .devices
                                                .iter()
                                                .find(|device| {
                                                    device.category == DeviceCategory::Gpu
                                                        && device.available
                                                })
                                                .map(|device| device.id.clone());
                                        }
                                        self.status = "Hardware capabilities loaded.".into();
                                    }
                                    Err(problem) => self.status = problem.to_string(),
                                }
                            }
                            if let Some(benchmarks) = response["data"].get("benchmarks") {
                                match serde_json::from_value(benchmarks.clone()) {
                                    Ok(items) => {
                                        self.benchmarks = items;
                                        if self.selected.is_none() {
                                            self.selected = self
                                                .benchmarks
                                                .iter()
                                                .find(|item| {
                                                    item.suite_id == self.selected_suite
                                                        && self.benchmark_available(item)
                                                })
                                                .map(|item| item.id.clone());
                                        }
                                    }
                                    Err(problem) => self.status = problem.to_string(),
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
    fn tabs(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .fill(SURFACE)
            .inner_margin(egui::Margin::symmetric(18, 12))
            .stroke(egui::Stroke::new(1.0, BORDER))
            .corner_radius(10)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new("GLUJ-BENCH")
                                .size(21.0)
                                .strong()
                                .color(egui::Color32::WHITE),
                        );
                        ui.label(
                            egui::RichText::new("Hardware performance lab")
                                .size(11.0)
                                .color(MUTED),
                        );
                    });
                    ui.add_space(30.0);
                    nav_button(ui, &mut self.tab, Tab::Overview, "Overview");
                    nav_button(ui, &mut self.tab, Tab::Benchmarks, "Benchmarks");
                    nav_button(ui, &mut self.tab, Tab::Results, "Results");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let connected = self.worker.child.is_some();
                        status_pill(
                            ui,
                            if connected {
                                "WORKER ONLINE"
                            } else {
                                "WORKER OFFLINE"
                            },
                            if connected {
                                egui::Color32::from_rgb(93, 214, 139)
                            } else {
                                egui::Color32::from_rgb(239, 103, 119)
                            },
                        );
                    });
                });
            });
        ui.add_space(18.0);
    }
    fn overview(&self, ui: &mut egui::Ui) {
        section_heading(
            ui,
            "Hardware overview",
            "Detected compute devices and their exposed capabilities.",
        );
        ui.add_space(12.0);
        ui.columns(3, |columns| {
            summary_card(
                &mut columns[0],
                "DEVICES",
                self.devices.len().to_string(),
                ACCENT,
            );
            summary_card(
                &mut columns[1],
                "AVAILABLE",
                self.devices
                    .iter()
                    .filter(|device| device.available)
                    .count()
                    .to_string(),
                egui::Color32::from_rgb(93, 214, 139),
            );
            summary_card(
                &mut columns[2],
                "BENCHMARKS",
                self.benchmarks.len().to_string(),
                ACCENT_BLUE,
            );
        });
        ui.add_space(12.0);
        status_banner(ui, &self.status, self.worker.child.is_some());
        ui.add_space(12.0);
        egui::ScrollArea::vertical().show(ui, |ui| {
            for device in &self.devices {
                device_card(ui, device);
                ui.add_space(10.0);
            }
        });
    }
    fn benchmarks(&mut self, ui: &mut egui::Ui) {
        section_heading(
            ui,
            "Benchmark lab",
            "Choose a workload, then run it against the selected hardware.",
        );
        ui.add_space(12.0);
        egui::Frame::new()
            .fill(SURFACE)
            .inner_margin(10)
            .stroke(egui::Stroke::new(1.0, BORDER))
            .corner_radius(9)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    self.suite_button(ui, "cpu.bandwidth", "CPU · Bandwidth");
                    self.suite_button(ui, "cpu.performance", "CPU · Performance");
                    self.suite_button(ui, "gpu.bandwidth", "GPU · Bandwidth");
                    self.suite_button(ui, "gpu.performance", "GPU · Performance");
                });
            });
        ui.add_space(10.0);
        ui.label(egui::RichText::new(match self.selected_suite.as_str() {
            "cpu.bandwidth" => {
                "Read, write, and copy bandwidth with per-operation sample statistics."
            }
            "cpu.performance" => {
                "Aggregate logical-processor throughput, the matching single-thread INT64 test, and compute-vs-memory diagnosis."
            }
            "gpu.bandwidth" => {
                "Estimated effective L2/L3, GPU-local memory, and bidirectional host-device bandwidth. Cache names are inferred from measured plateaus."
            }
            "gpu.performance" => {
                "Vector shader and hardware cooperative-matrix throughput. Matrix entries are independently capability-gated; unavailable numeric formats remain visible with the exact reason."
            }
            _ => "Select a benchmark suite.",
        }).color(MUTED));
        if self.selected_suite.starts_with("gpu.") {
            let gpu_devices: Vec<_> = self
                .devices
                .iter()
                .filter(|device| device.category == DeviceCategory::Gpu && device.available)
                .map(|device| (device.id.clone(), device.name.clone()))
                .collect();
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("GPU ADAPTER")
                        .size(11.0)
                        .strong()
                        .color(MUTED),
                );
                let selected_text = self
                    .selected_gpu_id
                    .as_ref()
                    .and_then(|id| gpu_devices.iter().find(|device| &device.0 == id))
                    .map(|device| device.1.as_str())
                    .unwrap_or("No available GPU");
                egui::ComboBox::from_id_salt("gpu-adapter")
                    .selected_text(selected_text)
                    .show_ui(ui, |ui| {
                        for (id, name) in &gpu_devices {
                            if ui
                                .selectable_value(&mut self.selected_gpu_id, Some(id.clone()), name)
                                .changed()
                            {
                                self.selected = self
                                    .benchmarks
                                    .iter()
                                    .find(|item| {
                                        item.suite_id == self.selected_suite
                                            && item.available
                                            && item.supported_device_ids.contains(id)
                                    })
                                    .map(|item| item.id.clone());
                            }
                        }
                    });
            });
        }
        ui.add_space(12.0);
        egui::ScrollArea::vertical()
            .max_height(360.0)
            .show(ui, |ui| {
                for benchmark in self
                    .benchmarks
                    .iter()
                    .filter(|item| item.suite_id == self.selected_suite)
                {
                    let selected = self.selected.as_deref() == Some(&benchmark.id);
                    let available = self.benchmark_available(benchmark);
                    let fill = if selected {
                        egui::Color32::from_rgb(27, 55, 60)
                    } else {
                        SURFACE
                    };
                    egui::Frame::new()
                        .fill(fill)
                        .inner_margin(egui::Margin::symmetric(14, 11))
                        .stroke(egui::Stroke::new(
                            1.0,
                            if selected { ACCENT } else { BORDER },
                        ))
                        .corner_radius(8)
                        .show(ui, |ui| {
                            ui.add_enabled_ui(available && self.active_request.is_none(), |ui| {
                                let response = ui.selectable_label(
                                    selected,
                                    egui::RichText::new(&benchmark.name).strong().size(14.0),
                                );
                                if response.clicked() {
                                    self.selected = Some(benchmark.id.clone());
                                }
                            });
                            ui.label(
                                egui::RichText::new(&benchmark.workload)
                                    .size(11.0)
                                    .color(MUTED),
                            );
                            if !available {
                                ui.colored_label(
                                    egui::Color32::from_rgb(245, 188, 89),
                                    format!(
                                        "Unavailable · {}",
                                        if benchmark.available {
                                            "unsupported_on_selected_adapter"
                                        } else {
                                            &benchmark.unavailable_reason
                                        }
                                    ),
                                );
                            }
                        });
                    ui.add_space(8.0);
                }
            });
        ui.add_space(8.0);
        if self.active_request.is_some() {
            egui::Frame::new()
                .fill(SURFACE_RAISED)
                .inner_margin(12)
                .corner_radius(8)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.strong(&self.progress_message);
                    });
                    ui.add(
                        egui::ProgressBar::new(self.progress)
                            .show_percentage()
                            .fill(ACCENT),
                    );
                    if ui.button("Cancel benchmark").clicked() {
                        self.cancel();
                    }
                });
        } else {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(self.selected.is_some(), primary_button("Run selected"))
                    .clicked()
                {
                    self.run_selected();
                }
                if ui.add(egui::Button::new("Run all available")).clicked() {
                    self.run_all();
                }
                ui.label(
                    egui::RichText::new("5 samples · ~5 seconds per operation")
                        .size(11.0)
                        .color(MUTED),
                );
            });
        }
    }

    fn suite_button(&mut self, ui: &mut egui::Ui, suite: &str, label: &str) {
        let selected = self.selected_suite == suite;
        let button = egui::Button::new(egui::RichText::new(label).strong())
            .selected(selected)
            .fill(if selected {
                egui::Color32::from_rgb(29, 75, 76)
            } else {
                SURFACE_RAISED
            })
            .stroke(egui::Stroke::new(
                1.0,
                if selected { ACCENT } else { BORDER },
            ));
        if ui.add(button).clicked() {
            self.selected_suite = suite.into();
            self.selected = self
                .benchmarks
                .iter()
                .find(|item| item.suite_id == self.selected_suite && self.benchmark_available(item))
                .map(|item| item.id.clone());
        }
    }

    fn benchmark_available(&self, benchmark: &BenchmarkDescriptor) -> bool {
        if !benchmark.available {
            return false;
        }
        if benchmark.category != gluj_bench_core::BenchmarkCategory::Gpu {
            return true;
        }
        self.selected_gpu_id
            .as_ref()
            .is_some_and(|id| benchmark.supported_device_ids.contains(id))
    }
    fn results(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            section_heading(
                ui,
                "Results",
                "Measured performance with five-sample statistics.",
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if !self.results.is_empty() && ui.button("Clear results").clicked() {
                    self.results.clear();
                }
            });
        });
        ui.add_space(12.0);
        if self.results.is_empty() {
            egui::Frame::new()
                .fill(SURFACE)
                .inner_margin(30)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(10)
                .show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new("No results yet").size(18.0).strong());
                        ui.label(
                            egui::RichText::new("Run a benchmark to populate this view.")
                                .color(MUTED),
                        );
                    });
                });
            return;
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (result_index, result) in self.results.iter().rev().enumerate() {
                egui::Frame::new()
                    .fill(SURFACE)
                    .inner_margin(16)
                    .stroke(egui::Stroke::new(1.0, BORDER))
                    .corner_radius(10)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(&result.benchmark_id)
                                    .size(16.0)
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    status_pill(
                                        ui,
                                        &format!("{:.2} ms", result.elapsed_ns as f64 / 1e6),
                                        ACCENT_BLUE,
                                    );
                                },
                            );
                        });
                        ui.add_space(8.0);
                        egui::Grid::new(format!("result-{}-{}", result.benchmark_id, result_index))
                            .striped(true)
                            .min_col_width(120.0)
                            .show(ui, |ui| {
                                table_header(ui, "OPERATION");
                                table_header(ui, "MEDIAN");
                                table_header(ui, "MIN / MAX");
                                table_header(ui, "STD DEV");
                                ui.end_row();
                                for metric in &result.metrics {
                                    let s = &metric.statistics;
                                    ui.label(egui::RichText::new(&metric.name).strong());
                                    let median = ui.label(
                                        egui::RichText::new(format_metric(
                                            metric.value,
                                            &metric.unit,
                                        ))
                                        .strong()
                                        .color(ACCENT),
                                    );
                                    if metric.unit == "operations/s" {
                                        median.on_hover_text(format!(
                                            "{:.4} TOPS; raw value: {:.0} operations/s",
                                            metric.value / 1e12,
                                            metric.value
                                        ));
                                    }
                                    ui.label(format!(
                                        "{} / {}",
                                        format_metric(s.minimum, &metric.unit),
                                        format_metric(s.maximum, &metric.unit)
                                    ));
                                    ui.label(format_metric(s.standard_deviation, &metric.unit));
                                    ui.end_row();
                                }
                            });
                        show_bound_diagnosis(ui, &result.workload_metadata);
                        egui::CollapsingHeader::new("Technical details")
                            .id_salt(format!(
                                "technical-details-{}-{}",
                                result.benchmark_id, result_index
                            ))
                            .show(ui, |ui| {
                                egui::Grid::new(format!(
                                    "metadata-{}-{}",
                                    result.benchmark_id, result_index
                                ))
                                .striped(true)
                                .show(ui, |ui| {
                                    for (key, value) in &result.workload_metadata {
                                        if is_diagnosis_key(key) {
                                            continue;
                                        }
                                        ui.label(egui::RichText::new(key).color(MUTED));
                                        ui.monospace(value);
                                        ui.end_row();
                                    }
                                });
                            });
                    });
                ui.add_space(10.0);
            }
        });
    }
}

impl eframe::App for GlujBenchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.receive_worker_events();
        egui::Frame::new()
            .inner_margin(egui::Margin::symmetric(18, 12))
            .show(ui, |ui| {
                self.tabs(ui);
                match self.tab {
                    Tab::Overview => self.overview(ui),
                    Tab::Benchmarks => self.benchmarks(ui),
                    Tab::Results => self.results(ui),
                }
            });
        ui.ctx().request_repaint_after(Duration::from_millis(100));
    }
}

fn nav_button(ui: &mut egui::Ui, current: &mut Tab, tab: Tab, label: &str) {
    let selected = *current == tab;
    let text = egui::RichText::new(label)
        .strong()
        .color(if selected { ACCENT } else { MUTED });
    if ui
        .add(
            egui::Button::new(text)
                .selected(selected)
                .fill(if selected {
                    egui::Color32::from_rgb(26, 57, 61)
                } else {
                    SURFACE
                })
                .stroke(egui::Stroke::new(
                    1.0,
                    if selected {
                        ACCENT
                    } else {
                        egui::Color32::TRANSPARENT
                    },
                )),
        )
        .clicked()
    {
        *current = tab;
    }
}

fn section_heading(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    ui.vertical(|ui| {
        ui.label(
            egui::RichText::new(title)
                .size(24.0)
                .strong()
                .color(egui::Color32::WHITE),
        );
        ui.label(egui::RichText::new(subtitle).size(12.0).color(MUTED));
    });
}

fn status_pill(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    egui::Frame::new()
        .fill(color.gamma_multiply(0.12))
        .stroke(egui::Stroke::new(1.0, color.gamma_multiply(0.65)))
        .corner_radius(20)
        .inner_margin(egui::Margin::symmetric(10, 5))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).size(10.0).strong().color(color));
        });
}

fn summary_card(ui: &mut egui::Ui, label: &str, value: String, color: egui::Color32) {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .corner_radius(9)
        .inner_margin(14)
        .show(ui, |ui| {
            ui.set_min_height(58.0);
            ui.label(egui::RichText::new(label).size(10.0).strong().color(MUTED));
            ui.label(egui::RichText::new(value).size(25.0).strong().color(color));
        });
}

fn status_banner(ui: &mut egui::Ui, status: &str, connected: bool) {
    let color = if connected {
        egui::Color32::from_rgb(93, 214, 139)
    } else {
        egui::Color32::from_rgb(239, 103, 119)
    };
    egui::Frame::new()
        .fill(color.gamma_multiply(0.08))
        .stroke(egui::Stroke::new(1.0, color.gamma_multiply(0.45)))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(12, 9))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.colored_label(color, "●");
                ui.label(status);
            });
        });
}

fn device_card(ui: &mut egui::Ui, device: &DeviceDescriptor) {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .corner_radius(10)
        .inner_margin(16)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(&device.name)
                            .size(16.0)
                            .strong()
                            .color(egui::Color32::WHITE),
                    );
                    ui.label(
                        egui::RichText::new(&device.id)
                            .size(10.0)
                            .monospace()
                            .color(MUTED),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    status_pill(
                        ui,
                        if device.available {
                            "AVAILABLE"
                        } else {
                            "UNAVAILABLE"
                        },
                        if device.available {
                            egui::Color32::from_rgb(93, 214, 139)
                        } else {
                            egui::Color32::from_rgb(245, 188, 89)
                        },
                    );
                    status_pill(
                        ui,
                        match device.category {
                            DeviceCategory::Cpu => "CPU",
                            DeviceCategory::Memory => "MEMORY",
                            DeviceCategory::Gpu => "GPU",
                        },
                        ACCENT_BLUE,
                    );
                });
            });
            ui.add_space(6.0);
            ui.label(egui::RichText::new(&device.status).color(MUTED));
            if !device.caches.is_empty() {
                ui.add_space(7.0);
                ui.horizontal_wrapped(|ui| {
                    for cache in &device.caches {
                        status_pill(
                            ui,
                            &format!(
                                "L{} {:?} · {} KiB",
                                cache.level,
                                cache.kind,
                                cache.size_bytes / 1024
                            ),
                            ACCENT,
                        );
                    }
                });
            }
            if !device.properties.is_empty() {
                egui::CollapsingHeader::new("Capabilities and topology")
                    .id_salt(format!("device-capabilities-{}", device.id))
                    .show(ui, |ui| {
                        egui::Grid::new(format!("device-properties-{}", device.id))
                            .striped(true)
                            .show(ui, |ui| {
                                for (key, value) in &device.properties {
                                    ui.label(egui::RichText::new(key).color(MUTED));
                                    ui.monospace(value);
                                    ui.end_row();
                                }
                            });
                    });
            }
        });
}

fn primary_button(label: &'static str) -> egui::Button<'static> {
    egui::Button::new(
        egui::RichText::new(label)
            .strong()
            .color(egui::Color32::from_rgb(7, 24, 25)),
    )
    .fill(ACCENT)
    .stroke(egui::Stroke::new(1.0, ACCENT))
}

fn table_header(ui: &mut egui::Ui, label: &str) {
    ui.label(egui::RichText::new(label).size(10.0).strong().color(MUTED));
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

fn format_metric(value: f64, unit: &str) -> String {
    if unit == "bytes/s" {
        return format!("{:.2} GB/s", value / 1e9);
    }
    if unit == "operations/s" {
        let (scale, suffix) = if value >= 1e12 {
            (1e12, "TOPS")
        } else if value >= 1e6 {
            (1e6, "MOPS")
        } else if value >= 1e3 {
            (1e3, "KOPS")
        } else {
            (1.0, "OPS")
        };
        return format!("{} {suffix}", format_grouped_2(value / scale));
    }
    if unit == "MB/s" {
        return format!("{} MB/s", format_grouped_2(value));
    }
    if unit == "strings/s" {
        let (scale, suffix) = if value >= 1e6 {
            (1e6, "M strings/s")
        } else if value >= 1e3 {
            (1e3, "K strings/s")
        } else {
            (1.0, "strings/s")
        };
        return format!("{} {suffix}", format_grouped_2(value / scale));
    }
    if unit == "primes/s" {
        let (scale, suffix) = if value >= 1e6 {
            (1e6, "M primes/s")
        } else if value >= 1e3 {
            (1e3, "K primes/s")
        } else {
            (1.0, "primes/s")
        };
        return format!("{} {suffix}", format_grouped_2(value / scale));
    }
    format!("{value:.2} {unit}")
}

fn format_grouped_2(value: f64) -> String {
    let raw = format!("{:.2}", value.abs());
    let (integer, fraction) = raw.split_once('.').unwrap_or((&raw, "00"));
    let mut grouped = String::with_capacity(raw.len() + integer.len() / 3);
    if value.is_sign_negative() {
        grouped.push('-');
    }
    for (index, character) in integer.chars().enumerate() {
        if index != 0 && (integer.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(character);
    }
    grouped.push('.');
    grouped.push_str(fraction);
    grouped
}

fn is_diagnosis_key(key: &str) -> bool {
    matches!(
        key,
        "bound_classification"
            | "cpu_runnable_percent"
            | "large_data_slowdown_percent"
            | "memory_wait_pressure_percent"
            | "memory_sensitivity_ratio"
            | "classification_method"
            | "classification_is_inference"
            | "cpu_activity_interpretation"
            | "diagnostic_domain"
            | "implied_output_bandwidth_bytes_per_second"
    )
}

fn show_bound_diagnosis(ui: &mut egui::Ui, metadata: &std::collections::BTreeMap<String, String>) {
    let Some(classification) = metadata.get("bound_classification") else {
        return;
    };
    ui.add_space(6.0);
    let (label, color) = if classification == "memory_bandwidth_bound" {
        ("MEMORY BANDWIDTH BOUND", egui::Color32::YELLOW)
    } else {
        ("COMPUTE BOUND", egui::Color32::LIGHT_GREEN)
    };
    ui.horizontal(|ui| {
        ui.strong("Bound diagnosis:");
        ui.colored_label(color, label);
    });
    if metadata.get("execution_domain").map(String::as_str) == Some("cooperative_matrix") {
        ui.small(
            metadata
                .get("classification_method")
                .map(String::as_str)
                .unwrap_or("Register-resident cooperative-matrix workload."),
        );
        ui.small("Classification is inferred from the deliberately high arithmetic intensity; it is not a hardware stall-counter reading.");
        return;
    }
    if metadata.get("diagnostic_domain").map(String::as_str) == Some("gpu") {
        let sensitivity = metadata
            .get("memory_sensitivity_ratio")
            .and_then(|value| value.parse::<f64>().ok())
            .map(|value| format!("{:.2}%", value * 100.0))
            .unwrap_or_else(|| "unknown".into());
        let slowdown = metadata
            .get("large_data_slowdown_percent")
            .map(String::as_str)
            .unwrap_or("unknown");
        let output_bandwidth = metadata
            .get("implied_output_bandwidth_bytes_per_second")
            .and_then(|value| value.parse::<f64>().ok())
            .map(|value| format_metric(value, "bytes/s"))
            .unwrap_or_else(|| "unknown".into());
        ui.small(format!(
            "Arithmetic-intensity sensitivity: {sensitivity}  |  Low-intensity slowdown: {slowdown}%  |  Implied output traffic: {output_bandwidth}"
        ));
        ui.small("Classification compares normalized throughput at two arithmetic intensities. It is an inference, not a hardware stall-counter reading.");
        return;
    }
    let runnable = metadata
        .get("cpu_runnable_percent")
        .map(String::as_str)
        .unwrap_or("unknown");
    let pressure = metadata
        .get("large_data_slowdown_percent")
        .or_else(|| metadata.get("memory_wait_pressure_percent"))
        .map(String::as_str)
        .unwrap_or("unknown");
    let sensitivity = metadata
        .get("memory_sensitivity_ratio")
        .and_then(|value| value.parse::<f64>().ok())
        .map(|value| format!("{:.2}%", value * 100.0))
        .unwrap_or_else(|| "unknown".into());
    ui.small(format!(
        "CPU runnable: {runnable}%  |  Large-data slowdown: {pressure}%  |  Large/cache throughput: {sensitivity}"
    ));
    ui.small("Classification is inferred from working-set sensitivity. Slowdown shows memory influence, not necessarily bandwidth saturation; CPU runnable time is not a hardware stall counter.");
}

#[cfg(test)]
mod tests {
    use super::{format_grouped_2, format_metric};

    #[test]
    fn operation_rates_use_readable_si_prefixes() {
        assert_eq!(format_metric(12_500.0, "operations/s"), "12.50 KOPS");
        assert_eq!(
            format_metric(129_718_800_000.0, "operations/s"),
            "129,718.80 MOPS"
        );
        assert_eq!(
            format_metric(3_200_000_000_000.0, "operations/s"),
            "3.20 TOPS"
        );
        assert_eq!(
            format_metric(1_800_000_000_000.0, "operations/s"),
            "1.80 TOPS"
        );
        assert_eq!(format_metric(3_744.186_882_15, "MB/s"), "3,744.19 MB/s");
        assert_eq!(
            format_metric(1_141_432_000.0, "strings/s"),
            "1,141.43 M strings/s"
        );
        assert_eq!(format_metric(12_345_678.0, "primes/s"), "12.35 M primes/s");
    }

    #[test]
    fn grouped_numbers_keep_the_decimal_in_the_correct_place() {
        assert_eq!(format_grouped_2(129_718.8), "129,718.80");
        assert_eq!(format_grouped_2(-1_234.5), "-1,234.50");
    }
}
