//! Result reading, explicit history curation and plan inspection.
use super::*;
use gluj_bench_core::{
    CompletionStatus, RunAnnotation, RunIndex, RunIndexEntry, RunRecord, RunSource, RunStore,
};
use std::collections::BTreeSet;
const PAGE_SIZE: usize = 30;
#[derive(Default)]
pub(super) struct HistoryState {
    open: bool,
    entries: Vec<RunIndexEntry>,
    annotations: BTreeMap<String, RunAnnotation>,
    selected: Option<String>,
    record: Option<RunRecord>,
    pub(super) viewed_run: Option<String>,
    checked: BTreeSet<String>,
    query: String,
    page: usize,
    pending_delete: Option<String>,
    status: String,
    import_error: String,
    observations: presentation::observations::Selection,
    status_filter: usize,
    io: Option<Receiver<HistoryEvent>>,
}
enum HistoryEvent {
    Loaded(String, Result<Box<RunRecord>, String>),
    Finished(Result<String, String>),
    Imported(Result<String, String>),
}
fn write_export(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    let temporary = path.with_extension(format!(
        "{}.{}.tmp",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let outcome = (|| {
        let mut f = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&temporary, path)
    })()
    .map_err(|e: std::io::Error| e.to_string());
    let _ = fs::remove_file(temporary);
    outcome
}
pub(super) fn timestamp(ms: u64) -> String {
    let seconds = ms / 1000;
    let z = (seconds / 86400) as i64 + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02} UTC",
        seconds % 86400 / 3600,
        seconds % 3600 / 60
    )
}
#[cfg(test)]
fn label(entry: &RunIndexEntry, annotations: &BTreeMap<String, RunAnnotation>) -> String {
    let a = annotations.get(&entry.run_id).cloned().unwrap_or_default();
    format!(
        "{}{}{} | {} | {:?} | {:?}",
        if a.pinned { "[Pinned] " } else { "" },
        if a.archived { "[Archived] " } else { "" },
        if a.name.is_empty() {
            entry.benchmark_id.as_str()
        } else {
            a.name.as_str()
        },
        entry
            .started_unix_ms
            .map(timestamp)
            .unwrap_or_else(|| "Time not recorded".into()),
        entry.status,
        entry.source
    )
}
fn history_title(benchmark_id: &str, benchmarks: &[BenchmarkDescriptor]) -> String {
    benchmarks
        .iter()
        .find(|benchmark| benchmark.id == benchmark_id)
        .map(families::mode_name)
        .unwrap_or_else(|| benchmark_id.to_owned())
}

impl HistoryState {
    fn visible(&self, benchmarks: &[BenchmarkDescriptor]) -> Vec<&RunIndexEntry> {
        let query = self.query.trim().to_lowercase();
        self.entries
            .iter()
            .rev()
            .filter(|e| {
                let a = self.annotations.get(&e.run_id).cloned().unwrap_or_default();
                (a.archived == (self.status_filter == 4))
                    && match self.status_filter {
                        1 => e.status == CompletionStatus::Completed,
                        2 => matches!(
                            e.status,
                            CompletionStatus::Cancelled | CompletionStatus::Failed
                        ),
                        3 => e.source == RunSource::ImportedComparison,
                        _ => true,
                    }
                    && (query.is_empty()
                        || [&e.benchmark_id, &e.run_id, &a.name, &a.notes]
                            .iter()
                            .any(|value| value.to_lowercase().contains(&query))
                        || history_title(&e.benchmark_id, benchmarks)
                            .to_lowercase()
                            .contains(&query))
            })
            .collect()
    }
    fn page_entries(&self, benchmarks: &[BenchmarkDescriptor]) -> Vec<&RunIndexEntry> {
        self.visible(benchmarks)
            .into_iter()
            .skip(self.page * PAGE_SIZE)
            .take(PAGE_SIZE)
            .collect()
    }

    fn clear_hidden_selection(&mut self, benchmarks: &[BenchmarkDescriptor]) {
        if let Some(id) = &self.selected
            && !self
                .page_entries(benchmarks)
                .iter()
                .any(|entry| &entry.run_id == id)
        {
            self.selected = None;
            self.record = None;
            self.pending_delete = None;
        }
    }
}
pub(super) fn sized_metrics(result: &BenchmarkResult) -> Vec<(u64, &Metric)> {
    if scaling_result(result) {
        let mut values = scaling_compute_tiers(result);
        values.reverse();
        return values;
    }
    let mut values: Vec<_> = result
        .metrics
        .iter()
        .filter_map(|m| {
            let (size, operation) = m.name.strip_prefix("working_set_")?.split_once('.')?;
            if !matches!(
                operation,
                "read" | "write" | "copy" | "host_to_device" | "device_to_host"
            ) {
                return None;
            }
            Some((size.parse::<u64>().ok()?, m))
        })
        .collect();
    let operation_order = |m: &Metric| match m.name.rsplit('.').next().unwrap_or("") {
        "read" | "host_to_device" => 0,
        "write" | "device_to_host" => 1,
        _ => 2,
    };
    values.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(operation_order(a.1).cmp(&operation_order(b.1)))
    });
    values
}
pub(super) fn typical(group: &[&BenchmarkResult], operation: Option<&str>) -> String {
    let Some(latest) = group.first() else {
        return "N/A".into();
    };
    let reference = match operation {
        Some(name) => latest.metrics.iter().find(|m| m.name == name),
        None => featured_metric(latest),
    };
    let Some(reference) = reference else {
        return "N/A".into();
    };
    let mut values: Vec<_> = group
        .iter()
        .filter(|r| same_run_configuration(r, latest))
        .filter_map(|r| {
            r.metrics
                .iter()
                .find(|m| m.name == reference.name && m.unit == reference.unit)
        })
        .filter(|m| m.value.is_finite())
        .map(|m| m.value)
        .collect();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return "N/A".into();
    }
    let middle = values.len() / 2;
    let value = if values.len() % 2 == 0 {
        values[middle - 1] / 2. + values[middle] / 2.
    } else {
        values[middle]
    };
    format_metric(value, &reference.unit)
}
fn measured_text(metric: &Metric) -> String {
    let s = &metric.statistics;
    if s.sample_count == 0 {
        return format!(
            "{} | sample coverage not recorded",
            format_metric(metric.value, &metric.unit)
        );
    }
    format!(
        "Median {} | observed {} - {} | {} observations",
        format_metric(s.median, &metric.unit),
        format_metric(s.minimum, &metric.unit),
        format_metric(s.maximum, &metric.unit),
        s.sample_count
    )
}
impl App {
    pub(super) fn history_busy(&self) -> bool {
        self.history.io.is_some()
    }
    pub(super) fn reload_history(&mut self) {
        let Some(path) = &self.native_history_path else {
            return;
        };
        let store = RunStore::new(path);
        let outcome = (|| {
            let index = match fs::read(path.join("index.json")) {
                Ok(bytes) => {
                    serde_json::from_slice::<RunIndex>(&bytes).map_err(|e| e.to_string())?
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    store.rebuild_index().map_err(|e| e.to_string())?
                }
                Err(e) => return Err(e.to_string()),
            };
            if index.version != gluj_bench_core::RUN_FORMAT_VERSION {
                return Err("Unsupported native index version. Files preserved.".into());
            }
            let annotations = store.annotations().map_err(|e| e.to_string())?;
            Ok((index.runs, annotations))
        })();
        match outcome {
            Ok((entries, annotations)) => {
                self.history.entries = entries;
                self.history.annotations = annotations;
                self.history.page = self.history.page.min(
                    self.history
                        .visible(&self.benchmarks)
                        .len()
                        .saturating_sub(1)
                        / PAGE_SIZE,
                );
                self.history.clear_hidden_selection(&self.benchmarks);
            }
            Err(e) => {
                self.history.status =
                    format!("Could not read history: {e}. Existing files preserved.")
            }
        }
    }
    pub(super) fn restore_pinned_baselines(&mut self) {
        let ids: Vec<_> = self
            .history
            .annotations
            .iter()
            .filter(|(_, a)| a.pinned)
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            self.use_history_baseline(&id);
        }
    }
    fn history_io(&mut self, job: impl FnOnce() -> HistoryEvent + Send + 'static) {
        if self.history.io.is_some() {
            self.history.status = "Wait for the current history operation.".into();
            return;
        }
        let (sender, receiver) = mpsc::channel();
        match thread::Builder::new()
            .name("history-io".into())
            .spawn(move || {
                let _ = sender.send(job());
            }) {
            Ok(_) => {
                self.history.io = Some(receiver);
                self.history.status = "Loading / saving history…".into();
            }
            Err(e) => self.history.status = e.to_string(),
        }
        self.ui_dirty = true;
    }
    pub(super) fn receive_history_events(&mut self) {
        let event = self.history.io.as_ref().map(|events| events.try_recv());
        match event {
            Some(Ok(event)) => {
                self.history.io = None;
                match event {
                    HistoryEvent::Loaded(id, record) => {
                        if self.history.selected.as_deref() == Some(&id) {
                            match record {
                                Ok(record) => {
                                    self.history.record = Some(*record);
                                    self.history.status.clear();
                                }
                                Err(e) => {
                                    self.history.status =
                                        format!("Could not load record: {e}. File preserved.")
                                }
                            }
                        } else {
                            self.history.status.clear();
                        }
                    }
                    HistoryEvent::Finished(outcome) => {
                        self.reload_history();
                        self.history.status = outcome.unwrap_or_else(|e| {
                            format!("History operation failed: {e}. Source files preserved.")
                        });
                    }
                    HistoryEvent::Imported(outcome) => {
                        self.reload_history();
                        match outcome {
                            Ok(message) => self.history.status = message,
                            Err(error) => {
                                self.history.status.clear();
                                self.history.import_error = error;
                            }
                        }
                    }
                }
                self.ui_dirty = true;
            }
            Some(Err(mpsc::TryRecvError::Disconnected)) => {
                self.history.io = None;
                self.history.status =
                    "History operation stopped without a result. Files retained.".into();
                self.ui_dirty = true;
            }
            _ => {}
        }
    }
    pub(super) fn restore_plan_status(&mut self) {
        let Some(root) = &self.native_history_path else {
            return;
        };
        let path = root.join("latest-plan.json");
        if !path.exists() {
            return;
        }
        match fs::read(path).map_err(|e| e.to_string()).and_then(|bytes| {
            serde_json::from_slice::<gluj_bench_core::SuitePlan>(&bytes).map_err(|e| e.to_string())
        }) {
            Ok(mut plan) => {
                if let Err(e) = plan.validate() {
                    self.status = format!("Could not restore previous plan: {e}. File preserved.");
                    return;
                }
                let mut interrupted = false;
                for c in &mut plan.conditions {
                    if c.status == CompletionStatus::Running {
                        c.status = CompletionStatus::Cancelled;
                        interrupted = true;
                    } else if c.status == CompletionStatus::Pending {
                        c.status = CompletionStatus::Skipped;
                        interrupted = true;
                    }
                }
                self.active_plan = Some(plan);
                if interrupted {
                    self.status="Previous plan was interrupted. Its completed/partial status is preserved; no work was resumed automatically.".into();
                }
            }
            Err(e) => {
                self.status = format!("Could not restore previous plan: {e}. File preserved.")
            }
        }
    }
    fn use_history_baseline(&mut self, id: &str) {
        let Some(path) = &self.native_history_path else {
            return;
        };
        match RunStore::new(path).load(id) {
            Ok(record) => {
                if let Some(mut result) = record.result {
                    let a = self
                        .history
                        .annotations
                        .get(id)
                        .cloned()
                        .unwrap_or_default();
                    result
                        .device_metadata
                        .insert("native_run_id".into(), id.into());
                    result
                        .device_metadata
                        .insert("saved_setup".into(), format!("baseline-{id}"));
                    result.device_metadata.insert(
                        "saved_baseline_name".into(),
                        if a.name.is_empty() {
                            format!("Saved run {id}")
                        } else {
                            a.name
                        },
                    );
                    if record.source == RunSource::ImportedComparison {
                        let component = component_label(&result, &self.devices);
                        result.device_metadata.insert(
                            "saved_component_name".into(),
                            format!("Imported | {component}"),
                        );
                    }
                    normalize_results(std::slice::from_mut(&mut result));
                    self.baseline_results.retain(|old| {
                        old.device_metadata
                            .get("native_run_id")
                            .is_none_or(|v| v != id)
                    });
                    self.baseline_results.push(result);
                    self.comparison_selection = 0;
                    self.history.status =
                        "Saved run added to Compare with. Pin it to restore it at startup.".into();
                } else {
                    self.history.status="This run has no completed score; its partial raw samples remain available in native export.".into();
                }
            }
            Err(e) => self.history.status = e.to_string(),
        }
    }
    fn select_history(&mut self, id: &str) {
        if self.history.io.is_some()
            || !self
                .history
                .page_entries(&self.benchmarks)
                .iter()
                .any(|entry| entry.run_id == id)
        {
            return;
        }
        self.history.pending_delete = None;
        if self
            .history
            .record
            .as_ref()
            .is_some_and(|record| record.run_id == id)
        {
            return;
        }
        self.history.record = None;
        self.history.selected = Some(id.into());
        if let Some(path) = &self.native_history_path {
            let root = path.clone();
            self.history.observations = Default::default();
            let id = id.to_owned();
            self.history_io(move || {
                let record = RunStore::new(root)
                    .load(&id)
                    .map(Box::new)
                    .map_err(|e| e.to_string());
                HistoryEvent::Loaded(id, record)
            });
        }
    }
    fn history_action(&mut self, action: &str, name: String, notes: String) {
        if self.history.io.is_some() && action != "close" {
            self.history.status = "Wait for the current history operation.".into();
            self.ui_dirty = true;
            return;
        }
        match action {
            "open" => {
                self.history.open = true;
                self.reload_history();
            }
            "close" => {
                self.history.open = false;
                self.history.pending_delete = None;
            }
            "refresh" => self.reload_history(),
            "next" => {
                if (self.history.page + 1) * PAGE_SIZE
                    < self.history.visible(&self.benchmarks).len()
                {
                    self.history.page += 1;
                }
            }
            "previous" => self.history.page = self.history.page.saturating_sub(1),
            "baseline" => {
                if let Some(id) = self.history.selected.clone() {
                    self.use_history_baseline(&id);
                }
            }
            "save" | "pin" | "archive" => {
                if let (Some(id), Some(path)) = (
                    self.history.selected.clone(),
                    self.native_history_path.clone(),
                ) {
                    let mut a = self
                        .history
                        .annotations
                        .get(&id)
                        .cloned()
                        .unwrap_or_default();
                    if action == "save" {
                        a.name = name;
                        a.notes = notes;
                    } else if action == "pin" {
                        a.pinned = !a.pinned;
                    } else {
                        a.archived = !a.archived;
                    }
                    let message = match (action, a.pinned, a.archived) {
                        ("pin", true, _) => "Baseline pinned.",
                        ("pin", false, _) => "Baseline unpinned.",
                        ("archive", _, true) => "Run archived.",
                        ("archive", _, false) => "Run restored.",
                        _ => "Name and notes saved.",
                    };
                    match RunStore::new(path).annotate(&id, a) {
                        Ok(()) => {
                            self.reload_history();
                            self.history.status = message.into();
                        }
                        Err(e) => self.history.status = format!("Could not save annotation: {e}"),
                    }
                }
            }
            "delete" => {
                self.history.pending_delete = self.history.selected.clone();
                self.history.status.clear();
            }
            "keep" => self.history.pending_delete = None,
            "confirm-delete" => {
                if let (Some(id), Some(path)) = (
                    self.history.pending_delete.take(),
                    self.native_history_path.clone(),
                ) {
                    match RunStore::new(path).delete_run(&id) {
                        Ok(()) => {
                            self.history.checked.remove(&id);
                            self.history.record = None;
                            self.history.selected = None;
                            self.reload_history();
                            self.history.status = "Native record deleted.".into();
                        }
                        Err(e) => self.history.status = format!("Could not delete record: {e}"),
                    }
                }
            }
            "view" => {
                if let Some(record) = &self.history.record
                    && let Some(mut result) = record.result.clone()
                {
                    normalize_results(std::slice::from_mut(&mut result));
                    let existing = self.results.iter().position(|candidate| {
                        candidate.device_metadata.get("native_run_id") == Some(&record.run_id)
                            || (record.source == RunSource::LegacyLocal
                                && !candidate.device_metadata.contains_key("native_run_id")
                                && candidate == &result)
                    });
                    let destination = if let Some(index) = existing {
                        &mut self.results[index]
                    } else {
                        self.results.push(result);
                        self.results.last_mut().unwrap()
                    };
                    destination
                        .device_metadata
                        .insert("native_run_id".into(), record.run_id.clone());
                    if let Some(started) = record.started_unix_ms {
                        destination
                            .device_metadata
                            .entry("saved_at".into())
                            .or_insert_with(|| (started / 1000).to_string());
                    }
                    let component = result_component_key(destination);
                    let mut keys = Vec::new();
                    for result in &self.results {
                        let key = result_component_key(result);
                        if !keys.contains(&key) {
                            keys.push(key);
                        }
                    }
                    self.result_component =
                        keys.iter().position(|key| *key == component).unwrap_or(0);
                    self.result_selection = component_results(&self.results, &component)
                        .iter()
                        .position(|group| {
                            group.iter().any(|result| {
                                result.device_metadata.get("native_run_id") == Some(&record.run_id)
                            })
                        })
                        .unwrap_or(0);
                    self.history.viewed_run = Some(record.run_id.clone());
                    self.comparison_selection = 0;
                    self.result_details_expanded = false;
                    self.primary_selection = 0;
                    self.history.open = false;
                }
            }
            _ => {}
        }
        self.history.clear_hidden_selection(&self.benchmarks);
        self.ui_dirty = true;
    }
    pub(super) fn present_selected_measurement(
        &self,
        window: &MainWindow,
        result: &BenchmarkResult,
        row: &mut ResultRow,
    ) {
        let sizes = sized_metrics(result);
        let choices: Vec<SharedString> = sizes
            .iter()
            .map(|(size, m)| {
                format!(
                    "{} | {}",
                    format_binary_size(*size as f64),
                    display_metric_name(m.name.split('.').next_back().unwrap_or(&m.name))
                )
                .into()
            })
            .collect();
        window.set_primary_choices(ModelRc::new(VecModel::from(choices)));
        let index = self.primary_selection.min(sizes.len().saturating_sub(1));
        window.set_primary_selection(index as i32);
        let metric = sizes
            .get(index)
            .map(|(_, m)| *m)
            .or_else(|| featured_metric(result));
        window.set_memory_dataset_rows(ModelRc::new(VecModel::from(if scaling_result(result) {
            vec![]
        } else {
            presentation::datasets::rows(result)
        })));
        let (memory_chart, memory_points) =
            if scaling_result(result) || self.active_request.is_some() {
                Default::default()
            } else {
                presentation::datasets::curve(result)
            };
        window.set_memory_dataset_chart_visible(memory_chart.size().width > 0);
        window.set_memory_dataset_chart(memory_chart);
        window.set_memory_dataset_chart_points(ModelRc::new(VecModel::from(memory_points)));
        window.set_result_is_demand(is_offered_load(result));
        let bytes = sizes.get(index).map(|(size, _)| *size).unwrap_or(0);
        window.set_demand_facts(ModelRc::new(VecModel::from(
            presentation::datasets::demand(result, bytes),
        )));
        window.set_core_comparison(ModelRc::new(VecModel::from(
            metric
                .map(|m| presentation::datasets::core_counts(result, m, &self.results))
                .unwrap_or_default(),
        )));
        window.set_selected_metric_label(
            if metric.is_some_and(|m| m.statistics.sample_count > 0) {
                "Median"
            } else {
                "Reported value"
            }
            .into(),
        );
        window.set_selected_metric_facts(ModelRc::new(VecModel::from(
            metric
                .map(|m| {
                    vec![
                        presentation::fact(
                            "Observed range",
                            if m.statistics.sample_count > 0 {
                                format!(
                                    "{} – {}",
                                    format_metric(m.statistics.minimum, &m.unit),
                                    format_metric(m.statistics.maximum, &m.unit)
                                )
                            } else {
                                "Not recorded".into()
                            },
                        ),
                        presentation::fact(
                            "Observations",
                            if m.statistics.sample_count > 0 {
                                m.statistics.sample_count.to_string()
                            } else {
                                "Not recorded".into()
                            },
                        ),
                    ]
                })
                .unwrap_or_default(),
        )));
        if let Some(metric) = metric {
            row.primary_value = format_metric(metric.value, &metric.unit).into();
            row.summary = format!(
                "Measured: {}{}",
                sizes
                    .get(index)
                    .map(|(n, _)| format!("{} | ", format_binary_size(*n as f64)))
                    .unwrap_or_default(),
                measured_text(metric)
            )
            .into();
        }
        let validation = if result.workload_metadata.contains_key("result_validation") {
            "Runner checks recorded"
        } else {
            "Scope not recorded"
        };
        let mut technical = vec![presentation::fact(
            "Validation scope",
            result
                .workload_metadata
                .get("result_validation")
                .map(String::as_str)
                .unwrap_or("Not recorded"),
        )];
        technical.extend(
            result
                .workload_metadata
                .iter()
                .map(|(k, v)| presentation::technical_fact(&presentation::friendly(k), v)),
        );
        technical.extend(
            result
                .device_metadata
                .iter()
                .map(|(k, v)| presentation::technical_fact(&presentation::friendly(k), v)),
        );
        let query = self.presentation.technical_query.to_lowercase();
        technical.retain(|f| {
            format!("{} {}", f.label, f.value)
                .to_lowercase()
                .contains(&query)
        });
        window.set_result_technical(ModelRc::new(VecModel::from(technical)));
        let drift = result
            .workload_metadata
            .get("compute_reference_drift_percent")
            .map(|s| format!(" · Reference drift {s}%"))
            .unwrap_or_default();
        let calibration = if result
            .workload_metadata
            .get("gpu_saturation_boundary_peak")
            .is_some_and(|s| s == "true")
        {
            " · Calibration limit reached"
        } else if result
            .workload_metadata
            .get("gpu_saturation_peak_repeatable")
            .is_some_and(|s| s == "false")
        {
            " · Calibration spread recorded"
        } else {
            ""
        };
        window.set_measurement_context(
            format!("Validation: {validation}{drift}{calibration}").into(),
        );
        if let Some(metric) = metric {
            let key = self.selected_comparison_key(result);
            let note = if let Some((old, previous)) =
                comparison_metric(result, &metric.name, &self.baseline_results, key.as_deref())
            {
                let delta = (metric.value / previous.value - 1.) * 100.;
                let direction = if metric.unit == "ns" {
                    if delta < 0. {
                        "lower latency/cost"
                    } else {
                        "higher latency/cost"
                    }
                } else if delta >= 0. {
                    "higher throughput"
                } else {
                    "lower throughput"
                };
                if delta.abs() < 0.05 {
                    "No change vs saved baseline".into()
                } else {
                    format!(
                        "{delta:+.1}% {direction} · {} → {} · {}",
                        format_metric(previous.value, &previous.unit),
                        format_metric(metric.value, &metric.unit),
                        component_label(old, &self.devices)
                    )
                }
            } else {
                self.comparison_reason(result, &metric.name, key.as_deref())
            };
            let compatible =
                comparison_metric(result, &metric.name, &self.baseline_results, key.as_deref())
                    .is_some();
            let details = if compatible {
                vec![]
            } else {
                let old = self.baseline_results.iter().rev().find(|r| {
                    r.benchmark_id == result.benchmark_id
                        && key.as_ref().is_none_or(|k| result_component_key(r) == *k)
                });
                old.map(|old| {
                    let mut fields = old
                        .workload_metadata
                        .keys()
                        .chain(result.workload_metadata.keys())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .filter(|k| {
                            old.workload_metadata.get(*k) != result.workload_metadata.get(*k)
                        })
                        .map(|k| {
                            presentation::technical_fact(
                                &presentation::friendly(k),
                                &format!(
                                    "{} → {}",
                                    old.workload_metadata
                                        .get(k)
                                        .map(String::as_str)
                                        .unwrap_or("Not recorded"),
                                    result
                                        .workload_metadata
                                        .get(k)
                                        .map(String::as_str)
                                        .unwrap_or("Not recorded")
                                ),
                            )
                        })
                        .collect::<Vec<_>>();
                    if fields.is_empty() {
                        fields.push(presentation::fact("Comparison", &note));
                    }
                    fields
                })
                .unwrap_or_default()
            };
            let has_differences = !details.is_empty();
            window.set_comparison_facts(ModelRc::new(VecModel::from(details)));
            window.set_comparison_note(if compatible {
                note.into()
            } else if has_differences {
                "Not comparable · Inspect configuration differences".into()
            } else {
                "No matching baseline".into()
            });
        }
    }
    fn selected_comparison_key(&self, current: &BenchmarkResult) -> Option<String> {
        let mut keys = Vec::new();
        for r in self
            .baseline_results
            .iter()
            .rev()
            .filter(|r| component_family(r) == component_family(current))
        {
            let key = result_component_key(r);
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        self.comparison_selection
            .checked_sub(1)
            .and_then(|i| keys.get(i).cloned())
    }
    fn comparison_reason(
        &self,
        current: &BenchmarkResult,
        metric: &str,
        key: Option<&str>,
    ) -> String {
        let candidates: Vec<_> = self
            .baseline_results
            .iter()
            .filter(|r| key.is_none_or(|k| result_component_key(r) == k))
            .collect();
        if !candidates
            .iter()
            .any(|r| r.benchmark_id == current.benchmark_id)
        {
            return "No saved baseline for this workload in the selected source.".into();
        }
        if !candidates
            .iter()
            .any(|r| r.benchmark_id == current.benchmark_id && same_run_configuration(r, current))
        {
            let old = candidates
                .iter()
                .rev()
                .find(|r| r.benchmark_id == current.benchmark_id)
                .expect("workload checked");
            let mut differences = Vec::new();
            if result_load_signature(old) != result_load_signature(current) {
                differences.push("load, worker count or allocation budget differs".into());
            }
            let keys: BTreeSet<_> = old
                .workload_metadata
                .keys()
                .chain(current.workload_metadata.keys())
                .collect();
            for key in keys {
                let relevant = key.ends_with("revision")
                    || key.contains("preset")
                    || key.contains("dataset")
                    || key.starts_with("access_")
                    || key.starts_with("transfer_")
                    || key.starts_with("matrix_")
                    || matches!(
                        key.as_str(),
                        "memory_copy_path"
                            | "memory_operation"
                            | "memory_kernel_identity"
                            | "staging_buffers"
                            | "data_type"
                            | "thread_mode"
                            | "arithmetic_iterations"
                            | "sustained_seconds"
                    );
                if relevant && old.workload_metadata.get(key) != current.workload_metadata.get(key)
                {
                    differences.push(format!(
                        "{}: {} -> {}",
                        display_metric_name(key),
                        old.workload_metadata
                            .get(key)
                            .map(String::as_str)
                            .unwrap_or("not recorded"),
                        current
                            .workload_metadata
                            .get(key)
                            .map(String::as_str)
                            .unwrap_or("not recorded")
                    ));
                }
                if differences.len() >= 5 {
                    break;
                }
            }
            return format!(
                "Not comparable: {}.",
                if differences.is_empty() {
                    "recorded execution or workload context differs".into()
                } else {
                    differences.join("; ")
                }
            );
        }
        format!(
            "No compatible measured {metric} with matching units and recorded workload context in the selected baseline."
        )
    }
    pub(super) fn mark_pending_skipped(&mut self) {
        if let Some(plan) = &mut self.active_plan {
            for c in &mut plan.conditions {
                if c.status == CompletionStatus::Pending {
                    c.status = CompletionStatus::Skipped;
                }
            }
        }
    }
    pub(super) fn persist_plan_status(&mut self) {
        if let (Some(plan), Some(root)) = (&self.active_plan, &self.native_history_path) {
            let outcome = (|| {
                fs::create_dir_all(root)?;
                let bytes = serde_json::to_vec_pretty(plan)?;
                let path = root.join("latest-plan.json");
                let temporary = root.join("latest-plan.tmp");
                fs::write(&temporary, bytes)?;
                fs::rename(temporary, path)
            })()
            .map_err(|e: std::io::Error| e.to_string());
            if let Err(e) = outcome {
                self.status = format!("Plan status could not be saved: {e}");
            }
        }
    }
    fn queue_action(&mut self, index: usize) {
        let Some(plan) = &mut self.active_plan else {
            return;
        };
        let Some(condition) = plan.conditions.get_mut(index) else {
            return;
        };
        match condition.status {
            CompletionStatus::Pending => {
                let offset = plan.conditions[..index]
                    .iter()
                    .filter(|c| c.status == CompletionStatus::Pending)
                    .count();
                self.queue.remove(offset);
                plan.conditions[index].status = CompletionStatus::Skipped;
            }
            CompletionStatus::Running => {
                if let Some(request) = self.active_request.clone() {
                    self.continue_after_cancel = true;
                    let id = self.next_id("skip");
                    if let Err(e)=self.worker.send(json!({"protocol":PROTOCOL_VERSION,"id":id,"command":"cancel","arguments":{"request_id":request}})){self.continue_after_cancel=false;self.status=e;}
                }
            }
            _ => {}
        }
        self.persist_plan_status();
        self.ui_dirty = true;
    }
    fn run_task(&mut self, task: &str) {
        if self.active_request.is_some() {
            return;
        }
        let plan = match task {
            "compare" => self.prepare_plan(self.custom_plan.clone(), "configuration-comparison"),
            "cpu" => {
                let old = self.selected_suite.clone();
                self.selected_suite = "cpu.performance".into();
                let p = self.curated_plan(None);
                self.selected_suite = old;
                p
            }
            "memory" => {
                let old = self.selected_suite.clone();
                let gpu = old.starts_with("gpu.");
                self.selected_suite = if gpu {
                    "gpu.bandwidth"
                } else {
                    "cpu.bandwidth"
                }
                .into();
                let p = self.curated_plan(Some(if gpu {
                    "gpu.local_bandwidth"
                } else {
                    "cpu.memory_bandwidth"
                }));
                self.selected_suite = old;
                p
            }
            "large" => self
                .selected
                .as_ref()
                .ok_or_else(|| "Select a buffered workload first.".into())
                .and_then(|id| self.queued_run(id))
                .and_then(|mut run| {
                    if configurable_compute(&run.benchmark_id) {
                        run.settings.vector_preset = 6;
                        run.settings.vector_size = "256 MiB".into();
                        run.settings.vector_sweep = true;
                    } else if gluj_bench_core::gpu_memory(&run.benchmark_id)
                        || gluj_bench_core::gpu_transfer(&run.benchmark_id)
                        || run.benchmark_id == CPU_RAM_BANDWIDTH_ID
                    {
                        run.settings.dataset_mode = 1;
                        run.settings.dataset_mib = 256;
                    } else {
                        return Err(
                            "Choose a configurable buffered compute or memory mode for this task."
                                .into(),
                        );
                    }
                    self.prepare_plan(vec![run], "large-dataset")
                }),
            _ => return,
        };
        // Task buttons prepare reviewable conditions; user starts the custom plan.
        match plan {
            Ok(p) => {
                self.custom_plan = p.runs.into();
                self.presentation.scope = 1;
                self.status = format!("Custom list: {} tests.", self.custom_plan.len());
            }
            Err(e) => self.status = e,
        }
        self.ui_dirty = true;
    }
    pub(super) fn wire_phase_eight(window: &MainWindow, app: &Rc<RefCell<Self>>) {
        let weak = Rc::downgrade(app);
        let win = window.as_weak();
        window.on_open_result_observations(move || {
            if let (Some(a), Some(w)) = (weak.upgrade(), win.upgrade()) {
                let mut a = a.borrow_mut();
                let root = a.native_history_path.clone();
                a.result_observations.open(&w, root.as_deref());
            }
        });
        let weak = Rc::downgrade(app);
        let win = window.as_weak();
        window.on_result_observation_change(move |action, index| {
            if let (Some(a), Some(w)) = (weak.upgrade(), win.upgrade()) {
                a.borrow_mut()
                    .result_observations
                    .change(&w, action.as_str(), index);
            }
        });
        let weak = Rc::downgrade(app);
        window.on_history_action(move|action|{let Some(app)=weak.upgrade() else{return;};if app.borrow().active_request.is_some(){return;}
            if action=="export" {
                let (root,ids)={let a=app.borrow();let ids:Vec<String>=if a.history.checked.is_empty(){a.history.record.iter().map(|r| r.run_id.clone()).collect()}else{a.history.checked.iter().cloned().collect()};(a.native_history_path.clone(),ids)};
                if !ids.is_empty() && let Some(root)=root && let Some(path)=rfd::FileDialog::new().add_filter("Native Gluj-Bench records", &["json"]).set_file_name("Gluj-Bench-runs.json").save_file() {
                    app.borrow_mut().history_io(move||HistoryEvent::Finished((||{let bytes=RunStore::new(root).export_runs(&ids).map_err(|e|e.to_string())?;write_export(&path,&bytes)?;Ok(format!("Exported {} full native records; raw samples and provenance retained.",ids.len()))})()));
                }return;
            }
            if action=="import-native" {
                if let Some(path)=rfd::FileDialog::new().add_filter("Native Gluj-Bench records", &["json"]).pick_file(){app.borrow_mut().import_native(path);}return;
            }
            app.borrow_mut().history_action(action.as_str(),String::new(),String::new());
        });
        let weak = Rc::downgrade(app);
        window.on_save_history_annotation(move |name, notes| {
            if let Some(a) = weak.upgrade() {
                a.borrow_mut()
                    .history_action("save", name.to_string(), notes.to_string());
            }
        });
        let weak = Rc::downgrade(app);
        window.on_select_history(move |id| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                a.select_history(id.as_str());
                a.ui_dirty = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_check_history(move |id, checked| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                if a.history
                    .page_entries(&a.benchmarks)
                    .iter()
                    .any(|entry| entry.run_id == id.as_str())
                {
                    if checked {
                        a.history.checked.insert(id.to_string());
                    } else {
                        a.history.checked.remove(id.as_str());
                    }
                }
                a.ui_dirty = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_search_history(move |query| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                a.history.query = query.to_string();
                a.history.page = 0;
                let app = &mut *a;
                app.history.clear_hidden_selection(&app.benchmarks);
                a.ui_dirty = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_select_primary(move |index| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                a.primary_selection = index.max(0) as usize;
                a.ui_dirty = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_queue_action(move |index| {
            if let Some(a) = weak.upgrade() {
                a.borrow_mut().queue_action(index.max(0) as usize);
            }
        });
        let weak = Rc::downgrade(app);
        window.on_history_observation_change(move |action, index| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                a.history.observations.change(action.as_str(), index);
                a.ui_dirty = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_history_filter(move |index| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                a.history.status_filter = index.max(0) as usize;
                a.history.page = 0;
                let app = &mut *a;
                app.history.clear_hidden_selection(&app.benchmarks);
                a.ui_dirty = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_prepare_task(move |task| {
            if let Some(a) = weak.upgrade() {
                a.borrow_mut().run_task(task.as_str());
            }
        });
    }
    fn import_native(&mut self, path: PathBuf) {
        let Some(root) = self.native_history_path.clone() else {
            return;
        };
        self.history.open = true;
        self.history.import_error.clear();
        self.history_io(move || {
            HistoryEvent::Imported((|| {
                use std::io::Read;
                let mut bytes = Vec::new();
                fs::File::open(&path)
                    .map_err(|e| e.to_string())?
                    .take(gluj_bench_core::NATIVE_EXPORT_LIMIT + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|e| e.to_string())?;
                let bundle =
                    gluj_bench_core::NativeExport::decode(&bytes).map_err(|e| e.to_string())?;
                let store = RunStore::new(root);
                let mut count = 0;
                for mut record in bundle.records {
                    let original = record.run_id.clone();
                    let a = bundle.annotations.get(&original).cloned();
                    record.run_id = RunRecord::start(
                        &record.requested_benchmark_id,
                        &gluj_bench_core::BenchmarkConfig::default(),
                        vec![],
                    )
                    .run_id;
                    let previous_source = record.source.clone();
                    record.source = RunSource::ImportedComparison;
                    record.source_reference = Some(format!(
                        "{} | original run {} | previous source {:?}: {}",
                        path.display(),
                        original,
                        previous_source,
                        record.source_reference.as_deref().unwrap_or("none")
                    ));
                    store.save(&record).map_err(|e| {
                        format!("Imported {count} records before storage failure: {e}")
                    })?;
                    if let Some(mut a) = a {
                        a.pinned = false;
                        store
                            .annotate(&record.run_id, a)
                            .map_err(|e| e.to_string())?;
                    }
                    count += 1;
                }
                Ok(format!("Imported {count} run records for comparison."))
            })())
        });
    }
    pub(super) fn refresh_phase_eight(window: &MainWindow, app: &Self) {
        window.set_test_is_gpu_memory(
            app.selected
                .as_deref()
                .is_some_and(gluj_bench_core::gpu_memory),
        );
        window.set_history_open(app.history.open);
        window.set_history_status(app.history.status.clone().into());
        window.set_history_import_error(app.history.import_error.as_str().into());
        window.set_history_delete_pending(app.history.pending_delete.is_some());
        window.set_history_busy(app.history.io.is_some());
        window.set_history_can_export(
            !app.history.checked.is_empty() || app.history.record.is_some(),
        );
        let active_id = app.history.selected.as_deref().unwrap_or_default();
        let annotation = app
            .history
            .annotations
            .get(active_id)
            .cloned()
            .unwrap_or_default();
        window.set_history_active_id(active_id.into());
        window.set_history_pinned(annotation.pinned);
        window.set_history_archived(annotation.archived);
        let rows: Vec<HistoryRow> = app
            .history
            .page_entries(&app.benchmarks)
            .iter()
            .map(|e| {
                let annotation = app
                    .history
                    .annotations
                    .get(&e.run_id)
                    .cloned()
                    .unwrap_or_default();
                let name = if annotation.name.is_empty() {
                    history_title(&e.benchmark_id, &app.benchmarks)
                } else {
                    annotation.name
                };
                HistoryRow {
                    id: e.run_id.as_str().into(),
                    name: format!(
                        "{}{}",
                        if annotation.pinned { "Pinned | " } else { "" },
                        name,
                    )
                    .into(),
                    date: e
                        .started_unix_ms
                        .map(timestamp)
                        .unwrap_or_else(|| "Not recorded".into())
                        .into(),
                    status: format!("{:?}", e.status).into(),
                    source: match e.source {
                        RunSource::ImportedComparison => "Imported",
                        RunSource::LegacyLocal => "Legacy",
                        RunSource::LocalMeasurement => "Local",
                    }
                    .into(),
                    observations: e.sample_count.to_string().into(),
                    checked: app.history.checked.contains(&e.run_id),
                }
            })
            .collect();
        window.set_history_active_title(
            rows.iter()
                .find(|row| row.id.as_str() == active_id)
                .map(|row| row.name.clone())
                .unwrap_or_default(),
        );
        window.set_history_rows(ModelRc::new(VecModel::from(rows)));
        let total = app.history.visible(&app.benchmarks).len();
        let mut counts = format!("{total} {}", if total == 1 { "run" } else { "runs" });
        if !app.history.checked.is_empty() {
            counts.push_str(&format!(
                "  |  {} checked for export",
                app.history.checked.len()
            ));
        }
        window.set_history_counts(counts.into());
        window.set_history_page_index(app.history.page as i32);
        window.set_history_page_count(total.div_ceil(PAGE_SIZE) as i32);
        window.set_history_filter_index(app.history.status_filter as i32);
        window.set_history_query(app.history.query.as_str().into());
        window.set_history_name(annotation.name.into());
        window.set_history_notes(annotation.notes.into());
        window.set_history_selected(app.history.record.is_some());
        window.set_history_has_result(
            app.history
                .record
                .as_ref()
                .is_some_and(|record| record.result.is_some()),
        );
        let (facts, provenance) = app
            .history
            .record
            .as_ref()
            .map(|r| {
                use crate::presentation::fact;
                (
                    vec![
                        fact("Status", format!("{:?}", r.status)),
                        fact("Coverage", format!("{:?}", r.sample_coverage)),
                        fact("Raw observations", r.samples.len().to_string()),
                        fact("Validation", presentation::friendly(&r.validation.status)),
                    ],
                    vec![
                        fact("Run ID", &r.run_id),
                        fact("Benchmark", &r.requested_benchmark_id),
                        fact(
                            "Application",
                            r.application_version.as_deref().unwrap_or("Not recorded"),
                        ),
                        fact(
                            "Build",
                            r.build_revision.as_deref().unwrap_or("Not recorded"),
                        ),
                        fact("Source", format!("{:?}", r.source)),
                        fact(
                            "Source reference",
                            r.source_reference.as_deref().unwrap_or("Local"),
                        ),
                        fact("Validation scope", &r.validation.scope),
                        fact(
                            "Error",
                            r.error
                                .as_ref()
                                .map(|e| e.message.clone())
                                .unwrap_or_else(|| "None".into()),
                        ),
                    ],
                )
            })
            .unwrap_or_default();
        window.set_history_facts(ModelRc::new(VecModel::from(facts)));
        window.set_history_provenance(ModelRc::new(VecModel::from(provenance)));
        let partial = app.history.record.as_ref().filter(|record| {
            app.history.open
                && app.active_request.is_none()
                && record.result.is_none()
                && !record.samples.is_empty()
        });
        window.set_history_has_partial_observations(partial.is_some());
        window.set_history_observations(
            partial
                .map(|record| app.history.observations.view(record))
                .unwrap_or_default(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn measured(value: f64) -> BenchmarkResult {
        BenchmarkResult {
            benchmark_id: "gpu.bandwidth.vram".into(),
            device_id: "gpu:fixture".into(),
            elapsed_ns: 1000,
            metrics: vec![Metric {
                name: "working_set_1048576.read".into(),
                value,
                unit: "bytes/s".into(),
                statistics: gluj_bench_core::SampleStatistics {
                    sample_count: 6,
                    minimum: value * 0.9,
                    median: value,
                    maximum: value * 1.1,
                    standard_deviation: value * 0.01,
                },
            }],
            workload_metadata: BTreeMap::new(),
            device_metadata: BTreeMap::new(),
        }
    }
    fn temporary() -> PathBuf {
        std::env::temp_dir().join(format!(
            "gluj-phase8-{}",
            RunRecord::start("fixture", &Default::default(), vec![]).run_id
        ))
    }
    fn wait(app: &mut App) {
        let until = std::time::Instant::now() + Duration::from_secs(10);
        while app.history_busy() {
            app.receive_history_events();
            assert!(std::time::Instant::now() < until);
            thread::sleep(Duration::from_millis(5));
        }
    }
    #[test]
    fn selected_size_defaults_to_largest_and_typical_does_not_select_best() {
        let mut first = measured(20.);
        let second = measured(100.);
        let third = measured(30.);
        first.metrics.push(Metric {
            name: "working_set_4194304.read".into(),
            value: 5.,
            unit: "bytes/s".into(),
            statistics: Default::default(),
        });
        assert_eq!(sized_metrics(&first)[0].0, 4194304);
        assert_eq!(featured_metric(&first).unwrap().value, 5.);
        assert_eq!(
            typical(&[&second, &third], None),
            format_metric(65., "bytes/s")
        );
        assert!(measured_text(&third.metrics[0]).contains("6 observations"));
    }
    #[test]
    fn summary_cache_keeps_more_than_128_distinct_configurations() {
        let mut results = Vec::new();
        for i in 0..150 {
            let mut r = measured(20.);
            r.device_metadata
                .insert("saved_setup".into(), format!("setup-{i}"));
            remember_result(&mut results, &r);
        }
        assert_eq!(results.len(), 150);
    }
    #[test]
    fn history_search_matches_the_displayed_title() {
        let app = crate::configuration_tests::fixture();
        let descriptor = &app.benchmarks[0];
        let history = HistoryState {
            query: format!(" {} ", families::mode_name(descriptor).to_uppercase()),
            entries: vec![RunIndexEntry {
                run_id: "fixture-run".into(),
                source: RunSource::LocalMeasurement,
                started_unix_ms: None,
                benchmark_id: descriptor.id.clone(),
                status: CompletionStatus::Completed,
                sample_count: 0,
            }],
            ..Default::default()
        };
        assert_eq!(history.visible(&app.benchmarks).len(), 1);
    }

    #[test]
    fn opening_a_legacy_result_keeps_its_position_and_run_count() {
        let older = measured(30.);
        let newer = measured(40.);
        let record = RunRecord::legacy(
            older.clone(),
            RunSource::LegacyLocal,
            "results.json#0".into(),
        );
        let mut app = crate::configuration_tests::fixture();
        app.results = vec![older, newer.clone()];
        app.history.record = Some(record.clone());
        for _ in 0..2 {
            app.history_action("view", String::new(), String::new());
            assert_eq!(app.results.len(), 2);
            assert_eq!(app.results[1], newer);
            assert_eq!(
                app.history.viewed_run.as_deref(),
                Some(record.run_id.as_str())
            );
            assert_eq!(
                app.results[0].device_metadata["native_run_id"],
                record.run_id
            );
        }
    }

    #[test]
    fn history_names_pins_archive_and_background_import_survive_restart() {
        let path = temporary();
        let store = RunStore::new(&path);
        let original =
            RunRecord::legacy(measured(30.), RunSource::LocalMeasurement, "fixture".into());
        store.save(&original).unwrap();
        let mut app = crate::configuration_tests::fixture();
        app.native_history_path = Some(path.clone());
        app.reload_history();
        app.select_history(&original.run_id);
        wait(&mut app);
        assert_eq!(app.history.record.as_ref().unwrap().run_id, original.run_id);
        app.history_action(
            "save",
            "Before driver update".into(),
            "Quiet desktop".into(),
        );
        app.history_action("pin", String::new(), String::new());
        let mut reopened = crate::configuration_tests::fixture();
        reopened.native_history_path = Some(path.clone());
        reopened.reload_history();
        reopened.restore_pinned_baselines();
        assert_eq!(reopened.baseline_results.len(), 1);
        assert_eq!(
            reopened.baseline_results[0].device_metadata["saved_baseline_name"],
            "Before driver update"
        );
        app.history_action("archive", String::new(), String::new());
        assert!(app.history.visible(&app.benchmarks).is_empty());
        assert!(app.history.record.is_none());
        app.history.status_filter = 4;
        app.select_history(&original.run_id);
        wait(&mut app);
        assert_eq!(app.history.record.as_ref().unwrap().run_id, original.run_id);
        let bytes = store
            .export_runs(std::slice::from_ref(&original.run_id))
            .unwrap();
        let export = path.join("selected.json");
        write_export(&export, &bytes).unwrap();
        reopened.import_native(export);
        wait(&mut reopened);
        assert_eq!(reopened.history.entries.len(), 2);
        let imported = reopened
            .history
            .entries
            .iter()
            .find(|r| r.source == RunSource::ImportedComparison)
            .unwrap();
        let imported = store.load(&imported.run_id).unwrap();
        assert_eq!(imported.result, original.result);
        assert!(
            imported
                .source_reference
                .unwrap()
                .contains(&original.run_id)
        );
        assert!(!store.annotations().unwrap()[&imported.run_id].pinned);
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn skipping_one_pending_condition_preserves_other_work_and_persists_status() {
        let mut app = crate::configuration_tests::fixture();
        let id = app.selected.clone().unwrap();
        let run = app.queued_run(&id).unwrap();
        let mut other = run.clone();
        other.settings.ram_offload_percent = 50;
        let plan = app.prepare_plan(vec![run, other], "skip-test").unwrap();
        assert_eq!(plan.runs.len(), 2);
        app.queue = plan.runs;
        app.active_plan = Some(plan.suite);
        let path = temporary();
        app.native_history_path = Some(path.clone());
        app.queue_action(0);
        assert_eq!(app.queue.len(), 1);
        assert_eq!(
            app.active_plan.as_ref().unwrap().conditions[0].status,
            CompletionStatus::Skipped
        );
        let mut reopened = crate::configuration_tests::fixture();
        reopened.native_history_path = Some(path.clone());
        reopened.restore_plan_status();
        let restored = reopened.active_plan.unwrap();
        assert!(
            restored
                .conditions
                .iter()
                .all(|c| c.status == CompletionStatus::Skipped)
        );
        assert!(reopened.queue.is_empty());
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn stale_load_completion_does_not_replace_current_selection_and_failed_export_preserves_file() {
        let mut app = crate::configuration_tests::fixture();
        let (sender, events) = mpsc::channel();
        app.history.io = Some(events);
        app.history.selected = Some("new-selection".into());
        let record = RunRecord::start("fixture", &Default::default(), vec![]);
        sender
            .send(HistoryEvent::Loaded(
                "old-selection".into(),
                Ok(Box::new(record)),
            ))
            .unwrap();
        app.receive_history_events();
        assert!(app.history.record.is_none());
        let path = temporary();
        fs::create_dir_all(&path).unwrap();
        let target = path.join("records.json");
        fs::write(&target, b"previous export").unwrap();
        assert!(write_export(&path.join("missing").join("records.json"), b"new").is_err());
        assert_eq!(fs::read(target).unwrap(), b"previous export");
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn comparison_explains_the_actual_changed_access_option() {
        let mut app = crate::configuration_tests::fixture();
        let mut old = measured(20.);
        let mut current = old.clone();
        old.workload_metadata
            .insert("access_pattern".into(), "sequential".into());
        current
            .workload_metadata
            .insert("access_pattern".into(), "shuffled_blocks".into());
        app.baseline_results.push(old);
        let reason = app.comparison_reason(&current, &current.metrics[0].name, None);
        assert!(reason.contains("sequential -> shuffled_blocks"));
    }
    #[test]
    fn history_dates_and_search_preserve_unknown_timestamps() {
        assert_eq!(timestamp(0), "1970-01-01 00:00 UTC");
        assert_eq!(timestamp(1709164800000), "2024-02-29 00:00 UTC");
        let entry = RunIndexEntry {
            run_id: "fixture".into(),
            source: RunSource::LegacyLocal,
            started_unix_ms: None,
            benchmark_id: "cpu.fp32".into(),
            status: CompletionStatus::Completed,
            sample_count: 0,
        };
        assert!(label(&entry, &BTreeMap::new()).contains("Time not recorded"));
        let mut history = HistoryState {
            entries: vec![entry],
            ..Default::default()
        };
        history.query = "cpu".into();
        assert_eq!(history.visible(&[]).len(), 1);
        history.query = "missing".into();
        assert!(history.visible(&[]).is_empty());
    }
}
