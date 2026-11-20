//! Structured presentation only. Plans and measurements keep their native definitions.
use super::*;
use gluj_bench_core::{CompletionStatus, SuiteCondition};
use std::collections::BTreeSet;
pub(super) mod datasets;
pub(super) mod observations;

#[derive(Default)]
pub(super) struct State {
    pub scope: usize,
    pub single_test: bool,
    pub editing_saved_test: Option<String>,
    pub editor_test_name: Option<String>,
    pub editor_shorter_run: Option<bool>,
    pub saved_test_status: String,
    pub set_selected: bool,
    pub curated_set: Option<String>,
    pub test_editor: bool,
    pub editing_test: Option<usize>,
    pub inspecting_test: Option<usize>,
    latest_batch: Option<String>,
    batch_sequence: u64,
    query: String,
    excluded_tests: BTreeSet<String>,
    expanded_tests: BTreeSet<String>,
    queue_filter: usize,
    pub technical_query: String,
    pub hardware_query: String,
}
impl State {
    pub(super) fn builder() -> Self {
        Self {
            scope: 1,
            single_test: true,
            ..Default::default()
        }
    }
    pub(super) fn clear_test_selection(&mut self) {
        self.query.clear();
        self.excluded_tests.clear();
        self.expanded_tests.clear();
    }
    pub(super) fn begin_result_batch(&mut self) {
        self.batch_sequence = self.batch_sequence.wrapping_add(1);
        self.latest_batch = Some(format!(
            "{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            self.batch_sequence
        ));
    }
    pub(super) fn tag_result(&mut self, result: &mut BenchmarkResult) {
        if self.latest_batch.is_none() {
            self.begin_result_batch();
        }
        result
            .device_metadata
            .insert("saved_run_batch".into(), self.latest_batch.clone().unwrap());
    }
}
fn recorded_ms(result: &BenchmarkResult) -> Option<u64> {
    result
        .device_metadata
        .get("saved_at_unix_ms")
        .and_then(|v| v.parse().ok())
        .or_else(|| {
            result
                .device_metadata
                .get("saved_at")
                .and_then(|v| v.parse::<u64>().ok())?
                .checked_mul(1000)
        })
}
pub(super) fn result_timestamp(result: &BenchmarkResult) -> String {
    recorded_ms(result)
        .map(|ms| {
            format!(
                "{}:{:02} UTC",
                phase_eight::timestamp(ms).trim_end_matches(" UTC"),
                ms / 1000 % 60
            )
        })
        .unwrap_or_else(|| "Time not recorded".into())
}
fn result_run_key(result: &BenchmarkResult) -> Option<String> {
    for (field, prefix) in [("saved_run_batch", "batch"), ("native_run_id", "native")] {
        if let Some(id) = result
            .device_metadata
            .get(field)
            .filter(|id| !id.is_empty())
        {
            return Some(format!("{prefix}:{id}"));
        }
    }
    recorded_ms(result).map(|ms| {
        format!(
            "legacy:{ms}:{}:{}:{}",
            result.benchmark_id,
            result_component_key(result),
            result_configuration_label(result)
        )
    })
}
impl App {
    pub(super) fn result_is_new(&self, result: &BenchmarkResult) -> bool {
        let latest = self
            .presentation
            .latest_batch
            .as_ref()
            .map(|id| format!("batch:{id}"))
            .or_else(|| self.saved_results.last().and_then(result_run_key));
        let Some(latest) = latest else {
            return false;
        };
        if result_run_key(result).as_deref() == Some(latest.as_str()) {
            return true;
        }
        result
            .device_metadata
            .get("native_run_id")
            .is_some_and(|id| {
                self.saved_results.iter().any(|saved| {
                    saved.device_metadata.get("native_run_id") == Some(id)
                        && result_run_key(saved).as_deref() == Some(latest.as_str())
                })
            })
    }
}
fn model<T: Clone + 'static>(rows: Vec<T>) -> ModelRc<T> {
    ModelRc::new(VecModel::from(rows))
}
pub(super) fn fact(label: &str, value: impl Into<String>) -> DeviceFact {
    DeviceFact {
        label: label.into(),
        value: value.into().into(),
    }
}
pub(super) fn friendly(value: &str) -> String {
    value.replace('_', " ")
}
pub(super) fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    let needle = needle.trim();
    if needle.is_empty() {
        return true;
    }
    if needle.is_ascii() {
        let nb = needle.as_bytes();
        haystack
            .as_bytes()
            .windows(nb.len())
            .any(|window| window.eq_ignore_ascii_case(nb))
    } else {
        haystack.to_lowercase().contains(&needle.to_lowercase())
    }
}

pub(super) fn matches_pair_query(key: &str, value: &str, query: &str) -> bool {
    let query = query.trim();
    if query.is_empty() {
        return true;
    }
    if contains_ignore_ascii_case(key, query) || contains_ignore_ascii_case(value, query) {
        return true;
    }
    query.split_whitespace().all(|term| {
        contains_ignore_ascii_case(key, term) || contains_ignore_ascii_case(value, term)
    })
}

pub(super) fn clamp_display_text(value: &str, max_chars: usize, max_lines: usize) -> String {
    if value.is_empty() {
        return String::new();
    }
    let max_chars = max_chars.max(1);
    let max_lines = max_lines.max(1);
    if value.len() <= max_chars && value.bytes().filter(|b| *b == b'\n').count() < max_lines {
        return value.to_owned();
    }
    let mut lines = 1usize;
    let mut cutoff = None;
    for (chars, (idx, ch)) in value.char_indices().enumerate() {
        if chars >= max_chars {
            cutoff = Some(idx);
            break;
        }
        if ch == '\n' {
            if lines >= max_lines {
                cutoff = Some(idx);
                break;
            }
            lines += 1;
        }
    }
    let Some(end) = cutoff else {
        return value.to_owned();
    };
    let prefix = value[..end].trim_end();
    if max_lines <= 1 {
        format!("{prefix} … (truncated; {} bytes)", value.len())
    } else {
        format!("{prefix}\n… (truncated; {} bytes)", value.len())
    }
}

pub(super) fn bounded_fact(label: &str, value: &str) -> DeviceFact {
    fact(label, clamp_display_text(value, 240, 4))
}

pub(super) fn technical_fact(label: &str, value: &str) -> DeviceFact {
    fact(
        label,
        if value.len() > 180 {
            if value.len() > 65_536 {
                format!(
                    "Recorded payload · {} KiB · Native JSON",
                    value.len().div_ceil(1024)
                )
            } else {
                format!(
                    "Recorded payload · {} characters · Native JSON",
                    value.chars().count()
                )
            }
        } else {
            value.to_owned()
        },
    )
}
fn option(c: &SuiteCondition, key: &str, fallback: &str) -> String {
    c.condition
        .options
        .get(key)
        .map(|s| friendly(s))
        .unwrap_or_else(|| fallback.into())
}
fn test_key(scope: usize, c: &SuiteCondition) -> String {
    let options: BTreeMap<_, _> = c
        .condition
        .options
        .iter()
        .filter(|(key, _)| key.as_str() != "measurement_policy")
        .collect();
    serde_json::to_string(&(
        scope,
        &c.condition.benchmark_id,
        &c.condition.device_id,
        options,
        &c.condition.kernel_revision,
        &c.condition.timing_scope,
    ))
    .expect("resolved test identity uses serializable strings")
}
fn test_details(c: &SuiteCondition, row: &ConditionRow) -> Vec<DeviceFact> {
    vec![
        fact("Hardware", row.target.as_str()),
        fact("Dataset", row.dataset.as_str()),
        fact("Workers", row.workers.as_str()),
        fact("Observations", c.condition.policy.samples.to_string()),
    ]
}
pub(super) fn condition_row(c: &SuiteCondition, index: usize, app: &App) -> ConditionRow {
    let descriptor = app
        .benchmarks
        .iter()
        .find(|b| b.id == c.condition.benchmark_id);
    let title = descriptor
        .map(families::mode_name)
        .unwrap_or_else(|| c.condition.benchmark_id.clone());
    let group = descriptor
        .map(|b| families::title(app.family_of(b)))
        .unwrap_or("Other");
    let target = app
        .devices
        .iter()
        .find(|d| d.id == c.condition.device_id)
        .map(|d| d.name.clone())
        .unwrap_or_else(|| c.condition.device_id.clone());
    let mut dataset = c
        .condition
        .options
        .get("dataset_bytes")
        .and_then(|s| s.parse::<u64>().ok())
        .map(|n| format_binary_size(n as f64))
        .unwrap_or_else(|| {
            if c.condition
                .options
                .get("dataset_mode")
                .is_some_and(|s| s == "register")
            {
                "Registers".into()
            } else {
                "Runtime-selected".into()
            }
        });
    if c.condition
        .options
        .get("dataset_mode")
        .is_some_and(|s| s == "sweep")
    {
        dataset = format!("Sweep ≤ {dataset}");
    }
    let mut mode = [
        option(c, "access_pattern", ""),
        option(c, "matrix_traversal", ""),
        option(c, "memory_operation", ""),
        option(c, "memory_copy_path", ""),
        option(c, "transfer_direction", ""),
    ]
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect::<Vec<_>>()
    .join(" · ");
    for (key, label) in [
        ("access_realizations", "realizations"),
        ("matrix_tile_shape", "tile"),
        ("matrix_tile_reuse", "reuse"),
        ("staging_buffers", "ring"),
        ("transfer_batch", "batch"),
        ("transfer_submission", "submission"),
        ("arithmetic_iterations", "iterations"),
    ] {
        if let Some(value) = c.condition.options.get(key)
            && (key != "access_realizations" || value != "1")
        {
            if !mode.is_empty() {
                mode.push_str(" · ");
            }
            if key == "access_realizations" {
                mode.push_str(&format!("{value} {label}"));
            } else {
                mode.push_str(&format!("{label} {}", friendly(value)));
            }
        }
    }
    let workers = if c.condition.benchmark_id.starts_with("gpu.") {
        "GPU".into()
    } else if c.condition.benchmark_id.starts_with("cpu.latency.") {
        "1 pinned worker".into()
    } else {
        format!(
            "{} {}",
            option(c, "cpu_core_limit", "Automatic"),
            if c.condition
                .options
                .get("cpu_core_limit")
                .is_some_and(|s| s == "1")
            {
                "core"
            } else {
                "cores"
            }
        )
    };
    if let Some(first) = mode.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    ConditionRow {
        included: true,
        expanded: false,
        details: model(vec![]),
        index: index as i32,
        title: title.into(),
        group: group.into(),
        mode: mode.into(),
        dataset: dataset.into(),
        target: target.into(),
        workers: workers.into(),
        status: format!("{:?}", c.status).into(),
        action: match c.status {
            CompletionStatus::Running => "Cancel test",
            CompletionStatus::Pending => "Skip",
            _ => "",
        }
        .into(),
    }
}
pub(super) fn plan_facts(plan: &families::PreparedPlan) -> Vec<DeviceFact> {
    let known: Vec<_> = plan
        .suite
        .conditions
        .iter()
        .filter_map(|c| c.estimated_duration_ms)
        .collect();
    let complete_time = known.len() == plan.suite.conditions.len();
    let peak = plan
        .suite
        .conditions
        .iter()
        .filter_map(|c| c.estimated_peak_bytes)
        .max();
    let complete_memory = plan
        .suite
        .conditions
        .iter()
        .all(|c| c.estimated_peak_bytes.is_some());
    vec![
        fact("Tests to run", plan.runs.len().to_string()),
        fact(
            "Minimum measurement",
            if complete_time {
                format!("≥ {} s", known.iter().sum::<u64>() as f64 / 1000.)
            } else {
                "Resolves at execution".into()
            },
        ),
        fact(
            if complete_memory {
                "Estimated peak test buffers"
            } else {
                "Known buffer estimate · partial"
            },
            peak.map(|n| format_binary_size(n as f64))
                .unwrap_or_else(|| "Resolves at execution".into()),
        ),
    ]
}
impl App {
    fn presentation_base_plan(&self) -> Result<families::PreparedPlan, String> {
        if self.presentation.set_selected {
            if self.presentation.curated_set.is_some() {
                return self.selected_curated_plan();
            }
            if !self.presentation.test_editor {
                return self.prepare_set_plan(self.custom_plan.clone(), "custom");
            }
            return self
                .selected
                .as_deref()
                .ok_or("Choose a workload".into())
                .and_then(|id| self.configured_plan(id));
        }
        if self.presentation.scope == 0 {
            self.selected_family()
                .ok_or_else(|| "Select a test.".into())
                .and_then(|family| self.curated_plan(Some(family)))
        } else if !self.presentation.single_test && !self.custom_plan.is_empty() {
            self.prepare_plan(self.custom_plan.clone(), "custom")
        } else {
            self.selected
                .as_ref()
                .ok_or_else(|| "Select a test.".into())
                .and_then(|id| self.configured_plan(id))
        }
    }

    fn select_presentation_tests(
        &self,
        mut plan: families::PreparedPlan,
    ) -> Result<families::PreparedPlan, String> {
        let mut runs = VecDeque::new();
        let mut tests = Vec::new();
        for (run, test) in std::mem::take(&mut plan.runs)
            .into_iter()
            .zip(std::mem::take(&mut plan.suite.conditions))
        {
            if !self
                .presentation
                .excluded_tests
                .contains(&test_key(self.presentation.scope, &test))
            {
                runs.push_back(run);
                tests.push(test);
            }
        }
        if runs.is_empty() {
            return Err("Select at least one test.".into());
        }
        plan.runs = runs;
        plan.suite.conditions = tests;
        Ok(plan)
    }
    pub(super) fn presentation_plan(&self) -> Result<families::PreparedPlan, String> {
        self.select_presentation_tests(self.presentation_base_plan()?)
    }
    fn set_presentation_test_enabled(&mut self, index: usize, enabled: bool) {
        if self.active_request.is_some() {
            return;
        }
        let Ok(plan) = self.presentation_base_plan() else {
            return;
        };
        let Some(test) = plan.suite.conditions.get(index) else {
            return;
        };
        let key = test_key(self.presentation.scope, test);
        if enabled {
            self.presentation.excluded_tests.remove(&key);
        } else {
            self.presentation.excluded_tests.insert(key);
        }
    }
    fn toggle_presentation_details(&mut self, index: usize) {
        if self.active_request.is_some() {
            return;
        }
        let Ok(plan) = self.presentation_base_plan() else {
            return;
        };
        let Some(test) = plan.suite.conditions.get(index) else {
            return;
        };
        let key = test_key(self.presentation.scope, test);
        if !self.presentation.expanded_tests.remove(&key) {
            self.presentation.expanded_tests.insert(key);
        }
    }
    pub(super) fn wire_presentation(window: &MainWindow, app: &Rc<RefCell<Self>>) {
        let weak = Rc::downgrade(app);
        window.on_plan_scope_changed(move |index| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                a.presentation.scope = index.clamp(0, 1) as usize;
                a.dirty.presentation = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_plan_search(move |query| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                a.presentation.query = query.to_string();
                a.dirty.presentation = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_plan_action(move |action, index| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                match action.as_str() {
                    "run" => {
                        let plan = a.presentation_plan();
                        a.begin_plan(plan);
                        a.ui_dirty = true;
                    }
                    "include" => {
                        a.set_presentation_test_enabled(index.max(0) as usize, true);
                        a.dirty.presentation = true;
                    }
                    "exclude" => {
                        a.set_presentation_test_enabled(index.max(0) as usize, false);
                        a.dirty.presentation = true;
                    }
                    "details" => {
                        a.toggle_presentation_details(index.max(0) as usize);
                        a.dirty.presentation = true;
                    }
                    "remove"
                        if a.presentation.scope == 1
                            && !a.custom_plan.is_empty()
                            && a.active_request.is_none() =>
                    {
                        a.custom_plan.remove(index.max(0) as usize);
                        a.dirty.presentation = true;
                        a.dirty.setup = true;
                    }
                    _ => {}
                }
            }
        });
        let weak = Rc::downgrade(app);
        window.on_detail_search(move |kind, text| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                if kind == "hardware" {
                    a.presentation.hardware_query = text.to_string();
                    a.dirty.devices = true;
                } else {
                    a.presentation.technical_query = text.to_string();
                    a.dirty.result_technical = true;
                }
            }
        });
        let weak = Rc::downgrade(app);
        window.on_queue_filter_changed(move |index| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                a.presentation.queue_filter = index.max(0) as usize;
                a.dirty.presentation = true;
            }
        });
    }
    pub(super) fn refresh_presentation(window: &MainWindow, app: &Self) {
        window.set_batch_builder(!app.presentation.single_test);
        window.set_plan_scope(app.presentation.scope as i32);
        window.set_queue_filter_index(app.presentation.queue_filter as i32);
        if app.active_request.is_none() {
            let base = app.presentation_base_plan();
            let plan = base
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|base| app.select_presentation_tests(base.clone()));
            window.set_plan_valid(plan.as_ref().is_ok_and(|p| !p.runs.is_empty()));
            window.set_plan_error(
                plan.as_ref()
                    .err()
                    .map(|error| clamp_display_text(error, 1200, 12))
                    .unwrap_or_default()
                    .into(),
            );
            let total = base.as_ref().map(|p| p.runs.len()).unwrap_or(0);
            window.set_plan_filter_visible(total > 6 || !app.presentation.query.is_empty());
            let selected = plan.as_ref().map(|p| p.runs.len()).unwrap_or(0);
            let mut facts = plan.as_ref().map(plan_facts).unwrap_or_else(|_| {
                vec![
                    fact("Tests to run", "0"),
                    fact("Minimum measurement", "—"),
                    fact("Estimated peak test buffers", "—"),
                ]
            });
            facts[0].value = format!("{selected} / {total}").into();
            window.set_plan_facts(model(facts));
            let mut rows = base
                .as_ref()
                .map(|p| {
                    p.suite
                        .conditions
                        .iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let mut row = condition_row(c, i, app);
                            let key = test_key(app.presentation.scope, c);
                            row.included = !app.presentation.excluded_tests.contains(&key);
                            row.expanded = app.presentation.expanded_tests.contains(&key);
                            if row.expanded {
                                row.details = model(test_details(c, &row));
                            }
                            row
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let query = app.presentation.query.trim();
            if !query.is_empty() {
                rows.retain(|r| {
                    [
                        r.group.as_str(),
                        r.title.as_str(),
                        r.mode.as_str(),
                        r.dataset.as_str(),
                        r.target.as_str(),
                    ]
                    .iter()
                    .any(|field| contains_ignore_ascii_case(field, query))
                });
            }
            rows.sort_by(|a, b| {
                a.group
                    .cmp(&b.group)
                    .then(a.title.cmp(&b.title))
                    .then(a.index.cmp(&b.index))
            });
            window.set_plan_page(if query.is_empty() {
                "".into()
            } else {
                format!(
                    "{} matching {}",
                    rows.len(),
                    if rows.len() == 1 { "test" } else { "tests" }
                )
                .into()
            });
            window.set_plan_rows(model(rows));
        }
        let queue = app
            .active_plan
            .as_ref()
            .map(|p| {
                p.conditions
                    .iter()
                    .enumerate()
                    .map(|(i, c)| condition_row(c, i, app))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let counts = [
            "Running",
            "Pending",
            "Completed",
            "Cancelled",
            "Failed",
            "Skipped",
        ]
        .iter()
        .map(|s| {
            format!(
                "{} {}",
                queue.iter().filter(|r| r.status == *s).count(),
                s.to_lowercase()
            )
        })
        .collect::<Vec<_>>()
        .join(" · ");
        window.set_queue_summary(counts.into());
        let filtered = queue
            .into_iter()
            .filter(|r| match app.presentation.queue_filter {
                1 => r.status == "Running" || r.status == "Pending",
                2 => r.status == "Completed",
                3 => r.status == "Cancelled" || r.status == "Failed" || r.status == "Skipped",
                _ => true,
            })
            .collect();
        window.set_condition_queue(model(filtered));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration_tests::fixture;

    fn dated_result(id: &str, seconds: &str) -> BenchmarkResult {
        BenchmarkResult {
            benchmark_id: "cpu.performance.integer.i64".into(),
            device_id: "cpu:system".into(),
            elapsed_ns: 1,
            metrics: vec![],
            workload_metadata: BTreeMap::new(),
            device_metadata: BTreeMap::from([
                ("saved_at".into(), seconds.into()),
                ("native_run_id".into(), id.into()),
            ]),
        }
    }
    #[test]
    fn test_checkboxes_control_execution_without_hiding_rows_or_changing_with_sampling() {
        let mut app = fixture();
        app.presentation.scope = 1;
        let run = app.queued_run(GPU_OFFLOAD_ID).unwrap();
        app.custom_plan = [50, 75, 100]
            .into_iter()
            .map(|share| {
                let mut run = run.clone();
                run.settings.ram_offload_percent = share;
                run
            })
            .collect();
        assert_eq!(app.presentation_base_plan().unwrap().runs.len(), 3);
        app.set_presentation_test_enabled(1, false);
        app.presentation.query = "only show one row".into();
        let selected = app.presentation_plan().unwrap();
        assert_eq!(
            selected
                .runs
                .iter()
                .map(|run| run.settings.ram_offload_percent)
                .collect::<Vec<_>>(),
            vec![50, 100]
        );
        assert_eq!(app.presentation_base_plan().unwrap().runs.len(), 3);
        selected.suite.validate().unwrap();
        for (run, test) in selected.runs.iter().zip(&selected.suite.conditions) {
            let descriptor = app
                .benchmarks
                .iter()
                .find(|b| b.id == run.benchmark_id)
                .unwrap();
            let config =
                resolved_ui_config(&app.condition_for(descriptor), run, &app.devices).unwrap();
            assert_eq!(
                app.condition_for(descriptor)
                    .resolve(&config, &app.devices)
                    .unwrap(),
                test.condition
            );
        }
        app.settings.shorter_run = true;
        let short = app.presentation_plan().unwrap();
        assert_eq!(short.runs.len(), 2);
        assert!(
            short
                .suite
                .conditions
                .iter()
                .all(|test| test.condition.policy.samples == 3)
        );
        assert_eq!(plan_facts(&short)[1].value, "≥ 0.6 s");
        app.custom_plan.remove(0);
        assert_eq!(
            app.presentation_plan().unwrap().runs[0]
                .settings
                .ram_offload_percent,
            100
        );
        app.set_presentation_test_enabled(1, false);
        assert!(
            matches!(app.presentation_plan(), Err(error) if error.contains("Select at least one test"))
        );
        assert_eq!(app.presentation_base_plan().unwrap().runs.len(), 2);
        app.set_presentation_test_enabled(0, true);
        assert_eq!(
            app.presentation_plan().unwrap().runs[0]
                .settings
                .ram_offload_percent,
            75
        );
        app.active_request = Some("in-flight".into());
        app.set_presentation_test_enabled(0, false);
        assert_eq!(app.presentation_plan().unwrap().runs.len(), 1);
    }
    #[test]
    fn inline_test_details_follow_workload_identity_and_toggle_in_place() {
        let mut app = fixture();
        let base = app.presentation_base_plan().unwrap();
        let key = test_key(app.presentation.scope, &base.suite.conditions[0]);
        app.toggle_presentation_details(0);
        assert!(app.presentation.expanded_tests.contains(&key));
        let condition = &base.suite.conditions[0];
        let row = condition_row(condition, 0, &app);
        let details = test_details(condition, &row);
        assert!(
            details
                .iter()
                .any(|fact| fact.label == "Hardware" && fact.value == row.target)
        );
        assert!(
            details
                .iter()
                .any(|fact| fact.label == "Observations" && fact.value == "6")
        );
        app.settings.shorter_run = true;
        assert_eq!(
            test_key(
                app.presentation.scope,
                &app.presentation_base_plan().unwrap().suite.conditions[0]
            ),
            key
        );
        app.toggle_presentation_details(0);
        assert!(!app.presentation.expanded_tests.contains(&key));
    }

    #[test]
    fn result_times_use_recorded_precision_and_keep_unknown_dates_explicit() {
        let mut result = dated_result("one", "1709164801");
        assert_eq!(result_timestamp(&result), "2024-02-29 00:00:01 UTC");
        result
            .device_metadata
            .insert("saved_at_unix_ms".into(), "1709164802345".into());
        assert_eq!(result_timestamp(&result), "2024-02-29 00:00:02 UTC");
        result.device_metadata.clear();
        assert_eq!(result_timestamp(&result), "Time not recorded");
        result
            .device_metadata
            .insert("saved_at".into(), u64::MAX.to_string());
        assert_eq!(result_timestamp(&result), "Time not recorded");
    }
    #[test]
    fn new_flags_track_the_whole_latest_run_without_marking_imports_as_new() {
        let mut app = fixture();
        let older = dated_result("older", "1");
        app.saved_results.push(older.clone());
        assert!(app.result_is_new(&older));
        app.presentation.begin_result_batch();
        assert!(!app.result_is_new(&older));
        let mut first = dated_result("first", "2");
        app.presentation.tag_result(&mut first);
        app.saved_results.push(first.clone());
        let mut second = dated_result("second", "3");
        app.presentation.tag_result(&mut second);
        app.saved_results.push(second.clone());
        assert!(app.result_is_new(&first) && app.result_is_new(&second));
        assert!(!app.result_is_new(&dated_result("imported", "9999999999")));
        let mut native_view = first.clone();
        native_view.device_metadata.remove("saved_run_batch");
        assert!(app.result_is_new(&native_view));
        app.presentation = State::default();
        assert!(app.result_is_new(&first) && app.result_is_new(&second));
        assert!(!app.result_is_new(&older));
        app.presentation.begin_result_batch();
        assert!(!app.result_is_new(&first) && !app.result_is_new(&second));
    }

    #[test]
    fn curated_and_custom_resolve_the_previewed_conditions() {
        let mut app = fixture();
        assert_eq!(app.presentation.scope, 0);
        assert_eq!(
            app.presentation_plan().unwrap().suite.conditions,
            app.curated_plan(app.selected_family())
                .unwrap()
                .suite
                .conditions
        );
        let configured = app
            .configured_plan(app.selected.as_deref().unwrap())
            .unwrap();
        app.presentation.scope = 1;
        assert_eq!(
            app.presentation_plan().unwrap().suite.conditions,
            configured.suite.conditions
        );
        app.custom_plan = configured.runs.into();
        assert_eq!(
            app.presentation_plan().unwrap().suite.conditions,
            app.prepare_plan(app.custom_plan.clone(), "custom")
                .unwrap()
                .suite
                .conditions
        );
        app.custom_plan.clear();
        app.selected = None;
        assert!(app.presentation_plan().is_err());
    }

    #[test]
    fn adding_the_current_configuration_selects_custom_and_preserves_its_conditions() {
        let mut app = fixture();
        let expected = app
            .configured_plan(app.selected.as_deref().unwrap())
            .unwrap();
        app.add_custom_condition();
        assert_eq!(app.presentation.scope, 1);
        assert_eq!(
            app.presentation_plan().unwrap().suite.conditions,
            expected.suite.conditions
        );
        let count = app.custom_plan.len();
        app.add_custom_condition();
        assert_eq!(app.custom_plan.len(), count);
        app.custom_plan.clear();
        assert_eq!(
            app.presentation_plan().unwrap().suite.conditions,
            expected.suite.conditions
        );
        app.selected = None;
        app.presentation.scope = 0;
        assert!(app.presentation_plan().is_err());
    }

    #[test]
    fn unresolved_estimates_do_not_look_like_complete_totals() {
        let app = fixture();
        let mut plan = app.presentation_plan().unwrap();
        plan.suite.conditions[0].estimated_duration_ms = None;
        plan.suite.conditions[0].estimated_peak_bytes = None;
        let facts = plan_facts(&plan);
        assert_eq!(facts[1].value, "Resolves at execution");
        assert!(facts[2].label.contains("partial"));
        assert!(!facts[2].value.is_empty());
        for condition in &mut plan.suite.conditions {
            condition.estimated_peak_bytes = None;
        }
        assert_eq!(plan_facts(&plan)[2].value, "Resolves at execution");
    }

    #[test]
    fn condition_rows_keep_execution_index_and_friendly_target() {
        let app = fixture();
        let plan = app.presentation_plan().unwrap();
        let row = condition_row(&plan.suite.conditions[0], 51, &app);
        assert_eq!(row.index, 51);
        assert_eq!(row.target, "Discrete GPU");
        assert_eq!(row.action, "Skip");
        let mut current = plan.suite.conditions[0].clone();
        current.status = CompletionStatus::Running;
        assert_eq!(condition_row(&current, 51, &app).action, "Cancel test");
    }

    #[test]
    fn clamp_display_text_caps_characters_and_lines() {
        assert_eq!(clamp_display_text("short text", 20, 2), "short text");
        let long_single = "a".repeat(50);
        let clipped_line = clamp_display_text(&long_single, 10, 1);
        assert!(clipped_line.starts_with("aaaaaaaaaa … (truncated; 50 bytes)"));
        let multiline = "line 1\nline 2\nline 3\nline 4";
        let clipped_lines = clamp_display_text(multiline, 200, 2);
        assert_eq!(clipped_lines, "line 1\nline 2\n… (truncated; 27 bytes)");
        let fact_row = bounded_fact("Error", &long_single);
        assert_eq!(fact_row.label.as_str(), "Error");
        assert_eq!(fact_row.value.as_str(), long_single);
        let huge_fact = bounded_fact("Error", &"x".repeat(400));
        assert!(huge_fact.value.as_str().contains("truncated; 400 bytes"));
    }

    #[test]
    fn technical_fact_summarizes_medium_and_huge_payloads() {
        let short = technical_fact("Driver", "24.10.1");
        assert_eq!(short.value.as_str(), "24.10.1");

        let medium = "é".repeat(100);
        assert_eq!(medium.len(), 200);
        let medium_fact = technical_fact("Blob", &medium);
        assert_eq!(
            medium_fact.value.as_str(),
            "Recorded payload · 100 characters · Native JSON"
        );

        let huge = "x".repeat(65_537);
        let huge_fact = technical_fact("Large blob", &huge);
        assert_eq!(
            huge_fact.value.as_str(),
            "Recorded payload · 65 KiB · Native JSON"
        );
    }

    #[test]
    fn ascii_case_insensitive_search_matches_keys_and_values() {
        assert!(contains_ignore_ascii_case("Vulkan PCIe 4.0 x16", ""));
        assert!(contains_ignore_ascii_case(
            "Vulkan PCIe 4.0 x16",
            "pcie 4.0"
        ));
        assert!(!contains_ignore_ascii_case("short", "longer needle"));
        assert!(matches_pair_query("Adapter link", "Gen4 x16", "GEN4"));
        assert!(matches_pair_query("Adapter link", "Gen4 x16", "link"));
        assert!(!matches_pair_query("Adapter link", "Gen4 x16", "Gen5"));
    }

    #[test]
    #[ignore = "headless scoped-dirty and cache-preservation check, run separately"]
    fn headless_detail_and_history_search_preserve_result_presentations() {
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
        slint::platform::set_platform(Box::new(Headless(adapter))).unwrap();
        let window = MainWindow::new().unwrap();
        let mut state = fixture();
        state.results.push(BenchmarkResult {
            benchmark_id: "cpu.performance.float.f32".into(),
            device_id: "cpu:fixture".into(),
            elapsed_ns: 100,
            metrics: vec![Metric {
                name: "throughput".into(),
                value: 1.0,
                unit: "operations/s".into(),
                statistics: Default::default(),
            }],
            workload_metadata: BTreeMap::from([("access_pattern".into(), "sequential".into())]),
            device_metadata: BTreeMap::from([("driver_version".into(), "24.12.1".into())]),
        });
        let app = Rc::new(RefCell::new(state));
        App::wire(&window, &app);
        App::wire_presentation(&window, &app);
        App::wire_phase_eight(&window, &app);
        App::refresh(&window, &app.borrow());
        assert_eq!(app.borrow().result_presentations.borrow().len(), 1);
        {
            let mut app = app.borrow_mut();
            app.ui_dirty = false;
            app.dirty = Default::default();
        }

        window.invoke_detail_search("hardware".into(), "missing".into());
        window.invoke_detail_search("result".into(), "24.12".into());
        window.invoke_search_history("cpu".into());
        {
            let app = app.borrow();
            assert!(!app.ui_dirty);
            assert!(app.dirty.devices);
            assert!(app.dirty.result_technical);
            assert!(app.dirty.history);
        }

        App::flush_dirty(&window, &app);
        assert_eq!(app.borrow().result_presentations.borrow().len(), 1);
        assert_eq!(window.get_result_technical().row_count(), 1);
    }
}
