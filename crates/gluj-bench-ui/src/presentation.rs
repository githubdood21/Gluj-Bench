//! Structured presentation only. Plans and measurements keep their native definitions.
use super::*;
use gluj_bench_core::{CompletionStatus, SuiteCondition};
use std::collections::BTreeSet;
pub(super) mod datasets;
pub(super) mod observations;

#[derive(Default)]
pub(super) struct State {
    pub scope: usize,
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
pub(super) fn technical_fact(label: &str, value: &str) -> DeviceFact {
    fact(
        label,
        if value.len() > 180 {
            format!(
                "Recorded payload · {} characters · Native JSON",
                value.chars().count()
            )
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
fn condition_row(c: &SuiteCondition, index: usize, app: &App) -> ConditionRow {
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
        if self.presentation.scope == 0 {
            self.selected_family()
                .ok_or_else(|| "Select a test.".into())
                .and_then(|family| self.curated_plan(Some(family)))
        } else if !self.custom_plan.is_empty() {
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
    fn presentation_plan(&self) -> Result<families::PreparedPlan, String> {
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
                a.ui_dirty = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_plan_search(move |query| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                a.presentation.query = query.to_string();
                a.ui_dirty = true;
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
                    }
                    "include" => a.set_presentation_test_enabled(index.max(0) as usize, true),
                    "exclude" => a.set_presentation_test_enabled(index.max(0) as usize, false),
                    "details" => a.toggle_presentation_details(index.max(0) as usize),
                    "remove"
                        if a.presentation.scope == 1
                            && !a.custom_plan.is_empty()
                            && a.active_request.is_none() =>
                    {
                        a.custom_plan.remove(index.max(0) as usize);
                    }
                    _ => {}
                }
                a.ui_dirty = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_detail_search(move |kind, text| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                if kind == "hardware" {
                    a.presentation.hardware_query = text.to_string();
                } else {
                    a.presentation.technical_query = text.to_string();
                }
                a.ui_dirty = true;
            }
        });
        let weak = Rc::downgrade(app);
        window.on_queue_filter_changed(move |index| {
            if let Some(a) = weak.upgrade() {
                let mut a = a.borrow_mut();
                a.presentation.queue_filter = index.max(0) as usize;
                a.ui_dirty = true;
            }
        });
    }
    pub(super) fn refresh_presentation(window: &MainWindow, app: &Self) {
        window.set_plan_scope(app.presentation.scope as i32);
        window.set_queue_filter_index(app.presentation.queue_filter as i32);
        if app.active_request.is_none() {
            let base = app.presentation_base_plan();
            let plan = base
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|base| app.select_presentation_tests(base.clone()));
            window.set_plan_valid(plan.as_ref().is_ok_and(|p| !p.runs.is_empty()));
            window.set_plan_error(plan.as_ref().err().cloned().unwrap_or_default().into());
            let total = base.as_ref().map(|p| p.runs.len()).unwrap_or(0);
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
            let query = app.presentation.query.to_lowercase();
            rows.retain(|r| {
                format!(
                    "{} {} {} {} {}",
                    r.group, r.title, r.mode, r.dataset, r.target
                )
                .to_lowercase()
                .contains(&query)
            });
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
}
