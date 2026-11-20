use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct SavedTest {
    pub id: String,
    pub name: String,
    pub category: String,
    pub runs: Vec<QueuedRun>,
}

impl App {
    pub(super) fn builder_shorter_run(&self) -> bool {
        self.presentation
            .editor_shorter_run
            .unwrap_or(self.settings.shorter_run)
    }
    pub(super) fn editor_test_name(&self) -> String {
        self.presentation
            .editor_test_name
            .clone()
            .unwrap_or_else(|| {
                self.selected_family()
                    .map(families::title)
                    .unwrap_or("Select a test")
                    .into()
            })
    }

    fn saved_test_plan(&self, saved: &SavedTest) -> Result<families::PreparedPlan, String> {
        let mut runs = saved.runs.clone();
        for run in &mut runs {
            if !settings_valid(&run.settings) {
                return Err("Saved settings are invalid. Edit and save this test again.".into());
            }
            validated_run_settings(
                &run.settings,
                &run.settings.dataset_mib.to_string(),
                &run.benchmark_id,
            )?;
            run.physical_cores = self.physical_core_count();
            if run.benchmark_id.starts_with("gpu.") {
                run.gpu_id = self.selected_gpu_id.clone();
            }
        }
        self.prepare_set_plan(runs, &saved.id)
    }

    pub(super) fn test_set_coverage<'a>(
        &self,
        runs: impl IntoIterator<Item = &'a QueuedRun>,
    ) -> String {
        let mut count = 0;
        let mut types = Vec::new();
        for run in runs {
            count += 1;
            if let Some(mode) = self.benchmarks.iter().find(|b| b.id == run.benchmark_id) {
                let data_type = mode.data_type.trim().to_uppercase();
                if !data_type.is_empty()
                    && data_type != "WORKLOAD-SPECIFIC"
                    && !types.contains(&data_type)
                {
                    types.push(data_type);
                }
            }
        }
        let count = format!("{count} {}", if count == 1 { "test" } else { "tests" });
        if types.is_empty() {
            count
        } else {
            format!("{} \u{b7} {count}", types.join(", "))
        }
    }

    pub(super) fn saved_test_rows(&self) -> Vec<TemplateRow> {
        self.benchmark_preferences
            .saved_tests
            .iter()
            .filter(|test| test.category == self.selected_suite)
            .map(|test| {
                let plan = self.saved_test_plan(test);
                let mut workloads = Vec::new();
                for run in &test.runs {
                    if let Some(mode) = self.benchmarks.iter().find(|b| b.id == run.benchmark_id) {
                        let title = families::title(self.family_of(mode));
                        if !workloads.contains(&title) {
                            workloads.push(title);
                        }
                    }
                }
                TemplateRow {
                    id: test.id.as_str().into(),
                    title: test.name.as_str().into(),
                    detail: plan
                        .as_ref()
                        .err()
                        .cloned()
                        .unwrap_or_else(|| {
                            if workloads.is_empty() {
                                "User-created set".into()
                            } else {
                                workloads.join(" / ")
                            }
                        })
                        .into(),
                    group: "Your sets".into(),
                    coverage: self.test_set_coverage(&test.runs).into(),
                    curated: false,
                    available: plan.is_ok(),
                }
            })
            .collect()
    }

    fn commit_saved_tests(&mut self, tests: Vec<SavedTest>) -> Result<(), String> {
        if !self.benchmark_preferences_writable {
            return Err("Benchmark settings are read-only. This set could not be saved.".into());
        }
        let mut preferences = self.snapshot_benchmark_preferences();
        preferences.saved_tests = tests;
        write_json(&self.benchmark_preferences_path, &preferences)?;
        self.benchmark_preferences = preferences;
        Ok(())
    }

    fn load_set_test(&mut self, run: &QueuedRun) {
        self.selected = Some(run.benchmark_id.clone());
        if let Some(mode) = self.benchmarks.iter().find(|b| b.id == run.benchmark_id) {
            self.family_modes
                .insert(self.family_of(mode).into(), run.benchmark_id.clone());
        }
        self.run_overrides.insert(
            run.benchmark_id.clone(),
            TestOverride {
                settings: run.settings.clone(),
                dataset_text: run.settings.dataset_mib.to_string(),
            },
        );
        if configurable_compute(&run.benchmark_id) {
            self.vector_choices.insert(run.benchmark_id.clone(), 1);
        }
        self.presentation.editor_shorter_run = Some(run.settings.shorter_run);
    }

    pub(super) fn set_test_rows(&self) -> Vec<ConditionRow> {
        let runs = if self.presentation.curated_set.is_some() {
            self.selected_curated_plan()
                .map(|p| Vec::from(p.runs))
                .unwrap_or_default()
        } else if self.presentation.set_selected {
            self.custom_plan.clone()
        } else {
            Vec::new()
        };
        runs.into_iter()
            .enumerate()
            .map(|(index, run)| {
                let mut row = self
                    .prepare_set_plan(vec![run.clone()], "preview")
                    .ok()
                    .map(|plan| presentation::condition_row(&plan.suite.conditions[0], index, self))
                    .unwrap_or_else(|| ConditionRow {
                        index: index as i32,
                        title: self
                            .benchmarks
                            .iter()
                            .find(|b| b.id == run.benchmark_id)
                            .map(families::mode_name)
                            .unwrap_or(run.benchmark_id.clone())
                            .into(),
                        mode: "Unavailable on this device".into(),
                        ..Default::default()
                    });
                let mut details = vec![
                    DeviceFact {
                        label: "Workload".into(),
                        value: row.title.clone(),
                    },
                    DeviceFact {
                        label: "Dataset".into(),
                        value: row.dataset.clone(),
                    },
                    DeviceFact {
                        label: "Sampling".into(),
                        value: if run.settings.shorter_run {
                            "Shorter run"
                        } else {
                            "Extensive"
                        }
                        .into(),
                    },
                ];
                if let Ok(plan) = self.prepare_set_plan(vec![run], "settings") {
                    let condition = &plan.suite.conditions[0].condition;
                    for (key, label) in [
                        ("cpu_core_limit", "Physical cores"),
                        ("access_pattern", "Access pattern"),
                        ("access_realizations", "Random realizations"),
                        ("access_seed", "Random seed"),
                        ("matrix_shape", "Matrix shape"),
                        ("matrix_tile_shape", "Matrix tile"),
                        ("matrix_reuse", "Matrix reuse"),
                        ("matrix_k_block", "Matrix block"),
                        ("ram_offload_percent", "System RAM share"),
                    ] {
                        if let Some(value) = condition.options.get(key) {
                            details.push(DeviceFact {
                                label: label.into(),
                                value: value.as_str().into(),
                            });
                        }
                    }
                }
                row.details = ModelRc::new(VecModel::from(details));
                row.expanded = self.presentation.inspecting_test == Some(index);
                row
            })
            .collect()
    }

    pub(super) fn saved_test_action(&mut self, action: &str, argument: &str) {
        if self.active_request.is_some() {
            return;
        }
        // Curated definitions can only be selected, inspected or run, even via callbacks.
        if self.presentation.curated_set.is_some()
            && !matches!(action, "new" | "select" | "run" | "inspect")
        {
            return;
        }
        self.presentation.saved_test_status.clear();
        match action {
            "new" => {
                self.presentation.curated_set = None;
                self.presentation.set_selected = true;
                self.presentation.editing_saved_test = None;
                self.presentation.editor_test_name = Some("Untitled set".into());
                self.presentation.test_editor = false;
                self.presentation.editing_test = None;
                self.presentation.inspecting_test = None;
                self.presentation.scope = 1;
                self.presentation.single_test = false;
                self.presentation.clear_test_selection();
                self.custom_plan.clear();
            }
            "inspect" => {
                let index = argument.parse::<usize>().ok();
                self.presentation.inspecting_test = if self.presentation.inspecting_test == index {
                    None
                } else {
                    index
                };
            }
            "rename" if self.presentation.set_selected => {
                self.presentation.editor_test_name = Some(argument.into())
            }
            "add" if self.presentation.set_selected => {
                // A fresh draft starts with defaults, never a hidden copy of a curated test.
                let modes: Vec<String> = self
                    .visible_families()
                    .iter()
                    .flat_map(|f| self.runnable_modes(f))
                    .map(|b| b.id.clone())
                    .collect();
                for id in &modes {
                    let mut settings = AppSettings::default();
                    if configurable_compute(id) {
                        let presets = self.supported_presets(id, &settings);
                        settings.vector_preset = presets
                            .iter()
                            .find(|(id, _)| *id == 1)
                            .or_else(|| presets.iter().find(|(id, _)| *id == 6))
                            .or_else(|| presets.first())
                            .map(|(id, _)| *id)
                            .unwrap_or(6);
                    }
                    self.run_overrides.insert(
                        id.clone(),
                        TestOverride {
                            dataset_text: settings.dataset_mib.to_string(),
                            settings,
                        },
                    );
                    if configurable_compute(id) {
                        self.vector_choices.insert(id.clone(), 1);
                    }
                }
                if let Some(id) = modes.first() {
                    let run = QueuedRun {
                        benchmark_id: id.clone(),
                        settings: self.run_overrides[id].settings.clone(),
                        gpu_id: self.selected_gpu_id.clone(),
                        physical_cores: self.physical_core_count(),
                    };
                    self.load_set_test(&run);
                    self.presentation.test_editor = true;
                    self.presentation.editing_test = None;
                }
            }
            "edit" if self.presentation.set_selected => {
                if let Some((index, run)) = argument
                    .parse::<usize>()
                    .ok()
                    .and_then(|i| self.custom_plan.get(i).cloned().map(|r| (i, r)))
                {
                    if !settings_valid(&run.settings) {
                        self.presentation.saved_test_status =
                            "Invalid settings. Remove this test and add it again.".into();
                    } else {
                        self.load_set_test(&run);
                        self.presentation.test_editor = true;
                        self.presentation.editing_test = Some(index);
                    }
                }
            }
            "cancel" => {
                self.presentation.test_editor = false;
                self.presentation.editing_test = None;
            }
            "apply" if self.presentation.test_editor => {
                let outcome = self
                    .selected
                    .as_deref()
                    .ok_or("Choose a workload".into())
                    .and_then(|id| self.queued_run(id))
                    .and_then(|run| {
                        self.prepare_set_plan(vec![run.clone()], "test")?;
                        Ok(run)
                    });
                match outcome {
                    Ok(run) => {
                        if let Some(index) = self.presentation.editing_test {
                            if let Some(existing) = self.custom_plan.get_mut(index) {
                                *existing = run;
                            }
                        } else {
                            self.custom_plan.push(run);
                        }
                        self.presentation.test_editor = false;
                        self.presentation.editing_test = None;
                    }
                    Err(error) => self.presentation.saved_test_status = error,
                }
            }
            "remove" if self.presentation.set_selected && !self.presentation.test_editor => {
                if let Ok(index) = argument.parse::<usize>() {
                    if index < self.custom_plan.len() {
                        self.custom_plan.remove(index);
                    }
                    self.presentation.inspecting_test = None;
                }
            }
            "save" if self.presentation.set_selected && !self.presentation.test_editor => {
                let name = self.editor_test_name().trim().to_owned();
                let outcome = (|| {
                    if name.is_empty()
                        || name.chars().count() > 100
                        || name.chars().any(char::is_control)
                    {
                        return Err("Use a set name between 1 and 100 characters.".into());
                    }
                    self.prepare_set_plan(self.custom_plan.clone(), "validate")?;
                    let id = self
                        .presentation
                        .editing_saved_test
                        .clone()
                        .unwrap_or_else(|| {
                            format!(
                                "user:{:x}-{}",
                                SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_nanos(),
                                self.next_id("saved")
                            )
                        });
                    let test = SavedTest {
                        id: id.clone(),
                        name: name.clone(),
                        category: self.selected_suite.clone(),
                        runs: self.custom_plan.clone(),
                    };
                    let mut tests = self.benchmark_preferences.saved_tests.clone();
                    if let Some(existing) = tests.iter_mut().find(|t| t.id == id) {
                        *existing = test;
                    } else {
                        tests.push(test);
                    }
                    self.commit_saved_tests(tests)?;
                    self.presentation.editing_saved_test = Some(id);
                    self.presentation.editor_test_name = Some(name);
                    Ok::<_, String>("Set saved".into())
                })();
                self.presentation.saved_test_status = outcome.unwrap_or_else(|e| e);
            }
            "delete" => {
                if let Some(id) = self.presentation.editing_saved_test.clone() {
                    let mut tests = self.benchmark_preferences.saved_tests.clone();
                    tests.retain(|t| t.id != id);
                    match self.commit_saved_tests(tests) {
                        Ok(()) => {
                            self.presentation.editing_saved_test = None;
                            self.presentation.set_selected = false;
                            self.presentation.editor_test_name = None;
                            self.custom_plan.clear();
                        }
                        Err(error) => self.presentation.saved_test_status = error,
                    }
                }
            }
            "run" | "select" => {
                if let Some(test) = self
                    .benchmark_preferences
                    .saved_tests
                    .iter()
                    .find(|t| t.id == argument && t.category == self.selected_suite)
                    .cloned()
                {
                    if action == "run" {
                        self.begin_plan(self.saved_test_plan(&test));
                    } else {
                        self.presentation.curated_set = None;
                        self.presentation.set_selected = true;
                        self.presentation.test_editor = false;
                        self.presentation.editing_test = None;
                        self.presentation.inspecting_test = None;
                        self.presentation.editing_saved_test = Some(test.id);
                        self.presentation.editor_test_name = Some(test.name);
                        self.custom_plan = test.runs;
                        let cores = self.physical_core_count();
                        for run in &mut self.custom_plan {
                            run.physical_cores = cores;
                            if run.benchmark_id.starts_with("gpu.") {
                                run.gpu_id = self.selected_gpu_id.clone();
                            }
                        }
                        self.presentation.scope = 1;
                        self.presentation.single_test = false;
                        self.presentation.clear_test_selection();
                    }
                }
            }
            _ => return,
        }
        self.ui_dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::tests::catalogue;

    fn editor() -> App {
        let mut app = catalogue();
        app.selected_suite = "gpu.performance".into();
        app.select_family("gpu.vector");
        app.selected = Some(GPU_VECTOR_ID.into());
        app.presentation = presentation::State::builder();
        app.benchmark_preferences_writable = true;
        app.benchmark_preferences_path = std::env::temp_dir().join(format!(
            "gluj-saved-tests-{}-{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        app.saved_test_action("new", "");
        app.saved_test_action("add", "");
        app.saved_test_action("apply", "");
        assert_eq!(
            app.custom_plan.len(),
            1,
            "{}",
            app.presentation.saved_test_status
        );
        app
    }

    #[test]
    fn saved_test_create_restore_update_and_delete_preserve_curated_definitions() {
        let mut app = editor();
        app.saved_test_action("rename", "My GPU sweep");
        app.saved_test_action("save", "");
        assert_eq!(app.presentation.saved_test_status, "Set saved");
        let id = app.presentation.editing_saved_test.clone().unwrap();
        let original = app.benchmark_preferences.saved_tests[0].clone();
        let preferences = load_benchmark_preferences(&app.benchmark_preferences_path).unwrap();
        let mut restored = editor();
        restored.restore_benchmark_preferences(preferences);
        restored.saved_test_action("select", &id);
        assert_eq!(restored.editor_test_name(), "My GPU sweep");
        assert_eq!(restored.custom_plan, original.runs);
        assert_eq!(restored.saved_test_rows().len(), 1);
        restored.saved_test_action("rename", "Updated GPU sweep");
        restored.saved_test_action("save", "");
        assert_eq!(restored.benchmark_preferences.saved_tests.len(), 1);
        assert_eq!(restored.benchmark_preferences.saved_tests[0].id, id);
        restored.selected_suite = "cpu.performance".into();
        assert!(restored.saved_test_rows().is_empty());
        restored.selected_suite = "gpu.performance".into();
        restored.saved_test_action("delete", "");
        assert!(restored.benchmark_preferences.saved_tests.is_empty());
        assert!(restored.presentation.editing_saved_test.is_none());
        // Selecting curated sets never converts them into user-created sets.
        app.template_action("select", "gpu.vector.scaling@1");
        assert!(app.presentation.editing_saved_test.is_none());
        app.saved_test_action("rename", "Another copy");
        app.saved_test_action("save", "");
        assert_eq!(app.benchmark_preferences.saved_tests.len(), 1);
        fs::remove_file(&app.benchmark_preferences_path).unwrap();
        fs::remove_file(&restored.benchmark_preferences_path).unwrap();
    }

    #[test]
    fn failed_or_invalid_save_retains_previous_named_test() {
        let mut app = editor();
        app.saved_test_action("rename", "Working test");
        app.saved_test_action("save", "");
        let saved = app.benchmark_preferences.saved_tests.clone();
        app.saved_test_action("rename", "   ");
        app.saved_test_action("save", "");
        assert_eq!(app.benchmark_preferences.saved_tests, saved);
        app.saved_test_action("rename", "Unsaved rename");
        app.benchmark_preferences_writable = false;
        app.saved_test_action("save", "");
        assert_eq!(app.benchmark_preferences.saved_tests, saved);
        assert!(app.presentation.saved_test_status.contains("read-only"));
        app.benchmark_preferences_writable = true;
        let path = app.benchmark_preferences_path.clone();
        app.benchmark_preferences_path = path.join("unwritable.json");
        app.saved_test_action("save", "");
        assert_eq!(app.benchmark_preferences.saved_tests, saved);
        assert_ne!(app.presentation.saved_test_status, "Set saved");
        app.benchmark_preferences_path = path;
        fs::remove_file(&app.benchmark_preferences_path).unwrap();
    }

    #[test]
    fn old_preferences_load_without_saved_tests() {
        let preferences: BenchmarkPreferences =
            serde_json::from_str(r#"{"version":1,"tests":{}}"#).unwrap();
        assert!(preferences.saved_tests.is_empty());
    }

    #[test]
    fn selecting_and_renaming_preserve_individual_measurement_policies() {
        let mut app = editor();
        app.custom_plan[0].benchmark_id = GPU_VECTOR_ID.into();
        app.custom_plan[0].settings.shorter_run = true;
        let mut second = app.custom_plan[0].clone();
        second.benchmark_id = "gpu.performance.fp16".into();
        second.settings.shorter_run = false;
        app.custom_plan.push(second);
        app.saved_test_action("rename", "Mixed policy set");
        app.saved_test_action("save", "");
        assert_eq!(app.presentation.saved_test_status, "Set saved");
        let id = app.presentation.editing_saved_test.clone().unwrap();
        let original = app.custom_plan.clone();
        app.benchmarks
            .iter_mut()
            .find(|b| b.id == original[0].benchmark_id)
            .unwrap()
            .data_type = "fp32".into();
        app.benchmarks
            .iter_mut()
            .find(|b| b.id == original[1].benchmark_id)
            .unwrap()
            .data_type = "fp16".into();
        let row = &app.saved_test_rows()[0];
        assert_eq!(row.coverage, "FP32, FP16 \u{b7} 2 tests");
        assert_eq!(row.detail, "GPU vector compute");
        app.settings.shorter_run = true;
        app.saved_test_action("select", &id);
        app.saved_test_action("rename", "Renamed set");
        app.saved_test_action("save", "");
        let saved = &app.benchmark_preferences.saved_tests[0];
        assert_eq!(saved.runs, original);
        let plan = app.saved_test_plan(saved).unwrap();
        assert!(plan.runs[0].settings.shorter_run);
        assert!(!plan.runs[1].settings.shorter_run);
        assert_ne!(
            plan.suite.conditions[0].condition.policy.samples,
            plan.suite.conditions[1].condition.policy.samples
        );
        fs::remove_file(&app.benchmark_preferences_path).unwrap();
    }

    #[test]
    fn draft_add_edit_cancel_and_remove_are_explicit_and_independent() {
        let mut app = editor();
        let original = app.custom_plan.clone();
        app.saved_test_action("add", "");
        assert!(app.presentation.test_editor);
        app.builder_choice("workload", 1);
        app.saved_test_action("cancel", "");
        assert_eq!(app.custom_plan, original);
        app.saved_test_action("edit", "0");
        assert_eq!(
            app.selected.as_deref(),
            Some(original[0].benchmark_id.as_str())
        );
        app.edit_configuration(false, "measurement", 1);
        app.saved_test_action("cancel", "");
        assert_eq!(app.custom_plan, original);
        app.saved_test_action("edit", "0");
        app.edit_configuration(false, "measurement", 1);
        app.saved_test_action("apply", "");
        assert!(app.custom_plan[0].settings.shorter_run);
        assert!(!app.settings.shorter_run);
        app.saved_test_action("add", "");
        app.builder_choice("workload", 1);
        assert_eq!(app.selected_family(), Some("gpu.matrix"));
        app.saved_test_action("apply", "");
        assert_eq!(app.custom_plan.len(), 2);
        assert_eq!(app.custom_plan[0].benchmark_id, original[0].benchmark_id);
        app.saved_test_action("remove", "0");
        assert_eq!(app.custom_plan.len(), 1);
        assert!(app.custom_plan[0].benchmark_id.contains("matrix"));
    }

    #[test]
    fn blank_sets_cannot_save_and_invalid_edits_do_not_replace_tests() {
        let mut app = editor();
        app.saved_test_action("new", "");
        app.saved_test_action("save", "");
        assert!(app.benchmark_preferences.saved_tests.is_empty());
        app.saved_test_action("add", "");
        app.saved_test_action("apply", "");
        let original = app.custom_plan.clone();
        app.saved_test_action("edit", "0");
        app.selected = None;
        app.saved_test_action("apply", "");
        assert!(app.presentation.test_editor);
        assert_eq!(app.custom_plan, original);
        app.saved_test_action("cancel", "");
        app.saved_test_action("edit", "999");
        assert!(!app.presentation.test_editor);
    }
    #[test]
    fn invalid_saved_settings_can_be_inspected_and_removed_without_panicking() {
        let mut app = editor();
        let mut invalid = app.custom_plan[0].clone();
        invalid.settings.gpu_intensity = usize::MAX;
        app.benchmark_preferences.saved_tests.push(SavedTest {
            id: "user:invalid".into(),
            name: "Invalid old set".into(),
            category: app.selected_suite.clone(),
            runs: vec![invalid],
        });
        app.saved_test_action("select", "user:invalid");
        assert_eq!(app.custom_plan.len(), 1);
        assert!(!app.saved_test_rows()[0].available);
        assert_eq!(app.set_test_rows().len(), 1);
        app.saved_test_action("edit", "0");
        assert!(!app.presentation.test_editor);
        app.saved_test_action("remove", "0");
        app.saved_test_action("add", "");
        app.saved_test_action("apply", "");
        app.saved_test_action("save", "");
        assert_eq!(app.presentation.saved_test_status, "Set saved");
        fs::remove_file(&app.benchmark_preferences_path).unwrap();
    }
}
