use super::*;
use families::PreparedPlan;

// Bump when template defaults, coverage, or measurement policy changes.
const TEMPLATE_VERSION: u32 = 1;
#[derive(Clone, Copy)]
enum Coverage {
    All,
    Peak,
    Scaling,
    Offload,
}

struct Template {
    id: String,
    family: String,
    title: String,
    detail: &'static str,
    coverage: Coverage,
}

impl App {
    fn builder_workload_families(&self) -> Vec<String> {
        self.visible_families()
            .into_iter()
            .filter(|family| !self.runnable_modes(family).is_empty())
            .collect()
    }
    fn setup_templates(&self) -> Vec<Template> {
        let mut templates = Vec::new();
        for family in self.visible_families() {
            if self.runnable_modes(&family).is_empty() {
                continue;
            }
            let choices = match family.as_str() {
                "gpu.vector" => vec![
                    (
                        "peak",
                        "Vector peak throughput",
                        "Register arithmetic across supported precisions",
                        Coverage::Peak,
                    ),
                    (
                        "scaling",
                        "Vector dataset scaling",
                        "Throughput as GPU datasets grow",
                        Coverage::Scaling,
                    ),
                    (
                        "offload",
                        "System RAM offload",
                        "FP32 with 75% of data in system RAM",
                        Coverage::Offload,
                    ),
                ],
                "gpu.matrix" => vec![
                    (
                        "peak",
                        "Matrix peak throughput",
                        "Register matrix operations across supported precisions",
                        Coverage::Peak,
                    ),
                    (
                        "scaling",
                        "Matrix dataset scaling",
                        "Matrix throughput as operand datasets grow",
                        Coverage::Scaling,
                    ),
                ],
                "cpu.fp32" | "cpu.fp64" | "cpu.int64" => vec![
                    (
                        "peak",
                        "Peak throughput",
                        "Compute throughput with one and all supported cores",
                        Coverage::Peak,
                    ),
                    (
                        "scaling",
                        "Dataset scaling",
                        "Throughput across growing working sets",
                        Coverage::Scaling,
                    ),
                ],
                _ => {
                    let detail = match family.as_str() {
                        "cpu.memory_bandwidth" => "Read, write and copy across L1, L2, L3 and RAM",
                        "cpu.dependent_latency" => "Dependent reads across caches and system RAM",
                        "cpu.matrix" => "FP32 matrix products across shapes and dataset sizes",
                        "cpu.ascii" => "Character scans across growing input sizes",
                        "cpu.deflate" => "Independent-block compression and decompression",
                        "cpu.sieve" => "Prime sieve throughput across supported core counts",
                        "cpu.popcnt" => "Population count across growing datasets",
                        "cpu.aes_rounds" => "AES round throughput across supported core counts",
                        "gpu.local_bandwidth" => {
                            "Cache-oriented and streaming read, write and copy"
                        }
                        "gpu.host_transfers" => "Host-to-GPU and GPU-to-host transfer throughput",
                        _ => "Standard coverage for this workload",
                    };
                    vec![("standard", families::title(&family), detail, Coverage::All)]
                }
            };
            for (suffix, title, detail, coverage) in choices {
                let title = if family.starts_with("cpu.") && !matches!(coverage, Coverage::All) {
                    format!("{} · {title}", families::title(&family))
                } else {
                    title.into()
                };
                templates.push(Template {
                    id: format!("{family}.{suffix}@{TEMPLATE_VERSION}"),
                    family: family.clone(),
                    title,
                    detail,
                    coverage,
                });
            }
        }
        templates
    }

    fn setup_template_plan(&self, template: &Template) -> Result<PreparedPlan, String> {
        let base =
            self.curated_plan_with_settings(Some(&template.family), &AppSettings::default())?;
        let runs = base
            .runs
            .into_iter()
            .filter(|run| {
                let offload = gpu_offload(&run.benchmark_id);
                match template.coverage {
                    Coverage::All => true,
                    Coverage::Peak => {
                        !offload
                            && (!configurable_compute(&run.benchmark_id)
                                || run.settings.vector_preset == 1)
                    }
                    Coverage::Scaling => {
                        !offload
                            && configurable_compute(&run.benchmark_id)
                            && run.settings.vector_preset != 1
                    }
                    Coverage::Offload => offload,
                }
            })
            .collect();
        self.prepare_plan_with_measurement(runs, &format!("template:{}", template.id), false)
    }

    pub(super) fn template_action(&mut self, action: &str, id: &str) {
        if self.active_request.is_some() {
            return;
        }
        let Some(template) = self
            .setup_templates()
            .into_iter()
            .find(|template| template.id == id)
        else {
            return;
        };
        let plan = self.setup_template_plan(&template);
        if action == "run" {
            self.begin_plan(plan);
        } else if action == "customize" {
            match plan {
                Ok(plan) => {
                    let runs: Vec<_> = plan.runs.into();
                    let first = runs.first().unwrap();
                    self.selected = Some(first.benchmark_id.clone());
                    self.family_modes
                        .insert(template.family, first.benchmark_id.clone());
                    self.run_overrides.insert(
                        first.benchmark_id.clone(),
                        TestOverride {
                            settings: first.settings.clone(),
                            dataset_text: first.settings.dataset_mib.to_string(),
                        },
                    );
                    if configurable_compute(&first.benchmark_id) {
                        self.vector_choices.insert(first.benchmark_id.clone(), 1);
                    }
                    self.presentation.scope = 1;
                    self.presentation.single_test = runs.len() == 1;
                    self.presentation.clear_test_selection();
                    if runs.len() > 1 {
                        self.custom_plan = runs;
                    }
                    self.persist_benchmark_preferences();
                    self.status.clear();
                }
                Err(error) => self.status = error,
            }
        }
        self.ui_dirty = true;
    }

    fn builder_gpu_precisions(&self) -> Vec<&BenchmarkDescriptor> {
        self.selected_family()
            .map(|family| {
                let modes = self.runnable_modes(family);
                let mut precisions: Vec<_> = modes
                    .iter()
                    .copied()
                    .filter(|mode| !gpu_offload(&mode.id))
                    .collect();
                if !precisions.iter().any(|mode| mode.data_type == "fp32")
                    && let Some(offload) = modes.iter().find(|mode| gpu_offload(&mode.id))
                {
                    precisions.push(*offload);
                }
                precisions
            })
            .unwrap_or_default()
    }

    fn builder_gpu_paths(&self) -> Vec<(i32, &'static str)> {
        let selected = self.selected.as_deref().unwrap_or("");
        let id = if gpu_offload(selected) {
            GPU_VECTOR_ID
        } else {
            selected
        };
        let settings = self
            .effective_settings(id)
            .unwrap_or_else(|_| self.settings.clone());
        let presets = self.supported_presets(id, &settings);
        let mut paths = Vec::new();
        if presets.iter().any(|(preset, _)| *preset == 1) {
            paths.push((0, "Registers"));
        }
        if presets.iter().any(|(preset, _)| *preset == 6) {
            paths.push((1, "GPU memory"));
        }
        if id == GPU_VECTOR_ID
            && self
                .benchmarks
                .iter()
                .any(|mode| mode.id == GPU_OFFLOAD_ID && self.benchmark_available(mode))
            && self
                .supported_presets(GPU_OFFLOAD_ID, &settings)
                .iter()
                .any(|(preset, _)| *preset == 6)
        {
            paths.push((2, "System RAM"));
        }
        paths
    }

    fn builder_dataset_presets(&self) -> Vec<(usize, String)> {
        let id = self.selected.as_deref().unwrap_or("");
        let settings = self
            .effective_settings(id)
            .unwrap_or_else(|_| self.settings.clone());
        self.supported_presets(id, &settings)
            .into_iter()
            .filter(|(preset, _)| *preset != 1 && (*preset != 0 || settings.vector_preset == 0))
            .collect()
    }

    pub(super) fn builder_choice(&mut self, field: &str, index: i32) {
        if self.active_request.is_some() || index < 0 {
            return;
        }
        let index = index as usize;
        match field {
            "workload" => {
                if let Some(family) = self.builder_workload_families().get(index).cloned() {
                    self.select_family(&family);
                    self.presentation.clear_test_selection();
                }
            }
            "precision" => {
                if let Some(id) = self
                    .builder_gpu_precisions()
                    .get(index)
                    .map(|mode| mode.id.clone())
                {
                    let preserve_offload =
                        id == GPU_VECTOR_ID && self.selected.as_deref().is_some_and(gpu_offload);
                    let id = if preserve_offload {
                        GPU_OFFLOAD_ID.into()
                    } else {
                        id
                    };
                    self.selected = Some(id.clone());
                    if let Some(family) = self.selected_family().map(str::to_owned) {
                        self.family_modes.insert(family, id);
                    }
                    self.persist_benchmark_preferences();
                }
            }
            "placement" => {
                if let Some((path, _)) = self.builder_gpu_paths().get(index).copied() {
                    let current = self.selected.as_deref().unwrap_or("");
                    let id = if path == 2 {
                        GPU_OFFLOAD_ID.into()
                    } else if gpu_offload(current) {
                        GPU_VECTOR_ID.into()
                    } else {
                        current.to_owned()
                    };
                    self.selected = Some(id);
                    self.edit_configuration(true, "vector-preset", if path == 0 { 1 } else { 6 });
                    if let Some(family) = self.selected_family().map(str::to_owned) {
                        self.family_modes
                            .insert(family, self.selected.clone().unwrap());
                    }
                    self.persist_benchmark_preferences();
                }
            }
            "dataset" => {
                if let Some((preset, _)) = self.builder_dataset_presets().get(index) {
                    self.edit_configuration(true, "vector-preset", *preset as i32);
                }
            }
            _ => return,
        }
        self.presentation.scope = 1;
        self.ui_dirty = true;
    }

    pub(super) fn wire_benchmark_setup(window: &MainWindow, app: &Rc<RefCell<Self>>) {
        let weak = Rc::downgrade(app);
        let win = window.as_weak();
        window.on_builder_choice(move |field, index| {
            if let (Some(app), Some(window)) = (weak.upgrade(), win.upgrade()) {
                let mut app = app.borrow_mut();
                app.builder_choice(field.as_str(), index);
                App::refresh_benchmark_panel(&window, &app);
            }
        });
        let weak = Rc::downgrade(app);
        window.on_template_action(move |action, id| {
            if let Some(app) = weak.upgrade() {
                app.borrow_mut()
                    .template_action(action.as_str(), id.as_str());
            }
        });
        let weak = Rc::downgrade(app);
        window.on_batch_builder_changed(move |enabled| {
            if let Some(app) = weak.upgrade() {
                let mut app = app.borrow_mut();
                app.presentation.single_test = !enabled;
                app.presentation.scope = 1;
                app.presentation.clear_test_selection();
                app.ui_dirty = true;
            }
        });
    }

    pub(super) fn refresh_benchmark_setup(window: &MainWindow, app: &Self) {
        let rows = app
            .setup_templates()
            .into_iter()
            .filter_map(|template| {
                let plan = app.setup_template_plan(&template).ok()?;
                let seconds = plan
                    .suite
                    .conditions
                    .iter()
                    .filter_map(|condition| condition.estimated_duration_ms)
                    .sum::<u64>() as f64
                    / 1000.;
                let mut precisions = Vec::new();
                for run in &plan.runs {
                    if let Some(mode) = app
                        .benchmarks
                        .iter()
                        .find(|mode| mode.id == run.benchmark_id)
                    {
                        let precision = mode.data_type.to_uppercase();
                        if !precisions.contains(&precision) {
                            precisions.push(precision);
                        }
                    }
                }
                let group = match template.coverage {
                    Coverage::Offload => "System RAM",
                    _ if template.family == "gpu.vector" => "Vector",
                    _ if template.family == "gpu.matrix" => "Matrix",
                    _ if app.selected_suite.ends_with("bandwidth") => "Memory",
                    Coverage::Peak | Coverage::Scaling => {
                        families::title(&template.family).trim_start_matches("CPU ")
                    }
                    _ => "Workloads",
                };
                let prefix = format!("{} · ", families::title(&template.family));
                let title = template
                    .title
                    .strip_prefix(&prefix)
                    .unwrap_or(&template.title);
                Some(TemplateRow {
                    id: template.id.into(),
                    title: title.into(),
                    detail: template.detail.into(),
                    group: group.into(),
                    duration: format!("≥ {seconds} s").into(),
                    coverage: format!(
                        "{} · {} {}",
                        precisions.join(", "),
                        plan.runs.len(),
                        if plan.runs.len() == 1 {
                            "test"
                        } else {
                            "tests"
                        }
                    )
                    .into(),
                })
            })
            .collect::<Vec<_>>();
        window.set_setup_templates(ModelRc::new(VecModel::from(rows)));
        let families = app.builder_workload_families();
        window.set_builder_workloads(ModelRc::new(VecModel::from(
            families
                .iter()
                .map(|family| {
                    SharedString::from(match family.as_str() {
                        "gpu.vector" => "Vector",
                        "gpu.matrix" => "Matrix",
                        _ => families::title(family),
                    })
                })
                .collect::<Vec<_>>(),
        )));
        window.set_builder_workload_index(
            families
                .iter()
                .position(|family| Some(family.as_str()) == app.selected_family())
                .map(|index| index as i32)
                .unwrap_or(-1),
        );
        let gpu_compute = app.selected_suite == "gpu.performance";
        window.set_builder_gpu_compute(gpu_compute);
        let selected = app.selected.as_deref().unwrap_or("");
        let base = if gpu_offload(selected) {
            GPU_VECTOR_ID
        } else {
            selected
        };
        let precisions = app.builder_gpu_precisions();
        window.set_builder_precisions(ModelRc::new(VecModel::from(
            precisions
                .iter()
                .map(|mode| SharedString::from(mode.data_type.to_uppercase()))
                .collect::<Vec<_>>(),
        )));
        window.set_builder_precision_index(
            precisions
                .iter()
                .position(|mode| {
                    mode.id == base || (gpu_offload(selected) && gpu_offload(&mode.id))
                })
                .map(|index| index as i32)
                .unwrap_or(-1),
        );
        let settings = app
            .effective_settings(selected)
            .unwrap_or_else(|_| app.settings.clone());
        let path = if gpu_offload(selected) {
            2
        } else if settings.vector_preset == 1 {
            0
        } else {
            1
        };
        let paths = app.builder_gpu_paths();
        window.set_builder_placements(ModelRc::new(VecModel::from(
            paths
                .iter()
                .map(|(_, label)| SharedString::from(*label))
                .collect::<Vec<_>>(),
        )));
        window.set_builder_placement_index(
            paths
                .iter()
                .position(|(id, _)| *id == path)
                .map(|index| index as i32)
                .unwrap_or(-1),
        );
        let presets = app.builder_dataset_presets();
        window.set_builder_dataset_presets(ModelRc::new(VecModel::from(
            presets
                .iter()
                .map(|(_, label)| SharedString::from(label))
                .collect::<Vec<_>>(),
        )));
        window.set_builder_dataset_index(
            presets
                .iter()
                .position(|(id, _)| *id == settings.vector_preset)
                .map(|index| index as i32)
                .unwrap_or(-1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::tests::catalogue;
    use slint::Model;

    #[test]
    #[ignore = "headless gallery and builder render check, run separately"]
    fn headless_gallery_and_builder_callbacks_render_at_supported_window_sizes() {
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
        let mut state = catalogue();
        state.presentation = presentation::State::builder();
        state.selected_suite = "gpu.performance".into();
        state.select_family("gpu.vector");
        let app = Rc::new(RefCell::new(state));
        App::wire(&window, &app);
        App::wire_presentation(&window, &app);
        App::wire_benchmark_setup(&window, &app);
        App::refresh(&window, &app.borrow());
        assert!(window.get_setup_templates().row_count() > 0);
        assert_eq!(window.get_builder_workloads().row_count(), 2);
        assert!(!window.get_batch_builder());
        assert_eq!(
            window.get_test_vector_register(),
            app.borrow()
                .effective_settings(app.borrow().selected.as_deref().unwrap())
                .unwrap()
                .vector_preset
                == 1
        );
        window.show().unwrap();
        for (width, height, panel) in [(1280, 850, 0), (980, 680, 0), (980, 680, 1)] {
            adapter.set_size(slint::PhysicalSize::new(width, height));
            window.set_custom_panel(panel);
            window.window().request_redraw();
            let mut pixels = vec![slint::Rgb8Pixel::default(); width as usize * height as usize];
            assert!(adapter.draw_if_needed(|renderer| {
                renderer.render(&mut pixels, width as usize);
            }));
            assert!(
                pixels
                    .iter()
                    .any(|pixel| pixel.r != 0 || pixel.g != 0 || pixel.b != 0)
            );
            if let Some(directory) = std::env::var_os("GLUJ_BENCH_UI_PREVIEW_DIR") {
                let directory = PathBuf::from(directory);
                fs::create_dir_all(&directory).unwrap();
                let file =
                    fs::File::create(directory.join(format!("benchmarks-{width}-{panel}.png")))
                        .unwrap();
                let mut encoder = png::Encoder::new(file, width, height);
                encoder.set_color(png::ColorType::Rgb);
                let mut writer = encoder.write_header().unwrap();
                let bytes: Vec<_> = pixels
                    .iter()
                    .flat_map(|pixel| [pixel.r, pixel.g, pixel.b])
                    .collect();
                writer.write_image_data(&bytes).unwrap();
            }
        }
        window.invoke_template_action("customize".into(), "gpu.vector.scaling@1".into());
        App::refresh(&window, &app.borrow());
        assert_eq!(
            window.get_batch_builder(),
            app.borrow().custom_plan.len() > 1
        );
        window.invoke_batch_builder_changed(false);
        App::refresh(&window, &app.borrow());
        assert!(!window.get_batch_builder());
        window.invoke_select_suite("cpu.performance".into());
        assert!(!window.get_builder_gpu_compute());
        assert!(window.get_builder_workloads().row_count() > 2);
        window.hide().unwrap();
    }

    #[test]
    fn template_plans_are_independent_of_custom_settings_and_measurement_policy() {
        let mut app = catalogue();
        for suite in [
            "cpu.performance",
            "cpu.bandwidth",
            "gpu.performance",
            "gpu.bandwidth",
        ] {
            app.selected_suite = suite.into();
            for template in app.setup_templates() {
                let Ok(before) = app.setup_template_plan(&template) else {
                    continue;
                };
                app.settings.shorter_run = true;
                app.settings.cpu_intensity = 0;
                app.settings.gpu_intensity = 0;
                app.settings.access_pattern = 1;
                app.settings.access_fresh = true;
                app.settings.access_ensemble = true;
                app.settings.ram_budget_percent = 80;
                app.settings.vram_budget_percent = 80;
                let after = app.setup_template_plan(&template).unwrap();
                assert_eq!(before.runs, after.runs, "{}", template.id);
                assert_eq!(before.suite, after.suite);
                assert!(after.runs.iter().all(|run| !run.settings.shorter_run));
                assert!(after.suite.id.ends_with("@1"));
            }
        }
    }

    #[test]
    fn template_customization_copies_a_frozen_batch_and_preserves_global_defaults() {
        let mut app = catalogue();
        app.selected_suite = "gpu.performance".into();
        let defaults = app.settings.clone();
        let template = app
            .setup_templates()
            .into_iter()
            .find(|template| template.id == "gpu.vector.scaling@1")
            .unwrap();
        let before = app.setup_template_plan(&template).unwrap();
        app.template_action("customize", &template.id);
        assert_eq!(app.settings, defaults);
        assert_eq!(app.presentation.single_test, before.runs.len() == 1);
        if before.runs.len() > 1 {
            assert_eq!(app.custom_plan, Vec::from(before.runs));
            let saved = app.custom_plan.clone();
            app.edit_configuration(true, "gpu", 0);
            assert_eq!(app.custom_plan, saved);
        }
        assert_eq!(app.presentation.scope, 1);
    }

    #[test]
    fn gpu_builder_paths_and_precisions_follow_the_selected_workload() {
        let mut app = catalogue();
        app.selected_suite = "gpu.performance".into();
        app.select_family("gpu.vector");
        app.selected = Some(GPU_VECTOR_ID.into());
        let paths = app.builder_gpu_paths();
        let ram = paths.iter().position(|(path, _)| *path == 2).unwrap();
        app.builder_choice("placement", ram as i32);
        assert_eq!(app.selected.as_deref(), Some(GPU_OFFLOAD_ID));
        let precisions = app.builder_gpu_precisions();
        let fp16 = precisions
            .iter()
            .position(|mode| mode.id == "gpu.performance.fp16")
            .unwrap();
        app.builder_choice("precision", fp16 as i32);
        assert_eq!(app.selected.as_deref(), Some("gpu.performance.fp16"));
        assert!(!app.builder_gpu_paths().iter().any(|(path, _)| *path == 2));
        app.select_family("gpu.matrix");
        assert!(!app.builder_gpu_paths().iter().any(|(path, _)| *path == 2));
    }

    #[test]
    fn gallery_covers_all_runnable_families_in_each_category() {
        let mut app = catalogue();
        for suite in [
            "cpu.performance",
            "cpu.bandwidth",
            "gpu.performance",
            "gpu.bandwidth",
        ] {
            app.selected_suite = suite.into();
            let templates = app.setup_templates();
            for family in app.visible_families() {
                if !app.runnable_modes(&family).is_empty() {
                    assert!(
                        templates.iter().any(|template| template.family == family
                            && app.setup_template_plan(template).is_ok()),
                        "{family}"
                    );
                }
            }
        }
    }
}
