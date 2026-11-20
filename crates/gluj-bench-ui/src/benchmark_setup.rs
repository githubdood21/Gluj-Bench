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
        if id.starts_with("user:") {
            self.saved_test_action(
                if action == "customize" {
                    "select"
                } else {
                    action
                },
                id,
            );
            return;
        }
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
        } else if action == "select" || action == "customize" {
            // Older callback names also select read-only; they never create an editable copy.
            match plan {
                Ok(_) => {
                    self.presentation.curated_set = Some(template.id);
                    self.presentation.set_selected = true;
                    self.presentation.test_editor = false;
                    self.presentation.editing_test = None;
                    self.presentation.inspecting_test = None;
                    self.presentation.editing_saved_test = None;
                    self.presentation.editor_test_name = Some(template.title);
                    self.presentation.saved_test_status.clear();
                }
                Err(error) => self.status = error,
            }
        }
        self.ui_dirty = true;
    }

    pub(super) fn selected_curated_plan(&self) -> Result<PreparedPlan, String> {
        let template = self
            .setup_templates()
            .into_iter()
            .find(|template| Some(&template.id) == self.presentation.curated_set.as_ref())
            .ok_or("Select a set")?;
        self.setup_template_plan(&template)
    }

    fn missing_mode_label(mode: &BenchmarkDescriptor) -> String {
        let id = mode.id.as_str();
        if id.contains(".avx2.") {
            return "AVX2/FMA".into();
        }
        if id.contains(".avx512.") {
            return "AVX-512".into();
        }
        if id.ends_with(".aes") {
            return "AES".into();
        }
        if id.ends_with(".popcnt") {
            return "POPCNT".into();
        }
        if mode.category == BenchmarkCategory::Gpu {
            let format = if id.ends_with(".fp8") {
                "FP8".into()
            } else if id.contains(".fp16") {
                "FP16".into()
            } else if id.contains(".fp64") {
                "FP64".into()
            } else if id.contains(".int8") {
                "INT8".into()
            } else {
                mode.data_type.to_uppercase()
            };
            return if id.contains(".sparse.") {
                format!("Sparse {format}")
            } else {
                format
            };
        }
        families::mode_name(mode)
            .split(" | ")
            .next()
            .unwrap_or(&mode.name)
            .into()
    }

    fn gpu_format_unsupported(&self, mode: &BenchmarkDescriptor) -> bool {
        // App placeholders do not describe GPU capabilities. These formats are not probed yet.
        if mode.id.contains(".sparse.") || mode.id.ends_with(".fp8") {
            return false;
        }
        let matrix = self.family_of(mode) == "gpu.matrix";
        let key = match (matrix, Self::missing_mode_label(mode).as_str()) {
            (true, "FP16") => "cooperative_matrix_fp16",
            (true, "INT8") => "cooperative_matrix_int8",
            (false, "FP16") => "shader_f16",
            (false, "FP64") => "shader_f64",
            _ => return false,
        };
        self.devices
            .iter()
            .find(|gpu| Some(&gpu.id) == self.selected_gpu_id.as_ref())
            .and_then(|gpu| gpu.properties.get(key))
            .is_some_and(|value| value == "false")
    }

    fn cpu_mode_unsupported(&self, mode: &BenchmarkDescriptor) -> bool {
        !self.benchmark_available(mode)
            && matches!(
                mode.unavailable_reason.as_str(),
                "avx2_fma_not_supported"
                    | "avx512_not_supported"
                    | "aes_not_supported"
                    | "popcnt_not_supported"
                    | "x86_64_instruction_set_required"
            )
    }

    fn unsupported_list<'a>(
        &self,
        modes: impl IntoIterator<Item = &'a BenchmarkDescriptor>,
    ) -> String {
        let mut labels = Vec::new();
        for mode in modes {
            let label = Self::missing_mode_label(mode);
            if !labels.contains(&label) {
                labels.push(label);
            }
        }
        if labels.is_empty() {
            String::new()
        } else {
            format!("Unsupported:\n{}", labels.join("\n"))
        }
    }

    fn precision_support_hint(&self) -> String {
        let Some(family) = self.selected_family().filter(|f| f.starts_with("gpu.")) else {
            return String::new();
        };
        let hint = self.unsupported_list(self.benchmarks.iter().filter(|mode| {
            self.family_of(mode) == family
                && !gpu_offload(&mode.id)
                && self.gpu_format_unsupported(mode)
        }));
        if hint.is_empty() {
            "Unsupported:\nNone reported".into()
        } else {
            hint
        }
    }

    fn variant_support_hint(&self) -> String {
        let Some(family) = self.selected_family().filter(|f| f.starts_with("cpu.")) else {
            return String::new();
        };
        self.unsupported_list(
            self.family_members(family)
                .into_iter()
                .filter(|mode| self.cpu_mode_unsupported(mode)),
        )
    }

    fn workload_support_hint(&self) -> String {
        self.unsupported_list(
            self.visible_families()
                .iter()
                .filter(|family| self.runnable_modes(family).is_empty())
                .flat_map(|family| self.family_members(family))
                .filter(|mode| {
                    if mode.category == BenchmarkCategory::Gpu {
                        self.gpu_format_unsupported(mode)
                    } else {
                        self.cpu_mode_unsupported(mode)
                    }
                }),
        )
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
        if self.active_request.is_some()
            || index < 0
            || !self.presentation.test_editor
            || self.presentation.curated_set.is_some()
        {
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
        window.on_saved_test_action(move |action, argument| {
            if let Some(app) = weak.upgrade() {
                app.borrow_mut()
                    .saved_test_action(action.as_str(), argument.as_str());
            }
        });
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
        window.set_builder_test_name(app.editor_test_name().into());
        window.set_precision_support_hint(app.precision_support_hint().into());
        window.set_variant_support_hint(app.variant_support_hint().into());
        window.set_workload_support_hint(app.workload_support_hint().into());
        window.set_selected_template_id(
            app.presentation
                .curated_set
                .as_ref()
                .or(app.presentation.editing_saved_test.as_ref())
                .cloned()
                .unwrap_or_default()
                .into(),
        );
        window.set_set_selected(app.presentation.set_selected);
        window.set_set_read_only(app.presentation.curated_set.is_some());
        window.set_test_editor(app.presentation.test_editor);
        window.set_editing_set_test(app.presentation.editing_test.is_some());
        window.set_set_test_rows(ModelRc::new(VecModel::from(app.set_test_rows())));
        window.set_editing_saved_test(app.presentation.editing_saved_test.is_some());
        window.set_saved_test_status(app.presentation.saved_test_status.as_str().into());
        let mut rows = app
            .setup_templates()
            .into_iter()
            .filter_map(|template| {
                let plan = app.setup_template_plan(&template).ok()?;
                let coverage = app.test_set_coverage(&plan.runs);
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
                    curated: true,
                    available: true,
                    id: template.id.into(),
                    title: title.into(),
                    detail: template.detail.into(),
                    group: group.into(),
                    coverage: coverage.into(),
                })
            })
            .collect::<Vec<_>>();
        rows.extend(app.saved_test_rows());
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
        window.invoke_template_action("select".into(), "gpu.vector.scaling@1".into());
        App::refresh(&window, &app.borrow());
        window.show().unwrap();
        for (width, height, panel) in [(1280, 850, 0), (980, 680, 0), (980, 680, 1)] {
            if width == 980 {
                window.invoke_saved_test_action("new".into(), "".into());
                window.invoke_saved_test_action("add".into(), "".into());
                if panel == 0 {
                    window.invoke_saved_test_action("apply".into(), "".into());
                    // Exercise wrapped settings inside the fixed-width sidebar.
                    window.invoke_saved_test_action("inspect".into(), "0".into());
                }
                App::refresh(&window, &app.borrow());
            }
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
            if width == 1280 && panel == 0 {
                // The disabled Run button must not fall through to customization.
                for (x, y) in [(850., 240.), (278., 203.), (320., 240.), (900., 274.)] {
                    window.set_selected_template_id("".into());
                    let position = slint::LogicalPosition::new(x, y);
                    window
                        .window()
                        .dispatch_event(slint::platform::WindowEvent::PointerPressed {
                            position,
                            button: slint::platform::PointerEventButton::Left,
                        });
                    window
                        .window()
                        .dispatch_event(slint::platform::WindowEvent::PointerReleased {
                            position,
                            button: slint::platform::PointerEventButton::Left,
                        });
                    if x == 850. {
                        assert!(window.get_selected_template_id().is_empty());
                    } else {
                        assert_eq!(
                            window.get_selected_template_id(),
                            window.get_setup_templates().row_data(0).unwrap().id,
                            "row click at ({x}, {y})"
                        );
                        assert!(app.borrow().custom_plan.is_empty());
                        assert!(app.borrow().presentation.curated_set.is_some());
                        App::refresh(&window, &app.borrow());
                        assert!(window.get_set_read_only());
                        assert!(!window.get_test_editor());
                        assert!(window.get_set_test_rows().row_count() > 0);
                    }
                }
            }
        }
        window.invoke_saved_test_action("cancel".into(), "".into());
        App::refresh(&window, &app.borrow());
        assert!(!window.get_test_editor());
        window.invoke_select_suite("cpu.performance".into());
        assert!(!window.get_builder_gpu_compute());
        assert!(!window.get_set_selected());
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
    fn curated_selection_is_read_only_and_never_copies_into_the_builder() {
        let mut app = catalogue();
        app.selected_suite = "gpu.performance".into();
        let defaults = app.settings.clone();
        let before = app.custom_plan.clone();
        app.template_action("select", "gpu.vector.scaling@1");
        assert_eq!(app.settings, defaults);
        assert_eq!(app.custom_plan, before);
        assert!(app.presentation.curated_set.is_some());
        assert!(!app.presentation.test_editor);
        assert!(!app.set_test_rows().is_empty());
        for action in ["add", "edit", "remove", "rename", "save", "delete", "apply"] {
            app.saved_test_action(action, "0");
        }
        app.builder_choice("workload", 1);
        assert_eq!(app.custom_plan, before);
        assert_eq!(app.settings, defaults);
        assert!(app.benchmark_preferences.saved_tests.is_empty());
    }

    #[test]
    fn gpu_builder_paths_and_precisions_follow_the_selected_workload() {
        let mut app = catalogue();
        app.presentation.test_editor = true;
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

    fn support_catalogue() -> App {
        let mut app = catalogue();
        for mode in &mut app.benchmarks {
            if mode.category == BenchmarkCategory::Gpu {
                mode.data_type = if mode.id.contains(".fp16") {
                    "fp16"
                } else if mode.id.contains(".fp64") {
                    "fp64"
                } else if mode.id.contains(".fp8") {
                    "fp8"
                } else if mode.id.contains(".int8") {
                    "int8"
                } else {
                    "fp32"
                }
                .into();
            }
        }
        app.selected_suite = "gpu.performance".into();
        app.select_family("gpu.vector");
        app
    }

    #[test]
    fn precision_hint_lists_missing_formats_for_the_selected_adapter_only() {
        let mut app = support_catalogue();
        let fp64 = app
            .benchmarks
            .iter_mut()
            .find(|b| b.id == "gpu.performance.fp64")
            .unwrap();
        fp64.supported_device_ids = vec!["gpu:other".into()];
        fp64.available = true; // Available elsewhere must not imply support on this GPU.
        fp64.unavailable_reason.clear();
        let gpu = app
            .devices
            .iter_mut()
            .find(|d| Some(&d.id) == app.selected_gpu_id.as_ref())
            .unwrap();
        gpu.properties.insert("shader_f64".into(), "false".into());
        let mut other = gpu.clone();
        let hint = app.precision_support_hint();
        assert_eq!(hint, "Unsupported:\nFP64");
        assert!(!hint.contains("FP16"));
        assert!(!hint.contains("Sparse"));
        assert!(!hint.contains("FP8"));
        assert!(
            !app.builder_gpu_precisions()
                .iter()
                .any(|b| b.id == "gpu.performance.fp64")
        );
        other.id = "gpu:other".into();
        other.properties.insert("shader_f64".into(), "true".into());
        app.devices.push(other);
        for mode in &mut app.benchmarks {
            if mode.category == BenchmarkCategory::Gpu && mode.available {
                mode.supported_device_ids.push("gpu:other".into());
            }
        }
        app.selected_gpu_id = Some("gpu:other".into());
        assert_eq!(app.precision_support_hint(), "Unsupported:\nNone reported");
    }

    #[test]
    fn matrix_hint_reports_only_probed_gpu_formats_on_separate_lines() {
        let mut app = support_catalogue();
        let mode = app
            .benchmarks
            .iter_mut()
            .find(|b| b.id == GPU_MATRIX_ID)
            .unwrap();
        mode.available = true;
        mode.supported_device_ids = vec!["gpu:other".into()];
        let gpu = app
            .devices
            .iter_mut()
            .find(|d| Some(&d.id) == app.selected_gpu_id.as_ref())
            .unwrap();
        gpu.properties
            .insert("cooperative_matrix_fp16".into(), "true".into());
        gpu.properties
            .insert("storage_buffer_16bit_access".into(), "false".into());
        app.select_family("gpu.matrix");
        // Missing app paths and storage access must not be labelled as unsupported number formats.
        assert_eq!(app.precision_support_hint(), "Unsupported:\nNone reported");
        let gpu = app
            .devices
            .iter_mut()
            .find(|d| Some(&d.id) == app.selected_gpu_id.as_ref())
            .unwrap();
        gpu.properties
            .insert("cooperative_matrix_fp16".into(), "false".into());
        gpu.properties
            .insert("cooperative_matrix_int8".into(), "false".into());
        assert_eq!(app.precision_support_hint(), "Unsupported:\nFP16\nINT8");
        gpu_support_unknown_is_not_unsupported(&mut app);
    }

    fn gpu_support_unknown_is_not_unsupported(app: &mut App) {
        let gpu = app
            .devices
            .iter_mut()
            .find(|d| Some(&d.id) == app.selected_gpu_id.as_ref())
            .unwrap();
        gpu.properties.remove("cooperative_matrix_fp16");
        gpu.properties.remove("cooperative_matrix_int8");
        assert_eq!(app.precision_support_hint(), "Unsupported:\nNone reported");
    }

    #[test]
    fn missing_avx_is_explained_even_when_only_the_scalar_variant_remains() {
        let mut app = support_catalogue();
        app.selected_suite = "cpu.performance".into();
        for mode in &mut app.benchmarks {
            if mode.id.contains(".avx2.") {
                mode.available = false;
                mode.unavailable_reason = "avx2_fma_not_supported".into();
            }
        }
        app.select_family("cpu.fp32");
        assert_eq!(app.runnable_modes("cpu.fp32").len(), 1);
        assert_eq!(app.variant_support_hint(), "Unsupported:\nAVX2/FMA");
        assert!(!app.variant_support_hint().contains("AVX-512")); // No invented tests.
        let aes = app
            .benchmarks
            .iter_mut()
            .find(|b| b.id.ends_with(".aes"))
            .unwrap();
        aes.available = false;
        aes.unavailable_reason = "aes_not_supported".into();
        assert_eq!(app.workload_support_hint(), "Unsupported:\nAES");
        assert!(!app.workload_support_hint().contains("AVX2")); // Scalar FP32/64 workloads remain available.
        let mode = app
            .benchmarks
            .iter_mut()
            .find(|b| b.id == CPU_VECTOR_ID)
            .unwrap();
        mode.unavailable_reason = "cpu_topology_unavailable".into();
        assert!(app.variant_support_hint().is_empty());
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
