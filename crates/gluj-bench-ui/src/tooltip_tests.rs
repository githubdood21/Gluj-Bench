use super::*;

#[test]
#[ignore = "offscreen hover, focus and click check; run separately"]
fn hover_help_preserves_controls_and_stays_inside_the_window() {
    use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
    struct Headless(Rc<MinimalSoftwareWindow>);
    impl slint::platform::Platform for Headless {
        fn create_window_adapter(
            &self,
        ) -> Result<Rc<dyn slint::platform::WindowAdapter>, slint::PlatformError> {
            Ok(self.0.clone())
        }
    }
    fn draw(adapter: &MinimalSoftwareWindow, window: &MainWindow) -> Vec<slint::Rgb8Pixel> {
        slint::platform::update_timers_and_animations();
        window.window().request_redraw();
        let mut pixels = vec![slint::Rgb8Pixel::default(); 980 * 680];
        adapter.draw_if_needed(|renderer| {
            renderer.render(&mut pixels, 980);
        });
        pixels
    }
    fn move_to(window: &MainWindow, x: f32, y: f32) {
        window
            .window()
            .dispatch_event(slint::platform::WindowEvent::PointerMoved {
                position: slint::LogicalPosition::new(x, y),
            });
    }
    fn hover(adapter: &MinimalSoftwareWindow, window: &MainWindow, x: f32, y: f32) {
        move_to(window, x, y);
        draw(adapter, window);
        std::thread::sleep(std::time::Duration::from_millis(460));
        draw(adapter, window);
    }
    let adapter = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    slint::platform::set_platform(Box::new(Headless(adapter.clone()))).unwrap();
    let window = MainWindow::new().unwrap();
    let mut state = families::tests::catalogue();
    state.presentation = presentation::State::builder();
    state.selected_suite = "gpu.performance".into();
    state.page = 1;
    state.saved_test_action("new", "");
    state.saved_test_action("add", "");
    let app = Rc::new(RefCell::new(state));
    App::wire(&window, &app);
    App::wire_presentation(&window, &app);
    App::wire_benchmark_setup(&window, &app);
    App::refresh(&window, &app.borrow());
    adapter.set_size(slint::PhysicalSize::new(980, 680));
    window.set_custom_panel(1);
    window.show().unwrap();
    draw(&adapter, &window);
    // Circle hints show without clicking and disappear when the pointer leaves.
    hover(&adapter, &window, 785., 176.);
    assert!(window.get_help_visible());
    assert!(window.get_help_text().contains("Run each version"));
    assert!(window.get_help_left() >= 12.);
    assert!(window.get_help_left() + 344. <= 980. - 12.);
    move_to(&window, 500., 550.);
    draw(&adapter, &window);
    assert!(!window.get_help_visible());
    hover(&adapter, &window, 755., 596.);
    assert!(window.get_help_visible());
    assert!(window.get_help_text().contains("Finish sooner"));
    // Showing help above the footer must not move or swallow the Add action.
    hover(&adapter, &window, 850., 636.);
    assert!(window.get_help_visible());
    assert!(window.get_help_text().contains("Include this test"));
    if let Some(directory) = std::env::var_os("GLUJ_BENCH_UI_PREVIEW_DIR") {
        let pixels = draw(&adapter, &window);
        let file = fs::File::create(PathBuf::from(directory).join("hover-help-980.png")).unwrap();
        let mut encoder = png::Encoder::new(file, 980, 680);
        encoder.set_color(png::ColorType::Rgb);
        let mut writer = encoder.write_header().unwrap();
        let bytes: Vec<_> = pixels.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
        writer.write_image_data(&bytes).unwrap();
    }
    let position = slint::LogicalPosition::new(850., 636.);
    for event in [
        slint::platform::WindowEvent::PointerPressed {
            position,
            button: slint::platform::PointerEventButton::Left,
        },
        slint::platform::WindowEvent::PointerReleased {
            position,
            button: slint::platform::PointerEventButton::Left,
        },
    ] {
        window.window().dispatch_event(event);
    }
    assert_eq!(app.borrow().custom_plan.len(), 1);
    App::refresh(&window, &app.borrow());
    draw(&adapter, &window);
    assert!(!window.get_help_visible());
    // Keyboard focus explains controls too; Escape dismisses help without activating them.
    move_to(&window, 500., 550.);
    for _ in 0..64 {
        window
            .window()
            .dispatch_event(slint::platform::WindowEvent::KeyPressed {
                text: slint::platform::Key::Tab.into(),
            });
        window
            .window()
            .dispatch_event(slint::platform::WindowEvent::KeyReleased {
                text: slint::platform::Key::Tab.into(),
            });
        draw(&adapter, &window);
        if window.get_help_visible() {
            break;
        }
    }
    assert!(
        window.get_help_visible(),
        "tab navigation must reveal a control's explanation"
    );
    window
        .window()
        .dispatch_event(slint::platform::WindowEvent::KeyPressed {
            text: slint::platform::Key::Escape.into(),
        });
    draw(&adapter, &window);
    assert!(!window.get_help_visible());
    assert_eq!(app.borrow().custom_plan.len(), 1);
    // Menu item explanations use the popup's window position and sit beside the menu.
    window.invoke_select_page(2);
    window.set_history_open(true);
    draw(&adapter, &window);
    let position = slint::LogicalPosition::new(900., 170.);
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
    draw(&adapter, &window);
    hover(&adapter, &window, 800., 214.);
    assert!(
        window.get_help_visible(),
        "menu-item hover should show help"
    );
    assert!(
        window
            .get_help_text()
            .contains("Load complete test records")
    );
    // The CPU support hint stays available when AVX filtering leaves just one scalar option.
    {
        let mut app = app.borrow_mut();
        app.selected_suite = "cpu.performance".into();
        for mode in &mut app.benchmarks {
            if mode.id.contains(".avx2.") {
                mode.available = false;
                mode.unavailable_reason = "avx2_fma_not_supported".into();
            }
        }
        app.select_family("cpu.fp32");
    }
    App::refresh(&window, &app.borrow());
    assert!(window.get_variant_support_hint().contains("AVX2/FMA"));
    assert_eq!(slint::Model::row_count(&window.get_family_modes()), 1);
    // Preview the shared selector in the real test editor, rather than a mock control.
    {
        let mut app = app.borrow_mut();
        app.selected_suite = "gpu.performance".into();
        app.page = 1;
        app.saved_test_action("new", "");
        app.saved_test_action("add", "");
    }
    App::refresh(&window, &app.borrow());
    window.set_custom_panel(0);
    fn click(window: &MainWindow, x: f32, y: f32) {
        let position = slint::LogicalPosition::new(x, y);
        for event in [
            slint::platform::WindowEvent::PointerPressed {
                position,
                button: slint::platform::PointerEventButton::Left,
            },
            slint::platform::WindowEvent::PointerReleased {
                position,
                button: slint::platform::PointerEventButton::Left,
            },
        ] {
            window.window().dispatch_event(event);
        }
    }
    fn key(window: &MainWindow, key: slint::platform::Key) {
        window
            .window()
            .dispatch_event(slint::platform::WindowEvent::KeyPressed { text: key.into() });
        window
            .window()
            .dispatch_event(slint::platform::WindowEvent::KeyReleased { text: key.into() });
    }
    fn preview(adapter: &MinimalSoftwareWindow, window: &MainWindow, name: &str) {
        let pixels = draw(adapter, window);
        if let Some(directory) = std::env::var_os("GLUJ_BENCH_UI_PREVIEW_DIR") {
            let file = fs::File::create(PathBuf::from(directory).join(name)).unwrap();
            let mut encoder = png::Encoder::new(file, 980, 680);
            encoder.set_color(png::ColorType::Rgb);
            let mut writer = encoder.write_header().unwrap();
            let bytes: Vec<_> = pixels.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
            writer.write_image_data(&bytes).unwrap();
        }
    }
    window.invoke_builder_choice("workload".into(), 0);
    draw(&adapter, &window);
    // Selecting with the keyboard commits only on Return; Escape leaves the test unchanged.
    click(&window, 800., 214.);
    draw(&adapter, &window);
    preview(&adapter, &window, "selector-open-980.png");
    key(&window, slint::platform::Key::DownArrow);
    key(&window, slint::platform::Key::Return);
    draw(&adapter, &window);
    assert_eq!(window.get_builder_workload_index(), 1);
    click(&window, 800., 214.);
    draw(&adapter, &window);
    key(&window, slint::platform::Key::Home);
    key(&window, slint::platform::Key::Escape);
    draw(&adapter, &window);
    assert_eq!(window.get_builder_workload_index(), 1);
    // The entire option row is clickable, including its left padding.
    click(&window, 800., 214.);
    draw(&adapter, &window);
    click(&window, 658., 258.);
    draw(&adapter, &window);
    assert_eq!(window.get_builder_workload_index(), 0);
    // The real results layout gives a long reference comparison its own full-width line.
    window.set_page(2);
    window.set_history_open(false);
    window.set_result_count(1);
    window.set_selected_result(ResultRow {
        title: "GPU matrix throughput".into(),
        primary_value: "59.24 TOPS".into(),
        dataset_loss: "-60.3% at 5.91 GiB vs compute reference".into(),
        ..Default::default()
    });
    preview(&adapter, &window, "dataset-comparison-980.png");
    window.hide().unwrap();
}
