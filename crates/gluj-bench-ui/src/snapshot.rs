use super::*;
use std::{
    collections::BTreeSet,
    io::Cursor,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

const METADATA_KEY: &str = "gluj-bench-results";
const MAX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Default)]
pub(super) struct SnapshotState {
    pub open: bool,
    pub ui_dirty: std::cell::Cell<bool>,
    choices_model: Rc<VecModel<SnapshotChoice>>,
    pub choices: Vec<Choice>,
    pub image: Option<Arc<Rendered>>,
    pub preview: slint::Image,
    pub rendering: bool,
    io_pending: usize,
    source: Option<Arc<RenderSource>>,
    worker: Option<RenderWorker>,
    revision: u64,
    pub status: String,
}

// Only plain owned benchmark data crosses into the renderer. No App/Slint state is shared.
struct RenderSource {
    results: Vec<BenchmarkResult>,
    devices: Vec<DeviceDescriptor>,
    benchmarks: Vec<BenchmarkDescriptor>,
    fingerprint: Option<String>,
}
struct RenderJob {
    revision: u64,
    source: Arc<RenderSource>,
    selected: Vec<usize>,
}
enum Completion {
    Render {
        revision: u64,
        image: Result<Rendered, String>,
    },
    Status(String),
    Imported(Result<Vec<BenchmarkResult>, String>),
}
struct RenderWorker {
    requests: mpsc::Sender<RenderJob>,
    events: Receiver<Completion>,
    completed: mpsc::Sender<Completion>,
    latest: Arc<AtomicU64>,
}
impl Drop for RenderWorker {
    fn drop(&mut self) {
        self.latest.fetch_add(1, Ordering::Release);
    }
}
impl RenderWorker {
    fn start() -> Result<Self, String> {
        let (requests, jobs) = mpsc::channel::<RenderJob>();
        let (completed, events) = mpsc::channel();
        let sender = completed.clone();
        let latest = Arc::new(AtomicU64::new(0));
        let generation = latest.clone();
        thread::Builder::new()
            .name("snapshot-render".into())
            .spawn(move || {
                let mut options = None;
                while let Ok(mut job) = jobs.recv() {
                    // Debounce and drain obsolete selections, retaining only the newest request.
                    while let Ok(newest) = jobs.recv_timeout(Duration::from_millis(40)) {
                        job = newest;
                    }
                    if generation.load(Ordering::Acquire) != job.revision {
                        continue;
                    }
                    let fonts = options.get_or_insert_with(font_options);
                    let current = || generation.load(Ordering::Acquire) == job.revision;
                    let image = render(&job.source, job.selected, fonts, &current);
                    if current()
                        && sender
                            .send(Completion::Render {
                                revision: job.revision,
                                image,
                            })
                            .is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|e| format!("Could not start snapshot renderer: {e}"))?;
        Ok(Self {
            requests,
            events,
            completed,
            latest,
        })
    }
}

fn font_options() -> resvg::usvg::Options<'static> {
    let mut options = resvg::usvg::Options {
        font_family: "Segoe UI".into(),
        ..Default::default()
    };
    let fonts = options.fontdb_mut();
    #[cfg(windows)]
    for path in [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
    ] {
        let _ = fonts.load_font_file(path);
    }
    if fonts.is_empty() {
        fonts.load_system_fonts();
    }
    options
}

pub(super) struct Choice {
    pub result_index: usize,
    pub selected: bool,
    pub title: String,
    pub detail: String,
}

pub(super) struct Rendered {
    pub png: Vec<u8>,
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[derive(Serialize, Deserialize)]
struct Payload<T = Vec<BenchmarkResult>> {
    format: String,
    version: u32,
    app_version: String,
    generated_at: u64,
    selected_result_indices: Vec<usize>,
    results: T,
}

fn available(result: &BenchmarkResult) -> bool {
    featured_metric(result).is_some_and(|m| m.value.is_finite() && m.value > 0.0)
}

fn target_label(result: &BenchmarkResult, devices: &[DeviceDescriptor]) -> String {
    let kind = if component_family(result) == "gpu" {
        if gpu_offload(&result.benchmark_id) {
            "GPU + system RAM"
        } else {
            "GPU"
        }
    } else if result.device_id == "memory:system" || result.benchmark_id == "cpu.latency.memory" {
        "System RAM via CPU"
    } else {
        "CPU"
    };
    format!("{kind} \u{00b7} {}", component_label(result, devices))
}

fn ram_label(result: &BenchmarkResult, app: &RenderSource) -> String {
    let mut recorded = result.device_metadata.clone();
    // Legacy CPU identities include the memory configuration. GPU identities do not,
    // so they cannot attest to a previous host's RAM without recorded RAM fields.
    if !recorded.contains_key("saved_system_ram_bytes")
        && component_family(result) == "cpu"
        && let Some(setup) = recorded.get("saved_setup")
    {
        let mut current = result.clone();
        stamp_result(&mut current, &app.devices, app.fingerprint.as_deref());
        if current.device_metadata.get("saved_setup") == Some(setup) {
            recorded = current.device_metadata;
        }
    }
    let capacity = recorded
        .get("saved_system_ram_bytes")
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(|v| format_binary_size(v as f64))
        .unwrap_or_else(|| "capacity not recorded".into());
    let speeds = recorded
        .get("saved_system_ram_speed_mts")
        .map(|v| {
            v.split(',')
                .filter_map(|v| v.parse::<u32>().ok())
                .filter(|v| *v > 0)
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let speed = if speeds.is_empty() {
        "speed not recorded".into()
    } else {
        format!("{} MT/s configured", speeds.join(" / "))
    };
    format!("System RAM \u{00b7} {capacity} \u{00b7} {speed}")
}

pub(super) fn choices(app: &App) -> Vec<Choice> {
    let mut keys = Vec::new();
    for result in &app.results {
        let key = result_component_key(result);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    let mut rows = Vec::new();
    for key in keys {
        for group in component_results(&app.results, &key) {
            let result = group[0];
            if !available(result) {
                continue;
            }
            let index = app
                .results
                .iter()
                .position(|r| std::ptr::eq(r, result))
                .unwrap();

            rows.push(Choice {
                result_index: index,
                selected: false,
                title: result_title(result, &app.benchmarks),
                detail: format!(
                    "{} · {}",
                    target_label(result, &app.devices),
                    result_configuration_label(result)
                ),
            });
        }
    }
    // Highlight only the newest available configuration/setup for each common workload.
    let highlights = [
        CPU_VECTOR_ID,
        CPU_MATRIX_ID,
        CPU_RAM_BANDWIDTH_ID,
        "cpu.latency.memory",
        GPU_VECTOR_ID,
        GPU_MATRIX_ID,
        GPU_INT8_MATRIX_ID,
        "gpu.bandwidth.vram",
    ];
    let mut selected = BTreeSet::new();
    for index in (0..app.results.len()).rev() {
        let result = &app.results[index];
        if highlights.contains(&result.benchmark_id.as_str())
            && !selected.contains(&result.benchmark_id)
            && let Some(choice) = rows.iter_mut().find(|c| c.result_index == index)
        {
            choice.selected = true;
            selected.insert(result.benchmark_id.clone());
        }
    }
    if selected.is_empty() {
        for row in rows.iter_mut().take(4) {
            row.selected = true;
        }
    }
    rows
}

fn check_current(current: &dyn Fn() -> bool) -> Result<(), String> {
    if current() {
        Ok(())
    } else {
        Err("Snapshot selection changed.".into())
    }
}

// Serialization and PNG output also yield to a newer selection between writes.
struct CancellableWriter<'a> {
    bytes: Vec<u8>,
    current: &'a dyn Fn() -> bool,
}
impl Write for CancellableWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if !(self.current)() {
            return Err(std::io::Error::other("Snapshot selection changed."));
        }
        if self.bytes.len().saturating_add(bytes.len()) > MAX_BYTES {
            return Err(std::io::Error::other("Snapshot exceeds the 64 MiB limit."));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn rasterize(
    tree: &resvg::usvg::Tree,
    width: u32,
    height: u32,
    current: &dyn Fn() -> bool,
) -> Result<resvg::tiny_skia::Pixmap, String> {
    check_current(current)?;
    let mut pixels = resvg::tiny_skia::Pixmap::new(width, height)
        .ok_or("Could not allocate the snapshot image.")?;
    // Bound obsolete raster work to one 64-row strip, including for very tall cards.
    for y in (0..height).step_by(64) {
        check_current(current)?;
        let rows = (height - y).min(64);
        let start = y as usize * width as usize * 4;
        let len = rows as usize * width as usize * 4;
        let mut strip = resvg::tiny_skia::PixmapMut::from_bytes(
            &mut pixels.data_mut()[start..start + len],
            width,
            rows,
        )
        .ok_or("Could not allocate the snapshot strip.")?;
        resvg::render(
            tree,
            resvg::tiny_skia::Transform::from_translate(0.0, -(y as f32)),
            &mut strip,
        );
    }
    check_current(current)?;
    Ok(pixels)
}

fn render(
    app: &RenderSource,
    selected: Vec<usize>,
    options: &resvg::usvg::Options<'_>,
    current: &dyn Fn() -> bool,
) -> Result<Rendered, String> {
    if app.results.is_empty() || app.results.len() > 10000 {
        return Err("A snapshot supports between 1 and 10,000 completed results.".into());
    }
    if selected.is_empty() {
        return Err("Select at least one completed result for the picture.".into());
    }
    if selected.len() > 128 {
        return Err("Select up to 128 rows per picture. All results are still embedded.".into());
    }
    let mut hardware = Vec::new();
    for index in &selected {
        let result = app
            .results
            .get(*index)
            .ok_or("Results changed; reopen the snapshot.")?;
        let name = target_label(result, &app.devices);
        if !hardware.contains(&name) {
            hardware.push(name);
        }
    }
    // Include the rest of the detected hardware only when the selected measurements
    // belong to this exact setup. Older saved results keep their recorded hardware names.
    if selected.iter().all(|index| {
        let result = &app.results[*index];
        let Some(setup) = result.device_metadata.get("saved_setup") else {
            return false;
        };
        let mut current = result.clone();
        stamp_result(&mut current, &app.devices, app.fingerprint.as_deref());
        current.device_metadata.get("saved_setup") == Some(setup)
            && [
                "saved_system_cpu_name",
                "saved_system_ram_bytes",
                "saved_system_ram_speed_mts",
            ]
            .iter()
            .all(|key| result.device_metadata.get(*key) == current.device_metadata.get(*key))
    }) {
        hardware.clear();
        for device in app.devices.iter().filter(|d| d.available).take(8) {
            let line = match device.category {
                DeviceCategory::Cpu => Some(format!("CPU · {}", device.name)),
                DeviceCategory::Gpu => Some(format!(
                    "GPU · {}{}",
                    device.name,
                    device
                        .properties
                        .get("device_local_memory_bytes")
                        .and_then(|v| v.parse::<u64>().ok())
                        .map(|v| format!(" · {}", format_binary_size(v as f64)))
                        .unwrap_or_default()
                )),
                DeviceCategory::Memory => None,
            };
            if let Some(line) = line {
                hardware.push(line);
            }
        }
    }
    for index in &selected {
        let ram = ram_label(&app.results[*index], app);
        if !hardware.contains(&ram) {
            hardware.push(ram);
        }
    }
    let width = 1200;
    let header = 158 + hardware.len() as u32 * 34;
    let height = header + selected.len() as u32 * 132 + 86;
    let mut svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Segoe UI, sans-serif"><rect width="100%" height="100%" fill="#0c131e"/><rect x="0" y="0" width="1200" height="6" fill="#61dec8"/>"##
    );
    text_svg(
        &mut svg,
        (100, 66),
        30,
        "#edf3fc",
        "GLUJ-BENCH",
        true,
        false,
    );
    text_svg(
        &mut svg,
        (1152, 62),
        18,
        "#91a5bd",
        &format!("v{}", env!("CARGO_PKG_VERSION")),
        false,
        true,
    );
    text_svg(
        &mut svg,
        (40, 113),
        14,
        "#61dec8",
        "BENCHMARK SNAPSHOT",
        true,
        false,
    );
    for (i, line) in hardware.iter().enumerate() {
        text_svg(
            &mut svg,
            (40, 150 + i as u32 * 34),
            21,
            "#c2d0e3",
            &short(line, 92),
            false,
            false,
        );
    }
    for (i, index) in selected.iter().enumerate() {
        check_current(current)?;
        let result = &app.results[*index];
        let name = result_title(result, &app.benchmarks);
        let y = header + i as u32 * 132;
        svg.push_str(&format!(r##"<rect x="32" y="{y}" width="1136" height="120" rx="12" fill="#172333"/><rect x="32" y="{}" width="4" height="64" rx="2" fill="#61dec8"/>"##, y + 16));
        text_svg(
            &mut svg,
            (54, y + 29),
            20,
            "#edf3fc",
            &short(&name, 62),
            true,
            false,
        );
        text_svg(
            &mut svg,
            (54, y + 53),
            14,
            "#c2d0e3",
            &short(&target_label(result, &app.devices), 110),
            false,
            false,
        );
        text_svg(
            &mut svg,
            (54, y + 75),
            13,
            "#9bb0c8",
            &short(&result_configuration_label(result), 110),
            false,
            false,
        );
        let memory = ["read", "write", "copy"]
            .iter()
            .map(|name| {
                result
                    .metrics
                    .iter()
                    .find(|m| m.name == *name && m.unit == "bytes/s")
            })
            .collect::<Option<Vec<_>>>();
        if let Some(metrics) = memory {
            let scores = metrics
                .iter()
                .map(|m| format!("{} {}", m.name, format_metric(m.value, &m.unit)))
                .collect::<Vec<_>>()
                .join("  ·  ");
            text_svg(
                &mut svg,
                (1144, y + 106),
                18,
                "#70e4cc",
                &scores,
                true,
                true,
            );
        } else {
            text_svg(
                &mut svg,
                (1144, y + 106),
                26,
                "#70e4cc",
                &featured_metric(result)
                    .map(|m| format_metric(m.value, &m.unit))
                    .unwrap_or_else(|| "—".into()),
                true,
                true,
            );
        }
    }
    text_svg(
        &mut svg,
        (40, height - 38),
        14,
        "#91a5bd",
        "Full completed-results list embedded in this PNG · Import snapshot in Gluj-Bench to compare",
        false,
        false,
    );
    text_svg(
        &mut svg,
        (40, height - 17),
        12,
        "#7186a0",
        "Measured workloads and units. Different configurations are separate measurements.",
        false,
        false,
    );
    svg.push_str("</svg>");
    if !current() {
        return Err("Snapshot selection changed.".into());
    }
    let tree = resvg::usvg::Tree::from_str(&svg, options).map_err(|e| e.to_string())?;
    let mut pixels = rasterize(&tree, width, height, current)?;
    if let Ok(logo) = resvg::tiny_skia::Pixmap::decode_png(include_bytes!(
        "../../../assets/branding/loop-chip-g-preview.png"
    )) {
        pixels.draw_pixmap(
            0,
            0,
            logo.as_ref(),
            &resvg::tiny_skia::PixmapPaint::default(),
            resvg::tiny_skia::Transform::from_scale(
                52.0 / logo.width() as f32,
                52.0 / logo.height() as f32,
            )
            .post_translate(34.0, 35.0),
            None,
        );
    }
    if !current() {
        return Err("Snapshot selection changed.".into());
    }
    let rgba = pixels.data().to_vec(); // Opaque background: premultiplied and straight RGBA coincide.
    let payload = Payload {
        format: "gluj-bench-snapshot".into(),
        version: 1,
        app_version: env!("CARGO_PKG_VERSION").into(),
        generated_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        selected_result_indices: selected,
        results: &app.results,
    };
    let mut json = CancellableWriter {
        bytes: Vec::new(),
        current,
    };
    serde_json::to_writer(&mut json, &payload).map_err(|e| e.to_string())?;
    let json = String::from_utf8(json.bytes).map_err(|e| e.to_string())?;
    check_current(current)?;
    let mut png = CancellableWriter {
        bytes: Vec::new(),
        current,
    };
    {
        let mut encoder = png::Encoder::new(&mut png, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut metadata = png::text_metadata::ITXtChunk::new(METADATA_KEY, json);
        metadata.compressed = true;
        metadata.compress_text().map_err(|e| e.to_string())?;
        check_current(current)?;
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer
            .write_text_chunk(&metadata)
            .map_err(|e| e.to_string())?;
        {
            let mut stream = writer.stream_writer().map_err(|e| e.to_string())?;
            for strip in rgba.chunks(width as usize * 4 * 64) {
                check_current(current)?;
                stream.write_all(strip).map_err(|e| e.to_string())?;
            }
            stream.finish().map_err(|e| e.to_string())?;
        }
        writer.finish().map_err(|e| e.to_string())?;
    }
    check_current(current)?;
    let png = png.bytes;
    if png.len() > MAX_BYTES {
        return Err("The snapshot exceeds the 64 MiB PNG import limit.".into());
    }
    Ok(Rendered {
        png,
        rgba,
        width,
        height,
    })
}

fn short(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        text.into()
    } else {
        format!("{}…", text.chars().take(limit - 1).collect::<String>())
    }
}
fn text_svg(
    svg: &mut String,
    position: (u32, u32),
    size: u32,
    color: &str,
    text: &str,
    bold: bool,
    right: bool,
) {
    let (x, y) = position;
    let escaped = text
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;");
    svg.push_str(&format!(r#"<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{}" text-anchor="{}">{escaped}</text>"#, if bold { 700 } else { 400 }, if right { "end" } else { "start" }));
}

pub(super) fn import(bytes: &[u8]) -> Result<Vec<BenchmarkResult>, String> {
    if bytes.len() > MAX_BYTES {
        return Err("Snapshot PNG exceeds 64 MiB.".into());
    }
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_limits(png::Limits { bytes: MAX_BYTES });
    let reader = decoder
        .read_info()
        .map_err(|e| format!("Invalid PNG: {e}"))?;
    let mut metadata = reader.info().utf8_text.iter().find(|chunk| chunk.keyword == METADATA_KEY)
        .cloned().ok_or("This PNG has no Gluj-Bench result data. Use the original saved snapshot; sharing services may remove metadata.")?;
    metadata
        .decompress_text_with_limit(MAX_BYTES)
        .map_err(|e| format!("Invalid snapshot data: {e}"))?;
    let text = metadata.get_text().map_err(|e| e.to_string())?;
    if text.len() > MAX_BYTES {
        return Err("Snapshot result data exceeds 64 MiB.".into());
    }
    let payload: Payload =
        serde_json::from_str(&text).map_err(|e| format!("Invalid snapshot results: {e}"))?;
    if payload.format != "gluj-bench-snapshot" || payload.version != 1 {
        return Err("Unsupported snapshot format/version.".into());
    }
    if payload.results.is_empty() || payload.results.len() > 10000 {
        return Err("Snapshot must contain between 1 and 10,000 completed results.".into());
    }
    if payload.results.iter().any(|r| {
        r.benchmark_id.is_empty()
            || r.device_id.is_empty()
            || r.metrics.iter().any(|m| !m.value.is_finite())
    }) {
        return Err("Snapshot contains invalid result records.".into());
    }
    Ok(payload.results)
}

#[cfg(windows)]
pub(super) fn copy(image: &Rendered) -> Result<(), String> {
    // Offer both the PNG (with metadata) and a standard DIB for paste compatibility.
    let _clipboard = clipboard_win::Clipboard::new_attempts(10).map_err(|e| e.to_string())?;
    let mut dib = vec![0_u8; 40];
    dib[0..4].copy_from_slice(&40_u32.to_le_bytes());
    dib[4..8].copy_from_slice(&image.width.to_le_bytes());
    dib[8..12].copy_from_slice(&(-(image.height as i32)).to_le_bytes());
    dib[12..14].copy_from_slice(&1_u16.to_le_bytes());
    dib[14..16].copy_from_slice(&32_u16.to_le_bytes());
    for pixel in image.rgba.as_chunks::<4>().0 {
        dib.extend_from_slice(&[pixel[2], pixel[1], pixel[0], 255]);
    }
    clipboard_win::raw::set(clipboard_win::formats::CF_DIB, &dib).map_err(|e| e.to_string())?;
    let format =
        clipboard_win::register_format("PNG").ok_or("Could not register PNG clipboard format.")?;
    clipboard_win::raw::set_without_clear(format.get(), &image.png).map_err(|e| e.to_string())
}

#[cfg(not(windows))]
pub(super) fn copy(_: &Rendered) -> Result<(), String> {
    Err("Use Save PNG on this platform.".into())
}

impl App {
    pub(super) fn open_snapshot(&mut self) {
        self.snapshot.open = true;
        self.refresh_snapshot_choices();
    }
    pub(super) fn reset_snapshot_highlights(&mut self) {
        self.snapshot.choices = choices(self);
        self.snapshot.source = Some(Arc::new(self.snapshot_source()));
        self.snapshot.open = true;
        self.update_snapshot();
    }
    pub(super) fn refresh_snapshot_choices(&mut self) {
        if !self.snapshot.open {
            return;
        }
        let previous = self
            .snapshot
            .choices
            .iter()
            .map(|c| (c.result_index, c.selected))
            .collect::<BTreeMap<_, _>>();
        let mut next = choices(self);
        for choice in &mut next {
            if let Some(selected) = previous.get(&choice.result_index) {
                choice.selected = *selected;
            }
        }
        self.snapshot.choices = next;
        self.snapshot.source = Some(Arc::new(self.snapshot_source()));
        self.update_snapshot();
    }
    fn snapshot_source(&self) -> RenderSource {
        RenderSource {
            results: self.results.clone(),
            devices: self.devices.clone(),
            benchmarks: self.benchmarks.clone(),
            fingerprint: self.fingerprint.clone(),
        }
    }
    fn snapshot_worker(&mut self) -> Result<&mut RenderWorker, String> {
        if self.snapshot.worker.is_none() {
            self.snapshot.worker = Some(RenderWorker::start()?);
        }
        Ok(self.snapshot.worker.as_mut().unwrap())
    }
    pub(super) fn update_snapshot(&mut self) {
        self.snapshot.revision = self.snapshot.revision.wrapping_add(1);
        let revision = self.snapshot.revision;
        let selected = self
            .snapshot
            .choices
            .iter()
            .filter(|c| c.selected)
            .map(|c| c.result_index)
            .collect::<Vec<_>>();
        // Immediately invalidate the export, retaining the old preview until its replacement arrives.
        self.snapshot.image = None;
        self.snapshot.rendering = false;
        let queued = (|| {
            let worker = self.snapshot_worker()?;
            worker.latest.store(revision, Ordering::Release);
            if selected.is_empty() {
                return Err("Select at least one completed result for the picture.".into());
            }
            let source = self
                .snapshot
                .source
                .clone()
                .ok_or("Reopen the snapshot to load results.")?;
            self.snapshot
                .worker
                .as_ref()
                .unwrap()
                .requests
                .send(RenderJob {
                    revision,
                    source,
                    selected,
                })
                .map_err(|_| "Snapshot renderer stopped. Close and reopen the panel.".to_owned())
        })();
        match queued {
            Ok(()) => {
                self.snapshot.rendering = true;
                self.snapshot.status = "Updating snapshot…".into();
            }
            Err(error) => {
                self.snapshot.status = error;
                self.snapshot.preview = slint::Image::default();
            }
        }
        self.snapshot.ui_dirty.set(true);
    }
    pub(super) fn close_snapshot(&mut self) {
        self.snapshot.open = false;
        self.snapshot.rendering = false;
        self.snapshot.revision = self.snapshot.revision.wrapping_add(1);
        if let Some(worker) = &self.snapshot.worker {
            worker
                .latest
                .store(self.snapshot.revision, Ordering::Release);
        }
        self.snapshot.ui_dirty.set(true);
    }
    pub(super) fn receive_snapshot_events(&mut self) {
        let events = self
            .snapshot
            .worker
            .as_ref()
            .map(|w| w.events.try_iter().collect::<Vec<_>>())
            .unwrap_or_default();
        for event in events {
            match event {
                Completion::Render { revision, image } => {
                    if revision != self.snapshot.revision || !self.snapshot.open {
                        continue;
                    }
                    self.snapshot.rendering = false;
                    match image {
                        Ok(image) => {
                            let buffer =
                                slint::SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
                                    &image.rgba,
                                    image.width,
                                    image.height,
                                );
                            self.snapshot.preview = slint::Image::from_rgba8(buffer);
                            self.snapshot.image = Some(Arc::new(image));
                            self.snapshot.status = format!(
                                "{} selected · {} completed results embedded",
                                self.snapshot.choices.iter().filter(|c| c.selected).count(),
                                self.results.len()
                            );
                        }
                        Err(error) => {
                            self.snapshot.image = None;
                            self.snapshot.preview = slint::Image::default();
                            self.snapshot.status = error;
                        }
                    }
                    self.snapshot.ui_dirty.set(true);
                }
                Completion::Status(status) => {
                    self.snapshot.io_pending = self.snapshot.io_pending.saturating_sub(1);
                    self.snapshot.status = status;
                    self.snapshot.ui_dirty.set(true);
                }
                Completion::Imported(result) => {
                    self.snapshot.io_pending = self.snapshot.io_pending.saturating_sub(1);
                    match result {
                        Ok(results) => self.import_snapshot_results(results),
                        Err(error) => {
                            self.snapshot.status = format!("Could not import snapshot: {error}")
                        }
                    }
                    self.snapshot.ui_dirty.set(true);
                }
            }
        }
    }
    pub(super) fn refresh_snapshot(window: &MainWindow, app: &Self) {
        use slint::Model;
        app.snapshot.ui_dirty.set(false);
        window.set_snapshot_open(app.snapshot.open);
        window.set_snapshot_status(app.snapshot.status.clone().into());
        let model = &app.snapshot.choices_model;
        let rows = app.snapshot.choices.iter().map(|choice| SnapshotChoice {
            title: choice.title.clone().into(),
            detail: choice.detail.clone().into(),
            selected: choice.selected,
        });
        if model.row_count() != app.snapshot.choices.len() {
            model.set_vec(rows.collect::<Vec<_>>());
        } else {
            for (index, row) in rows.enumerate() {
                let old = model.row_data(index).unwrap();
                if old.title != row.title
                    || old.detail != row.detail
                    || old.selected != row.selected
                {
                    model.set_row_data(index, row);
                }
            }
        }
        window.set_snapshot_choices(ModelRc::from(model.clone()));
        window.set_snapshot_ready(app.snapshot.image.is_some());
        if app.snapshot.open {
            window.set_snapshot_preview(app.snapshot.preview.clone());
        }
    }
    pub(super) fn save_snapshot_in_background(&mut self, picture: Arc<Rendered>, path: PathBuf) {
        self.snapshot_io(move || {
            Completion::Status(match fs::write(&path, &picture.png) {
                Ok(()) => format!("Saved snapshot with full result data: {}", path.display()),
                Err(error) => format!("Could not save PNG: {error}"),
            })
        });
    }
    pub(super) fn copy_snapshot_in_background(&mut self) {
        if let Some(picture) = self.snapshot.image.clone() {
            self.snapshot_io(move || {
                Completion::Status(match copy(&picture) {
                    Ok(()) => "Image copied. Save the original PNG to keep its import data.".into(),
                    Err(error) => format!("Could not copy image: {error}"),
                })
            });
        }
    }
    pub(super) fn import_snapshot_in_background(&mut self, path: PathBuf) {
        self.snapshot_io(move || {
            Completion::Imported(fs::File::open(&path).map_err(|e| e.to_string()).and_then(
                |file| {
                    use std::io::Read;
                    let mut bytes = Vec::new();
                    file.take(MAX_BYTES as u64 + 1)
                        .read_to_end(&mut bytes)
                        .map_err(|e| e.to_string())?;
                    import(&bytes)
                },
            ))
        });
    }
    fn snapshot_io(&mut self, operation: impl FnOnce() -> Completion + Send + 'static) {
        let result = self
            .snapshot_worker()
            .map(|worker| worker.completed.clone())
            .and_then(|completed| {
                thread::Builder::new()
                    .name("snapshot-io".into())
                    .spawn(move || {
                        let _ = completed.send(operation());
                    })
                    .map_err(|e| e.to_string())
            });
        self.snapshot.status = match result {
            Ok(_) => {
                self.snapshot.io_pending += 1;
                "Processing snapshot…".into()
            }
            Err(error) => error,
        };
        self.snapshot.ui_dirty.set(true);
    }
    #[cfg(test)]
    pub(super) fn finish_snapshot_for_test(&mut self) {
        let until = std::time::Instant::now() + Duration::from_secs(10);
        while self.snapshot.rendering || self.snapshot.io_pending > 0 {
            self.receive_snapshot_events();
            assert!(
                std::time::Instant::now() < until,
                "Snapshot rendering timed out"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
    pub(super) fn import_snapshot_results(&mut self, mut results: Vec<BenchmarkResult>) {
        normalize_results(&mut results);
        let count = results.len();
        // Add as a separate comparison source, retaining the recorded hardware identity in the label.
        let source = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for result in &mut results {
            let setup = result
                .device_metadata
                .get("saved_setup")
                .cloned()
                .unwrap_or_else(|| "unknown".into());
            result
                .device_metadata
                .insert("saved_setup".into(), format!("snapshot-{source}-{setup}"));
            let name = component_label(result, &self.devices);
            result
                .device_metadata
                .insert("saved_component_name".into(), format!("Imported · {name}"));
        }
        self.baseline_results.extend(results);
        self.ui_dirty = true;
        self.comparison_selection = 0;
        self.snapshot.status = format!(
            "Imported {count} results for comparison. Choose an Imported setup under Compare with."
        );
        self.snapshot.ui_dirty.set(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_strips_match_full_render_and_cancel_mid_render() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="120" height="210"><rect width="120" height="210" fill="#123456"/><circle cx="60" cy="64" r="40" fill="#abcdef"/><path d="M0 120 L120 180 L30 210 Z" fill="#fff"/></svg>"##;
        let tree = resvg::usvg::Tree::from_str(svg, &Default::default()).unwrap();
        let striped = rasterize(&tree, 120, 210, &|| true).unwrap();
        let mut full = resvg::tiny_skia::Pixmap::new(120, 210).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut full.as_mut(),
        );
        assert_eq!(striped.data(), full.data());
        let checks = std::cell::Cell::new(0);
        let current = || {
            checks.set(checks.get() + 1);
            checks.get() < 4
        };
        assert!(
            rasterize(&tree, 120, 210, &current)
                .unwrap_err()
                .contains("selection changed")
        );
        assert_eq!(checks.get(), 4);
    }

    #[test]
    fn snapshot_updates_do_not_request_full_app_refresh() {
        let mut app = crate::configuration_tests::fixture();
        app.results = vec![measured(GPU_VECTOR_ID, 10e12)];
        app.ui_dirty = false;
        app.open_snapshot();
        assert!(app.snapshot.ui_dirty.get());
        assert!(!app.ui_dirty);
        app.snapshot.ui_dirty.set(false);
        app.finish_snapshot_for_test();
        assert!(app.snapshot.ui_dirty.get());
        assert!(!app.ui_dirty);
        app.snapshot.ui_dirty.set(false);
        app.close_snapshot();
        assert!(app.snapshot.ui_dirty.get());
        assert!(!app.ui_dirty);
    }

    #[test]
    fn cancelled_serialization_and_png_output_stop_writing() {
        let current = std::cell::Cell::new(true);
        let mut writer = CancellableWriter {
            bytes: Vec::new(),
            current: &|| current.get(),
        };
        writer.write_all(b"first").unwrap();
        current.set(false);
        assert!(writer.write_all(b"obsolete").is_err());
        assert_eq!(writer.bytes, b"first");
    }

    #[test]
    fn stale_completions_cannot_replace_latest_or_closed_preview() {
        let mut app = crate::configuration_tests::fixture();
        app.results = vec![measured(GPU_VECTOR_ID, 10e12)];
        let (requests, _jobs) = mpsc::channel();
        let (completed, events) = mpsc::channel();
        let latest = Arc::new(AtomicU64::new(0));
        app.snapshot.worker = Some(RenderWorker {
            requests,
            events,
            completed: completed.clone(),
            latest: latest.clone(),
        });
        app.open_snapshot();
        let old = app.snapshot.revision;
        app.update_snapshot();
        assert_eq!(latest.load(Ordering::Acquire), app.snapshot.revision);
        completed
            .send(Completion::Render {
                revision: old,
                image: Err("obsolete".into()),
            })
            .unwrap();
        app.receive_snapshot_events();
        assert!(app.snapshot.rendering);
        assert_ne!(app.snapshot.status, "obsolete");
        let revision = app.snapshot.revision;
        app.close_snapshot();
        assert_ne!(latest.load(Ordering::Acquire), revision);
        completed
            .send(Completion::Render {
                revision,
                image: Err("closed".into()),
            })
            .unwrap();
        app.receive_snapshot_events();
        assert!(!app.snapshot.rendering);
        assert_ne!(app.snapshot.status, "closed");
        assert!(app.snapshot.image.is_none());
    }

    #[test]
    fn rapid_changes_export_only_latest_selection_and_background_io_round_trips() {
        let mut app = crate::configuration_tests::fixture();
        app.results = vec![
            measured(GPU_VECTOR_ID, 10e12),
            measured(GPU_OFFLOAD_ID, 4e12),
        ];
        app.open_snapshot();
        for index in [0, 1, 0, 1] {
            for choice in &mut app.snapshot.choices {
                choice.selected = choice.result_index == index;
            }
            app.update_snapshot();
            assert!(app.snapshot.image.is_none());
        }
        app.finish_snapshot_for_test();
        let picture = app.snapshot.image.clone().unwrap();
        let reader = png::Decoder::new(Cursor::new(&picture.png))
            .read_info()
            .unwrap();
        let text = reader
            .info()
            .utf8_text
            .iter()
            .find(|t| t.keyword == METADATA_KEY)
            .unwrap()
            .get_text()
            .unwrap();
        let payload: Payload = serde_json::from_str(&text).unwrap();
        assert_eq!(payload.selected_result_indices, [1]);
        assert_eq!(payload.results.len(), 2);
        let path = std::env::temp_dir().join(format!(
            "gluj-snapshot-{}-{}.png",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        app.save_snapshot_in_background(picture, path.clone());
        app.finish_snapshot_for_test();
        assert!(path.exists());
        let mut target = crate::configuration_tests::fixture();
        target.import_snapshot_in_background(path.clone());
        target.finish_snapshot_for_test();
        assert_eq!(target.baseline_results.len(), 2);
        assert!(target.results.is_empty());
        assert!(target.saved_results.is_empty());
        fs::remove_file(path).unwrap();
    }

    fn measured(id: &str, value: f64) -> BenchmarkResult {
        let offload = gpu_offload(id);
        BenchmarkResult {
            benchmark_id: id.into(),
            device_id: "gpu:fixture".into(),
            elapsed_ns: 12345,
            metrics: vec![Metric {
                name: if offload {
                    "working_set_1048576.compute"
                } else {
                    "throughput"
                }
                .into(),
                value,
                unit: "operations/s".into(),
                statistics: Default::default(),
            }],
            workload_metadata: BTreeMap::from([
                (
                    gpu_preset_key(id).into(),
                    if offload { "custom" } else { "register" }.into(),
                ),
                (
                    "dataset_mode".into(),
                    if offload { "single" } else { "register" }.into(),
                ),
            ]),
            device_metadata: BTreeMap::from([(
                "saved_component_name".into(),
                "Example GPU & <hardware>".into(),
            )]),
        }
    }

    #[test]
    fn snapshot_targets_and_recorded_ram_survive_hardware_changes() {
        let mut app = crate::configuration_tests::fixture();
        let mut cpu = app.devices[0].clone();
        cpu.id = "cpu:system".into();
        cpu.name = "Example CPU".into();
        cpu.category = DeviceCategory::Cpu;
        cpu.properties.clear();
        let mut memory = cpu.clone();
        memory.id = "memory:system".into();
        memory.name = "System memory".into();
        memory.category = DeviceCategory::Memory;
        memory.properties = BTreeMap::from([
            ("installed_bytes".into(), (64_u64 << 30).to_string()),
            ("configured_speed_mts".into(), "3200".into()),
        ]);
        app.devices.extend([cpu, memory]);
        let mut cpu_result = measured(CPU_VECTOR_ID, 1e12);
        cpu_result.device_id = "cpu:system".into();
        cpu_result.device_metadata.clear();
        let mut gpu_result = measured(GPU_VECTOR_ID, 10e12);
        gpu_result.device_metadata.clear();
        stamp_result(&mut cpu_result, &app.devices, None);
        stamp_result(&mut gpu_result, &app.devices, None);
        assert_eq!(
            target_label(&cpu_result, &app.devices),
            "CPU \u{00b7} Example CPU"
        );
        assert!(target_label(&gpu_result, &app.devices).starts_with("GPU "));
        let mut ram_result = cpu_result.clone();
        ram_result.device_id = "memory:system".into();
        assert!(target_label(&ram_result, &app.devices).starts_with("System RAM via CPU"));
        let memory = app
            .devices
            .iter_mut()
            .find(|d| d.category == DeviceCategory::Memory)
            .unwrap();
        memory
            .properties
            .insert("installed_bytes".into(), (32_u64 << 30).to_string());
        memory
            .properties
            .insert("configured_speed_mts".into(), "5600".into());
        let source = app.snapshot_source();
        for result in [&cpu_result, &gpu_result] {
            let label = ram_label(result, &source);
            assert!(label.contains("64.00 GiB"));
            assert!(label.contains("3200 MT/s"));
            assert!(!label.contains("5600"));
        }
        gpu_result.device_metadata.remove("saved_system_ram_bytes");
        gpu_result
            .device_metadata
            .remove("saved_system_ram_speed_mts");
        assert!(ram_label(&gpu_result, &source).contains("capacity not recorded"));
        assert!(ram_label(&gpu_result, &source).contains("speed not recorded"));
        cpu_result
            .device_metadata
            .insert("saved_system_ram_speed_mts".into(), "0,unknown".into());
        assert!(ram_label(&cpu_result, &source).contains("speed not recorded"));
    }

    #[test]
    fn highlights_include_only_completed_results_and_latest_configuration() {
        let mut app = crate::configuration_tests::fixture();
        app.results = vec![
            measured(GPU_VECTOR_ID, 10e12),
            measured(GPU_VECTOR_ID, 12e12),
            measured(GPU_OFFLOAD_ID, 5e12),
            measured(GPU_MATRIX_ID, 0.0),
        ];
        app.results[0]
            .workload_metadata
            .insert("gpu_preset".into(), "small".into());
        app.results[0].metrics[0].name = "working_set_1048576.compute".into();
        let choices = choices(&app);
        assert_eq!(choices.len(), 3);
        assert_eq!(
            choices
                .iter()
                .filter(|c| c.selected)
                .map(|c| c.result_index)
                .collect::<Vec<_>>(),
            [1]
        );
        assert!(!choices.iter().any(|c| c.result_index == 3));
        app.results.clear();
        assert!(super::choices(&app).is_empty());
    }

    #[test]
    fn png_round_trip_embeds_unselected_results_and_all_measurements() {
        let mut app = crate::configuration_tests::fixture();
        app.results = vec![
            measured(GPU_VECTOR_ID, 10e12),
            measured(GPU_OFFLOAD_ID, 4e12),
            measured(GPU_MATRIX_ID, 20e12),
        ];
        app.open_snapshot();
        app.finish_snapshot_for_test();
        for choice in &mut app.snapshot.choices {
            choice.selected = choice.result_index == 0;
        }
        app.update_snapshot();
        app.finish_snapshot_for_test();
        let picture = app.snapshot.image.as_ref().unwrap();
        assert_eq!(&picture.png[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        assert_eq!(
            serde_json::to_value(import(&picture.png).unwrap()).unwrap(),
            serde_json::to_value(&app.results).unwrap()
        );
        let mut decoder = png::Decoder::new(Cursor::new(&picture.png))
            .read_info()
            .unwrap();
        let mut pixels = vec![0; decoder.output_buffer_size().unwrap()];
        decoder.next_frame(&mut pixels).unwrap();
        assert_eq!(pixels, picture.rgba);
        for choice in &mut app.snapshot.choices {
            choice.selected = false;
        }
        app.update_snapshot();
        app.finish_snapshot_for_test();
        assert!(app.snapshot.image.is_none());
    }

    fn metadata_png(text: Option<&str>) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, 1, 1);
            encoder.set_color(png::ColorType::Rgba);
            if let Some(text) = text {
                encoder
                    .add_itxt_chunk(METADATA_KEY.into(), text.into())
                    .unwrap();
            }
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[0, 0, 0, 255])
                .unwrap();
        }
        out
    }

    #[test]
    fn invalid_imports_fail_without_mutating_existing_results() {
        assert!(import(b"not a png").is_err());
        assert!(
            import(&metadata_png(None))
                .unwrap_err()
                .contains("no Gluj-Bench")
        );
        assert!(import(&metadata_png(Some("not json"))).is_err());
        let invalid = serde_json::json!({"format":"gluj-bench-snapshot","version":999,"app_version":"0.4.0","generated_at":1,"selected_result_indices":[],"results":[]});
        assert!(
            import(&metadata_png(Some(&invalid.to_string())))
                .unwrap_err()
                .contains("Unsupported")
        );
    }

    #[test]
    fn imported_full_results_are_comparison_data_not_completed_runs() {
        let mut source = crate::configuration_tests::fixture();
        source.results = vec![
            measured(GPU_VECTOR_ID, 10e12),
            measured(GPU_OFFLOAD_ID, 4e12),
        ];
        source.open_snapshot();
        source.finish_snapshot_for_test();
        let imported = import(&source.snapshot.image.as_ref().unwrap().png).unwrap();
        let mut target = crate::configuration_tests::fixture();
        target.results = vec![measured(GPU_VECTOR_ID, 12e12)];
        target.import_snapshot_results(imported);
        assert_eq!(target.results.len(), 1);
        assert!(target.saved_results.is_empty());
        assert_eq!(target.baseline_results.len(), 2);
        let comparison = comparison_metric(
            &target.results[0],
            "throughput",
            &target.baseline_results,
            None,
        )
        .unwrap();
        assert_eq!(comparison.1.value, 10e12);
        assert!(component_label(comparison.0, &target.devices).starts_with("Imported"));
    }

    #[test]
    fn open_snapshot_refresh_preserves_selection_and_embeds_new_runs() {
        let mut app = crate::configuration_tests::fixture();
        app.results = vec![
            measured(GPU_VECTOR_ID, 10e12),
            measured(GPU_OFFLOAD_ID, 4e12),
        ];
        app.open_snapshot();
        app.finish_snapshot_for_test();
        app.snapshot
            .choices
            .iter_mut()
            .find(|c| c.result_index == 0)
            .unwrap()
            .selected = false;
        app.snapshot
            .choices
            .iter_mut()
            .find(|c| c.result_index == 1)
            .unwrap()
            .selected = true;
        app.results.push(measured(GPU_MATRIX_ID, 20e12));
        app.refresh_snapshot_choices();
        app.finish_snapshot_for_test();
        assert!(
            !app.snapshot
                .choices
                .iter()
                .find(|c| c.result_index == 0)
                .unwrap()
                .selected
        );
        assert!(
            app.snapshot
                .choices
                .iter()
                .find(|c| c.result_index == 1)
                .unwrap()
                .selected
        );
        assert_eq!(
            import(&app.snapshot.image.as_ref().unwrap().png)
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    #[ignore = "renders an actual share PNG to target/ui-preview"]
    fn render_snapshot_preview() {
        let mut app = crate::configuration_tests::fixture();
        app.devices[0].name = "NVIDIA GeForce RTX 5070 Ti".into();
        let mut cpu = app.devices[0].clone();
        cpu.id = "cpu:system".into();
        cpu.name = "AMD Ryzen 9 5950X 16-Core Processor".into();
        cpu.category = DeviceCategory::Cpu;
        cpu.properties.clear();
        let mut memory = cpu.clone();
        memory.id = "memory:system".into();
        memory.name = "System memory".into();
        memory.category = DeviceCategory::Memory;
        memory.properties = BTreeMap::from([
            ("installed_bytes".into(), (64_u64 << 30).to_string()),
            ("configured_speed_mts".into(), "3200".into()),
        ]);
        app.devices.extend([cpu, memory]);
        app.results = vec![
            {
                let mut result = measured(CPU_VECTOR_ID, 1.25e12);
                result.device_id = "cpu:system".into();
                result
            },
            measured(GPU_VECTOR_ID, 31.75e12),
            measured(GPU_MATRIX_ID, 105.2e12),
            measured(GPU_INT8_MATRIX_ID, 210.4e12),
            measured(GPU_OFFLOAD_ID, 4.32e12),
        ];
        for (id, name) in [
            (CPU_VECTOR_ID, "FP32 vector compute"),
            (GPU_VECTOR_ID, "FP32 vector compute"),
            (GPU_MATRIX_ID, "FP16 matrix compute"),
            (GPU_INT8_MATRIX_ID, "INT8 matrix compute"),
            (GPU_OFFLOAD_ID, "FP32 RAM offload compute"),
        ] {
            let mut descriptor = app.benchmarks[0].clone();
            descriptor.id = id.into();
            descriptor.name = name.into();
            app.benchmarks.push(descriptor);
        }
        for result in &mut app.results {
            result.device_metadata.clear();
            let key = if result.benchmark_id == CPU_VECTOR_ID {
                "vector_preset"
            } else {
                gpu_preset_key(&result.benchmark_id)
            };
            result.workload_metadata.clear();
            result
                .workload_metadata
                .insert(key.into(), "register".into());
            result
                .workload_metadata
                .insert("dataset_mode".into(), "register".into());
            stamp_result(result, &app.devices, app.fingerprint.as_deref());
        }
        app.open_snapshot();
        app.finish_snapshot_for_test();
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/ui-preview");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("share-snapshot.png"),
            &app.snapshot.image.as_ref().unwrap().png,
        )
        .unwrap();
    }
}
