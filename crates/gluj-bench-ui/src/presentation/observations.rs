use super::*;
use gluj_bench_core::{RawSample, RunRecord, RunStore, SampleTimingScope};

const PAGE_SIZE: usize = 30;

#[derive(Default)]
pub(crate) struct Selection {
    series: usize,
    realization: usize,
    page: usize,
}

impl Selection {
    pub(crate) fn change(&mut self, action: &str, index: i32) {
        match action {
            "series" => {
                self.series = index.max(0) as usize;
                self.realization = 0;
                self.page = 0;
            }
            "realization" => {
                self.realization = index.max(0) as usize;
                self.page = 0;
            }
            "previous" => self.page = self.page.saturating_sub(1),
            "next" => self.page += 1,
            _ => {}
        }
    }

    pub(crate) fn view(&self, record: &RunRecord) -> ObservationView {
        let keys = choices(record);
        let series = self.series.min(keys.len().saturating_sub(1));
        let (realizations, samples) = selected(
            record,
            keys.get(series).map(|(key, _)| key),
            self.realization,
        );
        let realization = self.realization.min(realizations.len().saturating_sub(1));
        let count = samples.len();
        let pages = count.div_ceil(PAGE_SIZE);
        let page = self.page.min(pages.saturating_sub(1));
        let (chart, chart_points) = chart(&samples);
        let labels: Vec<_> = keys
            .iter()
            .map(|(_, sample)| series_label(sample))
            .collect();
        let mut label_counts = BTreeMap::new();
        for label in &labels {
            *label_counts.entry(label.as_str()).or_insert(0) += 1;
        }
        ObservationView {
            series: ModelRc::new(VecModel::from(
                labels
                    .iter()
                    .enumerate()
                    .map(|(index, label)| {
                        if label_counts[label.as_str()] > 1 {
                            format!("{label} · Series {}", index + 1).into()
                        } else {
                            label.as_str().into()
                        }
                    })
                    .collect::<Vec<SharedString>>(),
            )),
            realizations: ModelRc::new(VecModel::from(
                realizations
                    .into_iter()
                    .enumerate()
                    .map(|(index, _)| SharedString::from(format!("Realization {}", index + 1)))
                    .collect::<Vec<_>>(),
            )),
            series_index: series as i32,
            realization_index: realization as i32,
            rows: ModelRc::new(VecModel::from(rows(&samples, page))),
            chart_visible: chart.size().width > 0,
            chart,
            chart_points: ModelRc::new(VecModel::from(chart_points)),
            message: if count == 0 {
                "No raw observations were saved for this run.".into()
            } else {
                let chart_note = if count > 800 {
                    " · Chart uses min/max buckets"
                } else {
                    ""
                };
                format!(
                    "{count} observations · Page {} / {pages}{chart_note}",
                    page + 1
                )
                .into()
            },
            has_previous: page > 0,
            has_next: page + 1 < pages,
            loading: false,
        }
    }
}

#[derive(Default)]
pub(crate) struct ResultObservations {
    run_id: SharedString,
    record: Option<RunRecord>,
    pending: Option<Receiver<Result<RunRecord, String>>>,
    selection: Selection,
}

impl ResultObservations {
    pub(crate) fn open(&mut self, window: &MainWindow, root: Option<&std::path::Path>) {
        let run_id = window.get_selected_native_id();
        *self = Self {
            run_id: run_id.clone(),
            ..Self::default()
        };
        let Some(root) = root else {
            window.set_result_observations(ObservationView {
                message: "Run history is unavailable.".into(),
                ..Default::default()
            });
            return;
        };
        let store = RunStore::new(root);
        let (sender, receiver) = mpsc::channel();
        let id = run_id.to_string();
        match thread::Builder::new()
            .name("result-observations".into())
            .spawn(move || {
                let _ = sender.send(store.load(&id).map_err(|error| error.to_string()));
            }) {
            Ok(_) => {
                self.pending = Some(receiver);
                window.set_result_observations(ObservationView {
                    message: "Loading observations…".into(),
                    loading: true,
                    ..Default::default()
                });
            }
            Err(error) => Self::show_error(window, &error.to_string()),
        }
    }

    pub(crate) fn receive(&mut self, window: &MainWindow) {
        let Some(pending) = &self.pending else { return };
        let outcome = match pending.try_recv() {
            Ok(outcome) => outcome,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => Err("Load stopped before completion.".into()),
        };
        self.pending = None;
        if self.run_id != window.get_selected_native_id() {
            return;
        }
        match outcome {
            Ok(record) => {
                window.set_result_observations(self.selection.view(&record));
                self.record = Some(record);
            }
            Err(error) => Self::show_error(window, &error),
        }
    }

    pub(crate) fn change(&mut self, window: &MainWindow, action: &str, index: i32) {
        if self.run_id == window.get_selected_native_id()
            && let Some(record) = &self.record
        {
            self.selection.change(action, index);
            window.set_result_observations(self.selection.view(record));
        }
    }

    fn show_error(window: &MainWindow, error: &str) {
        window.set_result_observations(ObservationView {
            message: format!("Could not load observations. {error}").into(),
            ..Default::default()
        });
    }
}

pub(crate) fn series(sample: &RawSample) -> String {
    let phase = sample.block_id.split(':').next().unwrap_or("Main");
    format!(
        "{} · {} · {} · {:?} · {phase}",
        sample.condition_id, sample.metric, sample.unit, sample.timing_scope
    )
}
fn choices(record: &RunRecord) -> Vec<(String, &RawSample)> {
    let mut choices = BTreeMap::new();
    for sample in &record.samples {
        choices.entry(series(sample)).or_insert(sample);
    }
    choices.into_iter().collect()
}

fn series_label(sample: &RawSample) -> String {
    let mut fields = sample.condition_id.split(':');
    let benchmark = fields.next().unwrap_or_default();
    let metric = if let Some(level) = benchmark.strip_prefix("cpu.bandwidth.cache.l")
        && let Some(operation @ ("Read" | "Write" | "Copy")) = fields.next()
    {
        format!("L{level} {}", operation.to_ascii_lowercase())
    } else {
        display_metric_name(sample.metric.rsplit('.').next().unwrap_or(&sample.metric))
    };
    let size = sample.condition_id.split(':').find_map(|part| {
        part.strip_prefix("payload-")
            .or_else(|| part.strip_prefix("working_set_"))
            .and_then(|value| value.parse::<u64>().ok())
            .map(|bytes| format_binary_size(bytes as f64))
    });
    let timing = match sample.timing_scope {
        SampleTimingScope::ActiveExecution => "Active time",
        SampleTimingScope::GpuTimestampAndHostElapsed => "GPU + host",
        SampleTimingScope::HostCopyElapsed => "Host copy",
        SampleTimingScope::OfferedWindowWithAggregateService => "Offered window",
    };
    let phase = match sample.block_id.split(':').next() {
        Some("reference-before") => " · Reference before",
        Some("reference-after") => " · Reference after",
        Some("warmup") => " · Warm-up",
        _ => "",
    };
    match size {
        Some(size) => format!("{metric} · {size} · {timing}{phase}"),
        None => format!("{metric} · {timing}{phase}"),
    }
}
fn selected<'a>(
    record: &'a RunRecord,
    key: Option<&String>,
    realization: usize,
) -> (Vec<String>, Vec<&'a RawSample>) {
    let group = record
        .samples
        .iter()
        .filter(|s| Some(&series(s)) == key)
        .collect::<Vec<_>>();
    let realizations = group
        .iter()
        .map(|s| s.realization_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let chosen = realizations.get(realization.min(realizations.len().saturating_sub(1)));
    let samples = group
        .into_iter()
        .filter(|s| Some(&s.realization_id) == chosen)
        .collect();
    (realizations, samples)
}
// Retain the min/max of every chronological bucket. The table/export keeps all points.
fn envelope<'a>(samples: &[&'a RawSample]) -> Vec<&'a RawSample> {
    let width = samples.len().div_ceil(400).max(1);
    samples
        .chunks(width)
        .flat_map(|chunk| {
            let low = chunk
                .iter()
                .copied()
                .min_by(|a, b| a.value.total_cmp(&b.value))
                .unwrap();
            let high = chunk
                .iter()
                .copied()
                .max_by(|a, b| a.value.total_cmp(&b.value))
                .unwrap();
            if low.sequence == high.sequence {
                vec![low]
            } else if low.sequence < high.sequence {
                vec![low, high]
            } else {
                vec![high, low]
            }
        })
        .collect()
}
fn chart(samples: &[&RawSample]) -> (slint::Image, Vec<ChartPoint>) {
    let valid = samples
        .iter()
        .copied()
        .filter(|s| s.value.is_finite() && s.status == CompletionStatus::Completed)
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return Default::default();
    }
    let min = valid.iter().map(|s| s.value).fold(f64::INFINITY, f64::min);
    let max = valid
        .iter()
        .map(|s| s.value)
        .fold(f64::NEG_INFINITY, f64::max);
    let start = valid.iter().map(|s| s.offset_ns).min().unwrap();
    let end = valid.iter().map(|s| s.offset_ns).max().unwrap();
    let range = (max - min).max(max.abs() * 0.01).max(f64::EPSILON);
    let xy = |s: &RawSample| {
        (
            100. + 720. * s.offset_ns.saturating_sub(start) as f64 / (end - start).max(1) as f64,
            if max == min {
                110.
            } else {
                180. - 140. * (s.value - min) / range
            },
        )
    };
    let points = envelope(&valid);
    let coords = points
        .iter()
        .map(|s| {
            let (x, y) = xy(s);
            format!("{x:.2},{y:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let escape = |s: String| {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    };
    let high_label = escape(format_metric(max, &valid[0].unit));
    let low_label = escape(format_metric(min, &valid[0].unit));
    let y_labels = if high_label == low_label {
        format!(r#"<text x="8" y="114">{high_label}</text>"#)
    } else {
        format!(r#"<text x="8" y="40">{high_label}</text><text x="8" y="180">{low_label}</text>"#)
    };
    let end_label = if start == end {
        String::new()
    } else {
        format!(
            r#"<text x="680" y="210">{}</text>"#,
            gpu_timings::format_time(end as f64)
        )
    };
    let mut svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="900" height="240"><rect width="900" height="240" rx="8" fill="#222129"/><path d="M100 30V180H820" fill="none" stroke="#716a80"/><polyline points="{coords}" fill="none" stroke="#dcc66b" stroke-width="2"/><g fill="#c6bfce" font-family="sans-serif" font-size="12">{y_labels}<text x="100" y="210">{}</text>{end_label}<text x="360" y="232">Elapsed time from run start</text></g>"##,
        gpu_timings::format_time(start as f64)
    );
    let mut hover_points = Vec::with_capacity(points.len());
    for sample in points {
        let (x, y) = xy(sample);
        let value = format_metric(sample.value, &sample.unit);
        let offset = gpu_timings::format_time(sample.offset_ns as f64);
        hover_points.push(ChartPoint {
            x: x as f32,
            y: y as f32,
            bottom: 180.0,
            value_label: value.as_str().into(),
            x_label: offset.as_str().into(),
            series: display_metric_name(&sample.metric).into(),
            display_value: value.into(),
            context: format!("Observation #{} · {offset} from run start", sample.sequence).into(),
            accent: Color::from_rgb_u8(220, 198, 107),
        });
        svg.push_str(&format!(
            r##"<circle cx="{x:.2}" cy="{y:.2}" r="2.5" fill="#dcc66b"/>"##
        ));
    }
    svg.push_str("</svg>");
    (
        slint::Image::load_from_svg_data(svg.as_bytes()).unwrap_or_default(),
        hover_points,
    )
}
fn rows(samples: &[&RawSample], page: usize) -> Vec<ObservationRow> {
    samples
        .iter()
        .skip(page * PAGE_SIZE)
        .take(PAGE_SIZE)
        .map(|s| ObservationRow {
            sequence: format!("#{}", s.sequence).into(),
            value: format_metric(s.value, &s.unit).into(),
            offset: gpu_timings::format_time(s.offset_ns as f64).into(),
            active: gpu_timings::format_time(s.active_ns as f64).into(),
            elapsed: gpu_timings::format_time(s.elapsed_ns as f64).into(),
            status: format!("{:?}", s.status).into(),
            scope: format!("{:?}", s.timing_scope).into(),
            realization: s.realization_id.clone().into(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn display_envelope_keeps_spikes_and_chronology() {
        let samples = (0..2000)
            .map(|i| RawSample {
                version: 1,
                sequence: i,
                value: if i == 811 { 1e9 } else { 1. },
                offset_ns: i,
                block_id: "main".into(),
                realization_id: "seed".into(),
                condition_id: "size".into(),
                metric: "throughput".into(),
                unit: "operations/s".into(),
                useful_work: 1,
                useful_work_unit: "operations".into(),
                active_ns: 1,
                elapsed_ns: 1,
                reference_anchor: None,
                validation: gluj_bench_core::ValidationStatus {
                    status: "not_recorded".into(),
                    scope: "fixture".into(),
                },
                status: CompletionStatus::Completed,
                timing_scope: Default::default(),
            })
            .collect::<Vec<_>>();
        let refs = samples.iter().collect::<Vec<_>>();
        let shown = envelope(&refs);
        assert!(shown.len() <= 800);
        assert!(shown.iter().any(|s| s.sequence == 811));
        assert!(shown.windows(2).all(|s| s[0].sequence < s[1].sequence));
        assert_eq!(samples.len(), 2000);

        let (_, points) = chart(&refs);
        assert_eq!(points.len(), shown.len());
        let spike = points
            .iter()
            .find(|point| point.context.starts_with("Observation #811 ·"))
            .unwrap();
        let (x, y) = (spike.x / 2.0, spike.y / 2.0);
        let model = ModelRc::new(VecModel::from(points));
        let index = chart_hover::nearest(model.clone(), x, y, 450.0, 120.0, 240.0);
        let picked = slint::Model::row_data(&model, index as usize).unwrap();
        assert_eq!(picked.x_label, "811.0 ns");
        assert_eq!(picked.display_value, format_metric(1e9, "operations/s"));
    }
}
