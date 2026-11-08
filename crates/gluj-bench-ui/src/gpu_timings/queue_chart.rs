use super::{BenchmarkResult, ChartLabel, ChartPoint, Color, Metric, format_binary_size};
use std::fmt::Write;

fn metric<'a>(
    result: &'a BenchmarkResult,
    bytes: u64,
    suffix: &str,
    unit: &str,
) -> Option<&'a Metric> {
    super::super::demand_metric(result, bytes, suffix).filter(|m| m.unit == unit)
}

pub(super) fn chart(
    result: &BenchmarkResult,
) -> Option<(String, Vec<ChartLabel>, Vec<ChartPoint>)> {
    let sizes = super::super::scaling_compute_tiers(result)
        .into_iter()
        .map(|(bytes, _)| bytes)
        .filter(|bytes| {
            *bytes > 0
                && metric(result, *bytes, "unserved_requests", "requests").is_some()
                && metric(result, *bytes, "completed_requests", "requests").is_some()
        })
        .collect::<Vec<_>>();
    let (first, last) = (*sizes.first()?, *sizes.last()?);
    let span = ((last as f64).log2() - (first as f64).log2()).max(1.0);
    let x = |bytes: u64| {
        if first == last {
            490.0
        } else {
            110.0 + ((bytes as f64).log2() - (first as f64).log2()) / span * 760.0
        }
    };
    let response = |bytes| {
        (metric(result, bytes, "completed_requests", "requests")?.value > 0.0)
            .then(|| metric(result, bytes, "end_to_end_time", "ns"))
            .flatten()
    };
    let pending = |bytes| {
        metric(result, bytes, "unserved_requests", "requests").is_some_and(|m| m.value > 0.0)
    };
    let green = Color::from_rgb_u8(155, 212, 174);
    let red = Color::from_rgb_u8(240, 103, 113);
    let neutral = Color::from_rgb_u8(198, 191, 206);
    let label = |x, y, width, text: String, font_size, accent, alignment| ChartLabel {
        x,
        y,
        width,
        text: text.into(),
        font_size,
        accent,
        alignment,
    };
    // A conventional zero-based millisecond axis with 1/2/5 tick intervals.
    let raw_max_ms = sizes
        .iter()
        .filter_map(|bytes| response(*bytes))
        .map(|m| m.value / 1e6)
        .fold(0.001_f64, f64::max);
    let raw_step = raw_max_ms * 1.05 / 4.0;
    let power = 10_f64.powf(raw_step.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|v| v * power)
        .find(|v| *v >= raw_step)
        .unwrap();
    let maximum_ms = step * 4.0;
    let y = |ns: f64| 220.0 - ns / 1e6 / maximum_ms * 160.0;
    let mut labels = vec![
        label(
            80.0,
            8.0,
            790.0,
            "Average response time · lower is better".into(),
            16.0,
            neutral,
            0,
        ),
        label(
            110.0,
            32.0,
            740.0,
            "Red points: requests were still pending".into(),
            12.0,
            red,
            0,
        ),
    ];
    let mut hover = Vec::new();
    let mut svg = String::from(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="900" height="280" viewBox="0 0 900 280"><rect width="900" height="280" rx="10" fill="#222129"/>"##,
    );
    for tick in 0..=4 {
        let value_ms = step * tick as f64;
        let position = y(value_ms * 1e6);
        writeln!(
            svg,
            r##"<line x1="110" x2="870" y1="{position:.2}" y2="{position:.2}" stroke="#474352"/>"##
        )
        .unwrap();
        labels.push(label(
            0.0,
            (position - 9.0) as f32,
            100.0,
            if step >= 1.0 {
                format!("{value_ms:.0} ms")
            } else if step >= 0.1 {
                format!("{value_ms:.1} ms")
            } else {
                format!("{value_ms:.3} ms")
            },
            12.0,
            neutral,
            2,
        ));
    }
    // Missing completion times remain gaps; pending ages are never substituted for them.
    for pair in sizes.windows(2) {
        if let (Some(a), Some(b)) = (response(pair[0]), response(pair[1])) {
            let color = if pending(pair[0]) || pending(pair[1]) {
                "#f06771"
            } else {
                "#9bd4ae"
            };
            writeln!(svg, r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{color}" stroke-width="2.5"/>"#, x(pair[0]), y(a.value), x(pair[1]), y(b.value)).unwrap();
        }
    }
    for bytes in &sizes {
        if let Some(m) = response(*bytes) {
            let complete = metric(result, *bytes, "completed_requests", "requests")
                .unwrap()
                .value;
            let queued = metric(result, *bytes, "unserved_requests", "requests")
                .unwrap()
                .value;
            let window = metric(result, *bytes, "observation_window", "ns")
                .map(|m| super::format_time(m.value))
                .unwrap_or_else(|| "unavailable".into());
            let oldest = metric(result, *bytes, "oldest_pending_age", "ns")
                .map(|m| super::format_time(m.value))
                .unwrap_or_else(|| "unavailable".into());
            let (accent, color) = if pending(*bytes) {
                (red, "#f06771")
            } else {
                (green, "#9bd4ae")
            };
            let mut point = super::super::chart_hover::point(
                *bytes,
                m,
                [x(*bytes), y(m.value), 220.0],
                ("Average response time", 1e6, "ms"),
                accent,
            );
            point.series = format!("Average response time\n{complete} completed / {queued} pending\nOldest pending: {oldest}\nWindow: {window}").into();
            hover.push(point);
            writeln!(
                svg,
                r#"<circle cx="{:.2}" cy="{:.2}" r="{}" fill="{color}"/>"#,
                x(*bytes),
                y(m.value),
                if sizes.len() > 12 { 2.5 } else { 3.5 }
            )
            .unwrap();
        }
    }
    if hover.is_empty() {
        labels.push(label(
            150.0,
            128.0,
            660.0,
            "No requests completed; response time is unavailable.".into(),
            14.0,
            red,
            1,
        ));
    } else if hover.len() < sizes.len() {
        labels.push(label(
            110.0,
            196.0,
            740.0,
            "Gaps: no requests completed at those sizes".into(),
            11.0,
            neutral,
            2,
        ));
    }
    let mut previous_index = usize::MAX;
    let mut previous_x = -100.0;
    for tick in 0..=4 {
        let index = (sizes.len() - 1) * tick / 4;
        let position = x(sizes[index]);
        if index == previous_index
            || (tick != 4 && position - previous_x < 120.0)
            || (tick != 4 && index > 0 && x(last) - position < 120.0)
        {
            continue;
        }
        previous_index = index;
        previous_x = position;
        let (left, alignment) = if first == last {
            (position - 80.0, 1)
        } else if tick == 0 {
            (position, 0)
        } else if tick == 4 {
            (position - 140.0, 2)
        } else {
            (position - 70.0, 1)
        };
        labels.push(label(
            left as f32,
            231.0,
            140.0,
            format_binary_size(sizes[index] as f64),
            12.0,
            neutral,
            alignment,
        ));
    }
    labels.push(label(
        80.0,
        257.0,
        790.0,
        "Dataset size (logarithmic scale) →".into(),
        12.0,
        neutral,
        1,
    ));
    svg.push_str("</svg>");
    Some((svg, labels, hover))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gluj_bench_core::SampleStatistics;
    use std::collections::BTreeMap;
    fn result(completed: f64) -> BenchmarkResult {
        let mut result = BenchmarkResult {
            benchmark_id: "gpu.performance.fp32.offload.scaling".into(),
            device_id: "gpu:test".into(),
            elapsed_ns: 1,
            metrics: vec![],
            workload_metadata: BTreeMap::from([(
                "traffic_generation_policy".into(),
                "offered_load_v1".into(),
            )]),
            device_metadata: BTreeMap::new(),
        };
        for bytes in [1048576, 4194304] {
            for (suffix, value, unit) in [
                ("compute", 2e12, "operations/s"),
                ("completed_requests", completed, "requests"),
                ("unserved_requests", 100.0 - completed, "requests"),
                ("end_to_end_time", 50e6, "ns"),
                ("oldest_pending_age", 90e6, "ns"),
                ("observation_window", 100e6, "ns"),
            ] {
                result.metrics.push(Metric {
                    name: format!("working_set_{bytes}.{suffix}"),
                    value,
                    unit: unit.into(),
                    statistics: SampleStatistics {
                        sample_count: 5,
                        minimum: value,
                        median: value,
                        maximum: value,
                        standard_deviation: 0.0,
                    },
                });
            }
        }
        result
    }
    #[test]
    fn response_chart_uses_one_time_series_and_keeps_queue_details_in_hover() {
        let result = result(20.0);
        let (svg, labels, points) = super::super::interactive_chart(&result).unwrap();
        assert_eq!(points.len(), 2);
        assert!(
            labels
                .iter()
                .any(|l| l.text.contains("Average response time"))
        );
        assert!(
            labels
                .iter()
                .all(|l| !l.text.contains("GPU service") && !l.text.contains("observation window"))
        );
        assert!(
            points
                .iter()
                .all(|p| p.series.contains("20 completed / 80 pending")
                    && p.display_value == "50 ms")
        );
        assert!(points.iter().all(|p| p.series.contains("Window: 100.00 ms")
            && p.series.contains("Oldest pending: 90.00 ms")));
        let image = slint::Image::load_from_svg_data(svg.as_bytes())
            .unwrap()
            .to_rgba8()
            .unwrap();
        assert_eq!((image.width(), image.height()), (900, 280));
    }
    #[test]
    fn no_completions_does_not_invent_a_processing_time_and_missing_counts_are_not_backfilled() {
        let mut result = result(0.0);
        let (_, labels, points) = chart(&result).unwrap();
        assert!(points.is_empty());
        assert!(
            labels
                .iter()
                .any(|l| l.text.contains("No requests completed"))
        );
        result
            .metrics
            .retain(|m| !m.name.ends_with(".unserved_requests"));
        assert!(super::super::interactive_chart(&result).is_none());
    }
    #[test]
    fn pending_state_changes_color_without_changing_the_response_measurement() {
        let loaded = result(20.0);
        let clear = result(100.0);
        let (_, _, loaded_points) = chart(&loaded).unwrap();
        let (_, _, clear_points) = chart(&clear).unwrap();
        assert_eq!(
            loaded_points[0].display_value,
            clear_points[0].display_value
        );
        assert_eq!(loaded_points[0].accent, Color::from_rgb_u8(240, 103, 113));
        assert_eq!(clear_points[0].accent, Color::from_rgb_u8(155, 212, 174));
    }
}
