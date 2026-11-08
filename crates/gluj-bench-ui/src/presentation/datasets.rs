use super::*;

pub(crate) fn rows(result: &BenchmarkResult) -> Vec<DatasetRow> {
    phase_eight::sized_metrics(result)
        .into_iter()
        .map(|(size, m)| DatasetRow {
            size: format_binary_size(size as f64).into(),
            operation: display_metric_name(m.name.rsplit('.').next().unwrap_or(&m.name)).into(),
            median: format_metric(
                if m.statistics.sample_count > 0 {
                    m.statistics.median
                } else {
                    m.value
                },
                &m.unit,
            )
            .into(),
            range: if m.statistics.sample_count == 0 {
                "Not recorded".into()
            } else {
                format!(
                    "{} – {}",
                    format_metric(m.statistics.minimum, &m.unit),
                    format_metric(m.statistics.maximum, &m.unit)
                )
                .into()
            },
            count: m.statistics.sample_count.to_string().into(),
        })
        .collect()
}
pub(crate) fn curve(result: &BenchmarkResult) -> (slint::Image, Vec<ChartPoint>) {
    let points = phase_eight::sized_metrics(result)
        .into_iter()
        .filter(|(_, m)| m.unit == "bytes/s" && m.value.is_finite())
        .collect::<Vec<_>>();
    let low = points.iter().map(|(n, _)| *n).min().unwrap_or(0);
    let high = points.iter().map(|(n, _)| *n).max().unwrap_or(0);
    if low == 0 || low == high {
        return Default::default();
    }
    let ymax = points
        .iter()
        .map(|(_, m)| m.statistics.maximum.max(m.value))
        .fold(0., f64::max)
        .max(1.);
    let x = |n: u64| {
        100. + 720. * ((n as f64).ln() - (low as f64).ln())
            / ((high as f64).ln() - (low as f64).ln())
    };
    let y = |v: f64| 210. - 160. * v / ymax;
    let mut svg = String::from(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="900" height="290"><rect width="900" height="290" fill="#222129" rx="8"/><path d="M100 40V210H820" stroke="#716a80" fill="none"/><g fill="#c6bfce" font-family="sans-serif" font-size="12">"##,
    );
    let mut hover_points = Vec::new();
    svg.push_str(&format!("<text x=\"5\" y=\"50\">{}</text><text x=\"60\" y=\"210\">0</text><text x=\"100\" y=\"235\">{}</text><text x=\"750\" y=\"235\">{}</text><text x=\"320\" y=\"256\">Dataset size · logarithmic scale</text></g>",format_metric(ymax,"bytes/s"),format_binary_size(low as f64),format_binary_size(high as f64)));
    for (i, (operation, color, accent)) in [
        ("read", "#dcc66b", Color::from_rgb_u8(220, 198, 107)),
        ("write", "#aca5e3", Color::from_rgb_u8(172, 165, 227)),
        ("copy", "#d99aab", Color::from_rgb_u8(217, 154, 171)),
        (
            "host_to_device",
            "#dcc66b",
            Color::from_rgb_u8(220, 198, 107),
        ),
        (
            "device_to_host",
            "#aca5e3",
            Color::from_rgb_u8(172, 165, 227),
        ),
    ]
    .iter()
    .enumerate()
    {
        let mut series = points
            .iter()
            .filter(|(_, m)| m.name.ends_with(&format!(".{operation}")))
            .collect::<Vec<_>>();
        series.sort_by_key(|(size, _)| *size);
        if series.is_empty() {
            continue;
        }
        let coordinates = series
            .iter()
            .map(|(n, m)| format!("{:.2},{:.2}", x(*n), y(m.value)))
            .collect::<Vec<_>>()
            .join(" ");
        let operation_label = operation.replace('_', " ");
        svg.push_str(&format!(r##"<polyline points="{coordinates}" stroke="{color}" stroke-width="2" fill="none"/><text x="{}" y="280" fill="{color}" font-family="sans-serif" font-size="12">{operation_label}</text>"##,100+i*145));
        for (n, m) in series {
            let px = x(*n);
            let py = y(m.value);
            hover_points.push(chart_hover::point(
                *n,
                m,
                [px, py, 210.0],
                (&operation_label, 1e9, "GB/s"),
                *accent,
            ));
            if m.statistics.sample_count > 0 {
                svg.push_str(&format!(
                    r##"<path d="M{px:.2} {:.2}V{:.2}" stroke="{color}" opacity="0.5"/>"##,
                    y(m.statistics.minimum),
                    y(m.statistics.maximum)
                ));
            }
            svg.push_str(&format!(
                r##"<circle cx="{px:.2}" cy="{py:.2}" r="3" fill="{color}"/>"##,
            ));
        }
    }
    svg.push_str("</svg>");
    if hover_points.is_empty() {
        return Default::default();
    }
    (
        slint::Image::load_from_svg_data(svg.as_bytes()).unwrap_or_default(),
        hover_points,
    )
}
pub(crate) fn demand(result: &BenchmarkResult, bytes: u64) -> Vec<DeviceFact> {
    let mut facts = vec![];
    for (suffix, label) in [
        ("offered_requests", "Offered"),
        ("completed_requests", "Completed"),
        ("unserved_requests", "Pending at cutoff"),
        ("pending_submitted_requests", "Pending submitted"),
        ("pending_unsubmitted_requests", "Pending unsubmitted"),
        (
            "completion_observation_count",
            "Completed-only latency observations",
        ),
        ("completed_only_completion_p50", "Completed-only p50"),
        ("completed_only_completion_p90", "Completed-only p90"),
        ("completed_only_completion_p95", "Completed-only p95"),
        ("completed_only_completion_p99", "Completed-only p99"),
        ("oldest_pending_age", "Oldest pending age"),
        ("mean_pending_age", "Mean pending age"),
        ("youngest_pending_age", "Youngest pending age"),
        ("request_window", "Request window"),
        ("observation_window", "Observation window"),
    ] {
        if let Some(m) = demand_metric(result, bytes, suffix) {
            facts.push(fact(
                label,
                if m.unit == "ns" {
                    gpu_timings::format_time(m.value)
                } else {
                    format_metric(m.value, &m.unit)
                },
            ));
        }
    }
    facts
}
pub(crate) fn core_counts(
    current: &BenchmarkResult,
    metric: &Metric,
    results: &[BenchmarkResult],
) -> Vec<TuningBar> {
    if current
        .workload_metadata
        .get("core_experiment")
        .map(String::as_str)
        != Some("fixed_aggregate")
        || !current.device_metadata.contains_key("saved_setup")
        || metric.unit == "ns"
    {
        return vec![];
    }
    let mut normalized = current.clone();
    normalized
        .workload_metadata
        .insert("cpu_core_limit".into(), "1".into());
    let mut points = BTreeMap::new();
    for result in results {
        if result.benchmark_id != current.benchmark_id
            || result_component_key(result) != result_component_key(current)
            || result.workload_metadata.get("core_experiment")
                != current.workload_metadata.get("core_experiment")
        {
            continue;
        }
        let Some(cores) = result
            .workload_metadata
            .get("cpu_core_limit")
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|n| *n > 0)
        else {
            continue;
        };
        let mut candidate = result.clone();
        candidate
            .workload_metadata
            .insert("cpu_core_limit".into(), "1".into());
        if comparison_metric(
            &normalized,
            &metric.name,
            std::slice::from_ref(&candidate),
            None,
        )
        .is_none()
        {
            continue;
        }
        if let Some(m) = result.metrics.iter().find(|m| {
            m.name == metric.name && m.unit == metric.unit && m.value.is_finite() && m.value > 0.
        }) {
            points.insert(cores, m);
        }
    }
    if points.len() < 2 {
        return vec![];
    }
    let maximum = points.values().map(|m| m.value).fold(0., f64::max);
    let one = points.get(&1).map(|m| m.value);
    points
        .iter()
        .map(|(cores, m)| TuningBar {
            label: format!("{cores} cores").into(),
            value: format!(
                "{}{}",
                format_metric(m.value, &m.unit),
                one.map(|v| format!(" · {:.2}× one core", m.value / v))
                    .unwrap_or_default()
            )
            .into(),
            ratio: (m.value / maximum) as f32,
            minimum: 0.,
            maximum: 0.,
            range_visible: false,
            accent: Color::from_rgb_u8(220, 198, 107),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn measured(cores: u32, value: f64) -> BenchmarkResult {
        BenchmarkResult {
            benchmark_id: "cpu.performance.float.f64".into(),
            device_id: "cpu:system".into(),
            elapsed_ns: 1,
            metrics: vec![Metric {
                name: "compute".into(),
                value,
                unit: "operations/s".into(),
                statistics: gluj_bench_core::SampleStatistics {
                    sample_count: 6,
                    minimum: value,
                    maximum: value,
                    median: value,
                    standard_deviation: 0.,
                },
            }],
            workload_metadata: BTreeMap::from([
                ("core_experiment".into(), "fixed_aggregate".into()),
                ("cpu_core_limit".into(), cores.to_string()),
            ]),
            device_metadata: BTreeMap::from([("saved_setup".into(), "same-hardware".into())]),
        }
    }
    #[test]
    fn core_comparison_requires_fixed_input_hardware_and_measured_one_core_for_speedup() {
        let one = measured(1, 10.);
        let two = measured(2, 18.);
        let four = measured(4, 25.);
        let bars = core_counts(&two, &two.metrics[0], &[one.clone(), two.clone()]);
        assert_eq!(bars.len(), 2);
        assert!(bars[1].value.contains("1.80×"));
        let bars = core_counts(&two, &two.metrics[0], &[two.clone(), four.clone()]);
        assert_eq!(bars.len(), 2);
        assert!(bars.iter().all(|r| !r.value.contains("one core")));
        let mut other = four;
        other
            .device_metadata
            .insert("saved_setup".into(), "other-hardware".into());
        assert!(core_counts(&two, &two.metrics[0], &[two.clone(), other]).is_empty());
        let mut changing = two;
        changing
            .workload_metadata
            .insert("core_experiment".into(), "none".into());
        assert!(core_counts(&changing, &changing.metrics[0], &[one, changing.clone()]).is_empty());
    }
}
