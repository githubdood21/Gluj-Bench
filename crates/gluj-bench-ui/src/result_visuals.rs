use super::{BenchmarkResult, Color, Metric, TuningBar, TuningFact, format_metric};

pub(super) fn facts(result: &BenchmarkResult, metric: Option<&Metric>) -> Vec<TuningFact> {
    let Some(metric) = metric else {
        return Vec::new();
    };
    let stats = &metric.statistics;
    let range = stats.sample_count > 0
        && stats.minimum.is_finite()
        && stats.maximum.is_finite()
        && stats.minimum >= 0.0
        && stats.maximum >= stats.minimum;
    let variation = (stats.sample_count > 1
        && metric.value.is_finite()
        && metric.value > 0.0
        && stats.standard_deviation.is_finite())
    .then(|| stats.standard_deviation.abs() / metric.value * 100.0);
    [
        (
            "MEASURED THROUGHPUT",
            format_metric(metric.value, &metric.unit),
            super::display_metric_name(&metric.name),
            Color::from_rgb_u8(115, 220, 202),
        ),
        (
            "SAMPLE RANGE",
            if range {
                format!(
                    "{} – {}",
                    format_metric(stats.minimum, &metric.unit),
                    format_metric(stats.maximum, &metric.unit)
                )
            } else {
                "Not recorded".into()
            },
            format!(
                "{} samples · {:.2} ms total run",
                stats.sample_count,
                result.elapsed_ns as f64 / 1e6
            ),
            Color::from_rgb_u8(120, 169, 255),
        ),
        (
            "SAMPLE VARIATION",
            variation
                .map(|v| format!("{v:.1}%"))
                .unwrap_or_else(|| "Not enough samples".into()),
            "Standard deviation relative to measured throughput".into(),
            if variation.is_some_and(|v| v > 10.0) {
                Color::from_rgb_u8(237, 199, 120)
            } else {
                Color::from_rgb_u8(184, 198, 215)
            },
        ),
    ]
    .into_iter()
    .map(|(label, value, detail, accent)| TuningFact {
        label: label.into(),
        value: value.into(),
        detail: detail.into(),
        accent,
    })
    .collect()
}

pub(super) fn bars(result: &BenchmarkResult, metric: Option<&Metric>) -> Vec<TuningBar> {
    if result.benchmark_id.ends_with(".scaling") {
        return Vec::new();
    }
    let Some(metric) = metric else {
        return Vec::new();
    };
    let s = &metric.statistics;
    if s.sample_count < 2
        || !s.minimum.is_finite()
        || !s.median.is_finite()
        || !s.maximum.is_finite()
        || s.minimum < 0.0
        || s.minimum > s.median
        || s.median > s.maximum
        || s.maximum <= 0.0
    {
        return Vec::new();
    }
    [
        ("Sample minimum", s.minimum),
        ("Sample median", s.median),
        ("Sample maximum", s.maximum),
    ]
    .into_iter()
    .map(|(label, value)| TuningBar {
        label: label.into(),
        value: format_metric(value, &metric.unit).into(),
        ratio: (value / s.maximum) as f32,
        minimum: 0.0,
        maximum: 0.0,
        range_visible: false,
        accent: Color::from_rgb_u8(120, 169, 255),
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_sample_ranges_are_not_invented() {
        let result = BenchmarkResult {
            benchmark_id: "gpu.compute.fp16".into(),
            device_id: "fixture".into(),
            elapsed_ns: 1000,
            metrics: vec![],
            workload_metadata: Default::default(),
            device_metadata: Default::default(),
        };
        let metric = Metric {
            name: "throughput".into(),
            value: 10.0,
            unit: "operations/s".into(),
            statistics: Default::default(),
        };
        assert_eq!(facts(&result, Some(&metric))[1].value, "Not recorded");
        assert!(bars(&result, Some(&metric)).is_empty());
        assert!(facts(&result, None).is_empty());
    }
}
