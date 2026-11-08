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
    let mut facts: Vec<TuningFact> = [
        (
            if super::read_latency(result) {
                "MEASURED READ LATENCY"
            } else {
                "MEASURED THROUGHPUT"
            },
            format_metric(metric.value, &metric.unit),
            super::display_metric_name(&metric.name),
            Color::from_rgb_u8(220, 198, 107),
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
            Color::from_rgb_u8(172, 165, 227),
        ),
        (
            "SAMPLE VARIATION",
            variation
                .map(|v| format!("{v:.1}%"))
                .unwrap_or_else(|| "Not enough samples".into()),
            "Standard deviation relative to measured value".into(),
            if variation.is_some_and(|v| v > 10.0) {
                Color::from_rgb_u8(241, 181, 121)
            } else {
                Color::from_rgb_u8(198, 191, 206)
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
    .collect();
    if let Some(fact) = sustained_fact(result, metric) {
        facts.push(fact);
    }
    facts
}

pub(super) fn sustained_fact(result: &BenchmarkResult, metric: &Metric) -> Option<TuningFact> {
    let accent = Color::from_rgb_u8(172, 165, 227);
    let prefix = metric
        .name
        .strip_suffix(".compute")
        .map(|s| format!("{s}."))
        .unwrap_or_default();
    if let (Some(early), Some(late)) = (
        result
            .metrics
            .iter()
            .find(|m| m.name == format!("{prefix}early_throughput")),
        result
            .metrics
            .iter()
            .find(|m| m.name == format!("{prefix}late_throughput")),
    ) {
        return Some(TuningFact {label:"EARLY / LATE MEDIANS".into(),value:format!("{} → {}",format_metric(early.value,&early.unit),format_metric(late.value,&late.unit)).into(),detail:format!("First/last third of aggregate observations ({} each). Chronological worker samples remain in the native record; no thermal cause inferred.",early.statistics.sample_count).into(),accent});
    }
    let series: serde_json::Value =
        serde_json::from_str(result.workload_metadata.get("sustained_early_late")?).ok()?;
    let selected = metric
        .name
        .strip_prefix("working_set_")
        .and_then(|s| s.split_once('.'))
        .and_then(|(bytes, _)| bytes.parse::<u64>().ok());
    let values: Vec<_> = series
        .as_object()?
        .iter()
        .filter(|(key, _)| selected.is_none_or(|bytes| key.contains(&format!("bytes-{bytes}:"))))
        .filter_map(|(_, v)| {
            Some((
                v.get("early_mean")?.as_f64()?,
                v.get("late_mean")?.as_f64()?,
                v.get("unit")?.as_str()?,
            ))
        })
        .collect();
    if values.is_empty() {
        return None;
    }
    let (mut first_min, mut first_max, mut last_min, mut last_max) =
        (f64::INFINITY, 0.0_f64, f64::INFINITY, 0.0_f64);
    for (early, late, _) in &values {
        first_min = first_min.min(*early);
        first_max = first_max.max(*early);
        last_min = last_min.min(*late);
        last_max = last_max.max(*late);
    }
    Some(TuningFact {label:"SCOPED EARLY / LATE MEANS".into(),value:format!("{}–{} → {}–{}",format_metric(first_min,values[0].2),format_metric(first_max,values[0].2),format_metric(last_min,values[0].2),format_metric(last_max,values[0].2)).into(),detail:format!("Ranges over {} worker/condition/realization series, not total throughput. Full scoped time series retained in native records; no thermal cause inferred.",values.len()).into(),accent})
}

pub(super) fn bars(result: &BenchmarkResult, metric: Option<&Metric>) -> Vec<TuningBar> {
    if super::scaling_result(result) {
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
        accent: Color::from_rgb_u8(172, 165, 227),
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
