use super::{
    BenchmarkResult, Color, Metric, TuningBar, TuningFact, format_binary_size, format_metric,
};

#[derive(Default)]
pub(super) struct View {
    pub top: Vec<TuningFact>,
    pub bottom: Vec<TuningFact>,
    pub bars: Vec<TuningBar>,
    pub allocation: Vec<TuningBar>,
    pub actions: Vec<TuningFact>,
    pub quality: String,
}

fn number(result: &BenchmarkResult, key: &str) -> Option<f64> {
    result
        .workload_metadata
        .get(key)?
        .parse::<f64>()
        .ok()
        .filter(|n| n.is_finite() && *n >= 0.0)
}
fn fact(label: &str, value: String, detail: String, accent: Color) -> TuningFact {
    TuningFact {
        label: label.into(),
        value: value.into(),
        detail: detail.into(),
        accent,
    }
}
fn ratio(value: f64, maximum: f64) -> f32 {
    (value / maximum).clamp(0.0, 1.0) as f32
}
fn bar(label: &str, metric: &Metric, maximum: f64, accent: Color) -> TuningBar {
    let (minimum, upper) = super::chart_sample_range(metric);
    TuningBar {
        label: label.into(),
        value: format_metric(metric.value, &metric.unit).into(),
        ratio: ratio(metric.value, maximum),
        minimum: ratio(minimum, maximum),
        maximum: ratio(upper, maximum),
        range_visible: metric.statistics.sample_count > 1,
        accent,
    }
}

pub(super) fn view(result: &BenchmarkResult) -> View {
    let mut view = View::default();
    let tiers = super::chart_tiers(result);
    let Some(&(bytes, largest)) = tiers.last() else {
        return view;
    };
    let green = Color::from_rgb_u8(115, 220, 202);
    let blue = Color::from_rgb_u8(120, 169, 255);
    let amber = Color::from_rgb_u8(237, 199, 120);
    let neutral = Color::from_rgb_u8(184, 198, 215);
    let reference = result.metrics.iter().find(|m| {
        m.name == "measured_compute_ceiling"
            && m.unit == largest.unit
            && m.value.is_finite()
            && m.value > 0.0
    });
    let small = tiers
        .iter()
        .take(3)
        .max_by(|a, b| a.1.value.total_cmp(&b.1.value))
        .copied();
    let transition = number(result, "bandwidth_transition_working_set_bytes").filter(|n| *n > 0.0);
    view.top.push(fact(
        "DATASET TESTED",
        format_binary_size(bytes as f64),
        transition
            .map(|n| format!("Slowdown first detected at {}", format_binary_size(n)))
            .unwrap_or_else(|| {
                if tiers.len() == 1 {
                    "Single dataset measurement".into()
                } else {
                    "Largest size in this sweep".into()
                }
            }),
        green,
    ));
    view.top.push(fact(
        "LATEST THROUGHPUT",
        format_metric(largest.value, &largest.unit),
        "Measured at the largest dataset".into(),
        green,
    ));
    view.top.push(fact(
        "CHANGE VS REFERENCE",
        reference
            .map(|m| format!("{:+.1}%", (largest.value / m.value - 1.0) * 100.0))
            .unwrap_or_else(|| "Not recorded".into()),
        "Throughput comparison, not GPU stall time".into(),
        blue,
    ));
    let variation = (largest.statistics.sample_count > 1
        && largest.statistics.standard_deviation.is_finite())
    .then(|| largest.statistics.standard_deviation.abs() / largest.value * 100.0);
    view.bottom.push(fact(
        "SAMPLE VARIATION",
        variation
            .map(|n| format!("{n:.1}%"))
            .unwrap_or_else(|| "Unknown".into()),
        format!(
            "{} repeated samples at this size",
            largest.statistics.sample_count
        ),
        if variation.is_some_and(|n| n > 10.0) {
            amber
        } else {
            neutral
        },
    ));
    let drift = number(result, "compute_reference_drift_percent");
    view.bottom.push(fact(
        "REFERENCE DRIFT",
        drift
            .map(|n| format!("{n:.1}%"))
            .unwrap_or_else(|| "Not recorded".into()),
        "Change between pre/post reference runs".into(),
        if drift.is_some_and(|n| n > 5.0) {
            amber
        } else {
            neutral
        },
    ));
    let traffic = result.metrics.iter().find(|m| {
        m.name == format!("working_set_{bytes}.bandwidth")
            && m.unit == "bytes/s"
            && m.value.is_finite()
            && m.value > 0.0
    });
    view.bottom.push(fact(
        "EFFECTIVE TEST TRAFFIC",
        traffic
            .map(|m| format_metric(m.value, &m.unit))
            .unwrap_or_else(|| "Not recorded".into()),
        "Kernel accesses, including cache reuse".into(),
        blue,
    ));
    let mut rates = Vec::new();
    if let Some(reference) = reference {
        rates.push(("Measured compute reference", reference, amber));
    }
    if let Some((_, metric)) = small.filter(|(small_bytes, _)| *small_bytes < bytes) {
        rates.push(("Small dataset baseline", metric, blue));
    }
    rates.push(("Largest dataset", largest, green));
    let maximum = rates
        .iter()
        .map(|(_, m, _)| super::chart_sample_range(m).1)
        .fold(0.0, f64::max);
    for (name, metric, color) in rates {
        view.bars.push(bar(name, metric, maximum, color));
    }
    if let (Some(allocated), Some(budget)) = (
        number(result, "allocated_test_buffer_bytes"),
        number(result, "allocation_budget_bytes").filter(|n| *n > 0.0),
    ) {
        view.allocation.push(TuningBar {
            label: "Test buffers / allocation budget".into(),
            value: format!(
                "{} / {} · {:.1}%",
                format_binary_size(allocated),
                format_binary_size(budget),
                allocated / budget * 100.0
            )
            .into(),
            ratio: ratio(allocated, budget),
            minimum: 0.0,
            maximum: 0.0,
            range_visible: false,
            accent: blue,
        });
    }
    if let (Some(host), Some(local)) = (
        number(result, "host_allocated_test_buffer_bytes"),
        number(result, "device_allocated_test_buffer_bytes"),
    ) && host + local > 0.0
    {
        view.allocation.push(TuningBar {
            label: "System RAM share of test buffers".into(),
            value: format!(
                "{} RAM + {} VRAM · {:.0}% RAM",
                format_binary_size(host),
                format_binary_size(local),
                host / (host + local) * 100.0
            )
            .into(),
            ratio: ratio(host, host + local),
            minimum: 0.0,
            maximum: 0.0,
            range_visible: false,
            accent: green,
        });
    }
    let noisy = !super::consistent_metric(largest)
        || small.is_some_and(|(_, m)| !super::consistent_metric(m));
    view.quality = match (noisy, drift.is_some_and(|n| n > 5.0)) {
        (true, true) => "Repeat before judging a tuning change: some dataset samples are noisy, and the compute reference moved by more than 5%.".into(),
        (true, false) => "Some dataset samples are noisy or have insufficient repeats. Retest before judging small throughput changes.".into(),
        (false, true) => "Dataset samples are consistent, but the compute reference moved by more than 5%. Retest to confirm the reference comparison.".into(),
        (false, false) => "Measurements are repeatable at the compared dataset sizes. Tuning changes still need a measured retest.".into(),
    };
    let offload = super::gpu_offload(&result.benchmark_id);
    let cpu = result.benchmark_id.starts_with("cpu.");
    let first = if offload {
        "Try a lower GPU core limit in small steps, or smaller batches/fewer concurrent GPU jobs in your application.".into()
    } else if cpu {
        super::tuning_takeaway(result)
    } else if let Some(percent) = number(result, "suggested_core_frequency_limit_reduction_percent")
        .filter(|n| *n > 0.0 && *n <= 40.0)
    {
        format!("Try reducing your current GPU core limit by a further {percent:.0}%.")
    } else {
        "Repeat the current configuration to confirm the measurements before choosing a core-limit trial.".into()
    };
    view.actions
        .push(fact("1 · CHANGE ONE THING", first, String::new(), green));
    let shape = if cpu {
        "Keep the same aggregate dataset and arithmetic settings.".into()
    } else if offload {
        format!(
            "Keep the RAM offload share at {}% and the arithmetic settings unchanged.",
            result
                .workload_metadata
                .get("ram_offload_percent")
                .map(String::as_str)
                .unwrap_or("the recorded value")
        )
    } else {
        "Keep the same dataset and arithmetic settings.".into()
    };
    view.actions.push(fact(
        "2 · RETEST THE SAME WORK",
        format!("Rerun at {}.", format_binary_size(bytes as f64)),
        shape,
        blue,
    ));
    view.actions.push(fact("3 · KEEP OR REVERT", format!("For a tuning trial, retain at least {} (95% of this run).", format_metric(largest.value * 0.95, &largest.unit)),
        if offload { "Check application completion time for smaller batches; benchmark TOPS excludes pacing pauses.".into() }
        else { "Keep improvements too. Restore the previous setting if throughput loss exceeds 5%.".into() }, amber));
    view
}

#[cfg(test)]
mod tests {
    use super::*;
    use gluj_bench_core::SampleStatistics;
    use std::collections::BTreeMap;
    #[test]
    fn comparisons_and_allocation_use_actual_values_and_keep_drift_separate_from_noise() {
        let metric = |name: &str, value| Metric {
            name: name.into(),
            value,
            unit: "operations/s".into(),
            statistics: SampleStatistics {
                sample_count: 5,
                minimum: value * 0.99,
                median: value,
                maximum: value * 1.01,
                standard_deviation: value * 0.003,
            },
        };
        let result = BenchmarkResult {
            benchmark_id: "gpu.performance.fp32.offload.scaling".into(),
            device_id: "gpu:test".into(),
            elapsed_ns: 1,
            metrics: vec![
                metric("measured_compute_ceiling", 38.18e12),
                metric("working_set_1024000.compute", 18.08e12),
                metric("working_set_12884901888.compute", 3.16e12),
            ],
            workload_metadata: BTreeMap::from([
                ("compute_reference_drift_percent".into(), "8.87".into()),
                ("allocated_test_buffer_bytes".into(), "12884901888".into()),
                ("allocation_budget_bytes".into(), "22817013760".into()),
                ("ram_offload_percent".into(), "75".into()),
            ]),
            device_metadata: BTreeMap::new(),
        };
        let view = view(&result);
        assert_eq!(view.top[2].value, "-91.7%");
        assert_eq!(view.bottom[0].value, "0.3%");
        assert_eq!(view.bottom[1].value, "8.9%");
        assert!(view.quality.contains("reference moved"));
        assert_eq!(view.bars.len(), 3);
        assert!(view.bars[2].ratio < 0.09);
        assert!(view.allocation[0].value.contains("56.5%"));
        assert!(view.actions[2].value.contains("3.00 TOPS"));
    }
}
