use super::{
    BenchmarkResult, DatasetMeasurement, DeviceFact, Metric, ModelRc, VecModel, format_binary_size,
    format_metric,
};

fn fact(label: &str, value: impl Into<String>) -> DeviceFact {
    DeviceFact {
        label: label.into(),
        value: value.into().into(),
    }
}
fn sampling(label: &str, metric: &Metric) -> DeviceFact {
    let stats = &metric.statistics;
    let time = |value| {
        if metric.unit == "ns" {
            super::gpu_timings::format_time(value)
        } else {
            format_metric(value, &metric.unit)
        }
    };
    let value = if stats.sample_count == 0 {
        "Not recorded".into()
    } else if stats.minimum.is_finite()
        && stats.maximum.is_finite()
        && stats.minimum >= 0.0
        && stats.minimum <= metric.value
        && stats.maximum >= metric.value
    {
        format!(
            "{} samples · {} – {}",
            stats.sample_count,
            time(stats.minimum),
            time(stats.maximum)
        )
    } else {
        format!("{} samples · range unavailable", stats.sample_count)
    };
    fact(label, value)
}
pub(super) fn view(result: &BenchmarkResult, bytes: u64, compute: &Metric) -> DatasetMeasurement {
    let offered = super::is_offered_load(result);
    let mut facts = vec![fact(
        if result.benchmark_id == super::CPU_ASCII_ID {
            "Scanning speed"
        } else {
            "Throughput"
        },
        format_metric(compute.value, &compute.unit),
    )];
    let mut samples = vec![sampling("Throughput range", compute)];
    if let Some(early_late) = super::result_visuals::sustained_fact(result, compute) {
        samples.push(fact(
            early_late.label.as_str(),
            format!("{} — {}", early_late.value, early_late.detail),
        ));
    }

    if let Some(count) = result
        .workload_metadata
        .get("access_realizations")
        .filter(|n| n.as_str() != "1")
    {
        samples.push(fact(
            "Workload coverage",
            format!("{count} realizations · Equal coverage · Pooled displayed range"),
        ));
    }
    if let Some(batches) = super::gpu_timings::batch_summary(result, bytes) {
        samples.push(fact("Measured batch work", batches));
    }
    if let Some(bandwidth) = super::chart_bandwidth(result, bytes) {
        facts.push(fact(
            if super::gpu_offload(&result.benchmark_id) {
                "Host traffic (nominal)"
            } else {
                "Effective traffic"
            },
            format_metric(bandwidth.value, &bandwidth.unit),
        ));
        samples.push(sampling("Traffic range", bandwidth));
    }
    if offered {
        let completed = super::demand_metric(result, bytes, "completed_requests");
        if let Some(pending) = super::demand_metric(result, bytes, "unserved_requests") {
            facts.push(fact(
                "Pending at cutoff",
                format!("{} requests", pending.value),
            ));
        }
        if completed.is_some_and(|m| m.value == 0.0) {
            facts.push(fact("Average response", "No completed requests"));
        } else if let Some(delay) = super::demand_metric(result, bytes, "end_to_end_time")
            .filter(|m| m.unit == "ns" && m.value > 0.0)
        {
            facts.push(fact(
                "Average response",
                super::gpu_timings::format_time(delay.value),
            ));
            samples.push(sampling("Response-time range", delay));
        }
        if let Some(completed) = completed {
            samples.push(fact(
                "Completed before cutoff",
                format!("{} requests", completed.value),
            ));
        }
        if let Some(count) = super::demand_metric(result, bytes, "completion_observation_count") {
            samples.push(fact(
                "Completed-only delay observations",
                format!("{:.0} completions; pending requests excluded", count.value),
            ));
        }
        for (suffix, label) in [
            ("completed_only_completion_p50", "Completed-only delay p50"),
            ("completed_only_completion_p90", "Completed-only delay p90"),
            ("completed_only_completion_p95", "Completed-only delay p95"),
            ("completed_only_completion_p99", "Completed-only delay p99"),
        ] {
            if let Some(m) = super::demand_metric(result, bytes, suffix) {
                samples.push(fact(label, super::gpu_timings::format_time(m.value)));
            }
        }
        for (suffix, label) in [
            ("pending_submitted_requests", "Pending submitted requests"),
            (
                "pending_unsubmitted_requests",
                "Pending unsubmitted requests",
            ),
        ] {
            if let Some(m) = super::demand_metric(result, bytes, suffix) {
                samples.push(fact(
                    label,
                    format!("{:.0} requests (sample median)", m.value),
                ));
            }
        }
        for (suffix, label) in [
            ("request_window", "Scheduled request window"),
            ("observation_window", "Observation window"),
            ("oldest_pending_age", "Oldest pending request"),
            ("mean_pending_age", "Mean pending age"),
            ("youngest_pending_age", "Youngest pending request"),
        ] {
            if let Some(m) = super::demand_metric(result, bytes, suffix).filter(|m| m.unit == "ns")
            {
                samples.push(fact(label, super::gpu_timings::format_time(m.value)));
            }
        }
    } else if let Some((gpu, wall)) = super::gpu_timings::point(result, bytes) {
        facts.push(fact(
            "GPU time / pass",
            super::gpu_timings::format_time(gpu.value),
        ));
        facts.push(fact(
            "End-to-end / pass",
            super::gpu_timings::format_time(wall.value),
        ));
        samples.push(sampling("GPU-time range", gpu));
        samples.push(sampling("End-to-end range", wall));
    }
    if super::gpu_offload(&result.benchmark_id)
        && let Some(percent) = result
            .workload_metadata
            .get("ram_offload_percent")
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|p| *p <= 100)
    {
        let host = bytes / 100 * percent + bytes % 100 * percent / 100;
        samples.push(fact(
            "Memory split",
            format!(
                "{} RAM + {} VRAM",
                format_binary_size(host as f64),
                format_binary_size((bytes - host) as f64)
            ),
        ));
    }
    if let Some(shape) = super::cpu_matrix_tier_shape(result, bytes) {
        samples.push(fact("Matrix per worker", shape));
    }
    DatasetMeasurement {
        facts: ModelRc::new(VecModel::from(facts)),
        samples: ModelRc::new(VecModel::from(samples)),
    }
}

#[cfg(test)]
pub(super) fn text(view: &DatasetMeasurement) -> String {
    use slint::Model;
    view.facts
        .iter()
        .chain(view.samples.iter())
        .map(|fact| format!("{}: {}", fact.label, fact.value))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use gluj_bench_core::SampleStatistics;
    use slint::Model;
    use std::collections::BTreeMap;
    fn result() -> BenchmarkResult {
        BenchmarkResult {
            benchmark_id: super::super::GPU_OFFLOAD_ID.into(),
            device_id: "gpu:test".into(),
            elapsed_ns: 1,
            workload_metadata: BTreeMap::from([
                ("traffic_generation_policy".into(), "offered_load_v1".into()),
                ("ram_offload_percent".into(), "75".into()),
                (
                    "gpu_profile_launch_tuning".into(),
                    "verbose internal diagnostic".into(),
                ),
            ]),
            device_metadata: BTreeMap::new(),
            metrics: [
                ("compute", 2e12, "operations/s"),
                ("host_bandwidth", 20e9, "bytes/s"),
                ("end_to_end_time", 5e6, "ns"),
                ("completed_requests", 5.0, "requests"),
                ("unserved_requests", 0.0, "requests"),
            ]
            .into_iter()
            .map(|(suffix, value, unit)| Metric {
                name: format!("working_set_1048576.{suffix}"),
                value,
                unit: unit.into(),
                statistics: SampleStatistics {
                    sample_count: 5,
                    minimum: value * 0.99,
                    median: value,
                    maximum: value * 1.01,
                    standard_deviation: value * 0.01,
                },
            })
            .collect(),
        }
    }
    #[test]
    fn selected_dataset_has_four_facts_and_sampling_data_without_internal_metadata() {
        let result = result();
        let view = view(&result, 1048576, &result.metrics[0]);
        assert_eq!(view.facts.row_count(), 4);
        assert!(
            view.facts
                .iter()
                .any(|f| f.label == "Average response" && f.value == "5.00 ms")
        );
        assert!(
            view.facts
                .iter()
                .any(|f| f.label == "Pending at cutoff" && f.value == "0 requests")
        );
        assert!(
            view.samples
                .iter()
                .any(|f| f.label == "Memory split" && f.value.contains("768.00 KiB RAM"))
        );
        assert!(
            view.samples
                .iter()
                .any(|f| f.label == "Throughput range" && f.value.contains("5 samples"))
        );
        assert!(!text(&view).contains("verbose internal diagnostic"));
    }
    #[test]
    fn missing_completions_and_missing_ranges_are_not_zero_time_or_fabricated_samples() {
        let mut result = result();
        result.metrics[0].statistics.sample_count = 0;
        result.metrics[2].value = 0.0;
        result.metrics[3].value = 0.0;
        let view = view(&result, 1048576, &result.metrics[0]);
        assert!(
            view.facts
                .iter()
                .any(|f| f.label == "Average response" && f.value == "No completed requests")
        );
        assert!(
            view.samples
                .iter()
                .any(|f| f.label == "Throughput range" && f.value == "Not recorded")
        );
        assert!(
            view.samples
                .iter()
                .all(|f| f.label != "Response-time range")
        );
    }
}
