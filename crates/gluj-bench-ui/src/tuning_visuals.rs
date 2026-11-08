use super::{BenchmarkResult, format_binary_size, format_metric};

#[derive(Default)]
pub(super) struct View {
    pub balance: super::ComputeBalance,
    pub quality: String,
}
fn ratio(value: f64, maximum: f64) -> f32 {
    (value / maximum).clamp(0.0, 1.0) as f32
}
fn vram_reference_label(result: &BenchmarkResult) -> &'static str {
    if result
        .workload_metadata
        .get("gpu_saturation_boundary_peak")
        .is_some_and(|s| s == "true")
        || result
            .workload_metadata
            .get("gpu_saturation_peak_repeatable")
            .is_some_and(|s| s == "false")
    {
        "Best observed VRAM-only reference"
    } else {
        "VRAM-only compute reference"
    }
}
pub(super) fn view(result: &BenchmarkResult) -> View {
    if super::is_offered_load(result) {
        return offered_view(result);
    }
    let mut view = View::default();
    let tiers = super::chart_tiers(result);
    let Some(&(_, largest)) = tiers.last() else {
        return view;
    };
    let reference = super::dataset_compute_reference(result).filter(|m| m.unit == largest.unit);
    // Ignore the initial occupancy ramp: compare the slowest median at/after the
    // sweep peak, rather than mistaking a tiny underfilled dispatch for a slowdown.
    let peak_index = tiers
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.1.value.total_cmp(&b.1.1.value))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let &(slow_bytes, slowest) = tiers[peak_index..]
        .iter()
        .min_by(|a, b| a.1.value.total_cmp(&b.1.value))
        .unwrap();
    let peak = tiers[peak_index].1;
    let baseline = reference.unwrap_or(peak);
    let maximum = baseline.value.max(slowest.value);
    let lost = (baseline.value - slowest.value).max(0.0);
    view.balance = super::ComputeBalance {
        visible: true,
        delay: Default::default(),
        delay_warning: false,
        has_loss: lost > 0.0,
        reference: format!("{}: {}", if super::gpu_offload(&result.benchmark_id) && reference.is_some_and(|m| m.name == "measured_dataset_compute_peak") { vram_reference_label(result) } else if reference.is_some_and(|m| m.name == "measured_dataset_compute_peak") { if result.workload_metadata.get("gpu_saturation_boundary_peak").is_some_and(|s| s == "true") || result.workload_metadata.get("gpu_saturation_peak_repeatable").is_some_and(|s| s == "false") { "Best observed dataset reference" } else { "Measured dataset peak" } } else if reference.is_some() { "Default compute reference" } else { "Measured sweep peak" }, format_metric(baseline.value, &baseline.unit)).into(),
        settled: format!("{}: {} at {}", if tiers.len() == 1 { "Measured throughput" } else { "Slowest after peak" }, format_metric(slowest.value, &slowest.unit), format_binary_size(slow_bytes as f64)).into(),
        loss: format!("Reference difference: {} · {:.1}%", format_metric(lost, &baseline.unit), lost / baseline.value * 100.0).into(),
        note: if slowest.name.ends_with(".active_compute") { "Active GPU throughput; pacing waits excluded. The bar shows the measured reference gap, not measured stall time." } else { "Measured throughput medians; the bar shows the reference gap, not measured stall time." }.into(),
        retained_ratio: ratio(slowest.value, maximum),
        reference_ratio: ratio(baseline.value, maximum),
        loss_ratio: ratio(lost, maximum),
    };
    let drift = result
        .workload_metadata
        .get("compute_reference_drift_percent")
        .and_then(|n| n.parse::<f64>().ok())
        .filter(|n| n.is_finite() && *n >= 0.0);
    if slowest.statistics.sample_count == 0 {
        view.quality = "Dataset spread and observation coverage were not recorded.".into();
    } else {
        view.quality = format!(
            "Dataset: {} recorded batches; minimum {}, median {}, maximum {}; SD/median {:.1}%. These are descriptive observations, not confidence intervals.{}",
            slowest.statistics.sample_count,
            format_metric(slowest.statistics.minimum, &slowest.unit),
            format_metric(slowest.value, &slowest.unit),
            format_metric(slowest.statistics.maximum, &slowest.unit),
            slowest.statistics.standard_deviation.abs()
                / slowest.value.abs().max(f64::MIN_POSITIVE)
                * 100.0,
            drift
                .map(|value| format!(" Reference drift: {value:.1}%."))
                .unwrap_or_default()
        );
    }
    if let Some((gpu, wall)) = super::gpu_timings::point(result, slow_bytes) {
        let pacing = slowest.name.ends_with(".active_compute");
        let mut note = format!("{} GPU/pass", super::gpu_timings::format_time(gpu.value));
        if pacing {
            note.push_str(&format!(
                " · {} elapsed/pass incl. pacing",
                super::gpu_timings::format_time(wall.value)
            ));
        }
        note.push_str(
            " · GPU time includes compute, memory access and dispatch/synchronization overhead.",
        );
        view.balance.note = note.into();
    }
    if result
        .workload_metadata
        .get("gpu_saturation_boundary_peak")
        .is_some_and(|s| s == "true")
    {
        view.quality.push_str(" Calibration was still rising at its limit; the start is the best observed size, not a confirmed plateau.");
    }
    if result
        .workload_metadata
        .get("gpu_saturation_peak_repeatable")
        .is_some_and(|s| s == "false")
    {
        view.quality
            .push_str(" Dataset-peak calibration had more than 3% observed SD/median.");
    }
    let variation = slowest.statistics.standard_deviation.abs() / slowest.value * 100.0;
    if (5.0..=10.0).contains(&variation) {
        view.quality
            .push_str(" Observed dataset SD/median is between 5% and 10%.");
    }
    view
}

fn offered_view(result: &BenchmarkResult) -> View {
    let tiers = super::chart_tiers(result);
    let value = |bytes, suffix| {
        super::demand_metric(result, bytes, suffix)
            .map(|m| m.value)
            .unwrap_or(0.0)
    };
    let Some(&(bytes, delivered)) = tiers.iter().min_by(|a, b| {
        value(a.0, "demand_fulfillment")
            .total_cmp(&value(b.0, "demand_fulfillment"))
            .then_with(|| value(b.0, "end_to_end_time").total_cmp(&value(a.0, "end_to_end_time")))
    }) else {
        return View::default();
    };
    let reference = super::dataset_compute_reference(result).filter(|m| m.unit == delivered.unit);
    let baseline = reference.unwrap_or_else(|| {
        tiers
            .iter()
            .max_by(|a, b| a.1.value.total_cmp(&b.1.value))
            .unwrap()
            .1
    });
    let reference_label = match reference.map(|m| m.name.as_str()) {
        Some("measured_dataset_compute_peak") if super::gpu_offload(&result.benchmark_id) => {
            vram_reference_label(result)
        }
        Some("measured_dataset_compute_peak") => "Measured dataset peak",
        Some(_) => "Default compute reference",
        None => "Measured sweep peak",
    };
    let pending = value(bytes, "unserved_requests");
    let completed = value(bytes, "completed_requests");
    let fulfillment = value(bytes, "demand_fulfillment");
    let delay = value(bytes, "end_to_end_time");
    let oldest = value(bytes, "oldest_pending_age");
    let service = value(bytes, "gpu_execution_time");
    let lost = (baseline.value - delivered.value).max(0.0);
    let maximum = baseline.value.max(delivered.value).max(1.0);
    View {
        balance: super::ComputeBalance {
            visible: true, has_loss: lost > 0.0,
            reference: format!("{reference_label}: {}", format_metric(baseline.value, &baseline.unit)).into(),
            settled: format!("Average delivered: {} at {}", format_metric(delivered.value, &delivered.unit), format_binary_size(bytes as f64)).into(),
            loss: format!("Below reference: {} · {:.1}%", format_metric(lost, &delivered.unit), if baseline.value > 0.0 { lost / baseline.value * 100.0 } else { 0.0 }).into(),
            delay: format!("Average request delay: {} · Oldest pending: {}", if completed > 0.0 { super::gpu_timings::format_time(delay) } else { "no completions".into() }, super::gpu_timings::format_time(oldest)).into(),
            delay_warning: pending > 0.0 || delay > service * 1.5,
            note: format!("Demand met: {fulfillment:.1}% · Pending at cutoff: {pending:.0} requests · GPU service: {}", super::gpu_timings::format_time(service)).into(),
            retained_ratio: ratio(delivered.value, maximum), reference_ratio: ratio(baseline.value, maximum), loss_ratio: ratio(lost, maximum),
        },
        quality: "Requests arrive at the configured rate independently of completion. Eight real GPU requests can be in flight; excess demand remains a logical backlog. Average throughput counts observed completions before cutoff. Pending includes unfinished and not yet submitted work; only submitted work is drained afterwards, without throughput credit. The VRAM-only compute reference is measured separately without RAM offload; it is not the peak of the offload curve. The bar compares average delivered throughput with the measured compute reference; configured request rates can also lower the average, so the gap alone is not evidence of memory stalls. Request delay includes queueing, CPU submission and completion polling; it is not a measurement of memory stall time. Each sample starts with an empty queue.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::Metric;
    use super::*;

    #[test]
    fn variation_drift_and_legacy_guidance_never_request_another_run() {
        for variation in [0.0, 0.03, 0.08, 0.25] {
            let mut result = single_matrix_result();
            result.metrics[1].statistics.standard_deviation = result.metrics[1].value * variation;
            result
                .workload_metadata
                .insert("compute_reference_drift_percent".into(), "20".into());
            result.workload_metadata.insert(
                "tuning_guidance".into(),
                "Retest noisy samples; rerun before judging changes.".into(),
            );
            let row = super::super::result_row(&result, &[], &[]);
            for text in [
                &row.tuning_quality,
                &row.tuning_guidance,
                &row.tuning_stats,
                &row.tuning_tldr,
            ] {
                let lower = text.to_lowercase();
                assert!(
                    !lower.contains("rerun")
                        && !lower.contains("retest")
                        && !lower.contains("repeat before"),
                    "{text}"
                );
            }
            assert!(row.tuning_quality.contains("20.0%"));
            assert!(
                row.compute_balance
                    .loss
                    .starts_with("Reference difference:")
            );
        }
    }
    use gluj_bench_core::SampleStatistics;
    use std::collections::BTreeMap;
    fn single_matrix_result() -> BenchmarkResult {
        let metric = |name: &str, value, noise| Metric {
            name: name.into(),
            value,
            unit: "operations/s".into(),
            statistics: SampleStatistics {
                sample_count: 5,
                minimum: value * 0.9,
                median: value,
                maximum: value * 1.1,
                standard_deviation: value * noise,
            },
        };
        BenchmarkResult {
            benchmark_id: super::super::GPU_MATRIX_ID.into(),
            device_id: "gpu:fixture".into(),
            elapsed_ns: 1,
            metrics: vec![
                metric("measured_compute_ceiling", 105.83e12, 0.12),
                metric("working_set_268435456.compute", 56.78e12, 0.053),
            ],
            workload_metadata: BTreeMap::from([
                ("compute_reference_drift_percent".into(), "6.5".into()),
                ("dataset_mode".into(), "single".into()),
                ("gpu_matrix_preset".into(), "custom".into()),
                (
                    "tuning_guidance".into(),
                    "No frequency-limit trial suggested.".into(),
                ),
            ]),
            device_metadata: BTreeMap::new(),
        }
    }
    #[test]
    fn offered_load_shows_average_and_red_delay_instead_of_active_capacity() {
        let mut result = single_matrix_result();
        result.benchmark_id = "gpu.performance.fp32.offload.scaling".into();
        result
            .workload_metadata
            .insert("traffic_generation_policy".into(), "offered_load_v1".into());
        result.metrics[0].value = 50e12;
        result.metrics[1].value = 2e12;
        for (suffix, value) in [
            ("active_compute", 50e12),
            ("requested_compute", 10e12),
            ("demand_fulfillment", 20.0),
            ("unserved_requests", 80.0),
            ("completed_requests", 20.0),
            ("end_to_end_time", 50e6),
            ("gpu_execution_time", 2e6),
            ("oldest_pending_age", 80e6),
        ] {
            let mut m = result.metrics[1].clone();
            m.name = format!("working_set_268435456.{suffix}");
            m.value = value;
            result.metrics.push(m);
        }
        let shown = view(&result).balance;
        assert!(shown.delay_warning && shown.has_loss);
        assert!(shown.delay.contains("50.00 ms") && shown.delay.contains("80.00 ms"));
        assert!(shown.settled.contains("2.00 TOPS"));
        assert!(shown.reference.contains("50.00 TOPS"));
        assert!(shown.loss.contains("96.0%"));
        result
            .metrics
            .iter_mut()
            .find(|m| m.name.ends_with(".requested_compute"))
            .unwrap()
            .value = 1000e12;
        let changed = view(&result).balance;
        assert_eq!(changed.reference, shown.reference);
        assert_eq!(changed.loss, shown.loss);
        assert_eq!(changed.retained_ratio, shown.retained_ratio);
        assert_eq!(super::super::chart_tiers(&result)[0].1.value, 2e12);
        let (_, labels, _) = super::super::scaling_chart_interactive(&result, true).unwrap();
        assert!(labels.iter().any(|l| l.text.contains("Average completed")));
        assert!(
            labels
                .iter()
                .any(|l| l.text.contains("Reference 50.00 TOPS"))
        );
    }

    #[test]
    fn offload_vram_reference_is_distinct_from_measured_sweep_peak() {
        let mut result = single_matrix_result();
        result.benchmark_id = "gpu.performance.fp32.offload.scaling".into();
        result
            .workload_metadata
            .insert("traffic_generation_policy".into(), "offered_load_v1".into());
        result.metrics[0].name = "measured_dataset_compute_peak".into();
        result.metrics[0].value = 48.31e12;
        result.metrics[1].value = 2.32e12;
        let mut first = result.metrics[1].clone();
        first.name = "working_set_50331648.compute".into();
        first.value = 2.57e12;
        result.metrics.push(first);
        let shown = view(&result).balance;
        assert!(
            shown
                .reference
                .contains("VRAM-only compute reference: 48.31 TOPS")
        );
        let tiers = super::super::chart_tiers(&result);
        assert_eq!(tiers.len(), 2);
        assert_eq!(tiers[0].1.value, 2.57e12);
        assert_eq!(tiers[1].1.value, 2.32e12);
        let (_, labels, _) = super::super::scaling_chart_interactive(&result, true).unwrap();
        assert!(
            labels
                .iter()
                .any(|l| l.text.contains("VRAM ref 48.31 TOPS"))
        );
    }

    #[test]
    fn balance_uses_slowest_median_after_peak_and_ignores_initial_underfill() {
        let mut result = single_matrix_result();
        result.metrics[0].value = 100e12;
        let template = result.metrics[1].clone();
        result.metrics.truncate(1);
        for (bytes, value) in [
            (262144, 10e12),
            (1048576, 90e12),
            (4194304, 40e12),
            (8388608, 50e12),
        ] {
            let mut metric = template.clone();
            metric.name = format!("working_set_{bytes}.compute");
            metric.value = value;
            result.metrics.push(metric);
        }
        let shown = view(&result).balance;
        assert!(shown.visible && shown.has_loss);
        assert!(shown.settled.contains("40.00 TOPS at 4.00 MiB"));
        assert!(shown.loss.contains("60.00 TOPS · 60.0%"));
        assert!((shown.retained_ratio - 0.4).abs() < 0.0001);
        assert!((shown.loss_ratio - 0.6).abs() < 0.0001);
        assert_eq!(shown.reference_ratio, 1.0);
    }

    #[test]
    fn balance_without_register_reference_uses_measured_peak_and_handles_single_size() {
        let mut result = single_matrix_result();
        result.metrics.remove(0);
        let shown = view(&result).balance;
        assert!(shown.reference.starts_with("Measured sweep peak"));
        assert!(shown.settled.starts_with("Measured throughput"));
        assert!(!shown.has_loss);
        assert_eq!(shown.retained_ratio, 1.0);
        assert_eq!(shown.loss_ratio, 0.0);
    }

    #[test]
    fn summary_shows_per_pass_cost_and_separates_intentional_pacing() {
        let mut result = single_matrix_result();
        result.benchmark_id = super::super::GPU_OFFLOAD_ID.into();
        for (suffix, value, unit) in [
            ("active_compute", 2e12, "operations/s"),
            ("gpu_execution_time", 12580.0, "ns"),
            ("end_to_end_time", 10e6, "ns"),
        ] {
            let mut metric = result.metrics[1].clone();
            metric.name = format!("working_set_268435456.{suffix}");
            metric.value = value;
            metric.unit = unit.into();
            result.metrics.push(metric);
        }
        let shown = view(&result).balance;
        assert!(shown.note.contains("12.58 µs GPU/pass"));
        assert!(shown.note.contains("10.00 ms elapsed/pass incl. pacing"));
        assert!(shown.note.contains("dispatch/synchronization"));
    }

    #[test]
    fn memory_pressure_uses_one_memory_clock_sentence_without_compute_clock_advice() {
        let mut result = single_matrix_result();
        assert!(super::super::tuning_quick_goal(&result).contains("Compare dataset sizes"));
        let mut small = result.metrics[1].clone();
        small.name = "working_set_262144.compute".into();
        small.value = 93.18e12;
        small.statistics.standard_deviation = small.value * 0.01;
        result.metrics.insert(1, small);
        assert_eq!(
            super::super::tuning_quick_goal(&result),
            "Try increasing VRAM clock speeds for more throughput, or lowering the core clock for less heat and power consumption."
        );
        let notes = super::super::performance_option(&result);
        assert!(!notes.contains("then test compute clocks"));
        result.benchmark_id = super::super::CPU_INT64_ID.into();
        result.device_id = "cpu:test".into();
        assert_eq!(
            super::super::tuning_quick_goal(&result),
            "Try increasing RAM clock speeds for more throughput, or lowering the core clock for less heat and power consumption."
        );
        result.metrics[2].value = 94e12;
        assert!(super::super::tuning_quick_goal(&result).contains("Compare dataset sizes"));
    }

    #[test]
    fn large_single_dataset_gap_offers_experiment_without_inferred_clock_headroom() {
        let result = single_matrix_result();
        let row = super::super::result_row(&result, &[], &[]);
        let view = view(&result);
        assert!(view.balance.loss.contains("46.3%"));
        assert!((view.balance.retained_ratio - 0.5365).abs() < 0.001);
        assert!(view.quality.contains("Reference drift:"));
        assert!(view.quality.contains("Observed dataset SD/median"));
        assert!(row.tuning_tldr.starts_with("Trial:"));
        assert!(row.tuning_tldr.len() < 100);
        assert!(row.tuning_tldr.contains("65% → 50%"));

        assert!(!row.tuning_goal.contains("compute ↑"));

        assert!(row.tuning_guidance.contains("Only one dataset"));
        assert!(row.tuning_guidance.contains("same kernel"));
        assert!(row.tuning_guidance.contains("not an estimate derived"));
        assert!(!row.tuning_guidance.starts_with("No frequency-limit trial"));
        assert!(
            !result
                .workload_metadata
                .contains_key("suggested_core_frequency_limit_reduction_percent")
        );
    }
    #[test]
    fn red_loss_excludes_pacing_and_disappears_when_active_throughput_has_no_loss() {
        let mut result = single_matrix_result();
        result.benchmark_id = super::super::GPU_OFFLOAD_ID.into();
        let mut active = result.metrics[1].clone();
        active.name = "working_set_268435456.active_compute".into();
        active.value = result.metrics[0].value * 0.9;
        result.metrics.push(active);
        let shown = view(&result);

        assert!(shown.balance.has_loss);
        assert!(shown.balance.loss.contains("10.0%"));
        result.metrics[2].value = result.metrics[0].value * 1.1;
        let shown = view(&result);

        assert!(!shown.balance.has_loss);
        assert_eq!(shown.balance.loss_ratio, 0.0);
    }

    #[test]
    fn noisy_dataset_keeps_baseline_first_and_estimated_trials_keep_their_existing_advice() {
        let mut result = single_matrix_result();
        result.metrics[1].statistics.standard_deviation = result.metrics[1].value * 0.15;
        let noisy = view(&result);
        assert!(noisy.quality.contains("SD/median"));

        result.metrics[1].statistics.standard_deviation = result.metrics[1].value * 0.01;
        result.workload_metadata.insert(
            "suggested_core_frequency_limit_reduction_percent".into(),
            "10".into(),
        );
        assert!(super::super::gpu_gap_advice(&result).is_some());
        result
            .workload_metadata
            .remove("suggested_core_frequency_limit_reduction_percent");
        result.metrics[1].value = 100e12;
        assert!(super::super::gpu_gap_advice(&result).is_none());
    }

    #[test]
    fn severe_cpu_gap_replaces_a_saved_small_reduction_with_four_then_two_core_candidates() {
        let mut result = single_matrix_result();
        result.benchmark_id = super::super::CPU_VECTOR_ID.into();
        result.device_id = "cpu:system".into();
        result.metrics[0].value = 100e12;
        result.metrics[0].statistics.standard_deviation = 1e12;
        result.metrics[1].value = 5e12;
        result.metrics[1].statistics.standard_deviation = 0.01e12;
        let mut small = result.metrics[0].clone();
        small.name = "working_set_262144.compute".into();
        result.metrics.insert(1, small);
        result
            .workload_metadata
            .insert("thread_mode".into(), "physical_cores".into());
        result
            .workload_metadata
            .insert("thread_count".into(), "16".into());
        result
            .workload_metadata
            .insert("suggested_cpu_core_count_trial".into(), "14".into());
        let row = super::super::result_row(&result, &[], &[]);
        let view = view(&result);
        assert!(view.balance.loss.contains("95.0%"));
        assert!(row.tuning_tldr.contains("CPU 16 → 4 → 2 cores"));

        assert!(!row.tuning_goal.contains("compute ↑"));
        assert!(!row.tuning_goal.contains("compute ↑"));
        assert!(row.tuning_guidance.contains("not a measured optimum"));
        assert!(
            row.tuning_stats
                .contains("Candidate CPU allocations: 4 then 2")
        );
        // Logical worker counts cannot be presented as a physical-core allocation.
        result
            .workload_metadata
            .insert("thread_mode".into(), "logical_processors".into());
        assert!(super::super::cpu_efficiency_advice(&result).is_none());
    }
    #[test]
    fn severe_gpu_gap_explores_lower_limits_without_compounding_or_promising_power_savings() {
        let mut result = single_matrix_result();
        result.metrics[0].value = 100e12;
        result.metrics[0].statistics.standard_deviation = 1e12;
        result.metrics[1].value = 5e12;
        result.metrics[1].statistics.standard_deviation = 0.01e12;
        result.workload_metadata.insert(
            "suggested_core_frequency_limit_reduction_percent".into(),
            "40".into(),
        );
        let row = super::super::result_row(&result, &[], &[]);
        let view = view(&result);
        assert!(view.balance.loss.contains("95.0%"));
        assert!(row.tuning_tldr.contains("50% → 25%"));

        assert!(row.tuning_guidance.contains("not compounded"));
        assert!(row.tuning_guidance.contains("95% of the original baseline"));
        assert!(!row.tuning_goal.contains("compute ↑"));
        assert!(row.tuning_guidance.contains("does not measure"));
        assert!(!row.tuning_guidance.contains("predicted 95% power savings"));
    }

    #[test]
    fn balance_uses_actual_values_and_keeps_drift_separate_from_noise() {
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
        assert!(view.balance.loss.contains("91.7%"));
        assert!(view.balance.retained_ratio < 0.09);
        assert!(view.quality.contains("Reference drift:"));
    }
}
