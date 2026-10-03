use gluj_bench_core::{BenchmarkConfig, BenchmarkResult, Metric, SampleStatistics};
use std::collections::BTreeMap;

pub(super) fn reference_config(config: &BenchmarkConfig) -> BenchmarkConfig {
    let mut reference = config.clone();
    reference.samples = config.samples.clamp(3, 5);
    reference.target_duration_ms = config.target_duration_ms.clamp(150, 500);
    reference
}

pub(super) fn validate_reference(
    initial: BenchmarkResult,
    mut check: BenchmarkResult,
) -> BenchmarkResult {
    let before = initial.metrics[0].value;
    let after = check.metrics[0].value;
    let drift = if before > 0.0 {
        ((after / before - 1.0) * 100.0).abs()
    } else {
        f64::INFINITY
    };
    check
        .workload_metadata
        .insert("compute_reference_drift_percent".into(), drift.to_string());
    check.workload_metadata.insert(
        "compute_reference_selection".into(),
        "latest_post_sweep".into(),
    );
    check
}

/// Compare against an empirical compute ceiling, not a manufacturer's theoretical peak.
/// Clock headroom is an idealized roofline estimate, never a measured frequency response.
pub(super) fn add_reference_analysis(
    metrics: &mut Vec<Metric>,
    metadata: &mut BTreeMap<String, String>,
    reference: &BenchmarkResult,
    memory_transition_observed: bool,
) {
    let Some(ceiling) = reference.metrics.first() else {
        return;
    };
    let tiers = metrics
        .iter()
        .filter(|m| m.name.ends_with(".compute"))
        .cloned()
        .collect::<Vec<_>>();
    let Some(largest) = tiers.last() else { return };
    let valid = ceiling.value.is_finite() && ceiling.value > 0.0;
    let mut measured = ceiling.clone();
    measured.name = "measured_compute_ceiling".into();
    metrics.insert(0, measured);
    metadata.insert(
        "compute_reference_kind".into(),
        "measured_register_resident; not_theoretical_peak".into(),
    );
    metadata.insert(
        "compute_reference_benchmark".into(),
        reference.benchmark_id.clone(),
    );
    if let Some(selection) = reference
        .workload_metadata
        .get("compute_reference_selection")
    {
        metadata.insert("compute_reference_selection".into(), selection.clone());
    }
    metadata.insert(
        "clock_analysis_revision".into(),
        "empirical-reference-1".into(),
    );
    let drift = reference
        .workload_metadata
        .get("compute_reference_drift_percent")
        .and_then(|s| s.parse::<f64>().ok());
    if let Some(drift) = drift {
        metadata.insert(
            "compute_reference_drift_percent".into(),
            format!("{drift:.2}"),
        );
    }
    if !valid {
        return;
    }
    for tier in &tiers {
        let delta = |value: f64| (value / ceiling.value - 1.0) * 100.0;
        metrics.push(Metric {
            name: tier.name.replace(".compute", ".reference_delta"),
            value: delta(tier.value),
            unit: "%".into(),
            statistics: SampleStatistics {
                sample_count: tier.statistics.sample_count,
                minimum: delta(tier.statistics.minimum),
                median: delta(tier.statistics.median),
                maximum: delta(tier.statistics.maximum),
                standard_deviation: tier.statistics.standard_deviation / ceiling.value * 100.0,
            },
        });
    }
    let ratio = largest.value / ceiling.value;
    metadata.insert(
        "largest_vs_compute_reference_percent".into(),
        format!("{:.2}", ratio * 100.0),
    );
    metadata.insert(
        "largest_compute_delta_percent".into(),
        format!("{:.2}", (ratio - 1.0) * 100.0),
    );
    let relative_noise =
        |metric: &Metric| metric.statistics.standard_deviation.abs() / metric.value;
    let stable = |metric: &Metric| {
        metric.statistics.sample_count >= 2
            && metric.value.is_finite()
            && metric.value > 0.0
            && metric.statistics.standard_deviation.is_finite()
            && relative_noise(metric) <= 0.10
    };
    // Anchor trials to the smaller of the register reference and this kernel's
    // small-set rate. Kernel overhead must not become assumed clock headroom.
    let small = tiers
        .iter()
        .take(3)
        .max_by(|a, b| a.value.total_cmp(&b.value))
        .unwrap();
    // Require the observed gap to exceed twice the combined relative sample
    // deviation. This permits exploratory trials for meaningful but noisy drops
    // without presenting them as validated low-loss clock settings.
    let gap_noise = relative_noise(ceiling).hypot(relative_noise(largest));
    let trial_anchor = if small.value < ceiling.value {
        small
    } else {
        ceiling
    };
    let trial_gap = 1.0 - largest.value / trial_anchor.value;
    let trial_noise = relative_noise(trial_anchor).hypot(relative_noise(largest));
    let eligible = reference
        .workload_metadata
        .get("bound_classification")
        .is_some_and(|s| s == "compute_bound")
        && stable(ceiling)
        && stable(small)
        && stable(largest)
        && drift.is_some_and(|drift| drift.is_finite())
        && ratio > 0.0
        && ratio < 0.85
        && trial_gap > 0.15
        && trial_gap > 2.0 * trial_noise
        && memory_transition_observed;
    let guidance = if eligible {
        let headroom = (1.0 - ratio) * 100.0;
        let trial_headroom = trial_gap * 100.0;
        let bandwidth_gap = (1.0 / ratio - 1.0) * 100.0;
        // Leave one quarter of the modelled headroom unused; round down in 5% steps.
        // This is an exploratory frequency-limit change, not a verified response.
        let reduction = ((trial_headroom * 0.75 / 5.0).floor() * 5.0).clamp(5.0, 40.0);
        let exploratory = [ceiling, small, largest]
            .iter()
            .any(|metric| relative_noise(metric) > 0.02)
            || drift.is_some_and(|drift| drift > 5.0)
            || !(0.75..=1.10).contains(&(small.value / ceiling.value));
        metadata.insert(
            "tuning_confidence".into(),
            if exploratory {
                "exploratory"
            } else {
                "consistent_samples"
            }
            .into(),
        );
        let mut uncertainty = if exploratory {
            format!(
                " Sample variation at the largest size is {:.1}%; repeat measurements before judging the 0–5% performance-loss target.",
                relative_noise(largest) * 100.0
            )
        } else {
            String::new()
        };
        if drift.is_some_and(|drift| drift > 5.0) {
            uncertainty.push_str(&format!(" The compute reference changed by {:.1}% during the run, so treat this as exploratory.", drift.unwrap()));
        }
        if small.value < ceiling.value * 0.75 {
            uncertainty.push_str(" The trial uses the scaling test's smaller baseline to avoid counting kernel overhead as spare core capacity.");
        }
        metadata.insert(
            "frequency_trial_anchor_operations_per_second".into(),
            trial_anchor.value.to_string(),
        );
        metadata.insert(
            "frequency_trial_headroom_percent".into(),
            format!("{trial_headroom:.2}"),
        );
        metadata.insert(
            "idealized_core_headroom_percent".into(),
            format!("{headroom:.2}"),
        );
        metadata.insert(
            "idealized_bandwidth_increase_percent".into(),
            format!("{bandwidth_gap:.2}"),
        );
        metadata.insert(
            "suggested_core_underclock_trial_percent".into(),
            format!("{reduction:.0}"),
        );
        metadata.insert(
            "suggested_core_frequency_limit_reduction_percent".into(),
            format!("{reduction:.0}"),
        );
        format!(
            "Suggested starting point: reduce your current GPU core-frequency limit by a further {reduction:.0}%. The largest data set is {headroom:.1}% below the vector/matrix reference. The trial uses a {trial_headroom:.1}% gap against the smaller compute baseline as possible headroom. This changes only the frequency ceiling. Rerun the same data set and compare against its original throughput, then adjust in small steps to stay within 0–5% performance loss. Raise the limit if loss exceeds 5%. Check your apps too. This is untested guidance, not an automatically applied setting.{uncertainty}"
        )
    } else {
        format!(
            "No frequency-limit trial suggested: the reference, sample consistency, small-set agreement, or sustained memory-pressure evidence is insufficient. Largest/reference gap: {:.1}%; combined sample variation: {:.1}%. A throughput gap alone does not establish how much a lower core limit will affect performance.",
            (1.0 - ratio) * 100.0,
            gap_noise * 100.0
        )
    };
    metadata.insert("tuning_guidance".into(), guidance);
    metadata.insert("clock_analysis_limit".into(), "Estimate for this workload only. Core and memory clocks were not changed; frequency response, temperatures, power limits, and overclock stability were not measured.".into());
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metric(name: &str, value: f64) -> Metric {
        Metric {
            name: name.into(),
            value,
            unit: "operations/s".into(),
            statistics: SampleStatistics {
                sample_count: 3,
                minimum: value,
                median: value,
                maximum: value,
                standard_deviation: value * 0.01,
            },
        }
    }
    #[test]
    fn model_reports_delta_and_conservative_trial_instead_of_full_headroom() {
        let mut reference = BenchmarkResult {
            benchmark_id: "reference".into(),
            device_id: "gpu".into(),
            elapsed_ns: 1,
            metrics: vec![metric("throughput", 100.0)],
            device_metadata: BTreeMap::new(),
            workload_metadata: BTreeMap::from([(
                "bound_classification".into(),
                "compute_bound".into(),
            )]),
        };
        reference
            .workload_metadata
            .insert("compute_reference_drift_percent".into(), "0".into());
        let mut metrics = vec![
            metric("working_set_1.compute", 90.0),
            metric("working_set_2.compute", 50.0),
        ];
        let mut metadata = BTreeMap::new();
        add_reference_analysis(&mut metrics, &mut metadata, &reference, true);
        assert_eq!(metrics[0].name, "measured_compute_ceiling");
        assert_eq!(metrics.last().unwrap().value, -50.0);
        assert_eq!(metadata["idealized_core_headroom_percent"], "50.00");
        assert_eq!(metadata["idealized_bandwidth_increase_percent"], "100.00");
        assert_eq!(metadata["suggested_core_underclock_trial_percent"], "30");
        let mut check = reference.clone();
        check.metrics[0].value = 120.0;
        let checked = validate_reference(reference.clone(), check);
        assert_eq!(checked.metrics[0].value, 120.0);
        let mut drift_metrics = vec![
            metric("working_set_1.compute", 100.0),
            metric("working_set_2.compute", 50.0),
        ];
        let mut drift_metadata = BTreeMap::new();
        add_reference_analysis(&mut drift_metrics, &mut drift_metadata, &checked, true);
        assert_eq!(
            drift_metadata["suggested_core_underclock_trial_percent"],
            "35"
        );
        assert_eq!(drift_metadata["tuning_confidence"], "exploratory");
        for (small, large, deviation, transition) in [
            (40.0, 39.0, 0.01, true),
            (90.0, 50.0, 0.2, true),
            (90.0, 50.0, 0.01, false),
            (110.0, 120.0, 0.01, true),
        ] {
            let mut tiers = vec![
                metric("working_set_1.compute", small),
                metric("working_set_2.compute", large),
            ];
            tiers[1].statistics.standard_deviation = large * deviation;
            let mut meta = BTreeMap::new();
            add_reference_analysis(&mut tiers, &mut meta, &reference, transition);
            assert!(!meta.contains_key("suggested_core_underclock_trial_percent"));
        }
    }

    #[test]
    fn latest_lower_reference_drives_the_trial_instead_of_an_earlier_best() {
        let reference = |value| BenchmarkResult {
            benchmark_id: "gpu.performance.fp32".into(),
            device_id: "gpu".into(),
            elapsed_ns: 1,
            metrics: vec![metric("throughput", value)],
            device_metadata: BTreeMap::new(),
            workload_metadata: BTreeMap::from([(
                "bound_classification".into(),
                "compute_bound".into(),
            )]),
        };
        let latest = validate_reference(reference(100.0), reference(60.0));
        assert_eq!(latest.metrics[0].value, 60.0);
        assert_eq!(latest.metrics[0].statistics.median, 60.0);
        assert_eq!(
            latest.workload_metadata["compute_reference_selection"],
            "latest_post_sweep"
        );
        assert_eq!(
            latest.workload_metadata["compute_reference_drift_percent"],
            "40"
        );
        let mut metrics = vec![
            metric("working_set_1.compute", 90.0),
            metric("working_set_2.compute", 50.0),
        ];
        let mut metadata = BTreeMap::new();
        add_reference_analysis(&mut metrics, &mut metadata, &latest, true);
        assert_eq!(metrics[0].value, 60.0);
        assert_eq!(metadata["largest_compute_delta_percent"], "-16.67");
        assert_eq!(
            metadata["suggested_core_frequency_limit_reduction_percent"],
            "10"
        );
        assert_eq!(metadata["compute_reference_selection"], "latest_post_sweep");
    }

    #[test]
    fn matrix_gap_with_reference_drift_and_kernel_overhead_still_gets_a_trial() {
        let mut reference = BenchmarkResult {
            benchmark_id: "gpu.performance.matrix.fp16".into(),
            device_id: "gpu".into(),
            elapsed_ns: 1,
            metrics: vec![metric("throughput", 138.587246630601e12)],
            device_metadata: BTreeMap::new(),
            workload_metadata: BTreeMap::from([
                ("bound_classification".into(), "compute_bound".into()),
                ("compute_reference_drift_percent".into(), "10.16".into()),
            ]),
        };
        reference.metrics[0].statistics.standard_deviation = reference.metrics[0].value * 0.027603;
        let mut small = metric("working_set_262144.compute", 101.79079807733612e12);
        small.statistics.standard_deviation = small.value * 0.034199;
        let mut largest = metric("working_set_8589933568.compute", 48.39450891100962e12);
        largest.statistics.standard_deviation = largest.value * 0.064519;
        let mut metadata = BTreeMap::new();
        add_reference_analysis(&mut vec![small, largest], &mut metadata, &reference, true);
        assert_eq!(
            metadata["suggested_core_frequency_limit_reduction_percent"],
            "35"
        );
        assert_eq!(metadata["tuning_confidence"], "exploratory");
        assert!(metadata["tuning_guidance"].contains("changed by 10.2%"));
        assert!(metadata["tuning_guidance"].contains("avoid counting kernel overhead"));
    }

    #[test]
    fn meaningful_noisy_twelve_gib_drop_gets_an_exploratory_ten_percent_trial() {
        let mut reference = BenchmarkResult {
            benchmark_id: "gpu.performance.fp32".into(),
            device_id: "gpu".into(),
            elapsed_ns: 1,
            metrics: vec![metric("throughput", 51.15215688577806e12)],
            device_metadata: BTreeMap::new(),
            workload_metadata: BTreeMap::from([
                ("bound_classification".into(), "compute_bound".into()),
                ("compute_reference_drift_percent".into(), "0.14".into()),
            ]),
        };
        reference.metrics[0].statistics.standard_deviation = reference.metrics[0].value * 0.025704;
        let mut small = metric("working_set_258048.compute", 54.39027981764237e12);
        small.statistics.standard_deviation = small.value * 0.074747;
        let mut large = metric("working_set_12884889600.compute", 42.52320399493826e12);
        large.statistics.standard_deviation = large.value * 0.051440;
        let mut metrics = vec![small, large];
        let mut metadata = BTreeMap::new();
        add_reference_analysis(&mut metrics, &mut metadata, &reference, true);
        assert_eq!(
            metadata["suggested_core_frequency_limit_reduction_percent"],
            "10"
        );
        assert_eq!(metadata["tuning_confidence"], "exploratory");
        assert!(metadata["tuning_guidance"].contains("0–5% performance loss"));
        assert!(metadata["tuning_guidance"].contains("compare against its original throughput"));
        // A similarly noisy gap that is not distinguishable from variation is withheld.
        reference.metrics[0].statistics.standard_deviation = reference.metrics[0].value * 0.09;
        let mut metadata = BTreeMap::new();
        add_reference_analysis(&mut metrics, &mut metadata, &reference, true);
        assert!(!metadata.contains_key("suggested_core_frequency_limit_reduction_percent"));
    }
}
