mod queue_chart;
use super::{BenchmarkResult, ChartLabel, ChartPoint, Color, Metric, format_binary_size};

pub(super) fn format_time(ns: f64) -> String {
    if ns.abs() < 1000.0 {
        format!("{ns:.1} ns")
    } else if ns.abs() < 1e6 {
        format!("{:.2} µs", ns / 1000.0)
    } else if ns.abs() < 1e9 {
        format!("{:.2} ms", ns / 1e6)
    } else {
        format!("{:.2} s", ns / 1e9)
    }
}

fn metric<'a>(result: &'a BenchmarkResult, bytes: u64, suffix: &str) -> Option<&'a Metric> {
    result.metrics.iter().find(|m| {
        m.name == format!("working_set_{bytes}.{suffix}")
            && m.unit == "ns"
            && m.value.is_finite()
            && m.value > 0.0
    })
}

pub(super) fn point(result: &BenchmarkResult, bytes: u64) -> Option<(&Metric, &Metric)> {
    if !result.benchmark_id.starts_with("gpu.") || !result.benchmark_id.ends_with(".scaling") {
        return None;
    }
    Some((
        metric(result, bytes, "gpu_execution_time")?,
        metric(result, bytes, "end_to_end_time")?,
    ))
}

pub(super) fn batch_summary(result: &BenchmarkResult, bytes: u64) -> Option<String> {
    let series = result
        .workload_metadata
        .get("gpu_timing_batch_passes_series")?;
    let counts = series.split(',').find_map(|pair| {
        let (size, counts) = pair.split_once(':')?;
        (size.parse::<u64>().ok()? == bytes).then_some(counts)
    })?;
    let counts: Vec<u32> = counts
        .split('/')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    let (minimum, maximum) = (counts.iter().min()?, counts.iter().max()?);
    Some(format!(
        "{} observations; {minimum} to {maximum} passes per observation. Times use each observation's actual work.",
        counts.len()
    ))
}

#[cfg(test)]
fn batch_passes(result: &BenchmarkResult, bytes: u64) -> Option<u32> {
    result
        .workload_metadata
        .get("gpu_timing_batch_passes")?
        .split(',')
        .find_map(|pair| {
            let (size, passes) = pair.split_once(':')?;
            (size.parse::<u64>().ok()? == bytes)
                .then(|| passes.parse::<u32>().ok())
                .flatten()
                .filter(|count| *count > 0)
        })
}

#[cfg(test)]
pub(super) fn details(result: &BenchmarkResult, bytes: u64, throughput: &Metric) -> String {
    let mut text = String::new();
    if super::is_offered_load(result) {
        for (suffix, label) in [
            ("offered_requests", "Requested"),
            ("submitted_requests", "Submitted"),
            ("completed_requests", "Completed before cutoff"),
            ("unserved_requests", "Pending at cutoff"),
        ] {
            if let Some(m) = super::demand_metric(result, bytes, suffix) {
                text.push_str(&format!(
                    "\n{label}: {:.0} requests (sample median)",
                    m.value
                ));
            }
        }
        for (suffix, label) in [
            ("queue_wait", "Arrival to submission"),
            ("oldest_pending_age", "Oldest pending request"),
        ] {
            if let Some(m) = super::demand_metric(result, bytes, suffix) {
                text.push_str(&format!("\n{label}: {}", format_time(m.value)));
            }
        }
    }
    if let Some((gpu, wall)) = point(result, bytes) {
        for (label, metric) in [
            (
                if super::is_offered_load(result) {
                    "GPU service per request"
                } else {
                    "GPU execution per pass"
                },
                gpu,
            ),
            (
                if super::is_offered_load(result) {
                    "Request completion delay (arrival to observed completion)"
                } else {
                    "End-to-end per pass (incl. pacing)"
                },
                wall,
            ),
        ] {
            text.push_str(&format!(
                "\n{label}: {} (min {}, max {}; {} samples)",
                format_time(metric.value),
                format_time(metric.statistics.minimum),
                format_time(metric.statistics.maximum),
                metric.statistics.sample_count
            ));
        }
        if !super::is_offered_load(result)
            && let Some(passes) = batch_passes(result, bytes)
        {
            text.push_str(&format!(
                "\nSample batch: {passes} full dataset passes; timings normalized per pass."
            ));
        }
    }
    if let Some(density) = result
        .workload_metadata
        .get("gpu_profile_points_per_doubling")
    {
        text.push_str(&format!("\nSweep density: {density} measured intervals per size doubling. Each plotted point has its own repeated samples; denser sweeps take longer without shortening each point's sample budget."));
    }
    if let Some(start) = result
        .workload_metadata
        .get("gpu_saturation_start_bytes")
        .and_then(|s| s.parse::<u64>().ok())
    {
        text.push_str(&format!("\nAutomatic start: {} from same-kernel VRAM calibration; smaller warm-up probes are excluded from results. Targets below this start are raised within budget.", super::format_binary_size(start as f64)));
        if result
            .workload_metadata
            .get("gpu_saturation_boundary_peak")
            .is_some_and(|s| s == "true")
        {
            text.push_str(" Best throughput was still rising at the calibration limit; this is the best observed start, not a confirmed plateau.");
        }
        if result
            .workload_metadata
            .get("gpu_saturation_peak_repeatable")
            .is_some_and(|s| s == "false")
        {
            text.push_str(" Calibration was noisy; repeat before judging the peak.");
        }
    }
    if !super::is_offered_load(result)
        && result
            .workload_metadata
            .contains_key("gpu_command_recording_policy")
    {
        text.push_str("\nSample batch is pre-recorded and warmed; end-to-end sample time covers submission, completion and activity pacing.");
    }
    if let Some(groups) = result
        .workload_metadata
        .get("gpu_profile_workgroups_per_pass")
        .and_then(|values| {
            values.split(',').find_map(|pair| {
                let (size, groups) = pair.split_once(':')?;
                (size.parse::<u64>().ok()? == bytes).then_some(groups)
            })
        })
    {
        text.push_str(&format!(
            "\nWorkgroups per pass: {groups} (separate region/chunk dispatches)."
        ));
        if let Some(size) = result.workload_metadata.get("gpu_profile_workgroup_size") {
            text.push_str(&format!("\nThreads per workgroup: {size} (launch shape measured before the sweep; data and arithmetic unchanged)."));
        }
        text.push_str("\nSmall datasets may not fill the GPU; increasing dataset size can raise throughput before memory pressure appears. Warmup is excluded from the recorded samples.");
    }
    if super::gpu_offload(&result.benchmark_id)
        && let Some(share) = result
            .workload_metadata
            .get("ram_offload_percent")
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|s| *s <= 100)
    {
        let host = bytes / 100 * share + bytes % 100 * share / 100;
        if let Some(target) = result
            .workload_metadata
            .get("offload_traffic_bytes_per_second")
            .and_then(|s| s.parse::<f64>().ok())
        {
            let target = if result
                .workload_metadata
                .contains_key("offload_traffic_ramp_revision")
            {
                gluj_bench_core::dynamic_offload_traffic(target, bytes)
            } else {
                target
            };
            text.push_str(&format!(
                "\nDynamic traffic target: {}/s total · {}/s RAM · {:.6} passes/s at this size.",
                super::format_binary_size(target),
                super::format_binary_size(target * share as f64 / 100.0),
                target / bytes as f64
            ));
        }
        if let Some(target) = result
            .workload_metadata
            .get("offload_passes_per_second")
            .and_then(|s| s.parse::<f64>().ok())
        {
            text.push_str(&format!(
                "\nTraffic target: {target:.0} passes/s · {}/s total · {}/s RAM.",
                super::format_binary_size(bytes as f64 * target),
                super::format_binary_size(host as f64 * target)
            ));
        }
        if let Some(rate) = result.metrics.iter().find(|m| {
            m.name == format!("working_set_{bytes}.pass_rate")
                && m.value.is_finite()
                && m.value > 0.0
        }) {
            text.push_str(&format!("\nAchieved average over observation time: {:.2} passes/s · {}/s total · {}/s RAM (nominal accesses, not PCIe counters).", rate.value,
                super::format_binary_size(bytes as f64 * rate.value), super::format_binary_size(host as f64 * rate.value)));
        }

        text.push_str(&format!(
            "\nDataset split: {} RAM + {} VRAM (both inputs and output combined).",
            super::format_binary_size(host as f64),
            super::format_binary_size((bytes - host) as f64)
        ));
    }
    if throughput.name.ends_with(".active_compute") {
        if result
            .metrics
            .iter()
            .any(|m| m.name == format!("working_set_{bytes}.active_host_bandwidth"))
        {
            text.push_str("\nCompute and the traffic graph both use active GPU time. Achieved passes/s above includes intentional waits.");
        } else {
            text.push_str("\nActive host traffic was not recorded in this saved run; active compute and traffic comparison is unavailable. Averaged traffic remains in the raw measurements.");
        }
    }
    if !super::is_offered_load(result)
        && result.benchmark_id.starts_with("gpu.")
        && throughput.unit == "operations/s"
        && throughput.value.is_finite()
        && throughput.value > 0.0
        && let Some(reference) = super::dataset_compute_reference(result)
    {
        let retained = throughput.value / reference.value * 100.0;
        if retained.is_finite() {
            text.push_str(&format!(
                "\nThroughput retained vs compute reference: {retained:.1}%"
            ));
        }
    }
    text
}

pub(super) fn note(result: &BenchmarkResult) -> String {
    if super::is_offered_load(result) {
        return "Average time from request arrival to completion, including queueing. Red marks unfinished requests at cutoff. Hover for completed/pending counts and wait time. Gaps mean no requests completed.".into();
    }
    if !result.benchmark_id.starts_with("gpu.") {
        return String::new();
    }
    if !super::scaling_compute_tiers(result)
        .iter()
        .any(|(bytes, _)| point(result, *bytes).is_some())
    {
        return "Per-dataset timing data is not recorded in this result. Older throughput values are not converted into unrecorded timings.".into();
    }
    let mut text = if super::is_offered_load(result) {
        "GPU service measures execution after preceding output dependencies. Request completion measures arrival to host-observed completion, including queueing, submission and polling. Requests arrive independently of completions; activity pacing is disabled. Pending work at cutoff indicates unmet demand, not a failed request. Drain time receives no throughput credit.".to_owned()
    } else {
        gluj_bench_core::GPU_TIMING_EXPLANATION.to_owned()
    };
    if result
        .metrics
        .iter()
        .filter(|m| m.name.ends_with(".gpu_execution_time") || m.name.ends_with(".end_to_end_time"))
        .any(|m| !super::consistent_metric(m))
    {
        text.push_str(" Observed timing variation and recorded sample counts describe this run; small differences are descriptive.");
    }
    text
}

#[cfg(test)]
pub(super) fn chart(result: &BenchmarkResult) -> Option<(String, Vec<ChartLabel>)> {
    interactive_chart(result).map(|(svg, labels, _)| (svg, labels))
}

pub(super) fn interactive_chart(
    result: &BenchmarkResult,
) -> Option<(String, Vec<ChartLabel>, Vec<ChartPoint>)> {
    if super::is_offered_load(result) {
        return queue_chart::chart(result);
    }
    use std::fmt::Write;
    let points = super::scaling_compute_tiers(result)
        .into_iter()
        .filter(|(bytes, _)| *bytes > 0)
        .map(|(bytes, _)| (bytes, point(result, bytes)))
        .collect::<Vec<_>>();
    let valid = points
        .iter()
        .filter_map(|(bytes, pair)| pair.map(|pair| (*bytes, pair)))
        .collect::<Vec<_>>();
    let (first, last) = (valid.first()?, valid.last()?);
    let maximum = valid
        .iter()
        .flat_map(|(_, (gpu, wall))| [gpu, wall])
        .map(|m| super::chart_sample_range(m).1)
        .fold(0.0, f64::max)
        * 1.08;
    let (scale, unit) = if maximum < 1000.0 {
        (1.0, "ns")
    } else if maximum < 1e6 {
        (1000.0, "µs")
    } else if maximum < 1e9 {
        (1e6, "ms")
    } else {
        (1e9, "s")
    };
    let span = ((last.0 as f64).log2() - (first.0 as f64).log2()).max(1.0);
    let x = |bytes: u64| {
        if first.0 == last.0 {
            490.0
        } else {
            110.0 + ((bytes as f64).log2() - (first.0 as f64).log2()) / span * 760.0
        }
    };
    let y = |ns: f64| 220.0 - ns / maximum * 160.0;
    let label = |x, y, width, text: String, size, accent, alignment| ChartLabel {
        x,
        y,
        width,
        text: text.into(),
        font_size: size,
        accent,
        alignment,
    };
    let brass = Color::from_rgb_u8(220, 198, 107);
    let iris = Color::from_rgb_u8(172, 165, 227);
    let neutral = Color::from_rgb_u8(198, 191, 206);
    let mut labels = vec![
        label(
            80.0,
            8.0,
            790.0,
            format!("Measured time per dataset pass · {unit} · lower is faster"),
            16.0,
            neutral,
            0,
        ),
        label(
            80.0,
            32.0,
            330.0,
            (if super::is_offered_load(result) {
                "GPU service"
            } else {
                "GPU execution"
            })
            .into(),
            13.0,
            brass,
            0,
        ),
        label(
            440.0,
            32.0,
            400.0,
            (if super::is_offered_load(result) {
                "Request completion delay"
            } else {
                "End-to-end (includes pacing)"
            })
            .into(),
            13.0,
            iris,
            0,
        ),
    ];
    let mut svg = String::from(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="900" height="280" viewBox="0 0 900 280"><rect width="900" height="280" rx="10" fill="#222129"/>"##,
    );
    let mut hover_points = Vec::new();
    for tick in 0..=4 {
        let ns = maximum * tick as f64 / 4.0;
        writeln!(
            svg,
            r##"<line x1="110" x2="870" y1="{0:.2}" y2="{0:.2}" stroke="#474352"/>"##,
            y(ns)
        )
        .unwrap();
        labels.push(label(
            0.0,
            (y(ns) - 9.0) as f32,
            100.0,
            format!("{:.2} {unit}", ns / scale),
            13.0,
            neutral,
            2,
        ));
    }
    for pair in points.windows(2) {
        if let (Some(a), Some(b)) = (pair[0].1, pair[1].1) {
            for (from, to, color) in [(a.0, b.0, "#dcc66b"), (a.1, b.1, "#aca5e3")] {
                let (from_min, from_max) = super::chart_sample_range(from);
                let (to_min, to_max) = super::chart_sample_range(to);
                writeln!(svg, r#"<polygon points="{:.2},{:.2} {:.2},{:.2} {:.2},{:.2} {:.2},{:.2}" fill="{color}" fill-opacity="0.16"/><line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{color}" stroke-width="2.5"/>"#,
                    x(pair[0].0), y(from_min), x(pair[0].0), y(from_max), x(pair[1].0), y(to_max), x(pair[1].0), y(to_min),
                    x(pair[0].0), y(from.value), x(pair[1].0), y(to.value)).unwrap();
            }
        }
    }
    for (bytes, (gpu, wall)) in &valid {
        for (metric, color) in [(*gpu, "#dcc66b"), (*wall, "#aca5e3")] {
            hover_points.push(super::chart_hover::point(
                *bytes,
                metric,
                [x(*bytes), y(metric.value), 220.0],
                (
                    if color == "#dcc66b" {
                        if super::is_offered_load(result) {
                            "GPU service per request"
                        } else {
                            "GPU execution per pass"
                        }
                    } else if super::is_offered_load(result) {
                        "Request completion delay"
                    } else {
                        "End-to-end per pass"
                    },
                    scale,
                    unit,
                ),
                if color == "#dcc66b" { brass } else { iris },
            ));
            writeln!(
                svg,
                r#"<circle cx="{:.2}" cy="{:.2}" r="3.5" fill="{color}"/>"#,
                x(*bytes),
                y(metric.value)
            )
            .unwrap();
        }
    }
    let mut previous_index = usize::MAX;
    let mut previous_x = -100.0;
    for tick in 0..=4 {
        let index = (valid.len() - 1) * tick / 4;
        let bytes = valid[index].0;
        let position = x(bytes);
        if index == previous_index
            || (tick != 4 && position - previous_x < 120.0)
            || (valid.len() > 1 && tick != 4 && index != 0 && x(last.0) - position < 120.0)
        {
            continue;
        }
        previous_index = index;
        previous_x = position;
        let (left, alignment) = if valid.len() == 1 {
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
            if valid.len() == 1 { 160.0 } else { 140.0 },
            format_binary_size(bytes as f64),
            13.0,
            neutral,
            alignment,
        ));
    }
    labels.push(label(
        80.0,
        257.0,
        790.0,
        if valid.len() == 1 {
            "Tested dataset"
        } else {
            "Dataset size (logarithmic scale) →"
        }
        .into(),
        12.0,
        neutral,
        1,
    ));
    svg.push_str("</svg>");
    Some((svg, labels, hover_points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gluj_bench_core::SampleStatistics;
    use std::collections::BTreeMap;
    fn result() -> BenchmarkResult {
        let metric = |name: &str, value, unit: &str| Metric {
            name: name.into(),
            value,
            unit: unit.into(),
            statistics: SampleStatistics {
                sample_count: 3,
                minimum: value * 0.9,
                median: value,
                maximum: value * 1.1,
                standard_deviation: value * 0.01,
            },
        };
        BenchmarkResult {
            benchmark_id: "gpu.performance.fp32.offload.scaling".into(),
            device_id: "gpu:test".into(),
            elapsed_ns: 1,
            metrics: vec![
                metric("measured_compute_ceiling", 100.0, "operations/s"),
                metric("working_set_262144.compute", 120.0, "operations/s"),
                metric("working_set_1048576.compute", 25.0, "operations/s"),
                metric("working_set_262144.gpu_execution_time", 1000.0, "ns"),
                metric("working_set_262144.end_to_end_time", 2000.0, "ns"),
                metric("working_set_1048576.gpu_execution_time", 10000.0, "ns"),
                metric("working_set_1048576.end_to_end_time", 20000.0, "ns"),
            ],
            workload_metadata: BTreeMap::from([(
                "gpu_timing_batch_passes".into(),
                "262144:64,1048576:8".into(),
            )]),
            device_metadata: BTreeMap::new(),
        }
    }
    #[test]
    fn graphs_and_selected_dataset_display_real_times_and_batch_context() {
        let result = result();
        let (svg, labels) = chart(&result).unwrap();
        assert_eq!(svg.matches("<circle").count(), 4);
        assert_eq!(svg.matches("<polygon").count(), 2);
        assert!(labels.iter().any(|l| l.text.contains("µs")));
        assert!(
            labels
                .iter()
                .all(|l| !l.text.contains("wait") && !l.text.contains("busy"))
        );
        let text = details(&result, 262144, &result.metrics[1]);
        assert!(text.contains("1.00 µs"));
        assert!(text.contains("64 full dataset passes"));
        assert!(text.contains("120.0%"));
        let image = slint::Image::load_from_svg_data(svg.as_bytes())
            .unwrap()
            .to_rgba8()
            .unwrap();
        assert_eq!((image.width(), image.height()), (900, 280));
    }
    #[test]
    fn offload_details_explain_actual_workgroups_split_and_sampled_boundary() {
        let mut result = result();
        result.workload_metadata.extend([
            (
                "gpu_profile_workgroups_per_pass".into(),
                "262144:16+6,1048576:64+22".into(),
            ),
            ("ram_offload_percent".into(), "75".into()),
            ("bandwidth_transition_status".into(), "observed".into()),
            (
                "bandwidth_transition_previous_working_set_bytes".into(),
                "2097152".into(),
            ),
            (
                "bandwidth_transition_working_set_bytes".into(),
                "4194304".into(),
            ),
        ]);
        let text = details(&result, 262144, &result.metrics[1]);
        assert!(text.contains("16+6"));
        assert!(text.contains("192.00 KiB RAM + 64.00 KiB VRAM"));
        assert!(text.contains("may not fill the GPU"));
        let row = super::super::result_row(&result, &[], &[]);
        assert!(row.diagnosis.contains("2.00 MiB to 4.00 MiB"));
        assert!(row.diagnosis.contains("not a measured cache capacity"));
    }

    #[test]
    fn traffic_target_and_achieved_rate_are_distinct_from_resident_size() {
        let mut result = result();
        result
            .workload_metadata
            .insert("ram_offload_percent".into(), "75".into());
        result
            .workload_metadata
            .insert("offload_passes_per_second".into(), "100".into());
        let mut rate = result.metrics[1].clone();
        rate.name = "working_set_262144.pass_rate".into();
        rate.unit = "passes/s".into();
        rate.value = 98.0;
        result.metrics.push(rate);
        let text = details(&result, 262144, &result.metrics[1]);
        assert!(text.contains("100 passes/s · 25.00 MiB/s total · 18.75 MiB/s RAM"));
        assert!(text.contains("98.00 passes/s · 24.50 MiB/s total"));
        assert!(text.contains("not PCIe counters"));
        result.workload_metadata.remove("offload_passes_per_second");
        result
            .workload_metadata
            .insert("offload_traffic_bytes_per_second".into(), "8388608".into());
        let dynamic = details(&result, 262144, &result.metrics[1]);
        assert!(dynamic.contains(
            "Dynamic traffic target: 8.00 MiB/s total · 6.00 MiB/s RAM · 32.000000 passes/s"
        ));
    }

    #[test]
    fn old_results_are_not_back_calculated_and_cpu_results_are_untouched() {
        let mut result = result();
        result.metrics.retain(|m| m.unit != "ns");
        assert!(chart(&result).is_none());
        assert!(note(&result).contains("not recorded"));
        assert!(!note(&result).to_lowercase().contains("rerun"));
        result.benchmark_id = "cpu.performance.matrix.fp32.scaling".into();
        assert!(note(&result).is_empty());
        assert!(details(&result, 262144, &result.metrics[1]).is_empty());
    }
    #[test]
    fn single_dataset_and_invalid_timing_gaps_are_supported() {
        let mut result = result();
        let mut gap = result.clone();
        gap.metrics.push(Metric {
            name: "working_set_524288.compute".into(),
            value: 50.0,
            unit: "operations/s".into(),
            statistics: SampleStatistics::default(),
        });
        let (svg, _) = chart(&gap).unwrap();
        assert_eq!(svg.matches("<circle").count(), 4);
        assert!(!svg.contains("stroke-width=\"2.5\""));
        result.metrics.remove(1);
        let (svg, labels) = chart(&result).unwrap();
        assert_eq!(svg.matches("<circle").count(), 2);
        assert!(labels.iter().any(|l| l.text == "Tested dataset"));
        assert!(svg.contains("cx=\"490.00\""));
        assert_eq!(format_time(250.0), "250.0 ns");
        assert_eq!(format_time(2e6), "2.00 ms");
        assert_eq!(format_time(2e9), "2.00 s");
    }
}
