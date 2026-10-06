use gluj_bench_core::Metric;

/// Both arrays contain paired sample timings normalized by full dataset passes per batch.
pub(super) fn append_metrics(
    metrics: &mut Vec<Metric>,
    bytes: u64,
    gpu: &[f64],
    end_to_end: &[f64],
) {
    for (suffix, samples) in [("gpu_execution_time", gpu), ("end_to_end_time", end_to_end)] {
        let stats = crate::statistics(samples);
        metrics.push(Metric {
            name: format!("working_set_{bytes}.{suffix}"),
            value: stats.median,
            unit: "ns".into(),
            statistics: stats,
        });
    }
}

pub(super) fn per_pass(batch_ns: f64, passes: u32) -> f64 {
    debug_assert!(passes > 0);
    batch_ns / f64::from(passes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_normalization_preserves_measured_sample_statistics_and_pacing_gap() {
        let gpu = [4000.0, 8000.0, 12000.0].map(|ns| per_pass(ns, 4));
        let wall = [8000.0, 16000.0, 24000.0].map(|ns| per_pass(ns, 4));
        let mut metrics = vec![];
        append_metrics(&mut metrics, 1048576, &gpu, &wall);
        assert_eq!(metrics[0].value, 2000.0);
        assert_eq!(metrics[1].value, 4000.0);
        assert_eq!(metrics[0].statistics.sample_count, 3);
        assert_eq!(metrics[0].statistics.minimum, 1000.0);
        assert_eq!(metrics[0].statistics.maximum, 3000.0);
        assert_eq!(metrics[1].unit, "ns");
        assert!(metrics.iter().all(|m| !m.name.contains("estimate")));
    }
}
