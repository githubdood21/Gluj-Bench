//! Shared CPU workload identities and option validation.
use crate::{BenchmarkConfig, BenchmarkError};
pub fn cpu_data_mode(id: &str) -> bool {
    matches!(
        id,
        "cpu.performance.float.f64"
            | "cpu.performance.avx2.f64_fma"
            | "cpu.performance.extended.popcnt"
    )
}
pub fn cpu_codec(id: &str) -> bool {
    matches!(
        id,
        "cpu.performance.compression.deflate" | "cpu.performance.decompression.deflate"
    )
}
pub fn cpu_data_preset_key(id: &str) -> &str {
    if id == "cpu.performance.extended.popcnt" {
        "popcnt_preset"
    } else {
        "fp64_preset"
    }
}
pub fn cpu_matrix_shape(name: &str) -> Option<[usize; 3]> {
    match name {
        "square" => Some([128, 128, 128]),
        "tall" => Some([512, 128, 128]),
        "wide" => Some([128, 512, 128]),
        _ => None,
    }
}
pub fn validate_cpu_workload_options(
    id: &str,
    config: &BenchmarkConfig,
) -> Result<(), BenchmarkError> {
    let invalid = |s: &str| BenchmarkError::new("invalid_config", s);
    if config
        .options
        .get("core_experiment")
        .is_some_and(|v| v == "fixed_aggregate")
        && (config
            .options
            .get("dataset_mode")
            .is_some_and(|v| v == "register")
            || (id == "cpu.performance.matrix.fp32.scaling"
                && config
                    .options
                    .get("matrix_shape")
                    .is_none_or(|v| v == "legacy")))
    {
        return Err(invalid(
            "Fixed aggregate core experiments require buffered data or an explicit global matrix shape.",
        ));
    }
    if cpu_codec(id) {
        let bytes = config
            .options
            .get("codec_corpus_bytes")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(16 * 1024 * 1024);
        let block = config
            .options
            .get("codec_block_bytes")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(65536);
        let cores = config
            .options
            .get("cpu_core_limit")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        if cores > bytes.div_ceil(block) {
            return Err(invalid(
                "Choose at least one independent codec block per selected worker.",
            ));
        }
    }
    if id == "cpu.performance.matrix.fp32.scaling" {
        let shape = config
            .options
            .get("matrix_shape")
            .map(String::as_str)
            .unwrap_or("legacy");
        let reuse = config
            .options
            .get("matrix_reuse")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1);
        let block = config
            .options
            .get("matrix_k_block")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(128);
        if shape == "legacy" {
            if reuse != 1 || block != 128 {
                return Err(invalid(
                    "Matrix reuse and blocking controls require an explicit matrix shape.",
                ));
            }
        } else {
            let Some([m, _, _]) = cpu_matrix_shape(shape) else {
                return Err(invalid("Unknown CPU matrix shape."));
            };
            if config.options.contains_key("dataset_bytes")
                || config
                    .options
                    .get("matrix_preset")
                    .is_some_and(|s| s != "legacy")
                || config
                    .options
                    .get("dataset_mode")
                    .is_some_and(|s| s != "automatic")
            {
                return Err(invalid(
                    "Explicit matrix shapes define their allocation; remove dataset size/preset controls.",
                ));
            }
            let cores = config
                .options
                .get("cpu_core_limit")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            if cores > m / 8 {
                return Err(invalid(
                    "This matrix shape needs at least eight output rows per worker; choose fewer cores or a taller shape.",
                ));
            }
        }
    }
    Ok(())
}
