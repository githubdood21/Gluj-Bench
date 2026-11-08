//! Compatibility catalogue for the existing runners. Family UI consolidation is separate.
use crate::{
    BenchmarkDescriptor, CONTRACT_VERSION, CombinationRule, ConditionDescriptor, FamilyDescriptor,
    MetricDirection, OptionDescriptor, OptionDomain,
};
use std::collections::BTreeMap;

fn choice(key: &str, values: &[&str], default: Option<&str>) -> OptionDescriptor {
    OptionDescriptor {
        key: key.into(),
        domain: OptionDomain::Choice {
            values: values.iter().map(|v| (*v).into()).collect(),
        },
        default: default.map(str::to_owned),
    }
}
fn integer(key: &str, minimum: u64, maximum: u64, default: Option<u64>) -> OptionDescriptor {
    OptionDescriptor {
        key: key.into(),
        domain: OptionDomain::Integer { minimum, maximum },
        default: default.map(|v| v.to_string()),
    }
}

/// Every current descriptor has an explicit destination. Unknown extensions keep their own family.
pub fn family_id(id: &str) -> &str {
    match id {
        "cpu.bandwidth.cache.l1"
        | "cpu.bandwidth.cache.l2"
        | "cpu.bandwidth.cache.l3"
        | "cpu.bandwidth.memory" => "cpu.memory_bandwidth",
        "cpu.latency.cache.l1"
        | "cpu.latency.cache.l2"
        | "cpu.latency.cache.l3"
        | "cpu.latency.memory"
        | "cpu.latency.memory.localized" => "cpu.dependent_latency",
        "cpu.performance.float.f32"
        | "cpu.performance.avx2.f32_fma"
        | "cpu.performance.avx2.f32_fma.scaling" => "cpu.fp32",
        "cpu.performance.float.f64" | "cpu.performance.avx2.f64_fma" => "cpu.fp64",
        "cpu.performance.matrix.fp32.scaling" => "cpu.matrix",
        "cpu.performance.integer.i64" | "cpu.performance.single_thread.integer.i64" => "cpu.int64",
        "cpu.performance.string.ascii" => "cpu.ascii",
        "cpu.performance.compression.deflate" | "cpu.performance.decompression.deflate" => {
            "cpu.deflate"
        }
        "cpu.performance.prime.sieve" => "cpu.sieve",
        "cpu.performance.extended.popcnt" => "cpu.popcnt",
        "cpu.performance.extended.aes" => "cpu.aes_rounds",
        "gpu.bandwidth.cache" | "gpu.bandwidth.vram" => "gpu.local_bandwidth",
        "gpu.bandwidth.host_link" => "gpu.host_transfers",
        "gpu.performance.fp16"
        | "gpu.performance.fp32"
        | "gpu.performance.fp64"
        | "gpu.performance.fp32.scaling"
        | "gpu.performance.fp32.offload.scaling" => "gpu.vector",
        value if value.starts_with("gpu.performance.matrix.") => "gpu.matrix",
        _ => id,
    }
}

pub fn legacy_condition(descriptor: &BenchmarkDescriptor) -> ConditionDescriptor {
    let id = descriptor.id.as_str();
    let gpu = id.starts_with("gpu.");
    let latency = id.starts_with("cpu.latency.");
    let single = id == "cpu.performance.single_thread.integer.i64";
    let cpu_profile = matches!(
        id,
        "cpu.performance.avx2.f32_fma.scaling"
            | "cpu.performance.matrix.fp32.scaling"
            | "cpu.performance.integer.i64"
            | "cpu.performance.string.ascii"
    ) || crate::cpu_data_mode(id);
    let gpu_profile = matches!(
        id,
        "gpu.performance.fp32.scaling"
            | "gpu.performance.fp16"
            | "gpu.performance.fp64"
            | "gpu.performance.fp32.offload.scaling"
            | "gpu.performance.matrix.fp16.scaling"
            | "gpu.performance.matrix.int8.scaling"
    );
    let offload = id == "gpu.performance.fp32.offload.scaling";
    let memory = gpu_memory(id);
    let transfer = gpu_transfer(id);

    let mut options = vec![choice(
        if gpu {
            "gpu_activity_percent"
        } else {
            "cpu_activity_percent"
        },
        &["95", "99", "100"],
        Some("100"),
    )];
    if gpu {
        options.push(OptionDescriptor {
            key: "device_id".into(),
            domain: OptionDomain::Device,
            default: None,
        });
    } else if !latency && !single {
        // Resolve one common default instead of different hidden CLI/UI defaults.
        options.push(choice(
            "thread_mode",
            &["physical_cores", "logical_processors"],
            Some("physical_cores"),
        ));
        options.push(integer("cpu_core_limit", 0, u32::MAX.into(), Some(0)));
    }
    let mut combinations = Vec::new();
    options.push(choice(
        "measurement_policy",
        &["extensive", "shorter"],
        Some("extensive"),
    ));
    if !gpu
        || crate::gpu_buffered_vector(id)
        || gpu_memory(id)
        || gpu_transfer(id)
        || (gpu_profile && id.contains("matrix"))
    {
        options.push(integer("sustained_seconds", 0, 120, Some(0)));
    }
    if !gpu && !latency && !single {
        combinations.push(CombinationRule::NonZeroRequiresValue {
            key: "cpu_core_limit".into(),
            required_key: "thread_mode".into(),
            required_value: "physical_cores".into(),
        });
    }
    if memory || transfer {
        options.push(choice(
            "dataset_mode",
            &["automatic", "single", "sweep"],
            Some("automatic"),
        ));
        options.push(integer("dataset_bytes", 65536, u64::MAX, None));
        options.push(integer("vram_budget_percent", 20, 80, Some(25)));
        if transfer {
            options.push(integer("ram_budget_percent", 20, 80, Some(20)));
        }
        for mode in ["single", "sweep"] {
            combinations.push(CombinationRule::Requires {
                key: "dataset_mode".into(),
                value: mode.into(),
                required_key: "dataset_bytes".into(),
            });
        }
    }
    if memory {
        options.push(choice(
            "memory_operation",
            &["all", "read", "write", "copy"],
            Some("all"),
        ));
        options.push(choice(
            "memory_copy_path",
            &["shader", "native"],
            Some(if id == "gpu.bandwidth.vram" {
                "native"
            } else {
                "shader"
            }),
        ));
        options.push(choice(
            "access_pattern",
            &[
                "sequential",
                "shuffled_blocks",
                "rotating_regions",
                "lane_scattered",
            ],
            Some("sequential"),
        ));
    }
    if transfer {
        options.push(choice(
            "transfer_direction",
            &["both", "host_to_device", "device_to_host"],
            Some("both"),
        ));
        options.push(choice("staging_buffers", &["1", "3", "8"], Some("3")));
        options.push(choice("transfer_batch", &["1", "8", "32"], Some("8")));
        options.push(choice(
            "transfer_submission",
            &["serial", "batched"],
            Some("batched"),
        ));
    }
    if cpu_profile || gpu_profile {
        let has_register = !matches!(
            id,
            "cpu.performance.matrix.fp32.scaling"
                | "cpu.performance.string.ascii"
                | "gpu.performance.fp32.offload.scaling"
        );
        options.push(choice(
            "dataset_mode",
            if has_register {
                &["automatic", "single", "sweep", "register"]
            } else {
                &["automatic", "single", "sweep"]
            },
            Some("automatic"),
        ));
        options.push(integer("dataset_bytes", 1, u64::MAX, None));
        if cpu_profile || offload {
            options.push(integer("ram_budget_percent", 20, 80, Some(20)));
        }
        if cpu_profile && id != "cpu.performance.string.ascii" && !crate::cpu_data_mode(id) {
            options.push(integer("ram_budget_mib", 4, 4096, None));
            combinations.push(CombinationRule::Exclusive {
                keys: vec!["ram_budget_mib".into(), "ram_budget_percent".into()],
            });
        }
        if gpu_profile {
            options.push(integer("vram_budget_percent", 20, 80, Some(25)));
        }
        let (key, values, default): (&str, &[&str], &str) = match id {
            "cpu.performance.avx2.f32_fma.scaling" => (
                "vector_preset",
                &["legacy", "register", "l1", "l2", "l3", "ram", "custom"],
                "legacy",
            ),
            "cpu.performance.float.f64" | "cpu.performance.avx2.f64_fma" => (
                "fp64_preset",
                &["register", "l1", "l2", "l3", "ram", "custom"],
                "register",
            ),
            "cpu.performance.extended.popcnt" => (
                "popcnt_preset",
                &["register", "l1", "l2", "l3", "ram", "custom"],
                "register",
            ),
            "cpu.performance.integer.i64" => (
                "int64_preset",
                &["legacy", "register", "l1", "l2", "l3", "ram", "custom"],
                "register",
            ),
            "cpu.performance.matrix.fp32.scaling" => (
                "matrix_preset",
                &["legacy", "compute", "l1", "l2", "l3", "ram", "custom"],
                "legacy",
            ),
            "cpu.performance.string.ascii" => (
                "ascii_preset",
                &["legacy", "compute", "l1", "l2", "l3", "ram", "custom"],
                "compute",
            ),
            "gpu.performance.fp32.scaling" | "gpu.performance.fp16" | "gpu.performance.fp64" => (
                "gpu_preset",
                &["legacy", "register", "small", "vram", "custom"],
                "legacy",
            ),
            "gpu.performance.fp32.offload.scaling" => (
                "gpu_offload_preset",
                &["legacy", "small", "vram", "budget", "custom"],
                "legacy",
            ),
            "gpu.performance.matrix.fp16.scaling" => (
                "gpu_matrix_preset",
                &["legacy", "register", "small", "vram", "custom"],
                "legacy",
            ),
            _ => (
                "gpu_int8_matrix_preset",
                &["legacy", "register", "small", "vram", "custom"],
                "legacy",
            ),
        };
        let default = if matches!(id, "gpu.performance.fp16" | "gpu.performance.fp64") {
            "register"
        } else {
            default
        };
        options.push(choice(key, values, Some(default)));
        combinations.push(CombinationRule::Requires {
            key: key.into(),
            value: "custom".into(),
            required_key: "dataset_bytes".into(),
        });
        combinations.push(CombinationRule::Excludes {
            key: key.into(),
            value: "register".into(),
            excluded_key: "dataset_bytes".into(),
        });
        // Register mode must select an actual register kernel, not bypass a buffer control.
        combinations.push(CombinationRule::RequiresValue {
            key: "dataset_mode".into(),
            value: "register".into(),
            required_key: key.into(),
            required_value: "register".into(),
        });
        combinations.push(CombinationRule::RequiresValue {
            key: key.into(),
            value: "register".into(),
            required_key: "dataset_mode".into(),
            required_value: "register".into(),
        });
        combinations.push(CombinationRule::AllowedWith {
            key: "dataset_bytes".into(),
            other_key: key.into(),
            values: vec!["legacy".into(), "custom".into()],
        });
        combinations.push(CombinationRule::AllowedWith {
            key: "dataset_bytes".into(),
            other_key: "dataset_mode".into(),
            values: vec!["single".into(), "sweep".into()],
        });
        for mode in ["single", "sweep"] {
            combinations.push(CombinationRule::RequiresWhen {
                conditions: BTreeMap::from([
                    (key.into(), "legacy".into()),
                    ("dataset_mode".into(), mode.into()),
                ]),
                required_key: "dataset_bytes".into(),
            });
        }
        if has_register {
            for option in [
                "matrix_tile_reuse",
                "arithmetic_iterations",
                "max_working_set_bytes",
                "ram_budget_mib",
                "ram_budget_percent",
                "vram_budget_percent",
            ] {
                combinations.push(CombinationRule::AllowedWith {
                    key: option.into(),
                    other_key: "dataset_mode".into(),
                    values: vec!["automatic".into(), "single".into(), "sweep".into()],
                });
            }
        }
    }
    if matches!(
        id,
        "gpu.performance.fp32.scaling"
            | "gpu.performance.fp32.offload.scaling"
            | "gpu.performance.fp16"
            | "gpu.performance.fp64"
    ) {
        options.push(integer("max_working_set_bytes", 262_144, u64::MAX, None));
        options.push(integer("arithmetic_iterations", 1, 1024, Some(64)));
    }
    if gpu_profile && id.contains("matrix") {
        options.push(integer("matrix_tile_reuse", 1, 64, Some(8)));
        options.push(choice(
            "matrix_tile_shape",
            &["automatic"],
            Some("automatic"),
        ));
        options.push(choice(
            "matrix_traversal",
            &["sequential", "shuffled_tiles", "rotating_tiles"],
            Some("sequential"),
        ));
        for key in [
            "matrix_traversal",
            "access_seed",
            "access_seed_policy",
            "access_realizations",
        ] {
            combinations.push(CombinationRule::AllowedWith {
                key: key.into(),
                other_key: "dataset_mode".into(),
                values: vec!["automatic".into(), "single".into(), "sweep".into()],
            });
        }
    }
    if offload {
        options.push(choice(
            "ram_offload_percent",
            &["50", "75", "100"],
            Some("75"),
        ));
        options.push(integer("offload_passes_per_second", 1, 1000, None));
        options.push(integer(
            "offload_traffic_bytes_per_second",
            1_048_576,
            68_719_476_736,
            None,
        ));
        combinations.push(CombinationRule::Exclusive {
            keys: vec![
                "offload_passes_per_second".into(),
                "offload_traffic_bytes_per_second".into(),
            ],
        });
    }
    if crate::access_supported(id)
        || latency
        || crate::cpu_codec(id)
        || crate::gpu_buffered_vector(id)
        || memory
        || (gpu_profile && id.contains("matrix"))
    {
        options.push(integer(
            "access_seed",
            0,
            u64::MAX,
            Some(crate::DEFAULT_ACCESS_SEED),
        ));
        options.push(choice(
            "access_seed_policy",
            &["fixed", "fresh"],
            Some("fixed"),
        ));
        options.push(choice("access_realizations", &["1", "3"], Some("1")));
    }
    if id == "cpu.performance.string.ascii" {
        options.push(choice(
            "ascii_content",
            &["patterned", "mixed", "no_hits", "all_hits"],
            Some("patterned"),
        ));
    }
    if latency {
        options.push(integer("dataset_bytes", 64, u64::MAX, None));
        if !matches!(id, "cpu.latency.cache.l1" | "cpu.latency.memory") {
            options.push(integer("locality_block_bytes", 64, 1048576, Some(65536)));
        }
    }
    if crate::access_supported(id) || crate::cpu_codec(id) || crate::gpu_buffered_vector(id) {
        options.push(choice(
            "access_pattern",
            if crate::gpu_buffered_vector(id) {
                &[
                    "sequential",
                    "shuffled_blocks",
                    "rotating_regions",
                    "lane_scattered",
                ]
            } else if crate::cpu_codec(id) {
                &["sequential", "shuffled_blocks"]
            } else {
                &[
                    "sequential",
                    "shuffled_blocks",
                    "rotating_regions",
                    "scattered",
                    "hot_cold",
                ]
            },
            Some("sequential"),
        ));
        if !crate::cpu_codec(id) && !crate::gpu_buffered_vector(id) {
            options.push(integer("access_block_bytes", 64, 1048576, Some(65536)));
        }
    }
    if matches!(
        id,
        "cpu.performance.integer.i64" | "cpu.performance.avx2.f32_fma.scaling"
    ) || crate::cpu_data_mode(id)
        || (crate::gpu_buffered_vector(id) && !offload)
    {
        for key in [
            "access_pattern",
            "access_seed",
            "access_seed_policy",
            "access_realizations",
            "access_block_bytes",
        ] {
            combinations.push(CombinationRule::AllowedWith {
                key: key.into(),
                other_key: "dataset_mode".into(),
                values: vec!["automatic".into(), "single".into(), "sweep".into()],
            });
        }
    }
    if id == "cpu.bandwidth.memory" {
        options.push(choice(
            "dataset_mode",
            &["automatic", "single", "sweep"],
            Some("automatic"),
        ));
        options.push(integer("dataset_bytes", 65536, u64::MAX, None));
        for value in ["single", "sweep"] {
            combinations.push(CombinationRule::Requires {
                key: "dataset_mode".into(),
                value: value.into(),
                required_key: "dataset_bytes".into(),
            });
        }
        combinations.push(CombinationRule::AllowedWith {
            key: "dataset_bytes".into(),
            other_key: "dataset_mode".into(),
            values: vec!["single".into(), "sweep".into()],
        });
    }
    if crate::cpu_codec(id) {
        options.push(choice(
            "codec_corpus",
            &["compressible", "mixed", "incompressible"],
            Some("mixed"),
        ));
        options.push(integer(
            "codec_corpus_bytes",
            65536,
            268435456,
            Some(16777216),
        ));
        options.push(integer("codec_block_bytes", 4096, 4194304, Some(65536)));
        options.push(integer("codec_level", 0, 9, Some(6)));
    }
    if id == "cpu.performance.prime.sieve" {
        options.push(integer("sieve_limit", 10000, 100000000, Some(1000000)));
    }
    if id == "cpu.performance.matrix.fp32.scaling" {
        options.push(choice(
            "matrix_shape",
            &["legacy", "square", "tall", "wide"],
            Some("legacy"),
        ));
        options.push(choice("matrix_reuse", &["1", "4", "16"], Some("1")));
        options.push(choice("matrix_k_block", &["32", "128", "256"], Some("128")));
    }
    if crate::cpu_data_mode(id)
        || crate::cpu_codec(id)
        || id == "cpu.performance.matrix.fp32.scaling"
    {
        options.push(choice(
            "core_experiment",
            &["none", "fixed_aggregate"],
            Some("none"),
        ));
    }
    ConditionDescriptor {
        version: CONTRACT_VERSION,
        benchmark_id: id.into(),
        family_id: family_id(id).into(),
        available: descriptor.available,
        unavailable_reason: descriptor.unavailable_reason.clone(),
        supported_device_ids: descriptor.supported_device_ids.clone(),
        options,
        combinations,
        metric_direction: if latency {
            MetricDirection::Lower
        } else {
            MetricDirection::Higher
        },
        timing_scope: if gpu_profile {
            "per_metric_gpu_active_and_host_elapsed"
        } else if gpu {
            "per_metric_vulkan_or_host_transfer_elapsed"
        } else {
            "active_execution_excluding_pacing"
        }
        .into(),
        kernel_revision: descriptor.metadata.get("kernel_revision").cloned(),
        capabilities: vec![],
    }
}

pub fn family_catalogue(conditions: Vec<ConditionDescriptor>) -> Vec<FamilyDescriptor> {
    let mut families = BTreeMap::<String, Vec<ConditionDescriptor>>::new();
    for condition in conditions {
        families
            .entry(condition.family_id.clone())
            .or_default()
            .push(condition);
    }
    families
        .into_iter()
        .map(|(id, conditions)| FamilyDescriptor {
            version: CONTRACT_VERSION,
            id,
            conditions,
        })
        .collect()
}

/// Modes sharing the buffered vector harness; register-only FP32 remains a compatibility ID.
pub fn gpu_buffered_vector(id: &str) -> bool {
    matches!(
        id,
        "gpu.performance.fp16"
            | "gpu.performance.fp64"
            | "gpu.performance.fp32.scaling"
            | "gpu.performance.fp32.offload.scaling"
    )
}

pub fn gpu_memory(id: &str) -> bool {
    matches!(id, "gpu.bandwidth.cache" | "gpu.bandwidth.vram")
}
pub fn gpu_transfer(id: &str) -> bool {
    id == "gpu.bandwidth.host_link"
}
