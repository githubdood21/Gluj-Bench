//! Versioned workload contracts. Scheduling policy is deliberately outside workload identity.
use crate::{BenchmarkConfig, BenchmarkError, BenchmarkResult, DeviceDescriptor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CONTRACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    Higher,
    Lower,
    Descriptive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OptionDomain {
    Choice { values: Vec<String> },
    Integer { minimum: u64, maximum: u64 },
    Device,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionDescriptor {
    pub key: String,
    pub domain: OptionDomain,
    pub default: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CombinationRule {
    Exclusive {
        keys: Vec<String>,
    },
    Requires {
        key: String,
        value: String,
        required_key: String,
    },
    Excludes {
        key: String,
        value: String,
        excluded_key: String,
    },
    RequiresValue {
        key: String,
        value: String,
        required_key: String,
        required_value: String,
    },
    AllowedWith {
        key: String,
        other_key: String,
        values: Vec<String>,
    },
    RequiresWhen {
        conditions: BTreeMap<String, String>,
        required_key: String,
    },
    NonZeroRequiresValue {
        key: String,
        required_key: String,
        required_value: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionDescriptor {
    pub version: u32,
    pub benchmark_id: String,
    pub family_id: String,
    pub available: bool,
    pub unavailable_reason: String,
    pub supported_device_ids: Vec<String>,
    pub options: Vec<OptionDescriptor>,
    pub combinations: Vec<CombinationRule>,
    pub metric_direction: MetricDirection,
    /// Metric-specific timing definitions remain in the result; this is the advertised scope.
    pub timing_scope: String,
    pub kernel_revision: Option<String>,
    pub capabilities: Vec<ModeCapability>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModeCapability {
    pub when: BTreeMap<String, String>,
    pub supported_device_ids: Vec<String>,
    pub unavailable_reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyDescriptor {
    pub version: u32,
    pub id: String,
    pub conditions: Vec<ConditionDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementPolicy {
    pub revision: String,
    pub target_duration_ms: u64,
    pub samples: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedCondition {
    pub version: u32,
    pub family_id: String,
    pub benchmark_id: String,
    pub device_id: String,
    pub options: BTreeMap<String, String>,
    pub policy: MeasurementPolicy,
    pub kernel_revision: Option<String>,
    pub timing_scope: String,
}

impl ResolvedCondition {
    pub fn config(&self) -> BenchmarkConfig {
        let mut options = self.options.clone();
        if self.policy.revision.starts_with("bounded-") {
            options.insert(
                "measurement_policy".into(),
                if self.policy.revision.contains("shorter") {
                    "shorter"
                } else {
                    "extensive"
                }
                .into(),
            );
        }
        BenchmarkConfig {
            target_duration_ms: self.policy.target_duration_ms,
            samples: self.policy.samples,
            options,
        }
    }
}

impl ConditionDescriptor {
    /// The same function is used by request previews and the executing registry.
    pub fn resolve(
        &self,
        config: &BenchmarkConfig,
        devices: &[DeviceDescriptor],
    ) -> Result<ResolvedCondition, BenchmarkError> {
        let measured_config = crate::resolve_measurement_config(config)?;
        let config = &measured_config;
        let invalid = |message: String| BenchmarkError::new("invalid_config", message);
        if self.version != CONTRACT_VERSION {
            return Err(invalid("Unsupported condition contract version.".into()));
        }
        if !self.available {
            return Err(BenchmarkError::new(
                "benchmark_unavailable",
                &self.unavailable_reason,
            ));
        }
        if config.samples == 0 || config.target_duration_ms == 0 {
            return Err(invalid(
                "Samples and target duration must be positive.".into(),
            ));
        }
        let mut options = BTreeMap::new();
        for key in config.options.keys() {
            if !self.options.iter().any(|o| &o.key == key) {
                return Err(BenchmarkError::new(
                    "unsupported_option",
                    format!("{} does not support '{key}'.", self.benchmark_id),
                ));
            }
        }
        for option in &self.options {
            let Some(value) = config.options.get(&option.key).or(option.default.as_ref()) else {
                continue;
            };
            let valid = match &option.domain {
                OptionDomain::Choice { values } => values.contains(value),
                OptionDomain::Integer { minimum, maximum } => value
                    .parse::<u64>()
                    .is_ok_and(|v| (*minimum..=*maximum).contains(&v)),
                OptionDomain::Device => self.supported_device_ids.contains(value),
            };
            if !valid {
                return Err(invalid(format!(
                    "Unsupported value '{value}' for '{}'.",
                    option.key
                )));
            }
            // Canonicalize integers so equivalent explicit configurations have one identity.
            let value = if matches!(option.domain, OptionDomain::Integer { .. }) {
                value.parse::<u64>().expect("validated integer").to_string()
            } else {
                value.clone()
            };
            options.insert(option.key.clone(), value);
        }
        // Contextual defaults describe the actual legacy mode, rather than passing
        // an ignored buffer setting to a register kernel.
        if let Some((key, preset)) = options
            .iter()
            .find(|(key, _)| key.ends_with("_preset"))
            .map(|(k, v)| (k.clone(), v.clone()))
            && !config.options.contains_key("dataset_mode")
        {
            if preset == "register" {
                options.insert("dataset_mode".into(), "register".into());
            } else if preset != "legacy"
                && (key.starts_with("gpu_") || key == "matrix_preset" || key == "ascii_preset")
            {
                options.insert("dataset_mode".into(), "single".into());
            }
        }
        if crate::access_supported(&self.benchmark_id)
            || self.benchmark_id.starts_with("cpu.latency.")
        {
            let access = crate::AccessPolicy::from_config(&BenchmarkConfig {
                options: options.clone(),
                ..config.clone()
            })?;
            if options.get("dataset_mode").is_some_and(|v| v == "register")
                && (access.pattern != crate::AccessPattern::Sequential || access.realizations != 1)
            {
                return Err(invalid(
                    "Access patterns require a buffered workload; choose sequential for registers."
                        .into(),
                ));
            }
        }
        // Inapplicable defaults are omitted. Explicit inapplicable controls fail below.
        for rule in &self.combinations {
            if let CombinationRule::Exclusive { keys } = rule
                && keys.iter().any(|key| config.options.contains_key(key))
            {
                for key in keys {
                    if !config.options.contains_key(key) {
                        options.remove(key);
                    }
                }
            }
            if let CombinationRule::AllowedWith {
                key,
                other_key,
                values,
            } = rule
                && !config.options.contains_key(key)
                && !options.get(other_key).is_some_and(|v| values.contains(v))
            {
                options.remove(key);
            }
        }
        for rule in &self.combinations {
            match rule {
                CombinationRule::Exclusive { keys }
                    if keys.iter().filter(|k| options.contains_key(*k)).count() > 1 =>
                {
                    return Err(invalid(format!("Choose only one of {}.", keys.join(", "))));
                }
                CombinationRule::Requires {
                    key,
                    value,
                    required_key,
                } if options.get(key) == Some(value) && !options.contains_key(required_key) => {
                    return Err(invalid(format!("{key}={value} requires {required_key}.")));
                }
                CombinationRule::Excludes {
                    key,
                    value,
                    excluded_key,
                } if options.get(key) == Some(value) && options.contains_key(excluded_key) => {
                    return Err(invalid(format!(
                        "{excluded_key} is not supported with {key}={value}."
                    )));
                }
                CombinationRule::RequiresValue {
                    key,
                    value,
                    required_key,
                    required_value,
                } if options.get(key) == Some(value)
                    && options.get(required_key) != Some(required_value) =>
                {
                    return Err(invalid(format!(
                        "{key}={value} requires {required_key}={required_value}."
                    )));
                }
                CombinationRule::AllowedWith {
                    key,
                    other_key,
                    values,
                } if options.contains_key(key)
                    && !options.get(other_key).is_some_and(|v| values.contains(v)) =>
                {
                    return Err(invalid(format!(
                        "{key} requires {other_key} to be one of {}.",
                        values.join(", ")
                    )));
                }
                CombinationRule::RequiresWhen {
                    conditions,
                    required_key,
                } if conditions.iter().all(|(k, v)| options.get(k) == Some(v))
                    && !options.contains_key(required_key) =>
                {
                    return Err(invalid(format!(
                        "The selected combination requires {required_key}."
                    )));
                }
                CombinationRule::NonZeroRequiresValue {
                    key,
                    required_key,
                    required_value,
                } if options
                    .get(key)
                    .and_then(|v| v.parse::<u64>().ok())
                    .is_some_and(|n| n > 0)
                    && options.get(required_key) != Some(required_value) =>
                {
                    return Err(invalid(format!(
                        "Non-zero {key} requires {required_key}={required_value}."
                    )));
                }
                _ => {}
            }
        }
        let applicable: Vec<_> = self
            .capabilities
            .iter()
            .filter(|cap| {
                cap.when
                    .iter()
                    .all(|(key, value)| options.get(key) == Some(value))
            })
            .collect();
        let supports_mode = |id: &String| {
            applicable
                .iter()
                .all(|cap| cap.supported_device_ids.contains(id))
        };
        let device_id = options
            .get("device_id")
            .cloned()
            .or_else(|| {
                self.supported_device_ids
                    .iter()
                    .find(|id| {
                        supports_mode(id) && devices.iter().any(|d| &d.id == *id && d.available)
                    })
                    .cloned()
            })
            .ok_or_else(|| {
                BenchmarkError::new("device_unavailable", "No supported device is available.")
            })?;
        if let Some(cap) = applicable
            .iter()
            .find(|cap| !cap.supported_device_ids.contains(&device_id))
        {
            return Err(BenchmarkError::new(
                "mode_unavailable",
                &cap.unavailable_reason,
            ));
        }
        if !devices.iter().any(|d| d.id == device_id && d.available) {
            return Err(BenchmarkError::new(
                "device_unavailable",
                format!("Device '{device_id}' is unavailable."),
            ));
        }
        if options.get("cpu_core_limit").is_some_and(|v| v == "0")
            && options
                .get("thread_mode")
                .is_some_and(|v| v == "physical_cores")
            && let Some(count) = devices
                .iter()
                .find(|d| d.id == device_id)
                .and_then(|d| d.properties.get("physical_cores"))
        {
            options.insert("cpu_core_limit".into(), count.clone());
        }
        if crate::gpu_memory(&self.benchmark_id)
            && options
                .get("memory_copy_path")
                .is_some_and(|v| v == "native")
            && options
                .get("memory_operation")
                .is_some_and(|v| matches!(v.as_str(), "all" | "copy"))
            && options
                .get("access_pattern")
                .is_some_and(|v| v != "sequential")
        {
            return Err(invalid(
                "Native copy requires sequential access; select shader copy for irregular access."
                    .into(),
            ));
        }
        if crate::gpu_transfer(&self.benchmark_id)
            && options
                .get("transfer_submission")
                .is_some_and(|v| v == "serial")
            && options.get("transfer_batch").is_some_and(|v| v != "1")
        {
            return Err(invalid(
                "Serial transfer submission requires transfer_batch=1.".into(),
            ));
        }
        crate::validate_cpu_workload_options(
            &self.benchmark_id,
            &BenchmarkConfig {
                options: options.clone(),
                ..config.clone()
            },
        )?;
        if self.options.iter().any(|o| o.key == "device_id") {
            options.insert("device_id".into(), device_id.clone());
        }
        let policy_name = options
            .remove("measurement_policy")
            .unwrap_or_else(|| "extensive".into());
        Ok(ResolvedCondition {
            version: CONTRACT_VERSION,
            family_id: self.family_id.clone(),
            benchmark_id: self.benchmark_id.clone(),
            device_id,
            options,
            policy: MeasurementPolicy {
                revision: format!("bounded-{policy_name}-v1"),
                target_duration_ms: config.target_duration_ms,
                samples: config.samples,
            },
            kernel_revision: self.kernel_revision.clone(),
            timing_scope: self.timing_scope.clone(),
        })
    }
}

/// Requested workload plus actual execution facts. Missing legacy facts stay missing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionIdentity {
    pub resolved: ResolvedCondition,
    pub actual: BTreeMap<String, String>,
    pub complete: bool,
}

impl ConditionIdentity {
    pub fn from_result(resolved: ResolvedCondition, result: &BenchmarkResult) -> Self {
        let mut actual = BTreeMap::new();
        // Include definitions, never observed performance or changing calibration measurements.
        for (key, value) in &result.workload_metadata {
            if key.ends_with("_revision")
                || key.ends_with("_definition")
                || matches!(
                    key.as_str(),
                    "data_type"
                        | "thread_count"
                        | "thread_mode"
                        | "cpu_core_limit"
                        | "working_set_bytes"
                        | "compute_scaling_tier_bytes"
                        | "offload_tier_bytes"
                        | "matrix_shape"
                        | "matrix_tile_reuse"
                        | "host_memory_placement"
                        | "gpu_timing_method"
                        | "seed"
                        | "generator_revision"
                        | "access_pattern"
                        | "auxiliary_bytes"
                        | "dataset_bytes"
                        | "dataset_mode"
                )
            {
                actual.insert(key.clone(), value.clone());
            }
        }
        // Legacy runners do not yet advertise a complete identity for every kernel.
        Self {
            resolved,
            actual,
            complete: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "reasons", rename_all = "snake_case")]
pub enum Comparability {
    Comparable,
    DescriptiveOnly(Vec<String>),
    Incompatible(Vec<String>),
}

pub fn compare_conditions(left: &ConditionIdentity, right: &ConditionIdentity) -> Comparability {
    let mut reasons = Vec::new();
    let a = &left.resolved;
    let b = &right.resolved;
    for (same, reason) in [
        (a.version == b.version, "condition contract version differs"),
        (
            a.benchmark_id == b.benchmark_id,
            "benchmark/kernel identity differs",
        ),
        (a.device_id == b.device_id, "device differs"),
        (a.options == b.options, "workload options differ"),
        (
            a.kernel_revision == b.kernel_revision,
            "kernel revision differs",
        ),
        (a.timing_scope == b.timing_scope, "timing scope differs"),
        (
            left.actual == right.actual,
            "actual workload definitions or allocation differ",
        ),
    ] {
        if !same {
            reasons.push(reason.into());
        }
    }
    if !reasons.is_empty() {
        return Comparability::Incompatible(reasons);
    }
    if a.policy != b.policy {
        reasons.push("measurement policy differs; compare descriptively".into());
    }
    if !left.complete || !right.complete {
        reasons.push("legacy workload identity is incomplete".into());
    }
    if a.kernel_revision.is_none() {
        reasons.push("kernel revision was not recorded".into());
    }
    if reasons.is_empty() {
        Comparability::Comparable
    } else {
        Comparability::DescriptiveOnly(reasons)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStatus {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationStatus {
    pub status: String,
    pub scope: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawSample {
    #[serde(default)]
    pub timing_scope: SampleTimingScope,
    pub version: u32,
    pub sequence: u64,
    pub block_id: String,
    pub realization_id: String,
    pub condition_id: String,
    pub metric: String,
    pub value: f64,
    pub unit: String,
    pub useful_work: u64,
    pub useful_work_unit: String,
    pub active_ns: u64,
    pub elapsed_ns: u64,
    pub offset_ns: u64,
    pub reference_anchor: Option<String>,
    pub validation: ValidationStatus,
    pub status: CompletionStatus,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleTimingScope {
    #[default]
    ActiveExecution,
    GpuTimestampAndHostElapsed,
    OfferedWindowWithAggregateService,
    HostCopyElapsed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseTiming {
    pub phase: String,
    pub condition: String,
    pub offset_ns: u64,
    pub elapsed_ns: u64,
    /// Worker/reference intervals can overlap; do not sum them as a global wall clock.
    pub inclusive: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadRealization {
    pub id: String,
    pub seed: Option<u64>,
    pub generator_revision: Option<String>,
    pub parameters: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleBlock {
    pub id: String,
    pub realization_id: String,
    pub order: u64,
    pub preparation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceAnchor {
    pub id: String,
    pub sample_sequences: Vec<u64>,
    pub scope: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuiteCondition {
    pub id: String,
    pub condition: ResolvedCondition,
    pub intended_coverage: Vec<String>,
    pub estimated_duration_ms: Option<u64>,
    pub estimated_peak_bytes: Option<u64>,
    pub status: CompletionStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuitePlan {
    pub version: u32,
    pub id: String,
    pub order_seed: u64,
    /// Vector order is authoritative; seeds never imply implicit reordering.
    pub conditions: Vec<SuiteCondition>,
}

impl SuitePlan {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        let mut ids = std::collections::BTreeSet::new();
        if self.version != CONTRACT_VERSION {
            return Err(BenchmarkError::new(
                "invalid_suite",
                "Unsupported suite version.",
            ));
        }
        for (index, entry) in self.conditions.iter().enumerate() {
            if entry.id.is_empty()
                || !ids.insert(&entry.id)
                || entry.condition.version != CONTRACT_VERSION
            {
                return Err(BenchmarkError::new(
                    "invalid_suite",
                    "Invalid or repeated condition ID/version.",
                ));
            }
            if self.conditions[..index]
                .iter()
                .any(|old| old.condition == entry.condition)
            {
                return Err(BenchmarkError::new(
                    "duplicate_condition",
                    "Suite repeats an identical resolved condition.",
                ));
            }
        }
        Ok(())
    }
}
