use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceCategory {
    Cpu,
    Memory,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkCategory {
    Cpu,
    Memory,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheKind {
    Data,
    Unified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheDescriptor {
    pub level: u8,
    pub kind: CacheKind,
    pub size_bytes: u64,
    pub line_size_bytes: u32,
    pub sharing_logical_processors: u32,
    pub instances: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceDescriptor {
    pub id: String,
    pub name: String,
    pub category: DeviceCategory,
    pub available: bool,
    pub status: String,
    #[serde(default)]
    pub properties: BTreeMap<String, String>,
    #[serde(default)]
    pub caches: Vec<CacheDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkDescriptor {
    pub id: String,
    pub name: String,
    pub category: BenchmarkCategory,
    pub workload: String,
    pub data_type: String,
    pub unit: String,
    #[serde(default)]
    pub supported_device_ids: Vec<String>,
    pub available: bool,
    pub unavailable_reason: String,
    #[serde(default)]
    pub suite_id: String,
    #[serde(default)]
    pub display_order: u32,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub target_duration_ms: u64,
    pub samples: u32,
    #[serde(default)]
    pub options: BTreeMap<String, String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            target_duration_ms: 5_000,
            samples: 5,
            options: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressUpdate {
    pub fraction: f64,
    pub phase: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SampleStatistics {
    pub sample_count: u32,
    pub minimum: f64,
    pub median: f64,
    pub maximum: f64,
    pub standard_deviation: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub statistics: SampleStatistics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_id: String,
    pub device_id: String,
    pub elapsed_ns: u64,
    #[serde(default)]
    pub metrics: Vec<Metric>,
    #[serde(default)]
    pub workload_metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub device_metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct BenchmarkError {
    pub code: String,
    pub message: String,
}

impl BenchmarkError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}
