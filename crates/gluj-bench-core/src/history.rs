//! Native immutable run records. The compact index is a rebuildable cache, never the source of truth.
use crate::{
    BenchmarkConfig, BenchmarkError, BenchmarkResult, CONTRACT_VERSION, CompletionStatus,
    ConditionIdentity, DeviceDescriptor, RawSample, ResolvedCondition, ValidationStatus,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

pub const RUN_FORMAT_VERSION: u32 = 1;
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunSource {
    LocalMeasurement,
    LegacyLocal,
    ImportedComparison,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleCoverage {
    SummaryOnly,
    Partial,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub version: u32,
    pub run_id: String,
    pub source: RunSource,
    /// None for old files without a recorded timestamp; migration time is not run time.
    pub started_unix_ms: Option<u64>,
    pub application_version: Option<String>,
    pub build_revision: Option<String>,
    pub hardware: Vec<DeviceDescriptor>,
    pub requested_benchmark_id: String,
    pub requested_config: Option<BenchmarkConfig>,
    pub resolved: Option<ResolvedCondition>,
    pub identity: Option<ConditionIdentity>,
    pub status: CompletionStatus,
    pub validation: ValidationStatus,
    pub sample_coverage: SampleCoverage,
    pub samples: Vec<RawSample>,
    pub realizations: Vec<crate::WorkloadRealization>,
    pub blocks: Vec<crate::SampleBlock>,
    pub reference_anchors: Vec<crate::ReferenceAnchor>,
    #[serde(default)]
    pub phase_timings: Vec<crate::PhaseTiming>,
    /// Exact original summary, including old IDs and revisions. UI normalization uses a copy.
    pub result: Option<BenchmarkResult>,
    pub error: Option<BenchmarkError>,
    pub source_reference: Option<String>,
}

impl RunRecord {
    pub fn start(id: &str, config: &BenchmarkConfig, hardware: Vec<DeviceDescriptor>) -> Self {
        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Self {
            version: RUN_FORMAT_VERSION,
            run_id: format!(
                "{:x}-{:x}-{:x}",
                elapsed.as_nanos(),
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ),
            source: RunSource::LocalMeasurement,
            started_unix_ms: u64::try_from(elapsed.as_millis()).ok(),
            application_version: Some(env!("CARGO_PKG_VERSION").into()),
            build_revision: option_env!("GLUJ_BENCH_BUILD_REVISION").map(str::to_owned),
            hardware,
            requested_benchmark_id: id.into(),
            requested_config: Some(config.clone()),
            resolved: None,
            identity: None,
            status: CompletionStatus::Running,
            validation: ValidationStatus {
                status: "not_recorded".into(),
                scope: "legacy runner".into(),
            },
            sample_coverage: SampleCoverage::SummaryOnly,
            samples: vec![],
            realizations: vec![],
            blocks: vec![],
            reference_anchors: vec![],
            phase_timings: vec![],
            result: None,
            error: None,
            source_reference: None,
        }
    }

    pub fn finish(&mut self, outcome: Result<BenchmarkResult, BenchmarkError>) {
        match outcome {
            Ok(result) => {
                if let Some(resolved) = self.resolved.clone() {
                    self.identity = Some(ConditionIdentity::from_result(resolved, &result));
                }
                if let Some(scope) = result.workload_metadata.get("result_validation") {
                    self.validation = ValidationStatus {
                        status: "runner_reported".into(),
                        scope: scope.clone(),
                    };
                }
                self.result = Some(result);
                self.status = CompletionStatus::Completed;
            }
            Err(error) => {
                self.validation = ValidationStatus {
                    status: if error.code.contains("validation")
                        || error.code.contains("invalid_observation")
                    {
                        "failed"
                    } else {
                        "not_recorded"
                    }
                    .into(),
                    scope: error.message.clone(),
                };
                self.status = if error.code == "cancelled" {
                    CompletionStatus::Cancelled
                } else {
                    CompletionStatus::Failed
                };
                self.error = Some(error);
                self.result = None;
            }
        }
    }

    pub fn legacy(result: BenchmarkResult, source: RunSource, reference: String) -> Self {
        let mut record = Self::start(&result.benchmark_id, &BenchmarkConfig::default(), vec![]);
        record.started_unix_ms = result
            .workload_metadata
            .get("saved_at_unix_ms")
            .and_then(|v| v.parse().ok())
            .or_else(|| {
                result
                    .device_metadata
                    .get("saved_at")
                    .and_then(|v| v.parse::<u64>().ok())
                    .and_then(|v| v.checked_mul(1000))
            });
        record.application_version = None;
        record.build_revision = None;
        record.requested_config = None;
        record.source = source;
        record.source_reference = Some(reference);
        record.finish(Ok(result));
        record
    }

    pub fn validate(&self) -> Result<(), BenchmarkError> {
        let invalid = |s| BenchmarkError::new("invalid_run_record", s);
        if self.version != RUN_FORMAT_VERSION || !valid_id(&self.run_id) {
            return Err(invalid("Unsupported run version or unsafe run ID."));
        }
        if self
            .resolved
            .as_ref()
            .is_some_and(|condition| condition.version != CONTRACT_VERSION)
            || self
                .identity
                .as_ref()
                .is_some_and(|identity| self.resolved.as_ref() != Some(&identity.resolved))
        {
            return Err(invalid(
                "Unsupported or inconsistent resolved condition identity.",
            ));
        }
        if (self.status == CompletionStatus::Completed) != self.result.is_some() {
            return Err(invalid(
                "Only a completed run may contain a successful summary.",
            ));
        }
        if matches!(
            self.status,
            CompletionStatus::Failed | CompletionStatus::Cancelled
        ) && self.error.is_none()
        {
            return Err(invalid("Failed/cancelled runs require an error."));
        }
        if self.result.is_some() && self.error.is_some() {
            return Err(invalid("A run cannot be both successful and failed."));
        }
        if self.sample_coverage == SampleCoverage::SummaryOnly && !self.samples.is_empty() {
            return Err(invalid("Summary-only runs cannot claim raw samples."));
        }
        if self.sample_coverage == SampleCoverage::Complete
            && (self.samples.is_empty() || self.status != CompletionStatus::Completed)
        {
            return Err(invalid(
                "Complete sample coverage requires a completed run and real samples.",
            ));
        }
        let mut previous = None;
        let unique_nonempty = |ids: Vec<&str>| {
            let mut seen = std::collections::BTreeSet::new();
            ids.iter().all(|id| !id.is_empty() && seen.insert(*id))
        };
        if !unique_nonempty(self.realizations.iter().map(|r| r.id.as_str()).collect())
            || !unique_nonempty(self.blocks.iter().map(|r| r.id.as_str()).collect())
            || !unique_nonempty(
                self.reference_anchors
                    .iter()
                    .map(|r| r.id.as_str())
                    .collect(),
            )
        {
            return Err(invalid(
                "Realization, block and anchor IDs must be nonempty and unique.",
            ));
        }
        for block in &self.blocks {
            if !self
                .realizations
                .iter()
                .any(|r| r.id == block.realization_id)
            {
                return Err(invalid("Sample block refers to an unknown realization."));
            }
        }
        for sample in &self.samples {
            if !self
                .blocks
                .iter()
                .any(|b| b.id == sample.block_id && b.realization_id == sample.realization_id)
                || sample.condition_id.is_empty()
                || sample
                    .reference_anchor
                    .as_ref()
                    .is_some_and(|id| !self.reference_anchors.iter().any(|a| &a.id == id))
            {
                return Err(invalid(
                    "Sample refers to an unknown block, realization or anchor.",
                ));
            }
            if sample.version != CONTRACT_VERSION
                || !sample.value.is_finite()
                || (sample.active_ns > sample.elapsed_ns
                    && matches!(
                        sample.timing_scope,
                        crate::SampleTimingScope::ActiveExecution
                            | crate::SampleTimingScope::HostCopyElapsed
                    ))
                || previous.is_some_and(|(sequence, offset)| {
                    sample.sequence <= sequence || sample.offset_ns < offset
                })
            {
                return Err(invalid(
                    "Invalid sample version, timing, value or chronological order.",
                ));
            }
            previous = Some((sample.sequence, sample.offset_ns));
        }
        for anchor in &self.reference_anchors {
            if anchor.sample_sequences.is_empty()
                || anchor.sample_sequences.iter().any(|sequence| {
                    !self
                        .samples
                        .iter()
                        .any(|sample| &sample.sequence == sequence)
                })
            {
                return Err(invalid("Reference anchor must refer to recorded samples."));
            }
        }
        if let Some(result) = &self.result {
            let canonical_alias = self.requested_benchmark_id
                == "cpu.performance.single_thread.integer.i64"
                && result.benchmark_id == "cpu.performance.integer.i64";
            if (result.benchmark_id != self.requested_benchmark_id && !canonical_alias)
                || self
                    .resolved
                    .as_ref()
                    .is_some_and(|condition| condition.benchmark_id != result.benchmark_id)
                || self
                    .resolved
                    .as_ref()
                    .is_some_and(|condition| condition.device_id != result.device_id)
            {
                return Err(invalid(
                    "Result does not match the requested benchmark/device.",
                ));
            }
            for metric in &result.metrics {
                let s = &metric.statistics;
                if [
                    metric.value,
                    s.minimum,
                    s.median,
                    s.maximum,
                    s.standard_deviation,
                ]
                .iter()
                .any(|v| !v.is_finite())
                {
                    return Err(invalid("Metric values must be finite."));
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunIndexEntry {
    pub run_id: String,
    pub source: RunSource,
    pub started_unix_ms: Option<u64>,
    pub benchmark_id: String,
    pub status: CompletionStatus,
    pub sample_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunIndex {
    pub version: u32,
    pub runs: Vec<RunIndexEntry>,
}

pub struct RunStore {
    root: PathBuf,
}
impl RunStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn load(&self, id: &str) -> Result<RunRecord, BenchmarkError> {
        if !valid_id(id) {
            return Err(BenchmarkError::new("invalid_run_id", "Unsafe run ID."));
        }
        let record: RunRecord = serde_json::from_slice(
            &fs::read(self.root.join("runs").join(format!("{id}.json"))).map_err(storage_error)?,
        )
        .map_err(storage_error)?;
        record.validate()?;
        if record.run_id != id {
            return Err(BenchmarkError::new(
                "invalid_run_record",
                "Filename and run ID differ.",
            ));
        }
        Ok(record)
    }
    /// Commit the immutable full record first. Rebuilding an index can always recover it.
    pub fn save(&self, record: &RunRecord) -> Result<(), BenchmarkError> {
        record.validate()?;
        let directory = self.root.join("runs");
        fs::create_dir_all(&directory).map_err(storage_error)?;
        let path = directory.join(format!("{}.json", record.run_id));
        let bytes = serde_json::to_vec(record).map_err(storage_error)?;
        write_atomic(&path, &bytes, false).map_err(storage_error)?;
        self.refresh_index()?;
        Ok(())
    }
    // Normal commits read the compact cache and only load records missing from it.
    // This keeps large raw histories from being reparsed for every new measurement.
    fn refresh_index(&self) -> Result<(), BenchmarkError> {
        let cached = fs::read(self.root.join("index.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<RunIndex>(&bytes).ok());
        let Some(mut index) = cached.filter(|index| index.version == RUN_FORMAT_VERSION) else {
            self.rebuild_index()?;
            return Ok(());
        };
        let mut ids = std::collections::BTreeSet::new();
        if index
            .runs
            .iter()
            .any(|entry| !valid_id(&entry.run_id) || !ids.insert(entry.run_id.clone()))
        {
            self.rebuild_index()?;
            return Ok(());
        }
        for entry in fs::read_dir(self.root.join("runs")).map_err(storage_error)? {
            let path = entry.map_err(storage_error)?.path();
            if path.extension().and_then(|v| v.to_str()) != Some("json") {
                continue;
            }
            let id = path
                .file_stem()
                .and_then(|v| v.to_str())
                .ok_or_else(|| BenchmarkError::new("invalid_run_id", "Invalid run filename."))?;
            if ids.insert(id.into()) {
                index.runs.push(index_entry(&self.load(id)?));
            }
        }
        index.runs.sort_by(|a, b| {
            a.started_unix_ms
                .cmp(&b.started_unix_ms)
                .then(a.run_id.cmp(&b.run_id))
        });
        write_atomic(
            &self.root.join("index.json"),
            &serde_json::to_vec(&index).map_err(storage_error)?,
            true,
        )
        .map_err(storage_error)
    }
    /// Read authoritative records rather than trust a stale/missing index after a crash.
    /// Malformed/future records produce an error and are preserved.
    pub fn rebuild_index(&self) -> Result<RunIndex, BenchmarkError> {
        let mut runs = Vec::new();
        let directory = self.root.join("runs");
        if directory.exists() {
            for entry in fs::read_dir(directory).map_err(storage_error)? {
                let path = entry.map_err(storage_error)?.path();
                if path.extension().and_then(|v| v.to_str()) != Some("json") {
                    continue;
                }
                let id = path.file_stem().and_then(|v| v.to_str()).ok_or_else(|| {
                    BenchmarkError::new("invalid_run_id", "Invalid run filename.")
                })?;
                let record = self.load(id)?;
                runs.push(index_entry(&record));
            }
        }
        runs.sort_by(|a, b| {
            a.started_unix_ms
                .cmp(&b.started_unix_ms)
                .then(a.run_id.cmp(&b.run_id))
        });
        let index = RunIndex {
            version: RUN_FORMAT_VERSION,
            runs,
        };
        fs::create_dir_all(&self.root).map_err(storage_error)?;
        write_atomic(
            &self.root.join("index.json"),
            &serde_json::to_vec(&index).map_err(storage_error)?,
            true,
        )
        .map_err(storage_error)?;
        Ok(index)
    }
    /// Idempotent migration; source JSON is never rewritten or normalized here.
    pub fn import_legacy_results(
        &self,
        bytes: &[u8],
        reference: &str,
        source: RunSource,
    ) -> Result<usize, BenchmarkError> {
        #[derive(Deserialize)]
        struct Legacy {
            version: u32,
            results: Vec<BenchmarkResult>,
        }
        let file: Legacy = serde_json::from_slice(bytes).map_err(storage_error)?;
        if file.version != 1 {
            return Err(BenchmarkError::new(
                "unsupported_results_version",
                "Only version-1 legacy results can be migrated.",
            ));
        }
        let index = self.rebuild_index()?;
        let mut existing = index
            .runs
            .iter()
            .filter(|entry| entry.source == source)
            .map(|entry| self.load(&entry.run_id))
            .collect::<Result<Vec<_>, _>>()?;
        let mut imported = 0;
        for (ordinal, result) in file.results.into_iter().enumerate() {
            if source == RunSource::LegacyLocal
                && let Some(id) = result.device_metadata.get("native_run_id")
                && let Ok(native) = self.load(id)
                && native.source == RunSource::LocalMeasurement
                && native.result.as_ref().is_some_and(|original| {
                    original.benchmark_id == result.benchmark_id
                        && original.device_id == result.device_id
                        && original.elapsed_ns == result.elapsed_ns
                        && original.metrics == result.metrics
                        && original.workload_metadata == result.workload_metadata
                })
            {
                continue;
            }
            let reference = format!("{reference}#{ordinal}");
            if existing.iter().any(|record| {
                record.source_reference.as_deref() == Some(&reference)
                    && record.result.as_ref() == Some(&result)
            }) {
                continue;
            }
            let record = RunRecord::legacy(result, source.clone(), reference);
            self.save(&record)?;
            existing.push(record);
            imported += 1;
        }
        Ok(imported)
    }
}

fn index_entry(record: &RunRecord) -> RunIndexEntry {
    RunIndexEntry {
        run_id: record.run_id.clone(),
        source: record.source.clone(),
        started_unix_ms: record.started_unix_ms,
        benchmark_id: record.requested_benchmark_id.clone(),
        status: record.status.clone(),
        sample_count: record.samples.len(),
    }
}

/// Hardware identity is evaluated separately from workload identity; imports cannot
/// become local measurements just because their device IDs happen to match.
pub fn compare_runs(left: &RunRecord, right: &RunRecord) -> crate::Comparability {
    use crate::Comparability;
    if left.status != CompletionStatus::Completed || right.status != CompletionStatus::Completed {
        return Comparability::Incompatible(vec![
            "only completed runs have comparable scores".into(),
        ]);
    }
    let stable_hardware = |run: &RunRecord| {
        run.hardware
            .iter()
            .map(|device| {
                let properties: std::collections::BTreeMap<_, _> = device
                    .properties
                    .iter()
                    .filter(|(key, _)| key.as_str() != "available_bytes_at_discovery")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                (
                    device.id.clone(),
                    device.name.clone(),
                    properties,
                    device.caches.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    if !left.hardware.is_empty()
        && !right.hardware.is_empty()
        && stable_hardware(left) != stable_hardware(right)
    {
        return Comparability::Incompatible(vec!["hardware or driver context differs".into()]);
    }
    let (Some(a), Some(b)) = (&left.identity, &right.identity) else {
        return Comparability::DescriptiveOnly(vec![
            "legacy result lacks resolved workload identity".into(),
        ]);
    };
    let mut comparison = crate::compare_conditions(a, b);
    let mut limitations = Vec::new();
    if left.source != RunSource::LocalMeasurement || right.source != RunSource::LocalMeasurement {
        limitations.push("contains a legacy or imported comparison source".into());
    }
    if left.hardware.is_empty() || right.hardware.is_empty() {
        limitations.push("hardware context was not recorded".into());
    }
    if left.sample_coverage != SampleCoverage::Complete
        || right.sample_coverage != SampleCoverage::Complete
    {
        limitations.push("raw sample coverage is incomplete".into());
    }
    if !limitations.is_empty() {
        match &mut comparison {
            Comparability::Comparable => comparison = Comparability::DescriptiveOnly(limitations),
            Comparability::DescriptiveOnly(reasons) => reasons.extend(limitations),
            Comparability::Incompatible(_) => {}
        }
    }
    comparison
}

fn storage_error(error: impl std::fmt::Display) -> BenchmarkError {
    BenchmarkError::new("run_storage", error.to_string())
}

/// Flush before publish. Immutable commits use a hard link for atomic no-clobber publication.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8], replace: bool) -> io::Result<()> {
    let temporary = path.with_extension(format!(
        "{}.{}.tmp",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        if replace {
            fs::rename(&temporary, path)
        } else {
            fs::hard_link(&temporary, path)
        }
    })();
    let _ = fs::remove_file(&temporary);
    result
}
