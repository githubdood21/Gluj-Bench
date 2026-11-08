//! Explicit history curation; annotations never rewrite measured records.
use crate::{BenchmarkError, RunRecord, RunStore};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs};

pub const NATIVE_EXPORT_LIMIT: u64 = 256 * 1024 * 1024;
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunAnnotation {
    pub name: String,
    pub notes: String,
    pub pinned: bool,
    pub archived: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeExport {
    pub format: String,
    pub version: u32,
    pub records: Vec<RunRecord>,
    pub annotations: BTreeMap<String, RunAnnotation>,
}
fn error(e: impl std::fmt::Display) -> BenchmarkError {
    BenchmarkError::new("history_action", e.to_string())
}
impl NativeExport {
    pub fn decode(bytes: &[u8]) -> Result<Self, BenchmarkError> {
        if bytes.len() as u64 > NATIVE_EXPORT_LIMIT {
            return Err(error(
                "Native import exceeds 256 MiB; split the selected records.",
            ));
        }
        let bundle: Self = serde_json::from_slice(bytes).map_err(error)?;
        if bundle.format != "gluj-bench-native" || bundle.version != 1 || bundle.records.is_empty()
        {
            return Err(error("Unsupported or empty native export."));
        }
        let mut ids = std::collections::BTreeSet::new();
        for record in &bundle.records {
            record.validate()?;
            if !ids.insert(&record.run_id) {
                return Err(error("Duplicate exported run ID."));
            }
        }
        if bundle.annotations.keys().any(|id| !ids.contains(id)) {
            return Err(error("Annotation refers to an absent exported run."));
        }
        validate_annotations(&bundle.annotations)?;
        Ok(bundle)
    }
}
fn validate_annotations(items: &BTreeMap<String, RunAnnotation>) -> Result<(), BenchmarkError> {
    if items
        .iter()
        .any(|(id, a)| !super::history::valid_id(id) || a.name.len() > 256 || a.notes.len() > 16384)
    {
        return Err(error(
            "Invalid annotation ID, name (256-byte limit), or notes (16 KiB limit).",
        ));
    }
    Ok(())
}
impl RunStore {
    pub fn annotations(&self) -> Result<BTreeMap<String, RunAnnotation>, BenchmarkError> {
        let path = self.root().join("annotations.json");
        if !path.exists() {
            return Ok(BTreeMap::new());
        }
        let annotations = serde_json::from_slice(&fs::read(path).map_err(error)?).map_err(error)?;
        validate_annotations(&annotations)?;
        Ok(annotations)
    }
    pub fn annotate(&self, id: &str, annotation: RunAnnotation) -> Result<(), BenchmarkError> {
        self.load(id)?;
        let mut all = self.annotations()?;
        all.insert(id.into(), annotation);
        validate_annotations(&all)?;
        fs::create_dir_all(self.root()).map_err(error)?;
        super::history::write_atomic(
            &self.root().join("annotations.json"),
            &serde_json::to_vec(&all).map_err(error)?,
            true,
        )
        .map_err(error)
    }
    /// Explicit user deletion only. Other records and original legacy files are untouched.
    pub fn delete_run(&self, id: &str) -> Result<(), BenchmarkError> {
        self.load(id)?;
        // Read sidecar first: malformed metadata must not be silently replaced.
        let mut annotations = self.annotations()?;
        fs::remove_file(self.root().join("runs").join(format!("{id}.json"))).map_err(error)?;
        annotations.remove(id);
        super::history::write_atomic(
            &self.root().join("annotations.json"),
            &serde_json::to_vec(&annotations).map_err(error)?,
            true,
        )
        .map_err(error)?;
        self.rebuild_index()?;
        Ok(())
    }
    pub fn export_runs(&self, ids: &[String]) -> Result<Vec<u8>, BenchmarkError> {
        if ids.is_empty() {
            return Err(error("Select at least one native run."));
        }
        let mut total = 0_u64;
        let mut records = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for id in ids {
            if !super::history::valid_id(id) || !seen.insert(id) {
                return Err(error("Invalid or duplicate export ID."));
            }
            total = total.saturating_add(
                fs::metadata(self.root().join("runs").join(format!("{id}.json")))
                    .map_err(error)?
                    .len(),
            );
            if total > NATIVE_EXPORT_LIMIT {
                return Err(error(
                    "Selected records exceed 256 MiB; export smaller selections. No samples were removed.",
                ));
            }
            records.push(self.load(id)?);
        }
        let annotations = self
            .annotations()?
            .into_iter()
            .filter(|(id, _)| ids.contains(id))
            .collect();
        let bundle = NativeExport {
            format: "gluj-bench-native".into(),
            version: 1,
            records,
            annotations,
        };
        let bytes = serde_json::to_vec(&bundle).map_err(error)?;
        if bytes.len() as u64 > NATIVE_EXPORT_LIMIT {
            return Err(error(
                "Native export exceeds 256 MiB; no samples were removed.",
            ));
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record() -> RunRecord {
        RunRecord::start("example", &crate::BenchmarkConfig::default(), vec![])
    }
    #[test]
    fn curation_export_restart_and_explicit_deletion_preserve_other_records() {
        let mut first = record();
        first.sample_coverage = crate::SampleCoverage::Partial;
        first.realizations.push(crate::WorkloadRealization {
            id: "realization".into(),
            seed: Some(42),
            generator_revision: Some("fixture-1".into()),
            parameters: Default::default(),
        });
        first.blocks.push(crate::SampleBlock {
            id: "block".into(),
            realization_id: "realization".into(),
            order: 0,
            preparation: "warm".into(),
        });
        first.samples.push(crate::RawSample {
            timing_scope: crate::SampleTimingScope::ActiveExecution,
            version: crate::CONTRACT_VERSION,
            sequence: 1,
            block_id: "block".into(),
            realization_id: "realization".into(),
            condition_id: "condition".into(),
            metric: "read".into(),
            value: 100.,
            unit: "bytes/s".into(),
            useful_work: 1,
            useful_work_unit: "bytes".into(),
            active_ns: 10,
            elapsed_ns: 20,
            offset_ns: 0,
            reference_anchor: None,
            validation: first.validation.clone(),
            status: crate::CompletionStatus::Completed,
        });
        let second = record();
        let path = std::env::temp_dir().join(format!("gluj-history-{}", first.run_id));
        let store = RunStore::new(&path);
        store.save(&first).unwrap();
        store.save(&second).unwrap();
        let annotation = RunAnnotation {
            name: "Before driver update".into(),
            notes: "Idle machine".into(),
            pinned: true,
            archived: true,
        };
        store.annotate(&first.run_id, annotation.clone()).unwrap();
        let reopened = RunStore::new(&path);
        assert_eq!(reopened.annotations().unwrap()[&first.run_id], annotation);
        let export = NativeExport::decode(
            &reopened
                .export_runs(std::slice::from_ref(&first.run_id))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(export.records, vec![first.clone()]);
        assert_eq!(export.annotations.len(), 1);
        assert!(reopened.delete_run("../escape").is_err());
        reopened.delete_run(&first.run_id).unwrap();
        assert_eq!(reopened.rebuild_index().unwrap().runs.len(), 1);
        assert_eq!(reopened.load(&second.run_id).unwrap(), second);
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn malformed_sidecar_is_preserved_and_export_rejects_duplicates() {
        let record = record();
        let path = std::env::temp_dir().join(format!("gluj-history-{}", record.run_id));
        let store = RunStore::new(&path);
        store.save(&record).unwrap();
        fs::write(path.join("annotations.json"), b"future format").unwrap();
        assert!(
            store
                .annotate(&record.run_id, RunAnnotation::default())
                .is_err()
        );
        assert_eq!(
            fs::read(path.join("annotations.json")).unwrap(),
            b"future format"
        );
        assert!(
            store
                .export_runs(&[record.run_id.clone(), record.run_id])
                .is_err()
        );
        fs::remove_dir_all(path).unwrap();
    }
}
