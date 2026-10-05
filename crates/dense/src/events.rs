//! Run event log, the same contract as `scripts/turntable_mesh/dense_events.py`
//! and `docs/ENGINE-CONTRACT.md`: one JSON object per line, appended.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

pub const SCHEMA: &str = "crisp3ds_dense_events_v1";

/// Append-only writer. `EventLog::none()` discards everything.
pub struct EventLog {
    path: Option<PathBuf>,
    stage: Option<String>,
}

impl EventLog {
    pub fn new(path: Option<&Path>, stage: &str) -> Self {
        EventLog { path: path.map(Path::to_path_buf), stage: Some(stage.to_string()) }
    }

    pub fn none() -> Self {
        EventLog { path: None, stage: None }
    }

    /// One write call per line, so appends from several stage processes never interleave.
    pub fn emit(&self, event_type: &str, fields: Value) -> anyhow::Result<()> {
        let Some(path) = &self.path else { return Ok(()) };
        let time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let mut record = Map::new();
        record.insert("type".into(), json!(event_type));
        record.insert("time".into(), json!(time));
        record.insert("stage".into(), json!(self.stage));
        if let Value::Object(extra) = fields {
            record.extend(extra);
        }
        let mut line = serde_json::to_string(&Value::Object(record))?;
        line.push('\n');
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(line.as_bytes())?;
        file.sync_all()?;
        Ok(())
    }

    pub fn progress(&self, fraction: f64, message: &str) -> anyhow::Result<()> {
        let fraction = (fraction.clamp(0.0, 1.0) * 1e4).round() / 1e4;
        self.emit("progress", json!({"fraction": fraction, "message": message}))
    }

    pub fn metric(&self, name: &str, value: f64) -> anyhow::Result<()> {
        self.emit("metric", json!({"name": name, "value": value}))
    }

    /// `path` must lie under the run directory (the directory of the event log).
    pub fn artifact(&self, kind: &str, path: &Path, label: &str, extra: Value) -> anyhow::Result<()> {
        let Some(log) = &self.path else { return Ok(()) };
        let root = log.parent().unwrap_or(Path::new("."));
        let relative = path.strip_prefix(root).unwrap_or(path);
        let relative = relative.to_string_lossy().replace('\\', "/");
        let mut fields = json!({"kind": kind, "path": relative, "label": label});
        if let (Value::Object(target), Value::Object(more)) = (&mut fields, extra) {
            target.extend(more);
        }
        self.emit("artifact", fields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_json_with_stage_and_relative_paths() {
        let dir = std::env::temp_dir().join(format!("crisp3ds-events-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("mesh")).unwrap();
        let log_path = dir.join("events.jsonl");
        let _ = std::fs::remove_file(&log_path);
        let log = EventLog::new(Some(&log_path), "mesh");
        log.progress(0.33333, "working").unwrap();
        log.artifact("final_mesh", &dir.join("mesh/mesh.stl"), "Final surface", json!({"triangles": 12})).unwrap();
        let text = std::fs::read_to_string(&log_path).unwrap();
        let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["stage"], "mesh");
        assert_eq!(lines[0]["fraction"], 0.3333);
        assert_eq!(lines[1]["path"], "mesh/mesh.stl");
        assert_eq!(lines[1]["triangles"], 12);
        EventLog::none().progress(0.5, "ignored").unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
