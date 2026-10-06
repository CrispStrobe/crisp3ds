//! Stopping and logging of a stage that runs in-process: a cancel flag another
//! thread may set, an optional `cancel` file a client may create (the run
//! directory contract), an optional deadline, and the stage's log file.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Why a stage stopped before it finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stopped {
    Cancelled,
    Deadline,
}

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Stopped::Cancelled => "cancelled on request",
            Stopped::Deadline => "deadline",
        })
    }
}

impl std::error::Error for Stopped {}

#[derive(Default)]
pub struct Control {
    cancel: Option<Arc<AtomicBool>>,
    cancel_file: Option<PathBuf>,
    deadline: Option<Instant>,
    log: Option<Mutex<File>>,
}

impl Control {
    /// Never stops; logs to standard output.
    pub fn none() -> Self {
        Control::default()
    }

    pub fn new(
        cancel: Option<Arc<AtomicBool>>,
        cancel_file: Option<PathBuf>,
        deadline: Option<Instant>,
        log: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let log = match log {
            Some(path) => Some(Mutex::new(File::create(path)?)),
            None => None,
        };
        Ok(Control { cancel, cancel_file, deadline, log })
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|flag| flag.load(Ordering::Relaxed)) || self.cancel_file.as_ref().is_some_and(|file| file.exists())
    }

    /// `Err(Stopped)` once the run was cancelled or the stage's deadline passed. Stages call this between units of work.
    pub fn check(&self) -> anyhow::Result<()> {
        if self.cancelled() {
            return Err(Stopped::Cancelled.into());
        }
        if self.deadline.is_some_and(|deadline| Instant::now() > deadline) {
            return Err(Stopped::Deadline.into());
        }
        Ok(())
    }

    /// One line into the stage log, or to standard output without one.
    pub fn log(&self, text: impl AsRef<str>) {
        match &self.log {
            Some(file) => {
                let mut file = file.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                let _ = writeln!(file, "{}", text.as_ref());
            }
            None => {
                println!("{}", text.as_ref());
                let _ = std::io::stdout().flush();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_on_flag_file_and_deadline() {
        assert!(Control::none().check().is_ok());
        let flag = Arc::new(AtomicBool::new(false));
        let file = std::env::temp_dir().join(format!("crisp3ds-cancel-{}", std::process::id()));
        let _ = std::fs::remove_file(&file);
        let control = Control::new(Some(flag.clone()), Some(file.clone()), None, None).unwrap();
        assert!(control.check().is_ok());
        std::fs::write(&file, "").unwrap();
        assert_eq!(control.check().unwrap_err().downcast::<Stopped>().unwrap(), Stopped::Cancelled);
        std::fs::remove_file(&file).unwrap();
        assert!(control.check().is_ok());
        flag.store(true, Ordering::Relaxed);
        assert!(control.cancelled());
        let late = Control::new(None, None, Some(Instant::now() - std::time::Duration::from_secs(1)), None).unwrap();
        assert_eq!(late.check().unwrap_err().downcast::<Stopped>().unwrap(), Stopped::Deadline);
    }
}
