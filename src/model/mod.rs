//! High-level model abstractions
//!
//! This module provides:
//! - `UniversalModel`: Unified boosting framework supporting multiple modes
//! - `AutoBuilder`: High-level AutoML interface for simplified training
//! - `AutoModel`: Self-contained trained model from AutoBuilder

mod auto;
mod builder;
mod config;
mod progress;
mod tuning;
mod universal;

pub use auto::AutoModel;
pub use builder::AutoBuilder;
pub use config::{AutoConfig, BuildPhaseTimes, BuildResult, TreeTunerConfig, TreeTuningResult, TuningLevel};
pub use progress::{ConsoleProgress, ProgressCallback, ProgressUpdate, QuietProgress, TrainingPhase};
pub use universal::{BoostingMode, ModeSelection, UniversalConfig, UniversalModel};
