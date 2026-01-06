//! High-level model abstractions
//!
//! This module provides:
//! - `UniversalModel`: Unified boosting framework supporting multiple modes
//! - `AutoBuilder`: High-level AutoML interface for simplified training
//! - `AutoModel`: Self-contained trained model from AutoBuilder

mod auto;
mod builder;
mod universal;

pub use auto::AutoModel;
pub use builder::{AutoBuilder, AutoConfig, BuildPhaseTimes, BuildResult, TuningLevel};
pub use universal::{BoostingMode, ModeSelection, UniversalConfig, UniversalModel};
