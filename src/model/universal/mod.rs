//! UniversalModel: Unified boosting framework
//!
//! Supports multiple boosting modes:
//! - **PureTree**: Standard GBDT (delegates to GBDTModel - GPU, conformal, multi-class)
//! - **LinearThenTree**: Linear model captures trend, trees capture residuals
//! - **RandomForest**: Parallel trees with bootstrap sampling
//!
//! # Design Rationale
//!
//! Most tabular problems are solved by Linear, Tree, or their combination.
//! UniversalModel provides a single interface for all three patterns.
//!
//! ## Architecture
//!
//! - **PureTree** wraps `GBDTModel` directly - you get GPU acceleration, conformal
//!   prediction, multi-class, and all mature GBDTModel features.
//! - **LinearThenTree** and **RandomForest** are specialized modes that don't
//!   fit the standard GBDT pattern.
//!
//! ## When to Use Each Mode
//!
//! - **PureTree**: Most tabular problems, categorical-heavy data
//! - **LinearThenTree**: Time-series with trends, extrapolation beyond training range
//! - **RandomForest**: When robustness and variance reduction are priorities
//!
//! # Automatic Mode Selection
//!
//! TreeBoost can automatically analyze your dataset and pick the best mode:
//!
//! ```ignore
//! use treeboost::{UniversalModel, MseLoss};
//!
//! // Let TreeBoost analyze the data and pick the best mode
//! let model = UniversalModel::auto(&dataset, &MseLoss)?;
//!
//! // See why it picked this mode
//! println!("{}", model.analysis_report().unwrap());
//! ```
//!
//! The analysis runs lightweight "probes" (quick linear and tree models on subsamples)
//! to measure signal strength WITHOUT the cost of full training. A 5-second analysis
//! beats a 4-hour search.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::model::{UniversalModel, BoostingMode, UniversalConfig};
//!
//! let config = UniversalConfig::default()
//!     .with_mode(BoostingMode::LinearThenTree)
//!     .with_num_rounds(100);
//!
//! let model = UniversalModel::train(&dataset, config)?;
//! let predictions = model.predict(&dataset);
//! ```

// Submodules
pub mod config;
pub mod mode;

// Core implementation modules (split for maintainability)
mod core; // Struct definition, accessors, helpers
mod incremental; // Incremental updates and TRB format
mod prediction; // Prediction methods
mod serialization;
mod training; // Training methods // TunableModel impl and tests

// Re-export everything from core
pub use core::{IncrementalUpdateReport, UniversalModel};

// Re-export key types
pub use config::{LinearSelectionMode, StackingStrategy, UniversalConfig, UniversalPreset};
pub use mode::{BoostingMode, ModeSelection};

// Implementation Note:
// UniversalModel implementation is now organized into logical modules:
// - core.rs: Struct definition (~640 lines) + basic accessors + helpers
// - training.rs: All train*() methods (~500 lines)
// - prediction.rs: All predict*() and analysis methods (~700 lines)
// - incremental.rs: update() and TRB save/load (~350 lines)
// - serialization.rs: TunableModel impl + tests (~600 lines)
//
// This split maintains clean module boundaries while keeping implementation details
// within each module. Cross-module calls use pub(super) visibility.
