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

// Core implementation (UniversalModel struct and all impls)
// Kept together to avoid tight coupling between split files
mod core;

// Re-export everything from core
pub use core::UniversalModel;

// Re-export key types
pub use config::UniversalConfig;
pub use mode::{BoostingMode, ModeSelection};

// Implementation Note:
// UniversalModel struct and impl blocks are kept in core.rs rather than being
// split into training.rs, prediction.rs, etc. While further splitting would
// reduce file size, it creates tight coupling between files (methods calling
// each other) and increases complexity. The current organization (enums + config
// extracted to separate files) provides clear structure while keeping related
// functionality together.
