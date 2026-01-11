//! GBDT booster module
//!
//! Provides the main training interface:
//! - `GBDTConfig`: Training configuration
//! - `GBDTModel`: Trained ensemble model
//!
//! ## Module Organization
//!
//! - `gbdt`: Core model struct, accessors, and basic utilities
//! - `training`: Training implementations (high-level and low-level APIs)
//! - `prediction`: Prediction implementations (inference methods)
//! - `analysis`: Feature importance and model analysis
//! - `conformal`: Conformal prediction intervals

pub mod analysis;
mod config;
pub mod conformal;
mod gbdt;
pub mod prediction;
pub mod training;

pub use config::{GBDTConfig, GbdtPreset, LossType, OutputType};
pub use gbdt::GBDTModel;
