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

mod config;
mod gbdt;
pub mod training;
pub mod prediction;
pub mod analysis;
pub mod conformal;

pub use config::{GBDTConfig, GbdtPreset, LossType};
pub use gbdt::GBDTModel;
