//! GBDT booster module
//!
//! Provides the main training interface:
//! - `GBDTConfig`: Training configuration
//! - `GBDTModel`: Trained ensemble model

mod config;
mod gbdt;

pub use config::{GBDTConfig, LossType, DEFAULT_MIN_EARLY_STOPPING_TREES};
pub use gbdt::GBDTModel;
