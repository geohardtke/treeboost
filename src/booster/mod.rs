//! GBDT booster module
//!
//! Provides the main training interface:
//! - `GBDTConfig`: Training configuration
//! - `GBDTModel`: Trained ensemble model

mod config;
mod gbdt;

pub use config::{GBDTConfig, GbdtPreset, LossType};
pub use gbdt::GBDTModel;
