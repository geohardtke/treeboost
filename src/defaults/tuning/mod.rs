//! Default constants for hyperparameter tuning
//!
//! This module groups default values for tuning-related operations:
//! - [`ltt`] - LinearThenTree tuning defaults
//! - [`seeds`] - Random seed defaults
//! - [`tuner`] - General tuner configuration defaults

pub mod ltt;
pub mod seeds;
pub mod tuner;

// Re-export all constants for convenience
pub use ltt::*;
pub use seeds::*;
pub use tuner::*;
