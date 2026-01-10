//! Default constants for feature engineering
//!
//! This module groups default values for feature-related operations:
//! - [`generation`] - Feature generation defaults (polynomials, interactions, ratios)
//! - [`linear_feature`] - Linear feature-specific defaults
//! - [`selection`] - Feature selection defaults
//! - [`smart`] - Smart feature engineering presets

pub mod generation;
pub mod linear_feature;
pub mod selection;
pub mod smart;

// Re-export all constants for convenience
pub use generation::*;
pub use linear_feature::*;
pub use selection::*;
pub use smart::*;
