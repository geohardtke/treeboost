//! Default constants for learner configurations
//!
//! This module groups default values for various learner types:
//! - [`gbdt`] - GBDT model configuration defaults
//! - [`linear`] - Linear model configuration defaults
//! - [`linear_tree`] - Linear tree model configuration defaults
//! - [`tree`] - Tree model configuration defaults
//! - [`universal`] - Universal model configuration defaults

pub mod gbdt;
pub mod linear;
pub mod linear_tree;
pub mod tree;
pub mod universal;

// Re-export all constants for convenience
pub use gbdt::*;
pub use linear::*;
pub use linear_tree::*;
pub use tree::*;
pub use universal::*;
