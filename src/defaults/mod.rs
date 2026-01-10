//! Default constants for TreeBoost configuration
//!
//! This module provides default values for all configurable parameters:
//!
//! - [`learners`] - Model configuration defaults (GBDT, linear, tree, and variants)
//! - [`features`] - Feature engineering defaults (generation, selection, smart presets)
//! - [`tuning`] - Hyperparameter tuning defaults (tuner, LinearThenTree, seeds)
//!
//! Top-level modules for infrastructure defaults:
//! - [`analysis`], [`auto`], [`backend`], [`bundler`]
//! - [`ensemble`], [`preprocessing`], [`split`]

// Grouped submodules
pub mod features;
pub mod learners;
pub mod tuning;

// Infrastructure modules (kept at root level)
pub mod analysis;
pub mod auto;
pub mod backend;
pub mod bundler;
pub mod ensemble;
pub mod preprocessing;
pub mod split;

// Re-export everything for backward compatibility
pub use analysis::*;
pub use auto::*;
pub use backend::*;
pub use bundler::*;
pub use ensemble::*;
pub use features::*;
pub use learners::*;
pub use preprocessing::*;
pub use split::*;
pub use tuning::*;
