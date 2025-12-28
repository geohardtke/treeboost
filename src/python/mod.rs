//! Python bindings for TreeBoost
//!
//! This module provides comprehensive Python bindings for the TreeBoost library,
//! organized into submodules for better code organization:
//!
//! - `bindings`: Core GBDT types (GBDTConfig, GBDTModel)
//! - `dataset`: Data types and loading (BinnedDataset, splits)
//! - `encoding`: Feature encoding (TargetEncoder, CategoryFilter)
//! - `inference`: Prediction types (Prediction, ConformalPredictor)
//! - `loss`: Loss functions (MseLoss, PseudoHuberLoss, etc.)
//! - `tuner`: Hyperparameter tuning (AutoTuner, TunerConfig)

pub mod bindings;
pub mod dataset;
pub mod encoding;
pub mod inference;
pub mod loss;
pub mod tuner;

use pyo3::prelude::*;

/// Register all Python bindings with the module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core GBDT types
    bindings::register_module(m)?;

    // Dataset types
    dataset::register(m)?;

    // Encoding types
    encoding::register(m)?;

    // Inference types
    inference::register(m)?;

    // Loss functions
    loss::register(m)?;

    // Tuner types
    tuner::register(m)?;

    Ok(())
}
