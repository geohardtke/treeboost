//! Python bindings for inference module
//!
//! Provides access to:
//! - Prediction: Point predictions with optional intervals
//! - ConformalPredictor: Distribution-free prediction intervals

mod predict;

use pyo3::prelude::*;

/// Register all inference classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    predict::register(m)
}
