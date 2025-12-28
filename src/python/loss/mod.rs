//! Python bindings for loss module
//!
//! Provides access to:
//! - MseLoss: Mean Squared Error (regression)
//! - PseudoHuberLoss: Robust loss (regression with outliers)
//! - BinaryLogLoss: Cross-entropy (binary classification)
//! - MultiClassLogLoss: Softmax cross-entropy (multi-class)
//! - sigmoid, softmax: Activation functions

mod functions;

use pyo3::prelude::*;

/// Register all loss classes and functions with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    functions::register(m)
}
