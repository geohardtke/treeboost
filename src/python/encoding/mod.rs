//! Python bindings for encoding module
//!
//! Provides access to:
//! - Target encoding: OrderedTargetEncoder, EncodingMap
//! - Category filtering: CountMinSketch, CategoryFilter, CategoryMapping

mod filter;
mod target;

use pyo3::prelude::*;

/// Register all encoding classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    target::register(m)?;
    filter::register(m)?;
    Ok(())
}
