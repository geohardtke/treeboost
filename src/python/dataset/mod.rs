//! Python bindings for dataset module
//!
//! Provides access to:
//! - Data types: FeatureType, FeatureInfo, BinnedDataset
//! - Data loading: DatasetLoader
//! - Data splitting: HoldoutSplit, KFoldSplit
//! - Data pipeline: DataPipeline, PipelineConfig

mod loader;
mod pipeline;
mod splits;
pub(crate) mod types;

// Re-export types used by other python modules
pub(crate) use types::PyBinnedDataset;

use pyo3::prelude::*;

/// Register all dataset classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    types::register(m)?;
    loader::register(m)?;
    pipeline::register(m)?;
    splits::register(m)?;
    Ok(())
}
