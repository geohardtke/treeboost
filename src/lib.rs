//! TreeBoost: High-performance Gradient Boosted Decision Tree engine
//!
//! A pure Rust GBDT implementation designed for large-scale tabular data
//! with robust handling of dirty/noisy data.
//!
//! # Key Features
//!
//! - **Histogram-based training**: u8 bins for memory efficiency
//! - **Shannon Entropy regularized splits**: Drift-resilient objective
//! - **Pseudo-Huber loss**: Robust to outliers
//! - **Ordered Target Encoding**: High-cardinality categoricals without leakage
//! - **Split Conformal Prediction**: Distribution-free prediction intervals
//! - **Zero-copy serialization**: Fast model loading via rkyv
//!
//! # Example
//!
//! ```ignore
//! use treeboost::{GBDTConfig, GBDTModel};
//! use treeboost::dataset::DatasetLoader;
//!
//! // Load data
//! let loader = DatasetLoader::new(255);
//! let dataset = loader.load_parquet("data.parquet", "target", None)?;
//!
//! // Configure and train
//! let config = GBDTConfig::new()
//!     .with_num_rounds(100)
//!     .with_max_depth(6)
//!     .with_pseudo_huber_loss(1.0)
//!     .with_entropy_weight(0.1);
//!
//! let model = GBDTModel::train(&dataset, config)?;
//!
//! // Predict
//! let predictions = model.predict(&dataset);
//! ```

pub mod booster;
pub mod dataset;
pub mod encoding;
pub mod histogram;
pub mod inference;
pub mod kernel;
pub mod loss;
pub mod serialize;
pub mod tree;

#[cfg(feature = "python")]
mod python;

// Re-exports for convenience
pub use booster::{GBDTConfig, GBDTModel};
pub use dataset::{BinnedDataset, QuantileBinner};
pub use inference::Prediction;
pub use loss::{LossFunction, MseLoss, PseudoHuberLoss};
pub use tree::{InteractionConstraints, MonotonicConstraint};

/// Library error type
#[derive(Debug, thiserror::Error)]
pub enum TreeBoostError {
    #[error("Data error: {0}")]
    Data(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, TreeBoostError>;

// Python module entry point
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)
}
