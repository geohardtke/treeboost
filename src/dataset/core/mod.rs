//! Core binned dataset types
//!
//! This module contains the fundamental data structures for binned datasets:
//!
//! - [`BinnedDataset`] - The main dataset structure with columnar u8 storage
//! - [`BinEntry`] - Histogram bin entry for gradient/hessian accumulation
//! - [`FeatureInfo`] - Feature metadata including name, type, and bin boundaries
//! - [`FeatureType`] - Enum distinguishing numeric from categorical features
//! - [`SparseColumn`] - CSR-like sparse storage for high-sparsity features

mod dataset;
mod feature_info;
mod sparse;

// Re-export all public types
pub use dataset::BinnedDataset;
pub use feature_info::{BinEntry, FeatureInfo, FeatureType, DEFAULT_BIN, SPARSITY_THRESHOLD};
pub use sparse::SparseColumn;
