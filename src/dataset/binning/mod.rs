//! Binning and quantization for feature discretization
//!
//! This module provides:
//!
//! - [`QuantileBinner`] - T-Digest based quantile binning for continuous features
//! - [`FeatureBundler`] - Exclusive Feature Bundling (EFB) for mutually exclusive features
//! - [`PackedColumn`] - 4-bit packed storage for memory efficiency

mod binner;
mod bundler;
pub mod packed;

// Re-export binner types
pub use binner::{DatasetBinner, QuantileBinner, DEFAULT_NUM_BINS, MAX_BINS};

// Re-export bundler types
pub use bundler::{BundledDataset, BundlerConfig, BundlingResult, FeatureBundle, FeatureBundler};

// Re-export packed storage types
pub use packed::{
    can_pack, optimal_storage, FeatureStorage, PackedColumn, PackedDataset, StorageMode,
};
