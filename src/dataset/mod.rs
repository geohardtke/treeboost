//! Dataset module: Data loading, binning, and columnar storage
//!
//! This module provides efficient data structures for GBDT training:
//!
//! ## Core Types
//!
//! - [`BinnedDataset`] - Columnar u8 storage for binned features
//! - [`BinEntry`] - Histogram bin entry for gradient/hessian accumulation
//! - [`FeatureInfo`] - Feature metadata including name, type, and bin boundaries
//! - [`SparseColumn`] - Sparse storage for high-sparsity features
//!
//! ## Binning & Quantization
//!
//! - [`QuantileBinner`] - T-Digest based quantile binning
//! - [`FeatureBundler`] - Exclusive Feature Bundling (EFB) for mutually exclusive features
//! - [`PackedColumn`] - 4-bit packed storage for memory efficiency
//!
//! ## I/O & Preprocessing
//!
//! - [`DatasetLoader`] - CSV/Parquet loading to BinnedDataset
//! - [`DataPipeline`] - Full dirty data pipeline (CMS filter → Target Encode → Bin)
//! - [`ColumnPermutation`] - Cache-aware column reordering
//!
//! ## Splitting
//!
//! - [`split_holdout`] - Train/validation/calibration splits
//! - [`split_kfold`] - K-fold cross-validation splits

// =============================================================================
// Submodules
// =============================================================================

/// Core binned dataset types
pub mod core;

/// Binning and quantization
pub mod binning;

/// I/O operations (loading, reordering)
pub mod io;

/// Preprocessing pipeline
pub mod pipeline;

/// Feature extraction for linear models
pub mod feature_extractor;

/// Data splitting utilities
pub mod split;

// =============================================================================
// Backward-Compatible Re-Exports
// =============================================================================

// Core types (from core/)
pub use core::{
    BinEntry, BinnedDataset, FeatureInfo, FeatureType, SparseColumn, DEFAULT_BIN,
    SPARSITY_THRESHOLD,
};

// Binning types (from binning/)
pub use binning::{
    can_pack, optimal_storage, BundledDataset, BundlerConfig, BundlingResult, DatasetBinner,
    FeatureBundle, FeatureBundler, FeatureStorage, PackedColumn, PackedDataset, QuantileBinner,
    StorageMode, DEFAULT_NUM_BINS, MAX_BINS,
};

// I/O types (from io/)
pub use io::{
    reorder_dataset, AccessTracker, ColumnPermutation, DatasetLoader, OrderingStrategy,
    ReorderBuilder,
};

// Pipeline types (from pipeline/)
pub use pipeline::{CategoricalEncodingState, DataPipeline, PipelineConfig, PipelineState};

// Split types (from split/)
pub use split::{split_holdout, split_holdout_by_era, split_kfold, HoldoutSplit, KFoldSplit};
