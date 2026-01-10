//! Dataset module: Data loading, binning, and columnar storage
//!
//! This module provides efficient data structures for GBDT training:
//! - `BinnedDataset`: Columnar u8 storage for binned features
//! - `QuantileBinner`: T-Digest based quantile binning
//! - `DataPipeline`: Full dirty data pipeline (CMS filter → Target Encode → Bin)
//! - `PackedColumn`: 4-bit packed storage for memory efficiency
//! - `ColumnPermutation`: Cache-aware column reordering
//! - `FeatureBundler`: Exclusive Feature Bundling (EFB) for mutually exclusive features
//! - Data loading utilities for Polars DataFrames

mod binned;
mod binner;
mod bundler;
pub mod feature_extractor;
mod loader;
pub mod packed;
mod pipeline;
mod reorder;
pub mod split;

pub use binned::{
    BinEntry, BinnedDataset, FeatureInfo, FeatureType, SparseColumn, DEFAULT_BIN,
    SPARSITY_THRESHOLD,
};
pub use binner::{DatasetBinner, QuantileBinner, DEFAULT_NUM_BINS, MAX_BINS};
pub use bundler::{BundledDataset, BundlerConfig, BundlingResult, FeatureBundle, FeatureBundler};
pub use loader::DatasetLoader;
pub use packed::{
    can_pack, optimal_storage, FeatureStorage, PackedColumn, PackedDataset, StorageMode,
};
pub use pipeline::{CategoricalEncodingState, DataPipeline, PipelineConfig, PipelineState};
pub use reorder::{
    reorder_dataset, AccessTracker, ColumnPermutation, OrderingStrategy, ReorderBuilder,
};
pub use split::{split_holdout, split_holdout_by_era, split_kfold, HoldoutSplit, KFoldSplit};
