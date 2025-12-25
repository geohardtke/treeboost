//! Dataset module: Data loading, binning, and columnar storage
//!
//! This module provides efficient data structures for GBDT training:
//! - `BinnedDataset`: Columnar u8 storage for binned features
//! - `QuantileBinner`: T-Digest based quantile binning
//! - `DataPipeline`: Full dirty data pipeline (CMS filter → Target Encode → Bin)
//! - Data loading utilities for Polars DataFrames

mod binned;
mod binner;
mod loader;
mod pipeline;

pub use binned::{BinEntry, BinnedDataset, FeatureInfo, FeatureType};
pub use binner::{DatasetBinner, QuantileBinner, DEFAULT_NUM_BINS, MAX_BINS};
pub use loader::DatasetLoader;
pub use pipeline::{CategoricalEncodingState, DataPipeline, PipelineConfig, PipelineState};
