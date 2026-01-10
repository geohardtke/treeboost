//! I/O operations for dataset loading and optimization
//!
//! This module provides:
//!
//! - [`DatasetLoader`] - CSV/Parquet loading to BinnedDataset
//! - [`ColumnPermutation`] - Cache-aware column reordering
//! - [`AccessTracker`] - Feature access pattern tracking

mod loader;
mod reorder;

// Re-export loader types
pub use loader::DatasetLoader;

// Re-export reorder types
pub use reorder::{reorder_dataset, AccessTracker, ColumnPermutation, OrderingStrategy, ReorderBuilder};
