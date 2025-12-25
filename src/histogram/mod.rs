//! Histogram construction for GBDT training
//!
//! Provides efficient parallel histogram building:
//! - Feature-parallel construction via Rayon
//! - Histogram Subtraction Trick for sibling nodes
//! - Cache-friendly memory access patterns

mod builder;
mod entry;

pub use builder::HistogramBuilder;
pub use entry::{Histogram, NodeHistograms, NUM_BINS};
