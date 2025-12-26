//! Histogram construction for GBDT training
//!
//! Provides efficient parallel histogram building:
//! - Feature-parallel construction via Rayon
//! - Histogram Subtraction Trick for sibling nodes
//! - Cache-friendly memory access patterns
//! - Fused gradient+histogram for eliminating cache pollution

mod builder;
mod entry;
mod fused;

pub use builder::HistogramBuilder;
pub use entry::{Histogram, NodeHistograms, NUM_BINS};
pub use fused::{FusedHistogramBuilder, FusedResult};
