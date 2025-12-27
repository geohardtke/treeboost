//! Histogram construction for GBDT training
//!
//! Provides efficient parallel histogram building:
//! - Feature-parallel construction via Rayon
//! - Histogram Subtraction Trick for sibling nodes
//! - Cache-friendly memory access patterns
//! - Fused gradient+histogram for eliminating cache pollution
//! - Era-stratified histograms for Directional Era Splitting (DES)

mod builder;
mod entry;
mod era;
mod fused;

pub use builder::HistogramBuilder;
pub use entry::{Histogram, NodeHistograms, NUM_BINS};
pub use era::{
    average_era_gain, has_directional_agreement, EraHistogramBuilder, EraHistograms,
    EraSplitStats,
};
pub use fused::{FusedHistogramBuilder, FusedResult};
