//! Categorical encoding module
//!
//! Provides robust encoding for high-cardinality categorical features:
//! - `CountMinSketch`: Probabilistic frequency filter for rare categories
//! - `OrderedTargetEncoder`: Streaming target encoding with M-Estimate smoothing

mod cms;
mod target;

pub use cms::{CategoryFilter, CategoryMapping, CountMinSketch};
pub use target::{EncodingMap, OrderedTargetEncoder};
