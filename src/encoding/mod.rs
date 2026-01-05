//! Production-grade categorical encoding for high-cardinality features
//!
//! This module provides **specialized encoders** designed for production GBDT systems
//! with robustness against rare categories and data drift.
//!
//! # Available Encoders
//!
//! - [`CountMinSketch`]: Probabilistic frequency filter for rare categories
//! - [`OrderedTargetEncoder`]: Streaming target encoding with M-Estimate smoothing
//! - [`CategoryFilter`]: Filters rare categories to "Unknown" bucket
//! - [`CategoryMapping`]: Maps categories to indices with rare handling
//!
//! # When to Use This Module vs `preprocessing::encoding`
//!
//! | Scenario | Use This Module | Use `preprocessing::encoding` |
//! |----------|-----------------|-------------------------------|
//! | High-cardinality (100+ categories) | ✅ | ❌ |
//! | Production inference with unseen categories | ✅ | ⚠️ |
//! | Rare category filtering | ✅ | ❌ |
//! | Target-based encoding with smoothing | ✅ | ❌ |
//! | Simple label/frequency encoding | ❌ | ✅ |
//! | One-hot encoding for linear models | ❌ | ✅ |
//! | Quick prototyping | ❌ | ✅ |
//!
//! # Design Philosophy
//!
//! These encoders are built for **production robustness**:
//! - **Rare category handling**: Categories below threshold → "Unknown"
//! - **Memory efficiency**: Count-Min Sketch uses O(1) space regardless of cardinality
//! - **Drift resistance**: Smoothed target encoding prevents overfitting to rare categories
//! - **Streaming**: Ordered target encoding processes data sequentially (no leakage)
//!
//! # Example: Production Pipeline
//!
//! ```ignore
//! use treeboost::encoding::{CountMinSketch, CategoryFilter, OrderedTargetEncoder};
//!
//! // Step 1: Filter rare categories (< 10 occurrences → "Unknown")
//! let mut cms = CountMinSketch::new(1000, 5);  // 1000 counters, 5 hash functions
//! for cat in &categories {
//!     cms.increment(cat);
//! }
//! let filter = CategoryFilter::new(&cms, 10);  // min_count = 10
//! let filtered: Vec<_> = categories.iter().map(|c| filter.filter(c)).collect();
//!
//! // Step 2: Target encode with smoothing
//! let mut encoder = OrderedTargetEncoder::new(30.0);  // m = 30 (smoothing)
//! let encoded = encoder.fit_transform(&filtered, &targets);
//! ```
//!
//! # See Also
//!
//! - [`crate::preprocessing::encoding`]: Simple encoders for general preprocessing

mod cms;
mod target;

pub use cms::{CategoryFilter, CategoryMapping, CountMinSketch};
pub use target::{EncodingMap, OrderedTargetEncoder};
