//! Feature metadata types for binned datasets
//!
//! This module contains the fundamental types for describing features:
//! - `BinEntry` - Histogram bin entry for gradient/hessian accumulation
//! - `FeatureType` - Enum for numeric vs categorical features
//! - `FeatureInfo` - Complete feature metadata including bins and boundaries

use bytemuck::{Pod, Zeroable};
use rkyv::{Archive, Deserialize, Serialize};

// =============================================================================
// Constants
// =============================================================================

/// Threshold for considering a feature sparse (fraction of default bin values)
/// Set to 0.9 (90% zeros) because sparse path overhead only pays off at high sparsity
pub const SPARSITY_THRESHOLD: f32 = 0.9;

/// Default bin value (typically 0, representing missing/zero values)
pub const DEFAULT_BIN: u8 = 0;

// =============================================================================
// BinEntry
// =============================================================================

/// Histogram bin entry for gradient/hessian accumulation
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, Archive, Serialize, Deserialize)]
pub struct BinEntry {
    /// Sum of gradients for samples in this bin
    pub sum_gradients: f32,
    /// Sum of hessians for samples in this bin
    pub sum_hessians: f32,
    /// Count of samples in this bin
    pub count: u32,
}

impl BinEntry {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sample's gradient and hessian to this bin
    #[inline]
    pub fn accumulate(&mut self, gradient: f32, hessian: f32) {
        self.sum_gradients += gradient;
        self.sum_hessians += hessian;
        self.count += 1;
    }

    /// Add multiple samples' gradients and hessians to this bin
    /// Used by sparse histogram building for default bin accumulation
    #[inline]
    pub fn accumulate_with_count(&mut self, gradient: f32, hessian: f32, count: u32) {
        self.sum_gradients += gradient;
        self.sum_hessians += hessian;
        self.count += count;
    }

    /// Merge another bin entry into this one
    #[inline]
    pub fn merge(&mut self, other: &BinEntry) {
        self.sum_gradients += other.sum_gradients;
        self.sum_hessians += other.sum_hessians;
        self.count += other.count;
    }

    /// Subtract another bin entry from this one (for Histogram Subtraction Trick)
    #[inline]
    pub fn subtract(&mut self, other: &BinEntry) {
        self.sum_gradients -= other.sum_gradients;
        self.sum_hessians -= other.sum_hessians;
        self.count -= other.count;
    }
}

// =============================================================================
// FeatureType
// =============================================================================

/// Feature type for determining how to handle the feature
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum FeatureType {
    /// Continuous numeric feature (binned via T-Digest quantiles)
    Numeric,
    /// Categorical feature (encoded via Ordered Target Encoding)
    Categorical,
}

// =============================================================================
// FeatureInfo
// =============================================================================

/// Feature metadata
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct FeatureInfo {
    /// Feature name
    pub name: String,
    /// Feature type
    pub feature_type: FeatureType,
    /// Number of bins used (max 256)
    pub num_bins: u8,
    /// Bin boundaries for numeric features (len = num_bins - 1)
    /// For categorical, this is empty (bins are category indices)
    pub bin_boundaries: Vec<f64>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin_entry_accumulate() {
        let mut entry = BinEntry::new();
        entry.accumulate(1.0, 2.0);
        entry.accumulate(0.5, 1.0);

        assert_eq!(entry.sum_gradients, 1.5);
        assert_eq!(entry.sum_hessians, 3.0);
        assert_eq!(entry.count, 2);
    }

    #[test]
    fn test_bin_entry_subtract() {
        let mut parent = BinEntry {
            sum_gradients: 10.0,
            sum_hessians: 20.0,
            count: 100,
        };
        let child = BinEntry {
            sum_gradients: 3.0,
            sum_hessians: 6.0,
            count: 30,
        };

        parent.subtract(&child);

        assert_eq!(parent.sum_gradients, 7.0);
        assert_eq!(parent.sum_hessians, 14.0);
        assert_eq!(parent.count, 70);
    }

    #[test]
    fn test_bin_entry_merge() {
        let mut entry1 = BinEntry {
            sum_gradients: 1.0,
            sum_hessians: 2.0,
            count: 10,
        };
        let entry2 = BinEntry {
            sum_gradients: 3.0,
            sum_hessians: 4.0,
            count: 20,
        };

        entry1.merge(&entry2);

        assert_eq!(entry1.sum_gradients, 4.0);
        assert_eq!(entry1.sum_hessians, 6.0);
        assert_eq!(entry1.count, 30);
    }

    #[test]
    fn test_bin_entry_accumulate_with_count() {
        let mut entry = BinEntry::new();
        entry.accumulate_with_count(5.0, 10.0, 5);

        assert_eq!(entry.sum_gradients, 5.0);
        assert_eq!(entry.sum_hessians, 10.0);
        assert_eq!(entry.count, 5);
    }
}
