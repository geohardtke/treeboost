//! Binned dataset with columnar u8 storage
//!
//! Provides memory-efficient storage for histogram-based GBDT training.
//! Features are discretized to u8 bins (256 values) for 8x memory reduction.

use bytemuck::{Pod, Zeroable};
use rkyv::{Archive, Deserialize, Serialize};

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

/// Feature type for determining how to handle the feature
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub enum FeatureType {
    /// Continuous numeric feature (binned via T-Digest quantiles)
    Numeric,
    /// Categorical feature (encoded via Ordered Target Encoding)
    Categorical,
}

/// Feature metadata
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
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

/// Columnar binned dataset for efficient histogram construction
///
/// Memory layout:
/// - Features stored column-major as contiguous u8 arrays
/// - Each feature column is `num_rows` bytes
/// - Total feature memory: `num_rows * num_features` bytes
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct BinnedDataset {
    /// Number of rows (samples)
    num_rows: usize,
    /// Feature data in column-major order: features[feature_idx * num_rows + row_idx]
    features: Vec<u8>,
    /// Target values (original scale, not binned)
    targets: Vec<f32>,
    /// Feature metadata
    feature_info: Vec<FeatureInfo>,
}

impl BinnedDataset {
    /// Create a new binned dataset
    pub fn new(
        num_rows: usize,
        features: Vec<u8>,
        targets: Vec<f32>,
        feature_info: Vec<FeatureInfo>,
    ) -> Self {
        debug_assert_eq!(features.len(), num_rows * feature_info.len());
        debug_assert_eq!(targets.len(), num_rows);

        Self {
            num_rows,
            features,
            targets,
            feature_info,
        }
    }

    /// Number of samples
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of features
    #[inline]
    pub fn num_features(&self) -> usize {
        self.feature_info.len()
    }

    /// Get feature info for a feature
    #[inline]
    pub fn feature_info(&self, feature_idx: usize) -> &FeatureInfo {
        &self.feature_info[feature_idx]
    }

    /// Get all feature info
    #[inline]
    pub fn all_feature_info(&self) -> &[FeatureInfo] {
        &self.feature_info
    }

    /// Get bin value for a specific row and feature
    #[inline]
    pub fn get_bin(&self, row_idx: usize, feature_idx: usize) -> u8 {
        self.features[feature_idx * self.num_rows + row_idx]
    }

    /// Get the entire column for a feature (contiguous slice)
    #[inline]
    pub fn feature_column(&self, feature_idx: usize) -> &[u8] {
        let start = feature_idx * self.num_rows;
        &self.features[start..start + self.num_rows]
    }

    /// Get target value for a row
    #[inline]
    pub fn target(&self, row_idx: usize) -> f32 {
        self.targets[row_idx]
    }

    /// Get all targets
    #[inline]
    pub fn targets(&self) -> &[f32] {
        &self.targets
    }

    /// Get mutable targets (for testing with outliers, etc.)
    #[inline]
    pub fn targets_mut(&mut self) -> &mut [f32] {
        &mut self.targets
    }

    /// Get raw bin value for original feature value using binary search
    pub fn bin_value(&self, feature_idx: usize, value: f64) -> u8 {
        let info = &self.feature_info[feature_idx];
        if info.bin_boundaries.is_empty() {
            return 0;
        }

        // Binary search for the appropriate bin
        match info.bin_boundaries.binary_search_by(|b| {
            b.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(idx) => (idx + 1).min(info.num_bins as usize - 1) as u8,
            Err(idx) => idx.min(info.num_bins as usize - 1) as u8,
        }
    }
}

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
    fn test_binned_dataset_access() {
        let num_rows = 4;

        // Column-major: feature 0 = [0,1,2,3], feature 1 = [10,11,12,13]
        let features = vec![0u8, 1, 2, 3, 10, 11, 12, 13];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![
            FeatureInfo {
                name: "f0".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 4,
                bin_boundaries: vec![0.5, 1.5, 2.5],
            },
            FeatureInfo {
                name: "f1".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 4,
                bin_boundaries: vec![10.5, 11.5, 12.5],
            },
        ];

        let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);

        assert_eq!(dataset.num_rows(), 4);
        assert_eq!(dataset.num_features(), 2);

        // Test individual access
        assert_eq!(dataset.get_bin(0, 0), 0);
        assert_eq!(dataset.get_bin(2, 0), 2);
        assert_eq!(dataset.get_bin(1, 1), 11);

        // Test column access
        assert_eq!(dataset.feature_column(0), &[0, 1, 2, 3]);
        assert_eq!(dataset.feature_column(1), &[10, 11, 12, 13]);

        // Test targets
        assert_eq!(dataset.target(0), 1.0);
        assert_eq!(dataset.targets(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
