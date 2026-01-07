//! Quantile-based feature binner using T-Digest
//!
//! Discretizes continuous features into u8 bins using robust quantile estimation.
//! T-Digest provides accurate quantiles even for skewed distributions.

use crate::dataset::{FeatureInfo, FeatureType};
use crate::Result;
use tdigest::TDigest;

/// Maximum number of bins (u8 range)
pub const MAX_BINS: usize = 256;

/// Default number of bins for numeric features
pub const DEFAULT_NUM_BINS: usize = 255;

/// Quantile-based binner using T-Digest
#[derive(Debug, Clone)]
pub struct QuantileBinner {
    /// Number of bins to create
    num_bins: usize,
    /// T-Digest compression parameter (higher = more accuracy, more memory)
    compression: f64,
}

impl Default for QuantileBinner {
    fn default() -> Self {
        Self::new(DEFAULT_NUM_BINS)
    }
}

impl QuantileBinner {
    /// Create a new quantile binner
    pub fn new(num_bins: usize) -> Self {
        assert!(
            num_bins > 0 && num_bins <= MAX_BINS,
            "num_bins must be in [1, 256]"
        );
        Self {
            num_bins,
            compression: 100.0, // Good balance of accuracy and memory
        }
    }

    /// Set T-Digest compression parameter
    pub fn with_compression(mut self, compression: f64) -> Self {
        self.compression = compression;
        self
    }

    /// Compute bin boundaries from a column of values
    ///
    /// Returns `num_bins - 1` cut points that define the bin edges.
    pub fn compute_boundaries(&self, values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }

        // Build T-Digest from values (excluding NaN)
        let valid_values: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();

        if valid_values.is_empty() {
            return Vec::new();
        }

        let digest = TDigest::new_with_size(self.compression as usize);
        let digest = digest.merge_unsorted(valid_values);

        // Compute quantile cut points
        let mut boundaries = Vec::with_capacity(self.num_bins - 1);
        for i in 1..self.num_bins {
            let q = i as f64 / self.num_bins as f64;
            let value = digest.estimate_quantile(q);

            // Avoid duplicate boundaries
            if boundaries.is_empty() || value > *boundaries.last().unwrap() {
                boundaries.push(value);
            }
        }

        boundaries
    }

    /// Bin a single value using precomputed boundaries
    #[inline]
    pub fn bin_value(value: f64, boundaries: &[f64]) -> u8 {
        if !value.is_finite() {
            return 0; // NaN/Inf goes to bin 0
        }

        if boundaries.is_empty() {
            return 0;
        }

        // Binary search for the bin
        match boundaries.binary_search_by(|b| b.partial_cmp(&value).unwrap()) {
            Ok(idx) => (idx + 1).min(255) as u8,
            Err(idx) => idx.min(255) as u8,
        }
    }

    /// Bin an entire column of values
    pub fn bin_column(&self, values: &[f64], boundaries: &[f64]) -> Vec<u8> {
        values
            .iter()
            .map(|&v| Self::bin_value(v, boundaries))
            .collect()
    }

    /// Create FeatureInfo for a numeric feature
    pub fn create_feature_info(&self, name: String, boundaries: Vec<f64>) -> FeatureInfo {
        let num_bins = (boundaries.len() + 1).min(MAX_BINS) as u8;
        FeatureInfo {
            name,
            feature_type: FeatureType::Numeric,
            num_bins,
            bin_boundaries: boundaries,
        }
    }
}

/// Builder for creating a BinnedDataset from raw data
pub struct DatasetBinner {
    binner: QuantileBinner,
}

impl DatasetBinner {
    pub fn new(num_bins: usize) -> Self {
        Self {
            binner: QuantileBinner::new(num_bins),
        }
    }

    /// Process a numeric column: compute boundaries and bin values
    pub fn process_numeric_column(
        &self,
        name: String,
        values: &[f64],
    ) -> Result<(Vec<u8>, FeatureInfo)> {
        let boundaries = self.binner.compute_boundaries(values);
        let binned = self.binner.bin_column(values, &boundaries);
        let info = self.binner.create_feature_info(name, boundaries);
        Ok((binned, info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_boundaries() {
        let binner = QuantileBinner::new(4);
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let boundaries = binner.compute_boundaries(&values);

        // Should have 3 boundaries for 4 bins
        assert!(boundaries.len() <= 3);
        // Boundaries should be sorted
        for i in 1..boundaries.len() {
            assert!(boundaries[i] > boundaries[i - 1]);
        }
    }

    #[test]
    fn test_bin_value() {
        let boundaries = vec![10.0, 20.0, 30.0];

        assert_eq!(QuantileBinner::bin_value(5.0, &boundaries), 0);
        assert_eq!(QuantileBinner::bin_value(10.0, &boundaries), 1);
        assert_eq!(QuantileBinner::bin_value(15.0, &boundaries), 1);
        assert_eq!(QuantileBinner::bin_value(25.0, &boundaries), 2);
        assert_eq!(QuantileBinner::bin_value(35.0, &boundaries), 3);
    }

    #[test]
    fn test_bin_column() {
        let binner = QuantileBinner::new(4);
        let boundaries = vec![10.0, 20.0, 30.0];
        let values = vec![5.0, 15.0, 25.0, 35.0];

        let binned = binner.bin_column(&values, &boundaries);

        assert_eq!(binned, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_nan_handling() {
        let boundaries = vec![10.0, 20.0];

        assert_eq!(QuantileBinner::bin_value(f64::NAN, &boundaries), 0);
        assert_eq!(QuantileBinner::bin_value(f64::INFINITY, &boundaries), 0);
        assert_eq!(QuantileBinner::bin_value(f64::NEG_INFINITY, &boundaries), 0);
    }

    #[test]
    fn test_empty_values() {
        let binner = QuantileBinner::new(4);
        let boundaries = binner.compute_boundaries(&[]);

        assert!(boundaries.is_empty());
    }
}
