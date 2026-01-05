//! Binned dataset with columnar u8 storage
//!
//! Provides memory-efficient storage for histogram-based GBDT training.
//! Features are discretized to u8 bins (256 values) for 8x memory reduction.
//!
//! # Sparsity Awareness
//!
//! For sparse features (many zeros), we store only non-zero entries and compute
//! the zero bin by subtraction: `zero_bin = total - sum(non_zero_bins)`.
//! This provides up to 20x speedup on 95% sparse data.
//!
//! # Data Layouts
//!
//! - **Column-major** (default): `bins[feature][row]` - optimal for scalar CPU
//! - **Row-major** (lazy): `bins[row][feature]` - optimal for GPU/tensor-tile
//!
//! Row-major layout is computed lazily on first GPU use and cached for reuse.

use bytemuck::{Pod, Zeroable};
use rkyv::{Archive, Deserialize, Serialize};
use std::sync::OnceLock;

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

/// Feature type for determining how to handle the feature
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum FeatureType {
    /// Continuous numeric feature (binned via T-Digest quantiles)
    Numeric,
    /// Categorical feature (encoded via Ordered Target Encoding)
    Categorical,
}

/// Feature metadata
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(serde::Serialize, serde::Deserialize)]
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

/// Threshold for considering a feature sparse (fraction of default bin values)
/// Set to 0.9 (90% zeros) because sparse path overhead only pays off at high sparsity
pub const SPARSITY_THRESHOLD: f32 = 0.9;

/// Default bin value (typically 0, representing missing/zero values)
pub const DEFAULT_BIN: u8 = 0;

/// Sparse column storage (CSR-like format)
///
/// Only stores non-default entries for memory efficiency.
/// For a feature with 95% zeros, this uses only 5% of dense storage.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct SparseColumn {
    /// Row indices of non-default entries
    pub indices: Vec<u32>,
    /// Bin values at those indices (all non-default)
    pub values: Vec<u8>,
    /// Total number of rows (for bounds checking)
    pub num_rows: usize,
}

impl SparseColumn {
    /// Create from dense column, extracting only non-default entries
    pub fn from_dense(dense: &[u8], default_bin: u8) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &bin) in dense.iter().enumerate() {
            if bin != default_bin {
                indices.push(i as u32);
                values.push(bin);
            }
        }

        Self {
            indices,
            values,
            num_rows: dense.len(),
        }
    }

    /// Number of non-default entries
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Sparsity ratio (fraction of default values)
    #[inline]
    pub fn sparsity(&self) -> f32 {
        if self.num_rows == 0 {
            return 1.0;
        }
        1.0 - (self.nnz() as f32 / self.num_rows as f32)
    }

    /// Check if this column is sparse enough to benefit from sparse processing
    #[inline]
    pub fn is_sparse(&self) -> bool {
        self.sparsity() >= SPARSITY_THRESHOLD
    }

    /// Iterate over (row_index, bin_value) pairs for non-default entries
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (usize, u8)> + '_ {
        self.indices
            .iter()
            .zip(self.values.iter())
            .map(|(&idx, &val)| (idx as usize, val))
    }
}

/// Columnar binned dataset for efficient histogram construction
///
/// Memory layout:
/// - Features stored column-major as contiguous u8 arrays
/// - Each feature column is `num_rows` bytes
/// - Total feature memory: `num_rows * num_features` bytes
///
/// Sparse features are additionally stored in CSR-like format for efficient
/// histogram building on sparse data.
///
/// For GPU backends, a row-major layout is computed lazily and cached.
#[derive(Archive, Serialize, Deserialize)]
pub struct BinnedDataset {
    /// Number of rows (samples)
    num_rows: usize,
    /// Feature data in column-major order: features[feature_idx * num_rows + row_idx]
    features: Vec<u8>,
    /// Target values (original scale, not binned)
    targets: Vec<f32>,
    /// Feature metadata
    feature_info: Vec<FeatureInfo>,
    /// Sparse representations for sparse features (None if dense)
    sparse_columns: Vec<Option<SparseColumn>>,
    /// Era indices for each sample (optional, for era-based splitting)
    /// Used by Directional Era Splitting (DES) to learn invariant patterns
    /// u16 supports up to 65536 eras
    era_indices: Option<Vec<u16>>,
    /// Number of unique eras (cached for efficiency)
    num_eras: usize,
    /// Cached row-major layout for GPU backends (lazily computed)
    /// Not serialized - recomputed on first GPU use after deserialization
    #[rkyv(with = rkyv::with::Skip)]
    row_major_cache: OnceLock<Vec<u8>>,
    /// Cached 4-bit packed row-major layout for GPU backends (lazily computed)
    /// Only used when all features have ≤16 bins
    #[rkyv(with = rkyv::with::Skip)]
    row_major_4bit_cache: OnceLock<Vec<u8>>,
}

impl std::fmt::Debug for BinnedDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinnedDataset")
            .field("num_rows", &self.num_rows)
            .field("num_features", &self.num_features())
            .field("features_len", &self.features.len())
            .field("sparse_features", &self.num_sparse_features())
            .field("max_bins", &self.max_bins())
            .field("supports_4bit", &self.supports_4bit())
            .field("has_eras", &self.era_indices.is_some())
            .field("num_eras", &self.num_eras)
            .field("row_major_cached", &self.row_major_cache.get().is_some())
            .field("row_major_4bit_cached", &self.row_major_4bit_cache.get().is_some())
            .finish()
    }
}

impl Clone for BinnedDataset {
    fn clone(&self) -> Self {
        Self {
            num_rows: self.num_rows,
            features: self.features.clone(),
            targets: self.targets.clone(),
            feature_info: self.feature_info.clone(),
            sparse_columns: self.sparse_columns.clone(),
            era_indices: self.era_indices.clone(),
            num_eras: self.num_eras,
            // Don't clone caches - they will be recomputed if needed
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
        }
    }
}

impl BinnedDataset {
    /// Create a new binned dataset
    ///
    /// Automatically detects sparse features and creates sparse representations.
    pub fn new(
        num_rows: usize,
        features: Vec<u8>,
        targets: Vec<f32>,
        feature_info: Vec<FeatureInfo>,
    ) -> Self {
        debug_assert_eq!(features.len(), num_rows * feature_info.len());
        debug_assert_eq!(targets.len(), num_rows);

        let num_features = feature_info.len();

        // Detect sparse features and create sparse representations
        let sparse_columns: Vec<Option<SparseColumn>> = (0..num_features)
            .map(|f| {
                let start = f * num_rows;
                let column = &features[start..start + num_rows];
                let sparse = SparseColumn::from_dense(column, DEFAULT_BIN);

                if sparse.is_sparse() {
                    Some(sparse)
                } else {
                    None
                }
            })
            .collect();

        Self {
            num_rows,
            features,
            targets,
            feature_info,
            sparse_columns,
            era_indices: None,
            num_eras: 0,
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
        }
    }

    /// Create a new binned dataset with era indices for era-based splitting
    ///
    /// Era indices must be in range [0, num_eras) where num_eras is automatically
    /// computed from the maximum era index + 1.
    ///
    /// # Arguments
    /// * `num_rows` - Number of samples
    /// * `features` - Column-major feature data
    /// * `targets` - Target values
    /// * `feature_info` - Feature metadata
    /// * `era_indices` - Era index for each sample (u16, supports up to 65536 eras)
    pub fn new_with_eras(
        num_rows: usize,
        features: Vec<u8>,
        targets: Vec<f32>,
        feature_info: Vec<FeatureInfo>,
        era_indices: Vec<u16>,
    ) -> Self {
        debug_assert_eq!(features.len(), num_rows * feature_info.len());
        debug_assert_eq!(targets.len(), num_rows);
        debug_assert_eq!(era_indices.len(), num_rows);

        let num_features = feature_info.len();

        // Detect sparse features
        let sparse_columns: Vec<Option<SparseColumn>> = (0..num_features)
            .map(|f| {
                let start = f * num_rows;
                let column = &features[start..start + num_rows];
                let sparse = SparseColumn::from_dense(column, DEFAULT_BIN);

                if sparse.is_sparse() {
                    Some(sparse)
                } else {
                    None
                }
            })
            .collect();

        // Compute number of eras from max era index
        let num_eras = era_indices.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);

        Self {
            num_rows,
            features,
            targets,
            feature_info,
            sparse_columns,
            era_indices: Some(era_indices),
            num_eras,
            row_major_cache: OnceLock::new(),
            row_major_4bit_cache: OnceLock::new(),
        }
    }

    /// Set era indices on an existing dataset
    ///
    /// # Arguments
    /// * `era_indices` - Era index for each sample (must have length == num_rows)
    pub fn set_era_indices(&mut self, era_indices: Vec<u16>) {
        debug_assert_eq!(era_indices.len(), self.num_rows);
        self.num_eras = era_indices.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);
        self.era_indices = Some(era_indices);
    }

    /// Check if era indices are available
    #[inline]
    pub fn has_eras(&self) -> bool {
        self.era_indices.is_some()
    }

    /// Get the number of eras (0 if no era indices)
    #[inline]
    pub fn num_eras(&self) -> usize {
        self.num_eras
    }

    /// Get era index for a row (panics if no era indices set)
    #[inline]
    pub fn era(&self, row_idx: usize) -> u16 {
        self.era_indices.as_ref().expect("No era indices set")[row_idx]
    }

    /// Get era indices slice (None if not set)
    #[inline]
    pub fn era_indices(&self) -> Option<&[u16]> {
        self.era_indices.as_deref()
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

    /// Check if a feature has a sparse representation
    #[inline]
    pub fn is_sparse(&self, feature_idx: usize) -> bool {
        self.sparse_columns
            .get(feature_idx)
            .map(|s| s.is_some())
            .unwrap_or(false)
    }

    /// Get sparse column for a feature (if available)
    #[inline]
    pub fn sparse_column(&self, feature_idx: usize) -> Option<&SparseColumn> {
        self.sparse_columns.get(feature_idx).and_then(|s| s.as_ref())
    }

    /// Get number of sparse features
    pub fn num_sparse_features(&self) -> usize {
        self.sparse_columns.iter().filter(|s| s.is_some()).count()
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

    /// Get the actual split value for a given feature and bin threshold
    ///
    /// For raw prediction without binning, we need the actual threshold value.
    /// Samples with value <= split_value go left.
    ///
    /// # Arguments
    /// * `feature_idx` - Feature index
    /// * `bin_threshold` - Bin threshold from tree split
    ///
    /// # Returns
    /// The actual split value (f64) for raw data comparison
    #[inline]
    pub fn get_split_value(&self, feature_idx: usize, bin_threshold: u8) -> f64 {
        let info = &self.feature_info[feature_idx];

        // Edge cases
        if info.bin_boundaries.is_empty() {
            return 0.0;
        }

        // bin_threshold is the largest bin that goes left
        // bin_boundaries[i] is the upper bound of bin i
        // So split_value = bin_boundaries[bin_threshold] (if exists)
        // For bin_threshold = 0, samples in bin 0 go left, threshold is boundaries[0]
        let idx = bin_threshold as usize;
        if idx < info.bin_boundaries.len() {
            info.bin_boundaries[idx]
        } else {
            // bin_threshold >= num_bins - 1, use the last boundary
            // or max value (everything goes left)
            info.bin_boundaries.last().copied().unwrap_or(f64::MAX)
        }
    }

    /// Get row-major layout for GPU backends (lazy conversion).
    ///
    /// Converts column-major `bins[feature][row]` to row-major `bins[row][feature]`.
    /// Result is cached for subsequent calls.
    ///
    /// # Returns
    /// A slice of row-major bin data: `bins[row * num_features + feature]`
    pub fn as_row_major(&self) -> &[u8] {
        self.row_major_cache.get_or_init(|| {
            let num_rows = self.num_rows;
            let num_features = self.num_features();
            let mut row_major = vec![0u8; num_rows * num_features];

            // Transpose: column-major → row-major
            // Column-major: features[feature * num_rows + row]
            // Row-major: row_major[row * num_features + feature]
            for row in 0..num_rows {
                for feature in 0..num_features {
                    row_major[row * num_features + feature] =
                        self.features[feature * num_rows + row];
                }
            }

            row_major
        })
    }

    /// Get maximum number of bins across all features.
    #[inline]
    pub fn max_bins(&self) -> u8 {
        self.feature_info
            .iter()
            .map(|f| f.num_bins)
            .max()
            .unwrap_or(0)
    }

    /// Check if dataset supports 4-bit bin packing.
    ///
    /// Returns true if all features have ≤16 bins, enabling 4-bit packing
    /// for 50% memory bandwidth reduction on GPU.
    #[inline]
    pub fn supports_4bit(&self) -> bool {
        self.max_bins() <= 16
    }

    /// Get 4-bit packed row-major layout for GPU backends (lazy conversion).
    ///
    /// Packs two 4-bit bin values into each byte (nibble packing).
    /// Only valid when all features have ≤16 bins.
    ///
    /// Layout: For each row, features are packed in pairs:
    /// - byte[i] = (feature[2i+1] << 4) | feature[2i]
    /// - If odd number of features, last nibble is padded with 0
    ///
    /// # Panics
    /// Panics if any feature has more than 16 bins.
    ///
    /// # Returns
    /// A slice of 4-bit packed row-major bin data
    pub fn as_row_major_4bit(&self) -> &[u8] {
        self.row_major_4bit_cache.get_or_init(|| {
            assert!(
                self.supports_4bit(),
                "4-bit packing requires all features to have ≤16 bins, max is {}",
                self.max_bins()
            );

            let num_rows = self.num_rows;
            let num_features = self.num_features();
            // Each pair of features packs into 1 byte
            let bytes_per_row = num_features.div_ceil(2);
            let mut packed = vec![0u8; num_rows * bytes_per_row];

            for row in 0..num_rows {
                let row_offset = row * bytes_per_row;
                for pair in 0..bytes_per_row {
                    let f0 = pair * 2;
                    let f1 = f0 + 1;

                    // Get bin values (4-bit, clamped to 0-15)
                    let bin0 = self.features[f0 * num_rows + row] & 0x0F;
                    let bin1 = if f1 < num_features {
                        self.features[f1 * num_rows + row] & 0x0F
                    } else {
                        0 // Padding for odd number of features
                    };

                    // Pack: low nibble = even feature, high nibble = odd feature
                    packed[row_offset + pair] = bin0 | (bin1 << 4);
                }
            }

            packed
        })
    }

    /// Get the number of bytes per row in 4-bit packed format.
    #[inline]
    pub fn bytes_per_row_4bit(&self) -> usize {
        self.num_features().div_ceil(2)
    }
}

// Implement BinStorage trait for use with backend abstraction
impl crate::backend::BinStorage for BinnedDataset {
    fn get_bin(&self, row: usize, feature: usize) -> u8 {
        self.features[feature * self.num_rows + row]
    }

    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_features(&self) -> usize {
        self.feature_info.len()
    }

    fn feature_column(&self, feature: usize) -> Option<&[u8]> {
        let start = feature * self.num_rows;
        Some(&self.features[start..start + self.num_rows])
    }

    fn sparse_column(&self, feature: usize) -> Option<&SparseColumn> {
        self.sparse_columns.get(feature).and_then(|s| s.as_ref())
    }

    fn as_row_major(&self) -> Option<&[u8]> {
        // Delegate to the lazy-cached method
        Some(BinnedDataset::as_row_major(self))
    }

    fn max_bins(&self) -> u8 {
        BinnedDataset::max_bins(self)
    }

    fn supports_4bit(&self) -> bool {
        BinnedDataset::supports_4bit(self)
    }

    fn as_row_major_4bit(&self) -> Option<&[u8]> {
        if self.supports_4bit() {
            Some(BinnedDataset::as_row_major_4bit(self))
        } else {
            None
        }
    }

    fn bytes_per_row_4bit(&self) -> usize {
        BinnedDataset::bytes_per_row_4bit(self)
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

    #[test]
    fn test_row_major_conversion() {
        let num_rows = 4;
        let num_features = 2;

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

        // Get row-major layout
        let row_major = dataset.as_row_major();

        // Verify row-major format: [row0_feat0, row0_feat1, row1_feat0, row1_feat1, ...]
        // Row 0: (0, 10), Row 1: (1, 11), Row 2: (2, 12), Row 3: (3, 13)
        assert_eq!(row_major.len(), num_rows * num_features);
        assert_eq!(row_major[0 * num_features + 0], 0); // row 0, feature 0
        assert_eq!(row_major[0 * num_features + 1], 10); // row 0, feature 1
        assert_eq!(row_major[1 * num_features + 0], 1); // row 1, feature 0
        assert_eq!(row_major[1 * num_features + 1], 11); // row 1, feature 1
        assert_eq!(row_major[3 * num_features + 0], 3); // row 3, feature 0
        assert_eq!(row_major[3 * num_features + 1], 13); // row 3, feature 1

        // Verify caching (second call returns same data)
        let row_major2 = dataset.as_row_major();
        assert_eq!(row_major.as_ptr(), row_major2.as_ptr());
    }

    #[test]
    fn test_row_major_via_bin_storage_trait() {
        use crate::backend::BinStorage;

        let features = vec![0u8, 1, 2, 3, 10, 11, 12, 13];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![
            FeatureInfo {
                name: "f0".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 4,
                bin_boundaries: vec![],
            },
            FeatureInfo {
                name: "f1".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 4,
                bin_boundaries: vec![],
            },
        ];

        let dataset = BinnedDataset::new(4, features, targets, feature_info);

        // Access via trait
        let storage: &dyn BinStorage = &dataset;
        let row_major = storage.as_row_major();
        assert!(row_major.is_some());

        let data = row_major.unwrap();
        assert_eq!(data.len(), 8);
        // Row 0: (0, 10)
        assert_eq!(data[0], 0);
        assert_eq!(data[1], 10);
    }

    #[test]
    fn test_4bit_packing() {
        let num_rows = 4;
        let _num_features = 3; // Odd number to test padding

        // Column-major: feature 0 = [1,2,3,4], feature 1 = [5,6,7,8], feature 2 = [9,10,11,12]
        // All values <= 15, so 4-bit packing is supported
        let features = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![
            FeatureInfo {
                name: "f0".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 16, // Max 16 bins for 4-bit
                bin_boundaries: vec![],
            },
            FeatureInfo {
                name: "f1".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 16,
                bin_boundaries: vec![],
            },
            FeatureInfo {
                name: "f2".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 16,
                bin_boundaries: vec![],
            },
        ];

        let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);

        // Verify 4-bit support
        assert!(dataset.supports_4bit());
        assert_eq!(dataset.max_bins(), 16);
        assert_eq!(dataset.bytes_per_row_4bit(), 2); // ceil(3/2) = 2 bytes per row

        // Get 4-bit packed data
        let packed = dataset.as_row_major_4bit();

        // Expected layout (2 bytes per row):
        // Row 0: byte0 = (5 << 4) | 1 = 0x51, byte1 = (0 << 4) | 9 = 0x09
        // Row 1: byte0 = (6 << 4) | 2 = 0x62, byte1 = (0 << 4) | 10 = 0x0A
        // Row 2: byte0 = (7 << 4) | 3 = 0x73, byte1 = (0 << 4) | 11 = 0x0B
        // Row 3: byte0 = (8 << 4) | 4 = 0x84, byte1 = (0 << 4) | 12 = 0x0C

        assert_eq!(packed.len(), num_rows * 2);

        // Verify row 0
        assert_eq!(packed[0], 0x51); // (5 << 4) | 1
        assert_eq!(packed[1], 0x09); // (0 << 4) | 9

        // Verify row 1
        assert_eq!(packed[2], 0x62); // (6 << 4) | 2
        assert_eq!(packed[3], 0x0A); // (0 << 4) | 10

        // Verify row 3
        assert_eq!(packed[6], 0x84); // (8 << 4) | 4
        assert_eq!(packed[7], 0x0C); // (0 << 4) | 12

        // Verify unpacking works correctly
        for row in 0..num_rows {
            let row_offset = row * 2;
            // Feature 0 (low nibble of byte 0)
            let bin0 = packed[row_offset] & 0x0F;
            assert_eq!(bin0, (row + 1) as u8);
            // Feature 1 (high nibble of byte 0)
            let bin1 = (packed[row_offset] >> 4) & 0x0F;
            assert_eq!(bin1, (row + 5) as u8);
            // Feature 2 (low nibble of byte 1)
            let bin2 = packed[row_offset + 1] & 0x0F;
            assert_eq!(bin2, (row + 9) as u8);
        }
    }

    #[test]
    fn test_4bit_not_supported_for_large_bins() {
        let features = vec![0u8, 1, 2, 3, 100, 101, 102, 103]; // Feature 1 has large values
        let targets = vec![1.0f32, 2.0, 3.0, 4.0];
        let feature_info = vec![
            FeatureInfo {
                name: "f0".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 16,
                bin_boundaries: vec![],
            },
            FeatureInfo {
                name: "f1".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 128, // More than 16 bins
                bin_boundaries: vec![],
            },
        ];

        let dataset = BinnedDataset::new(4, features, targets, feature_info);

        assert!(!dataset.supports_4bit());
        assert_eq!(dataset.max_bins(), 128);
    }

    #[test]
    fn test_4bit_via_bin_storage_trait() {
        use crate::backend::BinStorage;

        let features = vec![1u8, 2, 5, 6];
        let targets = vec![1.0f32, 2.0];
        let feature_info = vec![
            FeatureInfo {
                name: "f0".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 8,
                bin_boundaries: vec![],
            },
            FeatureInfo {
                name: "f1".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 8,
                bin_boundaries: vec![],
            },
        ];

        let dataset = BinnedDataset::new(2, features, targets, feature_info);
        let storage: &dyn BinStorage = &dataset;

        assert!(storage.supports_4bit());
        assert_eq!(storage.max_bins(), 8);

        let packed = storage.as_row_major_4bit();
        assert!(packed.is_some());

        let data = packed.unwrap();
        assert_eq!(data.len(), 2); // 2 rows, 1 byte per row (2 features)
        // Row 0: (5 << 4) | 1 = 0x51
        // Row 1: (6 << 4) | 2 = 0x62
        assert_eq!(data[0], 0x51);
        assert_eq!(data[1], 0x62);
    }
}
