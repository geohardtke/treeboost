//! Packed 4-bit bin storage for memory efficiency
//!
//! When features have ≤16 unique bins, we can pack two bins per byte,
//! reducing memory usage by 50% compared to u8 storage.
//!
//! This is particularly useful for:
//! - Low-cardinality categorical features
//! - Features with limited precision needs
//! - Memory-constrained environments

use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

/// Storage mode for binned features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize, Default)]
pub enum StorageMode {
    /// Standard u8 storage (256 bins max)
    #[default]
    U8,
    /// Packed 4-bit storage (16 bins max, 2x memory savings)
    Packed4Bit,
}

/// Packed column storing two 4-bit bins per byte
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct PackedColumn {
    /// Packed data: each byte contains 2 bins (high 4 bits + low 4 bits)
    data: Vec<u8>,
    /// Number of rows (for handling odd row counts)
    num_rows: usize,
}

impl PackedColumn {
    /// Create a new packed column from u8 bins
    ///
    /// # Panics
    /// Panics if any bin value exceeds 15 (4-bit max)
    pub fn from_bins(bins: &[u8]) -> Self {
        let num_rows = bins.len();
        let packed_len = num_rows.div_ceil(2);
        let mut data = Vec::with_capacity(packed_len);

        for chunk in bins.chunks(2) {
            debug_assert!(
                chunk[0] <= 15,
                "Bin value {} exceeds 4-bit max (15)",
                chunk[0]
            );

            let high = chunk[0] << 4;
            let low = if chunk.len() > 1 {
                debug_assert!(
                    chunk[1] <= 15,
                    "Bin value {} exceeds 4-bit max (15)",
                    chunk[1]
                );
                chunk[1]
            } else {
                0 // Padding for odd row count
            };
            data.push(high | low);
        }

        Self { data, num_rows }
    }

    /// Get bin value at row index
    #[inline]
    pub fn get(&self, row_idx: usize) -> u8 {
        debug_assert!(row_idx < self.num_rows);
        let byte_idx = row_idx / 2;
        let byte = self.data[byte_idx];
        if row_idx.is_multiple_of(2) {
            byte >> 4
        } else {
            byte & 0x0F
        }
    }

    /// Number of rows
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Memory usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Unpack to u8 slice for compatibility with histogram builder
    ///
    /// Uses SIMD-accelerated unpacking when available.
    pub fn unpack(&self) -> Vec<u8> {
        let mut result = vec![0u8; self.num_rows];
        self.unpack_to_buffer(&mut result);
        result
    }

    /// Unpack into a pre-allocated buffer
    ///
    /// Uses SIMD-accelerated unpacking (AVX2) when available.
    ///
    /// # Arguments
    /// * `buffer` - Output buffer, must have length >= self.num_rows
    pub fn unpack_to_buffer(&self, buffer: &mut [u8]) {
        debug_assert!(buffer.len() >= self.num_rows);

        // Handle odd row count: we may have packed one extra padding byte
        let full_bytes = self.num_rows / 2;

        if full_bytes > 0 {
            // Unpack full bytes using SIMD kernel
            crate::backend::scalar::kernel::unpack_4bit(
                &self.data[..full_bytes],
                &mut buffer[..full_bytes * 2],
            );
        }

        // Handle odd row count manually (last row if num_rows is odd)
        if self.num_rows % 2 == 1 {
            let last_byte = self.data[full_bytes];
            buffer[self.num_rows - 1] = last_byte >> 4;
        }
    }

    /// Unpack a range of rows into a buffer
    ///
    /// Optimized for histogram building where we process blocks of rows.
    ///
    /// # Arguments
    /// * `start_row` - First row to unpack
    /// * `count` - Number of rows to unpack
    /// * `buffer` - Output buffer, must have length >= count
    pub fn unpack_range(&self, start_row: usize, count: usize, buffer: &mut [u8]) {
        debug_assert!(start_row + count <= self.num_rows);
        debug_assert!(buffer.len() >= count);

        // Fast path: if start is even and count is even, we can use SIMD directly
        if start_row.is_multiple_of(2) && count.is_multiple_of(2) {
            let start_byte = start_row / 2;
            let byte_count = count / 2;
            crate::backend::scalar::kernel::unpack_4bit(
                &self.data[start_byte..start_byte + byte_count],
                &mut buffer[..count],
            );
            return;
        }

        // Fallback: handle unaligned access
        for (i, buf) in buffer[..count].iter_mut().enumerate() {
            *buf = self.get(start_row + i);
        }
    }

    /// Get raw packed data (for direct SIMD processing)
    #[inline]
    pub fn packed_data(&self) -> &[u8] {
        &self.data
    }

    /// Iterate over bins (for histogram building)
    #[inline]
    pub fn iter(&self) -> PackedColumnIter<'_> {
        PackedColumnIter {
            column: self,
            idx: 0,
        }
    }
}

/// Iterator over packed column bins
pub struct PackedColumnIter<'a> {
    column: &'a PackedColumn,
    idx: usize,
}

impl Iterator for PackedColumnIter<'_> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.column.num_rows {
            let val = self.column.get(self.idx);
            self.idx += 1;
            Some(val)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.column.num_rows - self.idx;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PackedColumnIter<'_> {}

/// Check if a column can be packed (all values ≤ 15)
pub fn can_pack(bins: &[u8]) -> bool {
    bins.iter().all(|&b| b <= 15)
}

/// Analyze column for optimal storage mode
pub fn optimal_storage(bins: &[u8]) -> StorageMode {
    if can_pack(bins) {
        StorageMode::Packed4Bit
    } else {
        StorageMode::U8
    }
}

use super::{BinnedDataset, FeatureInfo};

/// Feature storage that can be either u8 or packed 4-bit
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub enum FeatureStorage {
    /// Standard u8 storage
    U8(Vec<u8>),
    /// Packed 4-bit storage
    Packed(PackedColumn),
}

impl FeatureStorage {
    /// Get bin value at row index
    #[inline]
    pub fn get(&self, row_idx: usize) -> u8 {
        match self {
            FeatureStorage::U8(data) => data[row_idx],
            FeatureStorage::Packed(packed) => packed.get(row_idx),
        }
    }

    /// Number of rows
    #[inline]
    pub fn num_rows(&self) -> usize {
        match self {
            FeatureStorage::U8(data) => data.len(),
            FeatureStorage::Packed(packed) => packed.num_rows(),
        }
    }

    /// Memory usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        match self {
            FeatureStorage::U8(data) => data.len(),
            FeatureStorage::Packed(packed) => packed.memory_bytes(),
        }
    }

    /// Storage mode
    #[inline]
    pub fn mode(&self) -> StorageMode {
        match self {
            FeatureStorage::U8(_) => StorageMode::U8,
            FeatureStorage::Packed(_) => StorageMode::Packed4Bit,
        }
    }

    /// Create from bins with automatic mode selection
    pub fn from_bins(bins: Vec<u8>) -> Self {
        if can_pack(&bins) {
            FeatureStorage::Packed(PackedColumn::from_bins(&bins))
        } else {
            FeatureStorage::U8(bins)
        }
    }

    /// Create from bins with explicit mode
    pub fn from_bins_with_mode(bins: Vec<u8>, mode: StorageMode) -> Self {
        match mode {
            StorageMode::U8 => FeatureStorage::U8(bins),
            StorageMode::Packed4Bit => {
                debug_assert!(can_pack(&bins), "Cannot pack bins: values exceed 15");
                FeatureStorage::Packed(PackedColumn::from_bins(&bins))
            }
        }
    }

    /// Unpack to u8 vector
    pub fn to_u8(&self) -> Vec<u8> {
        match self {
            FeatureStorage::U8(data) => data.clone(),
            FeatureStorage::Packed(packed) => packed.unpack(),
        }
    }
}

/// Memory-optimized dataset with per-feature storage mode selection
///
/// Automatically uses 4-bit packing for features with ≤16 unique bins,
/// providing up to 50% memory savings for low-cardinality features.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct PackedDataset {
    /// Number of rows (samples)
    num_rows: usize,
    /// Per-feature storage (either u8 or packed)
    feature_data: Vec<FeatureStorage>,
    /// Target values
    targets: Vec<f32>,
    /// Feature metadata
    feature_info: Vec<FeatureInfo>,
}

impl PackedDataset {
    /// Create a new packed dataset from a BinnedDataset
    ///
    /// Automatically selects optimal storage mode for each feature.
    pub fn from_binned(dataset: &BinnedDataset) -> Self {
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        let feature_data: Vec<FeatureStorage> = (0..num_features)
            .map(|f| {
                let column = dataset.feature_column(f).to_vec();
                FeatureStorage::from_bins(column)
            })
            .collect();

        Self {
            num_rows,
            feature_data,
            targets: dataset.targets().to_vec(),
            feature_info: dataset.all_feature_info().to_vec(),
        }
    }

    /// Create with explicit storage modes per feature
    pub fn from_binned_with_modes(dataset: &BinnedDataset, modes: &[StorageMode]) -> Self {
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        debug_assert_eq!(modes.len(), num_features);

        let feature_data: Vec<FeatureStorage> = (0..num_features)
            .map(|f| {
                let column = dataset.feature_column(f).to_vec();
                FeatureStorage::from_bins_with_mode(column, modes[f])
            })
            .collect();

        Self {
            num_rows,
            feature_data,
            targets: dataset.targets().to_vec(),
            feature_info: dataset.all_feature_info().to_vec(),
        }
    }

    /// Number of rows
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of features
    #[inline]
    pub fn num_features(&self) -> usize {
        self.feature_data.len()
    }

    /// Get bin value for a specific row and feature
    #[inline]
    pub fn get_bin(&self, row_idx: usize, feature_idx: usize) -> u8 {
        self.feature_data[feature_idx].get(row_idx)
    }

    /// Get feature storage
    #[inline]
    pub fn feature_storage(&self, feature_idx: usize) -> &FeatureStorage {
        &self.feature_data[feature_idx]
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

    /// Get feature info
    #[inline]
    pub fn feature_info(&self, feature_idx: usize) -> &FeatureInfo {
        &self.feature_info[feature_idx]
    }

    /// Get all feature info
    #[inline]
    pub fn all_feature_info(&self) -> &[FeatureInfo] {
        &self.feature_info
    }

    /// Total memory usage for feature data (bytes)
    pub fn feature_memory_bytes(&self) -> usize {
        self.feature_data.iter().map(|f| f.memory_bytes()).sum()
    }

    /// Memory savings compared to u8 storage
    pub fn memory_savings(&self) -> f64 {
        let packed_size = self.feature_memory_bytes();
        let u8_size = self.num_rows * self.num_features();
        if u8_size == 0 {
            0.0
        } else {
            1.0 - (packed_size as f64 / u8_size as f64)
        }
    }

    /// Get storage modes for all features
    pub fn storage_modes(&self) -> Vec<StorageMode> {
        self.feature_data.iter().map(|f| f.mode()).collect()
    }

    /// Convert back to BinnedDataset (unpacks all features)
    ///
    /// Uses SIMD-accelerated unpacking for packed features.
    /// For datasets with many features, uses parallel unpacking.
    pub fn to_binned(&self) -> BinnedDataset {
        let num_features = self.num_features();
        let total_size = self.num_rows * num_features;
        let mut features = vec![0u8; total_size];

        // Use parallel unpacking for datasets with many features
        if num_features >= 8 {
            // Split into feature-sized chunks and unpack in parallel
            let chunk_size = self.num_rows;
            features
                .par_chunks_mut(chunk_size)
                .zip(self.feature_data.par_iter())
                .for_each(|(dest, storage)| match storage {
                    FeatureStorage::U8(data) => {
                        dest.copy_from_slice(data);
                    }
                    FeatureStorage::Packed(packed) => {
                        packed.unpack_to_buffer(dest);
                    }
                });
        } else {
            // Sequential for small feature counts
            for (f, storage) in self.feature_data.iter().enumerate() {
                let start = f * self.num_rows;
                let dest = &mut features[start..start + self.num_rows];

                match storage {
                    FeatureStorage::U8(data) => {
                        dest.copy_from_slice(data);
                    }
                    FeatureStorage::Packed(packed) => {
                        packed.unpack_to_buffer(dest);
                    }
                }
            }
        }

        BinnedDataset::new(
            self.num_rows,
            features,
            self.targets.clone(),
            self.feature_info.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_column_basic() {
        let bins = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let packed = PackedColumn::from_bins(&bins);

        assert_eq!(packed.num_rows(), 8);
        assert_eq!(packed.memory_bytes(), 4); // 8 bins / 2 = 4 bytes

        for (i, &expected) in bins.iter().enumerate() {
            assert_eq!(packed.get(i), expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_packed_column_odd_rows() {
        let bins = vec![15, 8, 3, 11, 7];
        let packed = PackedColumn::from_bins(&bins);

        assert_eq!(packed.num_rows(), 5);
        assert_eq!(packed.memory_bytes(), 3); // ceil(5/2) = 3 bytes

        for (i, &expected) in bins.iter().enumerate() {
            assert_eq!(packed.get(i), expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_packed_column_unpack() {
        let bins = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let packed = PackedColumn::from_bins(&bins);
        let unpacked = packed.unpack();

        assert_eq!(unpacked, bins);
    }

    #[test]
    fn test_packed_column_iterator() {
        let bins = vec![0, 15, 7, 8, 3];
        let packed = PackedColumn::from_bins(&bins);

        let collected: Vec<u8> = packed.iter().collect();
        assert_eq!(collected, bins);
    }

    #[test]
    fn test_can_pack() {
        assert!(can_pack(&[0, 1, 2, 3, 15]));
        assert!(can_pack(&[0, 15, 8, 7]));
        assert!(!can_pack(&[0, 1, 16])); // 16 exceeds 4-bit max
        assert!(!can_pack(&[255, 0]));
    }

    #[test]
    fn test_optimal_storage() {
        assert_eq!(optimal_storage(&[0, 1, 2, 3]), StorageMode::Packed4Bit);
        assert_eq!(optimal_storage(&[0, 1, 2, 255]), StorageMode::U8);
    }

    #[test]
    fn test_memory_savings() {
        let bins: Vec<u8> = (0..1000).map(|i| (i % 16) as u8).collect();

        let unpacked_size = bins.len(); // 1000 bytes
        let packed = PackedColumn::from_bins(&bins);
        let packed_size = packed.memory_bytes(); // 500 bytes

        assert_eq!(packed_size, 500);
        assert_eq!(unpacked_size / packed_size, 2); // 2x savings
    }

    #[test]
    fn test_feature_storage_auto() {
        // Packable column
        let bins_low = vec![0, 1, 2, 3, 4, 5];
        let storage_low = FeatureStorage::from_bins(bins_low.clone());
        assert_eq!(storage_low.mode(), StorageMode::Packed4Bit);
        for (i, &expected) in bins_low.iter().enumerate() {
            assert_eq!(storage_low.get(i), expected);
        }

        // Non-packable column
        let bins_high = vec![0, 1, 100, 200, 255];
        let storage_high = FeatureStorage::from_bins(bins_high.clone());
        assert_eq!(storage_high.mode(), StorageMode::U8);
        for (i, &expected) in bins_high.iter().enumerate() {
            assert_eq!(storage_high.get(i), expected);
        }
    }

    #[test]
    fn test_packed_dataset() {
        use crate::dataset::FeatureType;

        // Create a BinnedDataset with mixed storage potential
        let num_rows = 100;
        // f0: packable (0-15), f1: not packable (0-255)
        let mut features = Vec::with_capacity(num_rows * 2);
        for r in 0..num_rows {
            features.push((r % 16) as u8); // f0: packable
        }
        for r in 0..num_rows {
            features.push((r % 256) as u8); // f1: not packable
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| i as f32).collect();
        let feature_info = vec![
            FeatureInfo {
                name: "f0".to_string(),
                feature_type: FeatureType::Categorical,
                num_bins: 16,
                bin_boundaries: vec![],
            },
            FeatureInfo {
                name: "f1".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
            },
        ];

        let binned = BinnedDataset::new(num_rows, features, targets, feature_info);
        let packed = PackedDataset::from_binned(&binned);

        // Verify storage modes
        let modes = packed.storage_modes();
        assert_eq!(modes[0], StorageMode::Packed4Bit); // f0 packed
        assert_eq!(modes[1], StorageMode::U8); // f1 not packed

        // Verify data access
        for r in 0..num_rows {
            assert_eq!(packed.get_bin(r, 0), binned.get_bin(r, 0));
            assert_eq!(packed.get_bin(r, 1), binned.get_bin(r, 1));
        }

        // Verify memory savings (should be ~25% since f0 is halved)
        let savings = packed.memory_savings();
        assert!(savings > 0.2 && savings < 0.3, "Savings: {}", savings);

        // Verify round-trip
        let unpacked = packed.to_binned();
        for r in 0..num_rows {
            for f in 0..2 {
                assert_eq!(unpacked.get_bin(r, f), binned.get_bin(r, f));
            }
        }
    }
}
