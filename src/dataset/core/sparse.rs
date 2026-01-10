//! Sparse column storage for memory-efficient histogram building
//!
//! For features with high sparsity (many default/zero values), storing only
//! non-default entries provides significant memory and computation savings.

use rkyv::{Archive, Deserialize, Serialize};

use super::feature_info::SPARSITY_THRESHOLD;

// =============================================================================
// SparseColumn
// =============================================================================

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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DEFAULT_BIN;

    #[test]
    fn test_sparse_column_from_dense() {
        // Dense column with 80% zeros
        let dense = vec![0u8, 0, 1, 0, 0, 2, 0, 0, 0, 3];
        let sparse = SparseColumn::from_dense(&dense, DEFAULT_BIN);

        assert_eq!(sparse.num_rows, 10);
        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.indices, vec![2, 5, 9]);
        assert_eq!(sparse.values, vec![1, 2, 3]);
    }

    #[test]
    fn test_sparse_column_sparsity() {
        // 90% zeros (9 out of 10)
        let dense = vec![0u8, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let sparse = SparseColumn::from_dense(&dense, DEFAULT_BIN);

        assert_eq!(sparse.sparsity(), 0.9);
        assert!(sparse.is_sparse());
    }

    #[test]
    fn test_sparse_column_not_sparse() {
        // 50% zeros (not sparse enough)
        let dense = vec![0u8, 1, 0, 2, 0, 3, 0, 4, 0, 5];
        let sparse = SparseColumn::from_dense(&dense, DEFAULT_BIN);

        assert_eq!(sparse.sparsity(), 0.5);
        assert!(!sparse.is_sparse());
    }

    #[test]
    fn test_sparse_column_iter() {
        let dense = vec![0u8, 0, 5, 0, 7];
        let sparse = SparseColumn::from_dense(&dense, DEFAULT_BIN);

        let entries: Vec<_> = sparse.iter().collect();
        assert_eq!(entries, vec![(2, 5), (4, 7)]);
    }

    #[test]
    fn test_sparse_column_empty() {
        let sparse = SparseColumn::from_dense(&[], DEFAULT_BIN);

        assert_eq!(sparse.num_rows, 0);
        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.sparsity(), 1.0);
    }
}
