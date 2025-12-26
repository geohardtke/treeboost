//! Backend trait definitions for histogram building and split finding.
//!
//! These traits abstract over different hardware backends:
//! - Scalar (AVX2/NEON loads, scalar scatter) - current implementation
//! - WGPU (all GPUs via Vulkan/Metal/DX12) - future
//! - AVX-512 tensor-tile (vpconflictd) - future
//! - SVE2 tensor-tile (HISTCNT) - future
//! - Native backends: CUDA, ROCm, Metal - future extreme optimization

use crate::histogram::Histogram;

// Re-export SparseColumn from dataset to avoid duplication
pub use crate::dataset::SparseColumn;

/// Trait for accessing binned feature data.
///
/// Abstracts over different data layouts:
/// - Column-major (`bins[feature][row]`) for scalar backend
/// - Row-major (`bins[row][feature]`) for tensor-tile backends (GPU/AVX-512/SVE2)
pub trait BinStorage: Sync {
    /// Get the bin value for a specific row and feature.
    fn get_bin(&self, row: usize, feature: usize) -> u8;

    /// Total number of rows in the dataset.
    fn num_rows(&self) -> usize;

    /// Total number of features.
    fn num_features(&self) -> usize;

    /// Get a feature column as a contiguous slice (for scalar backend).
    /// Returns None if the storage is row-major.
    fn feature_column(&self, feature: usize) -> Option<&[u8]>;

    /// Get sparse column representation if available.
    fn sparse_column(&self, feature: usize) -> Option<&SparseColumn>;

    /// Check if a feature has sparse representation.
    fn is_sparse(&self, feature: usize) -> bool {
        self.sparse_column(feature).is_some()
    }

    /// Get the entire dataset as row-major layout (for tensor-tile backends).
    /// Returns None if the storage is column-major.
    fn as_row_major(&self) -> Option<&[u8]> {
        None
    }
}

/// Configuration for split finding.
#[derive(Debug, Clone, Copy)]
pub struct SplitConfig {
    /// L2 regularization parameter
    pub lambda: f32,
    /// Minimum samples required in each leaf
    pub min_samples_leaf: u32,
    /// Minimum hessian sum required in each leaf
    pub min_hessian_leaf: f32,
    /// Minimum gain to accept a split
    pub min_gain: f32,
    /// Shannon entropy regularization weight
    pub entropy_weight: f32,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            min_samples_leaf: 20,
            min_hessian_leaf: 1.0,
            min_gain: 0.0,
            entropy_weight: 0.0,
        }
    }
}

/// Result of split finding for a single feature.
#[derive(Debug, Clone, Copy)]
pub struct SplitCandidate {
    /// Feature index
    pub feature: usize,
    /// Bin threshold (split at bin <= threshold)
    pub threshold: u8,
    /// Gain from this split
    pub gain: f32,
    /// Left child gradient sum
    pub left_gradient: f32,
    /// Left child hessian sum
    pub left_hessian: f32,
    /// Left child sample count
    pub left_count: u32,
    /// Right child gradient sum
    pub right_gradient: f32,
    /// Right child hessian sum
    pub right_hessian: f32,
    /// Right child sample count
    pub right_count: u32,
}

/// Main trait for histogram building backends.
///
/// Different backends can implement this trait to provide hardware-accelerated
/// histogram building:
/// - `ScalarBackend`: Current CPU implementation (AVX2/NEON loads)
/// - `WgpuBackend`: GPU via WGPU (Vulkan/Metal/DX12) - future
/// - `Avx512Backend`: AVX-512 tensor-tile with vpconflictd - future
/// - `Sve2Backend`: ARM SVE2 tensor-tile with HISTCNT - future
pub trait HistogramBackend: Send + Sync {
    /// Human-readable name for this backend.
    fn name(&self) -> &'static str;

    /// Whether this backend uses tensor-tile (2D row-major) layout.
    /// True for GPU/AVX-512/SVE2, false for scalar.
    fn is_tensor_tile(&self) -> bool;

    /// Build histograms for all features at a tree node.
    ///
    /// # Arguments
    /// * `bins` - Binned feature data
    /// * `grad_hess` - Interleaved (gradient, hessian) pairs for each row
    /// * `row_indices` - Which rows belong to this node
    ///
    /// # Returns
    /// A vector of histograms, one per feature.
    fn build_histograms(
        &self,
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
    ) -> Vec<Histogram>;

    /// Build sibling histogram using the subtraction trick.
    ///
    /// For a parent node with histogram H_parent, if we compute the smaller
    /// child histogram H_smaller, we can derive the larger child as:
    /// H_larger = H_parent - H_smaller
    ///
    /// This halves the computation for child histogram building.
    fn build_histograms_sibling(
        &self,
        parent: &[Histogram],
        smaller_child: &[Histogram],
    ) -> Vec<Histogram>;

    /// Find the best split for each feature.
    ///
    /// # Arguments
    /// * `histograms` - One histogram per feature
    /// * `config` - Split finding configuration
    ///
    /// # Returns
    /// The best split candidate, or None if no valid split exists.
    fn find_best_split(
        &self,
        histograms: &[Histogram],
        config: &SplitConfig,
    ) -> Option<SplitCandidate>;
}
