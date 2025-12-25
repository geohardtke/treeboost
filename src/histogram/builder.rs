//! Parallel histogram construction
//!
//! # Performance Notes
//!
//! Histogram accumulation is inherently difficult to vectorize because:
//! 1. The scatter operation (accumulating into bins) has potential conflicts
//! 2. AVX2 gather has high latency for random access patterns
//! 3. True SIMD scatter requires AVX-512 conflict detection
//!
//! Our approach:
//! - **Contiguous rows** (e.g., root node): Use AVX2 `loadu_ps` for fast SIMD loads
//! - **Indexed rows** (e.g., child nodes): Use 8x unrolled scalar for ILP
//!
//! The scatter (accumulation) is always scalar due to bin conflicts.
//!
//! XGBoost/LightGBM use different strategies:
//! - XGBoost: Local buffers with prefetching
//! - LightGBM: Pre-reorders data for sequential access

use crate::dataset::BinnedDataset;
use crate::histogram::{Histogram, NodeHistograms};
use crate::kernel;
use rayon::prelude::*;

/// Histogram builder with feature-parallel construction
pub struct HistogramBuilder {
    /// Number of threads for parallel construction
    num_threads: usize,
}

impl Default for HistogramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HistogramBuilder {
    /// Create a new histogram builder
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Set number of threads
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Build histograms for all features at a node
    ///
    /// # Arguments
    /// * `dataset` - The binned dataset
    /// * `row_indices` - Indices of rows belonging to this node
    /// * `gradients` - Gradient for each row in the full dataset
    /// * `hessians` - Hessian for each row in the full dataset
    pub fn build(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> NodeHistograms {
        let num_features = dataset.num_features();

        // Check if rows are contiguous (0..n) - enables SIMD load optimization
        let is_contiguous = Self::is_contiguous(row_indices);

        // Build histograms in parallel across features
        let histograms: Vec<Histogram> = (0..num_features)
            .into_par_iter()
            .map(|feature_idx| {
                if is_contiguous {
                    self.build_single_feature_contiguous(
                        dataset,
                        feature_idx,
                        row_indices.len(),
                        gradients,
                        hessians,
                    )
                } else {
                    self.build_single_feature_indexed(
                        dataset,
                        feature_idx,
                        row_indices,
                        gradients,
                        hessians,
                    )
                }
            })
            .collect();

        NodeHistograms::from_vec(histograms)
    }

    /// Check if row_indices represents contiguous range 0..n
    #[inline]
    fn is_contiguous(row_indices: &[usize]) -> bool {
        if row_indices.is_empty() {
            return true;
        }
        // Check first element and length - if first is 0 and indices are sequential
        row_indices[0] == 0 && row_indices.last() == Some(&(row_indices.len() - 1))
    }

    /// Build histogram for contiguous rows using SIMD-optimized loads
    ///
    /// Uses AVX2 `loadu_ps` to load 8 gradients/hessians at once (sequential access).
    /// The scatter is still scalar due to bin conflicts.
    #[inline]
    fn build_single_feature_contiguous(
        &self,
        dataset: &BinnedDataset,
        feature_idx: usize,
        num_rows: usize,
        gradients: &[f32],
        hessians: &[f32],
    ) -> Histogram {
        let feature_column = dataset.feature_column(feature_idx);

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            kernel::histogram_accumulate_contiguous(
                feature_column.as_ptr(),
                num_rows,
                gradients.as_ptr(),
                hessians.as_ptr(),
                hist_grads.as_mut_ptr(),
                hist_hess.as_mut_ptr(),
                hist_counts.as_mut_ptr(),
            );
        }

        // Convert raw arrays to Histogram
        Histogram::from_raw_arrays(&hist_grads, &hist_hess, &hist_counts)
    }

    /// Build histogram for a single feature using indexed access
    ///
    /// Uses 8x unrolled scalar loop for best performance.
    /// Note: AVX2 gather was tested but is slower due to gather latency overhead
    /// and the fundamental scatter bottleneck in histogram accumulation.
    #[inline]
    fn build_single_feature_indexed(
        &self,
        dataset: &BinnedDataset,
        feature_idx: usize,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> Histogram {
        let feature_column = dataset.feature_column(feature_idx);
        self.build_single_feature_scalar(feature_column, row_indices, gradients, hessians)
    }

    /// Build histogram using scalar 8x unrolled loop
    ///
    /// This is the fastest path for histogram accumulation because:
    /// 1. 8x unrolling provides good ILP
    /// 2. AVX2 gather has high latency for scattered indices
    /// 3. The scatter operation cannot be vectorized due to bin conflicts
    #[inline]
    fn build_single_feature_scalar(
        &self,
        feature_column: &[u8],
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> Histogram {
        let mut histogram = Histogram::new();

        let len = row_indices.len();
        let chunks = len / 8;
        let remainder = len % 8;

        unsafe {
            // Process 8 samples at a time
            for i in 0..chunks {
                let base = i * 8;

                let idx0 = *row_indices.get_unchecked(base);
                let idx1 = *row_indices.get_unchecked(base + 1);
                let idx2 = *row_indices.get_unchecked(base + 2);
                let idx3 = *row_indices.get_unchecked(base + 3);
                let idx4 = *row_indices.get_unchecked(base + 4);
                let idx5 = *row_indices.get_unchecked(base + 5);
                let idx6 = *row_indices.get_unchecked(base + 6);
                let idx7 = *row_indices.get_unchecked(base + 7);

                let bin0 = *feature_column.get_unchecked(idx0) as usize;
                let bin1 = *feature_column.get_unchecked(idx1) as usize;
                let bin2 = *feature_column.get_unchecked(idx2) as usize;
                let bin3 = *feature_column.get_unchecked(idx3) as usize;
                let bin4 = *feature_column.get_unchecked(idx4) as usize;
                let bin5 = *feature_column.get_unchecked(idx5) as usize;
                let bin6 = *feature_column.get_unchecked(idx6) as usize;
                let bin7 = *feature_column.get_unchecked(idx7) as usize;

                let grad0 = *gradients.get_unchecked(idx0);
                let grad1 = *gradients.get_unchecked(idx1);
                let grad2 = *gradients.get_unchecked(idx2);
                let grad3 = *gradients.get_unchecked(idx3);
                let grad4 = *gradients.get_unchecked(idx4);
                let grad5 = *gradients.get_unchecked(idx5);
                let grad6 = *gradients.get_unchecked(idx6);
                let grad7 = *gradients.get_unchecked(idx7);

                let hess0 = *hessians.get_unchecked(idx0);
                let hess1 = *hessians.get_unchecked(idx1);
                let hess2 = *hessians.get_unchecked(idx2);
                let hess3 = *hessians.get_unchecked(idx3);
                let hess4 = *hessians.get_unchecked(idx4);
                let hess5 = *hessians.get_unchecked(idx5);
                let hess6 = *hessians.get_unchecked(idx6);
                let hess7 = *hessians.get_unchecked(idx7);

                histogram.bins_mut().get_unchecked_mut(bin0).accumulate(grad0, hess0);
                histogram.bins_mut().get_unchecked_mut(bin1).accumulate(grad1, hess1);
                histogram.bins_mut().get_unchecked_mut(bin2).accumulate(grad2, hess2);
                histogram.bins_mut().get_unchecked_mut(bin3).accumulate(grad3, hess3);
                histogram.bins_mut().get_unchecked_mut(bin4).accumulate(grad4, hess4);
                histogram.bins_mut().get_unchecked_mut(bin5).accumulate(grad5, hess5);
                histogram.bins_mut().get_unchecked_mut(bin6).accumulate(grad6, hess6);
                histogram.bins_mut().get_unchecked_mut(bin7).accumulate(grad7, hess7);
            }

            // Handle remainder
            let base = chunks * 8;
            for i in 0..remainder {
                let idx = *row_indices.get_unchecked(base + i);
                let bin = *feature_column.get_unchecked(idx) as usize;
                let grad = *gradients.get_unchecked(idx);
                let hess = *hessians.get_unchecked(idx);
                histogram.bins_mut().get_unchecked_mut(bin).accumulate(grad, hess);
            }
        }

        histogram
    }

    /// Build histograms using data parallelism (for large nodes)
    ///
    /// Chunks the rows across threads, builds partial histograms, then reduces.
    pub fn build_data_parallel(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> NodeHistograms {
        let num_features = dataset.num_features();

        // If small enough, use simple feature-parallel
        if row_indices.len() < 10000 {
            return self.build(dataset, row_indices, gradients, hessians);
        }

        // Chunk rows and build partial histograms
        let chunk_size = (row_indices.len() / self.num_threads).max(1000);

        let partial_histograms: Vec<NodeHistograms> = row_indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut node_hists = NodeHistograms::new(num_features);

                for feature_idx in 0..num_features {
                    let feature_column = dataset.feature_column(feature_idx);
                    let hist = node_hists.get_mut(feature_idx);

                    for &row_idx in chunk {
                        let bin = feature_column[row_idx];
                        hist.accumulate(bin, gradients[row_idx], hessians[row_idx]);
                    }
                }

                node_hists
            })
            .collect();

        // Reduce partial histograms
        let mut result = NodeHistograms::new(num_features);
        for partial in partial_histograms {
            result.merge(&partial);
        }

        result
    }

    /// Build sibling histogram using Histogram Subtraction Trick
    ///
    /// Instead of building the larger sibling directly,
    /// compute it as: sibling = parent - smaller_child
    ///
    /// This halves the computation for splits.
    pub fn build_sibling(
        parent: &NodeHistograms,
        smaller_child: &NodeHistograms,
    ) -> NodeHistograms {
        NodeHistograms::from_subtraction(parent, smaller_child)
    }
}

// Extend NodeHistograms with from_vec constructor
impl NodeHistograms {
    /// Create from a vector of histograms
    pub fn from_vec(histograms: Vec<Histogram>) -> Self {
        Self { histograms }
    }

    /// Get internal histograms vector
    pub fn into_vec(self) -> Vec<Histogram> {
        self.histograms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{BinnedDataset, FeatureInfo, FeatureType};

    fn create_test_dataset() -> BinnedDataset {
        let num_rows = 100;
        let num_features = 3;

        // Generate deterministic test data
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r + f * 7) % 256) as u8);
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| i as f32).collect();
        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("f{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_build_histograms() {
        let dataset = create_test_dataset();
        let num_rows = dataset.num_rows();

        let gradients: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.1).collect();
        let hessians: Vec<f32> = vec![1.0; num_rows];
        let row_indices: Vec<usize> = (0..num_rows).collect();

        let builder = HistogramBuilder::new();
        let hists = builder.build(&dataset, &row_indices, &gradients, &hessians);

        assert_eq!(hists.num_features(), 3);

        // Check total count
        let total_count: u32 = hists.get(0).bins().iter().map(|b| b.count).sum();
        assert_eq!(total_count, num_rows as u32);
    }

    #[test]
    fn test_subtraction_trick() {
        let dataset = create_test_dataset();
        let num_rows = dataset.num_rows();

        let gradients: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.1).collect();
        let hessians: Vec<f32> = vec![1.0; num_rows];

        let all_rows: Vec<usize> = (0..num_rows).collect();
        let left_rows: Vec<usize> = (0..num_rows / 2).collect();
        let right_rows: Vec<usize> = (num_rows / 2..num_rows).collect();

        let builder = HistogramBuilder::new();

        // Build parent and left child
        let parent = builder.build(&dataset, &all_rows, &gradients, &hessians);
        let left = builder.build(&dataset, &left_rows, &gradients, &hessians);

        // Build right using subtraction
        let right_subtracted = HistogramBuilder::build_sibling(&parent, &left);

        // Build right directly for comparison
        let right_direct = builder.build(&dataset, &right_rows, &gradients, &hessians);

        // Compare - should be identical (within floating point tolerance)
        for f in 0..dataset.num_features() {
            for bin in 0..=255u8 {
                let sub = right_subtracted.get(f).get(bin);
                let direct = right_direct.get(f).get(bin);

                assert!(
                    (sub.sum_gradients - direct.sum_gradients).abs() < 1e-5,
                    "Gradient mismatch at feature {} bin {}",
                    f,
                    bin
                );
                assert_eq!(sub.count, direct.count);
            }
        }
    }
}
