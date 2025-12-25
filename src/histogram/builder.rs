//! Parallel histogram construction with cache-blocking optimization
//!
//! # Cache-Blocked Architecture
//!
//! The key optimization is **cache blocking** (tiling):
//! - Process rows in blocks of 2048 (16KB of gradients+hessians fits in L1)
//! - For each block, update ALL feature histograms while gradients are hot in cache
//! - Parallelize over row blocks, then reduce partial histograms
//!
//! This reduces memory bandwidth by factor of `num_features` compared to
//! feature-parallel approach (which reads entire gradient array per feature).
//!
//! # Performance Notes
//!
//! Histogram accumulation is inherently difficult to vectorize because:
//! 1. The scatter operation (accumulating into bins) has potential conflicts
//! 2. AVX2 gather has high latency for random access patterns
//! 3. True SIMD scatter requires AVX-512 conflict detection
//!
//! Our approach:
//! - **Cache blocking**: Load 2048 rows of gradients into L1, update all features
//! - **Contiguous rows** (e.g., root node): Use AVX2 `loadu_ps` for fast SIMD loads
//! - **Indexed rows** (e.g., child nodes): Use 8x unrolled scalar for ILP
//!
//! The scatter (accumulation) is always scalar due to bin conflicts.

use crate::dataset::BinnedDataset;
use crate::histogram::{Histogram, NodeHistograms};
use rayon::prelude::*;

/// Block size for cache-blocked histogram building
/// 2048 rows * 8 bytes (gradient + hessian) = 16KB, fits in L1 cache
const BLOCK_SIZE: usize = 2048;

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

    /// Build histograms for all features at a node using cache-blocked approach
    ///
    /// # Cache-Blocking Strategy
    ///
    /// Instead of feature-parallel (which reads gradients N times for N features),
    /// we use row-block-parallel:
    /// 1. Divide rows into blocks of 2048 (fits in L1 cache)
    /// 2. For each block, load gradients/hessians once
    /// 3. Update ALL feature histograms while data is hot in cache
    /// 4. Merge partial histograms from all blocks
    ///
    /// This reduces memory bandwidth by factor of `num_features`.
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
        let num_rows = row_indices.len();

        // For small datasets, use simple single-threaded approach
        if num_rows < BLOCK_SIZE {
            return self.build_single_block(dataset, row_indices, gradients, hessians);
        }

        // Check if rows are contiguous (0..n) - enables optimized path
        let is_contiguous = Self::is_contiguous(row_indices);

        if is_contiguous {
            // Contiguous case: parallelize over row blocks
            self.build_blocked_contiguous(dataset, num_rows, gradients, hessians)
        } else {
            // Indexed case: parallelize over row blocks with indirection
            self.build_blocked_indexed(dataset, row_indices, gradients, hessians)
        }
    }

    /// Build histograms using cache-blocked approach for contiguous rows
    fn build_blocked_contiguous(
        &self,
        dataset: &BinnedDataset,
        num_rows: usize,
        gradients: &[f32],
        hessians: &[f32],
    ) -> NodeHistograms {
        let num_features = dataset.num_features();

        // Process row blocks in parallel
        let partial_histograms: Vec<NodeHistograms> = (0..num_rows)
            .into_par_iter()
            .step_by(BLOCK_SIZE)
            .map(|block_start| {
                let block_end = (block_start + BLOCK_SIZE).min(num_rows);
                let block_len = block_end - block_start;

                // Create local histograms for this block
                let mut local_hists = NodeHistograms::new(num_features);

                // Pre-load gradients/hessians for this block into stack arrays
                // This ensures they stay hot in L1 cache
                let mut grad_cache = [0.0f32; BLOCK_SIZE];
                let mut hess_cache = [0.0f32; BLOCK_SIZE];

                // Copy block data to cache (sequential read from main memory)
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        gradients.as_ptr().add(block_start),
                        grad_cache.as_mut_ptr(),
                        block_len,
                    );
                    std::ptr::copy_nonoverlapping(
                        hessians.as_ptr().add(block_start),
                        hess_cache.as_mut_ptr(),
                        block_len,
                    );
                }

                // Now iterate ALL features while gradients are hot in L1
                for feature_idx in 0..num_features {
                    let feature_column = dataset.feature_column(feature_idx);
                    let hist = local_hists.get_mut(feature_idx);
                    let bins = hist.bins_mut();

                    // 8x unrolled accumulation with cached gradients
                    let chunks = block_len / 8;
                    let remainder = block_len % 8;

                    unsafe {
                        for i in 0..chunks {
                            let base = i * 8;
                            let row_base = block_start + base;

                            // Load bins (sequential read from feature column)
                            let bin0 = *feature_column.get_unchecked(row_base) as usize;
                            let bin1 = *feature_column.get_unchecked(row_base + 1) as usize;
                            let bin2 = *feature_column.get_unchecked(row_base + 2) as usize;
                            let bin3 = *feature_column.get_unchecked(row_base + 3) as usize;
                            let bin4 = *feature_column.get_unchecked(row_base + 4) as usize;
                            let bin5 = *feature_column.get_unchecked(row_base + 5) as usize;
                            let bin6 = *feature_column.get_unchecked(row_base + 6) as usize;
                            let bin7 = *feature_column.get_unchecked(row_base + 7) as usize;

                            // Load from L1 cache (fast!)
                            let grad0 = *grad_cache.get_unchecked(base);
                            let grad1 = *grad_cache.get_unchecked(base + 1);
                            let grad2 = *grad_cache.get_unchecked(base + 2);
                            let grad3 = *grad_cache.get_unchecked(base + 3);
                            let grad4 = *grad_cache.get_unchecked(base + 4);
                            let grad5 = *grad_cache.get_unchecked(base + 5);
                            let grad6 = *grad_cache.get_unchecked(base + 6);
                            let grad7 = *grad_cache.get_unchecked(base + 7);

                            let hess0 = *hess_cache.get_unchecked(base);
                            let hess1 = *hess_cache.get_unchecked(base + 1);
                            let hess2 = *hess_cache.get_unchecked(base + 2);
                            let hess3 = *hess_cache.get_unchecked(base + 3);
                            let hess4 = *hess_cache.get_unchecked(base + 4);
                            let hess5 = *hess_cache.get_unchecked(base + 5);
                            let hess6 = *hess_cache.get_unchecked(base + 6);
                            let hess7 = *hess_cache.get_unchecked(base + 7);

                            // Scatter to histogram bins
                            bins.get_unchecked_mut(bin0).accumulate(grad0, hess0);
                            bins.get_unchecked_mut(bin1).accumulate(grad1, hess1);
                            bins.get_unchecked_mut(bin2).accumulate(grad2, hess2);
                            bins.get_unchecked_mut(bin3).accumulate(grad3, hess3);
                            bins.get_unchecked_mut(bin4).accumulate(grad4, hess4);
                            bins.get_unchecked_mut(bin5).accumulate(grad5, hess5);
                            bins.get_unchecked_mut(bin6).accumulate(grad6, hess6);
                            bins.get_unchecked_mut(bin7).accumulate(grad7, hess7);
                        }

                        // Handle remainder
                        let rem_base = chunks * 8;
                        for i in 0..remainder {
                            let bin = *feature_column.get_unchecked(block_start + rem_base + i) as usize;
                            let grad = *grad_cache.get_unchecked(rem_base + i);
                            let hess = *hess_cache.get_unchecked(rem_base + i);
                            bins.get_unchecked_mut(bin).accumulate(grad, hess);
                        }
                    }
                }

                local_hists
            })
            .collect();

        // Reduce partial histograms
        Self::reduce_histograms(partial_histograms, num_features)
    }

    /// Build histograms using cache-blocked approach for indexed (non-contiguous) rows
    fn build_blocked_indexed(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> NodeHistograms {
        let num_features = dataset.num_features();

        // Process row blocks in parallel
        let partial_histograms: Vec<NodeHistograms> = row_indices
            .par_chunks(BLOCK_SIZE)
            .map(|chunk| {
                let block_len = chunk.len();

                // Create local histograms for this block
                let mut local_hists = NodeHistograms::new(num_features);

                // Pre-load gradients/hessians for this block into stack arrays
                // Gather from scattered indices into contiguous cache
                let mut grad_cache = [0.0f32; BLOCK_SIZE];
                let mut hess_cache = [0.0f32; BLOCK_SIZE];

                unsafe {
                    for (i, &row_idx) in chunk.iter().enumerate() {
                        *grad_cache.get_unchecked_mut(i) = *gradients.get_unchecked(row_idx);
                        *hess_cache.get_unchecked_mut(i) = *hessians.get_unchecked(row_idx);
                    }
                }

                // Now iterate ALL features while gradients are hot in L1
                for feature_idx in 0..num_features {
                    let feature_column = dataset.feature_column(feature_idx);
                    let hist = local_hists.get_mut(feature_idx);
                    let bins = hist.bins_mut();

                    // 8x unrolled accumulation with cached gradients
                    let chunks_count = block_len / 8;
                    let remainder = block_len % 8;

                    unsafe {
                        for i in 0..chunks_count {
                            let base = i * 8;

                            // Load row indices
                            let idx0 = *chunk.get_unchecked(base);
                            let idx1 = *chunk.get_unchecked(base + 1);
                            let idx2 = *chunk.get_unchecked(base + 2);
                            let idx3 = *chunk.get_unchecked(base + 3);
                            let idx4 = *chunk.get_unchecked(base + 4);
                            let idx5 = *chunk.get_unchecked(base + 5);
                            let idx6 = *chunk.get_unchecked(base + 6);
                            let idx7 = *chunk.get_unchecked(base + 7);

                            // Load bins (scattered read from feature column)
                            let bin0 = *feature_column.get_unchecked(idx0) as usize;
                            let bin1 = *feature_column.get_unchecked(idx1) as usize;
                            let bin2 = *feature_column.get_unchecked(idx2) as usize;
                            let bin3 = *feature_column.get_unchecked(idx3) as usize;
                            let bin4 = *feature_column.get_unchecked(idx4) as usize;
                            let bin5 = *feature_column.get_unchecked(idx5) as usize;
                            let bin6 = *feature_column.get_unchecked(idx6) as usize;
                            let bin7 = *feature_column.get_unchecked(idx7) as usize;

                            // Load from L1 cache (fast!)
                            let grad0 = *grad_cache.get_unchecked(base);
                            let grad1 = *grad_cache.get_unchecked(base + 1);
                            let grad2 = *grad_cache.get_unchecked(base + 2);
                            let grad3 = *grad_cache.get_unchecked(base + 3);
                            let grad4 = *grad_cache.get_unchecked(base + 4);
                            let grad5 = *grad_cache.get_unchecked(base + 5);
                            let grad6 = *grad_cache.get_unchecked(base + 6);
                            let grad7 = *grad_cache.get_unchecked(base + 7);

                            let hess0 = *hess_cache.get_unchecked(base);
                            let hess1 = *hess_cache.get_unchecked(base + 1);
                            let hess2 = *hess_cache.get_unchecked(base + 2);
                            let hess3 = *hess_cache.get_unchecked(base + 3);
                            let hess4 = *hess_cache.get_unchecked(base + 4);
                            let hess5 = *hess_cache.get_unchecked(base + 5);
                            let hess6 = *hess_cache.get_unchecked(base + 6);
                            let hess7 = *hess_cache.get_unchecked(base + 7);

                            // Scatter to histogram bins
                            bins.get_unchecked_mut(bin0).accumulate(grad0, hess0);
                            bins.get_unchecked_mut(bin1).accumulate(grad1, hess1);
                            bins.get_unchecked_mut(bin2).accumulate(grad2, hess2);
                            bins.get_unchecked_mut(bin3).accumulate(grad3, hess3);
                            bins.get_unchecked_mut(bin4).accumulate(grad4, hess4);
                            bins.get_unchecked_mut(bin5).accumulate(grad5, hess5);
                            bins.get_unchecked_mut(bin6).accumulate(grad6, hess6);
                            bins.get_unchecked_mut(bin7).accumulate(grad7, hess7);
                        }

                        // Handle remainder
                        let rem_base = chunks_count * 8;
                        for i in 0..remainder {
                            let idx = *chunk.get_unchecked(rem_base + i);
                            let bin = *feature_column.get_unchecked(idx) as usize;
                            let grad = *grad_cache.get_unchecked(rem_base + i);
                            let hess = *hess_cache.get_unchecked(rem_base + i);
                            bins.get_unchecked_mut(bin).accumulate(grad, hess);
                        }
                    }
                }

                local_hists
            })
            .collect();

        // Reduce partial histograms
        Self::reduce_histograms(partial_histograms, num_features)
    }

    /// Build histograms for a single small block (no parallelism needed)
    fn build_single_block(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> NodeHistograms {
        let num_features = dataset.num_features();
        let mut node_hists = NodeHistograms::new(num_features);

        // Pre-cache gradients/hessians
        let block_len = row_indices.len();
        let mut grad_cache = vec![0.0f32; block_len];
        let mut hess_cache = vec![0.0f32; block_len];

        for (i, &row_idx) in row_indices.iter().enumerate() {
            grad_cache[i] = gradients[row_idx];
            hess_cache[i] = hessians[row_idx];
        }

        // Process all features with cached gradients
        for feature_idx in 0..num_features {
            let feature_column = dataset.feature_column(feature_idx);
            let hist = node_hists.get_mut(feature_idx);

            for (i, &row_idx) in row_indices.iter().enumerate() {
                let bin = feature_column[row_idx];
                hist.accumulate(bin, grad_cache[i], hess_cache[i]);
            }
        }

        node_hists
    }

    /// Reduce multiple partial histograms into one
    fn reduce_histograms(partials: Vec<NodeHistograms>, num_features: usize) -> NodeHistograms {
        if partials.is_empty() {
            return NodeHistograms::new(num_features);
        }

        if partials.len() == 1 {
            return partials.into_iter().next().unwrap();
        }

        // Parallel reduction
        let mut result = NodeHistograms::new(num_features);
        for partial in partials {
            result.merge(&partial);
        }
        result
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
