//! Fused Gradient+Histogram computation for eliminating cache pollution
//!
//! # The Problem
//!
//! Traditional GBDT training has two separate phases per round:
//! 1. Gradient computation: reads targets/predictions, writes gradients/hessians
//! 2. Histogram building: reads gradients/hessians, reads dataset bins
//!
//! When the dataset (50MB for 500k×100) exceeds L3 cache (32MB typical),
//! the gradient phase evicts dataset from cache. Histogram building then
//! must reload the entire dataset from RAM, causing 80%+ slowdown.
//!
//! # The Solution
//!
//! Fused computation combines both phases in a single pass:
//! ```text
//! For each row block (2048 rows, fits in L1):
//!     For each row:
//!         1. g = loss.gradient(target, prediction)
//!         2. h = loss.hessian(target, prediction)
//!         3. For each feature:
//!             bin = dataset[row, feature]
//!             histogram[feature][bin] += (g, h)
//! ```
//!
//! This achieves:
//! - Single memory pass over data
//! - Perfect cache utilization (all data stays in L1/L2)
//! - Theoretical minimum memory bandwidth
//!
//! # Usage
//!
//! The fused builder is used ONLY for the root node histogram. Child nodes
//! reuse the computed gradients with the regular HistogramBuilder, which
//! applies the histogram subtraction trick.

use crate::dataset::{BinnedDataset, SparseColumn, DEFAULT_BIN};
use crate::histogram::NodeHistograms;
use crate::loss::LossFunction;
use rayon::prelude::*;

/// Block size for cache-blocked processing
/// 2048 rows × (4 bytes gradient + 4 bytes hessian) = 16KB, fits in L1 cache
const BLOCK_SIZE: usize = 2048;

/// Result of fused gradient+histogram computation
pub struct FusedResult {
    /// Root node histograms for all features
    pub histograms: NodeHistograms,
    /// Total gradient sum (for root node weight)
    pub total_gradient: f32,
    /// Total hessian sum (for root node weight)
    pub total_hessian: f32,
}

/// Fused gradient and histogram builder
///
/// Computes gradients AND builds histograms in a single pass over the data,
/// eliminating cache pollution between the two phases.
pub struct FusedHistogramBuilder {
    /// Number of threads for parallel construction
    num_threads: usize,
}

impl Default for FusedHistogramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FusedHistogramBuilder {
    /// Create a new fused builder
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

    /// Build root histograms with fused gradient computation
    ///
    /// This is the main entry point for fused computation. It:
    /// 1. Computes gradients/hessians for all rows
    /// 2. Builds histograms for all features
    /// 3. Does both in a single pass per row block
    ///
    /// # Arguments
    /// * `dataset` - The binned dataset
    /// * `row_indices` - Indices of rows to process (for subsampling)
    /// * `targets` - Target values for all rows
    /// * `predictions` - Current predictions for all rows
    /// * `loss_fn` - Loss function for gradient/hessian computation
    /// * `gradients` - Output buffer for gradients (will be written)
    /// * `hessians` - Output buffer for hessians (will be written)
    ///
    /// # Returns
    /// `FusedResult` containing histograms and gradient sums
    ///
    /// Parameters represent distinct algorithm inputs that cannot be meaningfully grouped.
    #[allow(clippy::too_many_arguments)]
    pub fn build_root(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        targets: &[f32],
        predictions: &[f32],
        loss_fn: &dyn LossFunction,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> FusedResult {
        let num_rows = row_indices.len();

        // For small datasets, use single-threaded path
        if num_rows < BLOCK_SIZE {
            return self.build_single_block(
                dataset,
                row_indices,
                targets,
                predictions,
                loss_fn,
                gradients,
                hessians,
            );
        }

        // Check if rows are contiguous (0..n) - enables optimized path
        if Self::is_contiguous(row_indices) {
            self.build_blocked_contiguous(
                dataset,
                num_rows,
                targets,
                predictions,
                loss_fn,
                gradients,
                hessians,
            )
        } else {
            self.build_blocked_indexed(
                dataset,
                row_indices,
                targets,
                predictions,
                loss_fn,
                gradients,
                hessians,
            )
        }
    }

    /// Fused build for contiguous rows (optimized path)
    #[allow(clippy::too_many_arguments)]
    fn build_blocked_contiguous(
        &self,
        dataset: &BinnedDataset,
        num_rows: usize,
        targets: &[f32],
        predictions: &[f32],
        loss_fn: &dyn LossFunction,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> FusedResult {
        let num_features = dataset.num_features();

        // Collect partial results from parallel blocks
        let partial_results: Vec<(NodeHistograms, f32, f32)> = (0..num_rows)
            .into_par_iter()
            .step_by(BLOCK_SIZE)
            .map(|block_start| {
                let block_end = (block_start + BLOCK_SIZE).min(num_rows);
                let block_len = block_end - block_start;

                // Create local histograms for this block
                let mut local_hists = NodeHistograms::new(num_features);

                // Local gradient/hessian cache (stays in L1)
                let mut grad_cache = [0.0f32; BLOCK_SIZE];
                let mut hess_cache = [0.0f32; BLOCK_SIZE];

                // Phase 1: Compute gradients for this block (vectorizable)
                for i in 0..block_len {
                    let row = block_start + i;
                    let (g, h) = loss_fn.gradient_hessian(targets[row], predictions[row]);
                    grad_cache[i] = g;
                    hess_cache[i] = h;
                }

                // Write gradients back to output buffers
                // SAFETY: Each block writes to non-overlapping regions
                unsafe {
                    let grad_ptr = gradients.as_ptr() as *mut f32;
                    let hess_ptr = hessians.as_ptr() as *mut f32;
                    std::ptr::copy_nonoverlapping(
                        grad_cache.as_ptr(),
                        grad_ptr.add(block_start),
                        block_len,
                    );
                    std::ptr::copy_nonoverlapping(
                        hess_cache.as_ptr(),
                        hess_ptr.add(block_start),
                        block_len,
                    );
                }

                // Compute block sums (needed for sparse default bin calculation)
                let block_grad: f32 = grad_cache[..block_len].iter().sum();
                let block_hess: f32 = hess_cache[..block_len].iter().sum();

                // Phase 2: Build histograms while gradients are hot in L1
                for feature_idx in 0..num_features {
                    // Check for sparse feature
                    if let Some(sparse_col) = dataset.sparse_column(feature_idx) {
                        // Sparse path: only iterate non-default entries
                        Self::build_sparse_histogram_block(
                            local_hists.get_mut(feature_idx),
                            sparse_col,
                            block_start,
                            block_len,
                            &grad_cache,
                            &hess_cache,
                            block_grad,
                            block_hess,
                        );
                    } else {
                        // Dense path: iterate all rows
                        let feature_column = dataset.feature_column(feature_idx);
                        let hist = local_hists.get_mut(feature_idx);
                        let bins = hist.bins_mut();

                        // 8x unrolled for ILP
                        let chunks = block_len / 8;
                        let remainder = block_len % 8;

                        unsafe {
                            for i in 0..chunks {
                                let base = i * 8;
                                let row_base = block_start + base;

                                // Load bins (sequential from feature column)
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
                                let bin =
                                    *feature_column.get_unchecked(block_start + rem_base + i) as usize;
                                let grad = *grad_cache.get_unchecked(rem_base + i);
                                let hess = *hess_cache.get_unchecked(rem_base + i);
                                bins.get_unchecked_mut(bin).accumulate(grad, hess);
                            }
                        }
                    }
                }

                (local_hists, block_grad, block_hess)
            })
            .collect();

        // Reduce partial results
        self.reduce_results(partial_results, num_features)
    }

    /// Fused build for indexed (non-contiguous) rows
    #[allow(clippy::too_many_arguments)]
    fn build_blocked_indexed(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        targets: &[f32],
        predictions: &[f32],
        loss_fn: &dyn LossFunction,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> FusedResult {
        let num_features = dataset.num_features();

        // Collect partial results from parallel blocks
        let partial_results: Vec<(NodeHistograms, f32, f32)> = row_indices
            .par_chunks(BLOCK_SIZE)
            .map(|chunk| {
                let block_len = chunk.len();

                // Create local histograms for this block
                let mut local_hists = NodeHistograms::new(num_features);

                // Local gradient/hessian cache (stays in L1)
                let mut grad_cache = [0.0f32; BLOCK_SIZE];
                let mut hess_cache = [0.0f32; BLOCK_SIZE];

                // Phase 1: Compute gradients for this block
                for (i, &row_idx) in chunk.iter().enumerate() {
                    let (g, h) = loss_fn.gradient_hessian(targets[row_idx], predictions[row_idx]);
                    grad_cache[i] = g;
                    hess_cache[i] = h;

                    // Write to output buffers
                    // SAFETY: row_idx values are unique across all chunks
                    unsafe {
                        let grad_ptr = gradients.as_ptr() as *mut f32;
                        let hess_ptr = hessians.as_ptr() as *mut f32;
                        *grad_ptr.add(row_idx) = g;
                        *hess_ptr.add(row_idx) = h;
                    }
                }

                // Compute block sums (needed for sparse default bin calculation)
                let block_grad: f32 = grad_cache[..block_len].iter().sum();
                let block_hess: f32 = hess_cache[..block_len].iter().sum();

                // Phase 2: Build histograms while gradients are hot in L1
                for feature_idx in 0..num_features {
                    // Check for sparse feature
                    if let Some(sparse_col) = dataset.sparse_column(feature_idx) {
                        // Sparse path: only iterate non-default entries
                        Self::build_sparse_histogram_indexed(
                            local_hists.get_mut(feature_idx),
                            sparse_col,
                            chunk,
                            &grad_cache,
                            &hess_cache,
                            block_len,
                            block_grad,
                            block_hess,
                        );
                    } else {
                        // Dense path: iterate all rows
                        let feature_column = dataset.feature_column(feature_idx);
                        let hist = local_hists.get_mut(feature_idx);
                        let bins = hist.bins_mut();

                        // 8x unrolled for ILP
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

                                // Load bins (scattered from feature column)
                                let bin0 = *feature_column.get_unchecked(idx0) as usize;
                                let bin1 = *feature_column.get_unchecked(idx1) as usize;
                                let bin2 = *feature_column.get_unchecked(idx2) as usize;
                                let bin3 = *feature_column.get_unchecked(idx3) as usize;
                                let bin4 = *feature_column.get_unchecked(idx4) as usize;
                                let bin5 = *feature_column.get_unchecked(idx5) as usize;
                                let bin6 = *feature_column.get_unchecked(idx6) as usize;
                                let bin7 = *feature_column.get_unchecked(idx7) as usize;

                                // Load from L1 cache
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
                }

                (local_hists, block_grad, block_hess)
            })
            .collect();

        // Reduce partial results
        self.reduce_results(partial_results, num_features)
    }

    /// Single block build for small datasets
    #[allow(clippy::too_many_arguments)]
    fn build_single_block(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        targets: &[f32],
        predictions: &[f32],
        loss_fn: &dyn LossFunction,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> FusedResult {
        let num_features = dataset.num_features();
        let num_rows = row_indices.len();

        let mut histograms = NodeHistograms::new(num_features);
        let mut total_gradient = 0.0f32;
        let mut total_hessian = 0.0f32;

        // Local cache
        let mut grad_cache = vec![0.0f32; num_rows];
        let mut hess_cache = vec![0.0f32; num_rows];

        // Phase 1: Compute gradients
        for (i, &row_idx) in row_indices.iter().enumerate() {
            let (g, h) = loss_fn.gradient_hessian(targets[row_idx], predictions[row_idx]);
            grad_cache[i] = g;
            hess_cache[i] = h;
            gradients[row_idx] = g;
            hessians[row_idx] = h;
            total_gradient += g;
            total_hessian += h;
        }

        // Phase 2: Build histograms
        for feature_idx in 0..num_features {
            let feature_column = dataset.feature_column(feature_idx);
            let hist = histograms.get_mut(feature_idx);

            for (i, &row_idx) in row_indices.iter().enumerate() {
                let bin = feature_column[row_idx];
                hist.accumulate(bin, grad_cache[i], hess_cache[i]);
            }
        }

        FusedResult {
            histograms,
            total_gradient,
            total_hessian,
        }
    }

    /// Reduce partial results into final result
    fn reduce_results(
        &self,
        partials: Vec<(NodeHistograms, f32, f32)>,
        num_features: usize,
    ) -> FusedResult {
        if partials.is_empty() {
            return FusedResult {
                histograms: NodeHistograms::new(num_features),
                total_gradient: 0.0,
                total_hessian: 0.0,
            };
        }

        if partials.len() == 1 {
            let (hists, grad, hess) = partials.into_iter().next().unwrap();
            return FusedResult {
                histograms: hists,
                total_gradient: grad,
                total_hessian: hess,
            };
        }

        // Parallel reduction of histograms
        let mut result_hists = NodeHistograms::new(num_features);
        let mut total_grad = 0.0f32;
        let mut total_hess = 0.0f32;

        for (partial_hists, grad, hess) in partials {
            result_hists.merge(&partial_hists);
            total_grad += grad;
            total_hess += hess;
        }

        FusedResult {
            histograms: result_hists,
            total_gradient: total_grad,
            total_hessian: total_hess,
        }
    }

    /// Check if row_indices represents contiguous range 0..n
    #[inline]
    fn is_contiguous(row_indices: &[usize]) -> bool {
        if row_indices.is_empty() {
            return true;
        }
        row_indices[0] == 0 && row_indices.last() == Some(&(row_indices.len() - 1))
    }

    /// Build histogram for a sparse feature block (contiguous rows)
    ///
    /// Only iterates non-default entries in the block, then computes
    /// the default bin by subtraction: default = total - sum(non_default)
    #[allow(clippy::too_many_arguments)]
    fn build_sparse_histogram_block(
        hist: &mut crate::histogram::Histogram,
        sparse_col: &SparseColumn,
        block_start: usize,
        block_len: usize,
        grad_cache: &[f32; BLOCK_SIZE],
        hess_cache: &[f32; BLOCK_SIZE],
        block_total_grad: f32,
        block_total_hess: f32,
    ) {
        let block_end = block_start + block_len;
        let bins = hist.bins_mut();

        let mut non_default_grad = 0.0f32;
        let mut non_default_hess = 0.0f32;
        let mut non_default_count = 0u32;

        // Binary search to find first index >= block_start
        let start_pos = sparse_col
            .indices
            .partition_point(|&idx| (idx as usize) < block_start);

        // Iterate only the non-default entries within this block
        for i in start_pos..sparse_col.indices.len() {
            let row_idx = sparse_col.indices[i] as usize;
            if row_idx >= block_end {
                break;
            }

            let bin = sparse_col.values[i];
            let cache_idx = row_idx - block_start;

            unsafe {
                let grad = *grad_cache.get_unchecked(cache_idx);
                let hess = *hess_cache.get_unchecked(cache_idx);

                bins.get_unchecked_mut(bin as usize).accumulate(grad, hess);
                non_default_grad += grad;
                non_default_hess += hess;
                non_default_count += 1;
            }
        }

        // Compute default bin by subtraction
        let default_grad = block_total_grad - non_default_grad;
        let default_hess = block_total_hess - non_default_hess;
        let default_count = block_len as u32 - non_default_count;

        if default_count > 0 {
            unsafe {
                bins.get_unchecked_mut(DEFAULT_BIN as usize)
                    .accumulate_with_count(default_grad, default_hess, default_count);
            }
        }
    }

    /// Build histogram for a sparse feature with indexed rows
    #[allow(clippy::too_many_arguments)]
    fn build_sparse_histogram_indexed(
        hist: &mut crate::histogram::Histogram,
        sparse_col: &SparseColumn,
        chunk: &[usize],
        grad_cache: &[f32; BLOCK_SIZE],
        hess_cache: &[f32; BLOCK_SIZE],
        block_len: usize,
        block_total_grad: f32,
        block_total_hess: f32,
    ) {
        let bins = hist.bins_mut();

        let mut non_default_grad = 0.0f32;
        let mut non_default_hess = 0.0f32;
        let mut non_default_count = 0u32;

        // For indexed rows, we need to check each row against sparse indices
        let chunk_min = chunk.first().copied().unwrap_or(0);
        let chunk_max = chunk.last().copied().unwrap_or(0);

        // Binary search to find the range of sparse entries that could overlap
        let sparse_start = sparse_col
            .indices
            .partition_point(|&idx| (idx as usize) < chunk_min);
        let sparse_end = sparse_col
            .indices
            .partition_point(|&idx| (idx as usize) <= chunk_max);

        let sparse_range = sparse_end - sparse_start;

        // Choose strategy based on which is smaller to iterate
        if sparse_range <= chunk.len() / 2 {
            // Sparse-first: iterate sparse entries in range, binary search in chunk
            for i in sparse_start..sparse_end {
                let row_idx = sparse_col.indices[i] as usize;
                let bin = sparse_col.values[i];
                // Binary search in chunk to find cache index
                if let Ok(cache_idx) = chunk.binary_search(&row_idx) {
                    unsafe {
                        let grad = *grad_cache.get_unchecked(cache_idx);
                        let hess = *hess_cache.get_unchecked(cache_idx);

                        bins.get_unchecked_mut(bin as usize).accumulate(grad, hess);
                        non_default_grad += grad;
                        non_default_hess += hess;
                        non_default_count += 1;
                    }
                }
            }
        } else {
            // Chunk-first: iterate chunk, binary search in sparse indices
            for (cache_idx, &row_idx) in chunk.iter().enumerate() {
                if let Ok(sparse_pos) = sparse_col.indices.binary_search(&(row_idx as u32)) {
                    let bin = sparse_col.values[sparse_pos];
                    unsafe {
                        let grad = *grad_cache.get_unchecked(cache_idx);
                        let hess = *hess_cache.get_unchecked(cache_idx);

                        bins.get_unchecked_mut(bin as usize).accumulate(grad, hess);
                        non_default_grad += grad;
                        non_default_hess += hess;
                        non_default_count += 1;
                    }
                }
            }
        }

        // Compute default bin by subtraction
        let default_grad = block_total_grad - non_default_grad;
        let default_hess = block_total_hess - non_default_hess;
        let default_count = block_len as u32 - non_default_count;

        if default_count > 0 {
            unsafe {
                bins.get_unchecked_mut(DEFAULT_BIN as usize)
                    .accumulate_with_count(default_grad, default_hess, default_count);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeatureInfo, FeatureType};
    use crate::loss::MseLoss;

    fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * (f + 1) * 17) % 256) as u8);
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32 * 0.01).sin()).collect();
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
    fn test_fused_basic() {
        let num_rows = 1000;
        let num_features = 10;
        let dataset = create_test_dataset(num_rows, num_features);
        let targets = dataset.targets().to_vec();
        let predictions = vec![0.0f32; num_rows];
        let mut gradients = vec![0.0f32; num_rows];
        let mut hessians = vec![0.0f32; num_rows];
        let row_indices: Vec<usize> = (0..num_rows).collect();

        let loss_fn = MseLoss::new();
        let builder = FusedHistogramBuilder::new();

        let result = builder.build_root(
            &dataset,
            &row_indices,
            &targets,
            &predictions,
            &loss_fn,
            &mut gradients,
            &mut hessians,
        );

        // Verify histograms
        assert_eq!(result.histograms.num_features(), num_features);

        // Verify total count
        let total_count: u32 = result.histograms.get(0).bins().iter().map(|b| b.count).sum();
        assert_eq!(total_count, num_rows as u32);

        // Verify gradients were computed
        assert!(gradients.iter().any(|&g| g != 0.0));
        assert!(hessians.iter().all(|&h| h == 1.0)); // MSE hessian is constant 1.0
    }

    #[test]
    fn test_fused_matches_separate() {
        use crate::histogram::HistogramBuilder;

        let num_rows = 5000;
        let num_features = 20;
        let dataset = create_test_dataset(num_rows, num_features);
        let targets = dataset.targets().to_vec();
        let predictions = vec![0.5f32; num_rows];
        let row_indices: Vec<usize> = (0..num_rows).collect();

        let loss_fn = MseLoss::new();

        // Method 1: Fused
        let mut fused_grads = vec![0.0f32; num_rows];
        let mut fused_hess = vec![0.0f32; num_rows];
        let fused_builder = FusedHistogramBuilder::new();
        let fused_result = fused_builder.build_root(
            &dataset,
            &row_indices,
            &targets,
            &predictions,
            &loss_fn,
            &mut fused_grads,
            &mut fused_hess,
        );

        // Method 2: Separate (traditional)
        let mut sep_grads = vec![0.0f32; num_rows];
        let mut sep_hess = vec![0.0f32; num_rows];
        for &idx in &row_indices {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            sep_grads[idx] = g;
            sep_hess[idx] = h;
        }
        let hist_builder = HistogramBuilder::new();
        let sep_hists = hist_builder.build(&dataset, &row_indices, &sep_grads, &sep_hess);

        // Compare gradients
        for i in 0..num_rows {
            assert!(
                (fused_grads[i] - sep_grads[i]).abs() < 1e-6,
                "Gradient mismatch at row {}",
                i
            );
            assert!(
                (fused_hess[i] - sep_hess[i]).abs() < 1e-6,
                "Hessian mismatch at row {}",
                i
            );
        }

        // Compare histograms
        for f in 0..num_features {
            for bin in 0..=255u8 {
                let fused_entry = fused_result.histograms.get(f).get(bin);
                let sep_entry = sep_hists.get(f).get(bin);

                assert!(
                    (fused_entry.sum_gradients - sep_entry.sum_gradients).abs() < 1e-4,
                    "Gradient sum mismatch at feature {} bin {}",
                    f,
                    bin
                );
                assert!(
                    (fused_entry.sum_hessians - sep_entry.sum_hessians).abs() < 1e-4,
                    "Hessian sum mismatch at feature {} bin {}",
                    f,
                    bin
                );
                assert_eq!(
                    fused_entry.count, sep_entry.count,
                    "Count mismatch at feature {} bin {}",
                    f,
                    bin
                );
            }
        }
    }

    #[test]
    fn test_fused_indexed_rows() {
        let num_rows = 1000;
        let num_features = 5;
        let dataset = create_test_dataset(num_rows, num_features);
        let targets = dataset.targets().to_vec();
        let predictions = vec![0.0f32; num_rows];
        let mut gradients = vec![0.0f32; num_rows];
        let mut hessians = vec![0.0f32; num_rows];

        // Use only even rows
        let row_indices: Vec<usize> = (0..num_rows).filter(|i| i % 2 == 0).collect();

        let loss_fn = MseLoss::new();
        let builder = FusedHistogramBuilder::new();

        let result = builder.build_root(
            &dataset,
            &row_indices,
            &targets,
            &predictions,
            &loss_fn,
            &mut gradients,
            &mut hessians,
        );

        // Verify total count matches number of indexed rows
        let total_count: u32 = result.histograms.get(0).bins().iter().map(|b| b.count).sum();
        assert_eq!(total_count, row_indices.len() as u32);
    }
}
