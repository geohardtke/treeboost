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

use crate::backend::scalar::kernel;
use crate::dataset::{BinnedDataset, SparseColumn, DEFAULT_BIN};
use crate::histogram::{Histogram, NodeHistograms};
use rayon::prelude::*;

/// Block size for cache-blocked histogram building
/// 2048 rows * 8 bytes (gradient + hessian) = 16KB, fits in L1 cache
const BLOCK_SIZE: usize = 2048;

/// Histogram builder with feature-parallel construction
///
/// Uses Rayon's work-stealing thread pool for parallelism.
/// Thread count is controlled globally via `rayon::ThreadPoolBuilder` or
/// the `RAYON_NUM_THREADS` environment variable.
#[derive(Debug, Clone, Copy)]
pub struct HistogramBuilder;

impl Default for HistogramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HistogramBuilder {
    /// Create a new histogram builder
    pub fn new() -> Self {
        Self
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

                // INTERLEAVED LAYOUT: Pack grad/hess together for better cache locality
                // Single cache line load gets both values (~17.8% faster than separate arrays)
                // Uses SIMD (AVX2/NEON) for fast interleaving when available
                let mut gh_cache = [(0.0f32, 0.0f32); BLOCK_SIZE];

                // Copy block data to interleaved cache using SIMD-optimized path
                unsafe {
                    kernel::copy_gh_interleaved(
                        gradients,
                        hessians,
                        block_start,
                        block_len,
                        &mut gh_cache,
                    );
                }

                // Compute block totals for sparse default bin subtraction
                let (block_total_grad, block_total_hess) = gh_cache[..block_len]
                    .iter()
                    .fold((0.0f32, 0.0f32), |(g_acc, h_acc), &(g, h)| {
                        (g_acc + g, h_acc + h)
                    });

                // Now iterate ALL features while gradients are hot in L1
                for feature_idx in 0..num_features {
                    // Check for sparse feature
                    if let Some(sparse_col) = dataset.sparse_column(feature_idx) {
                        // Sparse path: only iterate non-default entries in this block
                        Self::build_sparse_histogram_block_interleaved(
                            local_hists.get_mut(feature_idx),
                            sparse_col,
                            block_start,
                            block_len,
                            &gh_cache,
                            block_total_grad,
                            block_total_hess,
                        );
                    } else {
                        // Dense path: 8x unrolled accumulation
                        let feature_column = dataset.feature_column(feature_idx);
                        let hist = local_hists.get_mut(feature_idx);
                        let bins = hist.bins_mut();

                        let chunks = block_len / 8;
                        let remainder = block_len % 8;

                        unsafe {
                            // Prefetch distance (in rows ahead)
                            const PF_DIST: usize = 16;

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

                                // Prefetch future histogram bins (hide memory latency)
                                #[cfg(target_arch = "x86_64")]
                                if base + PF_DIST < block_len {
                                    use std::arch::x86_64::*;
                                    let pf_base = block_start + base + PF_DIST;
                                    let pf_bin0 = *feature_column.get_unchecked(pf_base) as usize;
                                    let pf_bin1 =
                                        *feature_column.get_unchecked(pf_base + 1) as usize;
                                    _mm_prefetch(
                                        bins.as_ptr().add(pf_bin0) as *const i8,
                                        _MM_HINT_T0,
                                    );
                                    _mm_prefetch(
                                        bins.as_ptr().add(pf_bin1) as *const i8,
                                        _MM_HINT_T0,
                                    );
                                }

                                // Load from L1 cache (fast!) - interleaved layout
                                // Single cache line gets both grad and hess
                                let (grad0, hess0) = *gh_cache.get_unchecked(base);
                                let (grad1, hess1) = *gh_cache.get_unchecked(base + 1);
                                let (grad2, hess2) = *gh_cache.get_unchecked(base + 2);
                                let (grad3, hess3) = *gh_cache.get_unchecked(base + 3);
                                let (grad4, hess4) = *gh_cache.get_unchecked(base + 4);
                                let (grad5, hess5) = *gh_cache.get_unchecked(base + 5);
                                let (grad6, hess6) = *gh_cache.get_unchecked(base + 6);
                                let (grad7, hess7) = *gh_cache.get_unchecked(base + 7);

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
                                let bin = *feature_column.get_unchecked(block_start + rem_base + i)
                                    as usize;
                                let (grad, hess) = *gh_cache.get_unchecked(rem_base + i);
                                bins.get_unchecked_mut(bin).accumulate(grad, hess);
                            }
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

                // INTERLEAVED LAYOUT: Pack grad/hess together for better cache locality
                // Uses scalar for indexed access (gather has high latency, no SIMD benefit)
                let mut gh_cache = [(0.0f32, 0.0f32); BLOCK_SIZE];

                kernel::copy_gh_indexed(gradients, hessians, chunk, &mut gh_cache);

                // Compute block totals for sparse default bin subtraction
                let (block_total_grad, block_total_hess) = gh_cache[..block_len]
                    .iter()
                    .fold((0.0f32, 0.0f32), |(g_acc, h_acc), &(g, h)| {
                        (g_acc + g, h_acc + h)
                    });

                // Now iterate ALL features while gradients are hot in L1
                for feature_idx in 0..num_features {
                    // Check for sparse feature - for indexed rows, use the simpler path
                    // since we need to intersect sparse indices with chunk indices
                    if let Some(sparse_col) = dataset.sparse_column(feature_idx) {
                        Self::build_sparse_histogram_indexed_interleaved(
                            local_hists.get_mut(feature_idx),
                            sparse_col,
                            chunk,
                            &gh_cache,
                            block_len,
                            block_total_grad,
                            block_total_hess,
                        );
                    } else {
                        // Dense path: 8x unrolled accumulation
                        let feature_column = dataset.feature_column(feature_idx);
                        let hist = local_hists.get_mut(feature_idx);
                        let bins = hist.bins_mut();

                        let chunks_count = block_len / 8;
                        let remainder = block_len % 8;

                        unsafe {
                            // Prefetch distance (in rows ahead)
                            const PF_DIST: usize = 16;

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

                                // Prefetch future histogram bins (hide memory latency)
                                #[cfg(target_arch = "x86_64")]
                                if base + PF_DIST < block_len {
                                    use std::arch::x86_64::*;
                                    let pf_idx0 = *chunk.get_unchecked(base + PF_DIST);
                                    let pf_idx1 = *chunk.get_unchecked(base + PF_DIST + 1);
                                    let pf_bin0 = *feature_column.get_unchecked(pf_idx0) as usize;
                                    let pf_bin1 = *feature_column.get_unchecked(pf_idx1) as usize;
                                    _mm_prefetch(
                                        bins.as_ptr().add(pf_bin0) as *const i8,
                                        _MM_HINT_T0,
                                    );
                                    _mm_prefetch(
                                        bins.as_ptr().add(pf_bin1) as *const i8,
                                        _MM_HINT_T0,
                                    );
                                }

                                // Load from L1 cache (fast!) - interleaved layout
                                let (grad0, hess0) = *gh_cache.get_unchecked(base);
                                let (grad1, hess1) = *gh_cache.get_unchecked(base + 1);
                                let (grad2, hess2) = *gh_cache.get_unchecked(base + 2);
                                let (grad3, hess3) = *gh_cache.get_unchecked(base + 3);
                                let (grad4, hess4) = *gh_cache.get_unchecked(base + 4);
                                let (grad5, hess5) = *gh_cache.get_unchecked(base + 5);
                                let (grad6, hess6) = *gh_cache.get_unchecked(base + 6);
                                let (grad7, hess7) = *gh_cache.get_unchecked(base + 7);

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
                                let (grad, hess) = *gh_cache.get_unchecked(rem_base + i);
                                bins.get_unchecked_mut(bin).accumulate(grad, hess);
                            }
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

        // Compute totals for sparse default bin subtraction
        let total_grad: f32 = grad_cache.iter().sum();
        let total_hess: f32 = hess_cache.iter().sum();
        let total_count = block_len as u32;

        // Process all features with cached gradients
        for feature_idx in 0..num_features {
            // Check if this feature has a sparse representation
            if let Some(sparse_col) = dataset.sparse_column(feature_idx) {
                // Sparse path: only iterate non-default entries
                Self::build_sparse_histogram(
                    node_hists.get_mut(feature_idx),
                    sparse_col,
                    row_indices,
                    gradients,
                    hessians,
                    total_grad,
                    total_hess,
                    total_count,
                );
            } else {
                // Dense path
                let feature_column = dataset.feature_column(feature_idx);
                let hist = node_hists.get_mut(feature_idx);

                for (i, &row_idx) in row_indices.iter().enumerate() {
                    let bin = feature_column[row_idx];
                    hist.accumulate(bin, grad_cache[i], hess_cache[i]);
                }
            }
        }

        node_hists
    }

    /// Build histogram for a sparse feature using default bin subtraction
    ///
    /// Only iterates non-default entries, then computes default bin by:
    /// `default_bin = total - sum(non_default_bins)`
    #[allow(clippy::too_many_arguments)]
    fn build_sparse_histogram(
        hist: &mut Histogram,
        sparse_col: &SparseColumn,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
        total_grad: f32,
        total_hess: f32,
        total_count: u32,
    ) {
        // Create a set of active row indices for O(1) lookup
        // For small row_indices, linear search is faster than HashSet
        let is_contiguous =
            row_indices.first() == Some(&0) && row_indices.last() == Some(&(row_indices.len() - 1));

        let mut non_default_grad = 0.0f32;
        let mut non_default_hess = 0.0f32;
        let mut non_default_count = 0u32;

        if is_contiguous {
            // Fast path: row_indices is 0..n, just check bounds
            let n = row_indices.len();
            for (&row_idx, &bin) in sparse_col.indices.iter().zip(sparse_col.values.iter()) {
                let row_idx = row_idx as usize;
                if row_idx < n {
                    let grad = gradients[row_idx];
                    let hess = hessians[row_idx];
                    hist.accumulate(bin, grad, hess);
                    non_default_grad += grad;
                    non_default_hess += hess;
                    non_default_count += 1;
                }
            }
        } else {
            // Slower path: need to check membership in row_indices
            // For efficiency, we use binary search since row_indices are sorted
            for (&row_idx, &bin) in sparse_col.indices.iter().zip(sparse_col.values.iter()) {
                let row_idx = row_idx as usize;
                if row_indices.binary_search(&row_idx).is_ok() {
                    let grad = gradients[row_idx];
                    let hess = hessians[row_idx];
                    hist.accumulate(bin, grad, hess);
                    non_default_grad += grad;
                    non_default_hess += hess;
                    non_default_count += 1;
                }
            }
        }

        // Compute default bin (bin 0) by subtraction
        let default_bin = hist.get_mut(DEFAULT_BIN);
        default_bin.sum_gradients = total_grad - non_default_grad;
        default_bin.sum_hessians = total_hess - non_default_hess;
        default_bin.count = total_count - non_default_count;
    }

    /// Build histogram for a sparse feature block with interleaved grad/hess cache
    fn build_sparse_histogram_block_interleaved(
        hist: &mut Histogram,
        sparse_col: &SparseColumn,
        block_start: usize,
        block_len: usize,
        gh_cache: &[(f32, f32); BLOCK_SIZE],
        block_total_grad: f32,
        block_total_hess: f32,
    ) {
        let block_end = block_start + block_len;

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
                let (grad, hess) = *gh_cache.get_unchecked(cache_idx);
                hist.accumulate(bin, grad, hess);
                non_default_grad += grad;
                non_default_hess += hess;
                non_default_count += 1;
            }
        }

        // Compute default bin (bin 0) by subtraction
        let default_bin = hist.get_mut(DEFAULT_BIN);
        default_bin.sum_gradients += block_total_grad - non_default_grad;
        default_bin.sum_hessians += block_total_hess - non_default_hess;
        default_bin.count += block_len as u32 - non_default_count;
    }

    /// Build histogram for sparse feature with indexed rows using interleaved grad/hess cache
    fn build_sparse_histogram_indexed_interleaved(
        hist: &mut Histogram,
        sparse_col: &SparseColumn,
        chunk: &[usize],
        gh_cache: &[(f32, f32); BLOCK_SIZE],
        block_len: usize,
        block_total_grad: f32,
        block_total_hess: f32,
    ) {
        let mut non_default_grad = 0.0f32;
        let mut non_default_hess = 0.0f32;
        let mut non_default_count = 0u32;

        // Find the range of chunk indices for quick rejection
        let chunk_min = chunk.first().copied().unwrap_or(0);
        let chunk_max = chunk.last().copied().unwrap_or(0);

        // Binary search to find the range of sparse entries that could overlap with chunk
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
                        let (grad, hess) = *gh_cache.get_unchecked(cache_idx);
                        hist.accumulate(bin, grad, hess);
                        non_default_grad += grad;
                        non_default_hess += hess;
                        non_default_count += 1;
                    }
                }
            }
        } else {
            // Chunk-first: iterate chunk, binary search in sparse indices
            for (cache_idx, &row_idx) in chunk.iter().enumerate() {
                // Binary search to see if this row has a non-default value
                if let Ok(sparse_pos) = sparse_col.indices.binary_search(&(row_idx as u32)) {
                    let bin = sparse_col.values[sparse_pos];
                    unsafe {
                        let (grad, hess) = *gh_cache.get_unchecked(cache_idx);
                        hist.accumulate(bin, grad, hess);
                        non_default_grad += grad;
                        non_default_hess += hess;
                        non_default_count += 1;
                    }
                }
                // If not found, this row has the default bin value (0)
            }
        }

        // Compute default bin (bin 0) by subtraction
        let default_bin = hist.get_mut(DEFAULT_BIN);
        default_bin.sum_gradients += block_total_grad - non_default_grad;
        default_bin.sum_hessians += block_total_hess - non_default_hess;
        default_bin.count += block_len as u32 - non_default_count;
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

    /// Build histogram for a single feature from raw column data
    ///
    /// This is a lower-level API for use by the backend abstraction.
    ///
    /// # Arguments
    /// * `feature_column` - Bin values for each row
    /// * `row_indices` - Which rows to process
    /// * `gradients` - Gradient values (full dataset)
    /// * `hessians` - Hessian values (full dataset)
    /// * `sparse_column` - Optional sparse representation of the feature
    pub fn build_single_feature(
        &self,
        feature_column: &[u8],
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
        sparse_column: Option<&SparseColumn>,
    ) -> Histogram {
        let mut hist = Histogram::new();

        if let Some(sparse_col) = sparse_column {
            // Compute totals for sparse default bin subtraction
            let total_grad: f32 = row_indices.iter().map(|&i| gradients[i]).sum();
            let total_hess: f32 = row_indices.iter().map(|&i| hessians[i]).sum();
            let total_count = row_indices.len() as u32;

            Self::build_sparse_histogram(
                &mut hist,
                sparse_col,
                row_indices,
                gradients,
                hessians,
                total_grad,
                total_hess,
                total_count,
            );
        } else {
            // Dense path
            for &row_idx in row_indices {
                let bin = feature_column[row_idx];
                hist.accumulate(bin, gradients[row_idx], hessians[row_idx]);
            }
        }

        hist
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
