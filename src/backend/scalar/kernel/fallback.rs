//! Scalar fallback implementations for GBDT operations
//!
//! These implementations work on any architecture without SIMD support.
//! They serve as the baseline and are used when SIMD is not available.

// ============================================================================
// Split Finding
// ============================================================================

/// Result of split finding for a single feature
#[derive(Debug, Clone, Copy)]
pub struct SplitCandidate {
    /// Bin threshold (samples with bin <= threshold go left)
    pub bin_threshold: u8,
    /// Split gain (higher is better)
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

impl Default for SplitCandidate {
    fn default() -> Self {
        Self {
            bin_threshold: 0,
            gain: f32::NEG_INFINITY,
            left_gradient: 0.0,
            left_hessian: 0.0,
            left_count: 0,
            right_gradient: 0.0,
            right_hessian: 0.0,
            right_count: 0,
        }
    }
}

impl SplitCandidate {
    /// Check if this split is valid
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.gain > f32::NEG_INFINITY && self.left_count > 0 && self.right_count > 0
    }
}

/// Parameters for split finding
#[derive(Clone, Copy)]
pub struct SplitParams {
    pub total_gradient: f32,
    pub total_hessian: f32,
    pub total_count: u32,
    pub lambda: f32,
    pub min_samples_leaf: u32,
    pub min_hessian_leaf: f32,
}

/// Find the best split for a histogram using scalar code
#[inline]
pub fn find_best_split_scalar(
    hist_grads: &[f32; 256],
    hist_hess: &[f32; 256],
    hist_counts: &[u32; 256],
    params: SplitParams,
) -> Option<SplitCandidate> {
    let SplitParams {
        total_gradient,
        total_hessian,
        total_count,
        lambda,
        min_samples_leaf,
        min_hessian_leaf,
    } = params;
    let mut best = SplitCandidate::default();

    // Parent score (constant, compute once)
    let parent_score = (total_gradient * total_gradient) / (total_hessian + lambda);

    // Cumulative sums for left child
    let mut left_g = 0.0f32;
    let mut left_h = 0.0f32;
    let mut left_n = 0u32;

    // Scan through bins 0..254 (last bin can't be a threshold since right would be empty)
    for bin in 0..255u8 {
        let bin_idx = bin as usize;

        // Skip empty bins
        if hist_counts[bin_idx] == 0 {
            continue;
        }

        // Accumulate left child stats
        left_g += hist_grads[bin_idx];
        left_h += hist_hess[bin_idx];
        left_n += hist_counts[bin_idx];

        // Compute right child stats
        let right_g = total_gradient - left_g;
        let right_h = total_hessian - left_h;
        let right_n = total_count - left_n;

        // Check leaf constraints
        if left_n < min_samples_leaf || right_n < min_samples_leaf {
            continue;
        }
        if left_h < min_hessian_leaf || right_h < min_hessian_leaf {
            continue;
        }

        // Compute gain: 0.5 * (left_score + right_score - parent_score)
        let left_score = (left_g * left_g) / (left_h + lambda);
        let right_score = (right_g * right_g) / (right_h + lambda);
        let gain = 0.5 * (left_score + right_score - parent_score);

        if gain > best.gain {
            best.bin_threshold = bin;
            best.gain = gain;
            best.left_gradient = left_g;
            best.left_hessian = left_h;
            best.left_count = left_n;
            best.right_gradient = right_g;
            best.right_hessian = right_h;
            best.right_count = right_n;
        }
    }

    if best.is_valid() {
        Some(best)
    } else {
        None
    }
}

/// Fallback SIMD implementation - just calls scalar
#[inline]
pub fn find_best_split_simd(
    hist_grads: &[f32; 256],
    hist_hess: &[f32; 256],
    hist_counts: &[u32; 256],
    params: SplitParams,
) -> Option<SplitCandidate> {
    find_best_split_scalar(hist_grads, hist_hess, hist_counts, params)
}

// ============================================================================
// 4-bit Unpacking
// ============================================================================

/// Scalar 4-bit unpacking
#[inline]
pub fn unpack_4bit_scalar(packed: &[u8], output: &mut [u8]) {
    debug_assert!(output.len() >= packed.len() * 2);

    for (i, &byte) in packed.iter().enumerate() {
        output[i * 2] = byte >> 4;
        output[i * 2 + 1] = byte & 0x0F;
    }
}

/// Fallback unpacking (just calls scalar)
#[inline]
pub fn unpack_4bit(packed: &[u8], output: &mut [u8]) {
    unpack_4bit_scalar(packed, output);
}

// ============================================================================
// Histogram Accumulation
// ============================================================================

/// Parameters for histogram accumulation
pub struct HistogramAccumParams {
    pub feature_bins: *const u8,
    pub row_indices: *const usize,
    pub num_rows: usize,
    pub gradients: *const f32,
    pub hessians: *const f32,
    pub hist_grads: *mut f32,
    pub hist_hess: *mut f32,
    pub hist_counts: *mut u32,
}

/// Scalar histogram accumulation with indexed rows
///
/// # Safety
/// All pointers must be valid and properly sized.
#[inline]
pub unsafe fn histogram_accumulate_scalar(params: HistogramAccumParams) {
    let HistogramAccumParams {
        feature_bins,
        row_indices,
        num_rows,
        gradients,
        hessians,
        hist_grads,
        hist_hess,
        hist_counts,
    } = params;
    // 8x unrolled for better ILP (matches our current best implementation)
    let chunks = num_rows / 8;
    let remainder = num_rows % 8;

    // Process 8 rows at a time
    for i in 0..chunks {
        let base = i * 8;

        // Load 8 row indices
        let idx0 = *row_indices.add(base);
        let idx1 = *row_indices.add(base + 1);
        let idx2 = *row_indices.add(base + 2);
        let idx3 = *row_indices.add(base + 3);
        let idx4 = *row_indices.add(base + 4);
        let idx5 = *row_indices.add(base + 5);
        let idx6 = *row_indices.add(base + 6);
        let idx7 = *row_indices.add(base + 7);

        // Load 8 bin indices
        let bin0 = *feature_bins.add(idx0) as usize;
        let bin1 = *feature_bins.add(idx1) as usize;
        let bin2 = *feature_bins.add(idx2) as usize;
        let bin3 = *feature_bins.add(idx3) as usize;
        let bin4 = *feature_bins.add(idx4) as usize;
        let bin5 = *feature_bins.add(idx5) as usize;
        let bin6 = *feature_bins.add(idx6) as usize;
        let bin7 = *feature_bins.add(idx7) as usize;

        // Load 8 gradients
        let grad0 = *gradients.add(idx0);
        let grad1 = *gradients.add(idx1);
        let grad2 = *gradients.add(idx2);
        let grad3 = *gradients.add(idx3);
        let grad4 = *gradients.add(idx4);
        let grad5 = *gradients.add(idx5);
        let grad6 = *gradients.add(idx6);
        let grad7 = *gradients.add(idx7);

        // Load 8 hessians
        let hess0 = *hessians.add(idx0);
        let hess1 = *hessians.add(idx1);
        let hess2 = *hessians.add(idx2);
        let hess3 = *hessians.add(idx3);
        let hess4 = *hessians.add(idx4);
        let hess5 = *hessians.add(idx5);
        let hess6 = *hessians.add(idx6);
        let hess7 = *hessians.add(idx7);

        // Accumulate all 8 (scatter operation)
        *hist_grads.add(bin0) += grad0;
        *hist_hess.add(bin0) += hess0;
        *hist_counts.add(bin0) += 1;

        *hist_grads.add(bin1) += grad1;
        *hist_hess.add(bin1) += hess1;
        *hist_counts.add(bin1) += 1;

        *hist_grads.add(bin2) += grad2;
        *hist_hess.add(bin2) += hess2;
        *hist_counts.add(bin2) += 1;

        *hist_grads.add(bin3) += grad3;
        *hist_hess.add(bin3) += hess3;
        *hist_counts.add(bin3) += 1;

        *hist_grads.add(bin4) += grad4;
        *hist_hess.add(bin4) += hess4;
        *hist_counts.add(bin4) += 1;

        *hist_grads.add(bin5) += grad5;
        *hist_hess.add(bin5) += hess5;
        *hist_counts.add(bin5) += 1;

        *hist_grads.add(bin6) += grad6;
        *hist_hess.add(bin6) += hess6;
        *hist_counts.add(bin6) += 1;

        *hist_grads.add(bin7) += grad7;
        *hist_hess.add(bin7) += hess7;
        *hist_counts.add(bin7) += 1;
    }

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let idx = *row_indices.add(base + i);
        let bin = *feature_bins.add(idx) as usize;
        let grad = *gradients.add(idx);
        let hess = *hessians.add(idx);

        *hist_grads.add(bin) += grad;
        *hist_hess.add(bin) += hess;
        *hist_counts.add(bin) += 1;
    }
}

/// Scalar histogram accumulation for contiguous rows (0..num_rows)
///
/// # Safety
/// All pointers must be valid and properly sized.
#[inline]
pub unsafe fn histogram_accumulate_contiguous_scalar(
    feature_bins: *const u8,
    num_rows: usize,
    gradients: *const f32,
    hessians: *const f32,
    hist_grads: *mut f32,
    hist_hess: *mut f32,
    hist_counts: *mut u32,
) {
    // 8x unrolled for better ILP
    let chunks = num_rows / 8;
    let remainder = num_rows % 8;

    // Process 8 rows at a time
    for i in 0..chunks {
        let base = i * 8;

        // Direct sequential access - no indirection
        let bin0 = *feature_bins.add(base) as usize;
        let bin1 = *feature_bins.add(base + 1) as usize;
        let bin2 = *feature_bins.add(base + 2) as usize;
        let bin3 = *feature_bins.add(base + 3) as usize;
        let bin4 = *feature_bins.add(base + 4) as usize;
        let bin5 = *feature_bins.add(base + 5) as usize;
        let bin6 = *feature_bins.add(base + 6) as usize;
        let bin7 = *feature_bins.add(base + 7) as usize;

        let grad0 = *gradients.add(base);
        let grad1 = *gradients.add(base + 1);
        let grad2 = *gradients.add(base + 2);
        let grad3 = *gradients.add(base + 3);
        let grad4 = *gradients.add(base + 4);
        let grad5 = *gradients.add(base + 5);
        let grad6 = *gradients.add(base + 6);
        let grad7 = *gradients.add(base + 7);

        let hess0 = *hessians.add(base);
        let hess1 = *hessians.add(base + 1);
        let hess2 = *hessians.add(base + 2);
        let hess3 = *hessians.add(base + 3);
        let hess4 = *hessians.add(base + 4);
        let hess5 = *hessians.add(base + 5);
        let hess6 = *hessians.add(base + 6);
        let hess7 = *hessians.add(base + 7);

        // Accumulate all 8
        *hist_grads.add(bin0) += grad0;
        *hist_hess.add(bin0) += hess0;
        *hist_counts.add(bin0) += 1;

        *hist_grads.add(bin1) += grad1;
        *hist_hess.add(bin1) += hess1;
        *hist_counts.add(bin1) += 1;

        *hist_grads.add(bin2) += grad2;
        *hist_hess.add(bin2) += hess2;
        *hist_counts.add(bin2) += 1;

        *hist_grads.add(bin3) += grad3;
        *hist_hess.add(bin3) += hess3;
        *hist_counts.add(bin3) += 1;

        *hist_grads.add(bin4) += grad4;
        *hist_hess.add(bin4) += hess4;
        *hist_counts.add(bin4) += 1;

        *hist_grads.add(bin5) += grad5;
        *hist_hess.add(bin5) += hess5;
        *hist_counts.add(bin5) += 1;

        *hist_grads.add(bin6) += grad6;
        *hist_hess.add(bin6) += hess6;
        *hist_counts.add(bin6) += 1;

        *hist_grads.add(bin7) += grad7;
        *hist_hess.add(bin7) += hess7;
        *hist_counts.add(bin7) += 1;
    }

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let bin = *feature_bins.add(base + i) as usize;
        let grad = *gradients.add(base + i);
        let hess = *hessians.add(base + i);

        *hist_grads.add(bin) += grad;
        *hist_hess.add(bin) += hess;
        *hist_counts.add(bin) += 1;
    }
}

// ============================================================================
// Histogram Merge/Subtract
// ============================================================================

/// Scalar merge of histogram gradient arrays
#[inline]
pub fn merge_histogram_grads_scalar(self_grads: &mut [f32; 256], other_grads: &[f32; 256]) {
    for i in 0..256 {
        self_grads[i] += other_grads[i];
    }
}

/// Scalar merge of histogram hessian arrays
#[inline]
pub fn merge_histogram_hess_scalar(self_hess: &mut [f32; 256], other_hess: &[f32; 256]) {
    for i in 0..256 {
        self_hess[i] += other_hess[i];
    }
}

/// Scalar merge of histogram count arrays
#[inline]
pub fn merge_histogram_counts_scalar(self_counts: &mut [u32; 256], other_counts: &[u32; 256]) {
    for i in 0..256 {
        self_counts[i] += other_counts[i];
    }
}

/// Scalar subtract of histogram gradient arrays
#[inline]
pub fn subtract_histogram_grads_scalar(self_grads: &mut [f32; 256], other_grads: &[f32; 256]) {
    for i in 0..256 {
        self_grads[i] -= other_grads[i];
    }
}

/// Scalar subtract of histogram hessian arrays
#[inline]
pub fn subtract_histogram_hess_scalar(self_hess: &mut [f32; 256], other_hess: &[f32; 256]) {
    for i in 0..256 {
        self_hess[i] -= other_hess[i];
    }
}

/// Scalar subtract of histogram count arrays
#[inline]
pub fn subtract_histogram_counts_scalar(self_counts: &mut [u32; 256], other_counts: &[u32; 256]) {
    for i in 0..256 {
        self_counts[i] -= other_counts[i];
    }
}

// ============================================================================
// Grad/Hess Interleaving
// ============================================================================

/// Block size for cache-blocked histogram building
pub const BLOCK_SIZE: usize = 2048;

/// Scalar copy of gradients and hessians to interleaved cache.
///
/// # Safety
/// - `gradients` and `hessians` must have at least `start + len` elements
/// - `gh_cache` must have capacity for `len` elements
#[inline]
pub unsafe fn copy_gh_interleaved_scalar(
    gradients: &[f32],
    hessians: &[f32],
    start: usize,
    len: usize,
    gh_cache: &mut [(f32, f32); BLOCK_SIZE],
) {
    for i in 0..len {
        let g = *gradients.get_unchecked(start + i);
        let h = *hessians.get_unchecked(start + i);
        *gh_cache.get_unchecked_mut(i) = (g, h);
    }
}

/// Scalar copy of gradients and hessians for indexed (non-contiguous) rows.
#[inline]
pub fn copy_gh_indexed_scalar(
    gradients: &[f32],
    hessians: &[f32],
    indices: &[usize],
    gh_cache: &mut [(f32, f32); BLOCK_SIZE],
) {
    unsafe {
        for (i, &row_idx) in indices.iter().enumerate() {
            let g = *gradients.get_unchecked(row_idx);
            let h = *hessians.get_unchecked(row_idx);
            *gh_cache.get_unchecked_mut(i) = (g, h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_accumulate_indexed() {
        let feature_bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 3];
        let row_indices: Vec<usize> = (0..10).collect();
        let gradients: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let hessians: Vec<f32> = vec![1.0; 10];

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            histogram_accumulate_scalar(HistogramAccumParams {
                feature_bins: feature_bins.as_ptr(),
                row_indices: row_indices.as_ptr(),
                num_rows: 10,
                gradients: gradients.as_ptr(),
                hessians: hessians.as_ptr(),
                hist_grads: hist_grads.as_mut_ptr(),
                hist_hess: hist_hess.as_mut_ptr(),
                hist_counts: hist_counts.as_mut_ptr(),
            });
        }

        // Bin 0: rows 0, 3, 6 -> grads 1+4+7=12
        assert!((hist_grads[0] - 12.0).abs() < 1e-5);
        assert_eq!(hist_counts[0], 3);

        // Bin 1: rows 1, 4, 7 -> grads 2+5+8=15
        assert!((hist_grads[1] - 15.0).abs() < 1e-5);
        assert_eq!(hist_counts[1], 3);

        // Bin 2: rows 2, 5, 8 -> grads 3+6+9=18
        assert!((hist_grads[2] - 18.0).abs() < 1e-5);
        assert_eq!(hist_counts[2], 3);

        // Bin 3: row 9 -> grad 10
        assert!((hist_grads[3] - 10.0).abs() < 1e-5);
        assert_eq!(hist_counts[3], 1);
    }

    #[test]
    fn test_scalar_accumulate_contiguous() {
        let feature_bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 3];
        let gradients: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let hessians: Vec<f32> = vec![1.0; 10];

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            histogram_accumulate_contiguous_scalar(
                feature_bins.as_ptr(),
                10,
                gradients.as_ptr(),
                hessians.as_ptr(),
                hist_grads.as_mut_ptr(),
                hist_hess.as_mut_ptr(),
                hist_counts.as_mut_ptr(),
            );
        }

        // Same expected results
        assert!((hist_grads[0] - 12.0).abs() < 1e-5);
        assert_eq!(hist_counts[0], 3);
        assert!((hist_grads[1] - 15.0).abs() < 1e-5);
        assert_eq!(hist_counts[1], 3);
        assert!((hist_grads[2] - 18.0).abs() < 1e-5);
        assert_eq!(hist_counts[2], 3);
        assert!((hist_grads[3] - 10.0).abs() < 1e-5);
        assert_eq!(hist_counts[3], 1);
    }
}
