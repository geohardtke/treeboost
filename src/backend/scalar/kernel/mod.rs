//! SIMD-optimized kernels for GBDT operations
//!
//! # Architecture Support
//!
//! - **x86_64**: AVX2 (baseline) with AVX-512 runtime upgrade
//! - **aarch64**: ARM NEON
//! - **Other**: Scalar fallback
//!
//! # Runtime Detection
//!
//! Feature availability is detected at runtime and cached for subsequent calls.
//! The first call to any kernel function triggers detection.

pub mod fallback;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

// x86_64 exports
#[cfg(target_arch = "x86_64")]
pub use x86::{find_best_split_scalar, find_best_split_simd, SplitCandidate};
#[cfg(target_arch = "x86_64")]
pub use x86::{unpack_4bit, unpack_4bit_scalar};

// aarch64 exports
#[cfg(target_arch = "aarch64")]
pub use arm::{find_best_split_scalar, find_best_split_simd, SplitCandidate};
#[cfg(target_arch = "aarch64")]
pub use fallback::{unpack_4bit, unpack_4bit_scalar};

// Other architectures - fallback
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use fallback::{find_best_split_scalar, find_best_split_simd, SplitCandidate};
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use fallback::{unpack_4bit, unpack_4bit_scalar};

use std::sync::OnceLock;

/// Cached SIMD level (detected once at first use)
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Detected SIMD capability level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD available, use scalar fallback
    Scalar,
    /// AVX2 available (256-bit, 8 x f32)
    Avx2,
    /// AVX-512 available (512-bit, 16 x f32)
    Avx512,
    /// ARM NEON available (128-bit, 4 x f32)
    Neon,
}

impl SimdLevel {
    /// Detect the best available SIMD level at runtime
    #[cfg(target_arch = "x86_64")]
    fn detect() -> Self {
        if std::arch::is_x86_feature_detected!("avx512f") {
            SimdLevel::Avx512
        } else if std::arch::is_x86_feature_detected!("avx2") {
            SimdLevel::Avx2
        } else {
            SimdLevel::Scalar
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect() -> Self {
        // NEON is always available on aarch64
        SimdLevel::Neon
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect() -> Self {
        SimdLevel::Scalar
    }
}

/// Get the current SIMD level (cached after first call)
#[inline]
pub fn simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(SimdLevel::detect)
}

/// Check if AVX2 is available
#[inline]
pub fn has_avx2() -> bool {
    matches!(simd_level(), SimdLevel::Avx2 | SimdLevel::Avx512)
}

/// Check if AVX-512 is available
#[inline]
pub fn has_avx512() -> bool {
    matches!(simd_level(), SimdLevel::Avx512)
}

/// Check if ARM NEON is available
#[inline]
pub fn has_neon() -> bool {
    matches!(simd_level(), SimdLevel::Neon)
}

// ============================================================================
// Public API - Runtime-dispatched split finding
// ============================================================================

/// Find the best split for a histogram with runtime SIMD dispatch
///
/// Automatically selects the best available implementation:
/// - AVX2/FMA on x86_64 with those features
/// - Scalar fallback otherwise
///
/// # Arguments
/// * `hist_grads` - Sum of gradients per bin [256]
/// * `hist_hess` - Sum of hessians per bin [256]
/// * `hist_counts` - Count per bin [256]
/// * `total_gradient` - Total gradient sum across all bins
/// * `total_hessian` - Total hessian sum across all bins
/// * `total_count` - Total sample count across all bins
/// * `lambda` - L2 regularization parameter
/// * `min_samples_leaf` - Minimum samples required in each leaf
/// * `min_hessian_leaf` - Minimum hessian sum required in each leaf
///
/// # Returns
/// The best split candidate, or None if no valid split exists
#[inline]
pub fn find_best_split(
    hist_grads: &[f32; 256],
    hist_hess: &[f32; 256],
    hist_counts: &[u32; 256],
    total_gradient: f32,
    total_hessian: f32,
    total_count: u32,
    lambda: f32,
    min_samples_leaf: u32,
    min_hessian_leaf: f32,
) -> Option<SplitCandidate> {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // Safety: we just checked for AVX2 support
            return unsafe {
                find_best_split_simd(
                    hist_grads, hist_hess, hist_counts,
                    total_gradient, total_hessian, total_count,
                    lambda, min_samples_leaf, min_hessian_leaf,
                )
            };
        }
    }

    find_best_split_scalar(
        hist_grads, hist_hess, hist_counts,
        total_gradient, total_hessian, total_count,
        lambda, min_samples_leaf, min_hessian_leaf,
    )
}

// ============================================================================
// Public API - Runtime-dispatched histogram accumulation
// ============================================================================

/// Accumulate gradient/hessian statistics into histogram bins
///
/// This is the core hot loop for GBDT training. For each row:
/// - Look up the bin index from feature_bins
/// - Add gradient[row] to histogram[bin].sum_gradients
/// - Add hessian[row] to histogram[bin].sum_hessians
/// - Increment histogram[bin].count
///
/// # Arguments
/// * `feature_bins` - Bin indices for each row (u8, 0-255)
/// * `row_indices` - Which rows to process
/// * `gradients` - Gradient values (full dataset)
/// * `hessians` - Hessian values (full dataset)
/// * `hist_grads` - Output: sum of gradients per bin [256]
/// * `hist_hess` - Output: sum of hessians per bin [256]
/// * `hist_counts` - Output: count per bin [256]
///
/// # Safety
/// All pointers must be valid and properly sized.
#[inline]
pub unsafe fn histogram_accumulate(
    feature_bins: *const u8,
    row_indices: *const usize,
    num_rows: usize,
    gradients: *const f32,
    hessians: *const f32,
    hist_grads: *mut f32,
    hist_hess: *mut f32,
    hist_counts: *mut u32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        match simd_level() {
            SimdLevel::Avx512 | SimdLevel::Avx2 => {
                x86::histogram_accumulate_avx2(
                    feature_bins, row_indices, num_rows,
                    gradients, hessians,
                    hist_grads, hist_hess, hist_counts,
                )
            }
            _ => {
                fallback::histogram_accumulate_scalar(
                    feature_bins, row_indices, num_rows,
                    gradients, hessians,
                    hist_grads, hist_hess, hist_counts,
                )
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON histogram uses scalar fallback (scatter is inherently sequential)
        // NEON is used for grad/hess loading in copy_gh_interleaved
        fallback::histogram_accumulate_scalar(
            feature_bins, row_indices, num_rows,
            gradients, hessians,
            hist_grads, hist_hess, hist_counts,
        )
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        fallback::histogram_accumulate_scalar(
            feature_bins, row_indices, num_rows,
            gradients, hessians,
            hist_grads, hist_hess, hist_counts,
        )
    }
}

/// Accumulate gradient/hessian for contiguous rows (0..num_rows)
///
/// Optimized path when processing all rows sequentially (e.g., root node).
/// Eliminates indirection through row_indices.
///
/// # Safety
/// All pointers must be valid and properly sized.
#[inline]
pub unsafe fn histogram_accumulate_contiguous(
    feature_bins: *const u8,
    num_rows: usize,
    gradients: *const f32,
    hessians: *const f32,
    hist_grads: *mut f32,
    hist_hess: *mut f32,
    hist_counts: *mut u32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        match simd_level() {
            SimdLevel::Avx512 | SimdLevel::Avx2 => {
                x86::histogram_accumulate_contiguous_avx2(
                    feature_bins, num_rows,
                    gradients, hessians,
                    hist_grads, hist_hess, hist_counts,
                )
            }
            _ => {
                fallback::histogram_accumulate_contiguous_scalar(
                    feature_bins, num_rows,
                    gradients, hessians,
                    hist_grads, hist_hess, hist_counts,
                )
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        fallback::histogram_accumulate_contiguous_scalar(
            feature_bins, num_rows,
            gradients, hessians,
            hist_grads, hist_hess, hist_counts,
        )
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        fallback::histogram_accumulate_contiguous_scalar(
            feature_bins, num_rows,
            gradients, hessians,
            hist_grads, hist_hess, hist_counts,
        )
    }
}

// ============================================================================
// Public API - Runtime-dispatched histogram merge
// ============================================================================

/// Merge histogram gradient arrays with runtime SIMD dispatch
///
/// Adds `other_grads` into `self_grads`.
#[inline]
pub fn merge_histogram_grads(self_grads: &mut [f32; 256], other_grads: &[f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                x86::merge_histogram_grads_avx2(self_grads, other_grads);
            }
            return;
        }
    }
    fallback::merge_histogram_grads_scalar(self_grads, other_grads);
}

/// Merge histogram hessian arrays with runtime SIMD dispatch
#[inline]
pub fn merge_histogram_hess(self_hess: &mut [f32; 256], other_hess: &[f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                x86::merge_histogram_hess_avx2(self_hess, other_hess);
            }
            return;
        }
    }
    fallback::merge_histogram_hess_scalar(self_hess, other_hess);
}

/// Merge histogram count arrays with runtime SIMD dispatch
#[inline]
pub fn merge_histogram_counts(self_counts: &mut [u32; 256], other_counts: &[u32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                x86::merge_histogram_counts_avx2(self_counts, other_counts);
            }
            return;
        }
    }
    fallback::merge_histogram_counts_scalar(self_counts, other_counts);
}

/// Subtract histogram gradient arrays with runtime SIMD dispatch
///
/// Subtracts `other_grads` from `self_grads`.
#[inline]
pub fn subtract_histogram_grads(self_grads: &mut [f32; 256], other_grads: &[f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                x86::subtract_histogram_grads_avx2(self_grads, other_grads);
            }
            return;
        }
    }
    fallback::subtract_histogram_grads_scalar(self_grads, other_grads);
}

/// Subtract histogram hessian arrays with runtime SIMD dispatch
#[inline]
pub fn subtract_histogram_hess(self_hess: &mut [f32; 256], other_hess: &[f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                x86::subtract_histogram_hess_avx2(self_hess, other_hess);
            }
            return;
        }
    }
    fallback::subtract_histogram_hess_scalar(self_hess, other_hess);
}

/// Subtract histogram count arrays with runtime SIMD dispatch
#[inline]
pub fn subtract_histogram_counts(self_counts: &mut [u32; 256], other_counts: &[u32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                x86::subtract_histogram_counts_avx2(self_counts, other_counts);
            }
            return;
        }
    }
    fallback::subtract_histogram_counts_scalar(self_counts, other_counts);
}

// ============================================================================
// Public API - Runtime-dispatched grad/hess interleaving
// ============================================================================

/// Block size for cache-blocked histogram building
pub const BLOCK_SIZE: usize = 2048;

/// Copy gradients and hessians to interleaved cache with SIMD optimization.
///
/// Loads gradients and hessians from separate arrays and interleaves them
/// into `[(g0, h0), (g1, h1), ...]` format for cache-friendly access during
/// histogram building.
///
/// # Arguments
/// * `gradients` - Source gradient array
/// * `hessians` - Source hessian array
/// * `start` - Starting index in source arrays
/// * `len` - Number of elements to copy
/// * `gh_cache` - Output interleaved cache
///
/// # Safety
/// - `gradients` and `hessians` must have at least `start + len` elements
/// - `gh_cache` must have capacity for `len` elements
#[inline]
pub unsafe fn copy_gh_interleaved(
    gradients: &[f32],
    hessians: &[f32],
    start: usize,
    len: usize,
    gh_cache: &mut [(f32, f32); BLOCK_SIZE],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            x86::copy_gh_interleaved_avx2(gradients, hessians, start, len, gh_cache);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        arm::copy_gh_interleaved_neon(gradients, hessians, start, len, gh_cache);
        return;
    }

    // Scalar fallback
    fallback::copy_gh_interleaved_scalar(gradients, hessians, start, len, gh_cache);
}

/// Copy gradients and hessians for indexed (non-contiguous) rows.
///
/// Uses scalar implementation since gather operations have high latency.
#[inline]
pub fn copy_gh_indexed(
    gradients: &[f32],
    hessians: &[f32],
    indices: &[usize],
    gh_cache: &mut [(f32, f32); BLOCK_SIZE],
) {
    fallback::copy_gh_indexed_scalar(gradients, hessians, indices, gh_cache);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let level = simd_level();
        println!("Detected SIMD level: {:?}", level);

        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, should at least have AVX2 on modern systems
            println!("AVX2 available: {}", has_avx2());
            println!("AVX-512 available: {}", has_avx512());
        }
    }

    #[test]
    fn test_histogram_accumulate_basic() {
        // Simple test with known values
        let feature_bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 0, 1];
        let row_indices: Vec<usize> = (0..8).collect();
        let gradients: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let hessians: Vec<f32> = vec![1.0; 8];

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            histogram_accumulate(
                feature_bins.as_ptr(),
                row_indices.as_ptr(),
                8,
                gradients.as_ptr(),
                hessians.as_ptr(),
                hist_grads.as_mut_ptr(),
                hist_hess.as_mut_ptr(),
                hist_counts.as_mut_ptr(),
            );
        }

        // Bin 0: rows 0, 3, 6 -> grads 1+4+7=12, counts=3
        assert!((hist_grads[0] - 12.0).abs() < 1e-5);
        assert_eq!(hist_counts[0], 3);

        // Bin 1: rows 1, 4, 7 -> grads 2+5+8=15, counts=3
        assert!((hist_grads[1] - 15.0).abs() < 1e-5);
        assert_eq!(hist_counts[1], 3);

        // Bin 2: rows 2, 5 -> grads 3+6=9, counts=2
        assert!((hist_grads[2] - 9.0).abs() < 1e-5);
        assert_eq!(hist_counts[2], 2);
    }

    #[test]
    fn test_histogram_accumulate_contiguous() {
        let feature_bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 0, 1];
        let gradients: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let hessians: Vec<f32> = vec![1.0; 8];

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            histogram_accumulate_contiguous(
                feature_bins.as_ptr(),
                8,
                gradients.as_ptr(),
                hessians.as_ptr(),
                hist_grads.as_mut_ptr(),
                hist_hess.as_mut_ptr(),
                hist_counts.as_mut_ptr(),
            );
        }

        // Same expected results as indexed version
        assert!((hist_grads[0] - 12.0).abs() < 1e-5);
        assert_eq!(hist_counts[0], 3);
        assert!((hist_grads[1] - 15.0).abs() < 1e-5);
        assert_eq!(hist_counts[1], 3);
        assert!((hist_grads[2] - 9.0).abs() < 1e-5);
        assert_eq!(hist_counts[2], 2);
    }
}
