//! SIMD-optimized kernels for histogram operations
//!
//! # Architecture Support
//!
//! - **x86_64**: AVX2 (baseline) with AVX-512 runtime upgrade
//! - **aarch64**: ARM NEON (future)
//! - **Other**: Scalar fallback
//!
//! # Runtime Detection
//!
//! Feature availability is detected at runtime and cached for subsequent calls.
//! The first call to any kernel function triggers detection.

pub mod fallback;

#[cfg(target_arch = "x86_64")]
pub mod x86;

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

    #[cfg(not(target_arch = "x86_64"))]
    fn detect() -> Self {
        // TODO: Add ARM NEON detection
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
            SimdLevel::Avx512 => {
                // TODO: Implement AVX-512 kernel
                x86::histogram_accumulate_avx2(
                    feature_bins, row_indices, num_rows,
                    gradients, hessians,
                    hist_grads, hist_hess, hist_counts,
                )
            }
            SimdLevel::Avx2 => {
                x86::histogram_accumulate_avx2(
                    feature_bins, row_indices, num_rows,
                    gradients, hessians,
                    hist_grads, hist_hess, hist_counts,
                )
            }
            SimdLevel::Scalar => {
                fallback::histogram_accumulate_scalar(
                    feature_bins, row_indices, num_rows,
                    gradients, hessians,
                    hist_grads, hist_hess, hist_counts,
                )
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
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
            SimdLevel::Avx512 => {
                // TODO: Implement AVX-512 kernel
                x86::histogram_accumulate_contiguous_avx2(
                    feature_bins, num_rows,
                    gradients, hessians,
                    hist_grads, hist_hess, hist_counts,
                )
            }
            SimdLevel::Avx2 => {
                x86::histogram_accumulate_contiguous_avx2(
                    feature_bins, num_rows,
                    gradients, hessians,
                    hist_grads, hist_hess, hist_counts,
                )
            }
            SimdLevel::Scalar => {
                fallback::histogram_accumulate_contiguous_scalar(
                    feature_bins, num_rows,
                    gradients, hessians,
                    hist_grads, hist_hess, hist_counts,
                )
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        fallback::histogram_accumulate_contiguous_scalar(
            feature_bins, num_rows,
            gradients, hessians,
            hist_grads, hist_hess, hist_counts,
        )
    }
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
