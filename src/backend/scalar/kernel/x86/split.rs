//! SIMD-optimized split finding kernels
//!
//! This module provides vectorized implementations of the split finding algorithm
//! used in GBDT training. The key insight is that:
//!
//! 1. We iterate over 256 bins computing prefix sums (cumulative gradients/hessians/counts)
//! 2. For each bin, we compute the gain using the Friedman MSE formula
//! 3. Both operations are highly parallelizable with SIMD
//!
//! # Algorithm
//!
//! For each bin threshold t:
//! - left_g = sum(gradients[0..=t])
//! - left_h = sum(hessians[0..=t])
//! - left_n = sum(counts[0..=t])
//! - right_g = total_g - left_g
//! - right_h = total_h - left_h
//! - right_n = total_n - left_n
//! - gain = 0.5 * (left_g² / (left_h + λ) + right_g² / (right_h + λ) - total_g² / (total_h + λ))
//!
//! # SIMD Strategy
//!
//! 1. **Prefix sums**: Compute cumulative sums for all 256 bins first
//! 2. **Parallel gain**: Compute gain for 8 bins at a time using AVX2
//! 3. **Horizontal max**: Find the maximum gain across all bins

use crate::backend::scalar::kernel::fallback::SplitParams;

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

/// Find the best split for a histogram using scalar code
///
/// This is the baseline implementation that works on all architectures.
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

/// Find the best split for a histogram using SIMD (AVX2)
///
/// This implementation uses AVX2 to:
/// 1. Compute prefix sums for all 256 bins
/// 2. Evaluate gain for 8 bins in parallel
/// 3. Find the maximum gain using horizontal operations
///
/// # Safety
/// Requires AVX2 support. Caller must check `is_x86_feature_detected!("avx2")`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn find_best_split_simd(
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
    use std::arch::x86_64::*;

    // Step 1: Compute prefix sums for all 256 bins
    // We'll store cumulative sums in separate arrays
    let mut prefix_grads = [0.0f32; 256];
    let mut prefix_hess = [0.0f32; 256];
    let mut prefix_counts = [0u32; 256];

    // Scalar prefix sum (hard to vectorize due to data dependency)
    // But we can still benefit from good cache behavior
    let mut sum_g = 0.0f32;
    let mut sum_h = 0.0f32;
    let mut sum_n = 0u32;

    for i in 0..256 {
        sum_g += hist_grads[i];
        sum_h += hist_hess[i];
        sum_n += hist_counts[i];
        prefix_grads[i] = sum_g;
        prefix_hess[i] = sum_h;
        prefix_counts[i] = sum_n;
    }

    // Step 2: Compute gains for 8 bins at a time using SIMD
    // We process bins 0..248 in chunks of 8, then handle 248..255 separately

    // Broadcast constants
    let total_g_vec = _mm256_set1_ps(total_gradient);
    let total_h_vec = _mm256_set1_ps(total_hessian);
    let total_n_vec = _mm256_set1_ps(total_count as f32);
    let lambda_vec = _mm256_set1_ps(lambda);
    let half_vec = _mm256_set1_ps(0.5);
    let min_samples_vec = _mm256_set1_ps(min_samples_leaf as f32);
    let min_hessian_vec = _mm256_set1_ps(min_hessian_leaf);
    let neg_inf_vec = _mm256_set1_ps(f32::NEG_INFINITY);

    // Parent score (constant)
    let parent_score = (total_gradient * total_gradient) / (total_hessian + lambda);
    let parent_score_vec = _mm256_set1_ps(parent_score);

    // Track best gain and corresponding bin
    let mut best_gain = f32::NEG_INFINITY;
    let mut best_bin = 0u8;
    let mut best_left_g = 0.0f32;
    let mut best_left_h = 0.0f32;
    let mut best_left_n = 0u32;

    // Process 8 bins at a time
    let num_chunks = 255 / 8; // 31 chunks (bins 0-247)

    for chunk in 0..num_chunks {
        let base = chunk * 8;

        // Load 8 prefix sums
        let left_g = _mm256_loadu_ps(prefix_grads.as_ptr().add(base));
        let left_h = _mm256_loadu_ps(prefix_hess.as_ptr().add(base));

        // Load counts and convert to float for comparison
        let left_n_int: [u32; 8] = [
            prefix_counts[base],
            prefix_counts[base + 1],
            prefix_counts[base + 2],
            prefix_counts[base + 3],
            prefix_counts[base + 4],
            prefix_counts[base + 5],
            prefix_counts[base + 6],
            prefix_counts[base + 7],
        ];
        let left_n = _mm256_cvtepi32_ps(_mm256_loadu_si256(left_n_int.as_ptr() as *const __m256i));

        // Compute right child stats: right = total - left
        let right_g = _mm256_sub_ps(total_g_vec, left_g);
        let right_h = _mm256_sub_ps(total_h_vec, left_h);
        let right_n = _mm256_sub_ps(total_n_vec, left_n);

        // Check min_samples constraint
        // valid = (left_n >= min_samples) && (right_n >= min_samples)
        let left_samples_ok = _mm256_cmp_ps(left_n, min_samples_vec, _CMP_GE_OQ);
        let right_samples_ok = _mm256_cmp_ps(right_n, min_samples_vec, _CMP_GE_OQ);
        let samples_ok = _mm256_and_ps(left_samples_ok, right_samples_ok);

        // Check min_hessian constraint
        let left_hess_ok = _mm256_cmp_ps(left_h, min_hessian_vec, _CMP_GE_OQ);
        let right_hess_ok = _mm256_cmp_ps(right_h, min_hessian_vec, _CMP_GE_OQ);
        let hess_ok = _mm256_and_ps(left_hess_ok, right_hess_ok);

        // Combined validity mask
        let valid_mask = _mm256_and_ps(samples_ok, hess_ok);

        // Compute gain for all 8 bins
        // left_score = left_g² / (left_h + λ)
        let left_g_sq = _mm256_mul_ps(left_g, left_g);
        let left_denom = _mm256_add_ps(left_h, lambda_vec);
        let left_score = _mm256_div_ps(left_g_sq, left_denom);

        // right_score = right_g² / (right_h + λ)
        let right_g_sq = _mm256_mul_ps(right_g, right_g);
        let right_denom = _mm256_add_ps(right_h, lambda_vec);
        let right_score = _mm256_div_ps(right_g_sq, right_denom);

        // gain = 0.5 * (left_score + right_score - parent_score)
        let sum_scores = _mm256_add_ps(left_score, right_score);
        let diff = _mm256_sub_ps(sum_scores, parent_score_vec);
        let gain = _mm256_mul_ps(half_vec, diff);

        // Apply validity mask (invalid bins get -inf gain)
        let gain_masked = _mm256_blendv_ps(neg_inf_vec, gain, valid_mask);

        // Extract gains and find maximum
        let gain_arr: [f32; 8] = std::mem::transmute(gain_masked);

        for i in 0..8 {
            if gain_arr[i] > best_gain {
                best_gain = gain_arr[i];
                best_bin = (base + i) as u8;
                best_left_g = prefix_grads[base + i];
                best_left_h = prefix_hess[base + i];
                best_left_n = prefix_counts[base + i];
            }
        }
    }

    // Handle remaining bins 248..254 (bin 255 can't be a threshold)
    for bin in (num_chunks * 8)..255 {
        let left_g = prefix_grads[bin];
        let left_h = prefix_hess[bin];
        let left_n = prefix_counts[bin];
        let right_g = total_gradient - left_g;
        let right_h = total_hessian - left_h;
        let right_n = total_count - left_n;

        if left_n < min_samples_leaf || right_n < min_samples_leaf {
            continue;
        }
        if left_h < min_hessian_leaf || right_h < min_hessian_leaf {
            continue;
        }

        let left_score = (left_g * left_g) / (left_h + lambda);
        let right_score = (right_g * right_g) / (right_h + lambda);
        let gain = 0.5 * (left_score + right_score - parent_score);

        if gain > best_gain {
            best_gain = gain;
            best_bin = bin as u8;
            best_left_g = left_g;
            best_left_h = left_h;
            best_left_n = left_n;
        }
    }

    if best_gain > f32::NEG_INFINITY && best_left_n > 0 && (total_count - best_left_n) > 0 {
        Some(SplitCandidate {
            bin_threshold: best_bin,
            gain: best_gain,
            left_gradient: best_left_g,
            left_hessian: best_left_h,
            left_count: best_left_n,
            right_gradient: total_gradient - best_left_g,
            right_hessian: total_hessian - best_left_h,
            right_count: total_count - best_left_n,
        })
    } else {
        None
    }
}

/// Non-x86 fallback - just calls scalar version
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn find_best_split_simd(
    hist_grads: &[f32; 256],
    hist_hess: &[f32; 256],
    hist_counts: &[u32; 256],
    params: SplitParams,
) -> Option<SplitCandidate> {
    find_best_split_scalar(hist_grads, hist_hess, hist_counts, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_histogram() -> ([f32; 256], [f32; 256], [u32; 256]) {
        let mut grads = [0.0f32; 256];
        let mut hess = [0.0f32; 256];
        let mut counts = [0u32; 256];

        // Create a clear split: bins 0-127 have negative gradients, 128-255 have positive
        for bin in 0..128 {
            grads[bin] = -1.0;
            hess[bin] = 1.0;
            counts[bin] = 1;
        }
        for bin in 128..256 {
            grads[bin] = 1.0;
            hess[bin] = 1.0;
            counts[bin] = 1;
        }

        (grads, hess, counts)
    }

    #[test]
    fn test_scalar_split_finding() {
        let (grads, hess, counts) = create_test_histogram();

        let result = find_best_split_scalar(
            &grads,
            &hess,
            &counts,
            SplitParams {
                total_gradient: 0.0,
                total_hessian: 256.0,
                total_count: 256,
                lambda: 0.0,
                min_samples_leaf: 1,
                min_hessian_leaf: 1.0,
            },
        );

        assert!(result.is_some());
        let split = result.unwrap();

        // Best split should be at bin 127
        assert_eq!(split.bin_threshold, 127);
        assert!(split.gain > 0.0);
        assert_eq!(split.left_count, 128);
        assert_eq!(split.right_count, 128);
    }

    #[test]
    fn test_simd_matches_scalar() {
        let (grads, hess, counts) = create_test_histogram();

        let params = SplitParams {
            total_gradient: 0.0,
            total_hessian: 256.0,
            total_count: 256,
            lambda: 1.0,
            min_samples_leaf: 1,
            min_hessian_leaf: 1.0,
        };

        let scalar_result = find_best_split_scalar(&grads, &hess, &counts, params);

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let simd_result = unsafe {
                    find_best_split_simd(&grads, &hess, &counts, params)
                };

                assert!(scalar_result.is_some());
                assert!(simd_result.is_some());

                let scalar = scalar_result.unwrap();
                let simd = simd_result.unwrap();

                assert_eq!(scalar.bin_threshold, simd.bin_threshold);
                assert!((scalar.gain - simd.gain).abs() < 1e-5);
                assert_eq!(scalar.left_count, simd.left_count);
                assert_eq!(scalar.right_count, simd.right_count);
            }
        }
    }

    #[test]
    fn test_min_samples_constraint() {
        let mut grads = [0.0f32; 256];
        let mut hess = [0.0f32; 256];
        let mut counts = [0u32; 256];

        // Only 2 samples: 1 in bin 0, 1 in bin 255
        grads[0] = -1.0;
        hess[0] = 1.0;
        counts[0] = 1;

        grads[255] = 1.0;
        hess[255] = 1.0;
        counts[255] = 1;

        // With min_samples_leaf = 5, no valid split should exist
        let result = find_best_split_scalar(
            &grads,
            &hess,
            &counts,
            SplitParams {
                total_gradient: 0.0,
                total_hessian: 2.0,
                total_count: 2,
                lambda: 0.0,
                min_samples_leaf: 5,
                min_hessian_leaf: 1.0,
            },
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_empty_histogram() {
        let grads = [0.0f32; 256];
        let hess = [0.0f32; 256];
        let counts = [0u32; 256];

        let result = find_best_split_scalar(
            &grads,
            &hess,
            &counts,
            SplitParams {
                total_gradient: 0.0,
                total_hessian: 0.0,
                total_count: 0,
                lambda: 1.0,
                min_samples_leaf: 1,
                min_hessian_leaf: 1.0,
            },
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_single_bin_histogram() {
        let mut grads = [0.0f32; 256];
        let mut hess = [0.0f32; 256];
        let mut counts = [0u32; 256];

        // All samples in bin 100
        grads[100] = 10.0;
        hess[100] = 100.0;
        counts[100] = 100;

        let result = find_best_split_scalar(
            &grads,
            &hess,
            &counts,
            SplitParams {
                total_gradient: 10.0,
                total_hessian: 100.0,
                total_count: 100,
                lambda: 1.0,
                min_samples_leaf: 1,
                min_hessian_leaf: 1.0,
            },
        );

        // No valid split since all samples are in one bin
        assert!(result.is_none());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_various_splits() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        // Test with various split points
        for split_point in [10, 50, 100, 127, 200, 250] {
            let mut grads = [0.0f32; 256];
            let mut hess = [0.0f32; 256];
            let mut counts = [0u32; 256];

            for bin in 0..=split_point {
                grads[bin] = -1.0;
                hess[bin] = 1.0;
                counts[bin] = 1;
            }
            for bin in (split_point + 1)..256 {
                grads[bin] = 1.0;
                hess[bin] = 1.0;
                counts[bin] = 1;
            }

            let total_left = (split_point + 1) as f32;
            let total_right = (255 - split_point) as f32;
            let total_g = total_right - total_left;

            let params = SplitParams {
                total_gradient: total_g,
                total_hessian: 256.0,
                total_count: 256,
                lambda: 0.0,
                min_samples_leaf: 1,
                min_hessian_leaf: 1.0,
            };

            let scalar = find_best_split_scalar(&grads, &hess, &counts, params);

            let simd = unsafe {
                find_best_split_simd(&grads, &hess, &counts, params)
            };

            assert!(scalar.is_some(), "Scalar should find split at {}", split_point);
            assert!(simd.is_some(), "SIMD should find split at {}", split_point);

            let s = scalar.unwrap();
            let v = simd.unwrap();

            assert_eq!(s.bin_threshold, v.bin_threshold,
                "Threshold mismatch for split_point {}: scalar={}, simd={}",
                split_point, s.bin_threshold, v.bin_threshold);
            assert!((s.gain - v.gain).abs() < 1e-4,
                "Gain mismatch for split_point {}: scalar={}, simd={}",
                split_point, s.gain, v.gain);
        }
    }
}
