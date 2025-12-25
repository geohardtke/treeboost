//! Scalar fallback implementations for histogram operations
//!
//! These implementations work on any architecture without SIMD support.
//! They serve as the baseline and are used when SIMD is not available.

/// Scalar histogram accumulation with indexed rows
///
/// # Safety
/// All pointers must be valid and properly sized.
#[inline]
pub unsafe fn histogram_accumulate_scalar(
    feature_bins: *const u8,
    row_indices: *const usize,
    num_rows: usize,
    gradients: *const f32,
    hessians: *const f32,
    hist_grads: *mut f32,
    hist_hess: *mut f32,
    hist_counts: *mut u32,
) {
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
            histogram_accumulate_scalar(
                feature_bins.as_ptr(),
                row_indices.as_ptr(),
                10,
                gradients.as_ptr(),
                hessians.as_ptr(),
                hist_grads.as_mut_ptr(),
                hist_hess.as_mut_ptr(),
                hist_counts.as_mut_ptr(),
            );
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
