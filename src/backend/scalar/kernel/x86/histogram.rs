//! x86_64 SIMD implementations for histogram operations
//!
//! # Key Optimizations
//!
//! 1. **Vectorized contiguous loads**: Use AVX2 `loadu_ps` for sequential access
//! 2. **8x unrolled scatter**: Scatter to bins with good ILP
//! 3. **Software prefetching**: Prefetch upcoming data
//!
//! # Architecture Notes
//!
//! The histogram scatter operation is inherently difficult to vectorize because
//! multiple rows may map to the same bin (conflict). However, the LOAD side can
//! be vectorized when data is contiguous (sequential row access).
//!
//! Key insight: AVX2 `loadu_ps` is much faster than `gather` for sequential data:
//! - `loadu_ps`: ~3-4 cycles latency, loads 8 contiguous floats
//! - `gather`: ~20-30 cycles latency, random access pattern

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 histogram accumulation with indexed rows
///
/// Uses AVX2 gather to load 8 gradients/hessians at once, then scatters
/// to histogram bins. The scatter is still scalar due to potential conflicts.
///
/// # Safety
/// - Requires AVX2 support
/// - All pointers must be valid and properly sized
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn histogram_accumulate_avx2(
    feature_bins: *const u8,
    row_indices: *const usize,
    num_rows: usize,
    gradients: *const f32,
    hessians: *const f32,
    hist_grads: *mut f32,
    hist_hess: *mut f32,
    hist_counts: *mut u32,
) {
    const PREFETCH_DISTANCE: usize = 64; // Prefetch 64 iterations ahead

    let chunks = num_rows / 8;
    let remainder = num_rows % 8;

    // Process 8 rows at a time
    for i in 0..chunks {
        let base = i * 8;

        // Prefetch upcoming data
        if base + PREFETCH_DISTANCE < num_rows {
            _mm_prefetch(
                row_indices.add(base + PREFETCH_DISTANCE) as *const i8,
                _MM_HINT_T0,
            );
        }

        // Load 8 row indices
        let idx0 = *row_indices.add(base);
        let idx1 = *row_indices.add(base + 1);
        let idx2 = *row_indices.add(base + 2);
        let idx3 = *row_indices.add(base + 3);
        let idx4 = *row_indices.add(base + 4);
        let idx5 = *row_indices.add(base + 5);
        let idx6 = *row_indices.add(base + 6);
        let idx7 = *row_indices.add(base + 7);

        // Load 8 bin indices (gather from u8 array)
        let bin0 = *feature_bins.add(idx0) as usize;
        let bin1 = *feature_bins.add(idx1) as usize;
        let bin2 = *feature_bins.add(idx2) as usize;
        let bin3 = *feature_bins.add(idx3) as usize;
        let bin4 = *feature_bins.add(idx4) as usize;
        let bin5 = *feature_bins.add(idx5) as usize;
        let bin6 = *feature_bins.add(idx6) as usize;
        let bin7 = *feature_bins.add(idx7) as usize;

        // Create gather indices for gradients/hessians
        let indices = _mm256_set_epi32(
            idx7 as i32, idx6 as i32, idx5 as i32, idx4 as i32,
            idx3 as i32, idx2 as i32, idx1 as i32, idx0 as i32,
        );

        // Gather 8 gradients using AVX2 gather
        let grads = _mm256_i32gather_ps(gradients, indices, 4);

        // Gather 8 hessians using AVX2 gather
        let hess = _mm256_i32gather_ps(hessians, indices, 4);

        // Extract individual values for scatter (no SIMD scatter in AVX2)
        let grad_arr = std::mem::transmute::<__m256, [f32; 8]>(grads);
        let hess_arr = std::mem::transmute::<__m256, [f32; 8]>(hess);

        // Scatter to histogram bins (must be scalar due to conflicts)
        *hist_grads.add(bin0) += grad_arr[0];
        *hist_hess.add(bin0) += hess_arr[0];
        *hist_counts.add(bin0) += 1;

        *hist_grads.add(bin1) += grad_arr[1];
        *hist_hess.add(bin1) += hess_arr[1];
        *hist_counts.add(bin1) += 1;

        *hist_grads.add(bin2) += grad_arr[2];
        *hist_hess.add(bin2) += hess_arr[2];
        *hist_counts.add(bin2) += 1;

        *hist_grads.add(bin3) += grad_arr[3];
        *hist_hess.add(bin3) += hess_arr[3];
        *hist_counts.add(bin3) += 1;

        *hist_grads.add(bin4) += grad_arr[4];
        *hist_hess.add(bin4) += hess_arr[4];
        *hist_counts.add(bin4) += 1;

        *hist_grads.add(bin5) += grad_arr[5];
        *hist_hess.add(bin5) += hess_arr[5];
        *hist_counts.add(bin5) += 1;

        *hist_grads.add(bin6) += grad_arr[6];
        *hist_hess.add(bin6) += hess_arr[6];
        *hist_counts.add(bin6) += 1;

        *hist_grads.add(bin7) += grad_arr[7];
        *hist_hess.add(bin7) += hess_arr[7];
        *hist_counts.add(bin7) += 1;
    }

    // Handle remainder (scalar)
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

/// AVX2 histogram accumulation for contiguous rows
///
/// Optimized path when rows are 0..num_rows (no indirection needed).
/// Uses AVX2 loads directly since data is contiguous.
///
/// # Safety
/// - Requires AVX2 support
/// - All pointers must be valid and properly sized
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn histogram_accumulate_contiguous_avx2(
    feature_bins: *const u8,
    num_rows: usize,
    gradients: *const f32,
    hessians: *const f32,
    hist_grads: *mut f32,
    hist_hess: *mut f32,
    hist_counts: *mut u32,
) {
    const PREFETCH_DISTANCE: usize = 64;

    let chunks = num_rows / 8;
    let remainder = num_rows % 8;

    // Process 8 rows at a time
    for i in 0..chunks {
        let base = i * 8;

        // Prefetch upcoming data (contiguous, so more effective)
        if base + PREFETCH_DISTANCE < num_rows {
            _mm_prefetch(
                feature_bins.add(base + PREFETCH_DISTANCE) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                gradients.add(base + PREFETCH_DISTANCE) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                hessians.add(base + PREFETCH_DISTANCE) as *const i8,
                _MM_HINT_T0,
            );
        }

        // Load 8 bin indices (contiguous u8 access)
        let bin0 = *feature_bins.add(base) as usize;
        let bin1 = *feature_bins.add(base + 1) as usize;
        let bin2 = *feature_bins.add(base + 2) as usize;
        let bin3 = *feature_bins.add(base + 3) as usize;
        let bin4 = *feature_bins.add(base + 4) as usize;
        let bin5 = *feature_bins.add(base + 5) as usize;
        let bin6 = *feature_bins.add(base + 6) as usize;
        let bin7 = *feature_bins.add(base + 7) as usize;

        // Load 8 gradients (contiguous, use AVX2 load)
        let grads = _mm256_loadu_ps(gradients.add(base));

        // Load 8 hessians (contiguous, use AVX2 load)
        let hess = _mm256_loadu_ps(hessians.add(base));

        // Extract for scatter
        let grad_arr = std::mem::transmute::<__m256, [f32; 8]>(grads);
        let hess_arr = std::mem::transmute::<__m256, [f32; 8]>(hess);

        // Scatter to histogram bins
        *hist_grads.add(bin0) += grad_arr[0];
        *hist_hess.add(bin0) += hess_arr[0];
        *hist_counts.add(bin0) += 1;

        *hist_grads.add(bin1) += grad_arr[1];
        *hist_hess.add(bin1) += hess_arr[1];
        *hist_counts.add(bin1) += 1;

        *hist_grads.add(bin2) += grad_arr[2];
        *hist_hess.add(bin2) += hess_arr[2];
        *hist_counts.add(bin2) += 1;

        *hist_grads.add(bin3) += grad_arr[3];
        *hist_hess.add(bin3) += hess_arr[3];
        *hist_counts.add(bin3) += 1;

        *hist_grads.add(bin4) += grad_arr[4];
        *hist_hess.add(bin4) += hess_arr[4];
        *hist_counts.add(bin4) += 1;

        *hist_grads.add(bin5) += grad_arr[5];
        *hist_hess.add(bin5) += hess_arr[5];
        *hist_counts.add(bin5) += 1;

        *hist_grads.add(bin6) += grad_arr[6];
        *hist_hess.add(bin6) += hess_arr[6];
        *hist_counts.add(bin6) += 1;

        *hist_grads.add(bin7) += grad_arr[7];
        *hist_hess.add(bin7) += hess_arr[7];
        *hist_counts.add(bin7) += 1;
    }

    // Handle remainder (scalar)
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
    fn test_avx2_accumulate_indexed() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let feature_bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 3];
        let row_indices: Vec<usize> = (0..10).collect();
        let gradients: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let hessians: Vec<f32> = vec![1.0; 10];

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            histogram_accumulate_avx2(
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
        assert!((hist_grads[0] - 12.0).abs() < 1e-5, "Bin 0 grad mismatch: {}", hist_grads[0]);
        assert_eq!(hist_counts[0], 3);

        // Bin 1: rows 1, 4, 7 -> grads 2+5+8=15
        assert!((hist_grads[1] - 15.0).abs() < 1e-5, "Bin 1 grad mismatch: {}", hist_grads[1]);
        assert_eq!(hist_counts[1], 3);

        // Bin 2: rows 2, 5, 8 -> grads 3+6+9=18
        assert!((hist_grads[2] - 18.0).abs() < 1e-5, "Bin 2 grad mismatch: {}", hist_grads[2]);
        assert_eq!(hist_counts[2], 3);

        // Bin 3: row 9 -> grad 10
        assert!((hist_grads[3] - 10.0).abs() < 1e-5, "Bin 3 grad mismatch: {}", hist_grads[3]);
        assert_eq!(hist_counts[3], 1);
    }

    #[test]
    fn test_avx2_accumulate_contiguous() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let feature_bins: Vec<u8> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 3];
        let gradients: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let hessians: Vec<f32> = vec![1.0; 10];

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            histogram_accumulate_contiguous_avx2(
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

    #[test]
    fn test_avx2_large_dataset() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        // Test with 100k rows
        let num_rows = 100_000;
        let feature_bins: Vec<u8> = (0..num_rows).map(|i| (i % 256) as u8).collect();
        let row_indices: Vec<usize> = (0..num_rows).collect();
        let gradients: Vec<f32> = vec![1.0; num_rows];
        let hessians: Vec<f32> = vec![1.0; num_rows];

        let mut hist_grads = [0.0f32; 256];
        let mut hist_hess = [0.0f32; 256];
        let mut hist_counts = [0u32; 256];

        unsafe {
            histogram_accumulate_avx2(
                feature_bins.as_ptr(),
                row_indices.as_ptr(),
                num_rows,
                gradients.as_ptr(),
                hessians.as_ptr(),
                hist_grads.as_mut_ptr(),
                hist_hess.as_mut_ptr(),
                hist_counts.as_mut_ptr(),
            );
        }

        // Each bin should have ~390 or 391 rows (100000/256)
        let expected_per_bin = num_rows / 256;
        for bin in 0..256 {
            let count = hist_counts[bin];
            assert!(
                count >= expected_per_bin as u32 - 1 && count <= expected_per_bin as u32 + 1,
                "Bin {} has unexpected count: {}", bin, count
            );
        }
    }
}
