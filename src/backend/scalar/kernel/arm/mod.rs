//! ARM aarch64 NEON kernel implementations
//!
//! This module contains NEON optimized implementations for GBDT operations.
//!
//! NEON is always available on aarch64, providing 128-bit SIMD (4 x f32).

use super::fallback::SplitCandidate;

/// Block size for cache-blocked histogram building
pub const BLOCK_SIZE: usize = 2048;

// Re-export split finding from fallback (NEON doesn't provide significant gains for split finding)
pub use super::fallback::{find_best_split_scalar, find_best_split_simd};

/// NEON copy of gradients and hessians to interleaved cache.
///
/// Uses NEON `vzipq_f32` to efficiently interleave 4 gradient/hessian pairs
/// at a time: `[g0,g1,g2,g3]` + `[h0,h1,h2,h3]` -> `[(g0,h0),(g1,h1),(g2,h2),(g3,h3)]`
///
/// # Safety
/// - `gradients` and `hessians` must have at least `start + len` elements
/// - `gh_cache` must have capacity for `len` elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn copy_gh_interleaved_neon(
    gradients: &[f32],
    hessians: &[f32],
    start: usize,
    len: usize,
    gh_cache: &mut [(f32, f32); BLOCK_SIZE],
) {
    use std::arch::aarch64::*;

    let chunks = len / 4;
    let remainder = len % 4;

    let grad_ptr = gradients.as_ptr().add(start);
    let hess_ptr = hessians.as_ptr().add(start);
    let cache_ptr = gh_cache.as_mut_ptr() as *mut f32;

    for i in 0..chunks {
        let offset = i * 4;

        // Load 4 gradients: [g0, g1, g2, g3]
        let grads = vld1q_f32(grad_ptr.add(offset));
        // Load 4 hessians: [h0, h1, h2, h3]
        let hess = vld1q_f32(hess_ptr.add(offset));

        // Interleave using vzip: produces [g0, h0, g1, h1] and [g2, h2, g3, h3]
        let interleaved = vzipq_f32(grads, hess);

        // Store interleaved pairs (8 floats = 4 pairs)
        let dst = cache_ptr.add(offset * 2);
        vst1q_f32(dst, interleaved.0);
        vst1q_f32(dst.add(4), interleaved.1);
    }

    // Handle remainder with scalar code
    let rem_start = chunks * 4;
    for i in 0..remainder {
        let idx = rem_start + i;
        let g = *gradients.get_unchecked(start + idx);
        let h = *hessians.get_unchecked(start + idx);
        *gh_cache.get_unchecked_mut(idx) = (g, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_gh_interleaved_neon() {
        let gradients: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let hessians: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let mut gh_cache = [(0.0f32, 0.0f32); BLOCK_SIZE];

        unsafe {
            copy_gh_interleaved_neon(&gradients, &hessians, 0, 32, &mut gh_cache);
        }

        // Verify interleaving is correct
        for i in 0..32 {
            let expected_g = i as f32 * 0.1;
            let expected_h = i as f32 * 0.01;
            let (actual_g, actual_h) = gh_cache[i];

            assert!(
                (actual_g - expected_g).abs() < 1e-6,
                "Gradient mismatch at {}: expected {}, got {}",
                i,
                expected_g,
                actual_g
            );
            assert!(
                (actual_h - expected_h).abs() < 1e-6,
                "Hessian mismatch at {}: expected {}, got {}",
                i,
                expected_h,
                actual_h
            );
        }
    }

    #[test]
    fn test_copy_gh_interleaved_neon_with_remainder() {
        // Test with non-multiple-of-4 length
        let gradients: Vec<f32> = (0..35).map(|i| i as f32).collect();
        let hessians: Vec<f32> = (0..35).map(|i| (i as f32) + 100.0).collect();
        let mut gh_cache = [(0.0f32, 0.0f32); BLOCK_SIZE];

        unsafe {
            copy_gh_interleaved_neon(&gradients, &hessians, 0, 35, &mut gh_cache);
        }

        // Verify all values including remainder
        for i in 0..35 {
            let (g, h) = gh_cache[i];
            assert_eq!(g, i as f32, "Gradient mismatch at {}", i);
            assert_eq!(h, i as f32 + 100.0, "Hessian mismatch at {}", i);
        }
    }
}
