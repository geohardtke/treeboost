//! POC: Optimized Histogram Building Approaches
//!
//! Tests various optimizations against our current 8x unrolled baseline:
//! 1. Prefetching (bins + gradients/hessians)
//! 2. 16x unrolling
//! 3. Interleaved grad/hess layout
//! 4. Multi-feature batching (2-4 features sharing grad/hess loads)
//! 5. SIMD gradient loads with scalar scatter
//!
//! Run: cargo run --release --example tensor_tile_poc

#![allow(dead_code)]

use std::arch::x86_64::*;
use std::time::Instant;

const NUM_BINS: usize = 256;
const NUM_ROWS: usize = 100_000;
const NUM_FEATURES: usize = 50;
const ITERATIONS: usize = 20;

// Prefetch distance
const PF_DIST: usize = 32;

// =============================================================================
// BASELINE: Current 8x unrolled approach
// =============================================================================

fn histogram_8x_unrolled(
    bins: &[Vec<u8>],
    gradients: &[f32],
    hessians: &[f32],
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_features = bins.len();
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    for f in 0..num_features {
        let bin_col = &bins[f];
        let hist = &mut histograms[f];

        let mut row = 0;
        let unroll_end = (num_rows / 8) * 8;

        while row < unroll_end {
            unsafe {
                let b0 = *bin_col.get_unchecked(row) as usize;
                let b1 = *bin_col.get_unchecked(row + 1) as usize;
                let b2 = *bin_col.get_unchecked(row + 2) as usize;
                let b3 = *bin_col.get_unchecked(row + 3) as usize;
                let b4 = *bin_col.get_unchecked(row + 4) as usize;
                let b5 = *bin_col.get_unchecked(row + 5) as usize;
                let b6 = *bin_col.get_unchecked(row + 6) as usize;
                let b7 = *bin_col.get_unchecked(row + 7) as usize;

                let g0 = *gradients.get_unchecked(row);
                let g1 = *gradients.get_unchecked(row + 1);
                let g2 = *gradients.get_unchecked(row + 2);
                let g3 = *gradients.get_unchecked(row + 3);
                let g4 = *gradients.get_unchecked(row + 4);
                let g5 = *gradients.get_unchecked(row + 5);
                let g6 = *gradients.get_unchecked(row + 6);
                let g7 = *gradients.get_unchecked(row + 7);

                let h0 = *hessians.get_unchecked(row);
                let h1 = *hessians.get_unchecked(row + 1);
                let h2 = *hessians.get_unchecked(row + 2);
                let h3 = *hessians.get_unchecked(row + 3);
                let h4 = *hessians.get_unchecked(row + 4);
                let h5 = *hessians.get_unchecked(row + 5);
                let h6 = *hessians.get_unchecked(row + 6);
                let h7 = *hessians.get_unchecked(row + 7);

                let e0 = hist.get_unchecked_mut(b0);
                e0.0 += g0; e0.1 += h0; e0.2 += 1;
                let e1 = hist.get_unchecked_mut(b1);
                e1.0 += g1; e1.1 += h1; e1.2 += 1;
                let e2 = hist.get_unchecked_mut(b2);
                e2.0 += g2; e2.1 += h2; e2.2 += 1;
                let e3 = hist.get_unchecked_mut(b3);
                e3.0 += g3; e3.1 += h3; e3.2 += 1;
                let e4 = hist.get_unchecked_mut(b4);
                e4.0 += g4; e4.1 += h4; e4.2 += 1;
                let e5 = hist.get_unchecked_mut(b5);
                e5.0 += g5; e5.1 += h5; e5.2 += 1;
                let e6 = hist.get_unchecked_mut(b6);
                e6.0 += g6; e6.1 += h6; e6.2 += 1;
                let e7 = hist.get_unchecked_mut(b7);
                e7.0 += g7; e7.1 += h7; e7.2 += 1;
            }
            row += 8;
        }

        while row < num_rows {
            let bin = bin_col[row] as usize;
            hist[bin].0 += gradients[row];
            hist[bin].1 += hessians[row];
            hist[bin].2 += 1;
            row += 1;
        }
    }

    histograms
}

// =============================================================================
// OPTIMIZATION 1: 8x unrolled with prefetching
// =============================================================================

fn histogram_8x_prefetch(
    bins: &[Vec<u8>],
    gradients: &[f32],
    hessians: &[f32],
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_features = bins.len();
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    for f in 0..num_features {
        let bin_col = &bins[f];
        let hist = &mut histograms[f];

        let mut row = 0;
        let unroll_end = (num_rows / 8) * 8;

        while row < unroll_end {
            unsafe {
                // Prefetch future data
                if row + PF_DIST < num_rows {
                    _mm_prefetch(
                        bin_col.as_ptr().add(row + PF_DIST) as *const i8,
                        _MM_HINT_T0,
                    );
                    _mm_prefetch(
                        gradients.as_ptr().add(row + PF_DIST) as *const i8,
                        _MM_HINT_T0,
                    );
                    _mm_prefetch(
                        hessians.as_ptr().add(row + PF_DIST) as *const i8,
                        _MM_HINT_T0,
                    );
                }

                let b0 = *bin_col.get_unchecked(row) as usize;
                let b1 = *bin_col.get_unchecked(row + 1) as usize;
                let b2 = *bin_col.get_unchecked(row + 2) as usize;
                let b3 = *bin_col.get_unchecked(row + 3) as usize;
                let b4 = *bin_col.get_unchecked(row + 4) as usize;
                let b5 = *bin_col.get_unchecked(row + 5) as usize;
                let b6 = *bin_col.get_unchecked(row + 6) as usize;
                let b7 = *bin_col.get_unchecked(row + 7) as usize;

                // Prefetch histogram bins we'll access
                _mm_prefetch(hist.as_ptr().add(b0) as *const i8, _MM_HINT_T0);
                _mm_prefetch(hist.as_ptr().add(b4) as *const i8, _MM_HINT_T0);

                let g0 = *gradients.get_unchecked(row);
                let g1 = *gradients.get_unchecked(row + 1);
                let g2 = *gradients.get_unchecked(row + 2);
                let g3 = *gradients.get_unchecked(row + 3);
                let g4 = *gradients.get_unchecked(row + 4);
                let g5 = *gradients.get_unchecked(row + 5);
                let g6 = *gradients.get_unchecked(row + 6);
                let g7 = *gradients.get_unchecked(row + 7);

                let h0 = *hessians.get_unchecked(row);
                let h1 = *hessians.get_unchecked(row + 1);
                let h2 = *hessians.get_unchecked(row + 2);
                let h3 = *hessians.get_unchecked(row + 3);
                let h4 = *hessians.get_unchecked(row + 4);
                let h5 = *hessians.get_unchecked(row + 5);
                let h6 = *hessians.get_unchecked(row + 6);
                let h7 = *hessians.get_unchecked(row + 7);

                let e0 = hist.get_unchecked_mut(b0);
                e0.0 += g0; e0.1 += h0; e0.2 += 1;
                let e1 = hist.get_unchecked_mut(b1);
                e1.0 += g1; e1.1 += h1; e1.2 += 1;
                let e2 = hist.get_unchecked_mut(b2);
                e2.0 += g2; e2.1 += h2; e2.2 += 1;
                let e3 = hist.get_unchecked_mut(b3);
                e3.0 += g3; e3.1 += h3; e3.2 += 1;
                let e4 = hist.get_unchecked_mut(b4);
                e4.0 += g4; e4.1 += h4; e4.2 += 1;
                let e5 = hist.get_unchecked_mut(b5);
                e5.0 += g5; e5.1 += h5; e5.2 += 1;
                let e6 = hist.get_unchecked_mut(b6);
                e6.0 += g6; e6.1 += h6; e6.2 += 1;
                let e7 = hist.get_unchecked_mut(b7);
                e7.0 += g7; e7.1 += h7; e7.2 += 1;
            }
            row += 8;
        }

        while row < num_rows {
            let bin = bin_col[row] as usize;
            hist[bin].0 += gradients[row];
            hist[bin].1 += hessians[row];
            hist[bin].2 += 1;
            row += 1;
        }
    }

    histograms
}

// =============================================================================
// OPTIMIZATION 2: 16x unrolling
// =============================================================================

fn histogram_16x_unrolled(
    bins: &[Vec<u8>],
    gradients: &[f32],
    hessians: &[f32],
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_features = bins.len();
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    for f in 0..num_features {
        let bin_col = &bins[f];
        let hist = &mut histograms[f];

        let mut row = 0;
        let unroll_end = (num_rows / 16) * 16;

        while row < unroll_end {
            unsafe {
                // Load 16 bins
                let b0 = *bin_col.get_unchecked(row) as usize;
                let b1 = *bin_col.get_unchecked(row + 1) as usize;
                let b2 = *bin_col.get_unchecked(row + 2) as usize;
                let b3 = *bin_col.get_unchecked(row + 3) as usize;
                let b4 = *bin_col.get_unchecked(row + 4) as usize;
                let b5 = *bin_col.get_unchecked(row + 5) as usize;
                let b6 = *bin_col.get_unchecked(row + 6) as usize;
                let b7 = *bin_col.get_unchecked(row + 7) as usize;
                let b8 = *bin_col.get_unchecked(row + 8) as usize;
                let b9 = *bin_col.get_unchecked(row + 9) as usize;
                let b10 = *bin_col.get_unchecked(row + 10) as usize;
                let b11 = *bin_col.get_unchecked(row + 11) as usize;
                let b12 = *bin_col.get_unchecked(row + 12) as usize;
                let b13 = *bin_col.get_unchecked(row + 13) as usize;
                let b14 = *bin_col.get_unchecked(row + 14) as usize;
                let b15 = *bin_col.get_unchecked(row + 15) as usize;

                // Load 16 gradients
                let g0 = *gradients.get_unchecked(row);
                let g1 = *gradients.get_unchecked(row + 1);
                let g2 = *gradients.get_unchecked(row + 2);
                let g3 = *gradients.get_unchecked(row + 3);
                let g4 = *gradients.get_unchecked(row + 4);
                let g5 = *gradients.get_unchecked(row + 5);
                let g6 = *gradients.get_unchecked(row + 6);
                let g7 = *gradients.get_unchecked(row + 7);
                let g8 = *gradients.get_unchecked(row + 8);
                let g9 = *gradients.get_unchecked(row + 9);
                let g10 = *gradients.get_unchecked(row + 10);
                let g11 = *gradients.get_unchecked(row + 11);
                let g12 = *gradients.get_unchecked(row + 12);
                let g13 = *gradients.get_unchecked(row + 13);
                let g14 = *gradients.get_unchecked(row + 14);
                let g15 = *gradients.get_unchecked(row + 15);

                // Load 16 hessians
                let h0 = *hessians.get_unchecked(row);
                let h1 = *hessians.get_unchecked(row + 1);
                let h2 = *hessians.get_unchecked(row + 2);
                let h3 = *hessians.get_unchecked(row + 3);
                let h4 = *hessians.get_unchecked(row + 4);
                let h5 = *hessians.get_unchecked(row + 5);
                let h6 = *hessians.get_unchecked(row + 6);
                let h7 = *hessians.get_unchecked(row + 7);
                let h8 = *hessians.get_unchecked(row + 8);
                let h9 = *hessians.get_unchecked(row + 9);
                let h10 = *hessians.get_unchecked(row + 10);
                let h11 = *hessians.get_unchecked(row + 11);
                let h12 = *hessians.get_unchecked(row + 12);
                let h13 = *hessians.get_unchecked(row + 13);
                let h14 = *hessians.get_unchecked(row + 14);
                let h15 = *hessians.get_unchecked(row + 15);

                // Update 16 histogram entries
                let e = hist.get_unchecked_mut(b0); e.0 += g0; e.1 += h0; e.2 += 1;
                let e = hist.get_unchecked_mut(b1); e.0 += g1; e.1 += h1; e.2 += 1;
                let e = hist.get_unchecked_mut(b2); e.0 += g2; e.1 += h2; e.2 += 1;
                let e = hist.get_unchecked_mut(b3); e.0 += g3; e.1 += h3; e.2 += 1;
                let e = hist.get_unchecked_mut(b4); e.0 += g4; e.1 += h4; e.2 += 1;
                let e = hist.get_unchecked_mut(b5); e.0 += g5; e.1 += h5; e.2 += 1;
                let e = hist.get_unchecked_mut(b6); e.0 += g6; e.1 += h6; e.2 += 1;
                let e = hist.get_unchecked_mut(b7); e.0 += g7; e.1 += h7; e.2 += 1;
                let e = hist.get_unchecked_mut(b8); e.0 += g8; e.1 += h8; e.2 += 1;
                let e = hist.get_unchecked_mut(b9); e.0 += g9; e.1 += h9; e.2 += 1;
                let e = hist.get_unchecked_mut(b10); e.0 += g10; e.1 += h10; e.2 += 1;
                let e = hist.get_unchecked_mut(b11); e.0 += g11; e.1 += h11; e.2 += 1;
                let e = hist.get_unchecked_mut(b12); e.0 += g12; e.1 += h12; e.2 += 1;
                let e = hist.get_unchecked_mut(b13); e.0 += g13; e.1 += h13; e.2 += 1;
                let e = hist.get_unchecked_mut(b14); e.0 += g14; e.1 += h14; e.2 += 1;
                let e = hist.get_unchecked_mut(b15); e.0 += g15; e.1 += h15; e.2 += 1;
            }
            row += 16;
        }

        // Handle remainder with 8x then scalar
        let unroll8_end = row + ((num_rows - row) / 8) * 8;
        while row < unroll8_end {
            unsafe {
                let b0 = *bin_col.get_unchecked(row) as usize;
                let b1 = *bin_col.get_unchecked(row + 1) as usize;
                let b2 = *bin_col.get_unchecked(row + 2) as usize;
                let b3 = *bin_col.get_unchecked(row + 3) as usize;
                let b4 = *bin_col.get_unchecked(row + 4) as usize;
                let b5 = *bin_col.get_unchecked(row + 5) as usize;
                let b6 = *bin_col.get_unchecked(row + 6) as usize;
                let b7 = *bin_col.get_unchecked(row + 7) as usize;

                let g0 = *gradients.get_unchecked(row);
                let g1 = *gradients.get_unchecked(row + 1);
                let g2 = *gradients.get_unchecked(row + 2);
                let g3 = *gradients.get_unchecked(row + 3);
                let g4 = *gradients.get_unchecked(row + 4);
                let g5 = *gradients.get_unchecked(row + 5);
                let g6 = *gradients.get_unchecked(row + 6);
                let g7 = *gradients.get_unchecked(row + 7);

                let h0 = *hessians.get_unchecked(row);
                let h1 = *hessians.get_unchecked(row + 1);
                let h2 = *hessians.get_unchecked(row + 2);
                let h3 = *hessians.get_unchecked(row + 3);
                let h4 = *hessians.get_unchecked(row + 4);
                let h5 = *hessians.get_unchecked(row + 5);
                let h6 = *hessians.get_unchecked(row + 6);
                let h7 = *hessians.get_unchecked(row + 7);

                let e = hist.get_unchecked_mut(b0); e.0 += g0; e.1 += h0; e.2 += 1;
                let e = hist.get_unchecked_mut(b1); e.0 += g1; e.1 += h1; e.2 += 1;
                let e = hist.get_unchecked_mut(b2); e.0 += g2; e.1 += h2; e.2 += 1;
                let e = hist.get_unchecked_mut(b3); e.0 += g3; e.1 += h3; e.2 += 1;
                let e = hist.get_unchecked_mut(b4); e.0 += g4; e.1 += h4; e.2 += 1;
                let e = hist.get_unchecked_mut(b5); e.0 += g5; e.1 += h5; e.2 += 1;
                let e = hist.get_unchecked_mut(b6); e.0 += g6; e.1 += h6; e.2 += 1;
                let e = hist.get_unchecked_mut(b7); e.0 += g7; e.1 += h7; e.2 += 1;
            }
            row += 8;
        }

        while row < num_rows {
            let bin = bin_col[row] as usize;
            hist[bin].0 += gradients[row];
            hist[bin].1 += hessians[row];
            hist[bin].2 += 1;
            row += 1;
        }
    }

    histograms
}

// =============================================================================
// OPTIMIZATION 3: Interleaved grad/hess layout (better cache line usage)
// =============================================================================

fn histogram_interleaved(
    bins: &[Vec<u8>],
    grad_hess: &[(f32, f32)],  // Interleaved gradients and hessians
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_features = bins.len();
    let num_rows = grad_hess.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    for f in 0..num_features {
        let bin_col = &bins[f];
        let hist = &mut histograms[f];

        let mut row = 0;
        let unroll_end = (num_rows / 8) * 8;

        while row < unroll_end {
            unsafe {
                let b0 = *bin_col.get_unchecked(row) as usize;
                let b1 = *bin_col.get_unchecked(row + 1) as usize;
                let b2 = *bin_col.get_unchecked(row + 2) as usize;
                let b3 = *bin_col.get_unchecked(row + 3) as usize;
                let b4 = *bin_col.get_unchecked(row + 4) as usize;
                let b5 = *bin_col.get_unchecked(row + 5) as usize;
                let b6 = *bin_col.get_unchecked(row + 6) as usize;
                let b7 = *bin_col.get_unchecked(row + 7) as usize;

                let gh0 = *grad_hess.get_unchecked(row);
                let gh1 = *grad_hess.get_unchecked(row + 1);
                let gh2 = *grad_hess.get_unchecked(row + 2);
                let gh3 = *grad_hess.get_unchecked(row + 3);
                let gh4 = *grad_hess.get_unchecked(row + 4);
                let gh5 = *grad_hess.get_unchecked(row + 5);
                let gh6 = *grad_hess.get_unchecked(row + 6);
                let gh7 = *grad_hess.get_unchecked(row + 7);

                let e = hist.get_unchecked_mut(b0); e.0 += gh0.0; e.1 += gh0.1; e.2 += 1;
                let e = hist.get_unchecked_mut(b1); e.0 += gh1.0; e.1 += gh1.1; e.2 += 1;
                let e = hist.get_unchecked_mut(b2); e.0 += gh2.0; e.1 += gh2.1; e.2 += 1;
                let e = hist.get_unchecked_mut(b3); e.0 += gh3.0; e.1 += gh3.1; e.2 += 1;
                let e = hist.get_unchecked_mut(b4); e.0 += gh4.0; e.1 += gh4.1; e.2 += 1;
                let e = hist.get_unchecked_mut(b5); e.0 += gh5.0; e.1 += gh5.1; e.2 += 1;
                let e = hist.get_unchecked_mut(b6); e.0 += gh6.0; e.1 += gh6.1; e.2 += 1;
                let e = hist.get_unchecked_mut(b7); e.0 += gh7.0; e.1 += gh7.1; e.2 += 1;
            }
            row += 8;
        }

        while row < num_rows {
            let bin = bin_col[row] as usize;
            let gh = grad_hess[row];
            hist[bin].0 += gh.0;
            hist[bin].1 += gh.1;
            hist[bin].2 += 1;
            row += 1;
        }
    }

    histograms
}

// =============================================================================
// OPTIMIZATION 4: Multi-feature batching (process 2 features sharing grad loads)
// Uses split_at_mut to safely borrow two histograms
// =============================================================================

fn histogram_multi_feature_2(
    bins: &[Vec<u8>],
    gradients: &[f32],
    hessians: &[f32],
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_features = bins.len();
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    // Process 2 features at a time, sharing gradient/hessian loads
    let mut f = 0;
    while f + 1 < num_features {
        let bin_col0 = &bins[f];
        let bin_col1 = &bins[f + 1];

        // Split histograms to borrow two at once
        let (left, right) = histograms.split_at_mut(f + 1);
        let hist0 = &mut left[f];
        let hist1 = &mut right[0];

        let mut row = 0;
        let unroll_end = (num_rows / 8) * 8;

        while row < unroll_end {
            unsafe {
                // Load gradients and hessians ONCE for both features
                let g0 = *gradients.get_unchecked(row);
                let g1 = *gradients.get_unchecked(row + 1);
                let g2 = *gradients.get_unchecked(row + 2);
                let g3 = *gradients.get_unchecked(row + 3);
                let g4 = *gradients.get_unchecked(row + 4);
                let g5 = *gradients.get_unchecked(row + 5);
                let g6 = *gradients.get_unchecked(row + 6);
                let g7 = *gradients.get_unchecked(row + 7);

                let h0 = *hessians.get_unchecked(row);
                let h1 = *hessians.get_unchecked(row + 1);
                let h2 = *hessians.get_unchecked(row + 2);
                let h3 = *hessians.get_unchecked(row + 3);
                let h4 = *hessians.get_unchecked(row + 4);
                let h5 = *hessians.get_unchecked(row + 5);
                let h6 = *hessians.get_unchecked(row + 6);
                let h7 = *hessians.get_unchecked(row + 7);

                // Feature 0
                let b00 = *bin_col0.get_unchecked(row) as usize;
                let b01 = *bin_col0.get_unchecked(row + 1) as usize;
                let b02 = *bin_col0.get_unchecked(row + 2) as usize;
                let b03 = *bin_col0.get_unchecked(row + 3) as usize;
                let b04 = *bin_col0.get_unchecked(row + 4) as usize;
                let b05 = *bin_col0.get_unchecked(row + 5) as usize;
                let b06 = *bin_col0.get_unchecked(row + 6) as usize;
                let b07 = *bin_col0.get_unchecked(row + 7) as usize;

                let e = hist0.get_unchecked_mut(b00); e.0 += g0; e.1 += h0; e.2 += 1;
                let e = hist0.get_unchecked_mut(b01); e.0 += g1; e.1 += h1; e.2 += 1;
                let e = hist0.get_unchecked_mut(b02); e.0 += g2; e.1 += h2; e.2 += 1;
                let e = hist0.get_unchecked_mut(b03); e.0 += g3; e.1 += h3; e.2 += 1;
                let e = hist0.get_unchecked_mut(b04); e.0 += g4; e.1 += h4; e.2 += 1;
                let e = hist0.get_unchecked_mut(b05); e.0 += g5; e.1 += h5; e.2 += 1;
                let e = hist0.get_unchecked_mut(b06); e.0 += g6; e.1 += h6; e.2 += 1;
                let e = hist0.get_unchecked_mut(b07); e.0 += g7; e.1 += h7; e.2 += 1;

                // Feature 1 (reuse g0-g7, h0-h7)
                let b10 = *bin_col1.get_unchecked(row) as usize;
                let b11 = *bin_col1.get_unchecked(row + 1) as usize;
                let b12 = *bin_col1.get_unchecked(row + 2) as usize;
                let b13 = *bin_col1.get_unchecked(row + 3) as usize;
                let b14 = *bin_col1.get_unchecked(row + 4) as usize;
                let b15 = *bin_col1.get_unchecked(row + 5) as usize;
                let b16 = *bin_col1.get_unchecked(row + 6) as usize;
                let b17 = *bin_col1.get_unchecked(row + 7) as usize;

                let e = hist1.get_unchecked_mut(b10); e.0 += g0; e.1 += h0; e.2 += 1;
                let e = hist1.get_unchecked_mut(b11); e.0 += g1; e.1 += h1; e.2 += 1;
                let e = hist1.get_unchecked_mut(b12); e.0 += g2; e.1 += h2; e.2 += 1;
                let e = hist1.get_unchecked_mut(b13); e.0 += g3; e.1 += h3; e.2 += 1;
                let e = hist1.get_unchecked_mut(b14); e.0 += g4; e.1 += h4; e.2 += 1;
                let e = hist1.get_unchecked_mut(b15); e.0 += g5; e.1 += h5; e.2 += 1;
                let e = hist1.get_unchecked_mut(b16); e.0 += g6; e.1 += h6; e.2 += 1;
                let e = hist1.get_unchecked_mut(b17); e.0 += g7; e.1 += h7; e.2 += 1;
            }
            row += 8;
        }

        // Handle remainder
        while row < num_rows {
            let g = gradients[row];
            let h = hessians[row];

            let b0 = bin_col0[row] as usize;
            hist0[b0].0 += g;
            hist0[b0].1 += h;
            hist0[b0].2 += 1;

            let b1 = bin_col1[row] as usize;
            hist1[b1].0 += g;
            hist1[b1].1 += h;
            hist1[b1].2 += 1;

            row += 1;
        }

        f += 2;
    }

    // Handle odd feature count
    if f < num_features {
        let bin_col = &bins[f];
        let hist = &mut histograms[f];

        for row in 0..num_rows {
            let bin = bin_col[row] as usize;
            hist[bin].0 += gradients[row];
            hist[bin].1 += hessians[row];
            hist[bin].2 += 1;
        }
    }

    histograms
}

// =============================================================================
// OPTIMIZATION 5: Multi-feature batching (4 features)
// =============================================================================

fn histogram_multi_feature_4(
    bins: &[Vec<u8>],
    gradients: &[f32],
    hessians: &[f32],
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_features = bins.len();
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    // Process 4 features at a time
    let mut f = 0;
    while f + 3 < num_features {
        let bin_cols: [&Vec<u8>; 4] = [&bins[f], &bins[f+1], &bins[f+2], &bins[f+3]];

        let mut row = 0;
        let unroll_end = (num_rows / 4) * 4;

        while row < unroll_end {
            unsafe {
                // Load 4 gradients and hessians ONCE
                let g0 = *gradients.get_unchecked(row);
                let g1 = *gradients.get_unchecked(row + 1);
                let g2 = *gradients.get_unchecked(row + 2);
                let g3 = *gradients.get_unchecked(row + 3);

                let h0 = *hessians.get_unchecked(row);
                let h1 = *hessians.get_unchecked(row + 1);
                let h2 = *hessians.get_unchecked(row + 2);
                let h3 = *hessians.get_unchecked(row + 3);

                // Update all 4 features with same grad/hess values
                for (fi, bin_col) in bin_cols.iter().enumerate() {
                    let hist = &mut histograms[f + fi];

                    let b0 = *bin_col.get_unchecked(row) as usize;
                    let b1 = *bin_col.get_unchecked(row + 1) as usize;
                    let b2 = *bin_col.get_unchecked(row + 2) as usize;
                    let b3 = *bin_col.get_unchecked(row + 3) as usize;

                    let e = hist.get_unchecked_mut(b0); e.0 += g0; e.1 += h0; e.2 += 1;
                    let e = hist.get_unchecked_mut(b1); e.0 += g1; e.1 += h1; e.2 += 1;
                    let e = hist.get_unchecked_mut(b2); e.0 += g2; e.1 += h2; e.2 += 1;
                    let e = hist.get_unchecked_mut(b3); e.0 += g3; e.1 += h3; e.2 += 1;
                }
            }
            row += 4;
        }

        // Handle remainder
        while row < num_rows {
            let g = gradients[row];
            let h = hessians[row];
            for (fi, bin_col) in bin_cols.iter().enumerate() {
                let bin = bin_col[row] as usize;
                let hist = &mut histograms[f + fi];
                hist[bin].0 += g;
                hist[bin].1 += h;
                hist[bin].2 += 1;
            }
            row += 1;
        }

        f += 4;
    }

    // Handle remaining features with single-feature approach
    while f < num_features {
        let bin_col = &bins[f];
        let hist = &mut histograms[f];

        for row in 0..num_rows {
            let bin = bin_col[row] as usize;
            hist[bin].0 += gradients[row];
            hist[bin].1 += hessians[row];
            hist[bin].2 += 1;
        }
        f += 1;
    }

    histograms
}

// =============================================================================
// OPTIMIZATION 6: AVX2 SIMD gradient loads with scalar scatter
// =============================================================================

#[target_feature(enable = "avx2")]
unsafe fn histogram_avx2_loads(
    bins: &[Vec<u8>],
    gradients: &[f32],
    hessians: &[f32],
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_features = bins.len();
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    for f in 0..num_features {
        let bin_col = &bins[f];
        let hist = &mut histograms[f];

        let mut row = 0;
        let unroll_end = (num_rows / 8) * 8;

        while row < unroll_end {
            // SIMD load 8 gradients and hessians
            let grads = _mm256_loadu_ps(gradients.as_ptr().add(row));
            let hesss = _mm256_loadu_ps(hessians.as_ptr().add(row));

            // Extract to scalar for histogram update (scatter is random)
            let mut g_arr = [0.0f32; 8];
            let mut h_arr = [0.0f32; 8];
            _mm256_storeu_ps(g_arr.as_mut_ptr(), grads);
            _mm256_storeu_ps(h_arr.as_mut_ptr(), hesss);

            // Scalar histogram updates (random scatter)
            let b0 = *bin_col.get_unchecked(row) as usize;
            let b1 = *bin_col.get_unchecked(row + 1) as usize;
            let b2 = *bin_col.get_unchecked(row + 2) as usize;
            let b3 = *bin_col.get_unchecked(row + 3) as usize;
            let b4 = *bin_col.get_unchecked(row + 4) as usize;
            let b5 = *bin_col.get_unchecked(row + 5) as usize;
            let b6 = *bin_col.get_unchecked(row + 6) as usize;
            let b7 = *bin_col.get_unchecked(row + 7) as usize;

            let e = hist.get_unchecked_mut(b0); e.0 += g_arr[0]; e.1 += h_arr[0]; e.2 += 1;
            let e = hist.get_unchecked_mut(b1); e.0 += g_arr[1]; e.1 += h_arr[1]; e.2 += 1;
            let e = hist.get_unchecked_mut(b2); e.0 += g_arr[2]; e.1 += h_arr[2]; e.2 += 1;
            let e = hist.get_unchecked_mut(b3); e.0 += g_arr[3]; e.1 += h_arr[3]; e.2 += 1;
            let e = hist.get_unchecked_mut(b4); e.0 += g_arr[4]; e.1 += h_arr[4]; e.2 += 1;
            let e = hist.get_unchecked_mut(b5); e.0 += g_arr[5]; e.1 += h_arr[5]; e.2 += 1;
            let e = hist.get_unchecked_mut(b6); e.0 += g_arr[6]; e.1 += h_arr[6]; e.2 += 1;
            let e = hist.get_unchecked_mut(b7); e.0 += g_arr[7]; e.1 += h_arr[7]; e.2 += 1;

            row += 8;
        }

        while row < num_rows {
            let bin = bin_col[row] as usize;
            hist[bin].0 += gradients[row];
            hist[bin].1 += hessians[row];
            hist[bin].2 += 1;
            row += 1;
        }
    }

    histograms
}

// =============================================================================
// MAIN: Run all benchmarks
// =============================================================================

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("          HISTOGRAM OPTIMIZATION POC");
    println!("═══════════════════════════════════════════════════════════════════════════════\n");

    println!("Configuration:");
    println!("  Rows:       {:>10}", NUM_ROWS);
    println!("  Features:   {:>10}", NUM_FEATURES);
    println!("  Bins:       {:>10}", NUM_BINS);
    println!("  Iterations: {:>10}", ITERATIONS);

    // Generate test data
    println!("\nGenerating test data...");
    let bins: Vec<Vec<u8>> = (0..NUM_FEATURES)
        .map(|f| {
            (0..NUM_ROWS)
                .map(|r| ((r * (f + 1) * 17) % 256) as u8)
                .collect()
        })
        .collect();

    let gradients: Vec<f32> = (0..NUM_ROWS)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let hessians: Vec<f32> = (0..NUM_ROWS)
        .map(|i| 1.0 + (i as f32 * 0.005).cos() * 0.1)
        .collect();

    let grad_hess: Vec<(f32, f32)> = gradients.iter()
        .zip(hessians.iter())
        .map(|(&g, &h)| (g, h))
        .collect();

    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("BENCHMARKS");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    // Warmup
    let _ = histogram_8x_unrolled(&bins, &gradients, &hessians);

    // Benchmark each approach
    let mut results: Vec<(&str, f64)> = Vec::new();

    // 1. Baseline: 8x unrolled
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = histogram_8x_unrolled(&bins, &gradients, &hessians);
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("8x unrolled (baseline)", time));

    // 2. 8x + prefetch
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = histogram_8x_prefetch(&bins, &gradients, &hessians);
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("8x + prefetch", time));

    // 3. 16x unrolled
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = histogram_16x_unrolled(&bins, &gradients, &hessians);
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("16x unrolled", time));

    // 4. Interleaved grad/hess
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = histogram_interleaved(&bins, &grad_hess);
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("Interleaved grad/hess", time));

    // 5. Multi-feature (2)
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = histogram_multi_feature_2(&bins, &gradients, &hessians);
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("Multi-feature (2)", time));

    // 6. Multi-feature (4)
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = histogram_multi_feature_4(&bins, &gradients, &hessians);
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("Multi-feature (4)", time));

    // 7. AVX2 loads
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe {
            let _ = histogram_avx2_loads(&bins, &gradients, &hessians);
        }
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("AVX2 SIMD loads", time));

    // Print results
    let baseline = results[0].1;
    println!("{:<25} {:>10} {:>10}", "Approach", "Time (ms)", "Speedup");
    println!("{}", "─".repeat(50));

    for (name, time) in &results {
        let speedup = baseline / time;
        let marker = if speedup > 1.05 { "✓" } else if speedup < 0.95 { "✗" } else { " " };
        println!("{:<25} {:>10.3} {:>9.2}x {}", name, time, speedup, marker);
    }

    // Find best
    let best = results.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();

    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("BEST APPROACH: {} ({:.3} ms)", best.0, best.1);
    println!("───────────────────────────────────────────────────────────────────────────────");

    if best.0 == "8x unrolled (baseline)" {
        println!("\nCurrent baseline is already optimal!");
        println!("None of the tested optimizations improve performance.");
    } else {
        println!("\nImprovement: {:.1}% faster than baseline",
                 (baseline - best.1) / baseline * 100.0);
    }
}
