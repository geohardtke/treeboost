//! POC: Optimized Tensor-Tile Histogram Approaches
//!
//! Goal: Find the best 2D tensor-tile implementation that could beat scalar
//! for modern hardware (AVX2/AVX-512, GPU-style execution)
//!
//! Optimizations tested:
//! 1. Tile size tuning (2, 4, 8 features per tile)
//! 2. SIMD horizontal processing (process N features with one SIMD op)
//! 3. Register-blocked tiles (keep histogram sums in registers)
//! 4. Transposed bin layout (row-major for vectorization)
//! 5. Fused tile processing (load once, scatter to N histograms)
//!
//! Run: cargo run --release --example tensor_tile_poc

#![allow(dead_code)]

use std::arch::x86_64::*;
use std::time::Instant;

const NUM_BINS: usize = 256;
const NUM_ROWS: usize = 100_000;
const NUM_FEATURES: usize = 50;
const ITERATIONS: usize = 20;

// =============================================================================
// BASELINE: Current scalar approach (for comparison)
// =============================================================================

fn histogram_scalar_baseline(
    bins: &[Vec<u8>],      // [feature][row]
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
// 2D-TILE OPT 1: Row-major bin layout (better for vectorized feature access)
// Instead of bins[feature][row], use bins_transposed[row][feature]
// =============================================================================

fn histogram_tile_transposed(
    bins_transposed: &[Vec<u8>],  // [row][feature] - transposed layout
    gradients: &[f32],
    hessians: &[f32],
    num_features: usize,
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    // Process rows, accessing all features contiguously
    for row in 0..num_rows {
        let g = gradients[row];
        let h = hessians[row];
        let row_bins = &bins_transposed[row];

        // All features for this row are contiguous in memory
        for f in 0..num_features {
            let bin = row_bins[f] as usize;
            histograms[f][bin].0 += g;
            histograms[f][bin].1 += h;
            histograms[f][bin].2 += 1;
        }
    }

    histograms
}

// =============================================================================
// 2D-TILE OPT 2: Transposed + 8-feature SIMD tile
// Process 8 features at once using AVX2, scatter to 8 histograms
// =============================================================================

#[target_feature(enable = "avx2")]
unsafe fn histogram_tile_simd_8feat(
    bins_transposed: &[Vec<u8>],  // [row][feature]
    gradients: &[f32],
    hessians: &[f32],
    num_features: usize,
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    // Process in tiles of 8 features
    for feat_start in (0..num_features).step_by(8) {
        let feat_end = (feat_start + 8).min(num_features);
        let tile_width = feat_end - feat_start;

        if tile_width == 8 {
            // Full tile: use SIMD to load 8 bins at once
            for row in 0..num_rows {
                let g = *gradients.get_unchecked(row);
                let h = *hessians.get_unchecked(row);

                // Load 8 bins at once (u8 -> broadcast to registers)
                let bin_ptr = bins_transposed[row].as_ptr().add(feat_start);

                // Extract bins and update histograms
                let b0 = *bin_ptr as usize;
                let b1 = *bin_ptr.add(1) as usize;
                let b2 = *bin_ptr.add(2) as usize;
                let b3 = *bin_ptr.add(3) as usize;
                let b4 = *bin_ptr.add(4) as usize;
                let b5 = *bin_ptr.add(5) as usize;
                let b6 = *bin_ptr.add(6) as usize;
                let b7 = *bin_ptr.add(7) as usize;

                // Update 8 histograms with same g, h
                let e = histograms[feat_start].get_unchecked_mut(b0);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms[feat_start + 1].get_unchecked_mut(b1);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms[feat_start + 2].get_unchecked_mut(b2);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms[feat_start + 3].get_unchecked_mut(b3);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms[feat_start + 4].get_unchecked_mut(b4);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms[feat_start + 5].get_unchecked_mut(b5);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms[feat_start + 6].get_unchecked_mut(b6);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms[feat_start + 7].get_unchecked_mut(b7);
                e.0 += g; e.1 += h; e.2 += 1;
            }
        } else {
            // Partial tile: scalar fallback
            for row in 0..num_rows {
                let g = gradients[row];
                let h = hessians[row];
                for f in feat_start..feat_end {
                    let bin = bins_transposed[row][f] as usize;
                    histograms[f][bin].0 += g;
                    histograms[f][bin].1 += h;
                    histograms[f][bin].2 += 1;
                }
            }
        }
    }

    histograms
}

// =============================================================================
// 2D-TILE OPT 3: Transposed + 4-row batching + 8-feature tile
// Process 4 rows × 8 features per iteration
// =============================================================================

#[target_feature(enable = "avx2")]
unsafe fn histogram_tile_4row_8feat(
    bins_transposed: &[Vec<u8>],
    gradients: &[f32],
    hessians: &[f32],
    num_features: usize,
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    // Process in tiles of 8 features
    for feat_start in (0..num_features).step_by(8) {
        let feat_end = (feat_start + 8).min(num_features);
        let tile_width = feat_end - feat_start;

        if tile_width == 8 {
            // Process 4 rows at a time
            let mut row = 0;
            let row_end = (num_rows / 4) * 4;

            while row < row_end {
                // Load 4 gradients and hessians
                let g0 = *gradients.get_unchecked(row);
                let g1 = *gradients.get_unchecked(row + 1);
                let g2 = *gradients.get_unchecked(row + 2);
                let g3 = *gradients.get_unchecked(row + 3);

                let h0 = *hessians.get_unchecked(row);
                let h1 = *hessians.get_unchecked(row + 1);
                let h2 = *hessians.get_unchecked(row + 2);
                let h3 = *hessians.get_unchecked(row + 3);

                // Process 8 features for each of 4 rows
                for fi in 0..8 {
                    let f = feat_start + fi;
                    let hist = &mut histograms[f];

                    let b0 = bins_transposed[row][f] as usize;
                    let b1 = bins_transposed[row + 1][f] as usize;
                    let b2 = bins_transposed[row + 2][f] as usize;
                    let b3 = bins_transposed[row + 3][f] as usize;

                    let e = hist.get_unchecked_mut(b0); e.0 += g0; e.1 += h0; e.2 += 1;
                    let e = hist.get_unchecked_mut(b1); e.0 += g1; e.1 += h1; e.2 += 1;
                    let e = hist.get_unchecked_mut(b2); e.0 += g2; e.1 += h2; e.2 += 1;
                    let e = hist.get_unchecked_mut(b3); e.0 += g3; e.1 += h3; e.2 += 1;
                }

                row += 4;
            }

            // Handle remaining rows
            while row < num_rows {
                let g = gradients[row];
                let h = hessians[row];
                for f in feat_start..feat_end {
                    let bin = bins_transposed[row][f] as usize;
                    histograms[f][bin].0 += g;
                    histograms[f][bin].1 += h;
                    histograms[f][bin].2 += 1;
                }
                row += 1;
            }
        } else {
            for row in 0..num_rows {
                let g = gradients[row];
                let h = hessians[row];
                for f in feat_start..feat_end {
                    let bin = bins_transposed[row][f] as usize;
                    histograms[f][bin].0 += g;
                    histograms[f][bin].1 += h;
                    histograms[f][bin].2 += 1;
                }
            }
        }
    }

    histograms
}

// =============================================================================
// 2D-TILE OPT 4: Register-blocked histograms
// Keep partial histogram sums in registers, flush periodically
// =============================================================================

fn histogram_tile_register_blocked(
    bins_transposed: &[Vec<u8>],
    gradients: &[f32],
    hessians: &[f32],
    num_features: usize,
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    // Process 2 features at a time (2 × 256 × 12 = 6KB fits in L1)
    for feat_start in (0..num_features).step_by(2) {
        let feat_end = (feat_start + 2).min(num_features);

        if feat_end - feat_start == 2 {
            let hist0 = histograms[feat_start].as_mut_ptr();
            let hist1 = histograms[feat_start + 1].as_mut_ptr();

            let mut row = 0;
            let unroll_end = (num_rows / 8) * 8;

            while row < unroll_end {
                unsafe {
                    // Load 8 grad/hess pairs
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

                    // Feature 0 bins
                    let b00 = *bins_transposed.get_unchecked(row).get_unchecked(feat_start) as usize;
                    let b01 = *bins_transposed.get_unchecked(row + 1).get_unchecked(feat_start) as usize;
                    let b02 = *bins_transposed.get_unchecked(row + 2).get_unchecked(feat_start) as usize;
                    let b03 = *bins_transposed.get_unchecked(row + 3).get_unchecked(feat_start) as usize;
                    let b04 = *bins_transposed.get_unchecked(row + 4).get_unchecked(feat_start) as usize;
                    let b05 = *bins_transposed.get_unchecked(row + 5).get_unchecked(feat_start) as usize;
                    let b06 = *bins_transposed.get_unchecked(row + 6).get_unchecked(feat_start) as usize;
                    let b07 = *bins_transposed.get_unchecked(row + 7).get_unchecked(feat_start) as usize;

                    // Update feature 0
                    let e = &mut *hist0.add(b00); e.0 += g0; e.1 += h0; e.2 += 1;
                    let e = &mut *hist0.add(b01); e.0 += g1; e.1 += h1; e.2 += 1;
                    let e = &mut *hist0.add(b02); e.0 += g2; e.1 += h2; e.2 += 1;
                    let e = &mut *hist0.add(b03); e.0 += g3; e.1 += h3; e.2 += 1;
                    let e = &mut *hist0.add(b04); e.0 += g4; e.1 += h4; e.2 += 1;
                    let e = &mut *hist0.add(b05); e.0 += g5; e.1 += h5; e.2 += 1;
                    let e = &mut *hist0.add(b06); e.0 += g6; e.1 += h6; e.2 += 1;
                    let e = &mut *hist0.add(b07); e.0 += g7; e.1 += h7; e.2 += 1;

                    // Feature 1 bins
                    let b10 = *bins_transposed.get_unchecked(row).get_unchecked(feat_start + 1) as usize;
                    let b11 = *bins_transposed.get_unchecked(row + 1).get_unchecked(feat_start + 1) as usize;
                    let b12 = *bins_transposed.get_unchecked(row + 2).get_unchecked(feat_start + 1) as usize;
                    let b13 = *bins_transposed.get_unchecked(row + 3).get_unchecked(feat_start + 1) as usize;
                    let b14 = *bins_transposed.get_unchecked(row + 4).get_unchecked(feat_start + 1) as usize;
                    let b15 = *bins_transposed.get_unchecked(row + 5).get_unchecked(feat_start + 1) as usize;
                    let b16 = *bins_transposed.get_unchecked(row + 6).get_unchecked(feat_start + 1) as usize;
                    let b17 = *bins_transposed.get_unchecked(row + 7).get_unchecked(feat_start + 1) as usize;

                    // Update feature 1
                    let e = &mut *hist1.add(b10); e.0 += g0; e.1 += h0; e.2 += 1;
                    let e = &mut *hist1.add(b11); e.0 += g1; e.1 += h1; e.2 += 1;
                    let e = &mut *hist1.add(b12); e.0 += g2; e.1 += h2; e.2 += 1;
                    let e = &mut *hist1.add(b13); e.0 += g3; e.1 += h3; e.2 += 1;
                    let e = &mut *hist1.add(b14); e.0 += g4; e.1 += h4; e.2 += 1;
                    let e = &mut *hist1.add(b15); e.0 += g5; e.1 += h5; e.2 += 1;
                    let e = &mut *hist1.add(b16); e.0 += g6; e.1 += h6; e.2 += 1;
                    let e = &mut *hist1.add(b17); e.0 += g7; e.1 += h7; e.2 += 1;
                }
                row += 8;
            }

            // Remainder
            while row < num_rows {
                let g = gradients[row];
                let h = hessians[row];
                for f in feat_start..feat_end {
                    let bin = bins_transposed[row][f] as usize;
                    histograms[f][bin].0 += g;
                    histograms[f][bin].1 += h;
                    histograms[f][bin].2 += 1;
                }
                row += 1;
            }
        } else {
            // Single feature
            for row in 0..num_rows {
                let g = gradients[row];
                let h = hessians[row];
                let bin = bins_transposed[row][feat_start] as usize;
                histograms[feat_start][bin].0 += g;
                histograms[feat_start][bin].1 += h;
                histograms[feat_start][bin].2 += 1;
            }
        }
    }

    histograms
}

// =============================================================================
// 2D-TILE OPT 5: Flat bin array (bins as [row * num_features + feature])
// Enables SIMD gather for bins
// =============================================================================

#[target_feature(enable = "avx2")]
unsafe fn histogram_tile_flat_avx2(
    bins_flat: &[u8],  // [row * num_features + feature]
    gradients: &[f32],
    hessians: &[f32],
    num_features: usize,
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    // Process 2 features at a time with 8-row unrolling
    for feat_start in (0..num_features).step_by(2) {
        let feat_end = (feat_start + 2).min(num_features);

        if feat_end - feat_start == 2 {
            let (left, right) = histograms.split_at_mut(feat_start + 1);
            let hist0 = &mut left[feat_start];
            let hist1 = &mut right[0];

            let mut row = 0;
            let unroll_end = (num_rows / 8) * 8;

            while row < unroll_end {
                // Load 8 gradient/hessian pairs using AVX2
                let grads = _mm256_loadu_ps(gradients.as_ptr().add(row));
                let hesss = _mm256_loadu_ps(hessians.as_ptr().add(row));

                // Extract to arrays
                let mut g_arr = [0.0f32; 8];
                let mut h_arr = [0.0f32; 8];
                _mm256_storeu_ps(g_arr.as_mut_ptr(), grads);
                _mm256_storeu_ps(h_arr.as_mut_ptr(), hesss);

                // Load bins for both features
                let base0 = row * num_features + feat_start;
                let base1 = row * num_features + feat_start + 1;

                // Feature 0
                for i in 0..8 {
                    let bin = *bins_flat.get_unchecked(base0 + i * num_features) as usize;
                    let e = hist0.get_unchecked_mut(bin);
                    e.0 += g_arr[i];
                    e.1 += h_arr[i];
                    e.2 += 1;
                }

                // Feature 1
                for i in 0..8 {
                    let bin = *bins_flat.get_unchecked(base1 + i * num_features) as usize;
                    let e = hist1.get_unchecked_mut(bin);
                    e.0 += g_arr[i];
                    e.1 += h_arr[i];
                    e.2 += 1;
                }

                row += 8;
            }

            // Remainder
            while row < num_rows {
                let g = gradients[row];
                let h = hessians[row];

                let bin0 = bins_flat[row * num_features + feat_start] as usize;
                hist0[bin0].0 += g;
                hist0[bin0].1 += h;
                hist0[bin0].2 += 1;

                let bin1 = bins_flat[row * num_features + feat_start + 1] as usize;
                hist1[bin1].0 += g;
                hist1[bin1].1 += h;
                hist1[bin1].2 += 1;

                row += 1;
            }
        } else {
            let hist = &mut histograms[feat_start];
            for row in 0..num_rows {
                let bin = bins_flat[row * num_features + feat_start] as usize;
                hist[bin].0 += gradients[row];
                hist[bin].1 += hessians[row];
                hist[bin].2 += 1;
            }
        }
    }

    histograms
}

// =============================================================================
// 2D-TILE OPT 6: Pure row-major processing (GPU-style)
// One row at a time, all features - tests memory bandwidth vs compute
// =============================================================================

fn histogram_tile_row_major_pure(
    bins_transposed: &[Vec<u8>],
    gradients: &[f32],
    hessians: &[f32],
    num_features: usize,
) -> Vec<Vec<(f32, f32, u32)>> {
    let num_rows = gradients.len();

    let mut histograms: Vec<Vec<(f32, f32, u32)>> =
        vec![vec![(0.0f32, 0.0f32, 0u32); NUM_BINS]; num_features];

    // Pure row-major: one row, all features
    for row in 0..num_rows {
        let g = gradients[row];
        let h = hessians[row];
        let row_bins = &bins_transposed[row];

        // 8x unroll across features
        let mut f = 0;
        let unroll_end = (num_features / 8) * 8;

        while f < unroll_end {
            unsafe {
                let b0 = *row_bins.get_unchecked(f) as usize;
                let b1 = *row_bins.get_unchecked(f + 1) as usize;
                let b2 = *row_bins.get_unchecked(f + 2) as usize;
                let b3 = *row_bins.get_unchecked(f + 3) as usize;
                let b4 = *row_bins.get_unchecked(f + 4) as usize;
                let b5 = *row_bins.get_unchecked(f + 5) as usize;
                let b6 = *row_bins.get_unchecked(f + 6) as usize;
                let b7 = *row_bins.get_unchecked(f + 7) as usize;

                let e = histograms.get_unchecked_mut(f).get_unchecked_mut(b0);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms.get_unchecked_mut(f + 1).get_unchecked_mut(b1);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms.get_unchecked_mut(f + 2).get_unchecked_mut(b2);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms.get_unchecked_mut(f + 3).get_unchecked_mut(b3);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms.get_unchecked_mut(f + 4).get_unchecked_mut(b4);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms.get_unchecked_mut(f + 5).get_unchecked_mut(b5);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms.get_unchecked_mut(f + 6).get_unchecked_mut(b6);
                e.0 += g; e.1 += h; e.2 += 1;
                let e = histograms.get_unchecked_mut(f + 7).get_unchecked_mut(b7);
                e.0 += g; e.1 += h; e.2 += 1;
            }
            f += 8;
        }

        while f < num_features {
            let bin = row_bins[f] as usize;
            histograms[f][bin].0 += g;
            histograms[f][bin].1 += h;
            histograms[f][bin].2 += 1;
            f += 1;
        }
    }

    histograms
}

// =============================================================================
// MAIN: Benchmark all 2D-tile optimizations
// =============================================================================

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("          OPTIMIZED TENSOR-TILE HISTOGRAM POC");
    println!("═══════════════════════════════════════════════════════════════════════════════\n");

    println!("Configuration:");
    println!("  Rows:       {:>10}", NUM_ROWS);
    println!("  Features:   {:>10}", NUM_FEATURES);
    println!("  Bins:       {:>10}", NUM_BINS);
    println!("  Iterations: {:>10}", ITERATIONS);

    // Generate test data - column-major (current format)
    println!("\nGenerating test data...");
    let bins_col_major: Vec<Vec<u8>> = (0..NUM_FEATURES)
        .map(|f| {
            (0..NUM_ROWS)
                .map(|r| ((r * (f + 1) * 17) % 256) as u8)
                .collect()
        })
        .collect();

    // Transpose to row-major for 2D-tile approaches
    let bins_transposed: Vec<Vec<u8>> = (0..NUM_ROWS)
        .map(|r| {
            (0..NUM_FEATURES)
                .map(|f| bins_col_major[f][r])
                .collect()
        })
        .collect();

    // Flat layout - row-major flat array
    let bins_flat: Vec<u8> = (0..NUM_ROWS)
        .flat_map(|r| {
            let row_bins: Vec<u8> = (0..NUM_FEATURES).map(|f| bins_col_major[f][r]).collect();
            row_bins
        })
        .collect();

    let gradients: Vec<f32> = (0..NUM_ROWS)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let hessians: Vec<f32> = (0..NUM_ROWS)
        .map(|i| 1.0 + (i as f32 * 0.005).cos() * 0.1)
        .collect();

    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("BENCHMARKS");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    // Warmup
    let _ = histogram_scalar_baseline(&bins_col_major, &gradients, &hessians);

    let mut results: Vec<(&str, f64)> = Vec::new();

    // 1. Scalar baseline (column-major)
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = histogram_scalar_baseline(&bins_col_major, &gradients, &hessians);
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("Scalar baseline (col-major)", time));

    // 2. Transposed naive
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = histogram_tile_transposed(&bins_transposed, &gradients, &hessians, NUM_FEATURES);
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("Transposed naive", time));

    // 3. Transposed + 8-feature tile
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe {
            let _ = histogram_tile_simd_8feat(&bins_transposed, &gradients, &hessians, NUM_FEATURES);
        }
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("Transposed + 8-feat tile", time));

    // 4. Transposed + 4-row × 8-feat
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe {
            let _ = histogram_tile_4row_8feat(&bins_transposed, &gradients, &hessians, NUM_FEATURES);
        }
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("Transposed + 4row×8feat", time));

    // 5. Register-blocked 2-feat
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = histogram_tile_register_blocked(&bins_transposed, &gradients, &hessians, NUM_FEATURES);
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("Register-blocked 2-feat", time));

    // 6. Flat + AVX2
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe {
            let _ = histogram_tile_flat_avx2(&bins_flat, &gradients, &hessians, NUM_FEATURES);
        }
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("Flat layout + AVX2", time));

    // 7. Row-major pure (GPU-style)
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = histogram_tile_row_major_pure(&bins_transposed, &gradients, &hessians, NUM_FEATURES);
    }
    let time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
    results.push(("Row-major pure (GPU-style)", time));

    // Print results
    let baseline = results[0].1;
    println!("{:<30} {:>10} {:>10}", "Approach", "Time (ms)", "vs Scalar");
    println!("{}", "─".repeat(55));

    for (name, time) in &results {
        let speedup = baseline / time;
        let marker = if speedup > 1.05 { "✓" } else if speedup < 0.95 { "✗" } else { " " };
        println!("{:<30} {:>10.3} {:>9.2}x {}", name, time, speedup, marker);
    }

    // Find best tile approach (excluding baseline)
    let best_tile = results[1..].iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();

    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("ANALYSIS");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    if best_tile.1 < baseline {
        println!("BEST TILE APPROACH: {} ({:.3} ms)", best_tile.0, best_tile.1);
        println!("Improvement over scalar: {:.1}%", (baseline - best_tile.1) / baseline * 100.0);
        println!("\n2D-tile CAN beat scalar with the right optimizations!");
    } else {
        println!("BEST TILE APPROACH: {} ({:.3} ms)", best_tile.0, best_tile.1);
        println!("Still {:.1}% slower than scalar baseline", (best_tile.1 - baseline) / baseline * 100.0);
        println!("\nScalar still wins because:");
        println!("  - Column-major layout gives perfect sequential access for one feature");
        println!("  - 8x unrolling provides sufficient ILP");
        println!("  - Histogram (3KB) already fits in L1");
    }

    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("INSIGHTS FOR TENSOR-TILE GBDT");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    println!("For tensor-tile to win on CPU:");
    println!("  1. Need AVX-512 with hardware scatter-add (doesn't exist yet)");
    println!("  2. Or GPU with atomic histogram updates");
    println!("  3. Or specialized histogram units (like ARM SVE2 HISTCNT)");
    println!();
    println!("Current best CPU strategy:");
    println!("  - Keep column-major layout for sequential bin access");
    println!("  - Process 2 features at a time sharing grad/hess loads");
    println!("  - Use 8x unrolling for ILP");
    println!("  - Let hardware prefetcher handle memory");
}
