//! Split finding tests for multi-output vector gain computation
//!
//! Split Gain Vector Summation (2D Multi-Label Support)

use treeboost::histogram::{VectorHistogram, VectorNodeHistograms};
use treeboost::tree::{VectorSplitFinder, VectorSplitInfo};

/// Test vector split gain computation with multiple outputs
///
/// Verifies that:
/// 1. Split gain is computed as sum of per-output gains: Gain = Σ_k (G_L,k² / (H_L,k + λ) + G_R,k² / (H_R,k + λ) - G_k² / (H_k + λ))
/// 2. Per-output gradient/hessian sums are correctly propagated
/// 3. Count validation works correctly
#[test]
fn test_split_gain_vector_summation() {
    let num_outputs = 2;
    let lambda = 1.0;
    let min_samples_leaf = 1;
    let min_hessian_leaf = 0.0;

    // Create a simple histogram with a clear split point
    let mut hist = VectorHistogram::new(num_outputs);

    // Left partition: bins 0-127 get positive gradients for both outputs
    // We'll put 10 samples in bin 0 with grads [1.0, 0.5] and hess [1.0, 1.0] each
    for _ in 0..10 {
        hist.accumulate(0, &[1.0, 0.5], &[1.0, 1.0]);
    }

    // Right partition: bins 128-255 get negative gradients
    // We'll put 10 samples in bin 128 with grads [-1.0, -0.5] and hess [1.0, 1.0] each
    for _ in 0..10 {
        hist.accumulate(128, &[-1.0, -0.5], &[1.0, 1.0]);
    }

    let finder = VectorSplitFinder::new(num_outputs)
        .with_lambda(lambda)
        .with_min_samples_leaf(min_samples_leaf)
        .with_min_hessian_leaf(min_hessian_leaf);

    // Find the best split
    let split = finder.find_best_split(&hist, 0);

    assert!(split.is_valid(), "split should be valid");
    assert_eq!(split.bin_threshold, 0, "should split at bin 0"); // samples <= 0 go left

    // Verify left statistics (bin 0)
    // Left: G_0 = 10.0, G_1 = 5.0, H_0 = 10.0, H_1 = 10.0, count = 10
    let (left_g0, left_h0) = split.left_stats(0);
    let (left_g1, left_h1) = split.left_stats(1);
    assert!((left_g0 - 10.0).abs() < 1e-5, "left_g0 = {}", left_g0);
    assert!((left_g1 - 5.0).abs() < 1e-5, "left_g1 = {}", left_g1);
    assert!((left_h0 - 10.0).abs() < 1e-5, "left_h0 = {}", left_h0);
    assert!((left_h1 - 10.0).abs() < 1e-5, "left_h1 = {}", left_h1);
    assert_eq!(split.left_count, 10);

    // Verify right statistics (bin 128)
    // Right: G_0 = -10.0, G_1 = -5.0, H_0 = 10.0, H_1 = 10.0, count = 10
    let (right_g0, right_h0) = split.right_stats(0);
    let (right_g1, right_h1) = split.right_stats(1);
    assert!((right_g0 - (-10.0)).abs() < 1e-5, "right_g0 = {}", right_g0);
    assert!((right_g1 - (-5.0)).abs() < 1e-5, "right_g1 = {}", right_g1);
    assert!((right_h0 - 10.0).abs() < 1e-5, "right_h0 = {}", right_h0);
    assert!((right_h1 - 10.0).abs() < 1e-5, "right_h1 = {}", right_h1);
    assert_eq!(split.right_count, 10);

    // Verify gain computation
    // For each output k:
    //   gain_k = 0.5 * [G_L,k²/(H_L,k + λ) + G_R,k²/(H_R,k + λ) - G_k²/(H_k + λ)]
    //
    // Output 0:
    //   G_L = 10, H_L = 10, G_R = -10, H_R = 10, G = 0, H = 20
    //   gain_0 = 0.5 * [100/(10+1) + 100/(10+1) - 0/(20+1)]
    //          = 0.5 * [9.09 + 9.09 - 0] = 9.09
    //
    // Output 1:
    //   G_L = 5, H_L = 10, G_R = -5, H_R = 10, G = 0, H = 20
    //   gain_1 = 0.5 * [25/(10+1) + 25/(10+1) - 0/(20+1)]
    //          = 0.5 * [2.27 + 2.27 - 0] = 2.27
    //
    // Total gain = 9.09 + 2.27 = 11.36
    let expected_gain_0 = 0.5 * (100.0 / 11.0 + 100.0 / 11.0);
    let expected_gain_1 = 0.5 * (25.0 / 11.0 + 25.0 / 11.0);
    let expected_total_gain = expected_gain_0 + expected_gain_1;

    assert!(
        (split.gain - expected_total_gain).abs() < 1e-4,
        "expected gain {}, got {}",
        expected_total_gain,
        split.gain
    );
}

/// Test that minimum samples constraint is respected
#[test]
fn test_split_vector_min_samples() {
    let num_outputs = 2;

    let mut hist = VectorHistogram::new(num_outputs);

    // Put 5 samples in bin 0
    for _ in 0..5 {
        hist.accumulate(0, &[1.0, 1.0], &[1.0, 1.0]);
    }

    // Put 5 samples in bin 128
    for _ in 0..5 {
        hist.accumulate(128, &[-1.0, -1.0], &[1.0, 1.0]);
    }

    // With min_samples_leaf = 6, no valid split should be found
    let finder = VectorSplitFinder::new(num_outputs)
        .with_lambda(1.0)
        .with_min_samples_leaf(6)
        .with_min_hessian_leaf(0.0);

    let split = finder.find_best_split(&hist, 0);
    assert!(
        !split.is_valid(),
        "split should be invalid with min_samples_leaf=6"
    );

    // With min_samples_leaf = 5, split should be valid
    let finder = VectorSplitFinder::new(num_outputs)
        .with_lambda(1.0)
        .with_min_samples_leaf(5)
        .with_min_hessian_leaf(0.0);

    let split = finder.find_best_split(&hist, 0);
    assert!(
        split.is_valid(),
        "split should be valid with min_samples_leaf=5"
    );
}

/// Test that minimum hessian constraint is respected per-output
#[test]
fn test_split_vector_min_hessian() {
    let num_outputs = 2;

    let mut hist = VectorHistogram::new(num_outputs);

    // Put samples with low hessians
    for _ in 0..10 {
        hist.accumulate(0, &[1.0, 1.0], &[0.05, 0.05]);
    }
    for _ in 0..10 {
        hist.accumulate(128, &[-1.0, -1.0], &[0.05, 0.05]);
    }

    // With min_hessian_leaf = 1.0, no valid split (hessians are too small)
    let finder = VectorSplitFinder::new(num_outputs)
        .with_lambda(1.0)
        .with_min_samples_leaf(1)
        .with_min_hessian_leaf(1.0);

    let split = finder.find_best_split(&hist, 0);
    assert!(
        !split.is_valid(),
        "split should be invalid with min_hessian_leaf=1.0"
    );

    // With min_hessian_leaf = 0.4, split should be valid (total hessian per side = 0.5)
    let finder = VectorSplitFinder::new(num_outputs)
        .with_lambda(1.0)
        .with_min_samples_leaf(1)
        .with_min_hessian_leaf(0.4);

    let split = finder.find_best_split(&hist, 0);
    assert!(
        split.is_valid(),
        "split should be valid with min_hessian_leaf=0.4"
    );
}

/// Test VectorSplitInfo helper methods
#[test]
fn test_vector_split_info_helpers() {
    let num_outputs = 3;
    let mut info = VectorSplitInfo::new(num_outputs);

    info.feature_idx = 5;
    info.bin_threshold = 127;
    info.gain = 10.5;
    info.left_count = 50;
    info.right_count = 50;

    // Set left stats for each output
    info.set_left_stats(0, 1.0, 0.5);
    info.set_left_stats(1, 2.0, 1.0);
    info.set_left_stats(2, 3.0, 1.5);

    // Set right stats for each output
    info.set_right_stats(0, -1.0, 0.5);
    info.set_right_stats(1, -2.0, 1.0);
    info.set_right_stats(2, -3.0, 1.5);

    // Verify accessors
    assert_eq!(info.num_outputs(), 3);

    let (lg0, lh0) = info.left_stats(0);
    assert_eq!(lg0, 1.0);
    assert_eq!(lh0, 0.5);

    let (rg2, rh2) = info.right_stats(2);
    assert_eq!(rg2, -3.0);
    assert_eq!(rh2, 1.5);

    // Verify all getters
    assert_eq!(info.left_gradients(), vec![1.0, 2.0, 3.0]);
    assert_eq!(info.left_hessians(), vec![0.5, 1.0, 1.5]);
    assert_eq!(info.right_gradients(), vec![-1.0, -2.0, -3.0]);
    assert_eq!(info.right_hessians(), vec![0.5, 1.0, 1.5]);
}

/// Test finding best split across all features
#[test]
fn test_find_best_split_all_features() {
    let num_features = 3;
    let num_outputs = 2;

    let mut hists = VectorNodeHistograms::new(num_features, num_outputs);

    // Feature 0: poor split (gradients evenly distributed)
    for _ in 0..10 {
        hists.get_mut(0).accumulate(0, &[0.5, 0.5], &[1.0, 1.0]);
        hists.get_mut(0).accumulate(128, &[0.5, 0.5], &[1.0, 1.0]);
    }

    // Feature 1: good split (gradients well separated)
    for _ in 0..10 {
        hists.get_mut(1).accumulate(50, &[2.0, 2.0], &[1.0, 1.0]);
    }
    for _ in 0..10 {
        hists.get_mut(1).accumulate(200, &[-2.0, -2.0], &[1.0, 1.0]);
    }

    // Feature 2: medium split
    for _ in 0..10 {
        hists.get_mut(2).accumulate(0, &[1.0, 1.0], &[1.0, 1.0]);
        hists.get_mut(2).accumulate(255, &[-1.0, -1.0], &[1.0, 1.0]);
    }

    let finder = VectorSplitFinder::new(num_outputs)
        .with_lambda(1.0)
        .with_min_samples_leaf(1);

    let best = finder.find_best_split_all_features(&hists);

    assert!(best.is_valid(), "should find a valid split");
    assert_eq!(best.feature_idx, 1, "feature 1 should have the best gain");
    assert_eq!(
        best.bin_threshold, 50,
        "should split at bin 50 for feature 1"
    );
}
