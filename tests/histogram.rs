//! Histogram tests for multi-output vector accumulation

use treeboost::histogram::{VectorHistogram, VectorNodeHistograms};

/// Test vector histogram accumulation with multiple outputs
///
/// Verifies that:
/// 1. Gradients and hessians are correctly accumulated per bin per output
/// 2. Counts are shared across outputs (same bin, same samples)
/// 3. Totals are computed correctly for each output
#[test]
fn test_histogram_vector_accumulation() {
    let num_outputs = 3;
    let mut hist = VectorHistogram::new(num_outputs);

    // Accumulate samples into bin 0
    // Sample 1: grads = [1.0, 2.0, 3.0], hess = [0.5, 0.5, 0.5]
    hist.accumulate(0, &[1.0, 2.0, 3.0], &[0.5, 0.5, 0.5]);

    // Sample 2: grads = [0.5, 1.0, 1.5], hess = [0.25, 0.25, 0.25]
    hist.accumulate(0, &[0.5, 1.0, 1.5], &[0.25, 0.25, 0.25]);

    // Accumulate sample into bin 255
    // Sample 3: grads = [3.0, 6.0, 9.0], hess = [1.0, 1.0, 1.0]
    hist.accumulate(255, &[3.0, 6.0, 9.0], &[1.0, 1.0, 1.0]);

    // Verify bin 0 accumulation
    let (grads_0, hess_0) = hist.get_output_stats(0, 0); // bin 0, output 0
    assert!((grads_0 - 1.5).abs() < 1e-6, "grads_0 = {}", grads_0);
    assert!((hess_0 - 0.75).abs() < 1e-6, "hess_0 = {}", hess_0);

    let (grads_1, hess_1) = hist.get_output_stats(0, 1); // bin 0, output 1
    assert!((grads_1 - 3.0).abs() < 1e-6, "grads_1 = {}", grads_1);
    assert!((hess_1 - 0.75).abs() < 1e-6, "hess_1 = {}", hess_1);

    let (grads_2, hess_2) = hist.get_output_stats(0, 2); // bin 0, output 2
    assert!((grads_2 - 4.5).abs() < 1e-6, "grads_2 = {}", grads_2);
    assert!((hess_2 - 0.75).abs() < 1e-6, "hess_2 = {}", hess_2);

    // Verify bin 0 count (shared across outputs)
    assert_eq!(hist.get_count(0), 2, "bin 0 should have 2 samples");

    // Verify bin 255
    let (grads_255_0, hess_255_0) = hist.get_output_stats(255, 0);
    assert!((grads_255_0 - 3.0).abs() < 1e-6);
    assert!((hess_255_0 - 1.0).abs() < 1e-6);

    let (grads_255_2, hess_255_2) = hist.get_output_stats(255, 2);
    assert!((grads_255_2 - 9.0).abs() < 1e-6);
    assert!((hess_255_2 - 1.0).abs() < 1e-6);

    assert_eq!(hist.get_count(255), 1, "bin 255 should have 1 sample");

    // Verify totals per output
    let (total_g0, total_h0) = hist.total_output(0);
    assert!((total_g0 - 4.5).abs() < 1e-6, "total_g0 = {}", total_g0);
    assert!((total_h0 - 1.75).abs() < 1e-6, "total_h0 = {}", total_h0);

    let (total_g1, total_h1) = hist.total_output(1);
    assert!((total_g1 - 9.0).abs() < 1e-6, "total_g1 = {}", total_g1);
    assert!((total_h1 - 1.75).abs() < 1e-6, "total_h1 = {}", total_h1);

    let (total_g2, total_h2) = hist.total_output(2);
    assert!((total_g2 - 13.5).abs() < 1e-6, "total_g2 = {}", total_g2);
    assert!((total_h2 - 1.75).abs() < 1e-6, "total_h2 = {}", total_h2);

    // Verify total count
    assert_eq!(hist.total_count(), 3, "total should be 3 samples");
}

/// Test vector histogram batch accumulation
///
/// Verifies the 8x unrolled batch accumulation path.
#[test]
fn test_histogram_vector_batch_accumulation() {
    let num_outputs = 2;
    let mut hist = VectorHistogram::new(num_outputs);

    // Create batch of 10 samples
    let bins: Vec<u8> = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    // Gradients: [output0, output1] interleaved per sample
    let gradients: Vec<f32> = vec![
        1.0, 2.0, // sample 0
        1.0, 2.0, // sample 1
        3.0, 4.0, // sample 2
        3.0, 4.0, // sample 3
        5.0, 6.0, // sample 4
        5.0, 6.0, // sample 5
        7.0, 8.0, // sample 6
        7.0, 8.0, // sample 7
        9.0, 10.0, // sample 8
        9.0, 10.0, // sample 9
    ];
    let hessians: Vec<f32> = vec![1.0; 20]; // All hessians = 1.0

    hist.accumulate_batch(&bins, &gradients, &hessians);

    // Verify bin 0: 2 samples with grads [1,2] each
    let (g0_out0, h0_out0) = hist.get_output_stats(0, 0);
    let (g0_out1, h0_out1) = hist.get_output_stats(0, 1);
    assert!((g0_out0 - 2.0).abs() < 1e-6); // 1.0 + 1.0
    assert!((g0_out1 - 4.0).abs() < 1e-6); // 2.0 + 2.0
    assert!((h0_out0 - 2.0).abs() < 1e-6);
    assert!((h0_out1 - 2.0).abs() < 1e-6);
    assert_eq!(hist.get_count(0), 2);

    // Verify bin 4: 2 samples with grads [9,10] each
    let (g4_out0, h4_out0) = hist.get_output_stats(4, 0);
    let (g4_out1, h4_out1) = hist.get_output_stats(4, 1);
    assert!((g4_out0 - 18.0).abs() < 1e-6); // 9.0 + 9.0
    assert!((g4_out1 - 20.0).abs() < 1e-6); // 10.0 + 10.0
    assert!((h4_out0 - 2.0).abs() < 1e-6);
    assert!((h4_out1 - 2.0).abs() < 1e-6);
    assert_eq!(hist.get_count(4), 2);

    // Verify total count
    assert_eq!(hist.total_count(), 10);
}

/// Test vector histogram subtraction trick
///
/// Verifies: sibling = parent - child for vector histograms.
#[test]
fn test_histogram_vector_subtraction() {
    let num_outputs = 2;

    // Parent histogram
    let mut parent = VectorHistogram::new(num_outputs);
    parent.accumulate(0, &[10.0, 20.0], &[5.0, 5.0]);
    parent.accumulate(1, &[5.0, 10.0], &[2.5, 2.5]);

    // Child histogram (subset of parent)
    let mut child = VectorHistogram::new(num_outputs);
    child.accumulate(0, &[3.0, 6.0], &[1.5, 1.5]);
    child.accumulate(1, &[2.0, 4.0], &[1.0, 1.0]);

    // Sibling = parent - child
    let sibling = VectorHistogram::from_subtraction(&parent, &child);

    // Verify bin 0
    let (g0_out0, h0_out0) = sibling.get_output_stats(0, 0);
    let (g0_out1, h0_out1) = sibling.get_output_stats(0, 1);
    assert!((g0_out0 - 7.0).abs() < 1e-6, "g0_out0 = {}", g0_out0);
    assert!((g0_out1 - 14.0).abs() < 1e-6, "g0_out1 = {}", g0_out1);
    assert!((h0_out0 - 3.5).abs() < 1e-6);
    assert!((h0_out1 - 3.5).abs() < 1e-6);

    // Verify bin 1
    let (g1_out0, h1_out0) = sibling.get_output_stats(1, 0);
    let (g1_out1, h1_out1) = sibling.get_output_stats(1, 1);
    assert!((g1_out0 - 3.0).abs() < 1e-6);
    assert!((g1_out1 - 6.0).abs() < 1e-6);
    assert!((h1_out0 - 1.5).abs() < 1e-6);
    assert!((h1_out1 - 1.5).abs() < 1e-6);
}

/// Test VectorNodeHistograms for multi-feature support
#[test]
fn test_vector_node_histograms() {
    let num_features = 3;
    let num_outputs = 2;

    let mut hists = VectorNodeHistograms::new(num_features, num_outputs);

    // Accumulate into different features
    hists.get_mut(0).accumulate(5, &[1.0, 2.0], &[0.5, 0.5]);
    hists.get_mut(1).accumulate(10, &[3.0, 4.0], &[1.0, 1.0]);
    hists.get_mut(2).accumulate(15, &[5.0, 6.0], &[1.5, 1.5]);

    assert_eq!(hists.num_features(), 3);
    assert_eq!(hists.num_outputs(), 2);

    // Verify feature 0
    let (g_f0, h_f0) = hists.get(0).get_output_stats(5, 0);
    assert!((g_f0 - 1.0).abs() < 1e-6);
    assert!((h_f0 - 0.5).abs() < 1e-6);

    // Verify feature 1
    let (g_f1, h_f1) = hists.get(1).get_output_stats(10, 1);
    assert!((g_f1 - 4.0).abs() < 1e-6);
    assert!((h_f1 - 1.0).abs() < 1e-6);

    // Verify feature 2
    let (g_f2, h_f2) = hists.get(2).get_output_stats(15, 0);
    assert!((g_f2 - 5.0).abs() < 1e-6);
    assert!((h_f2 - 1.5).abs() < 1e-6);
}

/// Test VectorNodeHistograms merge and subtraction
#[test]
fn test_vector_node_histograms_merge_subtract() {
    let num_features = 2;
    let num_outputs = 2;

    let mut hists1 = VectorNodeHistograms::new(num_features, num_outputs);
    hists1.get_mut(0).accumulate(0, &[10.0, 20.0], &[5.0, 5.0]);
    hists1.get_mut(1).accumulate(0, &[5.0, 10.0], &[2.5, 2.5]);

    let mut hists2 = VectorNodeHistograms::new(num_features, num_outputs);
    hists2.get_mut(0).accumulate(0, &[3.0, 6.0], &[1.5, 1.5]);
    hists2.get_mut(1).accumulate(0, &[2.0, 4.0], &[1.0, 1.0]);

    // Test merge
    let mut merged = VectorNodeHistograms::new(num_features, num_outputs);
    merged.get_mut(0).accumulate(0, &[10.0, 20.0], &[5.0, 5.0]);
    merged.get_mut(1).accumulate(0, &[5.0, 10.0], &[2.5, 2.5]);
    merged.merge(&hists2);

    let (g_merged, h_merged) = merged.get(0).get_output_stats(0, 0);
    assert!((g_merged - 13.0).abs() < 1e-6);
    assert!((h_merged - 6.5).abs() < 1e-6);

    // Test subtraction
    let sibling = VectorNodeHistograms::from_subtraction(&hists1, &hists2);
    let (g_sib, h_sib) = sibling.get(0).get_output_stats(0, 0);
    assert!((g_sib - 7.0).abs() < 1e-6);
    assert!((h_sib - 3.5).abs() < 1e-6);
}

/// Test that vector histogram is memory efficient with flat buffer layout
#[test]
fn test_histogram_vector_memory_layout() {
    let num_outputs = 4;
    let hist = VectorHistogram::new(num_outputs);

    // Verify internal buffer sizes
    // grad_hess_buffer: 256 bins * 4 outputs * 2 (grad+hess) = 2048 f32 values
    // counts: 256 u32 values
    let expected_gh_size = 256 * num_outputs * 2;
    let expected_count_size = 256;

    assert_eq!(hist.grad_hess_buffer_len(), expected_gh_size);
    assert_eq!(hist.counts_len(), expected_count_size);
}
