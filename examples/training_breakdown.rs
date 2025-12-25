//! Detailed training time breakdown
//!
//! Run: cargo run --release --example training_breakdown

use std::time::Instant;

use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use treeboost::histogram::HistogramBuilder;
use treeboost::tree::{SplitFinder, TreeGrower};

fn create_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
    let mut features = Vec::with_capacity(num_rows * num_features);
    for f in 0..num_features {
        for r in 0..num_rows {
            features.push(((r * (f + 1) * 17) % 256) as u8);
        }
    }

    let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32 * 0.01).sin()).collect();
    let feature_info = (0..num_features)
        .map(|i| FeatureInfo {
            name: format!("f{}", i),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
        })
        .collect();

    BinnedDataset::new(num_rows, features, targets, feature_info)
}

fn main() {
    println!("Training Time Breakdown");
    println!("=======================\n");

    let num_rows = 100_000;
    let num_features = 50;
    let iterations = 20;

    println!("Dataset: {} rows × {} features\n", num_rows, num_features);

    let dataset = create_dataset(num_rows, num_features);
    let gradients: Vec<f32> = (0..num_rows)
        .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
        .collect();
    let hessians: Vec<f32> = vec![1.0; num_rows];
    let row_indices: Vec<usize> = (0..num_rows).collect();

    // 1. Histogram building (root node - all rows)
    let builder = HistogramBuilder::new();
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = builder.build(&dataset, &row_indices, &gradients, &hessians);
    }
    let hist_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    // 2. Split finding
    let histograms = builder.build(&dataset, &row_indices, &gradients, &hessians);
    let split_finder = SplitFinder::new()
        .with_lambda(1.0)
        .with_min_samples_leaf(1)
        .with_min_hessian_leaf(1.0);

    let total_grad: f32 = gradients.iter().sum();
    let total_hess: f32 = hessians.iter().sum();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = split_finder.find_best_split(&histograms, total_grad, total_hess, num_rows as u32);
    }
    let split_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    // 3. Row partitioning (simulated split at middle)
    let start = Instant::now();
    for _ in 0..iterations {
        let mut left = Vec::with_capacity(num_rows / 2);
        let mut right = Vec::with_capacity(num_rows / 2);
        let feature_col = dataset.feature_column(0);
        for &row_idx in &row_indices {
            if feature_col[row_idx] <= 127 {
                left.push(row_idx);
            } else {
                right.push(row_idx);
            }
        }
        std::hint::black_box((left, right));
    }
    let partition_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    // 4. Gradient computation
    let predictions = vec![0.0f32; num_rows];
    let targets: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.01).collect();
    let mut grads = vec![0.0f32; num_rows];
    let mut hesss = vec![0.0f32; num_rows];

    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..num_rows {
            let residual = targets[i] - predictions[i];
            grads[i] = -residual; // MSE gradient
            hesss[i] = 1.0;       // MSE hessian
        }
    }
    let grad_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    // 5. Prediction update (batch add)
    let grower = TreeGrower::new()
        .with_max_depth(6)
        .with_max_leaves(31);
    let tree = grower.grow(&dataset, &gradients, &hessians);
    let mut preds = vec![0.0f32; num_rows];

    let start = Instant::now();
    for _ in 0..iterations {
        preds.fill(0.0);
        tree.predict_batch_add(&dataset, &mut preds);
    }
    let predict_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    // 6. Full single tree grow
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = grower.grow(&dataset, &gradients, &hessians);
    }
    let tree_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    // 7. NodeHistograms allocation
    let start = Instant::now();
    for _ in 0..(iterations * 100) {
        let _ = treeboost::histogram::NodeHistograms::new(num_features);
    }
    let alloc_time = start.elapsed().as_secs_f64() * 1000.0 / (iterations * 100) as f64;

    println!("Component Breakdown (per operation):");
    println!("------------------------------------");
    println!("{:<30} {:>10.3} ms", "Histogram build (root)", hist_time);
    println!("{:<30} {:>10.3} ms", "Split finding", split_time);
    println!("{:<30} {:>10.3} ms", "Row partitioning", partition_time);
    println!("{:<30} {:>10.3} ms", "Gradient computation", grad_time);
    println!("{:<30} {:>10.3} ms", "Prediction update (1 tree)", predict_time);
    println!("{:<30} {:>10.3} ms", "NodeHistograms alloc", alloc_time);
    println!("{:<30} {:>10.3} ms", "Full tree grow", tree_time);

    // Estimate per-round cost
    // A tree with 31 leaves has ~30 internal nodes, so ~30 splits
    // Each split: histogram for smaller child + split finding
    // Subtraction trick means ~15 histogram builds per tree
    let estimated_round = hist_time + split_time * 30.0 + partition_time * 30.0 + grad_time + predict_time;

    println!("\n\nEstimated vs Actual:");
    println!("--------------------");
    println!("{:<30} {:>10.3} ms", "Estimated per round", estimated_round);
    println!("{:<30} {:>10.3} ms", "Actual tree grow", tree_time);
    println!("{:<30} {:>10.3} ms", "Overhead", tree_time - (hist_time + split_time));

    // Compare to LightGBM
    println!("\n\nComparison to LightGBM:");
    println!("-----------------------");
    let lgb_per_round = 268.0 / 100.0; // 268ms for 100 rounds
    let our_per_round = tree_time + grad_time + predict_time;
    println!("{:<30} {:>10.3} ms", "LightGBM per round", lgb_per_round);
    println!("{:<30} {:>10.3} ms", "TreeBoost per round", our_per_round);
    println!("{:<30} {:>10.1}x", "Slowdown", our_per_round / lgb_per_round);

    // Allocation analysis
    println!("\n\nAllocation Analysis:");
    println!("--------------------");

    // Each SplitCandidate stores row_indices Vec
    // With 31 leaves, we have ~30 splits, each creates 2 children
    // Worst case: all 100k rows stored multiple times
    let row_idx_size = std::mem::size_of::<usize>();
    let approx_total_stored = num_rows * 2; // rough estimate (rows appear in multiple candidates)
    let bytes_for_indices = approx_total_stored * row_idx_size;
    println!("{:<30} {:>10} bytes", "Approx row indices stored", bytes_for_indices);
    println!("{:<30} {:>10.2} MB", "In MB", bytes_for_indices as f64 / 1_000_000.0);

    // Histogram allocations
    // Each NodeHistograms has 50 features × 256 bins × 12 bytes (grad + hess + count)
    let hist_size = num_features * 256 * 12;
    let total_hist_alloc = hist_size * 62; // ~62 nodes in a 31-leaf tree
    println!("{:<30} {:>10} bytes", "Histogram allocations", total_hist_alloc);
    println!("{:<30} {:>10.2} MB", "In MB", total_hist_alloc as f64 / 1_000_000.0);

    // Vec allocation cost measurement
    let start = Instant::now();
    for _ in 0..1000 {
        let v: Vec<usize> = Vec::with_capacity(num_rows / 2);
        std::hint::black_box(v);
    }
    let vec_alloc_time = start.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    println!("{:<30} {:>10.4} ms", "Vec<usize> alloc (50k cap)", vec_alloc_time);
    println!("{:<30} {:>10.3} ms", "30 such allocs", vec_alloc_time * 30.0);

    // Detailed breakdown of what happens in tree growing
    println!("\n\nDetailed Tree Growing Analysis:");
    println!("--------------------------------");

    // Simulate tree growing operations
    // 31 leaves = 30 splits
    // Each split needs: partition + histogram build (smaller) + 2x split finding

    // Histogram builds at different sizes (subtraction trick means we build smaller child)
    // Level 0: 1 node with 100k rows (root)
    // Level 1: build smaller of 2 children (~50k)
    // Level 2: build 2 smaller children (~25k each)
    // etc.

    let row_sizes = [100000, 50000, 25000, 12500, 6250, 3125]; // approximate node sizes by level

    println!("\nHistogram build by node size:");
    for &size in &row_sizes {
        if size < 100 { continue; }
        let subset: Vec<usize> = (0..size).collect();
        let start = Instant::now();
        for _ in 0..10 {
            let _ = builder.build(&dataset, &subset, &gradients[..size], &hessians[..size]);
        }
        let time = start.elapsed().as_secs_f64() * 1000.0 / 10.0;
        println!("  {:>6} rows: {:>8.3} ms", size, time);
    }

    // Histogram subtraction
    let half_rows: Vec<usize> = (0..num_rows/2).collect();
    let parent_hist = builder.build(&dataset, &row_indices, &gradients, &hessians);
    let child_hist = builder.build(&dataset, &half_rows, &gradients[..num_rows/2], &hessians[..num_rows/2]);

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = HistogramBuilder::build_sibling(&parent_hist, &child_hist);
    }
    let subtract_time = start.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    println!("\nHistogram subtraction:       {:>8.4} ms", subtract_time);

    // Total estimated histogram work for a 31-leaf tree
    // Using subtraction trick: only build histograms for smaller children
    // Approximate: 1 root + 15 smaller children across all levels
    let total_hist_time = hist_time + 0.5 * hist_time + 0.25 * hist_time * 2.0 + 0.125 * hist_time * 4.0;
    println!("\nEstimated total histogram:   {:>8.3} ms", total_hist_time);
    println!("Split finding (30 × 2):      {:>8.3} ms", split_time * 60.0);
    println!("Row partitioning (30×):      {:>8.3} ms", partition_time * 30.0);
    println!("Subtraction (30×):           {:>8.3} ms", subtract_time * 30.0);
    let total_estimated = total_hist_time + split_time * 60.0 + partition_time * 30.0 + subtract_time * 30.0;
    println!("TOTAL estimated:             {:>8.3} ms", total_estimated);
    println!("Actual tree grow:            {:>8.3} ms", tree_time);
    println!("Unexplained overhead:        {:>8.3} ms", tree_time - total_estimated);
}
