//! Deep profiler for large datasets
//!
//! Run: cargo run --release --example large_dataset_profiler
//!
//! This investigates:
//! 1. Scaling behavior from 100k to 1M rows
//! 2. Cache effects at different sizes
//! 3. Memory bandwidth bottlenecks
//! 4. Per-component scaling factors

use std::time::Instant;

use treeboost::booster::{GBDTConfig, GBDTModel, LossType};
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use treeboost::histogram::HistogramBuilder;
use treeboost::tree::TreeGrower;

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

fn profile_histogram_build(
    dataset: &BinnedDataset,
    row_indices: &[usize],
    gradients: &[f32],
    hessians: &[f32],
    iterations: usize,
) -> f64 {
    let builder = HistogramBuilder::new();

    // Warmup
    for _ in 0..2 {
        let _ = builder.build(dataset, row_indices, gradients, hessians);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = builder.build(dataset, row_indices, gradients, hessians);
    }
    start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
}

fn profile_tree_grow(
    dataset: &BinnedDataset,
    gradients: &[f32],
    hessians: &[f32],
    row_indices: &[usize],
    iterations: usize,
) -> f64 {
    let grower = TreeGrower::new()
        .with_max_depth(6)
        .with_max_leaves(31)
        .with_learning_rate(0.1);

    // Warmup
    for _ in 0..2 {
        let _ = grower.grow_with_indices(dataset, gradients, hessians, row_indices);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = grower.grow_with_indices(dataset, gradients, hessians, row_indices);
    }
    start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
}

fn profile_gradient_computation(
    targets: &[f32],
    predictions: &[f32],
    row_indices: &[usize],
    gradients: &mut [f32],
    hessians: &mut [f32],
    iterations: usize,
) -> f64 {
    let loss_fn = LossType::Mse.create();

    let start = Instant::now();
    for _ in 0..iterations {
        for &idx in row_indices {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            gradients[idx] = g;
            hessians[idx] = h;
        }
    }
    start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
}

fn profile_prediction_update(
    dataset: &BinnedDataset,
    gradients: &[f32],
    hessians: &[f32],
    predictions: &mut [f32],
    iterations: usize,
) -> f64 {
    let grower = TreeGrower::new()
        .with_max_depth(6)
        .with_max_leaves(31)
        .with_learning_rate(0.1);
    let tree = grower.grow(dataset, gradients, hessians);

    let start = Instant::now();
    for _ in 0..iterations {
        predictions.fill(0.0);
        tree.predict_batch_add(dataset, predictions);
    }
    start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
}

fn profile_full_training(
    dataset: &BinnedDataset,
    num_rounds: usize,
) -> (f64, f64) {
    let config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(6)
        .with_learning_rate(0.1);

    // Warmup
    let _ = GBDTModel::train_binned(dataset, config.clone());

    let start = Instant::now();
    let _ = GBDTModel::train_binned(dataset, config);
    let total = start.elapsed().as_secs_f64() * 1000.0;
    (total, total / num_rounds as f64)
}

fn main() {
    println!("Large Dataset Profiler");
    println!("======================\n");

    let num_features = 100; // Match "large" benchmark
    let num_rounds = 10;

    // Dataset sizes to test
    let sizes = [
        (100_000, "100k"),
        (250_000, "250k"),
        (500_000, "500k"),
        (750_000, "750k"),
        (1_000_000, "1M"),
    ];

    println!("Configuration: {} features, {} rounds per size\n", num_features, num_rounds);

    // =========================================================================
    // 1. SCALING ANALYSIS
    // =========================================================================
    println!("1. TRAINING TIME SCALING");
    println!("-------------------------");
    println!("{:>8} {:>12} {:>12} {:>12} {:>12}",
        "Rows", "Total (ms)", "Per-round", "ms/100k", "vs 100k");
    println!("{}", "-".repeat(60));

    let mut baseline_per_round = 0.0;
    let mut training_results = Vec::new();

    for (num_rows, label) in &sizes {
        let dataset = create_dataset(*num_rows, num_features);
        let (total, per_round) = profile_full_training(&dataset, num_rounds);

        let ms_per_100k = per_round / (*num_rows as f64 / 100_000.0);

        if baseline_per_round == 0.0 {
            baseline_per_round = per_round;
        }
        let vs_baseline = per_round / baseline_per_round;

        println!("{:>8} {:>12.1} {:>12.2} {:>12.2} {:>12.2}x",
            label, total, per_round, ms_per_100k, vs_baseline);

        training_results.push((*num_rows, total, per_round));
    }

    // =========================================================================
    // 2. COMPONENT BREAKDOWN BY SIZE
    // =========================================================================
    println!("\n2. COMPONENT BREAKDOWN BY SIZE");
    println!("-------------------------------");
    println!("{:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Rows", "Hist(ms)", "TreeGrow", "Gradient", "Predict", "Total");
    println!("{}", "-".repeat(70));

    for (num_rows, label) in &sizes {
        let dataset = create_dataset(*num_rows, num_features);
        let gradients: Vec<f32> = (0..*num_rows)
            .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
            .collect();
        let hessians = vec![1.0f32; *num_rows];
        let row_indices: Vec<usize> = (0..*num_rows).collect();
        let targets = dataset.targets().to_vec();
        let mut predictions = vec![0.0f32; *num_rows];
        let mut g_buf = vec![0.0f32; *num_rows];
        let mut h_buf = vec![0.0f32; *num_rows];

        let iterations = if *num_rows >= 500_000 { 3 } else { 5 };

        let hist_time = profile_histogram_build(&dataset, &row_indices, &gradients, &hessians, iterations);
        let tree_time = profile_tree_grow(&dataset, &gradients, &hessians, &row_indices, iterations);
        let grad_time = profile_gradient_computation(&targets, &predictions, &row_indices, &mut g_buf, &mut h_buf, iterations);
        let pred_time = profile_prediction_update(&dataset, &gradients, &hessians, &mut predictions, iterations);

        let total = hist_time + tree_time + grad_time + pred_time;

        println!("{:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            label, hist_time, tree_time, grad_time, pred_time, total);
    }

    // =========================================================================
    // 3. MEMORY BANDWIDTH ANALYSIS
    // =========================================================================
    println!("\n3. MEMORY BANDWIDTH ANALYSIS");
    println!("-----------------------------");

    // For histogram building, we read:
    // - row_indices: num_rows × 8 bytes
    // - gradients: num_rows × 4 bytes
    // - hessians: num_rows × 4 bytes
    // - bins per feature: num_rows × 1 byte × num_features (through row indices)

    println!("{:>8} {:>12} {:>12} {:>12} {:>12}",
        "Rows", "Data (MB)", "Time (ms)", "BW (GB/s)", "Peak%");
    println!("{}", "-".repeat(60));

    // Assume ~50 GB/s memory bandwidth on typical desktop
    let peak_bandwidth_gbs = 50.0;

    for (num_rows, label) in &sizes {
        let dataset = create_dataset(*num_rows, num_features);
        let gradients: Vec<f32> = (0..*num_rows)
            .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
            .collect();
        let hessians = vec![1.0f32; *num_rows];
        let row_indices: Vec<usize> = (0..*num_rows).collect();

        let iterations = if *num_rows >= 500_000 { 5 } else { 10 };
        let hist_time = profile_histogram_build(&dataset, &row_indices, &gradients, &hessians, iterations);

        // Data accessed per histogram build
        let bytes_accessed = (*num_rows * 8)  // row_indices
            + (*num_rows * 4)  // gradients
            + (*num_rows * 4)  // hessians
            + (*num_rows * num_features);  // bins (through row indices, cache blocked)

        let data_mb = bytes_accessed as f64 / 1_000_000.0;
        let time_s = hist_time / 1000.0;
        let bandwidth_gbs = data_mb / 1000.0 / time_s;
        let peak_pct = (bandwidth_gbs / peak_bandwidth_gbs) * 100.0;

        println!("{:>8} {:>12.1} {:>12.3} {:>12.1} {:>12.1}%",
            label, data_mb, hist_time, bandwidth_gbs, peak_pct);
    }

    // =========================================================================
    // 4. CACHE EFFECTS - SEQUENTIAL VS RANDOM ACCESS
    // =========================================================================
    println!("\n4. CACHE EFFECTS - SEQUENTIAL VS RANDOM ACCESS");
    println!("------------------------------------------------");

    for (num_rows, label) in &sizes[..3] { // First 3 sizes only
        println!("\n{} rows:", label);

        let dataset = create_dataset(*num_rows, num_features);
        let gradients: Vec<f32> = (0..*num_rows)
            .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
            .collect();
        let hessians = vec![1.0f32; *num_rows];

        // Sequential indices
        let seq_indices: Vec<usize> = (0..*num_rows).collect();

        // Random indices
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut rand_indices: Vec<usize> = (0..*num_rows).collect();
        rand_indices.shuffle(&mut rng);

        let iterations = if *num_rows >= 500_000 { 3 } else { 5 };

        let seq_time = profile_histogram_build(&dataset, &seq_indices, &gradients, &hessians, iterations);
        let rand_time = profile_histogram_build(&dataset, &rand_indices, &gradients, &hessians, iterations);

        println!("  Sequential histogram:    {:>8.2} ms", seq_time);
        println!("  Random histogram:        {:>8.2} ms", rand_time);
        println!("  Slowdown from random:    {:>8.2}x", rand_time / seq_time);

        // Tree grow comparison
        let seq_tree = profile_tree_grow(&dataset, &gradients, &hessians, &seq_indices, iterations);
        let rand_tree = profile_tree_grow(&dataset, &gradients, &hessians, &rand_indices, iterations);

        println!("  Sequential tree grow:    {:>8.2} ms", seq_tree);
        println!("  Random tree grow:        {:>8.2} ms", rand_tree);
        println!("  Slowdown from random:    {:>8.2}x", rand_tree / seq_tree);
    }

    // =========================================================================
    // 5. TREE DEPTH VS PERFORMANCE
    // =========================================================================
    println!("\n5. TREE DEPTH VS PERFORMANCE");
    println!("-----------------------------");

    let num_rows = 500_000;
    let dataset = create_dataset(num_rows, num_features);
    let gradients: Vec<f32> = (0..num_rows)
        .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
        .collect();
    let hessians = vec![1.0f32; num_rows];
    let row_indices: Vec<usize> = (0..num_rows).collect();

    println!("{:>6} {:>8} {:>12} {:>12} {:>12}",
        "Depth", "Leaves", "Time (ms)", "ms/leaf", "ms/100k");
    println!("{}", "-".repeat(55));

    for depth in [4, 5, 6, 7, 8] {
        let max_leaves = (1 << depth) - 1;
        let grower = TreeGrower::new()
            .with_max_depth(depth)
            .with_max_leaves(max_leaves)
            .with_learning_rate(0.1);

        // Warmup
        let _ = grower.grow_with_indices(&dataset, &gradients, &hessians, &row_indices);

        let iterations = 3;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = grower.grow_with_indices(&dataset, &gradients, &hessians, &row_indices);
        }
        let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let ms_per_leaf = time / max_leaves as f64;
        let ms_per_100k = time / (num_rows as f64 / 100_000.0);

        println!("{:>6} {:>8} {:>12.2} {:>12.3} {:>12.2}",
            depth, max_leaves, time, ms_per_leaf, ms_per_100k);
    }

    // =========================================================================
    // 6. FEATURE COUNT SCALING
    // =========================================================================
    println!("\n6. FEATURE COUNT SCALING (500k rows)");
    println!("-------------------------------------");

    let num_rows = 500_000;

    println!("{:>10} {:>12} {:>12} {:>12}",
        "Features", "Time (ms)", "ms/feature", "vs 50 feat");
    println!("{}", "-".repeat(50));

    let mut baseline_time = 0.0;

    for num_features in [50, 100, 150, 200] {
        let dataset = create_dataset(num_rows, num_features);
        let gradients: Vec<f32> = (0..num_rows)
            .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
            .collect();
        let hessians = vec![1.0f32; num_rows];
        let row_indices: Vec<usize> = (0..num_rows).collect();

        let iterations = 3;
        let time = profile_tree_grow(&dataset, &gradients, &hessians, &row_indices, iterations);

        if baseline_time == 0.0 {
            baseline_time = time;
        }
        let vs_baseline = time / baseline_time;
        let ms_per_feature = time / num_features as f64;

        println!("{:>10} {:>12.2} {:>12.3} {:>12.2}x",
            num_features, time, ms_per_feature, vs_baseline);
    }

    // =========================================================================
    // 7. LIGHTGBM COMPARISON AT SCALE
    // =========================================================================
    println!("\n7. VS LIGHTGBM AT SCALE");
    println!("-----------------------");

    // LightGBM reference times (from FIX_SPEED_CHECKLIST.md)
    let lgb_times = [
        (100_000, 574.0, "100k"),   // 100k×50, 100 rounds -> 574ms
        (500_000, 2722.0, "500k"),  // 500k×100, 100 rounds -> 2722ms
    ];

    println!("{:>8} {:>12} {:>12} {:>12} {:>12}",
        "Rows", "TreeBoost", "LightGBM", "Slowdown", "Gap/round");
    println!("{}", "-".repeat(60));

    for (num_rows, lgb_time, label) in lgb_times {
        let num_features = if num_rows == 100_000 { 50 } else { 100 };
        let num_rounds = 100;

        let dataset = create_dataset(num_rows, num_features);
        let config = GBDTConfig::new()
            .with_num_rounds(num_rounds)
            .with_max_depth(6)
            .with_learning_rate(0.1);

        // Warmup
        let _ = GBDTModel::train_binned(&dataset, config.clone());

        let start = Instant::now();
        let _ = GBDTModel::train_binned(&dataset, config);
        let our_time = start.elapsed().as_secs_f64() * 1000.0;

        let slowdown = our_time / lgb_time;
        let gap_per_round = (our_time - lgb_time) / num_rounds as f64;

        println!("{:>8} {:>12.1} {:>12.1} {:>12.2}x {:>12.2} ms",
            label, our_time, lgb_time, slowdown, gap_per_round);
    }

    // =========================================================================
    // 8. DETAILED 500K BREAKDOWN
    // =========================================================================
    println!("\n8. DETAILED 500K BREAKDOWN (per round)");
    println!("---------------------------------------");

    let num_rows = 500_000;
    let num_features = 100;
    let dataset = create_dataset(num_rows, num_features);

    let gradients: Vec<f32> = (0..num_rows)
        .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
        .collect();
    let hessians = vec![1.0f32; num_rows];
    let row_indices: Vec<usize> = (0..num_rows).collect();
    let targets = dataset.targets().to_vec();
    let mut predictions = vec![0.0f32; num_rows];
    let mut g_buf = vec![0.0f32; num_rows];
    let mut h_buf = vec![0.0f32; num_rows];

    let iterations = 5;

    let hist_time = profile_histogram_build(&dataset, &row_indices, &gradients, &hessians, iterations);
    let tree_time = profile_tree_grow(&dataset, &gradients, &hessians, &row_indices, iterations);
    let grad_time = profile_gradient_computation(&targets, &predictions, &row_indices, &mut g_buf, &mut h_buf, iterations);
    let pred_time = profile_prediction_update(&dataset, &gradients, &hessians, &mut predictions, iterations);

    let lgb_per_round = 2722.0 / 100.0; // 27.22 ms per round for LightGBM

    println!("Component             TreeBoost      LightGBM*     Gap");
    println!("{}", "-".repeat(55));
    println!("{:<20} {:>10.2} ms", "Root histogram", hist_time);
    println!("{:<20} {:>10.2} ms", "Full tree grow", tree_time);
    println!("{:<20} {:>10.2} ms", "Gradient comp", grad_time);
    println!("{:<20} {:>10.2} ms", "Predict update", pred_time);
    println!("{}", "-".repeat(55));

    let our_per_round = tree_time + grad_time + pred_time;
    println!("{:<20} {:>10.2} ms {:>10.2} ms {:>10.2} ms",
        "TOTAL/round", our_per_round, lgb_per_round, our_per_round - lgb_per_round);
    println!("\n* LightGBM times are estimates from benchmark data");

    // =========================================================================
    // 9. ALLOCATION PROFILING
    // =========================================================================
    println!("\n9. ALLOCATION ANALYSIS");
    println!("-----------------------");

    let num_rows = 500_000;
    let num_features = 100;

    // Measure allocation costs at scale
    let iterations = 10;

    // Row indices allocation
    let start = Instant::now();
    for _ in 0..iterations {
        let v: Vec<usize> = (0..num_rows).collect();
        std::hint::black_box(v);
    }
    let alloc_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("Vec<usize> alloc ({}k): {:>8.3} ms", num_rows/1000, alloc_time);

    // Gradient buffer allocation
    let start = Instant::now();
    for _ in 0..iterations {
        let v = vec![0.0f32; num_rows];
        std::hint::black_box(v);
    }
    let grad_alloc_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("Vec<f32> alloc ({}k):   {:>8.3} ms", num_rows/1000, grad_alloc_time);

    // Histogram allocation
    let start = Instant::now();
    for _ in 0..(iterations * 10) {
        let _ = treeboost::histogram::NodeHistograms::new(num_features);
    }
    let hist_alloc_time = start.elapsed().as_secs_f64() * 1000.0 / (iterations * 10) as f64;
    println!("NodeHistograms alloc:     {:>8.3} ms", hist_alloc_time);

    // Estimate total allocation cost per tree (31 leaves = ~62 NodeHistograms)
    let tree_alloc_cost = hist_alloc_time * 62.0;
    println!("Est. alloc cost/tree:     {:>8.3} ms (62 histograms)", tree_alloc_cost);

    // =========================================================================
    // 10. ISOLATION: WHAT IS SLOW?
    // =========================================================================
    println!("\n10. ISOLATION: WHERE IS THE TIME GOING?");
    println!("-----------------------------------------");

    let num_rows = 500_000;
    let num_features = 100;
    let dataset = create_dataset(num_rows, num_features);

    let gradients: Vec<f32> = (0..num_rows)
        .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
        .collect();
    let hessians = vec![1.0f32; num_rows];

    // Time the actual GBDTModel::train loop
    let config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_max_depth(6)
        .with_learning_rate(0.1);

    let start = Instant::now();
    let _ = GBDTModel::train_binned(&dataset, config.clone());
    let actual_train_10 = start.elapsed().as_secs_f64() * 1000.0;

    // Time isolated tree grows
    let grower = TreeGrower::new()
        .with_max_depth(6)
        .with_max_leaves(31)
        .with_learning_rate(0.1);
    let row_indices: Vec<usize> = (0..num_rows).collect();

    let start = Instant::now();
    for _ in 0..10 {
        let _ = grower.grow_with_indices(&dataset, &gradients, &hessians, &row_indices);
    }
    let isolated_10_trees = start.elapsed().as_secs_f64() * 1000.0;

    println!("10 isolated tree grows:   {:>8.2} ms ({:.2} ms/tree)",
        isolated_10_trees, isolated_10_trees / 10.0);
    println!("GBDTModel::train (10r):   {:>8.2} ms ({:.2} ms/round)",
        actual_train_10, actual_train_10 / 10.0);
    println!("Difference:               {:>8.2} ms ({:.2} ms/round)",
        actual_train_10 - isolated_10_trees, (actual_train_10 - isolated_10_trees) / 10.0);

    let lgb_10_rounds = 2722.0 / 10.0; // ~272ms for 10 rounds
    println!("\nLightGBM (10 rounds):     {:>8.2} ms ({:.2} ms/round)",
        lgb_10_rounds, lgb_10_rounds / 10.0);
    println!("TreeBoost slowdown:       {:>8.2}x", actual_train_10 / lgb_10_rounds);
    println!("Isolated trees slowdown:  {:>8.2}x", isolated_10_trees / lgb_10_rounds);

    // What's in the training loop that's NOT in isolated grow?
    let per_round_overhead = (actual_train_10 - isolated_10_trees) / 10.0;
    println!("\nPer-round overhead breakdown:");
    println!("  Gradient computation:   {:>8.3} ms", grad_time);
    println!("  Prediction update:      {:>8.3} ms", pred_time);
    println!("  Total measured:         {:>8.3} ms", grad_time + pred_time);
    println!("  Actual overhead:        {:>8.3} ms", per_round_overhead);
    println!("  Unexplained:            {:>8.3} ms", per_round_overhead - grad_time - pred_time);
}
