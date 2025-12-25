//! Detailed training time breakdown - measures actual code paths
//!
//! Run: cargo run --release --example detailed_breakdown

use std::time::Instant;

use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use treeboost::histogram::HistogramBuilder;
use treeboost::tree::{SplitFinder, TreeGrower};
use treeboost::booster::{GBDTConfig, GBDTModel, LossType};

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
    println!("Detailed Training Breakdown");
    println!("============================\n");

    let num_rows = 100_000;
    let num_features = 50;
    let num_rounds = 10;

    println!("Dataset: {} rows × {} features", num_rows, num_features);
    println!("Rounds: {}\n", num_rounds);

    let dataset = create_dataset(num_rows, num_features);

    // Create gradients that encourage splitting
    let gradients: Vec<f32> = (0..num_rows)
        .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
        .collect();
    let hessians: Vec<f32> = vec![1.0; num_rows];

    // =========================================================================
    // 1. Single tree grow (what we optimized)
    // =========================================================================
    println!("1. SINGLE TREE GROW");
    println!("-------------------");

    let grower = TreeGrower::new()
        .with_max_depth(6)
        .with_max_leaves(31)
        .with_learning_rate(0.1);

    // Warmup
    for _ in 0..3 {
        let _ = grower.grow(&dataset, &gradients, &hessians);
    }

    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = grower.grow(&dataset, &gradients, &hessians);
    }
    let tree_grow_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("Tree grow (avg of {}):     {:>8.3} ms", iterations, tree_grow_time);

    // =========================================================================
    // 2. Full training loop components
    // =========================================================================
    println!("\n2. FULL TRAINING LOOP BREAKDOWN");
    println!("--------------------------------");

    let config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(6)
        .with_learning_rate(0.1);

    // Warmup
    let _ = GBDTModel::train(&dataset, config.clone());

    // Time full training
    let start = Instant::now();
    let _ = GBDTModel::train(&dataset, config.clone());
    let total_train_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Total training time:       {:>8.3} ms", total_train_time);
    println!("Per round:                 {:>8.3} ms", total_train_time / num_rounds as f64);
    println!("Tree grow portion:         {:>8.3} ms ({:.1}%)",
        tree_grow_time * num_rounds as f64,
        (tree_grow_time * num_rounds as f64 / total_train_time) * 100.0);

    // =========================================================================
    // 3. Gradient + Prediction update
    // =========================================================================
    println!("\n3. GRADIENT & PREDICTION UPDATE");
    println!("--------------------------------");

    let tree = grower.grow(&dataset, &gradients, &hessians);
    let mut predictions = vec![0.0f32; num_rows];
    let targets = dataset.targets();

    // Gradient computation
    let mut grads = vec![0.0f32; num_rows];
    let mut hesss = vec![0.0f32; num_rows];

    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..num_rows {
            let residual = targets[i] - predictions[i];
            grads[i] = -residual;
            hesss[i] = 1.0;
        }
    }
    let grad_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("Gradient computation:      {:>8.3} ms", grad_time);

    // Prediction update (batch add)
    let start = Instant::now();
    for _ in 0..iterations {
        predictions.fill(0.0);
        tree.predict_batch_add(&dataset, &mut predictions);
    }
    let predict_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("Prediction update:         {:>8.3} ms", predict_time);

    // =========================================================================
    // 4. Histogram building analysis
    // =========================================================================
    println!("\n4. HISTOGRAM BUILDING ANALYSIS");
    println!("-------------------------------");

    let builder = HistogramBuilder::new();
    let row_indices: Vec<usize> = (0..num_rows).collect();

    // Full histogram build
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = builder.build(&dataset, &row_indices, &gradients, &hessians);
    }
    let full_hist_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!("Full histogram ({}k rows): {:>8.3} ms", num_rows/1000, full_hist_time);

    // Histogram at different sizes (simulating tree levels)
    let sizes = [50000, 25000, 12500, 6250, 3125, 1562];
    let mut total_hist_estimate = full_hist_time;

    for &size in &sizes {
        let subset: Vec<usize> = (0..size).collect();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = builder.build(&dataset, &subset, &gradients[..size], &hessians[..size]);
        }
        let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        println!("Histogram ({}k rows):      {:>8.3} ms", size/1000, time);
        // Subtraction trick: we only build the smaller child
        // At each level we have 2^level nodes, but only build half
    }

    // Histogram subtraction
    let parent_hist = builder.build(&dataset, &row_indices, &gradients, &hessians);
    let half_rows: Vec<usize> = (0..num_rows/2).collect();
    let child_hist = builder.build(&dataset, &half_rows, &gradients[..num_rows/2], &hessians[..num_rows/2]);

    let start = Instant::now();
    for _ in 0..(iterations * 100) {
        let _ = HistogramBuilder::build_sibling(&parent_hist, &child_hist);
    }
    let subtract_time = start.elapsed().as_secs_f64() * 1000.0 / (iterations * 100) as f64;
    println!("Histogram subtraction:     {:>8.4} ms", subtract_time);

    // =========================================================================
    // 5. Split finding analysis
    // =========================================================================
    println!("\n5. SPLIT FINDING ANALYSIS");
    println!("--------------------------");

    let histograms = builder.build(&dataset, &row_indices, &gradients, &hessians);
    let split_finder = SplitFinder::new()
        .with_lambda(1.0)
        .with_min_samples_leaf(1)
        .with_min_hessian_leaf(1.0);

    let total_grad: f32 = gradients.iter().sum();
    let total_hess: f32 = hessians.iter().sum();

    let start = Instant::now();
    for _ in 0..(iterations * 10) {
        let _ = split_finder.find_best_split(&histograms, total_grad, total_hess, num_rows as u32);
    }
    let split_time = start.elapsed().as_secs_f64() * 1000.0 / (iterations * 10) as f64;
    println!("Split finding (1 node):    {:>8.4} ms", split_time);
    println!("Split finding (30 nodes):  {:>8.3} ms", split_time * 30.0);

    // =========================================================================
    // 6. WHAT'S IN THE ACTUAL TREE GROW?
    // =========================================================================
    println!("\n6. TREE GROW COST BREAKDOWN");
    println!("----------------------------");

    // With subtraction trick, a 31-leaf tree needs:
    // - 1 root histogram (100k rows)
    // - ~15 smaller child histograms (using subtraction for the rest)
    // - 30 split findings (one per internal node)
    // - 30 histogram subtractions
    // - 30 partitions (now in-place)

    // Estimate histogram cost with subtraction trick
    // Level 0: 1 hist @ 100k = full_hist_time
    // Level 1: 1 hist @ 50k (smaller child)
    // Level 2: 2 hist @ 25k
    // Level 3: 4 hist @ 12.5k
    // Level 4: 8 hist @ 6.25k
    // Level 5: 15 hist @ 3k (remaining leaves)

    // Get times for each size
    let subset_50k: Vec<usize> = (0..50000).collect();
    let start = Instant::now();
    for _ in 0..iterations { let _ = builder.build(&dataset, &subset_50k, &gradients[..50000], &hessians[..50000]); }
    let hist_50k = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    let subset_25k: Vec<usize> = (0..25000).collect();
    let start = Instant::now();
    for _ in 0..iterations { let _ = builder.build(&dataset, &subset_25k, &gradients[..25000], &hessians[..25000]); }
    let hist_25k = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    let subset_12k: Vec<usize> = (0..12500).collect();
    let start = Instant::now();
    for _ in 0..iterations { let _ = builder.build(&dataset, &subset_12k, &gradients[..12500], &hessians[..12500]); }
    let hist_12k = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    let subset_6k: Vec<usize> = (0..6250).collect();
    let start = Instant::now();
    for _ in 0..iterations { let _ = builder.build(&dataset, &subset_6k, &gradients[..6250], &hessians[..6250]); }
    let hist_6k = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    let subset_3k: Vec<usize> = (0..3125).collect();
    let start = Instant::now();
    for _ in 0..iterations { let _ = builder.build(&dataset, &subset_3k, &gradients[..3125], &hessians[..3125]); }
    let hist_3k = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    let estimated_hist_total = full_hist_time + hist_50k + 2.0*hist_25k + 4.0*hist_12k + 8.0*hist_6k;
    let estimated_split_total = split_time * 30.0;
    let estimated_subtract_total = subtract_time * 30.0;

    println!("Histogram builds:          {:>8.3} ms", estimated_hist_total);
    println!("  Root (100k):             {:>8.3} ms", full_hist_time);
    println!("  Level 1 (50k×1):         {:>8.3} ms", hist_50k);
    println!("  Level 2 (25k×2):         {:>8.3} ms", 2.0*hist_25k);
    println!("  Level 3 (12k×4):         {:>8.3} ms", 4.0*hist_12k);
    println!("  Level 4 (6k×8):          {:>8.3} ms", 8.0*hist_6k);
    println!("Split finding (30×):       {:>8.3} ms", estimated_split_total);
    println!("Histogram subtraction(30×):{:>8.3} ms", estimated_subtract_total);

    let accounted = estimated_hist_total + estimated_split_total + estimated_subtract_total;
    let unaccounted = tree_grow_time - accounted;

    println!("---");
    println!("ACCOUNTED:                 {:>8.3} ms", accounted);
    println!("ACTUAL TREE GROW:          {:>8.3} ms", tree_grow_time);
    println!("UNACCOUNTED OVERHEAD:      {:>8.3} ms ({:.1}%)",
        unaccounted, (unaccounted / tree_grow_time) * 100.0);

    // =========================================================================
    // 7. LightGBM comparison
    // =========================================================================
    println!("\n7. VS LIGHTGBM");
    println!("--------------");
    let lgb_train = 268.0; // ms for 100 rounds on 100k×50
    let lgb_per_round = lgb_train / 100.0;
    let our_per_round = total_train_time / num_rounds as f64;

    println!("LightGBM (100k×50, 100r):  {:>8.1} ms total", lgb_train);
    println!("LightGBM per round:        {:>8.3} ms", lgb_per_round);
    println!("TreeBoost per round:       {:>8.3} ms", our_per_round);
    println!("Slowdown:                  {:>8.1}x", our_per_round / lgb_per_round);

    println!("\nPer-round breakdown:");
    println!("  Tree grow:               {:>8.3} ms", tree_grow_time);
    println!("  Gradient:                {:>8.3} ms", grad_time);
    println!("  Predict update:          {:>8.3} ms", predict_time);
    println!("  Total estimated:         {:>8.3} ms", tree_grow_time + grad_time + predict_time);

    // =========================================================================
    // 8. MANUAL TRAINING LOOP (like GBDTModel::train but with timing)
    // =========================================================================
    println!("\n8. MANUAL TRAINING LOOP BREAKDOWN");
    println!("----------------------------------");

    let loss_fn = LossType::Mse.create();
    let targets = dataset.targets();
    let train_indices: Vec<usize> = (0..num_rows).collect();

    // Compute base prediction
    let start = Instant::now();
    let base_prediction = 0.0f32; // simplified
    let mut predictions = vec![base_prediction; num_rows];
    let mut gradients = vec![0.0f32; num_rows];
    let mut hessians = vec![0.0f32; num_rows];
    let setup_time = start.elapsed().as_secs_f64() * 1000.0;
    println!("Setup (alloc buffers):     {:>8.3} ms", setup_time);

    let grower = TreeGrower::new()
        .with_max_depth(6)
        .with_max_leaves(31)
        .with_learning_rate(0.1);

    let mut total_gradient_time = 0.0;
    let mut total_tree_grow_time = 0.0;
    let mut total_predict_time = 0.0;
    let mut total_sample_time = 0.0;

    for _round in 0..num_rounds {
        // Gradient computation
        let start = Instant::now();
        for &idx in &train_indices {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            gradients[idx] = g;
            hessians[idx] = h;
        }
        total_gradient_time += start.elapsed().as_secs_f64() * 1000.0;

        // Sample indices (no subsampling, just copy)
        let start = Instant::now();
        let sample_indices: Vec<usize> = train_indices.clone();
        total_sample_time += start.elapsed().as_secs_f64() * 1000.0;

        // Tree grow
        let start = Instant::now();
        let tree = grower.grow_with_indices(&dataset, &gradients, &hessians, &sample_indices);
        total_tree_grow_time += start.elapsed().as_secs_f64() * 1000.0;

        // Prediction update
        let start = Instant::now();
        tree.predict_batch_add(&dataset, &mut predictions);
        total_predict_time += start.elapsed().as_secs_f64() * 1000.0;
    }

    println!("Gradient comp (total):     {:>8.3} ms ({:.3} ms/round)",
        total_gradient_time, total_gradient_time / num_rounds as f64);
    println!("Sample indices (total):    {:>8.3} ms ({:.3} ms/round)",
        total_sample_time, total_sample_time / num_rounds as f64);
    println!("Tree grow (total):         {:>8.3} ms ({:.3} ms/round)",
        total_tree_grow_time, total_tree_grow_time / num_rounds as f64);
    println!("Predict update (total):    {:>8.3} ms ({:.3} ms/round)",
        total_predict_time, total_predict_time / num_rounds as f64);

    let manual_total = total_gradient_time + total_sample_time + total_tree_grow_time + total_predict_time;
    println!("---");
    println!("MANUAL TOTAL:              {:>8.3} ms", manual_total);
    println!("GBDTModel::train:          {:>8.3} ms", total_train_time);
    println!("DIFFERENCE:                {:>8.3} ms", total_train_time - manual_total);

    // =========================================================================
    // 9. SHUFFLED INDICES TEST (what GBDTModel::train does)
    // =========================================================================
    println!("\n9. SHUFFLED INDICES IMPACT");
    println!("---------------------------");

    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let mut shuffled_indices: Vec<usize> = (0..num_rows).collect();
    shuffled_indices.shuffle(&mut rng);

    let mut predictions = vec![0.0f32; num_rows];
    let mut gradients = vec![0.0f32; num_rows];
    let mut hessians = vec![0.0f32; num_rows];

    let mut total_gradient_shuffled = 0.0;
    let mut total_tree_grow_shuffled = 0.0;
    let mut total_predict_shuffled = 0.0;

    for _round in 0..num_rounds {
        // Gradient computation with SHUFFLED indices
        let start = Instant::now();
        for &idx in &shuffled_indices {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            gradients[idx] = g;
            hessians[idx] = h;
        }
        total_gradient_shuffled += start.elapsed().as_secs_f64() * 1000.0;

        // Tree grow with SHUFFLED indices
        let start = Instant::now();
        let tree = grower.grow_with_indices(&dataset, &gradients, &hessians, &shuffled_indices);
        total_tree_grow_shuffled += start.elapsed().as_secs_f64() * 1000.0;

        // Prediction update
        let start = Instant::now();
        tree.predict_batch_add(&dataset, &mut predictions);
        total_predict_shuffled += start.elapsed().as_secs_f64() * 1000.0;
    }

    println!("With SEQUENTIAL indices:");
    println!("  Gradient:                {:>8.3} ms/round", total_gradient_time / num_rounds as f64);
    println!("  Tree grow:               {:>8.3} ms/round", total_tree_grow_time / num_rounds as f64);
    println!("  Predict:                 {:>8.3} ms/round", total_predict_time / num_rounds as f64);
    println!("\nWith SHUFFLED indices (like GBDTModel::train):");
    println!("  Gradient:                {:>8.3} ms/round", total_gradient_shuffled / num_rounds as f64);
    println!("  Tree grow:               {:>8.3} ms/round", total_tree_grow_shuffled / num_rounds as f64);
    println!("  Predict:                 {:>8.3} ms/round", total_predict_shuffled / num_rounds as f64);

    let seq_total = (total_gradient_time + total_tree_grow_time + total_predict_time) / num_rounds as f64;
    let shuf_total = (total_gradient_shuffled + total_tree_grow_shuffled + total_predict_shuffled) / num_rounds as f64;
    println!("\nTotal per round:");
    println!("  Sequential:              {:>8.3} ms", seq_total);
    println!("  Shuffled:                {:>8.3} ms", shuf_total);
    println!("  Slowdown:                {:>8.1}x", shuf_total / seq_total);

    // =========================================================================
    // 10. GBDTMODEL::TRAIN OVERHEAD INVESTIGATION
    // =========================================================================
    println!("\n10. GBDTModel::train OVERHEAD BREAKDOWN");
    println!("----------------------------------------");

    // Measure what GBDTModel::train does that we're not measuring

    // 1. split_for_training
    let start = Instant::now();
    for _ in 0..100 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut indices: Vec<usize> = (0..num_rows).collect();
        indices.shuffle(&mut rng);
        std::hint::black_box(&indices);
    }
    let split_time = start.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("split_for_training (shuffle): {:>8.3} ms", split_time);

    // 2. initial_prediction computation
    let start = Instant::now();
    for _ in 0..100 {
        let train_targets: Vec<f32> = (0..num_rows).map(|i| targets[i]).collect();
        let _base = train_targets.iter().sum::<f32>() / train_targets.len() as f32;
    }
    let init_pred_time = start.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("initial_prediction:           {:>8.3} ms", init_pred_time);

    // 3. TreeGrower setup
    let start = Instant::now();
    for _ in 0..1000 {
        let _grower = TreeGrower::new()
            .with_max_depth(6)
            .with_max_leaves(31)
            .with_lambda(1.0)
            .with_min_samples_leaf(1)
            .with_min_hessian_leaf(1.0)
            .with_entropy_weight(0.0)
            .with_min_gain(0.0)
            .with_learning_rate(0.1)
            .with_colsample(1.0);
    }
    let grower_setup_time = start.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    println!("TreeGrower setup:             {:>8.4} ms", grower_setup_time);

    // 4. sample_indices.extend_from_slice (what happens when no subsampling)
    let mut sample_buf: Vec<usize> = Vec::with_capacity(num_rows);
    let start = Instant::now();
    for _ in 0..100 {
        sample_buf.clear();
        sample_buf.extend_from_slice(&shuffled_indices);
    }
    let extend_time = start.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("sample_indices extend:        {:>8.3} ms", extend_time);

    // 5. trees.push(tree) - Vec allocation for tree storage
    let mut trees_vec: Vec<treeboost::tree::Tree> = Vec::with_capacity(num_rounds);
    let tree = grower.grow(&dataset, &gradients, &hessians);
    let start = Instant::now();
    for _ in 0..1000 {
        trees_vec.push(tree.clone());
    }
    let push_time = start.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    println!("trees.push (clone):           {:>8.4} ms", push_time);

    // 6. Full GBDTModel::train with timing points
    println!("\nComparing 10 rounds:");

    // What we measured in manual loop
    let manual_per_round = (total_gradient_time + total_sample_time + total_tree_grow_time + total_predict_time) / num_rounds as f64;

    // What GBDTModel::train does extra per round:
    // - Nothing per round that we haven't measured, EXCEPT...

    // Check if it's the sample_indices.clone() vs extend_from_slice
    let start = Instant::now();
    for _ in 0..100 {
        let _cloned: Vec<usize> = shuffled_indices.clone();
    }
    let clone_time = start.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("Vec::clone (100k usize):      {:>8.3} ms", clone_time);

    // One-time setup costs
    let one_time = split_time + init_pred_time + grower_setup_time;
    println!("\nOne-time setup costs:         {:>8.3} ms", one_time);
    println!("Per-round overhead (extend):  {:>8.3} ms × {} = {:.3} ms",
        extend_time, num_rounds, extend_time * num_rounds as f64);

    let total_overhead = one_time + extend_time * num_rounds as f64;
    println!("Total estimated overhead:     {:>8.3} ms", total_overhead);
    println!("Actual difference:            {:>8.3} ms", total_train_time - manual_total);
    println!("Still unexplained:            {:>8.3} ms",
        (total_train_time - manual_total) - total_overhead);

    // =========================================================================
    // 11. COMPARE MANUAL LOOP WITH SHUFFLED INDICES
    // =========================================================================
    println!("\n11. MANUAL LOOP WITH SHUFFLED INDICES");
    println!("--------------------------------------");
    println!("(This should match GBDTModel::train more closely)");

    // Reset state
    let mut predictions = vec![0.0f32; num_rows];
    let mut gradients = vec![0.0f32; num_rows];
    let mut hessians = vec![0.0f32; num_rows];
    let mut sample_indices_buf: Vec<usize> = Vec::with_capacity(num_rows);

    let start_full = Instant::now();

    // One-time: shuffle indices (what split_for_training does)
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let mut train_indices_shuffled: Vec<usize> = (0..num_rows).collect();
    train_indices_shuffled.shuffle(&mut rng);

    let mut loop_gradient_time = 0.0;
    let mut loop_sample_time = 0.0;
    let mut loop_tree_time = 0.0;
    let mut loop_predict_time = 0.0;

    for _round in 0..num_rounds {
        // Gradient computation with SHUFFLED indices
        let start = Instant::now();
        for &idx in &train_indices_shuffled {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            gradients[idx] = g;
            hessians[idx] = h;
        }
        loop_gradient_time += start.elapsed().as_secs_f64() * 1000.0;

        // Sample indices (extend from shuffled)
        let start = Instant::now();
        sample_indices_buf.clear();
        sample_indices_buf.extend_from_slice(&train_indices_shuffled);
        loop_sample_time += start.elapsed().as_secs_f64() * 1000.0;

        // Tree grow with SHUFFLED indices
        let start = Instant::now();
        let tree = grower.grow_with_indices(&dataset, &gradients, &hessians, &sample_indices_buf);
        loop_tree_time += start.elapsed().as_secs_f64() * 1000.0;

        // Prediction update
        let start = Instant::now();
        tree.predict_batch_add(&dataset, &mut predictions);
        loop_predict_time += start.elapsed().as_secs_f64() * 1000.0;
    }

    let full_loop_time = start_full.elapsed().as_secs_f64() * 1000.0;

    println!("Per-round breakdown (shuffled indices):");
    println!("  Gradient:                {:>8.3} ms", loop_gradient_time / num_rounds as f64);
    println!("  Sample indices:          {:>8.3} ms", loop_sample_time / num_rounds as f64);
    println!("  Tree grow:               {:>8.3} ms", loop_tree_time / num_rounds as f64);
    println!("  Predict:                 {:>8.3} ms", loop_predict_time / num_rounds as f64);

    let loop_total = loop_gradient_time + loop_sample_time + loop_tree_time + loop_predict_time;
    println!("\nTotals (10 rounds):");
    println!("  Sum of components:       {:>8.3} ms", loop_total);
    println!("  Full loop measured:      {:>8.3} ms", full_loop_time);
    println!("  Timing overhead:         {:>8.3} ms", full_loop_time - loop_total);

    println!("\nComparison:");
    println!("  Manual (shuffled):       {:>8.3} ms", full_loop_time);
    println!("  GBDTModel::train:        {:>8.3} ms", total_train_time);
    println!("  Difference:              {:>8.3} ms", total_train_time - full_loop_time);
}
