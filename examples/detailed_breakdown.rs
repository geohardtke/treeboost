//! Detailed training time breakdown - measures actual code paths
//!
//! Run: cargo run --release --example detailed_breakdown

use std::time::Instant;

use rayon::prelude::*;
use treeboost::booster::{GBDTConfig, GBDTModel, LossType};
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType, QuantileBinner};
use treeboost::histogram::HistogramBuilder;
use treeboost::tree::{SplitFinder, TreeGrower};

/// Generate raw f32 features (row-major) and targets
fn generate_raw_data(num_rows: usize, num_features: usize) -> (Vec<f32>, Vec<f32>) {
    let features: Vec<f32> = (0..num_rows * num_features)
        .map(|i| ((i * 17) % 1000) as f32 / 1000.0)
        .collect();
    let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32 * 0.01).sin()).collect();
    (features, targets)
}

/// Create pre-binned dataset (for train_binned benchmarks)
fn create_binned_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
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
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                    TREEBOOST DETAILED TRAINING BREAKDOWN");
    println!("═══════════════════════════════════════════════════════════════════════════════\n");

    let num_rows = 100_000;
    let num_features = 50;
    let num_rounds = 10;

    println!("Dataset:  {} rows × {} features", num_rows, num_features);
    println!("Rounds:   {}", num_rounds);
    println!("Max depth: 6, Max leaves: 31\n");

    // =========================================================================
    // PHASE 1: BINNING (raw data → binned)
    // =========================================================================
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!("PHASE 1: BINNING (Converting raw floats → u8 bins)");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    let (raw_features, raw_targets) = generate_raw_data(num_rows, num_features);
    let binner = QuantileBinner::new(255);

    // Sequential binning (old Python bindings behavior)
    let start = Instant::now();
    let mut _binned_seq = Vec::with_capacity(num_rows * num_features);
    for f in 0..num_features {
        let col: Vec<f64> = (0..num_rows)
            .map(|r| raw_features[r * num_features + f] as f64)
            .collect();
        let boundaries = binner.compute_boundaries(&col);
        let binned = binner.bin_column(&col, &boundaries);
        _binned_seq.extend(binned);
    }
    let sequential_binning_time = start.elapsed().as_secs_f64() * 1000.0;
    println!("Sequential binning:        {:>10.2} ms", sequential_binning_time);

    // Parallel binning (new Rust high-level API)
    let start = Instant::now();
    let _binned_par: Vec<Vec<u8>> = (0..num_features)
        .into_par_iter()
        .map(|f| {
            let col: Vec<f64> = (0..num_rows)
                .map(|r| raw_features[r * num_features + f] as f64)
                .collect();
            let boundaries = binner.compute_boundaries(&col);
            binner.bin_column(&col, &boundaries)
        })
        .collect();
    let parallel_binning_time = start.elapsed().as_secs_f64() * 1000.0;
    println!("Parallel binning (Rayon):  {:>10.2} ms", parallel_binning_time);
    println!(
        "Speedup:                   {:>10.1}x",
        sequential_binning_time / parallel_binning_time
    );

    // =========================================================================
    // PHASE 2: HIGH-LEVEL TRAIN (includes binning)
    // =========================================================================
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("PHASE 2: HIGH-LEVEL train() (Raw floats → Trained model)");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    let config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(6)
        .with_max_leaves(31)
        .with_learning_rate(0.1);

    // Warmup
    let _ = GBDTModel::train(&raw_features, num_features, &raw_targets, config.clone(), None);

    let start = Instant::now();
    let _ = GBDTModel::train(&raw_features, num_features, &raw_targets, config.clone(), None);
    let high_level_train_time = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "GBDTModel::train():        {:>10.2} ms (includes binning)",
        high_level_train_time
    );
    println!(
        "  ├─ Binning (parallel):   {:>10.2} ms ({:.1}%)",
        parallel_binning_time,
        parallel_binning_time / high_level_train_time * 100.0
    );
    println!(
        "  └─ Training:             {:>10.2} ms ({:.1}%)",
        high_level_train_time - parallel_binning_time,
        (high_level_train_time - parallel_binning_time) / high_level_train_time * 100.0
    );

    // =========================================================================
    // PHASE 3: LOW-LEVEL TRAIN_BINNED (pre-binned data)
    // =========================================================================
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("PHASE 3: LOW-LEVEL train_binned() (Pre-binned → Trained model)");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    let dataset = create_binned_dataset(num_rows, num_features);

    // Warmup
    let _ = GBDTModel::train_binned(&dataset, config.clone());

    let start = Instant::now();
    let _ = GBDTModel::train_binned(&dataset, config.clone());
    let train_binned_time = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "GBDTModel::train_binned(): {:>10.2} ms",
        train_binned_time
    );
    println!(
        "Per round:                 {:>10.2} ms",
        train_binned_time / num_rounds as f64
    );

    // =========================================================================
    // PHASE 4: PER-ROUND BREAKDOWN
    // =========================================================================
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("PHASE 4: PER-ROUND BREAKDOWN (What happens each boosting iteration)");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    let loss_fn = LossType::Mse.create();
    let targets = dataset.targets();
    let train_indices: Vec<usize> = (0..num_rows).collect();

    let mut predictions = vec![0.0f32; num_rows];
    let mut gradients = vec![0.0f32; num_rows];
    let mut hessians = vec![0.0f32; num_rows];

    let grower = TreeGrower::new()
        .with_max_depth(6)
        .with_max_leaves(31)
        .with_learning_rate(0.1);

    // Measure each component
    let iterations = 10;
    let mut gradient_times = Vec::with_capacity(iterations);
    let mut tree_grow_times = Vec::with_capacity(iterations);
    let mut predict_times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        // 1. Gradient computation
        let start = Instant::now();
        for &idx in &train_indices {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            gradients[idx] = g;
            hessians[idx] = h;
        }
        gradient_times.push(start.elapsed().as_secs_f64() * 1000.0);

        // 2. Tree grow (FUSED path - gradient+histogram in single pass)
        let start = Instant::now();
        let tree = grower.grow_fused(
            &dataset,
            &train_indices,
            targets,
            &predictions,
            loss_fn.as_ref(),
            &mut gradients,
            &mut hessians,
        );
        tree_grow_times.push(start.elapsed().as_secs_f64() * 1000.0);

        // 3. Prediction update
        let start = Instant::now();
        tree.predict_batch_add(&dataset, &mut predictions);
        predict_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let avg_gradient = gradient_times.iter().sum::<f64>() / iterations as f64;
    let avg_tree_grow = tree_grow_times.iter().sum::<f64>() / iterations as f64;
    let avg_predict = predict_times.iter().sum::<f64>() / iterations as f64;
    let avg_total = avg_gradient + avg_tree_grow + avg_predict;

    println!("FUSED PATH (no subsampling, no GOSS):");
    println!(
        "  1. Gradient computation: {:>10.3} ms ({:.1}%)",
        avg_gradient,
        avg_gradient / avg_total * 100.0
    );
    println!(
        "  2. Tree grow (fused):    {:>10.3} ms ({:.1}%)",
        avg_tree_grow,
        avg_tree_grow / avg_total * 100.0
    );
    println!(
        "  3. Prediction update:    {:>10.3} ms ({:.1}%)",
        avg_predict,
        avg_predict / avg_total * 100.0
    );
    println!("  ─────────────────────────────────────");
    println!("  Total per round:         {:>10.3} ms", avg_total);

    // =========================================================================
    // PHASE 5: TREE GROW INTERNAL BREAKDOWN
    // =========================================================================
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("PHASE 5: TREE GROW INTERNAL BREAKDOWN");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    // Prepare gradients for histogram building
    for &idx in &train_indices {
        let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
        gradients[idx] = g;
        hessians[idx] = h;
    }

    let builder = HistogramBuilder::new();
    let split_finder = SplitFinder::new()
        .with_lambda(1.0)
        .with_min_samples_leaf(1)
        .with_min_hessian_leaf(1.0);

    // Histogram building at various node sizes
    // Note: Real child nodes have scattered indices, not contiguous!
    // We test BOTH cases to show the difference
    println!("Histogram building (scales with node size):");
    println!("  [Contiguous path - root node]");
    let sizes = [
        (num_rows, "root"),
        (50_000, "level 1"),
        (25_000, "level 2"),
        (12_500, "level 3"),
        (6_250, "level 4"),
        (3_125, "level 5"),
    ];

    let mut hist_times = Vec::new();
    for (size, label) in sizes.iter() {
        let subset: Vec<usize> = (0..*size).collect();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = builder.build(&dataset, &subset, &gradients[..*size], &hessians[..*size]);
        }
        let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        hist_times.push(time);
        println!("  {:>6} rows ({}):    {:>10.3} ms", size, label, time);
    }

    // Test with SCATTERED indices (simulates real child nodes after partition)
    println!("\n  [Scattered path - child nodes after partition]");
    let mut scattered_hist_times = Vec::new();
    for (size, label) in sizes.iter().skip(1) {
        // Create scattered indices: every other row
        let scattered: Vec<usize> = (0..num_rows).step_by(num_rows / size).take(*size).collect();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = builder.build(&dataset, &scattered, &gradients, &hessians);
        }
        let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        scattered_hist_times.push(time);
        println!("  {:>6} rows ({}):    {:>10.3} ms (scattered)", size, label, time);
    }

    // Histogram subtraction
    let parent_hist = builder.build(&dataset, &train_indices, &gradients, &hessians);
    let half: Vec<usize> = (0..num_rows / 2).collect();
    let child_hist = builder.build(&dataset, &half, &gradients[..num_rows / 2], &hessians[..num_rows / 2]);

    let start = Instant::now();
    for _ in 0..iterations * 100 {
        let _ = HistogramBuilder::build_sibling(&parent_hist, &child_hist);
    }
    let subtract_time = start.elapsed().as_secs_f64() * 1000.0 / (iterations * 100) as f64;
    println!("\nHistogram subtraction:     {:>10.4} ms (nearly free!)", subtract_time);

    // Split finding
    let total_grad: f32 = gradients.iter().sum();
    let total_hess: f32 = hessians.iter().sum();

    let start = Instant::now();
    for _ in 0..iterations * 10 {
        let _ = split_finder.find_best_split(&parent_hist, total_grad, total_hess, num_rows as u32);
    }
    let split_time = start.elapsed().as_secs_f64() * 1000.0 / (iterations * 10) as f64;
    println!("Split finding (per node):  {:>10.4} ms", split_time);

    // Tree cost estimate (31-leaf tree)
    // With subtraction trick: build smaller child, subtract for larger
    // Level 0: 1 histogram @ 100k (contiguous - root)
    // Level 1: 1 histogram @ 50k (scattered - smaller child)
    // Level 2: 2 histograms @ 25k (scattered)
    // Level 3: 4 histograms @ 12.5k (scattered)
    // Level 4: 8 histograms @ 6.25k (scattered)
    // Total: ~15-16 histogram builds + 15 subtractions + 30 split finds

    // Use scattered times for child nodes (they have non-contiguous indices!)
    let estimated_hist_cost = hist_times[0]  // root (contiguous)
        + scattered_hist_times[0]            // level 1 (scattered)
        + 2.0 * scattered_hist_times[1]      // level 2
        + 4.0 * scattered_hist_times[2]      // level 3
        + 8.0 * scattered_hist_times[3];     // level 4
    let estimated_split_cost = split_time * 30.0;
    let estimated_subtract_cost = subtract_time * 15.0;
    let estimated_total = estimated_hist_cost + estimated_split_cost + estimated_subtract_cost;

    println!("\n31-leaf tree cost estimate:");
    println!("  Histogram builds:        {:>10.3} ms", estimated_hist_cost);
    println!("  Split finding (30×):     {:>10.3} ms", estimated_split_cost);
    println!("  Subtraction (15×):       {:>10.4} ms", estimated_subtract_cost);
    println!("  ─────────────────────────────────────");
    println!("  Estimated total:         {:>10.3} ms", estimated_total);
    println!("  Actual tree grow:        {:>10.3} ms", avg_tree_grow);
    println!(
        "  Unaccounted:             {:>10.3} ms (row partitioning, allocations)",
        avg_tree_grow - estimated_total
    );

    // =========================================================================
    // PHASE 6: COMPARISON WITH LIGHTGBM
    // =========================================================================
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("PHASE 6: COMPARISON WITH LIGHTGBM");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    // LightGBM benchmark reference (100k×50, 100 rounds)
    let lgb_total = 258.0; // ms from benchmark
    let lgb_per_round = lgb_total / 100.0;

    let our_per_round = train_binned_time / num_rounds as f64;
    let our_100_rounds = our_per_round * 100.0;

    println!("100k×50 dataset, 100 rounds:");
    println!("  LightGBM:                {:>10.1} ms ({:.2} ms/round)", lgb_total, lgb_per_round);
    println!(
        "  TreeBoost:               {:>10.1} ms ({:.2} ms/round)",
        our_100_rounds, our_per_round
    );
    println!(
        "  Gap:                     {:>10.1}x slower",
        our_per_round / lgb_per_round
    );

    println!("\nPer-round comparison:");
    println!("  LightGBM per round:      {:>10.2} ms", lgb_per_round);
    println!("  TreeBoost per round:     {:>10.2} ms", our_per_round);
    println!(
        "  Difference:              {:>10.2} ms",
        our_per_round - lgb_per_round
    );

    // =========================================================================
    // PHASE 7: WHERE IS THE TIME GOING?
    // =========================================================================
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("PHASE 7: TIME DISTRIBUTION SUMMARY");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    let binning_pct = parallel_binning_time / high_level_train_time * 100.0;
    let training_pct = 100.0 - binning_pct;

    let hist_pct = estimated_hist_cost / avg_tree_grow * 100.0;
    let split_pct = estimated_split_cost / avg_tree_grow * 100.0;
    let other_pct = 100.0 - hist_pct - split_pct;

    println!("HIGH-LEVEL train() breakdown:");
    println!("  ┌─ Binning:               {:>6.1}% ({:.1} ms)", binning_pct, parallel_binning_time);
    println!("  └─ Training:             {:>6.1}% ({:.1} ms)", training_pct, high_level_train_time - parallel_binning_time);

    println!("\nPer-round breakdown:");
    println!("  ┌─ Tree grow:             {:>6.1}% ({:.2} ms)", avg_tree_grow / avg_total * 100.0, avg_tree_grow);
    println!("  │   ├─ Histogram build:   {:>6.1}%", hist_pct);
    println!("  │   ├─ Split finding:     {:>6.1}%", split_pct);
    println!("  │   └─ Other (partition): {:>6.1}%", other_pct);
    println!("  ├─ Gradient compute:     {:>6.1}% ({:.2} ms)", avg_gradient / avg_total * 100.0, avg_gradient);
    println!("  └─ Prediction update:    {:>6.1}% ({:.2} ms)", avg_predict / avg_total * 100.0, avg_predict);

    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("OPTIMIZATION TARGETS:");
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!();
    println!("1. HISTOGRAM BUILDING ({:.1} ms, {:.0}% of tree grow)", estimated_hist_cost, hist_pct);
    println!("   - Current: Feature-parallel with Rayon");
    println!("   - LightGBM: SIMD-optimized inner loop (AVX2)");
    println!("   - Potential: 2-4x speedup with explicit SIMD");
    println!();
    println!("2. ROW PARTITIONING ({:.1} ms, {:.0}% of tree grow)", avg_tree_grow - estimated_total, other_pct);
    println!("   - Current: In-place partitioning");
    println!("   - LightGBM: Bin-based sorting, cache-aware");
    println!("   - Potential: Review memory access patterns");
    println!();
    println!("3. GRADIENT COMPUTATION ({:.2} ms)", avg_gradient);
    println!("   - Current: Fused with histogram in single pass");
    println!("   - Already optimized via fused path!");
    println!();
}
