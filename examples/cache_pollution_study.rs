//! Cache pollution study
//!
//! Run: cargo run --release --example cache_pollution_study
//!
//! Investigates why tree grow is 2x slower in training loop vs isolated

use std::time::Instant;

use treeboost::booster::LossType;
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
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

fn main() {
    println!("Cache Pollution Study");
    println!("======================\n");

    for &(num_rows, num_features, label) in &[
        (100_000, 50, "Medium (100k×50)"),
        (500_000, 100, "Large (500k×100)"),
    ] {
        println!("\n{}", label);
        println!("{}", "=".repeat(label.len()));

        let dataset = create_dataset(num_rows, num_features);
        let targets = dataset.targets().to_vec();
        let loss_fn = LossType::Mse.create();
        let num_rounds = 10;

        let grower = TreeGrower::new()
            .with_max_depth(6)
            .with_max_leaves(31)
            .with_learning_rate(0.1);

        let fixed_grads: Vec<f32> = (0..num_rows)
            .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
            .collect();
        let fixed_hess = vec![1.0f32; num_rows];
        let row_indices: Vec<usize> = (0..num_rows).collect();

        // Warmup
        for _ in 0..2 {
            let _ = grower.grow_with_indices(&dataset, &fixed_grads, &fixed_hess, &row_indices);
        }

        // =========================================================================
        // SCENARIO A: Isolated tree grows (same gradients, hot cache)
        // =========================================================================
        let start = Instant::now();
        for _ in 0..num_rounds {
            let _ = grower.grow_with_indices(&dataset, &fixed_grads, &fixed_hess, &row_indices);
        }
        let scenario_a = start.elapsed().as_secs_f64() * 1000.0;

        // =========================================================================
        // SCENARIO B: Gradient computation BETWEEN tree grows (no prediction)
        // =========================================================================
        let mut gradients = vec![0.0f32; num_rows];
        let mut hessians = vec![0.0f32; num_rows];
        let predictions = vec![0.0f32; num_rows];

        let start = Instant::now();
        for _ in 0..num_rounds {
            // Compute gradients (touches gradient/hessian arrays)
            for &idx in &row_indices {
                let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                gradients[idx] = g;
                hessians[idx] = h;
            }
            // Tree grow with fresh gradients
            let _ = grower.grow_with_indices(&dataset, &gradients, &hessians, &row_indices);
        }
        let scenario_b = start.elapsed().as_secs_f64() * 1000.0;

        // =========================================================================
        // SCENARIO C: Gradient + Tree grow + Prediction update (full loop)
        // =========================================================================
        let mut gradients = vec![0.0f32; num_rows];
        let mut hessians = vec![0.0f32; num_rows];
        let mut predictions = vec![0.0f32; num_rows];

        let start = Instant::now();
        for _ in 0..num_rounds {
            for &idx in &row_indices {
                let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                gradients[idx] = g;
                hessians[idx] = h;
            }
            let tree = grower.grow_with_indices(&dataset, &gradients, &hessians, &row_indices);
            tree.predict_batch_add(&dataset, &mut predictions);
        }
        let scenario_c = start.elapsed().as_secs_f64() * 1000.0;

        // =========================================================================
        // SCENARIO D: Measure gradient computation alone
        // =========================================================================
        let start = Instant::now();
        for _ in 0..num_rounds {
            for &idx in &row_indices {
                let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                gradients[idx] = g;
                hessians[idx] = h;
            }
        }
        let grad_only = start.elapsed().as_secs_f64() * 1000.0;

        // =========================================================================
        // SCENARIO E: Measure prediction update alone
        // =========================================================================
        let tree = grower.grow_with_indices(&dataset, &fixed_grads, &fixed_hess, &row_indices);
        predictions.fill(0.0);

        let start = Instant::now();
        for _ in 0..num_rounds {
            tree.predict_batch_add(&dataset, &mut predictions);
        }
        let pred_only = start.elapsed().as_secs_f64() * 1000.0;

        // =========================================================================
        // ANALYSIS
        // =========================================================================
        println!("\nScenario timings (total for {} rounds):", num_rounds);
        println!("  A) Isolated trees (hot cache):   {:>8.2} ms", scenario_a);
        println!("  B) Grad + Tree (no pred):        {:>8.2} ms", scenario_b);
        println!("  C) Grad + Tree + Pred (full):    {:>8.2} ms", scenario_c);
        println!("  D) Gradient comp only:           {:>8.2} ms", grad_only);
        println!("  E) Prediction update only:       {:>8.2} ms", pred_only);

        println!("\nPer-round analysis:");
        let tree_only = scenario_a / num_rounds as f64;
        let tree_with_grad = scenario_b / num_rounds as f64;
        let full_loop = scenario_c / num_rounds as f64;
        let grad_per = grad_only / num_rounds as f64;
        let pred_per = pred_only / num_rounds as f64;

        println!("  Isolated tree grow:              {:>8.3} ms", tree_only);
        println!("  Tree with gradient prep:         {:>8.3} ms", tree_with_grad);
        println!("  Full loop:                       {:>8.3} ms", full_loop);

        println!("\n  Gradient computation time:       {:>8.3} ms", grad_per);
        println!("  Prediction update time:          {:>8.3} ms", pred_per);

        let tree_slowdown = tree_with_grad - tree_only - grad_per;
        println!("\n  Tree grow slowdown from grad:    {:>8.3} ms ({:.0}%)",
            tree_slowdown, (tree_slowdown / tree_only) * 100.0);

        let expected_full = tree_only + grad_per + pred_per;
        let actual_overhead = full_loop - expected_full;
        println!("  Expected full loop (sum):        {:>8.3} ms", expected_full);
        println!("  Actual full loop:                {:>8.3} ms", full_loop);
        println!("  Cache pollution overhead:        {:>8.3} ms ({:.0}%)",
            actual_overhead, (actual_overhead / tree_only) * 100.0);

        // =========================================================================
        // SCENARIO F: What if we batch gradients differently?
        // =========================================================================
        println!("\nCache-aware alternatives:");

        // Prefetch gradients in blocks to improve locality
        let mut gradients = vec![0.0f32; num_rows];
        let mut hessians = vec![0.0f32; num_rows];
        let mut predictions = vec![0.0f32; num_rows];

        const BLOCK_SIZE: usize = 8192;

        let start = Instant::now();
        for _ in 0..num_rounds {
            // Compute gradients in cache-friendly blocks
            for chunk in row_indices.chunks(BLOCK_SIZE) {
                for &idx in chunk {
                    let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                    gradients[idx] = g;
                    hessians[idx] = h;
                }
            }
            let tree = grower.grow_with_indices(&dataset, &gradients, &hessians, &row_indices);
            tree.predict_batch_add(&dataset, &mut predictions);
        }
        let blocked_grad = start.elapsed().as_secs_f64() * 1000.0;

        println!("  Blocked gradient compute:        {:>8.2} ms ({:.2} ms/round)",
            blocked_grad, blocked_grad / num_rounds as f64);
        println!("  vs naive full loop:              {:>8.2}% improvement",
            (1.0 - blocked_grad / scenario_c) * 100.0);

        // =========================================================================
        // SCENARIO G: Interleave gradient + tree or do them separately
        // =========================================================================

        // Memory working set analysis
        let grad_hess_size = num_rows * 4 * 2; // gradients + hessians in bytes
        let dataset_size = num_rows * num_features; // bins
        let total_working_set = grad_hess_size + dataset_size;

        println!("\nMemory analysis:");
        println!("  Gradients + Hessians:            {:>8.2} MB", grad_hess_size as f64 / 1_000_000.0);
        println!("  Dataset bins:                    {:>8.2} MB", dataset_size as f64 / 1_000_000.0);
        println!("  Total working set:               {:>8.2} MB", total_working_set as f64 / 1_000_000.0);
        println!("  Typical L3 cache:                {:>8} MB", 32); // Modern CPU

        if total_working_set as f64 / 1_000_000.0 > 32.0 {
            println!("  WARNING: Working set exceeds L3 cache!");
            println!("  This explains why gradients evict dataset from cache");
        }
    }

    // =========================================================================
    // POTENTIAL SOLUTIONS
    // =========================================================================
    println!("\n\nPOTENTIAL SOLUTIONS");
    println!("====================");
    println!("1. Cache-blocked gradient computation (already chunked in histogram builder)");
    println!("2. Prefetch dataset bins during gradient computation");
    println!("3. Interleave: compute gradients for block, build partial histogram, repeat");
    println!("4. Stream processing: fuse gradient + histogram in single pass");
    println!("5. Reduce working set: pack gradient/hessian to f16 during training");
}
