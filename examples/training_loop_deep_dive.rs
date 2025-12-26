//! Deep dive into the training loop overhead
//!
//! Run: cargo run --release --example training_loop_deep_dive
//!
//! This investigates the ~17ms unexplained overhead per round in GBDTModel::train
//! when compared to isolated tree grows.

use std::time::Instant;

use treeboost::booster::{GBDTConfig, GBDTModel, LossType};
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
    println!("Training Loop Deep Dive");
    println!("========================\n");

    let num_rows = 500_000;
    let num_features = 100;
    let num_rounds = 10;

    println!("Dataset: {} rows × {} features, {} rounds\n", num_rows, num_features, num_rounds);

    let dataset = create_dataset(num_rows, num_features);
    let targets = dataset.targets().to_vec();

    // =========================================================================
    // BASELINE: Isolated tree grows (what we know is fast)
    // =========================================================================
    println!("1. BASELINE: ISOLATED TREE GROWS");
    println!("---------------------------------");

    let gradients: Vec<f32> = (0..num_rows)
        .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
        .collect();
    let hessians = vec![1.0f32; num_rows];
    let row_indices: Vec<usize> = (0..num_rows).collect();

    let grower = TreeGrower::new()
        .with_max_depth(6)
        .with_max_leaves(31)
        .with_learning_rate(0.1);

    // Warmup
    for _ in 0..2 {
        let _ = grower.grow_with_indices(&dataset, &gradients, &hessians, &row_indices);
    }

    let start = Instant::now();
    for _ in 0..num_rounds {
        let _ = grower.grow_with_indices(&dataset, &gradients, &hessians, &row_indices);
    }
    let isolated_time = start.elapsed().as_secs_f64() * 1000.0;
    println!("Isolated {} trees:         {:>8.2} ms ({:.2} ms/tree)",
        num_rounds, isolated_time, isolated_time / num_rounds as f64);

    // =========================================================================
    // BASELINE: GBDTModel::train (where the overhead is)
    // =========================================================================
    println!("\n2. GBDTModel::train TIMING");
    println!("--------------------------");

    let config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(6)
        .with_learning_rate(0.1);  // Early stopping and conformal disabled by default

    // Warmup
    let _ = GBDTModel::train_binned(&dataset, config.clone());

    let start = Instant::now();
    let _ = GBDTModel::train_binned(&dataset, config.clone());
    let train_time = start.elapsed().as_secs_f64() * 1000.0;
    println!("GBDTModel::train:          {:>8.2} ms ({:.2} ms/round)",
        train_time, train_time / num_rounds as f64);

    let overhead = train_time - isolated_time;
    println!("Total overhead:            {:>8.2} ms ({:.2} ms/round)",
        overhead, overhead / num_rounds as f64);

    // =========================================================================
    // MANUAL LOOP: Replicate GBDTModel::train exactly
    // =========================================================================
    println!("\n3. MANUAL LOOP REPLICATING GBDTModel::train");
    println!("---------------------------------------------");

    let loss_fn = LossType::Mse.create();

    // Initial setup (one-time)
    let setup_start = Instant::now();
    let base_prediction = 0.0f32;
    let mut predictions = vec![base_prediction; num_rows];
    let mut gradients = vec![0.0f32; num_rows];
    let mut hessians = vec![0.0f32; num_rows];
    let train_indices: Vec<usize> = (0..num_rows).collect();
    let mut sample_indices: Vec<usize> = Vec::with_capacity(train_indices.len());
    let mut trees: Vec<treeboost::tree::Tree> = Vec::with_capacity(num_rounds);
    let setup_time = setup_start.elapsed().as_secs_f64() * 1000.0;
    println!("Setup allocations:         {:>8.3} ms", setup_time);

    // Training loop with detailed timing
    let mut time_gradient = 0.0;
    let mut time_sample = 0.0;
    let mut time_grow = 0.0;
    let mut time_predict = 0.0;
    let mut time_push = 0.0;

    let loop_start = Instant::now();

    for _round in 0..num_rounds {
        // 1. Gradient computation
        let t = Instant::now();
        for &idx in &train_indices {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            gradients[idx] = g;
            hessians[idx] = h;
        }
        time_gradient += t.elapsed().as_secs_f64() * 1000.0;

        // 2. Sample indices (no subsampling = copy)
        let t = Instant::now();
        sample_indices.clear();
        sample_indices.extend_from_slice(&train_indices);
        time_sample += t.elapsed().as_secs_f64() * 1000.0;

        // 3. Tree grow
        let t = Instant::now();
        let tree = grower.grow_with_indices(&dataset, &gradients, &hessians, &sample_indices);
        time_grow += t.elapsed().as_secs_f64() * 1000.0;

        // 4. Prediction update
        let t = Instant::now();
        tree.predict_batch_add(&dataset, &mut predictions);
        time_predict += t.elapsed().as_secs_f64() * 1000.0;

        // 5. Push tree
        let t = Instant::now();
        trees.push(tree);
        time_push += t.elapsed().as_secs_f64() * 1000.0;
    }

    let loop_time = loop_start.elapsed().as_secs_f64() * 1000.0;

    println!("\nPer-round breakdown:");
    println!("  Gradient comp:           {:>8.3} ms", time_gradient / num_rounds as f64);
    println!("  Sample indices:          {:>8.3} ms", time_sample / num_rounds as f64);
    println!("  Tree grow:               {:>8.3} ms", time_grow / num_rounds as f64);
    println!("  Predict update:          {:>8.3} ms", time_predict / num_rounds as f64);
    println!("  Tree push:               {:>8.3} ms", time_push / num_rounds as f64);

    let sum_components = time_gradient + time_sample + time_grow + time_predict + time_push;
    println!("\nTotal breakdown:");
    println!("  Sum of components:       {:>8.2} ms", sum_components);
    println!("  Actual loop time:        {:>8.2} ms", loop_time);
    println!("  Timing overhead:         {:>8.2} ms", loop_time - sum_components);

    // =========================================================================
    // KEY COMPARISON
    // =========================================================================
    println!("\n4. KEY COMPARISON");
    println!("------------------");
    println!("Manual loop total:         {:>8.2} ms", loop_time + setup_time);
    println!("GBDTModel::train:          {:>8.2} ms", train_time);
    println!("Difference:                {:>8.2} ms", train_time - (loop_time + setup_time));

    let manual_overhead = (loop_time + setup_time) - isolated_time;
    println!("\nManual overhead (vs isolated): {:>8.2} ms", manual_overhead);
    println!("This overhead is from:");
    println!("  Gradient comp:           {:>8.3} ms", time_gradient);
    println!("  Sample indices:          {:>8.3} ms", time_sample);
    println!("  Predict update:          {:>8.3} ms", time_predict);
    println!("  Tree push:               {:>8.3} ms", time_push);
    println!("  TOTAL known overhead:    {:>8.3} ms", time_gradient + time_sample + time_predict + time_push);

    let unknown = manual_overhead - (time_gradient + time_sample + time_predict + time_push);
    println!("  UNKNOWN overhead:        {:>8.3} ms", unknown);

    // =========================================================================
    // 5. CACHE POLLUTION: Tree grow time with changing gradients
    // =========================================================================
    println!("\n5. CACHE POLLUTION INVESTIGATION");
    println!("---------------------------------");

    // Reset
    let mut gradients = vec![0.0f32; num_rows];
    let mut hessians = vec![0.0f32; num_rows];
    let mut predictions = vec![0.0f32; num_rows];

    // A) Tree grow with SAME gradients (cache warm)
    let fixed_grads: Vec<f32> = (0..num_rows)
        .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
        .collect();
    let fixed_hess = vec![1.0f32; num_rows];

    let start = Instant::now();
    for _ in 0..num_rounds {
        let _ = grower.grow_with_indices(&dataset, &fixed_grads, &fixed_hess, &train_indices);
    }
    let warm_cache_time = start.elapsed().as_secs_f64() * 1000.0;

    // B) Tree grow with CHANGING gradients (cache cold each iteration)
    let start = Instant::now();
    for round in 0..num_rounds {
        // Compute new gradients (simulates training)
        for &idx in &train_indices {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            gradients[idx] = g;
            hessians[idx] = h;
        }

        let tree = grower.grow_with_indices(&dataset, &gradients, &hessians, &train_indices);
        tree.predict_batch_add(&dataset, &mut predictions);
    }
    let cold_cache_time = start.elapsed().as_secs_f64() * 1000.0;

    // C) Tree grow with changing gradients but NO prediction update
    let mut gradients = vec![0.0f32; num_rows];
    let mut hessians = vec![0.0f32; num_rows];

    let start = Instant::now();
    for round in 0..num_rounds {
        // Compute new gradients but no prediction update
        for idx in 0..num_rows {
            gradients[idx] = (round as f32 + idx as f32).sin();
            hessians[idx] = 1.0;
        }
        let _ = grower.grow_with_indices(&dataset, &gradients, &hessians, &train_indices);
    }
    let cold_no_pred_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Warm cache (same grads):   {:>8.2} ms ({:.2} ms/tree)", warm_cache_time, warm_cache_time / num_rounds as f64);
    println!("Cold (new grads, +pred):   {:>8.2} ms ({:.2} ms/tree)", cold_cache_time, cold_cache_time / num_rounds as f64);
    println!("Cold (new grads, no pred): {:>8.2} ms ({:.2} ms/tree)", cold_no_pred_time, cold_no_pred_time / num_rounds as f64);
    println!("\nCache pollution from changing gradients: {:>8.2} ms ({:.2} ms/round)",
        cold_no_pred_time - warm_cache_time, (cold_no_pred_time - warm_cache_time) / num_rounds as f64);

    // =========================================================================
    // 6. MEMORY TRAFFIC ANALYSIS
    // =========================================================================
    println!("\n6. MEMORY TRAFFIC ANALYSIS");
    println!("---------------------------");

    // Data touched per round:
    // - Targets: num_rows × 4 bytes (read)
    // - Predictions: num_rows × 4 bytes (read)
    // - Gradients: num_rows × 4 bytes (write)
    // - Hessians: num_rows × 4 bytes (write)
    // - train_indices: num_rows × 8 bytes (read)
    // - sample_indices: num_rows × 8 bytes (write then read)
    // - Dataset bins: num_rows × num_features bytes (read)
    // - Histogram storage: num_features × 256 × 12 bytes × ~62 nodes

    let gradient_traffic = num_rows * 4 * 2; // read + write
    let sample_traffic = num_rows * 8 * 2;    // write + read
    let predict_traffic = num_rows * 4 * 2;   // read predictions + write

    println!("Per-round memory traffic:");
    println!("  Gradient comp:           {:>8.1} MB", (num_rows * 4 * 4) as f64 / 1_000_000.0);
    println!("  Sample indices:          {:>8.1} MB", sample_traffic as f64 / 1_000_000.0);
    println!("  Predict update:          {:>8.1} MB", predict_traffic as f64 / 1_000_000.0);
    println!("  Dataset bins (tree):     {:>8.1} MB", (num_rows * num_features) as f64 / 1_000_000.0);

    let total_extra_traffic = gradient_traffic + sample_traffic + predict_traffic;
    println!("  Extra traffic/round:     {:>8.1} MB", total_extra_traffic as f64 / 1_000_000.0);

    // Assuming 50 GB/s memory bandwidth
    let bandwidth = 50_000.0; // MB/s
    let estimated_time = (total_extra_traffic as f64 / 1_000_000.0) / bandwidth * 1000.0;
    println!("  Estimated time @ 50GB/s: {:>8.3} ms", estimated_time);

    // =========================================================================
    // 7. WHAT IF WE DON'T COPY SAMPLE INDICES?
    // =========================================================================
    println!("\n7. SAMPLE INDICES COPY OVERHEAD");
    println!("--------------------------------");

    // Instead of copying, just use train_indices directly
    let mut predictions = vec![0.0f32; num_rows];
    let mut gradients = vec![0.0f32; num_rows];
    let mut hessians = vec![0.0f32; num_rows];

    let start = Instant::now();
    for _round in 0..num_rounds {
        for &idx in &train_indices {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            gradients[idx] = g;
            hessians[idx] = h;
        }
        // Use train_indices directly (no copy)
        let tree = grower.grow_with_indices(&dataset, &gradients, &hessians, &train_indices);
        tree.predict_batch_add(&dataset, &mut predictions);
    }
    let no_copy_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("With sample_indices copy:  {:>8.2} ms", loop_time);
    println!("Without copy (direct use): {:>8.2} ms", no_copy_time);
    println!("Savings:                   {:>8.2} ms", loop_time - no_copy_time);

    // =========================================================================
    // 8. GRADIENT COMPUTATION PARALLELISM
    // =========================================================================
    println!("\n8. GRADIENT COMPUTATION PARALLELISM");
    println!("------------------------------------");

    use rayon::prelude::*;

    let mut predictions = vec![0.0f32; num_rows];
    let mut gradients = vec![0.0f32; num_rows];
    let mut hessians = vec![0.0f32; num_rows];

    // Sequential
    let start = Instant::now();
    for _ in 0..num_rounds {
        for &idx in &train_indices {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            gradients[idx] = g;
            hessians[idx] = h;
        }
    }
    let seq_grad_time = start.elapsed().as_secs_f64() * 1000.0;

    // Parallel with chunks
    let start = Instant::now();
    for _ in 0..num_rounds {
        train_indices.par_chunks(8192).for_each(|chunk| {
            for &idx in chunk {
                let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                unsafe {
                    let grad_ptr = gradients.as_ptr() as *mut f32;
                    let hess_ptr = hessians.as_ptr() as *mut f32;
                    *grad_ptr.add(idx) = g;
                    *hess_ptr.add(idx) = h;
                }
            }
        });
    }
    let par_grad_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("Sequential gradient:       {:>8.3} ms ({:.3} ms/round)", seq_grad_time, seq_grad_time / num_rounds as f64);
    println!("Parallel gradient:         {:>8.3} ms ({:.3} ms/round)", par_grad_time, par_grad_time / num_rounds as f64);

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n9. SUMMARY: WHERE IS THE 17ms OVERHEAD?");
    println!("----------------------------------------");

    let per_round_overhead = overhead / num_rounds as f64;
    println!("Total overhead per round:  {:>8.2} ms", per_round_overhead);
    println!("\nBreakdown:");
    println!("  Gradient comp:           {:>8.3} ms ({:.0}%)",
        time_gradient / num_rounds as f64,
        (time_gradient / num_rounds as f64 / per_round_overhead) * 100.0);
    println!("  Sample indices:          {:>8.3} ms ({:.0}%)",
        time_sample / num_rounds as f64,
        (time_sample / num_rounds as f64 / per_round_overhead) * 100.0);
    println!("  Predict update:          {:>8.3} ms ({:.0}%)",
        time_predict / num_rounds as f64,
        (time_predict / num_rounds as f64 / per_round_overhead) * 100.0);

    let cache_pollution = (cold_no_pred_time - warm_cache_time) / num_rounds as f64;
    println!("  Cache pollution:         {:>8.3} ms ({:.0}%)",
        cache_pollution,
        (cache_pollution / per_round_overhead) * 100.0);

    let accounted = time_gradient / num_rounds as f64
        + time_sample / num_rounds as f64
        + time_predict / num_rounds as f64
        + cache_pollution;
    println!("  ---");
    println!("  ACCOUNTED:               {:>8.3} ms ({:.0}%)", accounted, (accounted / per_round_overhead) * 100.0);
    println!("  UNACCOUNTED:             {:>8.3} ms ({:.0}%)",
        per_round_overhead - accounted,
        ((per_round_overhead - accounted) / per_round_overhead) * 100.0);
}
