//! Benchmark histogram builder and tree growing scaling with number of features
//!
//! Run: cargo run --release --example histogram_scaling

use std::time::Instant;

use treeboost::booster::GBDTConfig;
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use treeboost::histogram::HistogramBuilder;
use treeboost::tree::TreeGrower;

fn create_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
    // Generate features (columnar layout)
    let mut features = Vec::with_capacity(num_rows * num_features);
    for f in 0..num_features {
        for r in 0..num_rows {
            features.push(((r * (f + 1) * 17) % 256) as u8);
        }
    }

    let targets: Vec<f32> = (0..num_rows).map(|i| i as f32).collect();
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

fn benchmark_histogram_build(num_rows: usize, num_features: usize, iterations: usize) -> f64 {
    let dataset = create_dataset(num_rows, num_features);
    let gradients: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.1).collect();
    let hessians: Vec<f32> = vec![1.0; num_rows];
    let row_indices: Vec<usize> = (0..num_rows).collect();

    let builder = HistogramBuilder::new();

    // Warmup
    for _ in 0..3 {
        let _ = builder.build(&dataset, &row_indices, &gradients, &hessians);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = builder.build(&dataset, &row_indices, &gradients, &hessians);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn benchmark_single_tree(num_rows: usize, num_features: usize, iterations: usize) -> f64 {
    let dataset = create_dataset(num_rows, num_features);
    // Create gradients that encourage splitting
    let gradients: Vec<f32> = (0..num_rows)
        .map(|i| if i < num_rows / 2 { -1.0 } else { 1.0 })
        .collect();
    let hessians: Vec<f32> = vec![1.0; num_rows];

    let grower = TreeGrower::new()
        .with_max_depth(6)
        .with_max_leaves(31)
        .with_learning_rate(0.1);

    // Warmup
    for _ in 0..2 {
        let _ = grower.grow(&dataset, &gradients, &hessians);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = grower.grow(&dataset, &gradients, &hessians);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn benchmark_full_training(num_rows: usize, num_features: usize, num_rounds: usize) -> f64 {
    let dataset = create_dataset(num_rows, num_features);

    let config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(6)
        .with_learning_rate(0.1);

    // Warmup
    let _ = treeboost::booster::GBDTModel::train(&dataset, config.clone());

    // Benchmark
    let start = Instant::now();
    let _ = treeboost::booster::GBDTModel::train(&dataset, config);
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0
}

fn main() {
    println!("TreeBoost Scaling Benchmark");
    println!("===========================\n");

    let num_rows = 100_000;

    // ==========================================================================
    // 1. Histogram Builder Scaling
    // ==========================================================================
    println!("1. HISTOGRAM BUILDER SCALING");
    println!("----------------------------");
    println!("Rows: {}\n", num_rows);

    let feature_counts = [5, 10, 20, 50, 100, 200, 300];
    let iterations = 10;

    println!("{:>10} {:>12} {:>12} {:>12}", "Features", "Time (ms)", "Per Feature", "Ratio");
    println!("{}", "-".repeat(50));

    let mut base_per_feature = 0.0;

    for &num_features in &feature_counts {
        let time_ms = benchmark_histogram_build(num_rows, num_features, iterations);
        let per_feature = time_ms / num_features as f64;

        let ratio = if base_per_feature > 0.0 {
            per_feature / base_per_feature
        } else {
            base_per_feature = per_feature;
            1.0
        };

        println!(
            "{:>10} {:>12.3} {:>12.4} {:>12.2}x",
            num_features, time_ms, per_feature, ratio
        );
    }

    // ==========================================================================
    // 2. Single Tree Growing Scaling
    // ==========================================================================
    println!("\n\n2. SINGLE TREE GROWING SCALING");
    println!("-------------------------------");
    println!("Rows: {}, max_depth=6, max_leaves=31\n", num_rows);

    println!("{:>10} {:>12} {:>12} {:>12}", "Features", "Time (ms)", "Per Feature", "Ratio");
    println!("{}", "-".repeat(50));

    let mut base_per_feature = 0.0;

    for &num_features in &feature_counts {
        let time_ms = benchmark_single_tree(num_rows, num_features, 5);
        let per_feature = time_ms / num_features as f64;

        let ratio = if base_per_feature > 0.0 {
            per_feature / base_per_feature
        } else {
            base_per_feature = per_feature;
            1.0
        };

        println!(
            "{:>10} {:>12.3} {:>12.4} {:>12.2}x",
            num_features, time_ms, per_feature, ratio
        );
    }

    // ==========================================================================
    // 3. Full Training Scaling (10 rounds)
    // ==========================================================================
    println!("\n\n3. FULL TRAINING SCALING (10 rounds)");
    println!("-------------------------------------");
    println!("Rows: {}, max_depth=6\n", num_rows);

    println!("{:>10} {:>12} {:>12} {:>12}", "Features", "Time (ms)", "Per Feature", "Ratio");
    println!("{}", "-".repeat(50));

    let mut base_per_feature = 0.0;

    for &num_features in &feature_counts {
        let time_ms = benchmark_full_training(num_rows, num_features, 10);
        let per_feature = time_ms / num_features as f64;

        let ratio = if base_per_feature > 0.0 {
            per_feature / base_per_feature
        } else {
            base_per_feature = per_feature;
            1.0
        };

        println!(
            "{:>10} {:>12.3} {:>12.4} {:>12.2}x",
            num_features, time_ms, per_feature, ratio
        );
    }

    // ==========================================================================
    // Summary
    // ==========================================================================
    println!("\n\nSCALING ANALYSIS:");
    println!("-----------------");
    println!("If 'Per Feature' time stays constant → O(n) scaling (good)");
    println!("If 'Per Feature' time grows with features → worse than O(n) (bad)");
    println!("'Ratio' shows how much worse per-feature time is vs baseline (5 features)");
}
