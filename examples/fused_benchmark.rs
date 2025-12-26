//! Benchmark comparing fused vs separate gradient+histogram paths
//!
//! Run: cargo run --release --example fused_benchmark

use std::time::Instant;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

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
    println!("Fused vs Separate Gradient+Histogram Benchmark");
    println!("================================================\n");

    for &(num_rows, num_features, label) in &[
        (100_000, 50, "Medium (100k×50)"),
        (500_000, 100, "Large (500k×100)"),
    ] {
        println!("{}", label);
        println!("{}", "-".repeat(label.len()));

        let dataset = create_dataset(num_rows, num_features);
        let num_rounds = 10;

        // Test 1: Fused path (no subsampling)
        let config_fused = GBDTConfig::new()
            .with_num_rounds(num_rounds)
            .with_max_depth(6)
            .with_learning_rate(0.1);

        // Warmup
        let _ = GBDTModel::train_binned(&dataset, config_fused.clone());

        let start = Instant::now();
        let _ = GBDTModel::train_binned(&dataset, config_fused.clone());
        let fused_time = start.elapsed().as_secs_f64() * 1000.0;

        // Test 2: Non-fused path using GOSS to force separate gradient computation
        // GOSS needs all gradients first to sample, so it uses separate path
        // Using high top_rate (0.8) + other_rate (0.19) = 99% data but forces separate gradient computation
        let config_separate = GBDTConfig::new()
            .with_num_rounds(num_rounds)
            .with_max_depth(6)
            .with_learning_rate(0.1)
            .with_goss_rates(0.8, 0.19); // Forces separate path while using ~99% of data

        // Warmup
        let _ = GBDTModel::train_binned(&dataset, config_separate.clone());

        let start = Instant::now();
        let _ = GBDTModel::train_binned(&dataset, config_separate.clone());
        let separate_time = start.elapsed().as_secs_f64() * 1000.0;

        let improvement = (1.0 - fused_time / separate_time) * 100.0;

        println!("  Fused path:      {:>8.1} ms ({:.1} ms/round)", fused_time, fused_time / num_rounds as f64);
        println!("  Separate path:   {:>8.1} ms ({:.1} ms/round)", separate_time, separate_time / num_rounds as f64);
        println!("  Improvement:     {:>8.1}%", improvement);
        println!();
    }

    // LightGBM reference
    println!("LightGBM Reference (from benchmarks):");
    println!("  Medium (100k×50, 100r):   574 ms (5.74 ms/round)");
    println!("  Large (500k×100, 100r):   2722 ms (27.22 ms/round)");
}
