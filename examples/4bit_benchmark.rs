//! Benchmark comparing 4-bit vs 8-bit bin packing on GPU.
//!
//! Run with: cargo run --release --features gpu --example 4bit_benchmark

use std::time::{Duration, Instant};

use treeboost::backend::wgpu::WgpuBackend;
use treeboost::backend::BinStorage;
use treeboost::backend::HistogramBackend;
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

fn create_dataset(num_rows: usize, num_features: usize, max_bins: u8) -> BinnedDataset {
    let mut features = vec![0u8; num_rows * num_features];
    for f in 0..num_features {
        for r in 0..num_rows {
            features[f * num_rows + r] = (r % max_bins as usize) as u8;
        }
    }

    let targets: Vec<f32> = (0..num_rows).map(|i| i as f32).collect();
    let feature_info: Vec<FeatureInfo> = (0..num_features)
        .map(|i| FeatureInfo {
            name: format!("feature_{}", i),
            feature_type: FeatureType::Numeric,
            num_bins: max_bins,
            bin_boundaries: vec![],
        })
        .collect();

    BinnedDataset::new(num_rows, features, targets, feature_info)
}

fn benchmark<F>(name: &str, iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..3 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations as u32;
    println!("  {}: {:?} ({} iters)", name, per_iter, iterations);
    per_iter
}

fn main() {
    println!("4-Bit vs 8-Bit Bin Packing GPU Benchmark");
    println!("=========================================\n");

    let backend = match WgpuBackend::new() {
        Some(b) => {
            println!("GPU: {}", b.device_name());
            println!("Backend: {:?}", b.backend_type());
            println!("Subgroups: {}", if b.subgroups_available() { "available" } else { "not available" });
            println!();
            b
        }
        None => {
            eprintln!("No GPU available! This benchmark requires a GPU.");
            return;
        }
    };

    let row_counts = [10_000, 50_000, 100_000, 250_000, 500_000];
    let num_features = 20;
    let iterations = 50;

    println!("Configuration:");
    println!("  Features: {}", num_features);
    println!("  Iterations: {}", iterations);
    println!();

    println!("| {:>8} | {:>12} | {:>12} | {:>8} | {:>12} |",
             "Rows", "8-bit (ms)", "4-bit (ms)", "Speedup", "BW Saved");
    println!("|{:-<10}|{:-<14}|{:-<14}|{:-<10}|{:-<14}|",
             "", "", "", "", "");

    for &num_rows in &row_counts {
        // Create 8-bit dataset (256 bins)
        let dataset_8bit = create_dataset(num_rows, num_features, 255);
        assert!(!dataset_8bit.supports_4bit());

        // Create 4-bit dataset (16 bins)
        let dataset_4bit = create_dataset(num_rows, num_features, 16);
        assert!(dataset_4bit.supports_4bit());

        // Generate gradients/hessians
        let grad_hess: Vec<(f32, f32)> = (0..num_rows)
            .map(|i| ((i as f32 * 0.01).sin(), 1.0))
            .collect();

        let row_indices: Vec<usize> = (0..num_rows).collect();

        // Benchmark 8-bit
        let mut time_8bit = Duration::ZERO;
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = backend.build_histograms(&dataset_8bit, &grad_hess, &row_indices);
            time_8bit += start.elapsed();
        }
        let avg_8bit = time_8bit / iterations as u32;

        // Benchmark 4-bit
        let mut time_4bit = Duration::ZERO;
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = backend.build_histograms(&dataset_4bit, &grad_hess, &row_indices);
            time_4bit += start.elapsed();
        }
        let avg_4bit = time_4bit / iterations as u32;

        let speedup = avg_8bit.as_secs_f64() / avg_4bit.as_secs_f64();

        // Calculate memory bandwidth saved
        let size_8bit = num_rows * num_features;
        let size_4bit = num_rows * ((num_features + 1) / 2);
        let bw_saved = 100.0 * (1.0 - size_4bit as f64 / size_8bit as f64);

        println!("| {:>8} | {:>12.3} | {:>12.3} | {:>7.2}x | {:>11.1}% |",
                 num_rows,
                 avg_8bit.as_secs_f64() * 1000.0,
                 avg_4bit.as_secs_f64() * 1000.0,
                 speedup,
                 bw_saved);
    }

    println!("\nNote: 4-bit path is only used when ALL features have <=16 bins.");
    println!("Memory bandwidth for bins is reduced by ~50% with 4-bit packing.");
    println!("Actual speedup depends on GPU memory bandwidth and compute ratio.");
}
