//! Benchmark full histogram building with SIMD optimization.
//!
//! Run with: cargo run --release --example histogram_benchmark

use std::time::Instant;
use treeboost::{BinnedDataset, FeatureInfo, FeatureType, HistogramBuilder};

fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
    // Generate deterministic test data
    let mut features = Vec::with_capacity(num_rows * num_features);
    for f in 0..num_features {
        for r in 0..num_rows {
            features.push(((r + f * 7) % 256) as u8);
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

fn benchmark_histogram_build(
    dataset: &BinnedDataset,
    row_indices: &[usize],
    gradients: &[f32],
    hessians: &[f32],
    iterations: usize,
) -> std::time::Duration {
    let builder = HistogramBuilder::new();

    let start = Instant::now();
    for _ in 0..iterations {
        let hists = builder.build(dataset, row_indices, gradients, hessians);
        std::hint::black_box(&hists);
    }
    start.elapsed()
}

fn main() {
    println!("Full Histogram Build Benchmark");
    println!("===============================");

    #[cfg(target_arch = "x86_64")]
    {
        println!("Architecture: x86_64");
        println!("AVX2 available: {}", is_x86_feature_detected!("avx2"));
        println!();
    }

    // Test configurations
    let configs = [
        (10_000, 20, 100),   // Small: 10K rows, 20 features, 100 iterations
        (50_000, 50, 50),    // Medium: 50K rows, 50 features, 50 iterations
        (100_000, 100, 20),  // Large: 100K rows, 100 features, 20 iterations
    ];

    for &(num_rows, num_features, iterations) in &configs {
        println!(
            "\nDataset: {} rows x {} features ({} iterations)",
            num_rows, num_features, iterations
        );
        println!("{}", "-".repeat(60));

        // Create dataset
        let dataset = create_test_dataset(num_rows, num_features);

        // Create gradients/hessians
        let gradients: Vec<f32> = (0..num_rows).map(|i| (i as f32 * 0.1) % 10.0).collect();
        let hessians: Vec<f32> = vec![1.0; num_rows];

        // Contiguous row indices (root node case)
        let row_indices: Vec<usize> = (0..num_rows).collect();

        // Run benchmark
        let duration = benchmark_histogram_build(&dataset, &row_indices, &gradients, &hessians, iterations);
        let per_iter_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;
        let throughput = (num_rows * num_features) as f64 / per_iter_ms / 1000.0; // Million cells/ms

        println!("  Time per build:  {:>8.3} ms", per_iter_ms);
        println!("  Throughput:      {:>8.2} M cells/ms", throughput);
        println!("  Total cells:     {:>8} K", num_rows * num_features / 1000);
    }

    // Compare with indexed (non-contiguous) access
    println!("\n\nIndexed vs Contiguous Access (50K rows, 50 features)");
    println!("====================================================");

    let num_rows = 50_000;
    let num_features = 50;
    let iterations = 50;

    let dataset = create_test_dataset(num_rows, num_features);
    let gradients: Vec<f32> = (0..num_rows).map(|i| (i as f32 * 0.1) % 10.0).collect();
    let hessians: Vec<f32> = vec![1.0; num_rows];

    // Contiguous (root node)
    let contiguous_indices: Vec<usize> = (0..num_rows).collect();
    let contiguous_time = benchmark_histogram_build(&dataset, &contiguous_indices, &gradients, &hessians, iterations);
    let contiguous_ms = contiguous_time.as_secs_f64() * 1000.0 / iterations as f64;

    // Indexed (child node - first half)
    let indexed_indices: Vec<usize> = (0..num_rows / 2).collect();
    let indexed_time = benchmark_histogram_build(&dataset, &indexed_indices, &gradients, &hessians, iterations);
    let indexed_ms = indexed_time.as_secs_f64() * 1000.0 / iterations as f64;

    println!("  Contiguous ({} rows): {:>8.3} ms", num_rows, contiguous_ms);
    println!("  Indexed ({} rows):    {:>8.3} ms", num_rows / 2, indexed_ms);
    println!(
        "  Ratio (indexed/contiguous, adjusted for row count): {:.2}x",
        indexed_ms / contiguous_ms * 2.0
    );

    println!("\n\nNote: SIMD optimization applies to the grad/hess interleaving phase");
    println!("in the contiguous path. The scatter (bin accumulation) is always");
    println!("scalar due to random access patterns.");
}
