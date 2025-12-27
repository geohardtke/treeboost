//! Benchmark comparing CPU vs GPU tree building strategies
//!
//! Compares:
//! 1. CPU partition + GPU histograms (best-first, histogram subtraction)
//! 2. GPU partition + GPU histograms (level-wise, current implementation)
//! 3. Full GPU pipeline (all operations on GPU, minimal PCIe transfers)
//!
//! Usage:
//!   cargo run --release --features gpu --example full_gpu_benchmark

use std::time::{Duration, Instant};

#[cfg(feature = "gpu")]
use std::sync::Arc;
#[cfg(feature = "gpu")]
use treeboost::backend::wgpu::{FullGpuTreeBuilder, GpuDevice};
use treeboost::backend::BackendType;
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use treeboost::tree::TreeGrower;

fn main() {
    println!("=== Tree Building Strategy Comparison ===\n");

    #[cfg(not(feature = "gpu"))]
    {
        println!("This benchmark requires the 'gpu' feature.");
        println!("Run with: cargo run --release --features gpu --example full_gpu_benchmark");
        return;
    }

    #[cfg(feature = "gpu")]
    run_benchmark();
}

#[cfg(feature = "gpu")]
fn run_benchmark() {
    let device = match GpuDevice::new() {
        Some(d) => Arc::new(d),
        None => {
            println!("No GPU available. Exiting.");
            return;
        }
    };

    println!("GPU: {} ({:?})\n", device.name(), device.backend());

    let warmup_trees = 2;
    let bench_trees = 10;
    let max_depth = 6;

    // Test configurations
    let configs = [
        (50_000, 20),  // 50K rows, 20 features
        (100_000, 20), // 100K rows
        (200_000, 20), // 200K rows
        (500_000, 20), // 500K rows
    ];

    println!(
        "{:>10} | {:>8} | {:>14} | {:>14} | {:>14}",
        "Rows", "Features", "CPU+GPUHist", "GPU Levelwise", "GPU BestFirst"
    );
    println!("{}", "-".repeat(78));

    for &(num_rows, num_features) in &configs {
        let (dataset, gradients, hessians) = generate_test_data(num_rows, num_features);

        // Approach 1: CPU partition + GPU histograms (best-first)
        let cpu_gpu_time = benchmark_cpu_partition_gpu_hist(
            &dataset,
            &gradients,
            &hessians,
            max_depth,
            warmup_trees,
            bench_trees,
        );

        // Approach 2: Full GPU level-wise (no subtraction trick)
        let full_gpu_levelwise_time = benchmark_full_gpu_levelwise(
            Arc::clone(&device),
            &dataset,
            &gradients,
            &hessians,
            max_depth,
            warmup_trees,
            bench_trees,
        );

        // Approach 3: Full GPU best-first (with subtraction trick)
        let full_gpu_bestfirst_time = benchmark_full_gpu_bestfirst(
            Arc::clone(&device),
            &dataset,
            &gradients,
            &hessians,
            max_depth,
            warmup_trees,
            bench_trees,
        );

        let cpu_gpu_ms = cpu_gpu_time.as_secs_f64() * 1000.0;
        let full_gpu_levelwise_ms = full_gpu_levelwise_time.as_secs_f64() * 1000.0;
        let full_gpu_bestfirst_ms = full_gpu_bestfirst_time.as_secs_f64() * 1000.0;

        println!(
            "{:>10} | {:>8} | {:>11.2}ms | {:>11.2}ms | {:>11.2}ms",
            num_rows, num_features, cpu_gpu_ms, full_gpu_levelwise_ms, full_gpu_bestfirst_ms,
        );
    }

    println!();
    println!("Strategy explanations:");
    println!("  CPU+GPUHist:     CPU best-first partition + GPU histogram building");
    println!("                   (histogram subtraction trick, priority queue)");
    println!("  GPU Levelwise:   Full GPU level-wise, no histogram subtraction");
    println!("                   (builds all histograms at each level)");
    println!("  GPU BestFirst:   GPU histogram building + CPU priority queue");
    println!("                   (histogram subtraction trick, best of both)");
}

fn generate_test_data(num_rows: usize, num_features: usize) -> (BinnedDataset, Vec<f32>, Vec<f32>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Generate bins: column-major layout for BinnedDataset
    let mut features = Vec::with_capacity(num_rows * num_features);
    for f in 0..num_features {
        for r in 0..num_rows {
            let mut hasher = DefaultHasher::new();
            (r, f).hash(&mut hasher);
            let bin = (hasher.finish() % 256) as u8;
            features.push(bin);
        }
    }

    // Generate targets
    let targets: Vec<f32> = (0..num_rows)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            (hasher.finish() as f32 / u64::MAX as f32) * 10.0 - 5.0
        })
        .collect();

    let feature_info: Vec<FeatureInfo> = (0..num_features)
        .map(|i| FeatureInfo {
            name: format!("f{}", i),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
        })
        .collect();

    let dataset = BinnedDataset::new(num_rows, features, targets.clone(), feature_info);

    // Generate gradients/hessians that encourage splitting
    let gradients: Vec<f32> = (0..num_rows)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (i, "grad").hash(&mut hasher);
            (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect();

    let hessians: Vec<f32> = vec![1.0; num_rows];

    (dataset, gradients, hessians)
}

fn benchmark_cpu_partition_gpu_hist(
    dataset: &BinnedDataset,
    gradients: &[f32],
    hessians: &[f32],
    max_depth: usize,
    warmup_trees: usize,
    bench_trees: usize,
) -> Duration {
    let grower = TreeGrower::new()
        .with_max_depth(max_depth)
        .with_max_leaves(1 << max_depth)
        .with_min_samples_leaf(10)
        .with_backend(BackendType::Wgpu)
        .with_gpu_batch_size(32);

    // Warmup
    for _ in 0..warmup_trees {
        let _ = grower.grow(dataset, gradients, hessians);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..bench_trees {
        let _ = grower.grow(dataset, gradients, hessians);
    }
    start.elapsed() / bench_trees as u32
}

#[cfg(feature = "gpu")]
fn benchmark_full_gpu_levelwise(
    device: Arc<GpuDevice>,
    dataset: &BinnedDataset,
    gradients: &[f32],
    hessians: &[f32],
    max_depth: usize,
    warmup_trees: usize,
    bench_trees: usize,
) -> Duration {
    let mut builder = FullGpuTreeBuilder::new(device);

    let row_indices: Vec<usize> = (0..dataset.num_rows()).collect();
    let max_leaves = 1 << max_depth;
    let lambda = 1.0f32;
    let min_samples_leaf = 10usize;
    let min_hessian_leaf = 1.0f32;
    let min_gain = 0.0f32;
    let learning_rate = 0.1f32;

    // Warmup
    for _ in 0..warmup_trees {
        let _ = builder.build_tree(
            dataset,
            gradients,
            hessians,
            &row_indices,
            max_depth,
            max_leaves,
            lambda,
            min_samples_leaf,
            min_hessian_leaf,
            min_gain,
            learning_rate,
        );
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..bench_trees {
        let _ = builder.build_tree(
            dataset,
            gradients,
            hessians,
            &row_indices,
            max_depth,
            max_leaves,
            lambda,
            min_samples_leaf,
            min_hessian_leaf,
            min_gain,
            learning_rate,
        );
    }
    start.elapsed() / bench_trees as u32
}

#[cfg(feature = "gpu")]
fn benchmark_full_gpu_bestfirst(
    device: Arc<GpuDevice>,
    dataset: &BinnedDataset,
    gradients: &[f32],
    hessians: &[f32],
    max_depth: usize,
    warmup_trees: usize,
    bench_trees: usize,
) -> Duration {
    let mut builder = FullGpuTreeBuilder::new(device);

    let row_indices: Vec<usize> = (0..dataset.num_rows()).collect();
    let max_leaves = 1 << max_depth;
    let lambda = 1.0f32;
    let min_samples_leaf = 10usize;
    let min_hessian_leaf = 1.0f32;
    let min_gain = 0.0f32;
    let learning_rate = 0.1f32;

    // Warmup
    for _ in 0..warmup_trees {
        let _ = builder.build_tree_best_first(
            dataset,
            gradients,
            hessians,
            &row_indices,
            max_depth,
            max_leaves,
            lambda,
            min_samples_leaf,
            min_hessian_leaf,
            min_gain,
            learning_rate,
        );
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..bench_trees {
        let _ = builder.build_tree_best_first(
            dataset,
            gradients,
            hessians,
            &row_indices,
            max_depth,
            max_leaves,
            lambda,
            min_samples_leaf,
            min_hessian_leaf,
            min_gain,
            learning_rate,
        );
    }
    start.elapsed() / bench_trees as u32
}
