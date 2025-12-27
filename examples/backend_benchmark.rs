//! Comprehensive benchmark comparing all backends: CPU, WGPU, and CUDA.
//!
//! Compares:
//! - Full CPU (scalar backend, best-first with histogram subtraction)
//! - WGPU Hybrid (GPU histogram + CPU partition, best-first)
//! - WGPU Full (GPU histogram + GPU partition, level-wise)
//! - CUDA Hybrid (GPU histogram + CPU partition, best-first)
//! - CUDA Full (GPU histogram + GPU partition, level-wise)
//!
//! Run with:
//!   cargo run --release --example backend_benchmark
//!   cargo run --release --features gpu --example backend_benchmark
//!   cargo run --release --features cuda --example backend_benchmark
//!   cargo run --release --features "gpu cuda" --example backend_benchmark

use std::time::{Duration, Instant};

use treeboost::backend::BackendType;
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use treeboost::tree::TreeGrower;

#[cfg(feature = "gpu")]
use std::sync::Arc;
#[cfg(feature = "gpu")]
use treeboost::backend::wgpu::{FullGpuTreeBuilder as WgpuFullBuilder, GpuDevice as WgpuDevice};

#[cfg(feature = "cuda")]
use std::sync::Arc as CudaArc;
#[cfg(feature = "cuda")]
use treeboost::backend::cuda::{CudaDevice, FullCudaTreeBuilder};

/// Benchmark result for a single configuration
#[derive(Clone)]
#[allow(dead_code)]
struct BenchResult {
    name: String,
    rows: usize,
    features: usize,
    time_ms: f64,
    trees_per_sec: f64,
}

impl BenchResult {
    fn new(name: &str, rows: usize, features: usize, duration: Duration, num_trees: usize) -> Self {
        let time_ms = duration.as_secs_f64() * 1000.0 / num_trees as f64;
        let trees_per_sec = num_trees as f64 / duration.as_secs_f64();
        Self {
            name: name.to_string(),
            rows,
            features,
            time_ms,
            trees_per_sec,
        }
    }
}

/// Create test dataset with deterministic values
fn create_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut features = Vec::with_capacity(num_rows * num_features);
    for f in 0..num_features {
        for r in 0..num_rows {
            let mut hasher = DefaultHasher::new();
            (r, f).hash(&mut hasher);
            features.push((hasher.finish() % 256) as u8);
        }
    }

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

    BinnedDataset::new(num_rows, features, targets, feature_info)
}

/// Create gradients and hessians that encourage splitting
fn create_grad_hess(num_rows: usize) -> (Vec<f32>, Vec<f32>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let gradients: Vec<f32> = (0..num_rows)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (i, "grad").hash(&mut hasher);
            (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect();

    let hessians: Vec<f32> = vec![1.0; num_rows];

    (gradients, hessians)
}

/// Benchmark Full CPU (scalar backend with TreeGrower)
fn bench_full_cpu(
    dataset: &BinnedDataset,
    gradients: &[f32],
    hessians: &[f32],
    max_depth: usize,
    warmup: usize,
    iterations: usize,
) -> Duration {
    let grower = TreeGrower::new()
        .with_max_depth(max_depth)
        .with_max_leaves(1 << max_depth)
        .with_min_samples_leaf(10)
        .with_backend(BackendType::Scalar);

    // Warmup
    for _ in 0..warmup {
        let _ = grower.grow(dataset, gradients, hessians);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let tree = grower.grow(dataset, gradients, hessians);
        std::hint::black_box(&tree);
    }
    start.elapsed()
}

/// Benchmark WGPU Hybrid (GPU histogram + CPU partition, best-first)
#[cfg(feature = "gpu")]
fn bench_wgpu_hybrid(
    dataset: &BinnedDataset,
    gradients: &[f32],
    hessians: &[f32],
    max_depth: usize,
    warmup: usize,
    iterations: usize,
) -> Option<Duration> {
    // Check if WGPU is available
    if treeboost::backend::WgpuBackend::new().is_none() {
        return None;
    }

    let grower = TreeGrower::new()
        .with_max_depth(max_depth)
        .with_max_leaves(1 << max_depth)
        .with_min_samples_leaf(10)
        .with_backend(BackendType::Wgpu)
        .with_gpu_batch_size(32);

    // Warmup
    for _ in 0..warmup {
        let _ = grower.grow(dataset, gradients, hessians);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let tree = grower.grow(dataset, gradients, hessians);
        std::hint::black_box(&tree);
    }
    Some(start.elapsed())
}

/// Benchmark WGPU Full GPU (level-wise)
#[cfg(feature = "gpu")]
fn bench_wgpu_full(
    device: Arc<WgpuDevice>,
    dataset: &BinnedDataset,
    gradients: &[f32],
    hessians: &[f32],
    max_depth: usize,
    warmup: usize,
    iterations: usize,
) -> Duration {
    let mut builder = WgpuFullBuilder::new(device);

    let row_indices: Vec<usize> = (0..dataset.num_rows()).collect();
    let max_leaves = 1 << max_depth;
    let lambda = 1.0f32;
    let min_samples_leaf = 10usize;
    let min_hessian_leaf = 1.0f32;
    let min_gain = 0.0f32;
    let learning_rate = 0.1f32;

    // Warmup
    for _ in 0..warmup {
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
    for _ in 0..iterations {
        let tree = builder.build_tree(
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
        std::hint::black_box(&tree);
    }
    start.elapsed()
}

/// Benchmark CUDA Hybrid (GPU histogram + CPU partition, best-first)
#[cfg(feature = "cuda")]
fn bench_cuda_hybrid(
    dataset: &BinnedDataset,
    gradients: &[f32],
    hessians: &[f32],
    max_depth: usize,
    warmup: usize,
    iterations: usize,
) -> Option<Duration> {
    // Check if CUDA is available
    if treeboost::backend::CudaBackend::new().is_none() {
        return None;
    }

    let grower = TreeGrower::new()
        .with_max_depth(max_depth)
        .with_max_leaves(1 << max_depth)
        .with_min_samples_leaf(10)
        .with_backend(BackendType::Cuda)
        .with_gpu_batch_size(32);

    // Warmup
    for _ in 0..warmup {
        let _ = grower.grow(dataset, gradients, hessians);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let tree = grower.grow(dataset, gradients, hessians);
        std::hint::black_box(&tree);
    }
    Some(start.elapsed())
}

/// Benchmark CUDA Full GPU (level-wise)
#[cfg(feature = "cuda")]
fn bench_cuda_full(
    device: CudaArc<CudaDevice>,
    dataset: &BinnedDataset,
    gradients: &[f32],
    hessians: &[f32],
    max_depth: usize,
    warmup: usize,
    iterations: usize,
) -> Duration {
    let mut builder = FullCudaTreeBuilder::new(device);

    let row_indices: Vec<usize> = (0..dataset.num_rows()).collect();
    let max_leaves = 1 << max_depth;
    let lambda = 1.0f32;
    let min_samples_leaf = 10usize;
    let min_hessian_leaf = 1.0f32;
    let min_gain = 0.0f32;
    let learning_rate = 0.1f32;

    // Warmup
    for _ in 0..warmup {
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
    for _ in 0..iterations {
        let tree = builder.build_tree(
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
        std::hint::black_box(&tree);
    }
    start.elapsed()
}

fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              TreeBoost Backend Benchmark (CPU vs GPU)                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Modes:                                                                      ║");
    println!("║    - Hybrid: GPU histogram + CPU partition (best-first, subtraction trick)  ║");
    println!("║    - Full:   GPU histogram + GPU partition (level-wise, minimal PCIe)       ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_system_info() {
    println!("System Information:");
    #[cfg(target_arch = "x86_64")]
    {
        println!("  Architecture: x86_64");
        println!("  AVX2:         {}", if is_x86_feature_detected!("avx2") { "Yes" } else { "No" });
        println!("  AVX-512:      {}", if is_x86_feature_detected!("avx512f") { "Yes" } else { "No" });
    }
    #[cfg(target_arch = "aarch64")]
    println!("  Architecture: aarch64");
    println!();
}

fn main() {
    print_header();
    print_system_info();

    // Detect available backends
    println!("Backend Availability:");
    println!("  Scalar (CPU):  Always available");

    #[cfg(feature = "gpu")]
    let wgpu_device = {
        match WgpuDevice::new() {
            Some(d) => {
                println!("  WGPU:          {} ({:?})", d.name(), d.backend());
                Some(Arc::new(d))
            }
            None => {
                println!("  WGPU:          Not available (no GPU detected)");
                None
            }
        }
    };
    #[cfg(not(feature = "gpu"))]
    {
        println!("  WGPU:          Not compiled (use --features gpu)");
        let wgpu_device: Option<()> = None;
        let _ = wgpu_device;
    }

    #[cfg(feature = "cuda")]
    let cuda_device = {
        match CudaDevice::new() {
            Some(d) => {
                println!("  CUDA:          {}", d.name());
                Some(CudaArc::new(d))
            }
            None => {
                println!("  CUDA:          Not available (no NVIDIA GPU or driver)");
                None
            }
        }
    };
    #[cfg(not(feature = "cuda"))]
    {
        println!("  CUDA:          Not compiled (use --features cuda)");
        let cuda_device: Option<()> = None;
        let _ = cuda_device;
    }

    println!();

    // Benchmark configurations
    let configs = [
        (50_000, 20, 2, 10),   // (rows, features, warmup, iterations)
        (100_000, 20, 2, 10),
        (200_000, 20, 2, 8),
        (500_000, 20, 2, 5),
    ];
    let max_depth = 6;

    // Run benchmarks
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("  Benchmark Results (tree building, depth={})", max_depth);
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    // Table header
    print!("{:>10} │", "Rows");
    print!(" {:>12}", "CPU");
    #[cfg(feature = "gpu")]
    if wgpu_device.is_some() {
        print!(" │ {:>12} │ {:>12}", "WGPU Hybrid", "WGPU Full");
    }
    #[cfg(feature = "cuda")]
    if cuda_device.is_some() {
        print!(" │ {:>12} │ {:>12}", "CUDA Hybrid", "CUDA Full");
    }
    println!();

    print!("{:>10} │", "");
    print!(" {:>12}", "(ms/tree)");
    #[cfg(feature = "gpu")]
    if wgpu_device.is_some() {
        print!(" │ {:>12} │ {:>12}", "(ms/tree)", "(ms/tree)");
    }
    #[cfg(feature = "cuda")]
    if cuda_device.is_some() {
        print!(" │ {:>12} │ {:>12}", "(ms/tree)", "(ms/tree)");
    }
    println!();

    println!("{}", "─".repeat(90));

    let mut all_results: Vec<Vec<Option<BenchResult>>> = Vec::new();

    for &(num_rows, num_features, warmup, iterations) in &configs {
        let dataset = create_dataset(num_rows, num_features);
        let (gradients, hessians) = create_grad_hess(num_rows);

        let mut row_results: Vec<Option<BenchResult>> = Vec::new();

        // CPU benchmark
        let cpu_time = bench_full_cpu(&dataset, &gradients, &hessians, max_depth, warmup, iterations);
        let cpu_result = BenchResult::new("CPU", num_rows, num_features, cpu_time, iterations);

        print!("{:>10} │", num_rows);
        print!(" {:>12.2}", cpu_result.time_ms);
        row_results.push(Some(cpu_result));

        // WGPU benchmarks
        #[cfg(feature = "gpu")]
        {
            if let Some(ref device) = wgpu_device {
                // WGPU Hybrid
                if let Some(wgpu_hybrid_time) = bench_wgpu_hybrid(&dataset, &gradients, &hessians, max_depth, warmup, iterations) {
                    let result = BenchResult::new("WGPU Hybrid", num_rows, num_features, wgpu_hybrid_time, iterations);
                    print!(" │ {:>12.2}", result.time_ms);
                    row_results.push(Some(result));
                } else {
                    print!(" │ {:>12}", "N/A");
                    row_results.push(None);
                }

                // WGPU Full
                let wgpu_full_time = bench_wgpu_full(
                    Arc::clone(device),
                    &dataset,
                    &gradients,
                    &hessians,
                    max_depth,
                    warmup,
                    iterations,
                );
                let result = BenchResult::new("WGPU Full", num_rows, num_features, wgpu_full_time, iterations);
                print!(" │ {:>12.2}", result.time_ms);
                row_results.push(Some(result));
            }
        }

        // CUDA benchmarks
        #[cfg(feature = "cuda")]
        {
            if let Some(ref device) = cuda_device {
                // CUDA Hybrid
                if let Some(cuda_hybrid_time) = bench_cuda_hybrid(&dataset, &gradients, &hessians, max_depth, warmup, iterations) {
                    let result = BenchResult::new("CUDA Hybrid", num_rows, num_features, cuda_hybrid_time, iterations);
                    print!(" │ {:>12.2}", result.time_ms);
                    row_results.push(Some(result));
                } else {
                    print!(" │ {:>12}", "N/A");
                    row_results.push(None);
                }

                // CUDA Full
                let cuda_full_time = bench_cuda_full(
                    CudaArc::clone(device),
                    &dataset,
                    &gradients,
                    &hessians,
                    max_depth,
                    warmup,
                    iterations,
                );
                let result = BenchResult::new("CUDA Full", num_rows, num_features, cuda_full_time, iterations);
                print!(" │ {:>12.2}", result.time_ms);
                row_results.push(Some(result));
            }
        }

        println!();
        all_results.push(row_results);
    }

    // Print speedup summary
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("  Speedup vs CPU (higher is better)");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    print!("{:>10} │", "Rows");
    #[cfg(feature = "gpu")]
    if wgpu_device.is_some() {
        print!(" {:>12} │ {:>12}", "WGPU Hybrid", "WGPU Full");
    }
    #[cfg(feature = "cuda")]
    if cuda_device.is_some() {
        print!(" │ {:>12} │ {:>12}", "CUDA Hybrid", "CUDA Full");
    }
    println!();
    println!("{}", "─".repeat(70));

    for (i, &(num_rows, _, _, _)) in configs.iter().enumerate() {
        let row = &all_results[i];
        let cpu_time = row[0].as_ref().map(|r| r.time_ms).unwrap_or(1.0);

        print!("{:>10} │", num_rows);

        let mut idx = 1;
        #[cfg(feature = "gpu")]
        if wgpu_device.is_some() {
            // WGPU Hybrid speedup
            if let Some(ref result) = row.get(idx).and_then(|r| r.as_ref()) {
                let speedup = cpu_time / result.time_ms;
                print!(" {:>11.2}x", speedup);
            } else {
                print!(" {:>12}", "N/A");
            }
            idx += 1;

            // WGPU Full speedup
            if let Some(ref result) = row.get(idx).and_then(|r| r.as_ref()) {
                let speedup = cpu_time / result.time_ms;
                print!(" │ {:>11.2}x", speedup);
            } else {
                print!(" │ {:>12}", "N/A");
            }
            idx += 1;
        }

        #[cfg(feature = "cuda")]
        if cuda_device.is_some() {
            // CUDA Hybrid speedup
            if let Some(ref result) = row.get(idx).and_then(|r| r.as_ref()) {
                let speedup = cpu_time / result.time_ms;
                print!(" │ {:>11.2}x", speedup);
            } else {
                print!(" │ {:>12}", "N/A");
            }
            idx += 1;

            // CUDA Full speedup
            if let Some(ref result) = row.get(idx).and_then(|r| r.as_ref()) {
                let speedup = cpu_time / result.time_ms;
                print!(" │ {:>11.2}x", speedup);
            } else {
                print!(" │ {:>12}", "N/A");
            }
            let _ = idx;
        }

        println!();
    }

    // CUDA vs WGPU comparison if both available
    #[cfg(all(feature = "gpu", feature = "cuda"))]
    if wgpu_device.is_some() && cuda_device.is_some() {
        println!();
        println!("═══════════════════════════════════════════════════════════════════════════════");
        println!("  CUDA vs WGPU (speedup, >1 means CUDA is faster)");
        println!("═══════════════════════════════════════════════════════════════════════════════");
        println!();
        println!("{:>10} │ {:>15} │ {:>15}", "Rows", "Hybrid", "Full");
        println!("{}", "─".repeat(50));

        for (i, &(num_rows, _, _, _)) in configs.iter().enumerate() {
            let row = &all_results[i];
            print!("{:>10} │", num_rows);

            // Hybrid comparison (indices: 1=WGPU Hybrid, 3=CUDA Hybrid)
            if let (Some(Some(wgpu)), Some(Some(cuda))) = (row.get(1), row.get(3)) {
                let speedup = wgpu.time_ms / cuda.time_ms;
                print!(" {:>14.2}x", speedup);
            } else {
                print!(" {:>15}", "N/A");
            }

            // Full comparison (indices: 2=WGPU Full, 4=CUDA Full)
            if let (Some(Some(wgpu)), Some(Some(cuda))) = (row.get(2), row.get(4)) {
                let speedup = wgpu.time_ms / cuda.time_ms;
                print!(" │ {:>14.2}x", speedup);
            } else {
                print!(" │ {:>15}", "N/A");
            }

            println!();
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("  Legend:");
    println!("    Hybrid = GPU histogram + CPU partition (best-first growth)");
    println!("    Full   = GPU histogram + GPU partition (level-wise growth)");
    println!("═══════════════════════════════════════════════════════════════════════════════");
}
