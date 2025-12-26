//! Deep dive into GPU vs CPU backend usage during training
//!
//! Run: cargo run --release --features gpu --example training_loop_deep_dive
//!
//! This profiles which backend is used at each point during training and
//! measures timing to identify optimization opportunities.

use std::time::{Duration, Instant};

use treeboost::backend::{BackendConfig, BackendSelector, BackendType};
use treeboost::booster::{GBDTConfig, GBDTModel, LossType};
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use treeboost::tree::TreeGrower;

#[cfg(feature = "gpu")]
use treeboost::backend::wgpu::{GpuProfileData, WgpuBackend};

fn create_dataset(num_rows: usize, num_features: usize, seed: u64) -> BinnedDataset {
    let mut state = seed;
    let mut next_rand = || -> f32 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((state >> 16) & 0x7FFF) as f32 / 32767.0
    };

    // Column-major bin storage
    let mut bins = Vec::with_capacity(num_rows * num_features);
    for _f in 0..num_features {
        for _r in 0..num_rows {
            bins.push((next_rand() * 255.0) as u8);
        }
    }

    // Targets with pattern
    let targets: Vec<f32> = (0..num_rows)
        .map(|i| {
            let f0 = bins[i] as f32 / 255.0;
            let f1 = bins[num_rows + i] as f32 / 255.0;
            f0 * 10.0 + f1 * 5.0 + next_rand() * 0.5
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

    BinnedDataset::new(num_rows, bins, targets, feature_info)
}

fn section_header(title: &str) {
    println!("\n\n{}", "=".repeat(70));
    println!("{}", title);
    println!("{}", "=".repeat(70));
}

/// Format duration as microseconds or milliseconds depending on magnitude
fn fmt_duration(d: Duration) -> String {
    let us = d.as_nanos() as f64 / 1000.0;
    if us < 1000.0 {
        format!("{:>8.1} µs", us)
    } else {
        format!("{:>8.3} ms", us / 1000.0)
    }
}

/// Format duration with percentage of total
fn fmt_duration_pct(d: Duration, total: Duration) -> String {
    let pct = d.as_secs_f64() / total.as_secs_f64() * 100.0;
    format!("{} ({:>5.1}%)", fmt_duration(d), pct)
}

#[cfg(feature = "gpu")]
fn print_gpu_profile(profile: &GpuProfileData, label: &str) {
    println!("\n  {} ({} rows, {} features, {} indices):",
        label, profile.num_rows, profile.num_features, profile.num_indices);
    println!("  {}", "-".repeat(60));

    // CPU Preprocessing
    println!("  CPU Preprocessing:");
    println!("    Indices convert (usize→u32):  {}", fmt_duration_pct(profile.indices_convert, profile.total));
    println!("    Bins pack/align check:        {}", fmt_duration_pct(profile.bins_pack, profile.total));

    // GPU Buffer Management
    println!("  GPU Buffer Management:");
    println!("    Buffer allocation:            {}", fmt_duration_pct(profile.buffer_alloc, profile.total));
    println!("    Upload params:                {}", fmt_duration_pct(profile.upload_params, profile.total));
    if profile.bins_cached {
        println!("    Upload bins:                  CACHED (skipped)");
    } else {
        println!("    Upload bins:                  {}", fmt_duration_pct(profile.upload_bins, profile.total));
    }
    println!("    Upload grad/hess:             {}", fmt_duration_pct(profile.upload_grad_hess, profile.total));
    println!("    Upload indices:               {}", fmt_duration_pct(profile.upload_indices, profile.total));

    // GPU Execution
    println!("  GPU Execution:");
    println!("    Bind group create:            {}", fmt_duration_pct(profile.bind_group_create, profile.total));
    println!("    Encode commands:              {}", fmt_duration_pct(profile.encode_commands, profile.total));
    println!("    GPU compute (submit+wait):    {}", fmt_duration_pct(profile.gpu_execute, profile.total));

    // Results Download
    println!("  Results Download:");
    println!("    Download histograms:          {}", fmt_duration_pct(profile.download_results, profile.total));
    println!("    Unpack to Histogram structs:  {}", fmt_duration_pct(profile.unpack_histograms, profile.total));

    println!("  {}", "-".repeat(60));
    println!("  TOTAL:                          {}", fmt_duration(profile.total));

    // Breakdown summary
    let cpu_time = profile.indices_convert + profile.bins_pack + profile.unpack_histograms;
    let upload_time = profile.upload_params + profile.upload_bins + profile.upload_grad_hess + profile.upload_indices;
    let download_time = profile.download_results;
    let gpu_overhead = profile.buffer_alloc + profile.bind_group_create + profile.encode_commands;
    let gpu_compute = profile.gpu_execute;

    println!("\n  Time breakdown:");
    println!("    CPU work:       {}", fmt_duration_pct(cpu_time, profile.total));
    println!("    Upload:         {}", fmt_duration_pct(upload_time, profile.total));
    println!("    GPU overhead:   {}", fmt_duration_pct(gpu_overhead, profile.total));
    println!("    GPU compute:    {}", fmt_duration_pct(gpu_compute, profile.total));
    println!("    Download:       {}", fmt_duration_pct(download_time, profile.total));
}

fn main() {
    println!("Training Loop Deep Dive: CPU vs GPU Backend Analysis");
    println!("=====================================================\n");

    let num_rows = 500_000;
    let num_features = 20;
    let num_rounds = 10;
    let max_depth = 6;

    println!("Configuration:");
    println!("  Rows:       {:>10}", num_rows);
    println!("  Features:   {:>10}", num_features);
    println!("  Rounds:     {:>10}", num_rounds);
    println!("  Max Depth:  {:>10}", max_depth);

    let dataset = create_dataset(num_rows, num_features, 42);

    // =========================================================================
    // SECTION 1: Isolated Histogram Building - CPU vs GPU by row count
    // =========================================================================
    section_header("SECTION 1: ISOLATED HISTOGRAM BUILDING BY ROW COUNT");

    println!("\nThis shows where GPU becomes faster than CPU for histogram building.");
    println!("The GPU backend internally falls back to CPU for rows < 5K.\n");

    let grad_hess: Vec<(f32, f32)> = (0..num_rows)
        .map(|i| (-(dataset.targets()[i]), 1.0))
        .collect();

    // Test different row counts
    let test_sizes = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000];

    println!("{:>10} | {:>12} | {:>12} | {:>10} | GPU Used?", "Rows", "Scalar (ms)", "GPU (ms)", "Speedup");
    println!("{}", "-".repeat(70));

    let scalar_config = BackendConfig {
        preferred: BackendType::Scalar,
        ..Default::default()
    };
    let scalar_backend = BackendSelector::with_config(scalar_config).select(num_rows);

    #[cfg(feature = "gpu")]
    let gpu_config = BackendConfig {
        preferred: BackendType::Wgpu,
        ..Default::default()
    };
    #[cfg(feature = "gpu")]
    let gpu_backend = BackendSelector::with_config(gpu_config).select(num_rows);

    for &size in &test_sizes {
        if size > num_rows {
            continue;
        }

        let row_indices: Vec<usize> = (0..size).collect();

        // Warmup
        let _ = scalar_backend.build_histograms(&dataset, &grad_hess, &row_indices);
        #[cfg(feature = "gpu")]
        let _ = gpu_backend.build_histograms(&dataset, &grad_hess, &row_indices);

        // Benchmark scalar
        let iterations = if size < 10_000 { 20 } else { 5 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = scalar_backend.build_histograms(&dataset, &grad_hess, &row_indices);
        }
        let scalar_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        // Benchmark GPU
        #[cfg(feature = "gpu")]
        let (gpu_time, gpu_used) = {
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = gpu_backend.build_histograms(&dataset, &grad_hess, &row_indices);
            }
            let time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
            let used = size >= 5_000; // GPU_MIN_ROWS threshold
            (time, used)
        };

        #[cfg(not(feature = "gpu"))]
        let (gpu_time, gpu_used) = (f64::NAN, false);

        let speedup = scalar_time / gpu_time;
        let speedup_str = if speedup.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.2}x", speedup)
        };
        let gpu_used_str = if gpu_used { "Yes (GPU)" } else { "No (CPU fallback)" };

        println!("{:>10} | {:>12.3} | {:>12.3} | {:>10} | {}",
            size, scalar_time, gpu_time, speedup_str, gpu_used_str);
    }

    // =========================================================================
    // SECTION 1B: Detailed GPU Profiling
    // =========================================================================
    #[cfg(feature = "gpu")]
    {
        section_header("SECTION 1B: DETAILED GPU OPERATION BREAKDOWN");

        println!("\nThis shows EXACTLY where time is spent in GPU histogram building.");
        println!("Each operation is timed individually to identify bottlenecks.\n");

        let gpu_backend = match WgpuBackend::new() {
            Some(b) => b,
            None => {
                println!("No GPU available, skipping detailed profiling");
                return;
            }
        };

        println!("GPU Device: {}", gpu_backend.device_name());
        println!("Backend: {:?}", gpu_backend.backend_type());

        // Profile at different row counts
        let profile_sizes = [10_000, 50_000, 100_000, 250_000, 500_000];

        for &size in &profile_sizes {
            if size > num_rows {
                continue;
            }

            let row_indices: Vec<usize> = (0..size).collect();

            // Warmup (also primes the bins cache)
            let _ = gpu_backend.build_histograms_profiled(&dataset, &grad_hess, &row_indices);

            // Profile (second call will have bins cached)
            let (_hists, profile) = gpu_backend.build_histograms_profiled(&dataset, &grad_hess, &row_indices);

            print_gpu_profile(&profile, &format!("Profile @ {} rows", size));
        }

        // Compare first call (cold) vs second call (warm/cached)
        println!("\n\n  === Cold vs Warm Start Comparison ===");
        println!("  (First call uploads bins, subsequent calls have bins cached)");

        // Create a fresh backend to see cold start
        let fresh_backend = WgpuBackend::new().unwrap();
        let row_indices: Vec<usize> = (0..100_000).collect();

        let (_hists, cold_profile) = fresh_backend.build_histograms_profiled(&dataset, &grad_hess, &row_indices);
        let (_hists, warm_profile) = fresh_backend.build_histograms_profiled(&dataset, &grad_hess, &row_indices);

        println!("\n  Cold start (first call):");
        println!("    Bins upload: {} (bins_cached: {})",
            fmt_duration(cold_profile.upload_bins), cold_profile.bins_cached);
        println!("    Total:       {}", fmt_duration(cold_profile.total));

        println!("\n  Warm start (second call):");
        println!("    Bins upload: {} (bins_cached: {})",
            fmt_duration(warm_profile.upload_bins), warm_profile.bins_cached);
        println!("    Total:       {}", fmt_duration(warm_profile.total));

        let savings = cold_profile.total.as_secs_f64() - warm_profile.total.as_secs_f64();
        println!("\n  Caching saves: {:.3} ms per histogram build", savings * 1000.0);
    }

    // =========================================================================
    // SECTION 2: Full Training Comparison
    // =========================================================================
    section_header("SECTION 2: FULL TRAINING TIME COMPARISON");

    // Train with Scalar backend
    println!("\nTraining with Scalar backend...");
    let scalar_config = GBDTConfig::new()
        .with_num_rounds(num_rounds)
        .with_max_depth(max_depth)
        .with_learning_rate(0.1)
        .with_backend(BackendType::Scalar);

    // Warmup
    let _ = GBDTModel::train_binned(&dataset, scalar_config.clone());

    let start = Instant::now();
    let _ = GBDTModel::train_binned(&dataset, scalar_config.clone());
    let scalar_train_time = start.elapsed();
    println!("  Total time: {:.2} ms ({:.2} ms/round)",
        scalar_train_time.as_secs_f64() * 1000.0,
        scalar_train_time.as_secs_f64() * 1000.0 / num_rounds as f64);

    // Train with GPU backend
    #[cfg(feature = "gpu")]
    let gpu_train_time = {
        println!("\nTraining with GPU backend...");
        let gpu_config = GBDTConfig::new()
            .with_num_rounds(num_rounds)
            .with_max_depth(max_depth)
            .with_learning_rate(0.1)
            .with_backend(BackendType::Wgpu);

        // Warmup
        let _ = GBDTModel::train_binned(&dataset, gpu_config.clone());

        let start = Instant::now();
        let _ = GBDTModel::train_binned(&dataset, gpu_config.clone());
        let gpu_train_time = start.elapsed();
        println!("  Total time: {:.2} ms ({:.2} ms/round)",
            gpu_train_time.as_secs_f64() * 1000.0,
            gpu_train_time.as_secs_f64() * 1000.0 / num_rounds as f64);

        let speedup = scalar_train_time.as_secs_f64() / gpu_train_time.as_secs_f64();
        println!("\n  Overall GPU Speedup: {:.2}x", speedup);
        gpu_train_time
    };

    // =========================================================================
    // SECTION 3: Per-Operation Breakdown
    // =========================================================================
    section_header("SECTION 3: PER-OPERATION BREAKDOWN");

    let loss_fn = LossType::Mse.create();
    let targets = dataset.targets().to_vec();

    #[cfg(feature = "gpu")]
    let backends: Vec<(&str, BackendType)> = vec![
        ("Scalar (CPU)", BackendType::Scalar),
        ("GPU (Wgpu)", BackendType::Wgpu),
    ];

    #[cfg(not(feature = "gpu"))]
    let backends: Vec<(&str, BackendType)> = vec![
        ("Scalar (CPU)", BackendType::Scalar),
    ];

    for (backend_name, backend_type) in backends {
        println!("\n{} Backend:", backend_name);
        println!("{}", "-".repeat(50));

        let grower = TreeGrower::new()
            .with_max_depth(max_depth)
            .with_max_leaves(63)
            .with_learning_rate(0.1)
            .with_backend(backend_type.clone());

        let mut predictions = vec![0.0f32; num_rows];
        let mut gradients = vec![0.0f32; num_rows];
        let mut hessians = vec![0.0f32; num_rows];
        let train_indices: Vec<usize> = (0..num_rows).collect();

        let mut time_gradient_ms = 0.0;
        let mut time_grow_ms = 0.0;
        let mut time_predict_ms = 0.0;

        // Warmup
        for &idx in &train_indices {
            let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
            gradients[idx] = g;
            hessians[idx] = h;
        }
        let _ = grower.grow_with_indices(&dataset, &gradients, &hessians, &train_indices);

        // Reset
        predictions.fill(0.0);

        let start = Instant::now();
        for _round in 0..num_rounds {
            // Gradient computation
            let t = Instant::now();
            for &idx in &train_indices {
                let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                gradients[idx] = g;
                hessians[idx] = h;
            }
            time_gradient_ms += t.elapsed().as_secs_f64() * 1000.0;

            // Tree grow (includes histogram building + split finding)
            let t = Instant::now();
            let tree = grower.grow_with_indices(&dataset, &gradients, &hessians, &train_indices);
            time_grow_ms += t.elapsed().as_secs_f64() * 1000.0;

            // Prediction update
            let t = Instant::now();
            tree.predict_batch_add(&dataset, &mut predictions);
            time_predict_ms += t.elapsed().as_secs_f64() * 1000.0;
        }
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!("  Gradient computation: {:>8.2} ms ({:>5.1}%)",
            time_gradient_ms, time_gradient_ms / total_ms * 100.0);
        println!("  Tree growing:         {:>8.2} ms ({:>5.1}%)  <- Histogram + Split",
            time_grow_ms, time_grow_ms / total_ms * 100.0);
        println!("  Prediction update:    {:>8.2} ms ({:>5.1}%)",
            time_predict_ms, time_predict_ms / total_ms * 100.0);
        println!("  ---------------------------------");
        println!("  Total:                {:>8.2} ms", total_ms);
        println!("  Per round:            {:>8.2} ms", total_ms / num_rounds as f64);
    }

    // =========================================================================
    // SECTION 4: GPU Internal Fallback Behavior
    // =========================================================================
    #[cfg(feature = "gpu")]
    {
        section_header("SECTION 4: GPU INTERNAL FALLBACK ANALYSIS");

        println!("\nThe GPU backend internally falls back to CPU for small row counts.");
        println!("Threshold: GPU_MIN_ROWS = 5,000 rows");
        println!("\nDuring tree growing, nodes get progressively smaller:");

        // Estimate node sizes at each depth
        println!("\nEstimated node sizes (balanced binary tree):");
        println!("{:>8} | {:>12} | {:>8} | {}", "Depth", "Rows/Node", "# Nodes", "Backend");
        println!("{}", "-".repeat(50));

        for depth in 0..=max_depth {
            let nodes_at_depth = 1usize << depth;
            let rows_per_node = num_rows / nodes_at_depth;
            let backend = if rows_per_node >= 5_000 { "GPU" } else { "CPU (fallback)" };
            println!("{:>8} | {:>12} | {:>8} | {}",
                depth, rows_per_node, nodes_at_depth, backend);
        }

        println!("\nWith max_depth={} and {} rows:", max_depth, num_rows);
        let gpu_depth_limit = (num_rows as f64 / 5_000.0).log2().floor() as usize;
        println!("  - GPU used for depths 0-{} (rows >= 5K)", gpu_depth_limit.min(max_depth));
        println!("  - CPU fallback for depths {}-{} (rows < 5K)", (gpu_depth_limit + 1).min(max_depth), max_depth);

        // Count histogram operations by estimated backend
        let mut gpu_hist_count = 0;
        let mut cpu_hist_count = 0;
        for depth in 0..=max_depth {
            let nodes_at_depth = 1usize << depth;
            let rows_per_node = num_rows / nodes_at_depth;
            // Each internal node needs histogram build for smaller child
            // (larger child uses subtraction)
            if depth < max_depth {
                if rows_per_node / 2 >= 5_000 {
                    gpu_hist_count += nodes_at_depth;
                } else {
                    cpu_hist_count += nodes_at_depth;
                }
            }
        }
        // Add root histogram
        if num_rows >= 5_000 {
            gpu_hist_count += 1;
        } else {
            cpu_hist_count += 1;
        }

        println!("\nEstimated histogram builds per tree:");
        println!("  GPU histograms:  ~{}", gpu_hist_count);
        println!("  CPU histograms:  ~{}", cpu_hist_count);
        println!("  Total:           ~{}", gpu_hist_count + cpu_hist_count);
    }

    // =========================================================================
    // SECTION 5: Data Transfer Analysis
    // =========================================================================
    section_header("SECTION 5: DATA TRANSFER ANALYSIS");

    let bins_size = num_rows * num_features;
    let grad_hess_size = num_rows * 8; // 2 x f32
    let hist_size = num_features * 256 * 12; // 3 x f32 per bin

    println!("\nData sizes:");
    println!("  Bins (dataset):          {:>8.2} MB (uploaded once, cached)", bins_size as f64 / 1_000_000.0);
    println!("  Grad/Hess (per round):   {:>8.2} MB (uploaded every round)", grad_hess_size as f64 / 1_000_000.0);
    println!("  Histograms (download):   {:>8.2} MB (per histogram build)", hist_size as f64 / 1_000_000.0);

    println!("\nPer-round GPU transfer:");
    println!("  Upload:   {:.2} MB (grad/hess)", grad_hess_size as f64 / 1_000_000.0);
    println!("  Download: {:.2} MB (histograms × builds)", hist_size as f64 / 1_000_000.0);

    // Estimate bandwidth requirements
    let transfer_per_round = grad_hess_size + hist_size * 10; // rough estimate
    println!("\n  Estimated transfer/round: {:.2} MB", transfer_per_round as f64 / 1_000_000.0);
    println!("  At 10 GB/s PCIe: {:.2} ms overhead", transfer_per_round as f64 / 10_000_000.0);

    // =========================================================================
    // SECTION 6: Optimization Opportunities
    // =========================================================================
    section_header("SECTION 6: OPTIMIZATION OPPORTUNITIES");

    println!("\nCurrent implementation:");
    println!("  ✓ Bins cached on GPU (no re-upload)");
    println!("  ✓ CPU fallback for small nodes (< 10K rows)");
    println!("  ✓ Sibling subtraction on CPU (trivial operation)");
    println!("  ✓ Split finding on CPU (O(256 × features), fast)");
    println!("  ✗ Grad/Hess uploaded every round");

    println!("\nPotential improvements:");
    println!("  1. Async grad/hess upload during tree traversal");
    println!("  2. Keep histograms on GPU, only download for splits");
    println!("  3. GPU split finding for very wide datasets (100+ features)");
    println!("  4. Batch multiple small histogram builds into one GPU dispatch");
    println!("  5. Double-buffering for grad/hess (overlap upload with compute)");

    // =========================================================================
    // SECTION 7: Recommendations
    // =========================================================================
    section_header("SECTION 7: RECOMMENDATIONS");

    println!("\nFor dataset: {} rows × {} features", num_rows, num_features);

    #[cfg(feature = "gpu")]
    {
        let speedup = scalar_train_time.as_secs_f64() / gpu_train_time.as_secs_f64();

        if speedup > 1.5 {
            println!("\n  ✓ RECOMMENDED: Use GPU backend (Wgpu)");
            println!("    Measured speedup: {:.2}x", speedup);
        } else if speedup > 1.0 {
            println!("\n  ~ GPU provides marginal benefit ({:.2}x)", speedup);
            println!("    Consider GPU for larger datasets");
        } else {
            println!("\n  ✗ RECOMMENDED: Use Scalar backend (CPU)");
            println!("    CPU is {:.2}x faster for this dataset", 1.0 / speedup);
        }

        println!("\nGuidelines:");
        println!("  - < 100K rows:   Use Scalar (GPU overhead dominates)");
        println!("  - 100K-500K:     Test both, GPU may help");
        println!("  - > 500K rows:   Use GPU (significant speedup)");
        println!("  - Many features: GPU helps more (parallel over features)");
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("\n  GPU feature not enabled.");
        println!("  Compile with: cargo run --release --features gpu --example training_loop_deep_dive");
    }
}
