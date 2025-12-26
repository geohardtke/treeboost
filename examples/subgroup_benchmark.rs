//! Benchmark: Subgroup Shader vs Base Shader Performance
//!
//! Compares GPU histogram building with and without subgroup optimizations.
//! Run with: cargo run --release --features gpu --example subgroup_benchmark

use std::time::Instant;
use treeboost::backend::wgpu::WgpuBackend;
use treeboost::backend::HistogramBackend;
use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 50;

fn main() {
    println!("Subgroup Shader Benchmark: Base vs Subgroup");
    println!("============================================\n");

    // Initialize GPU backend
    let backend = match WgpuBackend::new() {
        Some(b) => b,
        None => {
            println!("No GPU available");
            return;
        }
    };

    println!("GPU: {} ({:?})", backend.device_name(), backend.backend_type());

    let (min_sg, max_sg) = backend.subgroup_size();
    let sg_active = backend.has_subgroups();
    println!("Subgroups: {} (size: {}-{})",
        if sg_active { "ACTIVE" } else { "disabled" },
        min_sg, max_sg);

    if !sg_active {
        println!("\nWARNING: Subgroups not available - cannot compare shaders.");
        println!("Running base shader benchmark only.\n");
    }

    println!();

    // Test configurations
    let configs = [
        (50_000, 20),
        (100_000, 20),
        (250_000, 20),
        (500_000, 20),
        (1_000_000, 20),
    ];

    if sg_active {
        println!(
            "{:>10} | {:>12} | {:>12} | {:>8}",
            "Rows", "Base (ms)", "Subgroup (ms)", "Speedup"
        );
        println!("{}", "-".repeat(55));

        for (num_rows, num_features) in configs {
            let dataset = create_test_dataset(num_rows, num_features);
            let grad_hess = generate_grad_hess(num_rows);
            let indices: Vec<usize> = (0..num_rows).collect();

            // Warmup both paths
            for _ in 0..WARMUP_ITERS {
                let _ = backend.build_histograms_base_shader(&dataset, &grad_hess, &indices);
                let _ = backend.build_histograms(&dataset, &grad_hess, &indices);
            }

            // Benchmark base shader
            let start = Instant::now();
            for _ in 0..BENCH_ITERS {
                let _ = backend.build_histograms_base_shader(&dataset, &grad_hess, &indices);
            }
            let base_time = start.elapsed() / BENCH_ITERS as u32;

            // Benchmark subgroup shader
            let start = Instant::now();
            for _ in 0..BENCH_ITERS {
                let _ = backend.build_histograms(&dataset, &grad_hess, &indices);
            }
            let sg_time = start.elapsed() / BENCH_ITERS as u32;

            let speedup = base_time.as_secs_f64() / sg_time.as_secs_f64();

            println!(
                "{:>10} | {:>10.3} | {:>12.3} | {:>7.2}x",
                format_num(num_rows),
                base_time.as_secs_f64() * 1000.0,
                sg_time.as_secs_f64() * 1000.0,
                speedup,
            );
        }

        println!();
        println!("Interpretation:");
        println!("  Speedup > 1.0 = Subgroup shader is faster");
        println!("  Speedup < 1.0 = Base shader is faster");
        println!();

        // Detailed breakdown at 500K rows
        println!("--- Detailed Comparison @ 500K rows ---\n");

        let num_rows = 500_000;
        let num_features = 20;
        let dataset = create_test_dataset(num_rows, num_features);
        let grad_hess = generate_grad_hess(num_rows);
        let indices: Vec<usize> = (0..num_rows).collect();

        // Profile subgroup shader
        let (_, profile_sg) = backend.build_histograms_profiled(&dataset, &grad_hess, &indices);

        println!("Subgroup shader:");
        println!("  GPU compute: {:.3} ms", profile_sg.gpu_execute.as_secs_f64() * 1000.0);
        println!("  Total:       {:.3} ms", profile_sg.total.as_secs_f64() * 1000.0);

    } else {
        // Base shader only
        println!(
            "{:>10} | {:>12} | {:>12}",
            "Rows", "Time (ms)", "Throughput"
        );
        println!("{}", "-".repeat(42));

        for (num_rows, num_features) in configs {
            let dataset = create_test_dataset(num_rows, num_features);
            let grad_hess = generate_grad_hess(num_rows);
            let indices: Vec<usize> = (0..num_rows).collect();

            for _ in 0..WARMUP_ITERS {
                let _ = backend.build_histograms(&dataset, &grad_hess, &indices);
            }

            let start = Instant::now();
            for _ in 0..BENCH_ITERS {
                let _ = backend.build_histograms(&dataset, &grad_hess, &indices);
            }
            let elapsed = start.elapsed();
            let avg_time = elapsed / BENCH_ITERS as u32;

            let rows_per_sec = (num_rows as f64 * BENCH_ITERS as f64) / elapsed.as_secs_f64();
            let throughput = format!("{:.1}M rows/s", rows_per_sec / 1_000_000.0);

            println!(
                "{:>10} | {:>10.3} | {:>12}",
                format_num(num_rows),
                avg_time.as_secs_f64() * 1000.0,
                throughput,
            );
        }
    }
}

fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Column-major bins with locality (adjacent rows often have similar values)
    // This helps subgroup optimization since threads processing nearby rows
    // are more likely to hit the same bin
    let mut features = vec![0u8; num_rows * num_features];
    for f in 0..num_features {
        for r in 0..num_rows {
            // Groups of 8 rows get same base value
            let mut hasher = DefaultHasher::new();
            (r / 8, f).hash(&mut hasher);
            let base = (hasher.finish() % 200) as u8;
            let jitter = ((r * 7 + f * 13) % 10) as u8;
            features[f * num_rows + r] = base.wrapping_add(jitter);
        }
    }

    let targets: Vec<f32> = (0..num_rows).map(|i| i as f32).collect();
    let feature_info: Vec<FeatureInfo> = (0..num_features)
        .map(|i| FeatureInfo {
            name: format!("feature_{}", i),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
        })
        .collect();

    BinnedDataset::new(num_rows, features, targets, feature_info)
}

fn generate_grad_hess(num_rows: usize) -> Vec<(f32, f32)> {
    (0..num_rows)
        .map(|i| {
            let x = i as f32 / num_rows as f32;
            let grad = (x * 6.28).sin() * 2.0;
            let hess = 1.0 + (x * 3.14).cos().abs();
            (grad, hess)
        })
        .collect()
}

fn format_num(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}
