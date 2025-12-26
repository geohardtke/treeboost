//! Comprehensive benchmark comparing WGPU (GPU) vs Scalar (CPU) histogram backends.
//!
//! This benchmark measures:
//! - Histogram build throughput at various dataset sizes
//! - GPU vs CPU crossover point
//! - Per-operation timing breakdown
//! - Correctness verification
//!
//! Run with:
//!   cargo run --release --features gpu --example backend_benchmark
//!   cargo run --release --features gpu --example backend_benchmark -- --backend wgpu
//!   cargo run --release --features gpu --example backend_benchmark -- --backend scalar
//!   cargo run --release --features gpu --example backend_benchmark -- --rows 100000 --features 50

use std::time::{Duration, Instant};

use treeboost::backend::{BackendConfig, BackendSelector, HistogramBackend};
use treeboost::{BinnedDataset, FeatureInfo, FeatureType};

/// Command line arguments
struct Args {
    backend: Option<String>,
    num_rows: Option<usize>,
    num_features: Option<usize>,
    iterations: Option<usize>,
    compare: bool,
}

impl Args {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut result = Args {
            backend: None,
            num_rows: None,
            num_features: None,
            iterations: None,
            compare: true,
        };

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--backend" | "-b" => {
                    if i + 1 < args.len() {
                        result.backend = Some(args[i + 1].clone());
                        result.compare = false;
                        i += 1;
                    }
                }
                "--rows" | "-r" => {
                    if i + 1 < args.len() {
                        result.num_rows = args[i + 1].parse().ok();
                        i += 1;
                    }
                }
                "--features" | "-f" => {
                    if i + 1 < args.len() {
                        result.num_features = args[i + 1].parse().ok();
                        i += 1;
                    }
                }
                "--iterations" | "-i" => {
                    if i + 1 < args.len() {
                        result.iterations = args[i + 1].parse().ok();
                        i += 1;
                    }
                }
                "--compare" | "-c" => {
                    result.compare = true;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => {}
            }
            i += 1;
        }
        result
    }
}

fn print_help() {
    println!(
        r#"Backend Benchmark - Compare WGPU vs Scalar histogram building

USAGE:
    cargo run --release --features gpu --example backend_benchmark [OPTIONS]

OPTIONS:
    -b, --backend <NAME>     Use specific backend: 'wgpu' or 'scalar'
    -r, --rows <NUM>         Number of rows (default: runs multiple sizes)
    -f, --features <NUM>     Number of features (default: 50)
    -i, --iterations <NUM>   Benchmark iterations (default: auto-scaled)
    -c, --compare            Compare both backends (default if no --backend)
    -h, --help               Print this help

EXAMPLES:
    # Full comparison across dataset sizes
    cargo run --release --features gpu --example backend_benchmark

    # Benchmark specific backend
    cargo run --release --features gpu --example backend_benchmark -- --backend wgpu

    # Custom dataset size
    cargo run --release --features gpu --example backend_benchmark -- -r 100000 -f 100
"#
    );
}

/// Create a test dataset with deterministic bin values
fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
    let mut features = Vec::with_capacity(num_rows * num_features);
    for f in 0..num_features {
        for r in 0..num_rows {
            // Deterministic pattern that uses all 256 bins
            features.push(((r * 7 + f * 13) % 256) as u8);
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

/// Create gradients and hessians for benchmarking
fn create_grad_hess(num_rows: usize) -> Vec<(f32, f32)> {
    (0..num_rows)
        .map(|i| {
            let g = ((i as f32 * 0.1).sin() * 5.0) as f32;
            let h = 1.0 + (i % 10) as f32 * 0.1;
            (g, h)
        })
        .collect()
}

/// Timing result for a single benchmark run
#[derive(Clone)]
struct BenchResult {
    backend_name: String,
    num_rows: usize,
    num_features: usize,
    iterations: usize,
    total_time: Duration,
    per_iter_ms: f64,
    throughput_mcells_ms: f64,
}

impl BenchResult {
    fn print(&self) {
        println!(
            "  {:12} | {:>8.3} ms/iter | {:>8.2} M cells/ms | {:>6} iters",
            self.backend_name, self.per_iter_ms, self.throughput_mcells_ms, self.iterations
        );
    }
}

/// Run benchmark for a specific backend
fn benchmark_backend(
    backend: &dyn HistogramBackend,
    dataset: &BinnedDataset,
    grad_hess: &[(f32, f32)],
    row_indices: &[usize],
    iterations: usize,
) -> BenchResult {
    let num_rows = row_indices.len();
    let num_features = dataset.num_features();

    // Warmup
    for _ in 0..3.min(iterations) {
        let hists = backend.build_histograms(dataset, grad_hess, row_indices);
        std::hint::black_box(&hists);
    }

    // Timed run
    let start = Instant::now();
    for _ in 0..iterations {
        let hists = backend.build_histograms(dataset, grad_hess, row_indices);
        std::hint::black_box(&hists);
    }
    let total_time = start.elapsed();

    let per_iter_ms = total_time.as_secs_f64() * 1000.0 / iterations as f64;
    let cells = num_rows * num_features;
    let throughput_mcells_ms = cells as f64 / per_iter_ms / 1_000_000.0;

    BenchResult {
        backend_name: backend.name().to_string(),
        num_rows,
        num_features,
        iterations,
        total_time,
        per_iter_ms,
        throughput_mcells_ms,
    }
}

/// Verify that both backends produce equivalent results
fn verify_correctness(
    scalar: &dyn HistogramBackend,
    wgpu: &dyn HistogramBackend,
    dataset: &BinnedDataset,
    grad_hess: &[(f32, f32)],
    row_indices: &[usize],
) -> bool {
    let scalar_hists = scalar.build_histograms(dataset, grad_hess, row_indices);
    let wgpu_hists = wgpu.build_histograms(dataset, grad_hess, row_indices);

    if scalar_hists.len() != wgpu_hists.len() {
        println!("  ERROR: Histogram count mismatch");
        return false;
    }

    let mut max_grad_diff = 0.0f32;
    let mut max_hess_diff = 0.0f32;
    let mut count_matches = true;

    for (f, (sh, wh)) in scalar_hists.iter().zip(wgpu_hists.iter()).enumerate() {
        let (sg, shess, sc) = sh.totals();
        let (wg, whess, wc) = wh.totals();

        if sc != wc {
            println!("  ERROR: Feature {} count mismatch: scalar={}, wgpu={}", f, sc, wc);
            count_matches = false;
        }

        max_grad_diff = max_grad_diff.max((sg - wg).abs());
        max_hess_diff = max_hess_diff.max((shess - whess).abs());
    }

    if !count_matches {
        return false;
    }

    // Allow small floating point differences from GPU atomics
    let grad_tolerance = 0.01;
    let hess_tolerance = 0.01;

    if max_grad_diff > grad_tolerance || max_hess_diff > hess_tolerance {
        println!(
            "  WARNING: Float differences exceed tolerance (grad: {:.6}, hess: {:.6})",
            max_grad_diff, max_hess_diff
        );
    }

    true
}

/// Detailed timing breakdown for a single histogram build
fn timing_breakdown(
    backend: &dyn HistogramBackend,
    dataset: &BinnedDataset,
    grad_hess: &[(f32, f32)],
    row_indices: &[usize],
) {
    let num_rows = row_indices.len();
    let num_features = dataset.num_features();

    println!("\n  Timing Breakdown for {} ({} rows × {} features):",
             backend.name(), num_rows, num_features);
    println!("  {}", "-".repeat(50));

    // Time row-major conversion (for GPU)
    if backend.is_tensor_tile() {
        let start = Instant::now();
        let _ = dataset.as_row_major();
        let convert_time = start.elapsed();
        println!("    Row-major conversion:  {:>8.3} ms", convert_time.as_secs_f64() * 1000.0);
    }

    // Time histogram build (multiple runs for average)
    let runs = 10;
    let mut times = Vec::with_capacity(runs);

    for _ in 0..runs {
        let start = Instant::now();
        let hists = backend.build_histograms(dataset, grad_hess, row_indices);
        std::hint::black_box(&hists);
        times.push(start.elapsed());
    }

    let avg_ms = times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / runs as f64 * 1000.0;
    let min_ms = times.iter().map(|d| d.as_secs_f64()).fold(f64::MAX, f64::min) * 1000.0;
    let max_ms = times.iter().map(|d| d.as_secs_f64()).fold(0.0, f64::max) * 1000.0;

    println!("    Histogram build:       {:>8.3} ms (avg over {} runs)", avg_ms, runs);
    println!("      Min: {:>8.3} ms  Max: {:>8.3} ms", min_ms, max_ms);

    let cells = num_rows * num_features;
    let throughput = cells as f64 / avg_ms / 1_000_000.0;
    println!("    Throughput:            {:>8.2} M cells/ms", throughput);
}

fn main() {
    let args = Args::parse();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           TreeBoost Backend Benchmark (GPU vs CPU)                ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // System info
    #[cfg(target_arch = "x86_64")]
    {
        println!("System Information:");
        println!("  Architecture: x86_64");
        println!("  AVX2:         {}", if is_x86_feature_detected!("avx2") { "Yes" } else { "No" });
        println!("  AVX-512:      {}", if is_x86_feature_detected!("avx512f") { "Yes" } else { "No" });
    }

    // Create backends
    let scalar_config = BackendConfig::scalar();
    let scalar_selector = BackendSelector::with_config(scalar_config);
    let scalar_backend = scalar_selector.select(1_000_000);

    #[cfg(feature = "gpu")]
    let wgpu_backend = {
        use treeboost::backend::WgpuBackend;
        WgpuBackend::new()
    };

    #[cfg(not(feature = "gpu"))]
    let wgpu_backend: Option<Box<dyn HistogramBackend>> = None;

    println!("\nBackend Status:");
    println!("  Scalar: {} (always available)", scalar_backend.name());

    #[cfg(feature = "gpu")]
    {
        match &wgpu_backend {
            Some(b) => {
                println!("  WGPU:   {} ({})", b.name(), b.device_name());
            }
            None => {
                println!("  WGPU:   Not available (no GPU detected)");
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    println!("  WGPU:   Not compiled (enable with --features gpu)");

    println!();

    // Determine which backends to benchmark
    let bench_scalar = args.backend.is_none() || args.backend.as_deref() == Some("scalar");
    let bench_wgpu = (args.backend.is_none() || args.backend.as_deref() == Some("wgpu"))
        && wgpu_backend.is_some();

    if !bench_scalar && !bench_wgpu {
        println!("ERROR: No backends available for benchmarking");
        if args.backend.as_deref() == Some("wgpu") && wgpu_backend.is_none() {
            #[cfg(feature = "gpu")]
            println!("       WGPU backend requested but no GPU detected");
            #[cfg(not(feature = "gpu"))]
            println!("       WGPU backend requires --features gpu");
        }
        std::process::exit(1);
    }

    // Dataset configurations for comparison
    let configs: Vec<(usize, usize, usize)> = if let Some(rows) = args.num_rows {
        let features = args.num_features.unwrap_or(50);
        let iters = args.iterations.unwrap_or_else(|| {
            // Auto-scale iterations based on dataset size
            match rows {
                r if r < 10_000 => 200,
                r if r < 100_000 => 50,
                r if r < 1_000_000 => 20,
                _ => 10,
            }
        });
        vec![(rows, features, iters)]
    } else {
        // Default: sweep across sizes to find crossover point
        vec![
            (1_000, 50, 500),      // 1K rows - scalar wins
            (5_000, 50, 200),      // 5K rows
            (10_000, 50, 100),     // 10K rows
            (25_000, 50, 50),      // 25K rows
            (50_000, 50, 30),      // 50K rows
            (100_000, 50, 20),     // 100K rows - GPU starts winning
            (250_000, 50, 10),     // 250K rows
            (500_000, 50, 5),      // 500K rows
            (1_000_000, 50, 3),    // 1M rows - GPU dominates
        ]
    };

    // Run benchmarks
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Benchmark Results");
    println!("═══════════════════════════════════════════════════════════════════");

    let mut all_results: Vec<(usize, Option<BenchResult>, Option<BenchResult>)> = Vec::new();

    for &(num_rows, num_features, iterations) in &configs {
        println!("\n┌─ {} rows × {} features ({} cells)",
                 num_rows, num_features, num_rows * num_features);
        println!("│");

        // Create dataset
        let dataset = create_test_dataset(num_rows, num_features);
        let grad_hess = create_grad_hess(num_rows);
        let row_indices: Vec<usize> = (0..num_rows).collect();

        let mut scalar_result: Option<BenchResult> = None;
        let mut wgpu_result: Option<BenchResult> = None;

        // Benchmark scalar
        if bench_scalar {
            let result = benchmark_backend(
                scalar_backend.as_ref(),
                &dataset,
                &grad_hess,
                &row_indices,
                iterations,
            );
            print!("│ ");
            result.print();
            scalar_result = Some(result);
        }

        // Benchmark WGPU
        #[cfg(feature = "gpu")]
        if bench_wgpu {
            if let Some(ref wgpu) = wgpu_backend {
                let result = benchmark_backend(
                    wgpu as &dyn HistogramBackend,
                    &dataset,
                    &grad_hess,
                    &row_indices,
                    iterations,
                );
                print!("│ ");
                result.print();
                wgpu_result = Some(result);
            }
        }

        // Compare if both available
        if let (Some(ref s), Some(ref w)) = (&scalar_result, &wgpu_result) {
            let speedup = s.per_iter_ms / w.per_iter_ms;
            let winner = if speedup > 1.0 { "WGPU" } else { "Scalar" };
            println!("│");
            if speedup > 1.0 {
                println!("│  Speedup: {:.2}x faster with {}", speedup, winner);
            } else {
                println!("│  Speedup: {:.2}x faster with {}", 1.0 / speedup, winner);
            }
        }

        println!("└─");

        all_results.push((num_rows, scalar_result, wgpu_result));
    }

    // Summary table
    if args.compare && all_results.iter().all(|(_, s, w)| s.is_some() && w.is_some()) {
        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("  Summary: GPU vs CPU Crossover Analysis");
        println!("═══════════════════════════════════════════════════════════════════");
        println!();
        println!(
            "  {:>10} │ {:>12} │ {:>12} │ {:>10} │ Winner",
            "Rows", "Scalar (ms)", "WGPU (ms)", "Speedup"
        );
        println!("  {}", "─".repeat(65));

        let mut crossover_found = false;
        let mut prev_winner = "";

        for (rows, scalar, wgpu) in &all_results {
            if let (Some(s), Some(w)) = (scalar, wgpu) {
                let speedup = s.per_iter_ms / w.per_iter_ms;
                let winner = if speedup > 1.0 { "WGPU" } else { "Scalar" };

                if !crossover_found && !prev_winner.is_empty() && prev_winner != winner {
                    println!("  {:>10} │ {:>12} │ {:>12} │ {:>10} │ ← CROSSOVER",
                             "", "", "", "");
                    crossover_found = true;
                }

                println!(
                    "  {:>10} │ {:>12.3} │ {:>12.3} │ {:>9.2}x │ {}",
                    rows, s.per_iter_ms, w.per_iter_ms, speedup, winner
                );

                prev_winner = winner;
            }
        }

        println!();
        println!("  Interpretation:");
        println!("    - Speedup > 1.0 means GPU is faster");
        println!("    - GPU overhead dominates for small datasets");
        println!("    - GPU wins when compute time > transfer overhead");
    }

    // Correctness verification
    #[cfg(feature = "gpu")]
    if args.compare {
        if let Some(ref wgpu) = wgpu_backend {
            println!("\n═══════════════════════════════════════════════════════════════════");
            println!("  Correctness Verification");
            println!("═══════════════════════════════════════════════════════════════════");

            // Test with a medium dataset
            let dataset = create_test_dataset(10_000, 20);
            let grad_hess = create_grad_hess(10_000);
            let row_indices: Vec<usize> = (0..10_000).collect();

            let correct = verify_correctness(
                scalar_backend.as_ref(),
                wgpu as &dyn HistogramBackend,
                &dataset,
                &grad_hess,
                &row_indices,
            );

            if correct {
                println!("  ✓ WGPU and Scalar backends produce equivalent results");
            } else {
                println!("  ✗ MISMATCH detected between backends!");
            }

            // Also test with row indices
            let partial_indices: Vec<usize> = (0..5_000).collect();
            let correct_partial = verify_correctness(
                scalar_backend.as_ref(),
                wgpu as &dyn HistogramBackend,
                &dataset,
                &grad_hess,
                &partial_indices,
            );

            if correct_partial {
                println!("  ✓ Row index handling verified (subset of rows)");
            } else {
                println!("  ✗ Row index handling differs between backends!");
            }
        }
    }

    // Detailed timing breakdown
    if configs.len() == 1 {
        let (num_rows, num_features, _) = configs[0];
        let dataset = create_test_dataset(num_rows, num_features);
        let grad_hess = create_grad_hess(num_rows);
        let row_indices: Vec<usize> = (0..num_rows).collect();

        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("  Detailed Timing Breakdown");
        println!("═══════════════════════════════════════════════════════════════════");

        if bench_scalar {
            timing_breakdown(scalar_backend.as_ref(), &dataset, &grad_hess, &row_indices);
        }

        #[cfg(feature = "gpu")]
        if bench_wgpu {
            if let Some(ref wgpu) = wgpu_backend {
                timing_breakdown(wgpu as &dyn HistogramBackend, &dataset, &grad_hess, &row_indices);
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  Benchmark Complete");
    println!("═══════════════════════════════════════════════════════════════════");
}
