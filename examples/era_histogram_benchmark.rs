//! Benchmark comparing CPU vs GPU era histogram performance.
//!
//! Usage:
//!   cargo run --release --example era_histogram_benchmark
//!   cargo run --release --features gpu --example era_histogram_benchmark
//!   cargo run --release --features cuda --example era_histogram_benchmark
//!   cargo run --release --features gpu,cuda --example era_histogram_benchmark

use std::time::{Duration, Instant};
use treeboost::backend::{HistogramBackend, ScalarBackend};

#[cfg(feature = "gpu")]
use treeboost::backend::WgpuBackend;

#[cfg(feature = "cuda")]
use treeboost::backend::CudaBackend;

fn generate_test_data(
    num_rows: usize,
    num_features: usize,
    num_eras: usize,
) -> (Vec<u8>, Vec<(f32, f32)>, Vec<usize>, Vec<u16>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let bins: Vec<u8> = (0..num_rows * num_features)
        .map(|_| rng.gen_range(0..64))
        .collect();

    let grad_hess: Vec<(f32, f32)> = (0..num_rows)
        .map(|_| (rng.gen_range(-1.0..1.0), rng.gen_range(0.5..2.0)))
        .collect();

    let row_indices: Vec<usize> = (0..num_rows).collect();

    let era_indices: Vec<u16> = (0..num_rows)
        .map(|i| (i % num_eras) as u16)
        .collect();

    (bins, grad_hess, row_indices, era_indices)
}

struct RowMajorBins {
    bins: Vec<u8>,
    num_rows: usize,
    num_features: usize,
}

impl treeboost::backend::BinStorage for RowMajorBins {
    fn get_bin(&self, row: usize, feature: usize) -> u8 {
        self.bins[row * self.num_features + feature]
    }
    fn num_rows(&self) -> usize { self.num_rows }
    fn num_features(&self) -> usize { self.num_features }
    fn feature_column(&self, _feature: usize) -> Option<&[u8]> { None }
    fn sparse_column(&self, _feature: usize) -> Option<&treeboost::backend::SparseColumn> { None }
    fn as_row_major(&self) -> Option<&[u8]> { Some(&self.bins) }
    fn max_bins(&self) -> u8 { 64 }
}

fn benchmark_cpu(
    bins: &RowMajorBins,
    grad_hess: &[(f32, f32)],
    row_indices: &[usize],
    era_indices: &[u16],
    num_eras: usize,
    warmup: usize,
    iters: usize,
) -> Duration {
    let backend = ScalarBackend::new();
    for _ in 0..warmup {
        let _ = backend.build_era_histograms(bins, grad_hess, row_indices, era_indices, num_eras);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = backend.build_era_histograms(bins, grad_hess, row_indices, era_indices, num_eras);
    }
    start.elapsed() / iters as u32
}

#[cfg(feature = "gpu")]
fn benchmark_wgpu(
    bins: &RowMajorBins,
    grad_hess: &[(f32, f32)],
    row_indices: &[usize],
    era_indices: &[u16],
    num_eras: usize,
    warmup: usize,
    iters: usize,
) -> Option<Duration> {
    let backend = WgpuBackend::new()?;
    for _ in 0..warmup {
        let _ = backend.build_era_histograms(bins, grad_hess, row_indices, era_indices, num_eras);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = backend.build_era_histograms(bins, grad_hess, row_indices, era_indices, num_eras);
    }
    Some(start.elapsed() / iters as u32)
}

#[cfg(feature = "cuda")]
fn benchmark_cuda(
    bins: &RowMajorBins,
    grad_hess: &[(f32, f32)],
    row_indices: &[usize],
    era_indices: &[u16],
    num_eras: usize,
    warmup: usize,
    iters: usize,
) -> Option<Duration> {
    let backend = CudaBackend::new()?;
    for _ in 0..warmup {
        let _ = backend.build_era_histograms(bins, grad_hess, row_indices, era_indices, num_eras);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = backend.build_era_histograms(bins, grad_hess, row_indices, era_indices, num_eras);
    }
    Some(start.elapsed() / iters as u32)
}

fn fmt(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1000 { format!("{:>7} µs", us) }
    else { format!("{:>7.2} ms", us as f64 / 1000.0) }
}

fn main() {
    println!("Era Histogram Benchmark");
    println!("=======================\n");

    let configs = [
        (10_000, 50, 5),
        (50_000, 50, 5),
        (100_000, 50, 5),
        (100_000, 100, 5),
        (100_000, 50, 10),
        (500_000, 50, 5),
        (1_000_000, 50, 5),
    ];

    // Header
    print!("{:>10} {:>6} {:>5} {:>12}", "Rows", "Feats", "Eras", "CPU");
    #[cfg(feature = "gpu")]
    print!(" {:>12}", "WGPU");
    #[cfg(feature = "cuda")]
    print!(" {:>12}", "CUDA");
    println!();

    let len = 38
        + if cfg!(feature = "gpu") { 13 } else { 0 }
        + if cfg!(feature = "cuda") { 13 } else { 0 };
    println!("{}", "-".repeat(len));

    for (rows, feats, eras) in configs {
        let (bins_data, grad_hess, row_indices, era_indices) = generate_test_data(rows, feats, eras);
        let bins = RowMajorBins { bins: bins_data, num_rows: rows, num_features: feats };

        let cpu = benchmark_cpu(&bins, &grad_hess, &row_indices, &era_indices, eras, 3, 10);

        #[cfg(feature = "gpu")]
        let wgpu = benchmark_wgpu(&bins, &grad_hess, &row_indices, &era_indices, eras, 3, 10);

        #[cfg(feature = "cuda")]
        let cuda = benchmark_cuda(&bins, &grad_hess, &row_indices, &era_indices, eras, 3, 10);

        print!("{:>10} {:>6} {:>5} {:>12}", rows, feats, eras, fmt(cpu));

        #[cfg(feature = "gpu")]
        print!(" {:>12}", wgpu.map(fmt).unwrap_or_else(|| "N/A".into()));

        #[cfg(feature = "cuda")]
        print!(" {:>12}", cuda.map(fmt).unwrap_or_else(|| "N/A".into()));

        println!();
    }
}
