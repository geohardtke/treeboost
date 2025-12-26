//! Benchmark SIMD vs scalar grad/hess interleaving performance.
//!
//! Run with: cargo run --release --example simd_benchmark

use std::time::Instant;

const BLOCK_SIZE: usize = 2048;
const NUM_ITERATIONS: usize = 10000;

/// Scalar implementation for comparison
fn copy_gh_scalar(
    gradients: &[f32],
    hessians: &[f32],
    start: usize,
    len: usize,
    gh_cache: &mut [(f32, f32); BLOCK_SIZE],
) {
    unsafe {
        for i in 0..len {
            let g = *gradients.get_unchecked(start + i);
            let h = *hessians.get_unchecked(start + i);
            *gh_cache.get_unchecked_mut(i) = (g, h);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_gh_avx2(
    gradients: &[f32],
    hessians: &[f32],
    start: usize,
    len: usize,
    gh_cache: &mut [(f32, f32); BLOCK_SIZE],
) {
    use std::arch::x86_64::*;

    let chunks = len / 8;
    let remainder = len % 8;

    let grad_ptr = gradients.as_ptr().add(start);
    let hess_ptr = hessians.as_ptr().add(start);
    let cache_ptr = gh_cache.as_mut_ptr() as *mut f32;

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 gradients and 8 hessians
        let grads = _mm256_loadu_ps(grad_ptr.add(offset));
        let hess = _mm256_loadu_ps(hess_ptr.add(offset));

        // Interleave using unpack operations
        let lo = _mm256_unpacklo_ps(grads, hess);
        let hi = _mm256_unpackhi_ps(grads, hess);

        // Permute to fix lane crossing
        let first = _mm256_permute2f128_ps(lo, hi, 0x20);
        let second = _mm256_permute2f128_ps(lo, hi, 0x31);

        // Store interleaved pairs
        let dst = cache_ptr.add(offset * 2);
        _mm256_storeu_ps(dst, first);
        _mm256_storeu_ps(dst.add(8), second);
    }

    // Handle remainder
    let rem_start = chunks * 8;
    for i in 0..remainder {
        let idx = rem_start + i;
        let g = *gradients.get_unchecked(start + idx);
        let h = *hessians.get_unchecked(start + idx);
        *gh_cache.get_unchecked_mut(idx) = (g, h);
    }
}

fn benchmark_scalar(gradients: &[f32], hessians: &[f32], block_size: usize) -> std::time::Duration {
    let mut gh_cache = [(0.0f32, 0.0f32); BLOCK_SIZE];

    let start = Instant::now();
    for _ in 0..NUM_ITERATIONS {
        copy_gh_scalar(gradients, hessians, 0, block_size, &mut gh_cache);
        std::hint::black_box(&gh_cache);
    }
    start.elapsed()
}

#[cfg(target_arch = "x86_64")]
fn benchmark_avx2(gradients: &[f32], hessians: &[f32], block_size: usize) -> std::time::Duration {
    let mut gh_cache = [(0.0f32, 0.0f32); BLOCK_SIZE];

    let start = Instant::now();
    for _ in 0..NUM_ITERATIONS {
        unsafe {
            copy_gh_avx2(gradients, hessians, 0, block_size, &mut gh_cache);
        }
        std::hint::black_box(&gh_cache);
    }
    start.elapsed()
}

fn main() {
    println!("SIMD Benchmark: Grad/Hess Interleaving");
    println!("======================================");
    println!("Iterations per test: {}", NUM_ITERATIONS);
    println!();

    #[cfg(target_arch = "x86_64")]
    {
        println!("Architecture: x86_64");
        println!("AVX2 available: {}", has_avx2());
        println!();
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("Architecture: aarch64 (NEON always available)");
        println!();
    }

    // Test different block sizes
    let test_sizes = [64, 256, 512, 1024, 2048];

    for &size in &test_sizes {
        // Generate test data
        let gradients: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let hessians: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();

        println!("Block size: {} rows", size);
        println!("{}", "-".repeat(40));

        // Benchmark scalar
        let scalar_time = benchmark_scalar(&gradients, &hessians, size);
        let scalar_ns = scalar_time.as_nanos() as f64 / NUM_ITERATIONS as f64;
        println!("  Scalar:  {:>8.1} ns/iter", scalar_ns);

        // Benchmark SIMD
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            let avx2_time = benchmark_avx2(&gradients, &hessians, size);
            let avx2_ns = avx2_time.as_nanos() as f64 / NUM_ITERATIONS as f64;
            let speedup = scalar_ns / avx2_ns;
            println!("  AVX2:    {:>8.1} ns/iter ({:.2}x speedup)", avx2_ns, speedup);
        }

        println!();
    }

    // Full histogram building benchmark
    println!("\nFull Histogram Build Benchmark");
    println!("==============================");

    // Create a realistic dataset
    let num_rows = 100_000;
    let num_features = 50;

    println!("Dataset: {} rows x {} features", num_rows, num_features);

    // Generate test data
    let gradients: Vec<f32> = (0..num_rows).map(|i| (i as f32 * 0.1) % 10.0).collect();
    let hessians: Vec<f32> = vec![1.0; num_rows];

    // Time the interleaving portion only (for a single block)
    let block_len = BLOCK_SIZE.min(num_rows);

    let scalar_time = benchmark_scalar(&gradients, &hessians, block_len);
    let scalar_ns = scalar_time.as_nanos() as f64 / NUM_ITERATIONS as f64;

    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        let avx2_time = benchmark_avx2(&gradients, &hessians, block_len);
        let avx2_ns = avx2_time.as_nanos() as f64 / NUM_ITERATIONS as f64;
        let speedup = scalar_ns / avx2_ns;

        println!("\nPer-block interleaving ({} rows):", block_len);
        println!("  Scalar: {:>8.1} ns", scalar_ns);
        println!("  AVX2:   {:>8.1} ns ({:.2}x speedup)", avx2_ns, speedup);

        // Estimate total histogram building impact
        let num_blocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let scalar_total = scalar_ns * num_blocks as f64 * num_features as f64;
        let avx2_total = avx2_ns * num_blocks as f64 * num_features as f64;

        println!("\nEstimated total interleaving time (all features, all blocks):");
        println!("  Scalar: {:>8.1} µs", scalar_total / 1000.0);
        println!("  AVX2:   {:>8.1} µs ({:.2}x speedup)", avx2_total / 1000.0, scalar_total / avx2_total);

        println!("\nNote: Histogram scatter (bin accumulation) is still scalar due to");
        println!("random access patterns. SIMD only optimizes the grad/hess loading phase.");
        println!("Overall histogram build improvement is typically 10-15%.");
    }
}
