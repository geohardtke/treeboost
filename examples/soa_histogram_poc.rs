//! PoC: Compare AoS vs SoA histogram layouts for SIMD efficiency
//!
//! Run: cargo run --release --example soa_histogram_poc

use std::time::Instant;

const NUM_BINS: usize = 256;
const NUM_FEATURES: usize = 50;
const NUM_ROWS: usize = 100_000;
const ITERATIONS: usize = 10;

// ============================================================================
// Current layout: Array of Structures (AoS)
// ============================================================================

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct BinEntryAoS {
    sum_gradients: f32,
    sum_hessians: f32,
    count: u32,
}

struct HistogramAoS {
    bins: [BinEntryAoS; NUM_BINS],
}

impl HistogramAoS {
    fn new() -> Self {
        Self {
            bins: [BinEntryAoS::default(); NUM_BINS],
        }
    }

    #[inline]
    fn accumulate(&mut self, bin: usize, grad: f32, hess: f32) {
        unsafe {
            let entry = self.bins.get_unchecked_mut(bin);
            entry.sum_gradients += grad;
            entry.sum_hessians += hess;
            entry.count += 1;
        }
    }
}

// ============================================================================
// New layout: Structure of Arrays (SoA)
// ============================================================================

struct HistogramSoA {
    gradients: [f32; NUM_BINS],
    hessians: [f32; NUM_BINS],
    counts: [u32; NUM_BINS],
}

impl HistogramSoA {
    fn new() -> Self {
        Self {
            gradients: [0.0; NUM_BINS],
            hessians: [0.0; NUM_BINS],
            counts: [0; NUM_BINS],
        }
    }

    #[inline]
    fn accumulate(&mut self, bin: usize, grad: f32, hess: f32) {
        unsafe {
            *self.gradients.get_unchecked_mut(bin) += grad;
            *self.hessians.get_unchecked_mut(bin) += hess;
            *self.counts.get_unchecked_mut(bin) += 1;
        }
    }
}

// ============================================================================
// Benchmark: Single feature histogram building
// ============================================================================

fn bench_aos_single_feature(
    bins: &[u8],
    gradients: &[f32],
    hessians: &[f32],
) -> HistogramAoS {
    let mut hist = HistogramAoS::new();

    // 8x unrolled like current implementation
    let chunks = bins.len() / 8;
    let remainder = bins.len() % 8;

    unsafe {
        for i in 0..chunks {
            let base = i * 8;
            let b0 = *bins.get_unchecked(base) as usize;
            let b1 = *bins.get_unchecked(base + 1) as usize;
            let b2 = *bins.get_unchecked(base + 2) as usize;
            let b3 = *bins.get_unchecked(base + 3) as usize;
            let b4 = *bins.get_unchecked(base + 4) as usize;
            let b5 = *bins.get_unchecked(base + 5) as usize;
            let b6 = *bins.get_unchecked(base + 6) as usize;
            let b7 = *bins.get_unchecked(base + 7) as usize;

            let g0 = *gradients.get_unchecked(base);
            let g1 = *gradients.get_unchecked(base + 1);
            let g2 = *gradients.get_unchecked(base + 2);
            let g3 = *gradients.get_unchecked(base + 3);
            let g4 = *gradients.get_unchecked(base + 4);
            let g5 = *gradients.get_unchecked(base + 5);
            let g6 = *gradients.get_unchecked(base + 6);
            let g7 = *gradients.get_unchecked(base + 7);

            let h0 = *hessians.get_unchecked(base);
            let h1 = *hessians.get_unchecked(base + 1);
            let h2 = *hessians.get_unchecked(base + 2);
            let h3 = *hessians.get_unchecked(base + 3);
            let h4 = *hessians.get_unchecked(base + 4);
            let h5 = *hessians.get_unchecked(base + 5);
            let h6 = *hessians.get_unchecked(base + 6);
            let h7 = *hessians.get_unchecked(base + 7);

            hist.accumulate(b0, g0, h0);
            hist.accumulate(b1, g1, h1);
            hist.accumulate(b2, g2, h2);
            hist.accumulate(b3, g3, h3);
            hist.accumulate(b4, g4, h4);
            hist.accumulate(b5, g5, h5);
            hist.accumulate(b6, g6, h6);
            hist.accumulate(b7, g7, h7);
        }

        let rem_base = chunks * 8;
        for i in 0..remainder {
            let bin = *bins.get_unchecked(rem_base + i) as usize;
            let grad = *gradients.get_unchecked(rem_base + i);
            let hess = *hessians.get_unchecked(rem_base + i);
            hist.accumulate(bin, grad, hess);
        }
    }

    hist
}

fn bench_soa_single_feature(
    bins: &[u8],
    gradients: &[f32],
    hessians: &[f32],
) -> HistogramSoA {
    let mut hist = HistogramSoA::new();

    // 8x unrolled
    let chunks = bins.len() / 8;
    let remainder = bins.len() % 8;

    unsafe {
        for i in 0..chunks {
            let base = i * 8;
            let b0 = *bins.get_unchecked(base) as usize;
            let b1 = *bins.get_unchecked(base + 1) as usize;
            let b2 = *bins.get_unchecked(base + 2) as usize;
            let b3 = *bins.get_unchecked(base + 3) as usize;
            let b4 = *bins.get_unchecked(base + 4) as usize;
            let b5 = *bins.get_unchecked(base + 5) as usize;
            let b6 = *bins.get_unchecked(base + 6) as usize;
            let b7 = *bins.get_unchecked(base + 7) as usize;

            let g0 = *gradients.get_unchecked(base);
            let g1 = *gradients.get_unchecked(base + 1);
            let g2 = *gradients.get_unchecked(base + 2);
            let g3 = *gradients.get_unchecked(base + 3);
            let g4 = *gradients.get_unchecked(base + 4);
            let g5 = *gradients.get_unchecked(base + 5);
            let g6 = *gradients.get_unchecked(base + 6);
            let g7 = *gradients.get_unchecked(base + 7);

            let h0 = *hessians.get_unchecked(base);
            let h1 = *hessians.get_unchecked(base + 1);
            let h2 = *hessians.get_unchecked(base + 2);
            let h3 = *hessians.get_unchecked(base + 3);
            let h4 = *hessians.get_unchecked(base + 4);
            let h5 = *hessians.get_unchecked(base + 5);
            let h6 = *hessians.get_unchecked(base + 6);
            let h7 = *hessians.get_unchecked(base + 7);

            hist.accumulate(b0, g0, h0);
            hist.accumulate(b1, g1, h1);
            hist.accumulate(b2, g2, h2);
            hist.accumulate(b3, g3, h3);
            hist.accumulate(b4, g4, h4);
            hist.accumulate(b5, g5, h5);
            hist.accumulate(b6, g6, h6);
            hist.accumulate(b7, g7, h7);
        }

        let rem_base = chunks * 8;
        for i in 0..remainder {
            let bin = *bins.get_unchecked(rem_base + i) as usize;
            let grad = *gradients.get_unchecked(rem_base + i);
            let hess = *hessians.get_unchecked(rem_base + i);
            hist.accumulate(bin, grad, hess);
        }
    }

    hist
}

// ============================================================================
// Benchmark: Multi-feature with reduction (like real histogram builder)
// ============================================================================

fn bench_aos_multi_feature(
    feature_bins: &[Vec<u8>],  // [feature][row]
    gradients: &[f32],
    hessians: &[f32],
) -> Vec<HistogramAoS> {
    let num_features = feature_bins.len();
    let mut histograms: Vec<HistogramAoS> = (0..num_features)
        .map(|_| HistogramAoS::new())
        .collect();

    let num_rows = gradients.len();
    let chunks = num_rows / 8;
    let remainder = num_rows % 8;

    // Process all features for each row block (cache-blocked approach)
    unsafe {
        for i in 0..chunks {
            let base = i * 8;

            // Load gradients/hessians once
            let g0 = *gradients.get_unchecked(base);
            let g1 = *gradients.get_unchecked(base + 1);
            let g2 = *gradients.get_unchecked(base + 2);
            let g3 = *gradients.get_unchecked(base + 3);
            let g4 = *gradients.get_unchecked(base + 4);
            let g5 = *gradients.get_unchecked(base + 5);
            let g6 = *gradients.get_unchecked(base + 6);
            let g7 = *gradients.get_unchecked(base + 7);

            let h0 = *hessians.get_unchecked(base);
            let h1 = *hessians.get_unchecked(base + 1);
            let h2 = *hessians.get_unchecked(base + 2);
            let h3 = *hessians.get_unchecked(base + 3);
            let h4 = *hessians.get_unchecked(base + 4);
            let h5 = *hessians.get_unchecked(base + 5);
            let h6 = *hessians.get_unchecked(base + 6);
            let h7 = *hessians.get_unchecked(base + 7);

            // Update all features
            for f in 0..num_features {
                let bins = &feature_bins[f];
                let hist = histograms.get_unchecked_mut(f);

                let b0 = *bins.get_unchecked(base) as usize;
                let b1 = *bins.get_unchecked(base + 1) as usize;
                let b2 = *bins.get_unchecked(base + 2) as usize;
                let b3 = *bins.get_unchecked(base + 3) as usize;
                let b4 = *bins.get_unchecked(base + 4) as usize;
                let b5 = *bins.get_unchecked(base + 5) as usize;
                let b6 = *bins.get_unchecked(base + 6) as usize;
                let b7 = *bins.get_unchecked(base + 7) as usize;

                hist.accumulate(b0, g0, h0);
                hist.accumulate(b1, g1, h1);
                hist.accumulate(b2, g2, h2);
                hist.accumulate(b3, g3, h3);
                hist.accumulate(b4, g4, h4);
                hist.accumulate(b5, g5, h5);
                hist.accumulate(b6, g6, h6);
                hist.accumulate(b7, g7, h7);
            }
        }

        // Remainder
        let rem_base = chunks * 8;
        for i in 0..remainder {
            let idx = rem_base + i;
            let g = *gradients.get_unchecked(idx);
            let h = *hessians.get_unchecked(idx);

            for f in 0..num_features {
                let bin = *feature_bins[f].get_unchecked(idx) as usize;
                histograms.get_unchecked_mut(f).accumulate(bin, g, h);
            }
        }
    }

    histograms
}

fn bench_soa_multi_feature(
    feature_bins: &[Vec<u8>],
    gradients: &[f32],
    hessians: &[f32],
) -> Vec<HistogramSoA> {
    let num_features = feature_bins.len();
    let mut histograms: Vec<HistogramSoA> = (0..num_features)
        .map(|_| HistogramSoA::new())
        .collect();

    let num_rows = gradients.len();
    let chunks = num_rows / 8;
    let remainder = num_rows % 8;

    unsafe {
        for i in 0..chunks {
            let base = i * 8;

            let g0 = *gradients.get_unchecked(base);
            let g1 = *gradients.get_unchecked(base + 1);
            let g2 = *gradients.get_unchecked(base + 2);
            let g3 = *gradients.get_unchecked(base + 3);
            let g4 = *gradients.get_unchecked(base + 4);
            let g5 = *gradients.get_unchecked(base + 5);
            let g6 = *gradients.get_unchecked(base + 6);
            let g7 = *gradients.get_unchecked(base + 7);

            let h0 = *hessians.get_unchecked(base);
            let h1 = *hessians.get_unchecked(base + 1);
            let h2 = *hessians.get_unchecked(base + 2);
            let h3 = *hessians.get_unchecked(base + 3);
            let h4 = *hessians.get_unchecked(base + 4);
            let h5 = *hessians.get_unchecked(base + 5);
            let h6 = *hessians.get_unchecked(base + 6);
            let h7 = *hessians.get_unchecked(base + 7);

            for f in 0..num_features {
                let bins = &feature_bins[f];
                let hist = histograms.get_unchecked_mut(f);

                let b0 = *bins.get_unchecked(base) as usize;
                let b1 = *bins.get_unchecked(base + 1) as usize;
                let b2 = *bins.get_unchecked(base + 2) as usize;
                let b3 = *bins.get_unchecked(base + 3) as usize;
                let b4 = *bins.get_unchecked(base + 4) as usize;
                let b5 = *bins.get_unchecked(base + 5) as usize;
                let b6 = *bins.get_unchecked(base + 6) as usize;
                let b7 = *bins.get_unchecked(base + 7) as usize;

                hist.accumulate(b0, g0, h0);
                hist.accumulate(b1, g1, h1);
                hist.accumulate(b2, g2, h2);
                hist.accumulate(b3, g3, h3);
                hist.accumulate(b4, g4, h4);
                hist.accumulate(b5, g5, h5);
                hist.accumulate(b6, g6, h6);
                hist.accumulate(b7, g7, h7);
            }
        }

        let rem_base = chunks * 8;
        for i in 0..remainder {
            let idx = rem_base + i;
            let g = *gradients.get_unchecked(idx);
            let h = *hessians.get_unchecked(idx);

            for f in 0..num_features {
                let bin = *feature_bins[f].get_unchecked(idx) as usize;
                histograms.get_unchecked_mut(f).accumulate(bin, g, h);
            }
        }
    }

    histograms
}

// ============================================================================
// Benchmark: SoA with SIMD merge (the real benefit)
// ============================================================================

fn bench_soa_merge_simd(hist1: &HistogramSoA, hist2: &HistogramSoA) -> HistogramSoA {
    let mut result = HistogramSoA::new();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                // Merge gradients: 8 floats at a time
                for i in (0..NUM_BINS).step_by(8) {
                    let a = _mm256_loadu_ps(hist1.gradients.as_ptr().add(i));
                    let b = _mm256_loadu_ps(hist2.gradients.as_ptr().add(i));
                    let sum = _mm256_add_ps(a, b);
                    _mm256_storeu_ps(result.gradients.as_mut_ptr().add(i), sum);
                }

                // Merge hessians
                for i in (0..NUM_BINS).step_by(8) {
                    let a = _mm256_loadu_ps(hist1.hessians.as_ptr().add(i));
                    let b = _mm256_loadu_ps(hist2.hessians.as_ptr().add(i));
                    let sum = _mm256_add_ps(a, b);
                    _mm256_storeu_ps(result.hessians.as_mut_ptr().add(i), sum);
                }

                // Merge counts: 8 u32s at a time
                for i in (0..NUM_BINS).step_by(8) {
                    let a = _mm256_loadu_si256(hist1.counts.as_ptr().add(i) as *const __m256i);
                    let b = _mm256_loadu_si256(hist2.counts.as_ptr().add(i) as *const __m256i);
                    let sum = _mm256_add_epi32(a, b);
                    _mm256_storeu_si256(result.counts.as_mut_ptr().add(i) as *mut __m256i, sum);
                }

                return result;
            }
        }
    }

    // Scalar fallback
    for i in 0..NUM_BINS {
        result.gradients[i] = hist1.gradients[i] + hist2.gradients[i];
        result.hessians[i] = hist1.hessians[i] + hist2.hessians[i];
        result.counts[i] = hist1.counts[i] + hist2.counts[i];
    }
    result
}

fn bench_aos_merge(hist1: &HistogramAoS, hist2: &HistogramAoS) -> HistogramAoS {
    let mut result = HistogramAoS::new();
    for i in 0..NUM_BINS {
        result.bins[i].sum_gradients = hist1.bins[i].sum_gradients + hist2.bins[i].sum_gradients;
        result.bins[i].sum_hessians = hist1.bins[i].sum_hessians + hist2.bins[i].sum_hessians;
        result.bins[i].count = hist1.bins[i].count + hist2.bins[i].count;
    }
    result
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                    SoA vs AoS HISTOGRAM LAYOUT BENCHMARK");
    println!("═══════════════════════════════════════════════════════════════════════════════\n");

    // Generate test data
    let bins: Vec<u8> = (0..NUM_ROWS).map(|i| ((i * 17) % 256) as u8).collect();
    let gradients: Vec<f32> = (0..NUM_ROWS).map(|i| (i as f32 * 0.001).sin()).collect();
    let hessians: Vec<f32> = vec![1.0; NUM_ROWS];

    // Multi-feature data
    let feature_bins: Vec<Vec<u8>> = (0..NUM_FEATURES)
        .map(|f| (0..NUM_ROWS).map(|r| (((r + f) * 17) % 256) as u8).collect())
        .collect();

    println!("Test configuration:");
    println!("  Rows: {}", NUM_ROWS);
    println!("  Features: {}", NUM_FEATURES);
    println!("  Iterations: {}\n", ITERATIONS);

    // =========================================================================
    // Test 1: Single feature histogram building
    // =========================================================================
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!("TEST 1: Single feature histogram building");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    // Warmup
    let _ = bench_aos_single_feature(&bins, &gradients, &hessians);
    let _ = bench_soa_single_feature(&bins, &gradients, &hessians);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(bench_aos_single_feature(&bins, &gradients, &hessians));
    }
    let aos_time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(bench_soa_single_feature(&bins, &gradients, &hessians));
    }
    let soa_time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;

    println!("  AoS (current):     {:>8.3} ms", aos_time);
    println!("  SoA (proposed):    {:>8.3} ms", soa_time);
    println!("  Speedup:           {:>8.2}x", aos_time / soa_time);

    // =========================================================================
    // Test 2: Multi-feature histogram building (like real builder)
    // =========================================================================
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("TEST 2: Multi-feature histogram building ({} features)", NUM_FEATURES);
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    // Warmup
    let _ = bench_aos_multi_feature(&feature_bins, &gradients, &hessians);
    let _ = bench_soa_multi_feature(&feature_bins, &gradients, &hessians);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(bench_aos_multi_feature(&feature_bins, &gradients, &hessians));
    }
    let aos_multi_time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(bench_soa_multi_feature(&feature_bins, &gradients, &hessians));
    }
    let soa_multi_time = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;

    println!("  AoS (current):     {:>8.3} ms", aos_multi_time);
    println!("  SoA (proposed):    {:>8.3} ms", soa_multi_time);
    println!("  Speedup:           {:>8.2}x", aos_multi_time / soa_multi_time);

    // =========================================================================
    // Test 3: Histogram merge (where SoA really shines with SIMD)
    // =========================================================================
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("TEST 3: Histogram merge (SIMD benefit)");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    let aos_hist1 = bench_aos_single_feature(&bins, &gradients, &hessians);
    let aos_hist2 = bench_aos_single_feature(&bins, &gradients, &hessians);
    let soa_hist1 = bench_soa_single_feature(&bins, &gradients, &hessians);
    let soa_hist2 = bench_soa_single_feature(&bins, &gradients, &hessians);

    let merge_iters = ITERATIONS * 1000;

    let start = Instant::now();
    for _ in 0..merge_iters {
        std::hint::black_box(bench_aos_merge(&aos_hist1, &aos_hist2));
    }
    let aos_merge_time = start.elapsed().as_secs_f64() * 1_000_000.0 / merge_iters as f64;

    let start = Instant::now();
    for _ in 0..merge_iters {
        std::hint::black_box(bench_soa_merge_simd(&soa_hist1, &soa_hist2));
    }
    let soa_merge_time = start.elapsed().as_secs_f64() * 1_000_000.0 / merge_iters as f64;

    println!("  AoS merge:         {:>8.3} µs", aos_merge_time);
    println!("  SoA SIMD merge:    {:>8.3} µs", soa_merge_time);
    println!("  Speedup:           {:>8.2}x", aos_merge_time / soa_merge_time);

    // =========================================================================
    // Memory layout comparison
    // =========================================================================
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("MEMORY LAYOUT");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    let aos_size = std::mem::size_of::<HistogramAoS>();
    let soa_size = std::mem::size_of::<HistogramSoA>();

    println!("  AoS histogram size:  {} bytes ({} KB)", aos_size, aos_size / 1024);
    println!("  SoA histogram size:  {} bytes ({} KB)", soa_size, soa_size / 1024);
    println!("  BinEntry size:       {} bytes", std::mem::size_of::<BinEntryAoS>());

    println!("\n  AoS layout: [g0,h0,c0, g1,h1,c1, g2,h2,c2, ...]");
    println!("  SoA layout: [g0,g1,g2,...], [h0,h1,h2,...], [c0,c1,c2,...]");

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n───────────────────────────────────────────────────────────────────────────────");
    println!("SUMMARY");
    println!("───────────────────────────────────────────────────────────────────────────────\n");

    let build_benefit = aos_multi_time / soa_multi_time;
    let merge_benefit = aos_merge_time / soa_merge_time;

    if build_benefit > 1.05 {
        println!("  ✓ SoA is {:.1}% FASTER for histogram building", (build_benefit - 1.0) * 100.0);
    } else if build_benefit < 0.95 {
        println!("  ✗ SoA is {:.1}% SLOWER for histogram building", (1.0 - build_benefit) * 100.0);
    } else {
        println!("  ≈ SoA is SIMILAR for histogram building (within 5%)");
    }

    if merge_benefit > 1.05 {
        println!("  ✓ SoA is {:.1}% FASTER for histogram merge (SIMD)", (merge_benefit - 1.0) * 100.0);
    } else {
        println!("  ≈ SoA merge benefit is minimal");
    }

    println!();
}
