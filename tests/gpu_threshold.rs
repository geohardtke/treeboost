//! GPU vs CPU threshold benchmark
//!
//! This test measures the actual crossover point where GPU becomes faster than CPU
//! for histogram building. Run with:
//!
//! ```bash
//! cargo test --release --features cuda test_gpu_threshold -- --nocapture --ignored
//! ```

#[cfg(all(test, feature = "cuda"))]
mod threshold_tests {
    use std::time::Instant;
    use treeboost::backend::{CudaBackend, HistogramBackend, ScalarBackend};
    use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

    /// Create test data for histogram building
    fn create_test_data(num_rows: usize, num_features: usize) -> (BinnedDataset, Vec<(f32, f32)>) {
        // Generate random binned features (u8 values)
        let features: Vec<u8> = (0..num_rows * num_features)
            .map(|i| ((i as u64 * 17) % 256) as u8)
            .collect();

        // Generate random targets
        let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32 * 0.456) % 10.0).collect();

        // Create feature info (all numeric features)
        let feature_info: Vec<FeatureInfo> = (0..num_features)
            .map(|_| FeatureInfo {
                name: String::from("feat"),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
                impute_value: 0.0,
            })
            .collect();

        // Create binned dataset
        let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);

        // Generate grad/hess
        let grad_hess: Vec<(f32, f32)> = (0..num_rows)
            .map(|i| ((i as f32 * 0.456) % 1.0, 1.0))
            .collect();

        (dataset, grad_hess)
    }

    /// Measure histogram build time for a given backend and batch size
    fn measure_histogram_time(
        backend: &dyn HistogramBackend,
        dataset: &BinnedDataset,
        grad_hess: &[(f32, f32)],
        batch_size: usize,
        iterations: usize,
    ) -> f64 {
        let row_indices: Vec<usize> = (0..batch_size).collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = backend.build_histograms(dataset, grad_hess, &row_indices);
        }
        let elapsed = start.elapsed();

        elapsed.as_secs_f64() / iterations as f64
    }

    #[test]
    #[ignore] // Run explicitly with --ignored flag
    fn test_gpu_threshold() {
        println!("\n=== GPU vs CPU Threshold Benchmark ===\n");

        // Try to create backends
        let cuda_backend = CudaBackend::new();
        let scalar_backend = ScalarBackend::new();

        if cuda_backend.is_none() {
            println!("⚠️  CUDA not available - skipping benchmark");
            return;
        }

        let cuda_backend = cuda_backend.unwrap();
        println!("✓ CUDA backend initialized");
        println!("✓ Scalar backend initialized\n");

        // Test parameters
        let num_features = 100;
        let iterations = 20; // Average over multiple runs

        // Test different batch sizes
        let batch_sizes = vec![
            100, 200, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000,
        ];

        println!(
            "Testing with {} features, {} iterations per size\n",
            num_features, iterations
        );
        println!(
            "{:>10} {:>15} {:>15} {:>12}",
            "Batch Size", "CPU Time (ms)", "GPU Time (ms)", "Speedup"
        );
        println!("{}", "-".repeat(55));

        let mut crossover_point = None;

        for &batch_size in &batch_sizes {
            // Create test data for this batch size
            let (dataset, grad_hess) = create_test_data(batch_size * 2, num_features);

            // Measure CPU time
            let cpu_time = measure_histogram_time(
                &scalar_backend,
                &dataset,
                &grad_hess,
                batch_size,
                iterations,
            );

            // Measure GPU time
            let gpu_time =
                measure_histogram_time(&cuda_backend, &dataset, &grad_hess, batch_size, iterations);

            let speedup = cpu_time / gpu_time;

            println!(
                "{:>10} {:>15.3} {:>15.3} {:>12.2}x",
                batch_size,
                cpu_time * 1000.0,
                gpu_time * 1000.0,
                speedup
            );

            // Detect crossover point (where GPU becomes faster)
            if crossover_point.is_none() && speedup > 1.0 {
                crossover_point = Some(batch_size);
            }
        }

        println!("\n{}", "=".repeat(55));
        if let Some(threshold) = crossover_point {
            println!("✓ GPU becomes faster at ~{} rows", threshold);
            println!("\nRecommended CUDA_BATCH_THRESHOLD: {}", threshold);
        } else {
            println!("⚠️  GPU never became faster than CPU in tested range");
            println!("   Consider using CPU-only backend for this hardware");
        }
        println!("\nNote: These thresholds are hardware-specific.");
        println!("      Adjust based on your GPU model and workload.");
    }
}

#[cfg(all(test, feature = "gpu"))]
mod wgpu_threshold_tests {
    use std::time::Instant;
    use treeboost::backend::{HistogramBackend, ScalarBackend, WgpuBackend};
    use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

    /// Create test data for histogram building
    fn create_test_data(num_rows: usize, num_features: usize) -> (BinnedDataset, Vec<(f32, f32)>) {
        // Generate random binned features (u8 values)
        let features: Vec<u8> = (0..num_rows * num_features)
            .map(|i| ((i as u64 * 17) % 256) as u8)
            .collect();

        // Generate random targets
        let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32 * 0.456) % 10.0).collect();

        // Create feature info (all numeric features)
        let feature_info: Vec<FeatureInfo> = (0..num_features)
            .map(|_| FeatureInfo {
                name: String::from("feat"),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
                impute_value: 0.0,
            })
            .collect();

        // Create binned dataset
        let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);

        // Generate grad/hess
        let grad_hess: Vec<(f32, f32)> = (0..num_rows)
            .map(|i| ((i as f32 * 0.456) % 1.0, 1.0))
            .collect();

        (dataset, grad_hess)
    }

    /// Measure histogram build time for a given backend and batch size
    fn measure_histogram_time(
        backend: &dyn HistogramBackend,
        dataset: &BinnedDataset,
        grad_hess: &[(f32, f32)],
        batch_size: usize,
        iterations: usize,
    ) -> f64 {
        let row_indices: Vec<usize> = (0..batch_size).collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = backend.build_histograms(dataset, grad_hess, &row_indices);
        }
        let elapsed = start.elapsed();

        elapsed.as_secs_f64() / iterations as f64
    }

    #[test]
    #[ignore]
    fn test_wgpu_threshold() {
        println!("\n=== WGPU vs CPU Threshold Benchmark ===\n");

        // Try to create backends
        let wgpu_backend = WgpuBackend::new();
        let scalar_backend = ScalarBackend::new();

        if wgpu_backend.is_none() {
            println!("⚠️  WGPU not available - skipping benchmark");
            return;
        }

        let wgpu_backend = wgpu_backend.unwrap();
        println!("✓ WGPU backend initialized");
        println!("✓ Scalar backend initialized\n");

        // Test parameters
        let num_features = 100;
        let iterations = 20;

        // Test larger batch sizes for WGPU (higher overhead)
        let batch_sizes = vec![500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000];

        println!(
            "Testing with {} features, {} iterations per size\n",
            num_features, iterations
        );
        println!(
            "{:>10} {:>15} {:>15} {:>12}",
            "Batch Size", "CPU Time (ms)", "GPU Time (ms)", "Speedup"
        );
        println!("{}", "-".repeat(55));

        let mut crossover_point = None;

        for &batch_size in &batch_sizes {
            let (dataset, grad_hess) = create_test_data(batch_size * 2, num_features);

            let cpu_time = measure_histogram_time(
                &scalar_backend,
                &dataset,
                &grad_hess,
                batch_size,
                iterations,
            );

            let gpu_time =
                measure_histogram_time(&wgpu_backend, &dataset, &grad_hess, batch_size, iterations);

            let speedup = cpu_time / gpu_time;

            println!(
                "{:>10} {:>15.3} {:>15.3} {:>12.2}x",
                batch_size,
                cpu_time * 1000.0,
                gpu_time * 1000.0,
                speedup
            );

            if crossover_point.is_none() && speedup > 1.0 {
                crossover_point = Some(batch_size);
            }
        }

        println!("\n{}", "=".repeat(55));
        if let Some(threshold) = crossover_point {
            println!("✓ GPU becomes faster at ~{} rows", threshold);
            println!("\nRecommended WGPU_BATCH_THRESHOLD: {}", threshold);
        } else {
            println!("⚠️  GPU never became faster than CPU in tested range");
        }
        println!("\nNote: WGPU typically has 10-20x higher overhead than CUDA.");
    }
}
