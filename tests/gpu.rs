//! Integration tests for GPU backend
//!
//! These tests are conditional on the `gpu` feature flag

#[cfg(feature = "gpu")]
mod common;

#[cfg(feature = "gpu")]
use common::create_synthetic_dataset;

#[cfg(feature = "gpu")]
use treeboost::backend::{BackendConfig, BackendSelector, BackendType, SplitConfig};
#[cfg(feature = "gpu")]
use treeboost::booster::{GBDTConfig, GBDTModel};

/// Test that GPU backend produces correct histograms matching scalar backend
#[cfg(feature = "gpu")]
#[test]
fn test_gpu_backend_histogram_correctness() {
    // Create dataset large enough to benefit from GPU
    let dataset = create_synthetic_dataset(10000, 42);

    // Create gradient/hessian pairs
    let targets = dataset.targets();
    let grad_hess: Vec<(f32, f32)> = targets
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            // Simple gradient: prediction - target (assume pred=0)
            let grad = -t;
            let hess = 1.0 + (i % 3) as f32 * 0.1; // Varied hessian
            (grad, hess)
        })
        .collect();

    let row_indices: Vec<usize> = (0..dataset.num_rows()).collect();

    // Get scalar backend histograms
    let scalar_config = BackendConfig {
        preferred: BackendType::Scalar,
        ..Default::default()
    };
    let scalar_backend = BackendSelector::with_config(scalar_config).select(dataset.num_rows());
    let scalar_hists = scalar_backend.build_histograms(&dataset, &grad_hess, &row_indices);

    // Get GPU backend histograms
    let gpu_config = BackendConfig {
        preferred: BackendType::Wgpu,
        ..Default::default()
    };
    let gpu_selector = BackendSelector::with_config(gpu_config);
    let gpu_backend = gpu_selector.select(dataset.num_rows());

    // Skip test if GPU not available
    if gpu_backend.name() != "WGPU" {
        println!("GPU not available, skipping GPU histogram correctness test");
        return;
    }

    let gpu_hists = gpu_backend.build_histograms(&dataset, &grad_hess, &row_indices);

    // Compare histograms
    assert_eq!(
        scalar_hists.len(),
        gpu_hists.len(),
        "Histogram count mismatch"
    );

    for (f, (scalar_hist, gpu_hist)) in scalar_hists.iter().zip(gpu_hists.iter()).enumerate() {
        let (scalar_grad, scalar_hess, scalar_count) = scalar_hist.totals();
        let (gpu_grad, gpu_hess, gpu_count) = gpu_hist.totals();

        assert_eq!(
            scalar_count, gpu_count,
            "Feature {}: count mismatch ({} vs {})",
            f, scalar_count, gpu_count
        );

        // Allow small floating point tolerance for GPU atomics
        let grad_diff = (scalar_grad - gpu_grad).abs();
        let hess_diff = (scalar_hess - gpu_hess).abs();
        let grad_tol = scalar_grad.abs() * 0.01 + 0.1;
        let hess_tol = scalar_hess.abs() * 0.01 + 0.1;

        assert!(
            grad_diff < grad_tol,
            "Feature {}: gradient mismatch (scalar={}, gpu={}, diff={})",
            f,
            scalar_grad,
            gpu_grad,
            grad_diff
        );
        assert!(
            hess_diff < hess_tol,
            "Feature {}: hessian mismatch (scalar={}, gpu={}, diff={})",
            f,
            scalar_hess,
            gpu_hess,
            hess_diff
        );
    }

    println!(
        "GPU histogram correctness verified for {} features × {} rows",
        dataset.num_features(),
        dataset.num_rows()
    );
}

/// Test GPU backend with row subsampling
#[cfg(feature = "gpu")]
#[test]
fn test_gpu_backend_with_subsampling() {
    let dataset = create_synthetic_dataset(5000, 123);

    let grad_hess: Vec<(f32, f32)> = dataset.targets().iter().map(|&t| (-t, 1.0)).collect();

    // Use only even-indexed rows (50% subsample)
    let row_indices: Vec<usize> = (0..dataset.num_rows()).filter(|i| i % 2 == 0).collect();

    let gpu_config = BackendConfig {
        preferred: BackendType::Wgpu,
        ..Default::default()
    };
    let gpu_backend = BackendSelector::with_config(gpu_config).select(dataset.num_rows());

    if gpu_backend.name() != "WGPU" {
        println!("GPU not available, skipping GPU subsampling test");
        return;
    }

    let hists = gpu_backend.build_histograms(&dataset, &grad_hess, &row_indices);

    // Verify counts sum to subsampled size
    let total_count: u32 = hists[0].totals().2;
    assert_eq!(
        total_count as usize,
        row_indices.len(),
        "Total count should match subsampled row count"
    );

    println!(
        "GPU subsampling verified: {} rows from {} total",
        row_indices.len(),
        dataset.num_rows()
    );
}

/// Test GPU backend split finding matches scalar backend
#[cfg(feature = "gpu")]
#[test]
fn test_gpu_backend_split_finding() {
    let dataset = create_synthetic_dataset(10000, 456);

    let grad_hess: Vec<(f32, f32)> = dataset.targets().iter().map(|&t| (-t, 1.0)).collect();

    let row_indices: Vec<usize> = (0..dataset.num_rows()).collect();

    let split_config = SplitConfig {
        lambda: 1.0,
        min_samples_leaf: 10,
        min_hessian_leaf: 0.0,
        min_gain: 0.0,
        entropy_weight: 0.0,
    };

    // Scalar backend
    let scalar_backend = BackendSelector::with_config(BackendConfig {
        preferred: BackendType::Scalar,
        ..Default::default()
    })
    .select(dataset.num_rows());
    let scalar_hists = scalar_backend.build_histograms(&dataset, &grad_hess, &row_indices);
    let scalar_split = scalar_backend.find_best_split(&scalar_hists, &split_config);

    // GPU backend
    let gpu_backend = BackendSelector::with_config(BackendConfig {
        preferred: BackendType::Wgpu,
        ..Default::default()
    })
    .select(dataset.num_rows());

    if gpu_backend.name() != "WGPU" {
        println!("GPU not available, skipping GPU split finding test");
        return;
    }

    let gpu_hists = gpu_backend.build_histograms(&dataset, &grad_hess, &row_indices);
    let gpu_split = gpu_backend.find_best_split(&gpu_hists, &split_config);

    // Both should find a split
    assert!(scalar_split.is_some(), "Scalar should find a split");
    assert!(gpu_split.is_some(), "GPU should find a split");

    let scalar_split = scalar_split.unwrap();
    let gpu_split = gpu_split.unwrap();

    // Same feature and threshold (or very close gain)
    // Note: Due to floating point differences, the exact split may differ slightly
    let gain_diff = (scalar_split.gain - gpu_split.gain).abs();
    let gain_tol = scalar_split.gain.abs() * 0.05 + 0.1;

    assert!(
        gain_diff < gain_tol,
        "Split gain mismatch: scalar={} (feature={}, threshold={}), gpu={} (feature={}, threshold={})",
        scalar_split.gain,
        scalar_split.feature,
        scalar_split.threshold,
        gpu_split.gain,
        gpu_split.feature,
        gpu_split.threshold
    );

    println!(
        "GPU split finding verified: feature={}, threshold={}, gain={:.4}",
        gpu_split.feature, gpu_split.threshold, gpu_split.gain
    );
}

/// Test full GBDT training with GPU backend
#[cfg(feature = "gpu")]
#[test]
fn test_gpu_training_end_to_end() {
    // Create a larger dataset to benefit from GPU
    let dataset = create_synthetic_dataset(50000, 789);

    // Train with GPU backend
    let gpu_config = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(5)
        .with_learning_rate(0.1)
        .with_backend(BackendType::Wgpu);

    let gpu_model = GBDTModel::train_binned(&dataset, gpu_config).expect("GPU training failed");

    // Train with scalar backend for comparison
    let scalar_config = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(5)
        .with_learning_rate(0.1)
        .with_backend(BackendType::Scalar);

    let scalar_model =
        GBDTModel::train_binned(&dataset, scalar_config).expect("Scalar training failed");

    // Both models should produce similar predictions
    let gpu_preds = gpu_model.predict(&dataset);
    let scalar_preds = scalar_model.predict(&dataset);

    assert_eq!(gpu_preds.len(), scalar_preds.len());

    // Compute RMSE between predictions
    let mse: f32 = gpu_preds
        .iter()
        .zip(scalar_preds.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / gpu_preds.len() as f32;
    let rmse = mse.sqrt();

    // Predictions should be very similar (small numerical differences due to floating point)
    // Using a tolerance of 1% of the target range
    let target_range = dataset.targets().iter().cloned().fold(f32::MIN, f32::max)
        - dataset.targets().iter().cloned().fold(f32::MAX, f32::min);
    let tolerance = target_range * 0.01;

    println!(
        "GPU vs Scalar RMSE: {:.6}, tolerance: {:.6}",
        rmse, tolerance
    );
    assert!(
        rmse < tolerance,
        "GPU and Scalar predictions differ too much: RMSE={:.4} > tolerance={:.4}",
        rmse,
        tolerance
    );

    // Verify model quality (predictions should correlate with targets)
    let targets = dataset.targets();
    let gpu_mse: f32 = gpu_preds
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / gpu_preds.len() as f32;

    println!("GPU model MSE: {:.4}", gpu_mse);
    assert!(gpu_mse < 1.0, "GPU model MSE too high: {}", gpu_mse);
}

// =============================================================================
// CUDA-specific tests
// =============================================================================

/// Test CUDA backend initialization
#[cfg(feature = "cuda")]
#[test]
fn test_cuda_backend_initialization() {
    use treeboost::backend::cuda::CudaBackend;

    eprintln!("\n=== Testing CUDA Backend Initialization ===");

    match CudaBackend::new() {
        Some(backend) => {
            eprintln!("✓ CUDA backend initialized successfully");
            eprintln!("  Device: {}", backend.device().name());
        }
        None => {
            panic!("CUDA backend initialization failed! Check CUDA installation and driver.");
        }
    }
}

/// Test that auto-detection selects CUDA when available
#[cfg(feature = "cuda")]
#[test]
fn test_cuda_auto_detection() {
    let dataset = create_synthetic_dataset(15000, 42); // Must be > TENSOR_TILE_MIN_ROWS (10k)

    eprintln!("\n=== Testing CUDA Auto-Detection ===");

    // Use Auto backend type - should select CUDA
    let config = BackendConfig {
        preferred: BackendType::Auto,
        ..Default::default()
    };
    let backend = BackendSelector::with_config(config).select(dataset.num_rows());

    eprintln!("  Selected backend: {}", backend.name());

    // Should select CUDA if available
    assert_eq!(
        backend.name(),
        "CUDA",
        "Auto-detection should select CUDA backend when cuda feature is enabled"
    );
}

/// Test CUDA backend histogram correctness vs scalar
#[cfg(feature = "cuda")]
#[test]
fn test_cuda_backend_histogram_correctness() {
    let dataset = create_synthetic_dataset(1000, 42); // Small for fast test

    let targets = dataset.targets();
    let grad_hess: Vec<(f32, f32)> = targets
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let grad = -t;
            let hess = 1.0 + (i % 3) as f32 * 0.1;
            (grad, hess)
        })
        .collect();

    let row_indices: Vec<usize> = (0..dataset.num_rows()).collect();

    eprintln!("\n=== Testing CUDA Histogram Correctness ===");

    // Get scalar backend histograms
    let scalar_config = BackendConfig {
        preferred: BackendType::Scalar,
        ..Default::default()
    };
    let scalar_backend = BackendSelector::with_config(scalar_config).select(dataset.num_rows());
    let scalar_hists = scalar_backend.build_histograms(&dataset, &grad_hess, &row_indices);

    // Get CUDA backend histograms
    let cuda_config = BackendConfig {
        preferred: BackendType::Cuda,
        ..Default::default()
    };
    let cuda_backend = BackendSelector::with_config(cuda_config).select(dataset.num_rows());

    assert_eq!(cuda_backend.name(), "CUDA", "Should use CUDA backend");

    let cuda_hists = cuda_backend.build_histograms(&dataset, &grad_hess, &row_indices);

    // Compare histograms
    assert_eq!(
        scalar_hists.len(),
        cuda_hists.len(),
        "Histogram count mismatch"
    );

    for (f, (scalar_hist, cuda_hist)) in scalar_hists.iter().zip(cuda_hists.iter()).enumerate() {
        let (scalar_grad, scalar_hess, scalar_count) = scalar_hist.totals();
        let (cuda_grad, cuda_hess, cuda_count) = cuda_hist.totals();

        assert_eq!(
            scalar_count, cuda_count,
            "Feature {}: count mismatch ({} vs {})",
            f, scalar_count, cuda_count
        );

        // CUDA atomics may have small numerical differences
        let grad_diff = (scalar_grad - cuda_grad).abs();
        let hess_diff = (scalar_hess - cuda_hess).abs();
        let grad_tol = scalar_grad.abs() * 0.01 + 0.1;
        let hess_tol = scalar_hess.abs() * 0.01 + 0.1;

        assert!(
            grad_diff < grad_tol,
            "Feature {}: gradient mismatch (scalar={}, cuda={}, diff={})",
            f,
            scalar_grad,
            cuda_grad,
            grad_diff
        );
        assert!(
            hess_diff < hess_tol,
            "Feature {}: hessian mismatch (scalar={}, cuda={}, diff={})",
            f,
            scalar_hess,
            cuda_hess,
            hess_diff
        );
    }

    eprintln!(
        "  ✓ CUDA histogram correctness verified for {} features × {} rows",
        dataset.num_features(),
        dataset.num_rows()
    );
}

/// Test full GBDT training with CUDA backend
#[cfg(feature = "cuda")]
#[test]
fn test_cuda_training_end_to_end() {
    let dataset = create_synthetic_dataset(5000, 789); // Small dataset for fast test

    eprintln!("\n=== Testing CUDA End-to-End Training ===");

    // Train with CUDA backend
    let cuda_config = GBDTConfig::new()
        .with_num_rounds(5) // Few rounds for speed
        .with_max_depth(4)
        .with_learning_rate(0.1)
        .with_backend(BackendType::Cuda);

    let start = std::time::Instant::now();
    let cuda_model = GBDTModel::train_binned(&dataset, cuda_config).expect("CUDA training failed");
    let cuda_time = start.elapsed();

    // Train with scalar backend for comparison
    let scalar_config = GBDTConfig::new()
        .with_num_rounds(5) // Few rounds for speed
        .with_max_depth(4)
        .with_learning_rate(0.1)
        .with_backend(BackendType::Scalar);

    let start = std::time::Instant::now();
    let scalar_model =
        GBDTModel::train_binned(&dataset, scalar_config).expect("Scalar training failed");
    let scalar_time = start.elapsed();

    eprintln!("  CUDA training time: {:?}", cuda_time);
    eprintln!("  Scalar training time: {:?}", scalar_time);
    eprintln!(
        "  Speedup: {:.2}x",
        scalar_time.as_secs_f64() / cuda_time.as_secs_f64()
    );

    // Both models should produce similar predictions
    let cuda_preds = cuda_model.predict(&dataset);
    let scalar_preds = scalar_model.predict(&dataset);

    assert_eq!(cuda_preds.len(), scalar_preds.len());

    // Compute RMSE between predictions
    let mse: f32 = cuda_preds
        .iter()
        .zip(scalar_preds.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / cuda_preds.len() as f32;
    let rmse = mse.sqrt();

    let target_range = dataset.targets().iter().cloned().fold(f32::MIN, f32::max)
        - dataset.targets().iter().cloned().fold(f32::MAX, f32::min);
    let tolerance = target_range * 0.01;

    eprintln!(
        "  CUDA vs Scalar RMSE: {:.6}, tolerance: {:.6}",
        rmse, tolerance
    );

    assert!(
        rmse < tolerance,
        "CUDA and Scalar predictions differ too much: RMSE={:.4} > tolerance={:.4}",
        rmse,
        tolerance
    );

    // Verify model quality
    let targets = dataset.targets();
    let cuda_mse: f32 = cuda_preds
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / cuda_preds.len() as f32;

    eprintln!("  ✓ CUDA model MSE: {:.4}", cuda_mse);
    assert!(cuda_mse < 1.0, "CUDA model MSE too high: {}", cuda_mse);
}

/// Test CUDA backend is used during AutoTuner hyperparameter search
#[cfg(feature = "cuda")]
#[test]
fn test_cuda_autotuner() {
    use treeboost::tuner::{
        AutoTuner, EvalStrategy, GridStrategy, OptimizationMetric, ParameterSpace, SpacePreset,
        TunerConfig,
    };

    // Need 60k rows so that 20% validation split = 12k rows (> 10k threshold)
    let dataset = create_synthetic_dataset(60000, 42);

    eprintln!("\n=== Testing CUDA in AutoTuner ===");

    // Create base config with Auto backend - should auto-select CUDA
    let base_config = GBDTConfig::new()
        .with_num_rounds(10) // Few rounds for speed
        .with_max_depth(4)
        .with_learning_rate(0.1)
        .with_backend(BackendType::Auto); // Auto should select CUDA

    // Create tuner with minimal trials
    let tuner_config = TunerConfig::new()
        .with_iterations(2) // Just 2 trials for speed
        .with_grid_strategy(GridStrategy::LatinHypercube { n_samples: 2 })
        .with_eval_strategy(EvalStrategy::holdout(0.2))
        .with_optimization_metric(OptimizationMetric::ValidationLoss)
        .with_verbose(false);

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(ParameterSpace::with_preset(SpacePreset::Regression));

    eprintln!("  Running 2 tuning trials with CUDA backend...");

    let start = std::time::Instant::now();
    let (best_config, history) = tuner.tune(&dataset).expect("Tuning should succeed");
    let elapsed = start.elapsed();

    let best = history.best().expect("Should have best trial");

    eprintln!("  Tuning completed in {:?}", elapsed);
    eprintln!("  Best validation loss: {:.6}", best.val_metric);
    eprintln!(
        "  Best config: depth={}, lr={:.3}",
        best_config.max_depth, best_config.learning_rate
    );

    // With CUDA, 2 trials on 15k rows should be very fast (< 5 seconds)
    // With CPU, it would take much longer (> 10 seconds)
    assert!(
        elapsed.as_secs() < 10,
        "Tuning too slow ({:?}), CUDA may not be used",
        elapsed
    );

    eprintln!("  ✓ CUDA AutoTuner test passed (fast execution confirms GPU usage)");
}
