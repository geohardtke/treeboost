//! UniversalModel Example
//!
//! Demonstrates all three boosting modes:
//! - **PureTree**: Standard GBDT (best for most tabular problems)
//! - **LinearThenTree**: Linear model first, trees on residuals (best for trending data)
//! - **RandomForest**: Parallel trees with bagging (best for variance reduction)
//!
//! Run with: cargo run --release --example universal_model

#[path = "common/mod.rs"]
mod common;

use treeboost::dataset::BinnedDataset;
use treeboost::loss::MseLoss;
use treeboost::model::{BoostingMode, UniversalConfig, UniversalModel};

/// Generate synthetic dataset with a linear trend + non-linear component
fn create_synthetic_dataset(n_samples: usize, n_features: usize, seed: u64) -> BinnedDataset {
    let mut rng = common::SimpleRng::new(seed);

    // Generate features (column-major layout)
    let mut features = Vec::with_capacity(n_samples * n_features);
    for _f in 0..n_features {
        for _r in 0..n_samples {
            features.push((rng.next_f32() * 255.0) as u8);
        }
    }

    // Generate targets with both linear and non-linear components:
    // y = 10*f0 + 5*f1 - 3*f2 + sin(f0*10)*5 + noise
    // - Linear component: 10*f0 + 5*f1 - 3*f2
    // - Non-linear component: sin(f0*10)*5
    let targets: Vec<f32> = (0..n_samples)
        .map(|i| {
            let f0 = features[i] as f32 / 255.0;
            let f1 = features[n_samples + i] as f32 / 255.0;
            let f2 = features[2 * n_samples + i] as f32 / 255.0;
            // Linear trend
            let linear = 10.0 * f0 + 5.0 * f1 - 3.0 * f2;
            // Non-linear component
            let nonlinear = (f0 * 10.0).sin() * 5.0;
            // Noise
            let noise = rng.next_f32() * 0.5;
            linear + nonlinear + noise
        })
        .collect();

    let feature_info = common::create_feature_info(n_features, "feature");
    BinnedDataset::new(n_samples, features, targets, feature_info)
}

/// Calculate R² score
fn r_squared(predictions: &[f32], targets: &[f32]) -> f32 {
    let mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let ss_res: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();
    let ss_tot: f32 = targets.iter().map(|t| (t - mean).powi(2)).sum();
    1.0 - (ss_res / ss_tot)
}

/// Calculate MSE
fn mse(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / predictions.len() as f32
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("TreeBoost: UniversalModel Example - Three Boosting Modes");
    println!("{}", "=".repeat(70));
    println!();

    // Create synthetic data
    let n_samples = 5000;
    let n_features = 10;
    let seed = 42;

    println!("Dataset: {} samples, {} features", n_samples, n_features);
    println!("Target: y = 10*f0 + 5*f1 - 3*f2 + sin(f0*10)*5 + noise");
    println!("        (linear trend + non-linear component)");
    println!();

    let dataset = create_synthetic_dataset(n_samples, n_features, seed);
    let loss_fn = MseLoss;

    // Split into train/test
    let split_idx = (n_samples as f32 * 0.8) as usize;
    let train_data = common::extract_subset(&dataset, 0, split_idx);
    let test_data = common::extract_subset(&dataset, split_idx, n_samples);

    println!("Train: {} samples, Test: {} samples", train_data.num_rows(), test_data.num_rows());
    println!();

    // =========================================================================
    // Mode 1: PureTree (Standard GBDT)
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("MODE 1: PureTree (Standard GBDT)");
    println!("{}", "-".repeat(70));
    println!("Best for: Most tabular problems, categorical-heavy data");
    println!();

    let config = UniversalConfig::new()
        .with_mode(BoostingMode::PureTree)
        .with_num_rounds(100)
        .with_learning_rate(0.1)
        .with_seed(42);

    let start = std::time::Instant::now();
    let model = UniversalModel::train(&train_data, config, &loss_fn).expect("Training failed");
    let elapsed = start.elapsed();

    let test_preds = model.predict(&test_data);
    let test_r2 = r_squared(&test_preds, test_data.targets());
    let test_mse = mse(&test_preds, test_data.targets());

    println!("  Trees: {}", model.num_trees());
    println!("  Time:  {:.2?}", elapsed);
    println!("  Test R²:  {:.4}", test_r2);
    println!("  Test MSE: {:.4}", test_mse);
    println!();

    // =========================================================================
    // Mode 2: LinearThenTree (Residual Boosting)
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("MODE 2: LinearThenTree (Residual Boosting)");
    println!("{}", "-".repeat(70));
    println!("Best for: Time-series with trends, extrapolation beyond training range");
    println!("How it works:");
    println!("  1. Linear model captures global trend (10*f0 + 5*f1 - 3*f2)");
    println!("  2. Trees capture non-linear residuals (sin component)");
    println!();

    let config = UniversalConfig::new()
        .with_mode(BoostingMode::LinearThenTree)
        .with_num_rounds(80)           // Fewer tree rounds needed
        .with_linear_rounds(10)        // 10 linear boosting iterations first
        .with_learning_rate(0.1)
        .with_seed(42);

    let start = std::time::Instant::now();
    let model = UniversalModel::train(&train_data, config, &loss_fn).expect("Training failed");
    let elapsed = start.elapsed();

    let test_preds = model.predict(&test_data);
    let test_r2 = r_squared(&test_preds, test_data.targets());
    let test_mse = mse(&test_preds, test_data.targets());

    println!("  Linear: Yes (captures trend)");
    println!("  Trees:  {}", model.num_trees());
    println!("  Time:   {:.2?}", elapsed);
    println!("  Test R²:  {:.4}", test_r2);
    println!("  Test MSE: {:.4}", test_mse);
    println!();

    // =========================================================================
    // Mode 3: RandomForest (Bagging)
    // =========================================================================
    println!("{}", "-".repeat(70));
    println!("MODE 3: RandomForest (Bagging)");
    println!("{}", "-".repeat(70));
    println!("Best for: Robustness, variance reduction, avoiding overfitting");
    println!("How it works:");
    println!("  - Each tree trained on bootstrap sample (with replacement)");
    println!("  - Trees trained in PARALLEL (independent)");
    println!("  - Predictions averaged across all trees");
    println!();

    let config = UniversalConfig::new()
        .with_mode(BoostingMode::RandomForest)
        .with_num_rounds(100)          // Number of trees
        .with_subsample(0.7)           // Bootstrap sample ratio
        .with_seed(42);

    let start = std::time::Instant::now();
    let model = UniversalModel::train(&train_data, config, &loss_fn).expect("Training failed");
    let elapsed = start.elapsed();

    let test_preds = model.predict(&test_data);
    let test_r2 = r_squared(&test_preds, test_data.targets());
    let test_mse = mse(&test_preds, test_data.targets());

    println!("  Trees: {} (trained in parallel)", model.num_trees());
    println!("  Time:  {:.2?}", elapsed);
    println!("  Test R²:  {:.4}", test_r2);
    println!("  Test MSE: {:.4}", test_mse);
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("{}", "=".repeat(70));
    println!("Summary: Mode Selection Guide");
    println!("{}", "=".repeat(70));
    println!();
    println!("| Mode           | Best For                                      |");
    println!("|----------------|-----------------------------------------------|");
    println!("| PureTree       | General tabular, categorical features         |");
    println!("| LinearThenTree | Trending data, time-series, extrapolation     |");
    println!("| RandomForest   | Robustness, variance reduction, noisy data    |");
    println!();
    println!("Example completed successfully!");
}
