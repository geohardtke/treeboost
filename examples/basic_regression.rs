//! Basic Regression Example
//!
//! A simple end-to-end example showing how to:
//! - Generate synthetic regression data
//! - Train a GBDT model
//! - Make predictions
//! - Evaluate performance
//!
//! Run with: cargo run --release --example basic_regression

#[path = "common/mod.rs"]
mod common;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::BinnedDataset;

/// Generate synthetic regression data with a linear relationship
fn create_synthetic_dataset(n_samples: usize, n_features: usize, seed: u64) -> BinnedDataset {
    let mut rng = common::SimpleRng::new(seed);

    // Generate features (column-major layout)
    let mut features = Vec::with_capacity(n_samples * n_features);
    for _f in 0..n_features {
        for _r in 0..n_samples {
            features.push((rng.next_f32() * 255.0) as u8);
        }
    }

    // Generate targets: y = 10*f0 + 5*f1 - 3*f2 + noise
    // This tests if the model can learn the feature relationships
    let targets: Vec<f32> = (0..n_samples)
        .map(|i| {
            let f0 = features[i] as f32 / 255.0;
            let f1 = features[n_samples + i] as f32 / 255.0;
            let f2 = features[2 * n_samples + i] as f32 / 255.0;
            10.0 * f0 + 5.0 * f1 - 3.0 * f2 + rng.next_f32() * 0.5
        })
        .collect();

    let feature_info = common::create_feature_info(n_features, "feature");
    BinnedDataset::new(n_samples, features, targets, feature_info)
}

fn main() {
    println!("{}", "=".repeat(60));
    println!("TreeBoost: Basic Regression Example");
    println!("{}", "=".repeat(60));
    println!();

    // 1. Create synthetic dataset
    let n_samples = 5000;
    let n_features = 10;
    let seed = 42;

    println!("1. Generating synthetic regression dataset...");
    println!("   Samples: {}", n_samples);
    println!("   Features: {}", n_features);
    println!("   Relationship: y = 10*f0 + 5*f1 - 3*f2 + noise");
    println!();

    let dataset = create_synthetic_dataset(n_samples, n_features, seed);

    // 2. Configure the model
    println!("2. Configuring GBDT model...");
    let config = GBDTConfig::new()
        .with_num_rounds(100)
        .with_max_depth(5)
        .with_learning_rate(0.1)
        .with_subsample(0.8)
        .with_colsample(0.8)
        .with_seed(42);

    println!("   Rounds: 100");
    println!("   Max depth: 5");
    println!("   Learning rate: 0.1");
    println!("   Row sampling: 0.8");
    println!("   Feature sampling: 0.8");
    println!();

    // 3. Train the model
    println!("3. Training model...");
    let start = std::time::Instant::now();
    let model = GBDTModel::train_binned(&dataset, config).expect("Training failed");
    let elapsed = start.elapsed();

    println!("   Time: {:.2?}", elapsed);
    println!("   Trees: {}", model.num_trees());
    println!();

    // 4. Make predictions
    println!("4. Making predictions on training set...");
    let predictions = model.predict(&dataset);

    // 5. Evaluate performance
    println!("5. Evaluating performance...");

    // Calculate MSE
    let mse: f32 = predictions
        .iter()
        .zip(dataset.targets().iter())
        .map(|(pred, &target)| (pred - target).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;

    // Calculate MAE
    let mae: f32 = predictions
        .iter()
        .zip(dataset.targets().iter())
        .map(|(pred, &target)| (pred - target).abs())
        .sum::<f32>()
        / predictions.len() as f32;

    // Calculate R² (coefficient of determination)
    let mean_target = dataset.targets().iter().sum::<f32>() / dataset.targets().len() as f32;
    let ss_res: f32 = predictions
        .iter()
        .zip(dataset.targets().iter())
        .map(|(pred, &target)| (target - pred).powi(2))
        .sum();
    let ss_tot: f32 = dataset
        .targets()
        .iter()
        .map(|&target| (target - mean_target).powi(2))
        .sum();
    let r_squared = 1.0 - (ss_res / ss_tot);

    println!("   Mean Squared Error (MSE): {:.6}", mse);
    println!("   Mean Absolute Error (MAE): {:.6}", mae);
    println!("   R² Score: {:.6}", r_squared);
    println!();

    // 6. Feature importance
    println!("6. Feature Importance (top 5):");
    let importances = model.feature_importance();
    let mut indexed_importance: Vec<(usize, f32)> = importances
        .iter()
        .enumerate()
        .map(|(i, &imp)| (i, imp))
        .collect();
    indexed_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (feature_idx, importance) in indexed_importance.iter().take(5) {
        println!("   Feature {}: {:.6}", feature_idx, importance);
    }
    println!();

    // 7. Show prediction examples
    println!("7. Sample Predictions vs Actual:");
    for i in (0..predictions.len()).step_by(predictions.len() / 5) {
        let error = (predictions[i] - dataset.targets()[i]).abs();
        println!(
            "   Sample {}: Pred={:.4}, Actual={:.4}, Error={:.4}",
            i,
            predictions[i],
            dataset.targets()[i],
            error
        );
    }
    println!();

    println!("{}", "=".repeat(60));
    println!("Example completed successfully!");
    println!("{}", "=".repeat(60));
}
