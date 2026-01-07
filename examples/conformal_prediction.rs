//! Conformal Prediction Example
//!
//! Demonstrates split conformal prediction for uncertainty quantification:
//! - Calibration set usage for interval estimation
//! - Prediction intervals with guaranteed coverage
//! - Coverage validation
//!
//! Run with: cargo run --release --example conformal_prediction

#[path = "common/mod.rs"]
mod common;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::BinnedDataset;

/// Generate synthetic regression data with noise
fn create_synthetic_dataset(n_samples: usize, n_features: usize, seed: u64) -> BinnedDataset {
    let mut rng = common::SimpleRng::new(seed);

    // Generate features (column-major)
    let mut features = Vec::with_capacity(n_samples * n_features);
    for _f in 0..n_features {
        for _r in 0..n_samples {
            features.push((rng.next_f32() * 255.0) as u8);
        }
    }

    // Generate targets with significant noise
    let targets: Vec<f32> = (0..n_samples)
        .map(|i| {
            let f0 = features[i] as f32 / 255.0;
            let f1 = features[n_samples + i] as f32 / 255.0;
            10.0 * f0 + 5.0 * f1 + (rng.next_f32() - 0.5) * 4.0
        })
        .collect();

    let feature_info = common::create_feature_info(n_features, "feature");
    BinnedDataset::new(n_samples, features, targets, feature_info)
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("TreeBoost: Conformal Prediction Example");
    println!("{}", "=".repeat(70));
    println!();

    // 1. Create and split dataset
    let n_total = 6000;
    let n_train = 4000;
    let n_calib = 1000;
    let n_test = 1000;
    let n_features = 10;
    let seed = 42;

    println!("1. Generating dataset with noise...");
    println!("   Total samples: {}", n_total);
    println!(
        "   Training: {}, Calibration: {}, Test: {}",
        n_train, n_calib, n_test
    );
    println!("   Relationship: y = 10*f0 + 5*f1 + noise");
    println!();

    let full_dataset = create_synthetic_dataset(n_total, n_features, seed);

    // Split into train, calibration, and test sets
    let train_dataset = common::extract_subset(&full_dataset, 0, n_train);
    let calib_dataset = common::extract_subset(&full_dataset, n_train, n_train + n_calib);
    let test_dataset = common::extract_subset(&full_dataset, n_train + n_calib, n_total);

    // 2. Configure and train model on training set
    println!("2. Training base model...");
    let config = GBDTConfig::new()
        .with_num_rounds(50)
        .with_max_depth(5)
        .with_learning_rate(0.1)
        .with_subsample(0.8)
        .with_seed(42);

    let model = GBDTModel::train_binned(&train_dataset, config).expect("Training failed");
    println!("   Trained with {} trees", model.num_trees());
    println!();

    // 3. Get predictions on calibration set
    println!("3. Computing residuals on calibration set...");
    let calib_preds = model.predict(&calib_dataset);
    let calib_targets = calib_dataset.targets();

    // Compute residuals (absolute errors)
    let mut residuals: Vec<f32> = calib_preds
        .iter()
        .zip(calib_targets.iter())
        .map(|(pred, &target)| (target - pred).abs())
        .collect();

    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_residual = residuals.iter().sum::<f32>() / residuals.len() as f32;
    println!("   Mean absolute error (calibration): {:.4}", mean_residual);
    println!();

    // 4. Compute quantile for 90% coverage
    let coverage = 0.9;
    let quantile_idx =
        ((residuals.len() as f32 * coverage).ceil() as usize).min(residuals.len() - 1);
    let quantile = residuals[quantile_idx];

    println!(
        "4. Computing prediction intervals for {:.0}% coverage...",
        coverage * 100.0
    );
    println!("   Quantile of absolute errors: {:.4}", quantile);
    println!();

    // 5. Make predictions on test set with intervals
    println!("5. Making test predictions with intervals...");
    let test_preds = model.predict(&test_dataset);
    let test_targets = test_dataset.targets();

    // Compute intervals: [pred - quantile, pred + quantile]
    let intervals: Vec<(f32, f32, f32)> = test_preds
        .iter()
        .map(|&pred| (pred - quantile, pred, pred + quantile))
        .collect();

    println!();

    // 6. Evaluate coverage
    println!("6. Evaluating prediction interval coverage...");

    let covered = intervals
        .iter()
        .zip(test_targets.iter())
        .filter(|((lower, _, upper), &target)| target >= *lower && target <= *upper)
        .count();

    let actual_coverage = covered as f32 / test_targets.len() as f32;

    println!("   Target coverage: {:.1}%", coverage * 100.0);
    println!(
        "   Actual coverage: {:.1}% ({}/{})",
        actual_coverage * 100.0,
        covered,
        test_targets.len()
    );
    println!();

    // 7. Compute point prediction errors
    println!("7. Point prediction performance on test set...");

    let mae: f32 = test_preds
        .iter()
        .zip(test_targets.iter())
        .map(|(pred, &target)| (target - pred).abs())
        .sum::<f32>()
        / test_targets.len() as f32;

    let rmse: f32 = (test_preds
        .iter()
        .zip(test_targets.iter())
        .map(|(pred, &target)| (target - pred).powi(2))
        .sum::<f32>()
        / test_targets.len() as f32)
        .sqrt();

    println!("   Mean Absolute Error: {:.4}", mae);
    println!("   Root Mean Squared Error: {:.4}", rmse);
    println!();

    // 8. Show interval width statistics
    println!("8. Interval Width Statistics...");

    let widths: Vec<f32> = intervals
        .iter()
        .map(|(lower, _, upper)| upper - lower)
        .collect();
    let min_width = widths.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_width = widths.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_width = widths.iter().sum::<f32>() / widths.len() as f32;

    println!("   Minimum width: {:.4}", min_width);
    println!("   Maximum width: {:.4}", max_width);
    println!("   Mean width: {:.4}", mean_width);
    println!();

    // 9. Show sample predictions with intervals
    println!("9. Sample Predictions with Intervals:");
    println!(
        "   {:>6} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "ID", "Lower", "Point", "Upper", "Actual", "Covered"
    );
    println!("   {}", "-".repeat(60));

    for i in (0..test_targets.len()).step_by(test_targets.len() / 5) {
        let (lower, point, upper) = intervals[i];
        let actual = test_targets[i];
        let is_covered = actual >= lower && actual <= upper;
        println!(
            "   {:>6} {:>10.4} {:>10.4} {:>10.4} {:>8.4} {:>8}",
            i,
            lower,
            point,
            upper,
            actual,
            if is_covered { "YES" } else { "NO" }
        );
    }
    println!();

    println!("{}", "=".repeat(70));
    println!("Example completed successfully!");
    println!("{}", "=".repeat(70));
}
