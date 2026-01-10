//! Integration tests for core GBDT training and prediction

mod common;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::inference::ConformalPredictor;
use treeboost::serialize::{load_model, save_model};

use common::create_synthetic_dataset;

#[test]
fn test_basic_training_and_prediction() {
    let dataset = create_synthetic_dataset(1000, 42);

    let config = GBDTConfig::new()
        .with_num_rounds(50)
        .with_max_depth(4)
        .with_learning_rate(0.1);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    assert_eq!(model.num_trees(), 50);

    let predictions = model.predict(&dataset);
    assert_eq!(predictions.len(), 1000);

    // Check predictions are in reasonable range
    let targets = dataset.targets();
    let mse: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;

    // MSE should be reasonably low after 50 rounds
    assert!(mse < 5.0, "MSE {} is too high", mse);
}

#[test]
fn test_pseudo_huber_loss() {
    // Create dataset with outliers
    let mut dataset = create_synthetic_dataset(500, 123);

    // Add outliers to targets (simulate dirty data)
    let targets = dataset.targets_mut();
    targets[0] = 1000.0; // Extreme outlier
    targets[10] = -500.0;
    targets[50] = 2000.0;

    // Train with Pseudo-Huber loss (should be robust to outliers)
    let config = GBDTConfig::new()
        .with_num_rounds(30)
        .with_max_depth(3)
        .with_pseudo_huber_loss(1.0);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    let predictions = model.predict(&dataset);

    // Predictions for non-outlier points should be reasonable
    // (not pulled towards extreme values)
    let non_outlier_predictions: Vec<f32> = predictions
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 0 && *i != 10 && *i != 50)
        .map(|(_, &p)| p)
        .collect();

    let mean_pred: f32 =
        non_outlier_predictions.iter().sum::<f32>() / non_outlier_predictions.len() as f32;

    // Mean prediction should be in reasonable range (not pulled to extremes)
    assert!(
        mean_pred > 0.0 && mean_pred < 20.0,
        "Mean prediction {} is extreme",
        mean_pred
    );
}

#[test]
fn test_conformal_prediction() {
    let dataset = create_synthetic_dataset(500, 456);

    // Train with conformal prediction enabled
    let config = GBDTConfig::new()
        .with_num_rounds(30)
        .with_max_depth(4)
        .with_conformal(0.2, 0.9)
        .unwrap(); // 20% calibration, 90% coverage

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    assert!(model.conformal_quantile().is_some());

    let (predictions, lower, upper) = model.predict_with_intervals(&dataset);

    // All intervals should be valid
    for i in 0..predictions.len() {
        assert!(
            lower[i] < predictions[i],
            "Lower bound should be less than prediction"
        );
        assert!(
            upper[i] > predictions[i],
            "Upper bound should be greater than prediction"
        );
        assert!(lower[i] < upper[i], "Lower should be less than upper");
    }
}

#[test]
fn test_model_serialization() {
    let dataset = create_synthetic_dataset(200, 789);

    let config = GBDTConfig::new().with_num_rounds(10).with_max_depth(3);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");
    let original_predictions = model.predict(&dataset);

    // Save model
    let temp_dir = tempfile::tempdir().expect("Should create temp dir");
    let model_path = temp_dir.path().join("model.rkyv");

    save_model(&model, &model_path).expect("Should save model");

    // Load model
    let loaded_model = load_model(&model_path).expect("Should load model");

    // Verify
    assert_eq!(loaded_model.num_trees(), model.num_trees());
    assert_eq!(loaded_model.base_prediction(), model.base_prediction());

    let loaded_predictions = loaded_model.predict(&dataset);

    for (orig, loaded) in original_predictions.iter().zip(loaded_predictions.iter()) {
        assert!(
            (orig - loaded).abs() < 1e-6,
            "Predictions should match after load"
        );
    }
}

#[test]
fn test_feature_importance() {
    let dataset = create_synthetic_dataset(500, 321);

    let config = GBDTConfig::new().with_num_rounds(50).with_max_depth(5);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");
    let importances = model.feature_importance();

    assert_eq!(importances.len(), 5);

    // Importances should sum to ~1
    let total: f32 = importances.iter().sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "Importances should sum to 1, got {}",
        total
    );

    // All importances should be non-negative
    for (i, &imp) in importances.iter().enumerate() {
        assert!(
            imp >= 0.0,
            "Importance for feature {} should be non-negative",
            i
        );
    }

    // First two features should have higher importance (they define the target)
    // This is a soft check - may not always hold due to correlation
    let top_two: f32 = importances[0] + importances[1];
    assert!(
        top_two > 0.2,
        "Top two features should have significant importance"
    );
}

#[test]
fn test_conformal_predictor() {
    let predictions: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let actuals: Vec<f32> = vec![1.1, 1.9, 3.2, 3.8, 5.1];

    // Compute residuals
    let residuals: Vec<f32> = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (a - p).abs())
        .collect();

    let predictor = ConformalPredictor::from_residuals(&residuals, 0.9);

    // New predictions with intervals
    let new_predictions = vec![2.5, 3.5];
    let intervals = predictor.predict_batch(&new_predictions);

    assert_eq!(intervals.len(), 2);

    for interval in &intervals {
        let lower = interval.lower.unwrap();
        let upper = interval.upper.unwrap();
        assert!(lower < interval.point);
        assert!(interval.point < upper);
    }

    // Coverage should be at target level (approximately)
    let test_preds = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let test_actuals: Vec<f32> = test_preds.iter().map(|x| x + 0.1).collect();

    let coverage = predictor.empirical_coverage(&test_actuals, &test_preds);
    // Coverage should be high for these well-calibrated predictions
    assert!(coverage > 0.5, "Coverage {} is too low", coverage);
}

#[test]
fn test_entropy_regularization() {
    let dataset = create_synthetic_dataset(500, 654);

    // Train without entropy regularization
    let config_no_entropy = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(6)
        .with_entropy_weight(0.0);

    let model_no_entropy =
        GBDTModel::train_binned(&dataset, config_no_entropy).expect("Training should succeed");

    // Train with entropy regularization
    let config_entropy = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(6)
        .with_entropy_weight(0.1);

    let model_entropy =
        GBDTModel::train_binned(&dataset, config_entropy).expect("Training should succeed");

    // Both models should produce reasonable predictions
    let preds_no_entropy = model_no_entropy.predict(&dataset);
    let preds_entropy = model_entropy.predict(&dataset);

    // Basic sanity checks
    assert_eq!(preds_no_entropy.len(), 500);
    assert_eq!(preds_entropy.len(), 500);

    // Predictions should be similar but not identical
    let diff: f32 = preds_no_entropy
        .iter()
        .zip(preds_entropy.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / preds_no_entropy.len() as f32;

    // Some difference is expected due to regularization
    assert!(diff > 0.0, "Entropy regularization should have some effect");
}

#[test]
fn test_max_leaves_constraint() {
    let dataset = create_synthetic_dataset(500, 987);

    let config = GBDTConfig::new()
        .with_num_rounds(10)
        .with_max_leaves(8)
        .with_max_depth(10); // High max_depth, but leaves constrained

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // Each tree should have at most 8 leaves
    for tree in model.trees() {
        assert!(
            tree.num_leaves() <= 8,
            "Tree has {} leaves, expected <= 8",
            tree.num_leaves()
        );
    }
}
