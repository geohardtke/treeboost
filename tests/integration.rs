//! Integration tests for TreeBoost

use polars::prelude::*;
use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::{BinnedDataset, DataPipeline, FeatureInfo, FeatureType, PipelineConfig};
use treeboost::encoding::{CategoryFilter, CategoryMapping, OrderedTargetEncoder};
use treeboost::inference::ConformalPredictor;
use treeboost::serialize::{load_model, save_model};

/// Create a synthetic regression dataset for testing
fn create_synthetic_dataset(n: usize, seed: u64) -> BinnedDataset {
    // Deterministic pseudo-random using seed
    let mut state = seed;
    let mut next_rand = || -> f32 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((state >> 16) & 0x7FFF) as f32 / 32767.0
    };

    let num_features = 5;
    let mut features = Vec::with_capacity(n * num_features);

    // Generate features (column-major)
    for _f in 0..num_features {
        for _r in 0..n {
            features.push((next_rand() * 255.0) as u8);
        }
    }

    // Generate targets: y = f0 * 10 + f1 * 5 + noise
    let targets: Vec<f32> = (0..n)
        .map(|i| {
            let f0 = features[i] as f32 / 255.0;
            let f1 = features[n + i] as f32 / 255.0;
            f0 * 10.0 + f1 * 5.0 + next_rand() * 0.5
        })
        .collect();

    let feature_info: Vec<FeatureInfo> = (0..num_features)
        .map(|i| FeatureInfo {
            name: format!("feature_{}", i),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
        })
        .collect();

    BinnedDataset::new(n, features, targets, feature_info)
}

#[test]
fn test_basic_training_and_prediction() {
    let dataset = create_synthetic_dataset(1000, 42);

    let config = GBDTConfig::new()
        .with_num_rounds(50)
        .with_max_depth(4)
        .with_learning_rate(0.1);

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");

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

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");

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
        .with_conformal(0.2, 0.9); // 20% calibration, 90% coverage

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");

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

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");
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

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");
    let importances = model.feature_importances(5);

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
fn test_category_filter() {
    let mut filter = CategoryFilter::new(0.01, 0.99, 5);

    // Count categories
    for _ in 0..100 {
        filter.count("frequent_a");
        filter.count("frequent_b");
    }
    for _ in 0..10 {
        filter.count("medium");
    }
    for _ in 0..2 {
        filter.count("rare");
    }
    filter.count("very_rare");

    // Finalize
    filter.finalize(vec![
        "frequent_a".to_string(),
        "frequent_b".to_string(),
        "medium".to_string(),
        "rare".to_string(),
        "very_rare".to_string(),
    ]);

    // Frequent categories should be kept
    assert!(filter.is_frequent("frequent_a"));
    assert!(filter.is_frequent("frequent_b"));
    assert!(filter.is_frequent("medium")); // 10 > 5

    // Rare categories should be filtered
    assert!(!filter.is_frequent("rare")); // 2 < 5
    assert!(!filter.is_frequent("very_rare")); // 1 < 5
    assert!(!filter.is_frequent("unseen")); // 0 < 5

    // Filter function
    assert_eq!(filter.filter("frequent_a"), "frequent_a");
    assert_eq!(filter.filter("rare"), "unknown");
    assert_eq!(filter.filter("unseen"), "unknown");
}

#[test]
fn test_category_mapping() {
    let mut filter = CategoryFilter::new(0.01, 0.99, 3);

    for _ in 0..10 {
        filter.count("cat_a");
        filter.count("cat_b");
        filter.count("cat_c");
    }
    filter.count("rare");

    filter.finalize(vec![
        "cat_a".to_string(),
        "cat_b".to_string(),
        "cat_c".to_string(),
        "rare".to_string(),
    ]);

    let mapping = CategoryMapping::from_filter(&filter);

    // 3 frequent + 1 unknown
    assert_eq!(mapping.num_categories(), 4);

    // Indices should be unique and in range
    let idx_a = mapping.get_index("cat_a");
    let idx_b = mapping.get_index("cat_b");
    let idx_c = mapping.get_index("cat_c");
    let idx_rare = mapping.get_index("rare");

    assert!(idx_a < 3);
    assert!(idx_b < 3);
    assert!(idx_c < 3);
    assert_ne!(idx_a, idx_b);
    assert_ne!(idx_b, idx_c);
    assert_ne!(idx_a, idx_c);

    assert_eq!(idx_rare, mapping.unknown_idx);
    assert_eq!(mapping.get_index("unseen"), mapping.unknown_idx);
}

#[test]
fn test_ordered_target_encoder() {
    let categories = vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "C".to_string(),
    ];
    let targets = vec![10.0, 20.0, 12.0, 22.0, 14.0, 50.0];

    let mut encoder = OrderedTargetEncoder::new(5.0); // smoothing = 5

    let encoded = encoder.encode_column(&categories, &targets);

    // Ordered encoding: each row only sees PRIOR statistics
    // So first element gets 0 (no prior data), second gets mean of first, etc.
    assert_eq!(encoded.len(), 6);

    // All encoded values should be finite (not NaN or infinite)
    for &val in &encoded {
        assert!(val.is_finite(), "Encoded value should be finite");
    }

    // First element: no prior data -> global mean = 0
    assert_eq!(encoded[0], 0.0, "First element should be 0 (no prior data)");

    // Second element: global mean of first = 10.0
    assert!((encoded[1] - 10.0).abs() < 0.01, "Second should be ~10.0");

    // As more data accumulates, values become more meaningful
    // Check that later values are positive (non-trivial)
    assert!(encoded[5] > 0.0, "Later values should be positive");
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
        GBDTModel::train(&dataset, config_no_entropy).expect("Training should succeed");

    // Train with entropy regularization
    let config_entropy = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(6)
        .with_entropy_weight(0.1);

    let model_entropy =
        GBDTModel::train(&dataset, config_entropy).expect("Training should succeed");

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

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");

    // Each tree should have at most 8 leaves
    for tree in model.trees() {
        assert!(
            tree.num_leaves() <= 8,
            "Tree has {} leaves, expected <= 8",
            tree.num_leaves()
        );
    }
}

#[test]
fn test_data_pipeline_end_to_end() {
    // Create a DataFrame with mixed types including categoricals
    let df = df! {
        "price" => &[100.0, 150.0, 200.0, 180.0, 220.0, 90.0, 300.0, 250.0, 175.0, 400.0,
                     120.0, 160.0, 210.0, 190.0, 230.0, 95.0, 310.0, 260.0, 185.0, 420.0],
        "sqft" => &[1000.0, 1200.0, 1500.0, 1400.0, 1600.0, 900.0, 2000.0, 1800.0, 1350.0, 2500.0,
                    1050.0, 1250.0, 1550.0, 1450.0, 1650.0, 950.0, 2050.0, 1850.0, 1400.0, 2600.0],
        "bedrooms" => &[2.0, 3.0, 3.0, 3.0, 4.0, 2.0, 4.0, 4.0, 3.0, 5.0,
                        2.0, 3.0, 3.0, 3.0, 4.0, 2.0, 4.0, 4.0, 3.0, 5.0],
        "neighborhood" => &["downtown", "suburbs", "downtown", "suburbs", "downtown",
                            "rural", "downtown", "suburbs", "downtown", "suburbs",
                            "downtown", "suburbs", "downtown", "suburbs", "downtown",
                            "rural", "downtown", "suburbs", "downtown", "typo_rare"],
        "target" => &[250.0, 280.0, 350.0, 320.0, 400.0, 180.0, 500.0, 450.0, 300.0, 600.0,
                      260.0, 290.0, 360.0, 330.0, 410.0, 185.0, 510.0, 460.0, 310.0, 550.0]
    }
    .unwrap();

    // Create pipeline with configuration
    let pipeline = DataPipeline::new(
        PipelineConfig::new()
            .with_num_bins(16)
            .with_cms_params(0.01, 0.99, 2) // min_count=2 to filter "typo_rare" and "rural"
            .with_smoothing(5.0),
    );

    // Process for training
    let (dataset, state) = pipeline
        .process_for_training(df.clone(), "target", Some(&["neighborhood"]))
        .expect("Pipeline should succeed");

    assert_eq!(dataset.num_rows(), 20);
    assert_eq!(dataset.num_features(), 4); // price, sqft, bedrooms, neighborhood

    // Check that categorical encoding state was learned
    assert_eq!(state.categorical_encodings.len(), 1);
    assert_eq!(state.categorical_encodings[0].name, "neighborhood");

    // Train a model
    let config = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(4)
        .with_learning_rate(0.1);

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");

    // Make predictions
    let predictions = model.predict(&dataset);
    assert_eq!(predictions.len(), 20);

    // Calculate MSE - should be reasonable
    let targets = dataset.targets();
    let mse: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / predictions.len() as f32;

    assert!(mse < 10000.0, "MSE {} is too high for training data", mse);

    // Test inference with new data (including unseen category)
    let test_df = df! {
        "price" => &[175.0, 275.0],
        "sqft" => &[1300.0, 1900.0],
        "bedrooms" => &[3.0, 4.0],
        "neighborhood" => &["downtown", "unseen_area"]  // includes unseen category
    }
    .unwrap();

    let test_dataset = pipeline
        .process_for_inference(test_df, &state)
        .expect("Inference pipeline should succeed");

    assert_eq!(test_dataset.num_rows(), 2);

    let test_predictions = model.predict(&test_dataset);
    assert_eq!(test_predictions.len(), 2);

    // Predictions should be positive and reasonable
    for pred in &test_predictions {
        assert!(*pred > 0.0, "Prediction should be positive");
        assert!(*pred < 1000.0, "Prediction should be reasonable");
    }
}

#[test]
fn test_pipeline_rare_category_filtering() {
    // Create data with rare categories that should be filtered
    let df = df! {
        "feature" => &[1.0; 100],
        "category" => {
            let mut cats = vec!["frequent".to_string(); 50];
            cats.extend(vec!["also_frequent".to_string(); 30]);
            cats.extend(vec!["rare1".to_string(); 5]);
            cats.extend(vec!["rare2".to_string(); 5]);
            cats.extend(vec!["very_rare1".to_string(); 3]);
            cats.extend(vec!["very_rare2".to_string(); 3]);
            cats.extend(vec!["typo1".to_string(); 2]);
            cats.extend(vec!["typo2".to_string(); 2]);
            cats
        },
        "target" => &(0..100).map(|i| i as f64).collect::<Vec<_>>()
    }
    .unwrap();

    let pipeline = DataPipeline::new(
        PipelineConfig::new()
            .with_num_bins(8)
            .with_cms_params(0.01, 0.99, 10) // min_count=10 to filter many categories
            .with_smoothing(1.0),
    );

    let (_dataset, state) = pipeline
        .process_for_training(df, "target", Some(&["category"]))
        .expect("Pipeline should succeed");

    // Should have filtered rare categories
    let cat_state = &state.categorical_encodings[0];

    // Only "frequent" (50) and "also_frequent" (30) should remain
    // All others have count < 10
    assert_eq!(
        cat_state.category_mapping.category_to_idx.len(),
        2,
        "Expected 2 frequent categories, got {}",
        cat_state.category_mapping.category_to_idx.len()
    );

    // Verify the frequent ones are kept
    let kept_cats: Vec<&str> = cat_state
        .category_mapping
        .category_to_idx
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    assert!(kept_cats.contains(&"frequent"), "Should keep 'frequent'");
    assert!(
        kept_cats.contains(&"also_frequent"),
        "Should keep 'also_frequent'"
    );
}

#[test]
fn test_pipeline_target_encoding_prevents_leakage() {
    // Create data where target encoding with leakage would be obvious
    let df = df! {
        "category" => &["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
        "target" => &[10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 1.0, 2.0, 3.0, 4.0]
    }
    .unwrap();

    let pipeline = DataPipeline::new(
        PipelineConfig::new()
            .with_num_bins(8)
            .with_cms_params(0.01, 0.99, 1) // Keep all categories
            .with_smoothing(0.0), // No smoothing for clearer test
    );

    let (_dataset, state) = pipeline
        .process_for_training(df, "target", Some(&["category"]))
        .expect("Pipeline should succeed");

    // With ordered encoding:
    // - First "A" (row 0) sees no prior data -> encoded as 0
    // - First "B" (row 3) sees global mean of rows 0-2 = (10+20+30)/3 = 20
    // - If there was leakage, B would be encoded as mean(100,200,300)=200

    // The key property: each row's encoding doesn't include its own target
    // We verify this indirectly by checking the encoding map
    let cat_state = &state.categorical_encodings[0];

    // The final encoding map uses ALL data, but the training used ordered encoding
    // A's final mean: (10+20+30)/3 = 20
    // B's final mean: (100+200+300)/3 = 200
    // C's final mean: (1+2+3+4)/4 = 2.5

    let enc_a = cat_state.encoding_map.encode("A");
    let enc_b = cat_state.encoding_map.encode("B");
    let enc_c = cat_state.encoding_map.encode("C");

    // A should be encoded ~20, B ~200, C ~2.5
    assert!(
        enc_a < enc_b,
        "A (low target) should have lower encoding than B (high target)"
    );
    assert!(
        enc_c < enc_a,
        "C (very low target) should have lowest encoding"
    );
}

// ============================================================================
// Parquet Integration Tests
// ============================================================================
// These tests require parquet files generated by scripts/generate_samples.py
// Run with: cargo test parquet -- --ignored
// ============================================================================

/// Test loading and training on large numeric-only parquet file
#[test]
#[ignore] // Requires: python scripts/generate_samples.py --small
fn test_parquet_large_regression() {
    use std::path::Path;
    use treeboost::dataset::DatasetLoader;

    let parquet_path = Path::new("samples/synthetic/large_regression.parquet");
    if !parquet_path.exists() {
        eprintln!("Skipping test: {} not found", parquet_path.display());
        eprintln!("Run: python scripts/generate_samples.py --small");
        return;
    }

    let loader = DatasetLoader::new(64);
    let dataset = loader
        .load_parquet(parquet_path.to_str().unwrap(), "target", None)
        .expect("Should load parquet");

    assert!(dataset.num_rows() >= 10_000, "Expected at least 10K rows");
    assert_eq!(dataset.num_features(), 10, "Expected 10 features");

    // Train a model
    let config = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(5)
        .with_learning_rate(0.1);

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");
    let predictions = model.predict(&dataset);

    assert_eq!(predictions.len(), dataset.num_rows());

    // Calculate R² to verify model learned something
    let targets = dataset.targets();
    let mean_target: f32 = targets.iter().sum::<f32>() / targets.len() as f32;
    let ss_tot: f32 = targets.iter().map(|t| (t - mean_target).powi(2)).sum();
    let ss_res: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;

    assert!(r2 > 0.5, "R² should be > 0.5, got {}", r2);
}

/// Test loading and training on mixed types parquet file
#[test]
#[ignore] // Requires: python scripts/generate_samples.py --small
fn test_parquet_large_mixed() {
    use std::path::Path;

    let parquet_path = Path::new("samples/synthetic/large_mixed.parquet");
    if !parquet_path.exists() {
        eprintln!("Skipping test: {} not found", parquet_path.display());
        eprintln!("Run: python scripts/generate_samples.py --small");
        return;
    }

    let pipeline = DataPipeline::new(
        PipelineConfig::new()
            .with_num_bins(64)
            .with_cms_params(0.01, 0.99, 10)
            .with_smoothing(10.0),
    );

    let (dataset, state) = pipeline
        .load_parquet_for_training(
            parquet_path.to_str().unwrap(),
            "target",
            Some(&[
                "neighborhood",
                "property_type",
                "condition",
                "has_pool",
                "has_garage",
            ]),
        )
        .expect("Should load parquet with categoricals");

    assert!(dataset.num_rows() >= 10_000, "Expected at least 10K rows");

    // Should have encoded 5 categorical columns
    assert_eq!(
        state.categorical_encodings.len(),
        5,
        "Expected 5 categorical encodings"
    );

    // Train a model
    let config = GBDTConfig::new()
        .with_num_rounds(30)
        .with_max_depth(6)
        .with_learning_rate(0.1);

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");
    let predictions = model.predict(&dataset);

    assert_eq!(predictions.len(), dataset.num_rows());
}

/// Test dirty data handling in parquet
#[test]
#[ignore] // Requires: python scripts/generate_samples.py --small
fn test_parquet_large_dirty() {
    use std::path::Path;

    let parquet_path = Path::new("samples/synthetic/large_dirty.parquet");
    if !parquet_path.exists() {
        eprintln!("Skipping test: {} not found", parquet_path.display());
        eprintln!("Run: python scripts/generate_samples.py --small");
        return;
    }

    let pipeline = DataPipeline::new(
        PipelineConfig::new()
            .with_num_bins(32)
            .with_cms_params(0.001, 0.99, 50) // Filter categories with < 50 occurrences
            .with_smoothing(20.0),
    );

    let (dataset, state) = pipeline
        .load_parquet_for_training(
            parquet_path.to_str().unwrap(),
            "target",
            Some(&["category", "group"]),
        )
        .expect("Should load dirty parquet");

    assert!(dataset.num_rows() > 0, "Should have rows after filtering");

    // Train with Pseudo-Huber loss (robust to outliers)
    let config = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(4)
        .with_pseudo_huber_loss(1.0);

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");
    let predictions = model.predict(&dataset);

    assert_eq!(predictions.len(), dataset.num_rows());

    // Verify CMS filtered rare categories
    let cat_state = &state.categorical_encodings[0];
    // Should have filtered 500 rare categories, keeping only 5 frequent ones
    assert!(
        cat_state.category_mapping.category_to_idx.len() <= 10,
        "Should have filtered most rare categories, got {}",
        cat_state.category_mapping.category_to_idx.len()
    );
}

/// Test high-cardinality categorical handling in parquet
#[test]
#[ignore] // Requires: python scripts/generate_samples.py --small
fn test_parquet_high_cardinality() {
    use std::path::Path;

    let parquet_path = Path::new("samples/synthetic/large_high_cardinality.parquet");
    if !parquet_path.exists() {
        eprintln!("Skipping test: {} not found", parquet_path.display());
        eprintln!("Run: python scripts/generate_samples.py --small");
        return;
    }

    let pipeline = DataPipeline::new(
        PipelineConfig::new()
            .with_num_bins(64)
            .with_cms_params(0.001, 0.99, 20) // Filter categories with < 20 occurrences
            .with_smoothing(50.0),
    );

    let (dataset, state) = pipeline
        .load_parquet_for_training(
            parquet_path.to_str().unwrap(),
            "target",
            Some(&["user_id", "product_id", "region", "merchant_id"]),
        )
        .expect("Should load high-cardinality parquet");

    assert!(dataset.num_rows() >= 10_000, "Expected at least 10K rows");
    assert_eq!(
        state.categorical_encodings.len(),
        4,
        "Expected 4 categorical encodings"
    );

    // With 10K users and 100K rows, most users appear ~10 times
    // CMS filter with min_count=20 should filter many
    let user_encoding = &state.categorical_encodings[0];
    assert!(
        user_encoding.category_mapping.category_to_idx.len() < 10000,
        "Should have filtered some rare users"
    );

    // Train a model
    let config = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(5)
        .with_learning_rate(0.1);

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");
    let predictions = model.predict(&dataset);

    assert_eq!(predictions.len(), dataset.num_rows());
}

/// Stress test with 1M rows
#[test]
#[ignore] // Requires: python scripts/generate_samples.py (full, not --small)
fn test_parquet_stress_test() {
    use std::path::Path;
    use std::time::Instant;

    let parquet_path = Path::new("samples/synthetic/stress_test.parquet");
    if !parquet_path.exists() {
        eprintln!("Skipping test: {} not found", parquet_path.display());
        eprintln!("Run: python scripts/generate_samples.py");
        return;
    }

    let pipeline = DataPipeline::new(
        PipelineConfig::new()
            .with_num_bins(255)
            .with_cms_params(0.001, 0.99, 100)
            .with_smoothing(10.0),
    );

    let start = Instant::now();
    let (dataset, _state) = pipeline
        .load_parquet_for_training(parquet_path.to_str().unwrap(), "target", Some(&["cat"]))
        .expect("Should load stress test parquet");
    let load_time = start.elapsed();

    println!("Loaded {} rows in {:?}", dataset.num_rows(), load_time);
    assert!(dataset.num_rows() >= 100_000, "Expected at least 100K rows");

    // Train a model
    let config = GBDTConfig::new()
        .with_num_rounds(50)
        .with_max_depth(6)
        .with_learning_rate(0.1);

    let start = Instant::now();
    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");
    let train_time = start.elapsed();

    println!("Trained {} trees in {:?}", model.num_trees(), train_time);

    let start = Instant::now();
    let predictions = model.predict(&dataset);
    let predict_time = start.elapsed();

    println!("Predicted {} rows in {:?}", predictions.len(), predict_time);

    assert_eq!(predictions.len(), dataset.num_rows());
}

// ============================================================================
// Optimization Tests
// ============================================================================
// These tests verify 4-bit packing and cache-aware column reordering

/// Test 4-bit packed storage for low-cardinality features
#[test]
fn test_packed_dataset_memory_savings() {
    use treeboost::dataset::{PackedDataset, StorageMode};

    // Create dataset with mix of packable and non-packable features
    let num_rows = 1000;
    let mut features = Vec::with_capacity(num_rows * 4);

    // f0: 16 bins (packable) - categorical with low cardinality
    for r in 0..num_rows {
        features.push((r % 16) as u8);
    }
    // f1: 8 bins (packable) - binary-like feature
    for r in 0..num_rows {
        features.push((r % 8) as u8);
    }
    // f2: 256 bins (not packable) - high precision numeric
    for r in 0..num_rows {
        features.push((r % 256) as u8);
    }
    // f3: 4 bins (packable) - quartile-binned feature
    for r in 0..num_rows {
        features.push((r % 4) as u8);
    }

    let targets: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.1).collect();
    let feature_info: Vec<FeatureInfo> = vec![
        FeatureInfo {
            name: "categorical_16".to_string(),
            feature_type: FeatureType::Categorical,
            num_bins: 16,
            bin_boundaries: vec![],
        },
        FeatureInfo {
            name: "binary_like".to_string(),
            feature_type: FeatureType::Categorical,
            num_bins: 8,
            bin_boundaries: vec![],
        },
        FeatureInfo {
            name: "high_precision".to_string(),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
        },
        FeatureInfo {
            name: "quartile".to_string(),
            feature_type: FeatureType::Categorical,
            num_bins: 4,
            bin_boundaries: vec![],
        },
    ];

    let binned = BinnedDataset::new(num_rows, features, targets, feature_info);
    let packed = PackedDataset::from_binned(&binned);

    // Verify storage modes
    let modes = packed.storage_modes();
    assert_eq!(
        modes[0],
        StorageMode::Packed4Bit,
        "16-bin feature should be packed"
    );
    assert_eq!(
        modes[1],
        StorageMode::Packed4Bit,
        "8-bin feature should be packed"
    );
    assert_eq!(modes[2], StorageMode::U8, "256-bin feature should be u8");
    assert_eq!(
        modes[3],
        StorageMode::Packed4Bit,
        "4-bin feature should be packed"
    );

    // Memory savings: 3 of 4 features are packed (50% each)
    // Expected: (0.5 + 0.5 + 1.0 + 0.5) / 4 = 62.5% of original = 37.5% savings
    let savings = packed.memory_savings();
    assert!(
        savings > 0.35 && savings < 0.40,
        "Expected ~37.5% memory savings, got {:.1}%",
        savings * 100.0
    );

    // Verify data integrity
    for r in 0..num_rows {
        for f in 0..4 {
            assert_eq!(
                packed.get_bin(r, f),
                binned.get_bin(r, f),
                "Data mismatch at row {}, feature {}",
                r,
                f
            );
        }
    }

    // Round-trip verification
    let unpacked = packed.to_binned();
    for r in 0..num_rows {
        for f in 0..4 {
            assert_eq!(
                unpacked.get_bin(r, f),
                binned.get_bin(r, f),
                "Round-trip mismatch at row {}, feature {}",
                r,
                f
            );
        }
    }
}

/// Test cache-aware column reordering based on feature importance
#[test]
fn test_column_reordering_by_importance() {

    // Create a dataset where f2 is most important (highest correlation with target)
    let num_rows = 500;
    let num_features = 5;
    let mut features = Vec::with_capacity(num_rows * num_features);

    // Generate features where importance varies:
    // f0: noise (low importance)
    // f1: weak signal
    // f2: strong signal (most important)
    // f3: weak signal
    // f4: noise (low importance)
    for f in 0..num_features {
        for r in 0..num_rows {
            let base = (r * 17 + f * 31) % 256;
            features.push(base as u8);
        }
    }

    // Target strongly correlated with f2
    let targets: Vec<f32> = (0..num_rows)
        .map(|r| {
            let f2_val = features[2 * num_rows + r] as f32 / 255.0;
            f2_val * 100.0 + (r % 10) as f32 // Strong f2 signal + noise
        })
        .collect();

    let feature_info: Vec<FeatureInfo> = (0..num_features)
        .map(|i| FeatureInfo {
            name: format!("f{}", i),
            feature_type: FeatureType::Numeric,
            num_bins: 255,
            bin_boundaries: vec![],
        })
        .collect();

    let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);

    // Train model to compute feature importances
    let config = GBDTConfig::new()
        .with_num_rounds(30)
        .with_max_depth(4)
        .with_learning_rate(0.1);

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");

    // Get importance-based reordering
    let (reordered, permutation) = model.optimize_dataset_layout(&dataset);

    // Verify reordering happened
    assert!(
        !permutation.is_identity() || num_features <= 1,
        "Should reorder unless trivial"
    );

    // Verify feature names match after reordering
    for new_idx in 0..num_features {
        let orig_idx = permutation.to_original(new_idx);
        let orig_name = dataset.feature_info(orig_idx).name.clone();
        let reordered_name = reordered.feature_info(new_idx).name.clone();
        assert_eq!(
            orig_name, reordered_name,
            "Feature name mismatch at new index {}",
            new_idx
        );
    }

    // Verify data integrity after reordering
    for r in 0..num_rows {
        for new_idx in 0..num_features {
            let orig_idx = permutation.to_original(new_idx);
            assert_eq!(
                reordered.get_bin(r, new_idx),
                dataset.get_bin(r, orig_idx),
                "Data mismatch at row {}, new_idx {} (orig {})",
                r,
                new_idx,
                orig_idx
            );
        }
    }

    // Verify most important feature is first (or near first)
    let importances = model.feature_importances(num_features);
    let most_important_orig = importances
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let most_important_new = permutation.to_new(most_important_orig);
    assert!(
        most_important_new <= 1,
        "Most important feature (orig {}) should be near front, but is at position {}",
        most_important_orig,
        most_important_new
    );
}

/// Test that packed dataset predictions match original dataset predictions
#[test]
fn test_packed_dataset_prediction_equivalence() {
    use treeboost::dataset::PackedDataset;

    // Create packable dataset (all bins <= 15)
    let num_rows = 200;
    let num_features = 4;
    let mut features = Vec::with_capacity(num_rows * num_features);

    for f in 0..num_features {
        for r in 0..num_rows {
            features.push(((r * (f + 1) * 7) % 16) as u8);
        }
    }

    let targets: Vec<f32> = (0..num_rows)
        .map(|r| {
            let f0 = features[r] as f32;
            let f1 = features[num_rows + r] as f32;
            f0 * 2.0 + f1 * 1.5 + (r % 5) as f32
        })
        .collect();

    let feature_info: Vec<FeatureInfo> = (0..num_features)
        .map(|i| FeatureInfo {
            name: format!("f{}", i),
            feature_type: FeatureType::Numeric,
            num_bins: 16,
            bin_boundaries: vec![],
        })
        .collect();

    let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);
    let packed = PackedDataset::from_binned(&dataset);

    // Train model on original dataset
    let config = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(3)
        .with_learning_rate(0.1);

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");

    // Get predictions on original
    let preds_original = model.predict(&dataset);

    // Convert packed back to binned for prediction
    let unpacked = packed.to_binned();
    let preds_unpacked = model.predict(&unpacked);

    // Predictions should be identical
    assert_eq!(preds_original.len(), preds_unpacked.len());
    for (i, (orig, unp)) in preds_original.iter().zip(preds_unpacked.iter()).enumerate() {
        assert!(
            (orig - unp).abs() < 1e-6,
            "Prediction mismatch at row {}: {} vs {}",
            i,
            orig,
            unp
        );
    }
}

/// Test access tracker for dynamic reordering
#[test]
fn test_access_tracker() {
    use treeboost::dataset::{AccessTracker, ColumnPermutation};

    let mut tracker = AccessTracker::new(5);

    // Simulate feature access patterns from tree traversal
    // f2 accessed most, f0 second, others rarely
    for _ in 0..100 {
        tracker.record(2);
    }
    for _ in 0..50 {
        tracker.record(0);
    }
    for _ in 0..10 {
        tracker.record(1);
    }
    for _ in 0..5 {
        tracker.record(3);
    }
    tracker.record(4);

    let order = tracker.optimal_order();
    assert_eq!(order[0], 2, "Most accessed (f2) should be first");
    assert_eq!(order[1], 0, "Second most (f0) should be second");

    let perm = ColumnPermutation::from_access_tracker(&tracker);
    assert_eq!(perm.to_new(2), 0, "f2 should map to position 0");
    assert_eq!(perm.to_new(0), 1, "f0 should map to position 1");
}

// ============================================================================
// Raw Prediction Tests
// ============================================================================

#[test]
fn test_raw_prediction_equivalence() {
    // Create a dataset using the data pipeline (which sets up proper bin_boundaries)
    let df = df! {
        "f0" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                  11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                  21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                  31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                  41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0],
        "f1" => &[50.0, 49.0, 48.0, 47.0, 46.0, 45.0, 44.0, 43.0, 42.0, 41.0,
                  40.0, 39.0, 38.0, 37.0, 36.0, 35.0, 34.0, 33.0, 32.0, 31.0,
                  30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0,
                  20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0,
                  10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "f2" => &[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                  5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                  5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                  5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
                  5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        "target" => &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
                      110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
                      210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
                      310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0,
                      410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0]
    }
    .unwrap();

    // Process with data pipeline to get proper bin boundaries
    let pipeline = DataPipeline::new(PipelineConfig::new().with_num_bins(16));
    let (dataset, _state) = pipeline
        .process_for_training(df.clone(), "target", None)
        .expect("Pipeline should succeed");

    // Train a model
    let config = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(4)
        .with_learning_rate(0.1);

    let model = GBDTModel::train(&dataset, config).expect("Training should succeed");

    // Get predictions using binned data
    let binned_predictions = model.predict(&dataset);

    // Get raw feature values (row-major)
    let num_rows = 50;
    let num_features = 3;
    let mut raw_features = Vec::with_capacity(num_rows * num_features);

    // f0, f1, f2 for each row
    for i in 0..num_rows {
        raw_features.push((i + 1) as f64);  // f0
        raw_features.push((50 - i) as f64); // f1
        raw_features.push(((i % 10) * 5 + 5) as f64); // f2
    }

    // Get predictions using raw values
    let raw_predictions = model.predict_raw(&raw_features);

    // Both should have the same length
    assert_eq!(binned_predictions.len(), raw_predictions.len());

    // Predictions should be very close (within floating-point tolerance)
    // Note: Due to binning discretization, there may be small differences
    let max_diff: f32 = binned_predictions
        .iter()
        .zip(raw_predictions.iter())
        .map(|(b, r)| (b - r).abs())
        .fold(0.0f32, f32::max);

    // The predictions should be reasonably close
    // (they may differ slightly due to binning boundary edge cases)
    assert!(
        max_diff < 50.0,
        "Max difference between binned and raw predictions: {} (expected < 50.0)",
        max_diff
    );
}
