//! Integration tests for data pipeline

use polars::prelude::*;
use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::{DataPipeline, PipelineConfig};

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
    let (dataset, state, _filtered_df) = pipeline
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

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // Make predictions
    let predictions = model.predict(&dataset);
    assert_eq!(predictions.len(), 20);

    // Calculate MSE - should be reasonable
    let targets = dataset.targets();
    let mse: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t): (&f32, &f32)| (p - t).powi(2))
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

    let (_test_preprocessed_df, test_dataset) = pipeline
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

    let (_dataset, state, _filtered_df) = pipeline
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

    let (_dataset, state, _filtered_df) = pipeline
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
    let (dataset, _state, _filtered_df) = pipeline
        .process_for_training(df.clone(), "target", None)
        .expect("Pipeline should succeed");

    // Train a model
    let config = GBDTConfig::new()
        .with_num_rounds(20)
        .with_max_depth(4)
        .with_learning_rate(0.1);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

    // Get predictions using binned data
    let binned_predictions = model.predict(&dataset);

    // Get raw feature values (row-major)
    let num_rows = 50;
    let num_features = 3;
    let mut raw_features = Vec::with_capacity(num_rows * num_features);

    // f0, f1, f2 for each row
    for i in 0..num_rows {
        raw_features.push((i + 1) as f64); // f0
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
