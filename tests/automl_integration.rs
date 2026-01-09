//! Integration tests for AutoML pipeline
//!
//! These tests verify end-to-end functionality using synthetic datasets
//! that represent common real-world scenarios.

use polars::prelude::*;
use treeboost::{
    auto_train, auto_train_quick, auto_train_thorough, auto_train_with_mode, AutoBuilder,
    AutoConfig, BoostingMode, TuningLevel,
};

/// Create a simple regression dataset with linear trend + noise
fn create_linear_regression_dataset(n_rows: usize) -> DataFrame {
    let mut rng = fastrand::Rng::with_seed(42);

    let x1: Vec<f64> = (0..n_rows).map(|i| i as f64 / 10.0).collect();
    let x2: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 100.0).collect();
    let x3: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 50.0).collect();

    // Target: y = 2*x1 + 0.5*x2 - 0.1*x3 + noise
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .zip(x3.iter())
        .map(|((&x1, &x2), &x3)| 2.0 * x1 + 0.5 * x2 - 0.1 * x3 + rng.f64() * 5.0 - 2.5)
        .collect();

    df!(
        "x1" => x1,
        "x2" => x2,
        "x3" => x3,
        "target" => y
    )
    .unwrap()
}

/// Create a dataset with non-linear patterns (for tree-based models)
fn create_nonlinear_regression_dataset(n_rows: usize) -> DataFrame {
    let mut rng = fastrand::Rng::with_seed(123);

    let x1: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 10.0).collect();
    let x2: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 10.0).collect();

    // Non-linear: y = sin(x1) * x2 + noise
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&x1, &x2)| (x1.sin() * x2) + rng.f64() * 2.0 - 1.0)
        .collect();

    df!(
        "x1" => x1,
        "x2" => x2,
        "target" => y
    )
    .unwrap()
}

/// Create a dataset with categorical features
fn create_mixed_type_dataset(n_rows: usize) -> DataFrame {
    let mut rng = fastrand::Rng::with_seed(456);

    let numeric: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 100.0).collect();

    let categories = ["A", "B", "C", "D"];
    let categorical: Vec<&str> = (0..n_rows)
        .map(|_| categories[rng.usize(..categories.len())])
        .collect();

    // Target depends on both numeric and categorical
    let y: Vec<f64> = numeric
        .iter()
        .zip(categorical.iter())
        .map(|(&num, &cat)| {
            let cat_effect = match cat {
                "A" => 10.0,
                "B" => 20.0,
                "C" => 30.0,
                "D" => 40.0,
                _ => 0.0,
            };
            num * 0.5 + cat_effect + rng.f64() * 5.0
        })
        .collect();

    df!(
        "numeric_feature" => numeric,
        "categorical_feature" => categorical,
        "target" => y
    )
    .unwrap()
}

/// Create a dataset with high cardinality ID column (should be dropped)
fn create_dataset_with_id_column(n_rows: usize) -> DataFrame {
    let mut rng = fastrand::Rng::with_seed(789);

    let ids: Vec<i64> = (0..n_rows).map(|i| i as i64).collect();
    let feature: Vec<f64> = (0..n_rows).map(|_| rng.f64() * 100.0).collect();
    let target: Vec<f64> = feature
        .iter()
        .map(|&f| f * 2.0 + rng.f64() * 10.0)
        .collect();

    df!(
        "id" => ids,
        "feature" => feature,
        "target" => target
    )
    .unwrap()
}

#[test]
fn test_auto_train_linear_data() {
    let df = create_linear_regression_dataset(500);

    // Train with auto settings
    let model = auto_train(&df, "target").expect("Training should succeed");

    // Verify model can predict
    let predictions = model.predict(&df).expect("Prediction should succeed");
    assert_eq!(predictions.len(), 500);

    // Check that mode was selected
    let mode = model.mode();
    println!("Selected mode: {:?}", mode);

    // Verify summary works
    let summary = model.summary();
    assert!(summary.contains("TreeBoost Pipeline Report"));
    assert!(summary.contains("MODE SELECTION"));
}

#[test]
fn test_auto_train_nonlinear_data() {
    let df = create_nonlinear_regression_dataset(500);

    let model = auto_train(&df, "target").expect("Training should succeed");

    // Non-linear data should likely select PureTree
    let mode = model.mode();
    println!("Selected mode for non-linear data: {:?}", mode);

    // Verify prediction works
    let predictions = model.predict(&df).expect("Prediction should succeed");
    assert_eq!(predictions.len(), 500);
}

#[test]
fn test_auto_train_quick() {
    let df = create_linear_regression_dataset(300);

    let model = auto_train_quick(&df, "target").expect("Quick training should succeed");

    // Verify it produces a valid model
    let predictions = model.predict(&df).expect("Prediction should succeed");
    assert_eq!(predictions.len(), 300);

    // Check timing exists
    let build_time = model.build_time();
    println!("Quick build time: {:?}", build_time);
}

#[test]
fn test_auto_train_thorough() {
    let df = create_linear_regression_dataset(200);

    let model = auto_train_thorough(&df, "target").expect("Thorough training should succeed");

    // Verify it produces a valid model
    let predictions = model.predict(&df).expect("Prediction should succeed");
    assert_eq!(predictions.len(), 200);

    // Thorough should have tuning results
    if let Some(ltt_tuning) = model.ltt_tuning() {
        println!("LTT tuning R²: {:.4}", ltt_tuning.linear_r2);
    }
}

#[test]
fn test_auto_train_with_forced_mode() {
    let df = create_nonlinear_regression_dataset(300);

    // Force LinearThenTree even though data is non-linear
    let model = auto_train_with_mode(&df, "target", BoostingMode::LinearThenTree)
        .expect("Training with forced mode should succeed");

    assert_eq!(model.mode(), BoostingMode::LinearThenTree);

    let predictions = model.predict(&df).expect("Prediction should succeed");
    assert_eq!(predictions.len(), 300);
}

#[test]
fn test_auto_builder_with_config() {
    let df = create_linear_regression_dataset(400);

    let config = AutoConfig::default()
        .with_tuning(TuningLevel::Quick)
        .with_auto_features(true)
        .with_verbose(false);

    let builder = AutoBuilder::with_config(config);
    let result = builder.fit(&df, "target").expect("Fit should succeed");

    // Check that we got a result with metadata
    assert!(result.column_profile.is_some());

    // Create AutoModel from result
    let model = treeboost::AutoModel::from_build_result(result);
    let predictions = model.predict(&df).expect("Prediction should succeed");
    assert_eq!(predictions.len(), 400);
}

#[test]
fn test_profiler_drops_id_columns() {
    let df = create_dataset_with_id_column(500);

    let model = auto_train(&df, "target").expect("Training should succeed");

    // Verify the ID column was identified and dropped
    if let Some(profile) = model.column_profile() {
        let dropped_cols: Vec<_> = profile
            .drop_columns
            .iter()
            .map(|d| d.name.as_str())
            .collect();
        println!("Dropped columns: {:?}", dropped_cols);

        // The 'id' column should be dropped as ID-like
        assert!(
            dropped_cols.contains(&"id"),
            "ID column should be automatically dropped"
        );
    }
}

#[test]
fn test_mixed_type_dataset() {
    let df = create_mixed_type_dataset(400);

    // This will currently work only with numeric features
    // Categorical encoding is in the plan but extract_raw_features
    // currently only handles numerics
    let result = auto_train(&df, "target");

    match result {
        Ok(model) => {
            // If it succeeds, verify prediction works
            let predictions = model.predict(&df).expect("Prediction should succeed");
            assert_eq!(predictions.len(), 400);
        }
        Err(e) => {
            // Expected to fail until categorical encoding is fully integrated
            println!("Mixed type dataset error (expected): {}", e);
            assert!(
                e.to_string().contains("numeric") || e.to_string().contains("categorical"),
                "Error should mention numeric or categorical features"
            );
        }
    }
}

#[test]
fn test_phase_times_reported() {
    let df = create_linear_regression_dataset(300);

    let model = auto_train(&df, "target").expect("Training should succeed");

    let phase_times = model.phase_times();

    // Verify all phases have non-zero time (at least some work was done)
    println!("Profiling: {:?}", phase_times.profiling);
    println!("Preprocessing: {:?}", phase_times.preprocessing);
    println!("Feature Engineering: {:?}", phase_times.feature_engineering);
    println!("Analysis: {:?}", phase_times.analysis);
    println!("Tuning: {:?}", phase_times.tuning);
    println!("Training: {:?}", phase_times.training);

    // All phases should complete (even if nearly instant)
    // Just verify the structure is populated
}

#[test]
fn test_summary_contains_key_info() {
    let df = create_linear_regression_dataset(200);

    let model = auto_train(&df, "target").expect("Training should succeed");
    let summary = model.summary();

    // Verify summary contains expected sections
    assert!(summary.contains("TreeBoost Pipeline Report"));
    assert!(summary.contains("MODE SELECTION"));
    assert!(summary.contains("Total Time:"));
    assert!(summary.contains("Phase Breakdown:"));

    println!("=== AutoModel Summary ===\n{}", summary);
}

#[test]
fn test_autobuilder_verbose_mode() {
    let df = create_linear_regression_dataset(200);

    let builder = AutoBuilder::new().with_verbose(true);

    // Should print progress (we can't easily capture it in tests, but verify it runs)
    let result = builder.fit(&df, "target");
    assert!(result.is_ok(), "Verbose mode should not break training");
}

#[test]
fn test_different_tuning_levels() {
    let df = create_linear_regression_dataset(150);

    // Quick
    let model_quick = auto_train_quick(&df, "target").expect("Quick should succeed");
    let time_quick = model_quick.build_time();

    // Standard
    let model_standard = auto_train(&df, "target").expect("Standard should succeed");
    let time_standard = model_standard.build_time();

    // Thorough
    let model_thorough = auto_train_thorough(&df, "target").expect("Thorough should succeed");
    let time_thorough = model_thorough.build_time();

    println!(
        "Build times - Quick: {:?}, Standard: {:?}, Thorough: {:?}",
        time_quick, time_standard, time_thorough
    );

    // All should produce valid predictions
    assert_eq!(model_quick.predict(&df).unwrap().len(), 150);
    assert_eq!(model_standard.predict(&df).unwrap().len(), 150);
    assert_eq!(model_thorough.predict(&df).unwrap().len(), 150);
}

#[test]
fn test_time_budget_control() {
    use std::time::Duration;

    let df = create_linear_regression_dataset(300);

    // Set tight time budget (15 seconds)
    let builder = AutoBuilder::new()
        .with_time_budget(Duration::from_secs(15))
        .with_verbose(true);

    let start = std::time::Instant::now();
    let result = builder
        .fit(&df, "target")
        .expect("Training with time budget should succeed");
    let elapsed = start.elapsed();

    println!("Training with 15s budget completed in {:?}", elapsed);

    // Should complete within budget (with some tolerance)
    assert!(
        elapsed < Duration::from_secs(20),
        "Training took {:?}, exceeding budget tolerance",
        elapsed
    );

    // Should still produce a valid model
    let model = treeboost::AutoModel::from_build_result(result);
    let predictions = model.predict(&df).expect("Prediction should succeed");
    assert_eq!(predictions.len(), 300);
}

#[test]
fn test_time_budget_adaptations() {
    use std::time::Duration;

    let df = create_linear_regression_dataset(200);

    // Very tight budget - should skip features and tuning
    let builder = AutoBuilder::new()
        .with_time_budget(Duration::from_secs(5))
        .with_verbose(true)
        .with_auto_features(true) // Request features
        .with_tuning(TuningLevel::Standard); // Request tuning

    let result = builder
        .fit(&df, "target")
        .expect("Training with tight budget should succeed");

    // With tight budget, features and tuning may be skipped
    // Just verify it completed and works
    let model = treeboost::AutoModel::from_build_result(result);
    assert_eq!(model.predict(&df).unwrap().len(), 200);
    println!("Tight budget model trained successfully");
}

#[test]
fn test_time_budget_via_config() {
    use std::time::Duration;

    let df = create_linear_regression_dataset(150);

    let config = AutoConfig::default()
        .with_time_budget(Duration::from_secs(20))
        .with_tuning(TuningLevel::Thorough); // Request thorough but budget will adapt it

    let builder = AutoBuilder::with_config(config);
    let result = builder.fit(&df, "target").expect("Training should succeed");

    let model = treeboost::AutoModel::from_build_result(result);
    assert_eq!(model.predict(&df).unwrap().len(), 150);
    println!("Time budget via config works");
}

// =========================================================================
// Incremental Learning Tests
// =========================================================================

#[test]
fn test_auto_model_incremental_update() {
    use treeboost::AutoModel;

    // Create training data
    let df1 = create_linear_regression_dataset(200);

    // Train initial model with quick settings
    let mut model = AutoModel::train_quick(&df1, "target").expect("Initial training should succeed");
    let trees_before = model.num_trees();
    assert!(trees_before > 0, "Model should have trees after initial training");

    // Create update data (same schema)
    let df2 = create_linear_regression_dataset(100);

    // Update the model with 5 additional trees
    let report = model.update(&df2, 5).expect("Update should succeed");

    assert_eq!(report.trees_before, trees_before);
    assert_eq!(report.trees_added, 5);
    assert_eq!(report.trees_after, trees_before + 5);
    assert_eq!(report.rows_trained, 100);
    assert_eq!(report.target_column, "target");

    // Verify model still works
    let predictions = model.predict(&df1).expect("Prediction should work after update");
    assert_eq!(predictions.len(), 200);

    println!("Incremental update report: {}", report);
}

#[test]
fn test_auto_model_trb_save_load() {
    use treeboost::AutoModel;

    let df = create_linear_regression_dataset(200);

    // Train model
    let model = AutoModel::train_quick(&df, "target").expect("Training should succeed");
    let trees_before = model.num_trees();

    // Save to TRB format
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("model.trb");

    model.save_trb(&path, "Test model").expect("Save should succeed");
    assert!(path.exists(), "TRB file should exist");

    // Load from TRB
    let loaded = AutoModel::load_trb(&path, "target").expect("Load should succeed");

    // Verify tree count matches
    assert_eq!(loaded.num_trees(), trees_before, "Loaded model should have same number of trees");
    assert_eq!(loaded.mode(), model.mode(), "Loaded model should have same mode");

    // Note: predict() on DataFrame requires pipeline_state which isn't preserved in TRB format
    // Use predict_binned() with a BinnedDataset for TRB-loaded models, or reload the original model
}

#[test]
fn test_auto_model_trb_update_cycle() {
    use treeboost::AutoModel;

    let df1 = create_linear_regression_dataset(200);
    let df2 = create_linear_regression_dataset(100);

    // Train initial model
    let mut model = AutoModel::train_quick(&df1, "target").expect("Training should succeed");
    let trees_after_initial = model.num_trees();

    // Verify initial prediction works
    let initial_preds = model.predict(&df1).expect("Initial prediction should work");
    assert_eq!(initial_preds.len(), 200);

    // Save to TRB
    let dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("model.trb");
    model.save_trb(&path, "Initial training").expect("Save should succeed");

    let initial_size = std::fs::metadata(&path).unwrap().len();

    // Update model
    model.update(&df2, 5).expect("Update should succeed");
    let trees_after_update = model.num_trees();
    assert_eq!(trees_after_update, trees_after_initial + 5);

    // Verify updated model still works for prediction
    let updated_preds = model.predict(&df1).expect("Updated model prediction should work");
    assert_eq!(updated_preds.len(), 200);

    // Save update
    model
        .save_trb_update(&path, 100, "Update batch")
        .expect("Save update should succeed");

    // File should be larger
    let updated_size = std::fs::metadata(&path).unwrap().len();
    assert!(
        updated_size > initial_size,
        "TRB file should grow after update"
    );

    // Load and verify tree count
    let loaded = AutoModel::load_trb(&path, "target").expect("Load should succeed");
    assert_eq!(loaded.num_trees(), trees_after_update);

    // Note: predict() on DataFrame requires pipeline_state which isn't preserved in TRB format
    // The in-memory model (before save) works fine for predictions
}

#[test]
fn test_auto_model_update_report_display() {
    use treeboost::AutoModelUpdateReport;

    let report = AutoModelUpdateReport {
        rows_trained: 1000,
        trees_before: 10,
        trees_after: 20,
        trees_added: 10,
        mode: BoostingMode::PureTree,
        target_column: "price".to_string(),
    };

    let display = format!("{}", report);
    assert!(display.contains("1000 rows"));
    assert!(display.contains("price"));
    assert!(display.contains("10 trees added"));
}

#[test]
fn test_auto_model_config_preserved_across_updates() {
    use treeboost::AutoModel;

    let df = create_linear_regression_dataset(200);

    // Train with specific mode
    let mut model =
        AutoModel::train_with_mode(&df, "target", BoostingMode::PureTree).expect("Training should succeed");

    let original_mode = model.mode();
    assert_eq!(original_mode, BoostingMode::PureTree);

    // Update should preserve mode
    model.update(&df, 5).expect("Update should succeed");
    assert_eq!(model.mode(), original_mode, "Mode should be preserved after update");
}
