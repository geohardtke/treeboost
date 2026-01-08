use polars::prelude::*;
use treeboost::{
    dataset::feature_extractor::LinearFeatureConfig,
    learner::{LinearConfig, TreeConfig},
    model::{AutoConfig, AutoModel, BoostingMode, TuningLevel, UniversalConfig},
};

#[test]
fn test_ltt_pure_linear_data() {
    // Generate pure linear data: y = 2x + 3
    let x_values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let y_values: Vec<f64> = x_values.iter().map(|x| x * 2.0 + 3.0).collect();

    let df = DataFrame::new(vec![
        Series::new("x".into(), x_values.clone()).into(),
        Series::new("target".into(), y_values.clone()).into(),
    ])
    .unwrap();

    println!("\n=== Test: Pure Linear Data (y = 2x + 3) ===");
    println!(
        "Input DataFrame: {} rows × {} cols",
        df.height(),
        df.width()
    );

    // Configure with good hyperparameters for linear regression
    let linear_config = LinearConfig::ridge(0.01) // Very light regularization for pure linear data
        .with_shrinkage_factor(1.0) // Full step size
        .with_max_iter(500); // Many iterations for convergence

    let tree_config = TreeConfig::default()
        .with_max_depth(3)
        .with_min_samples_leaf(5); // No auto-stopping, train completely

    let univ_config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_linear_config(linear_config)
        .with_tree_config(tree_config)
        .with_num_rounds(50)
        .with_learning_rate(0.1);

    let config = AutoConfig::new()
        .with_auto_features(false)
        .with_tuning(TuningLevel::None) // No auto-tuning
        .with_custom_config(univ_config);

    let model = AutoModel::train_with_config(&df, "target", config).unwrap();

    let predictions = model.predict(&df).unwrap();

    // CRITICAL: Verify prediction dimensions match input
    assert_eq!(
        predictions.len(),
        df.height(),
        "Predictions length must match input DataFrame rows"
    );

    // Predictions should be close to actual values
    let rmse: f32 = predictions
        .iter()
        .zip(y_values.iter())
        .map(|(pred, actual)| (pred - *actual as f32).powi(2))
        .sum::<f32>()
        .sqrt()
        / predictions.len() as f32;

    println!("Predictions: {} rows", predictions.len());
    println!("Pure linear data RMSE: {:.4}", rmse);

    // For pure linear data, LTT should fit well
    // With y ranging from 3 to 201, RMSE < 5 is good (< 2.5% error)
    assert!(
        rmse < 5.0,
        "RMSE should be low for pure linear data, got {:.4}",
        rmse
    );
}

#[test]
fn test_ltt_linear_plus_residual() {
    // Generate data: y = 2x + sin(x)
    let x_values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let y_values: Vec<f64> = x_values.iter().map(|x| x * 2.0 + x.sin()).collect();

    let df = DataFrame::new(vec![
        Series::new("x".into(), x_values.clone()).into(),
        Series::new("target".into(), y_values.clone()).into(),
    ])
    .unwrap();

    println!("\n=== Test: Linear + Nonlinear (y = 2x + sin(x)) ===");
    println!(
        "Input DataFrame: {} rows × {} cols",
        df.height(),
        df.width()
    );

    // Configure with same good linear config as pure linear test
    // Linear should capture 2x just as well as test 1
    let linear_config = LinearConfig::ridge(0.01)
        .with_shrinkage_factor(1.0)
        .with_max_iter(500);

    // Trees need to fit sin(x) residual - use appropriate depth
    let tree_config = TreeConfig::default().with_max_depth(6);

    let univ_config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_linear_config(linear_config)
        .with_tree_config(tree_config)
        .with_num_rounds(200) // Enough rounds to fit sin(x)
        .with_learning_rate(0.1);

    let config = AutoConfig::new()
        .with_auto_features(false)
        .with_tuning(TuningLevel::None)
        .with_custom_config(univ_config);

    let model = AutoModel::train_with_config(&df, "target", config).unwrap();

    let predictions = model.predict(&df).unwrap();

    // CRITICAL: Verify prediction dimensions match input
    assert_eq!(
        predictions.len(),
        df.height(),
        "Predictions length must match input DataFrame rows"
    );

    let rmse: f32 = predictions
        .iter()
        .zip(y_values.iter())
        .map(|(pred, actual)| (pred - *actual as f32).powi(2))
        .sum::<f32>()
        .sqrt()
        / predictions.len() as f32;

    println!("Predictions: {} rows", predictions.len());
    println!("Linear + residual RMSE: {:.4}", rmse);

    // LTT should fit both linear and nonlinear components well
    // sin(x) has amplitude ~1, so RMSE < 0.5 is good
    assert!(
        rmse < 0.5,
        "RMSE should be low (LTT captures both linear trend and sin(x) residual), got {:.4}",
        rmse
    );
}

#[test]
fn test_ltt_with_categoricals() {
    // Generate data with numeric and categorical features
    let x_values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let cat_values: Vec<&str> = (0..100)
        .map(|i| match i % 4 {
            0 => "A",
            1 => "B",
            2 => "C",
            _ => "D",
        })
        .collect();
    let y_values: Vec<f64> = x_values.iter().map(|x| x * 2.0 + 3.0).collect();

    let df = DataFrame::new(vec![
        Series::new("x".into(), x_values.clone()).into(),
        Series::new("category".into(), cat_values).into(),
        Series::new("target".into(), y_values.clone()).into(),
    ])
    .unwrap();

    println!("\n=== Test: LTT with Categoricals ===");
    println!(
        "Input DataFrame: {} rows × {} cols (1 numeric, 1 categorical)",
        df.height(),
        df.width()
    );

    let config = AutoConfig::new()
        .with_mode(BoostingMode::LinearThenTree)
        .with_auto_features(false)
        .with_tuning(TuningLevel::None);

    let model = AutoModel::train_with_config(&df, "target", config).unwrap();

    let predictions = model.predict(&df).unwrap();

    // CRITICAL: Verify prediction dimensions match input
    assert_eq!(
        predictions.len(),
        df.height(),
        "Predictions length must match input DataFrame rows"
    );

    let rmse: f32 = predictions
        .iter()
        .zip(y_values.iter())
        .map(|(pred, actual)| (pred - *actual as f32).powi(2))
        .sum::<f32>()
        .sqrt()
        / predictions.len() as f32;

    println!("Predictions: {} rows", predictions.len());
    println!("LTT with categoricals RMSE: {:.4}", rmse);

    // Even with categoricals, should fit reasonably well
    assert!(
        rmse < 2.0,
        "RMSE should be low even with categorical features, got {:.4}",
        rmse
    );

    // Verify feature extractor stored the column type detection
    assert!(
        model.inner().feature_extractor().is_some(),
        "FeatureExtractor must be stored for LTT mode"
    );
}

#[test]
fn test_ltt_with_id_like_columns() {
    // Generate data with ID-like columns
    let x_values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let id_values: Vec<String> = (0..100).map(|i| format!("ID_{:04}", i)).collect();
    let y_values: Vec<f64> = x_values.iter().map(|x| x * 2.0 + 3.0).collect();

    let df = DataFrame::new(vec![
        Series::new("x".into(), x_values.clone()).into(),
        Series::new("id".into(), id_values).into(),
        Series::new("target".into(), y_values.clone()).into(),
    ])
    .unwrap();

    // Use same good config as first test
    let linear_config = LinearConfig::ridge(0.01)
        .with_shrinkage_factor(1.0)
        .with_max_iter(500);

    let tree_config = TreeConfig::default().with_max_depth(3);

    let univ_config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_linear_config(linear_config)
        .with_tree_config(tree_config)
        .with_num_rounds(50)
        .with_learning_rate(0.1);

    let config = AutoConfig::new()
        .with_auto_features(false)
        .with_custom_config(univ_config);

    let model = AutoModel::train_with_config(&df, "target", config).unwrap();

    let predictions = model.predict(&df).unwrap();

    let rmse: f32 = predictions
        .iter()
        .zip(y_values.iter())
        .map(|(pred, actual)| (pred - *actual as f32).powi(2))
        .sum::<f32>()
        .sqrt()
        / predictions.len() as f32;

    println!("LTT with ID-like columns RMSE: {:.4}", rmse);
    assert!(
        rmse < 5.0,
        "RMSE should be low, ID columns should be auto-excluded"
    );
}

#[test]
fn test_ltt_with_user_exclusions() {
    // Generate data
    let x_values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let corr_values: Vec<f64> = x_values.iter().map(|x| x * 3.0).collect();
    let y_values: Vec<f64> = x_values.iter().map(|x| x * 2.0 + 3.0).collect();

    let df = DataFrame::new(vec![
        Series::new("x".into(), x_values.clone()).into(),
        Series::new("correlated".into(), corr_values).into(),
        Series::new("target".into(), y_values.clone()).into(),
    ])
    .unwrap();

    // Configure linear features to exclude "correlated"
    let linear_config = LinearFeatureConfig::default().with_exclude_columns(&["correlated"]);

    let config = AutoConfig::new()
        .with_mode(BoostingMode::LinearThenTree)
        .with_linear_feature_config(linear_config)
        .with_tuning(TuningLevel::Quick);

    let model = AutoModel::train_with_config(&df, "target", config).unwrap();

    let predictions = model.predict(&df).unwrap();

    let rmse: f32 = predictions
        .iter()
        .zip(y_values.iter())
        .map(|(pred, actual)| (pred - *actual as f32).powi(2))
        .sum::<f32>()
        .sqrt()
        / predictions.len() as f32;

    println!("LTT with user exclusions RMSE: {:.4}", rmse);
    assert!(
        rmse < 2.0,
        "RMSE should be low with user-specified exclusions"
    );
}

#[test]
fn test_ltt_feature_extractor_storage() {
    // Generate data with mixed column types
    let x_values: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let cat_values: Vec<&str> = (0..50)
        .map(|i| match i % 3 {
            0 => "A",
            1 => "B",
            _ => "C",
        })
        .collect();
    let y_values: Vec<f64> = x_values.iter().map(|x| x * 2.0 + 3.0).collect();

    let df = DataFrame::new(vec![
        Series::new("x".into(), x_values.clone()).into(),
        Series::new("category".into(), cat_values).into(),
        Series::new("target".into(), y_values.clone()).into(),
    ])
    .unwrap();

    let config = AutoConfig::new()
        .with_mode(BoostingMode::LinearThenTree)
        .with_auto_features(false)
        .with_tuning(TuningLevel::None);

    let model = AutoModel::train_with_config(&df, "target", config).unwrap();

    // Verify feature extractor is stored
    assert!(
        model.inner().feature_extractor().is_some(),
        "FeatureExtractor should be stored in model"
    );

    let extractor = model.inner().feature_extractor().unwrap();
    println!("Feature extractor config: {:?}", extractor.config());
    println!(
        "Exclude categorical: {}",
        extractor.config().exclude_categorical
    );
    println!("Exclude ID: {}", extractor.config().exclude_id);

    println!("FeatureExtractor storage successful");
}

#[test]
fn test_ltt_with_pipeline_encoded_categoricals() {
    // This test simulates the real-world AutoBuilder flow:
    // 1. DataFrame with categoricals
    // 2. DataPipeline encodes them (target encoding)
    // 3. AutoBuilder extracts features from encoded DataFrame
    // 4. Feature counts should match between linear and tree

    use treeboost::dataset::DataPipeline;

    // Generate data with numeric and categorical features
    let x_values: Vec<f64> = (0..200).map(|i| i as f64 * 0.1).collect();
    let cat1_values: Vec<&str> = (0..200)
        .map(|i| match i % 4 {
            0 => "A",
            1 => "B",
            2 => "C",
            _ => "D",
        })
        .collect();
    let cat2_values: Vec<&str> = (0..200)
        .map(|i| match i % 3 {
            0 => "X",
            1 => "Y",
            _ => "Z",
        })
        .collect();
    // y depends on both x and categories
    let y_values: Vec<f64> = x_values
        .iter()
        .zip(cat1_values.iter())
        .map(|(x, &cat)| {
            let cat_effect = match cat {
                "A" => 10.0,
                "B" => 20.0,
                "C" => 30.0,
                _ => 40.0,
            };
            x * 2.0 + cat_effect
        })
        .collect();

    let df = DataFrame::new(vec![
        Series::new("x".into(), x_values.clone()).into(),
        Series::new("cat1".into(), cat1_values).into(),
        Series::new("cat2".into(), cat2_values).into(),
        Series::new("target".into(), y_values.clone()).into(),
    ])
    .unwrap();

    println!("\n=== Testing LTT with Pipeline-Encoded Categoricals ===");
    println!(
        "Original DataFrame: {} rows × {} cols",
        df.height(),
        df.width()
    );
    println!("Original dtypes:");
    for col in df.get_columns() {
        println!("  {} : {:?}", col.name(), col.dtype());
    }

    // Use AutoBuilder which internally uses DataPipeline
    // CRITICAL: No auto-tuning to ensure complete training
    let linear_config = LinearConfig::ridge(0.01)
        .with_shrinkage_factor(1.0)
        .with_max_iter(100);

    let tree_config = TreeConfig::default().with_max_depth(6);

    let univ_config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_linear_config(linear_config)
        .with_tree_config(tree_config)
        .with_num_rounds(100)
        .with_learning_rate(0.1);

    let config = AutoConfig::new()
        .with_auto_features(false)
        .with_tuning(TuningLevel::None) // No auto-tuning
        .with_verbose(false)
        .with_custom_config(univ_config);

    let model = AutoModel::train_with_config(&df, "target", config).unwrap();

    let predictions = model.predict(&df).unwrap();

    // CRITICAL: Verify prediction dimensions match input (this caught the bug!)
    assert_eq!(
        predictions.len(),
        df.height(),
        "Predictions length must match input DataFrame rows. \
         Mismatch indicates preprocessing inconsistency between train and predict."
    );

    let rmse: f32 = predictions
        .iter()
        .zip(y_values.iter())
        .map(|(pred, actual)| (pred - *actual as f32).powi(2))
        .sum::<f32>()
        .sqrt()
        / predictions.len() as f32;

    println!("Predictions: {} rows", predictions.len());
    println!("Pipeline-encoded categoricals RMSE: {:.4}", rmse);

    // Verify feature extractor captured the right features
    assert!(
        model.inner().feature_extractor().is_some(),
        "FeatureExtractor should be stored"
    );

    // With the bug fixed, RMSE should be reasonable (not >100)
    // This dataset has y = 2x + category_effect, so it's learnable
    assert!(
        rmse < 5.0,
        "RMSE too high - feature count mismatch or preprocessing issue. RMSE: {:.4}",
        rmse
    );

    println!("✓ LTT correctly handles pipeline-encoded categoricals");
}
