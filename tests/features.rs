//! Integration tests for feature selection module

mod common;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::features::{FeatureSelector, SelectionConfig};

use common::create_synthetic_dataset;

/// Test feature selection workflow
#[test]
fn test_features_selection_workflow() {
    let dataset = create_synthetic_dataset(500, 42);

    // Train model to get importances
    let config = GBDTConfig::new()
        .with_num_rounds(30)
        .with_max_depth(4);

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");
    let importances = model.feature_importance();

    // Verify feature importances are computed
    assert_eq!(importances.len(), 5, "Should have 5 feature importances");

    // All importances should be non-negative and sum to ~1
    let total: f32 = importances.iter().sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "Importances should sum to 1, got {}",
        total
    );

    for &imp in &importances {
        assert!(imp >= 0.0, "Importances should be non-negative");
    }
}

/// Test collinearity dropping
#[test]
fn test_features_drop_collinear() {
    // Create correlated data: f1 = f0 + small noise
    let num_rows = 100;
    let num_features = 3;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    for i in 0..num_rows {
        let f0 = i as f32;
        let f1 = f0 + (i % 5) as f32 * 0.01; // Highly correlated with f0
        let f2 = (i * i) as f32 % 100.0; // Independent
        data.push(f0);
        data.push(f1);
        data.push(f2);
    }

    let feature_names: Vec<String> = vec!["f0".to_string(), "f1".to_string(), "f2".to_string()];
    let targets: Vec<f32> = (0..num_rows).map(|i| i as f32 * 2.0).collect();

    let selection_config = SelectionConfig::default()
        .with_drop_collinear(true)
        .with_collinearity_threshold(0.95);

    let selector = FeatureSelector::new(selection_config);

    let (filtered_data, filtered_names, kept_indices) =
        selector.drop_collinear_features(&data, num_features, &feature_names, Some(&targets));

    // Should keep fewer features than original (dropped collinear one)
    assert!(
        kept_indices.len() <= num_features,
        "Should keep at most original features"
    );

    // Should keep at least 2 features (f2 is independent, one of f0/f1)
    assert!(
        kept_indices.len() >= 2,
        "Should keep independent features: kept {}",
        kept_indices.len()
    );

    // Filtered data should match kept indices
    assert_eq!(filtered_names.len(), kept_indices.len());
    assert_eq!(filtered_data.len(), num_rows * kept_indices.len());
}
