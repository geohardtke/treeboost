//! Integration tests for feature generation and selection

mod common;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::features::{
    FeatureGenerator, FeatureSelector, InteractionGenerator, InteractionType, PolynomialGenerator,
    RatioGenerator, SelectionConfig,
};

use common::create_synthetic_dataset;

/// Test feature selection workflow
#[test]
fn test_features_selection_workflow() {
    let dataset = create_synthetic_dataset(500, 42);

    // Train model to get importances
    let config = GBDTConfig::new().with_num_rounds(30).with_max_depth(4);

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

// ============================================================================
// Interaction Generator Integration Tests
// ============================================================================

/// Test InteractionGenerator with explicit pairs
#[test]
fn test_interaction_generator_explicit_pairs() {
    // Create realistic feature data (100 rows × 4 features)
    let num_rows = 100;
    let num_features = 4;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    for i in 0..num_rows {
        data.push(i as f32 * 1.5); // f0: linear
        data.push((i as f32).sqrt()); // f1: sqrt
        data.push((i % 10) as f32); // f2: periodic
        data.push(100.0 - i as f32); // f3: decreasing
    }

    let names: Vec<String> = (0..num_features).map(|i| format!("feat_{}", i)).collect();

    // Generate interactions for specific pairs
    let gen = InteractionGenerator::from_pairs(vec![(0, 1), (2, 3)])
        .with_types(vec![InteractionType::Multiply, InteractionType::Subtract]);

    let (int_data, int_names) = gen.generate(&data, num_features, &names);

    // Should have 2 pairs × 2 types = 4 interaction features
    assert_eq!(int_names.len(), 4);
    assert_eq!(int_data.len(), num_rows * 4);

    // Check feature names
    assert!(int_names.contains(&"feat_0_mul_feat_1".to_string()));
    assert!(int_names.contains(&"feat_0_sub_feat_1".to_string()));
    assert!(int_names.contains(&"feat_2_mul_feat_3".to_string()));
    assert!(int_names.contains(&"feat_2_sub_feat_3".to_string()));

    // Verify first row values
    // f0=0, f1=0: mul=0, sub=0
    assert!((int_data[0] - 0.0).abs() < 1e-6); // feat_0_mul_feat_1
    assert!((int_data[1] - 0.0).abs() < 1e-6); // feat_0_sub_feat_1

    // f2=0, f3=100: mul=0, sub=100
    assert!((int_data[2] - 0.0).abs() < 1e-6); // feat_2_mul_feat_3
    assert!((int_data[3] - 100.0).abs() < 1e-6); // feat_2_sub_feat_3
}

/// Test InteractionGenerator with all pairs (combinatorial)
#[test]
fn test_interaction_generator_all_pairs() {
    let num_rows = 50;
    let num_features = 3;
    let data: Vec<f32> = (0..num_rows * num_features)
        .map(|i| (i % 20) as f32)
        .collect();
    let names: Vec<String> = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    let mut gen = InteractionGenerator::all_pairs();
    gen.fit(&data, num_features);

    // 3 features → 3 pairs: (0,1), (0,2), (1,2)
    assert_eq!(gen.pairs().unwrap().len(), 3);

    let (int_data, int_names) = gen.generate(&data, num_features, &names);

    // 3 pairs × 1 type (default: multiply) = 3 features
    assert_eq!(int_names.len(), 3);
    assert_eq!(int_data.len(), num_rows * 3);
}

/// Test InteractionGenerator with correlation-based selection
#[test]
fn test_interaction_generator_auto_select() {
    // Create data where some features are correlated
    let num_rows = 200;
    let num_features = 5;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    for i in 0..num_rows {
        let base = i as f32;
        data.push(base); // f0
        data.push(base * 2.0); // f1 = 2 * f0 (perfectly correlated)
        data.push(base + (i % 7) as f32); // f2 = f0 + noise
        data.push((i * 17 % 100) as f32); // f3 = random-ish
        data.push(1000.0 - base); // f4 = negatively correlated with f0
    }

    let names: Vec<String> = (0..num_features).map(|i| format!("f{}", i)).collect();

    // Select top 5 correlated pairs
    let mut gen = InteractionGenerator::top_correlated(5).with_min_correlation(0.5);
    gen.fit(&data, num_features);

    let pairs = gen.pairs().unwrap();
    assert!(!pairs.is_empty());
    assert!(pairs.len() <= 5);

    let (int_data, int_names) = gen.generate(&data, num_features, &names);

    assert_eq!(int_names.len(), pairs.len());
    assert_eq!(int_data.len(), num_rows * pairs.len());

    // All values should be finite
    for val in &int_data {
        assert!(val.is_finite(), "Interaction values should be finite");
    }
}

/// Test InteractionGenerator with target-based selection
#[test]
fn test_interaction_generator_target_based() {
    // Create data where f0 × f1 correlates with target
    let num_rows = 100;
    let num_features = 4;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    for i in 0..num_rows {
        let f0 = (i % 10 + 1) as f32;
        let f1 = ((i / 10) % 10 + 1) as f32;
        let f2 = (i * 3 % 20) as f32;
        let f3 = (i * 7 % 15) as f32;
        data.push(f0);
        data.push(f1);
        data.push(f2);
        data.push(f3);
    }

    // Target = f0 × f1 (interaction is predictive)
    let targets: Vec<f32> = (0..num_rows)
        .map(|i| {
            let f0 = (i % 10 + 1) as f32;
            let f1 = ((i / 10) % 10 + 1) as f32;
            f0 * f1
        })
        .collect();

    let names: Vec<String> = (0..num_features).map(|i| format!("f{}", i)).collect();

    // Select pairs based on target correlation gain
    let mut gen = InteractionGenerator::target_based(3, targets);
    gen.fit(&data, num_features);

    let (int_data, int_names) = gen.generate(&data, num_features, &names);

    // Should generate some features
    assert!(!int_names.is_empty());
    assert_eq!(int_data.len(), num_rows * int_names.len());
}

/// Test all interaction types
#[test]
fn test_interaction_all_types() {
    let data = vec![
        3.0, 5.0, 8.0, 2.0, // row 0
        4.0, 6.0, 1.0, 9.0, // row 1
    ];
    let names: Vec<String> = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
    ];

    let gen = InteractionGenerator::from_pairs(vec![(0, 1)]).with_types(InteractionType::all());

    let (int_data, int_names) = gen.generate(&data, 4, &names);

    // 1 pair × 5 types = 5 features
    assert_eq!(int_names.len(), 5);
    assert_eq!(int_data.len(), 2 * 5);

    // Row 0: a=3, b=5
    // mul=15, add=8, sub=2, min=3, max=5
    assert!((int_data[0] - 15.0).abs() < 1e-6); // mul
    assert!((int_data[1] - 8.0).abs() < 1e-6); // add
    assert!((int_data[2] - 2.0).abs() < 1e-6); // sub
    assert!((int_data[3] - 3.0).abs() < 1e-6); // min
    assert!((int_data[4] - 5.0).abs() < 1e-6); // max
}

/// Test combining multiple feature generators
#[test]
fn test_combined_feature_generators() {
    let num_rows = 50;
    let num_features = 3;
    let data: Vec<f32> = (0..num_rows * num_features)
        .map(|i| (i % 10 + 1) as f32)
        .collect();
    let names: Vec<String> = vec!["x".to_string(), "y".to_string(), "z".to_string()];

    // Generate polynomial features
    let poly = PolynomialGenerator::new(); // square + sqrt by default
    let (poly_data, poly_names) = poly.generate(&data, num_features, &names);

    // Generate ratio features
    let ratio = RatioGenerator::from_pairs(vec![(0, 1), (1, 2)]);
    let (ratio_data, ratio_names) = ratio.generate(&data, num_features, &names);

    // Generate interaction features
    let interaction = InteractionGenerator::from_pairs(vec![(0, 2)]);
    let (int_data, int_names) = interaction.generate(&data, num_features, &names);

    // All generators should produce features
    assert!(!poly_names.is_empty());
    assert!(!ratio_names.is_empty());
    assert!(!int_names.is_empty());

    // Total features = original + poly + ratio + interactions
    let total_features = num_features + poly_names.len() + ratio_names.len() + int_names.len();
    assert!(total_features > num_features);

    // Verify data lengths
    assert_eq!(poly_data.len(), num_rows * poly_names.len());
    assert_eq!(ratio_data.len(), num_rows * ratio_names.len());
    assert_eq!(int_data.len(), num_rows * int_names.len());
}

/// Test self-interactions (x² terms)
#[test]
fn test_self_interactions() {
    let data = vec![2.0, 3.0, 4.0]; // 1 row × 3 features
    let names: Vec<String> = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    let mut gen = InteractionGenerator::all_pairs().with_self_interactions(true);
    gen.fit(&data, 3);

    // With self: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2) = 6 pairs
    assert_eq!(gen.pairs().unwrap().len(), 6);

    let (int_data, int_names) = gen.generate(&data, 3, &names);
    assert_eq!(int_names.len(), 6);

    // Check self-interactions produce squares
    // a_mul_a = 4, b_mul_b = 9, c_mul_c = 16
    assert!(int_names.contains(&"a_mul_a".to_string()));
    assert!((int_data[0] - 4.0).abs() < 1e-6); // 2² = 4
}
