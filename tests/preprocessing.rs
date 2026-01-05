//! Integration tests for preprocessing module

use treeboost::preprocessing::{
    FrequencyEncoder, ImputeStrategy, LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler,
    Scaler, SimpleImputer, StandardScaler, UnknownStrategy,
};

/// Test StandardScaler end-to-end workflow
#[test]
fn test_preprocessing_standard_scaler_workflow() {
    // Create row-major data: 100 rows × 3 features
    let num_rows = 100;
    let num_features = 3;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    for i in 0..num_rows {
        data.push(i as f32 * 10.0); // f0: 0, 10, 20, ...
        data.push((i as f32).powf(2.0)); // f1: 0, 1, 4, 9, ...
        data.push(1000.0 + i as f32 * 0.1); // f2: 1000.0, 1000.1, ...
    }

    let mut scaler = StandardScaler::new();

    // Fit on data
    scaler.fit(&data, num_features).expect("Fit should succeed");
    assert!(scaler.is_fitted());

    // Transform
    let mut transformed = data.clone();
    scaler
        .transform(&mut transformed, num_features)
        .expect("Transform should succeed");

    // Verify: Each feature should have mean ≈ 0, std ≈ 1
    for f in 0..num_features {
        let feature_vals: Vec<f32> = (0..num_rows)
            .map(|r| transformed[r * num_features + f])
            .collect();

        let mean: f32 = feature_vals.iter().sum::<f32>() / num_rows as f32;
        let var: f32 =
            feature_vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / num_rows as f32;
        let std = var.sqrt();

        assert!(
            mean.abs() < 0.01,
            "Feature {} mean should be ~0, got {}",
            f,
            mean
        );
        assert!(
            (std - 1.0).abs() < 0.1,
            "Feature {} std should be ~1, got {}",
            f,
            std
        );
    }
}

/// Test MinMaxScaler range normalization
#[test]
fn test_preprocessing_minmax_scaler() {
    let num_rows = 50;
    let num_features = 2;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    for i in 0..num_rows {
        data.push(i as f32); // f0: 0 to 49
        data.push(100.0 - i as f32); // f1: 100 to 51
    }

    let mut scaler = MinMaxScaler::new();
    scaler.fit(&data, num_features).expect("Fit should succeed");

    let mut transformed = data.clone();
    scaler
        .transform(&mut transformed, num_features)
        .expect("Transform should succeed");

    // Verify all values in [0, 1]
    for &v in &transformed {
        assert!(v >= 0.0 && v <= 1.0, "Value {} not in [0, 1]", v);
    }

    // Verify min and max values
    for f in 0..num_features {
        let feature_vals: Vec<f32> = (0..num_rows)
            .map(|r| transformed[r * num_features + f])
            .collect();

        let min = feature_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = feature_vals
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            min.abs() < 0.001,
            "Feature {} min should be 0, got {}",
            f,
            min
        );
        assert!(
            (max - 1.0).abs() < 0.001,
            "Feature {} max should be 1, got {}",
            f,
            max
        );
    }
}

/// Test RobustScaler with outliers
#[test]
fn test_preprocessing_robust_scaler_outliers() {
    let num_rows = 100;
    let num_features = 1;
    let mut data: Vec<f32> = Vec::with_capacity(num_rows * num_features);

    // Normal data with outliers
    for i in 0..num_rows {
        if i < 95 {
            data.push(i as f32 % 10.0); // Values 0-9
        } else {
            data.push(10000.0); // Outliers
        }
    }

    let mut robust = RobustScaler::new();
    robust.fit(&data, num_features).expect("Fit should succeed");

    let mut transformed = data.clone();
    robust
        .transform(&mut transformed, num_features)
        .expect("Transform should succeed");

    // Non-outlier values should be in reasonable range
    let non_outlier_vals: Vec<f32> = transformed[0..95].to_vec();
    let max_non_outlier = non_outlier_vals
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_non_outlier = non_outlier_vals.iter().cloned().fold(f32::INFINITY, f32::min);

    // Robust scaling should keep non-outliers in reasonable range
    assert!(
        max_non_outlier < 5.0,
        "Non-outliers should be scaled reasonably: {}",
        max_non_outlier
    );
    assert!(
        min_non_outlier > -5.0,
        "Non-outliers should be scaled reasonably: {}",
        min_non_outlier
    );
}

/// Test FrequencyEncoder with transform workflow
#[test]
fn test_preprocessing_frequency_encoder_workflow() {
    let categories = vec!["apple", "banana", "apple", "cherry", "apple", "banana"];

    let mut encoder = FrequencyEncoder::new().with_normalize(true);
    encoder.fit(&categories);

    assert!(encoder.is_fitted());
    assert_eq!(encoder.num_categories(), 3);

    let transformed = encoder
        .transform(&["apple", "banana", "cherry"])
        .expect("Transform should succeed");

    // apple: 3/6 = 0.5, banana: 2/6 = 0.333, cherry: 1/6 = 0.167
    assert!((transformed[0] - 0.5).abs() < 0.01);
    assert!((transformed[1] - 0.333).abs() < 0.01);
    assert!((transformed[2] - 0.167).abs() < 0.01);

    // Unknown category with default value
    let unknown = encoder.transform_single("mango");
    assert_eq!(unknown, Some(0.0), "Unknown category should return default");
}

/// Test LabelEncoder with inverse transform
#[test]
fn test_preprocessing_label_encoder_roundtrip() {
    let categories = vec!["red", "green", "blue", "red", "green"];

    let mut encoder = LabelEncoder::new();
    encoder.fit(&categories);

    let labels = encoder
        .transform(&categories)
        .expect("Transform should succeed");
    let reversed = encoder
        .inverse_transform(&labels)
        .expect("Inverse should succeed");

    assert_eq!(reversed, categories, "Roundtrip should preserve categories");

    // Verify alphabetical ordering
    assert_eq!(encoder.get_label("blue"), Some(0));
    assert_eq!(encoder.get_label("green"), Some(1));
    assert_eq!(encoder.get_label("red"), Some(2));
}

/// Test OneHotEncoder with drop_first for linear models
#[test]
fn test_preprocessing_onehot_encoder_drop_first() {
    let categories = vec!["A", "B", "C"];

    let mut encoder = OneHotEncoder::new()
        .with_drop_first(true)
        .with_unknown_strategy(UnknownStrategy::AllZeros);
    encoder.fit(&categories);

    // 3 categories with drop_first → 2 columns
    assert_eq!(encoder.num_columns(), 2);

    let encoded = encoder
        .transform(&["A", "B", "C", "unknown"])
        .expect("Transform should succeed");

    // A (dropped): [0, 0]
    // B: [1, 0]
    // C: [0, 1]
    // unknown: [0, 0]
    assert_eq!(
        encoded,
        vec![
            0.0, 0.0, // A (reference)
            1.0, 0.0, // B
            0.0, 1.0, // C
            0.0, 0.0, // unknown
        ]
    );
}

/// Test SimpleImputer with different strategies
#[test]
fn test_preprocessing_simple_imputer_strategies() {
    let num_features = 2;

    // Data with NaN values (row-major: 10 rows × 2 features)
    let data = vec![
        1.0,
        10.0,
        2.0,
        f32::NAN,
        f32::NAN,
        30.0,
        4.0,
        40.0,
        5.0,
        50.0,
        6.0,
        60.0,
        7.0,
        70.0,
        8.0,
        80.0,
        9.0,
        90.0,
        10.0,
        100.0,
    ];

    // Mean imputation
    let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
    imputer.fit(&data, num_features).expect("Fit should succeed");

    let mut imputed = data.clone();
    imputer
        .transform(&mut imputed, num_features)
        .expect("Transform should succeed");

    // No NaN values should remain
    for &v in &imputed {
        assert!(!v.is_nan(), "NaN should be imputed");
    }

    // Feature 0 mean: (1+2+4+5+6+7+8+9+10)/9 ≈ 5.78
    // Feature 1 mean: (10+30+40+50+60+70+80+90+100)/9 ≈ 58.89
    assert!(
        (imputed[4] - 5.78).abs() < 0.1,
        "Row 2, f0 should be imputed to mean"
    );
    assert!(
        (imputed[3] - 58.89).abs() < 0.1,
        "Row 1, f1 should be imputed to mean"
    );
}
