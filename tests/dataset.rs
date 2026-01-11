//! Dataset tests for multi-output 2D target ingestion

use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

fn create_test_feature_info(name: &str, num_bins: u8) -> FeatureInfo {
    FeatureInfo {
        name: name.to_string(),
        feature_type: FeatureType::Numeric,
        num_bins,
        bin_boundaries: vec![],
    }
}

/// Test 2D target ingestion with multiple output columns
///
/// Verifies that:
/// 1. `num_target_cols` is correctly stored
/// 2. Targets are flattened row-wise: [row0_label0, row0_label1, ..., row1_label0, ...]
/// 3. `get_target(row, label_idx)` accessor works correctly
/// 4. Validation: targets.len() == num_rows * num_target_cols
#[test]
fn test_dataset_2d_ingestion() {
    let num_rows = 4;
    let num_features = 2;
    let num_target_cols = 3;

    // Column-major features: feature 0 = [0,1,2,3], feature 1 = [10,11,12,13]
    let features = vec![0u8, 1, 2, 3, 10, 11, 12, 13];

    // Row-wise flattened targets: 4 rows × 3 labels
    // Row 0: [1.0, 0.0, 1.0]
    // Row 1: [0.0, 1.0, 0.0]
    // Row 2: [1.0, 1.0, 0.0]
    // Row 3: [0.0, 0.0, 1.0]
    let targets = vec![
        1.0, 0.0, 1.0, // row 0
        0.0, 1.0, 0.0, // row 1
        1.0, 1.0, 0.0, // row 2
        0.0, 0.0, 1.0, // row 3
    ];

    let feature_info = vec![
        create_test_feature_info("f0", 4),
        create_test_feature_info("f1", 14),
    ];

    let dataset =
        BinnedDataset::new_multioutput(num_rows, features, targets, feature_info, num_target_cols);

    // Verify basic properties
    assert_eq!(dataset.num_rows(), num_rows);
    assert_eq!(dataset.num_features(), num_features);
    assert_eq!(dataset.num_target_cols(), num_target_cols);
    assert!(dataset.is_multioutput());

    // Verify get_target accessor
    // Row 0: [1.0, 0.0, 1.0]
    assert_eq!(dataset.get_target(0, 0), 1.0);
    assert_eq!(dataset.get_target(0, 1), 0.0);
    assert_eq!(dataset.get_target(0, 2), 1.0);

    // Row 1: [0.0, 1.0, 0.0]
    assert_eq!(dataset.get_target(1, 0), 0.0);
    assert_eq!(dataset.get_target(1, 1), 1.0);
    assert_eq!(dataset.get_target(1, 2), 0.0);

    // Row 2: [1.0, 1.0, 0.0]
    assert_eq!(dataset.get_target(2, 0), 1.0);
    assert_eq!(dataset.get_target(2, 1), 1.0);
    assert_eq!(dataset.get_target(2, 2), 0.0);

    // Row 3: [0.0, 0.0, 1.0]
    assert_eq!(dataset.get_target(3, 0), 0.0);
    assert_eq!(dataset.get_target(3, 1), 0.0);
    assert_eq!(dataset.get_target(3, 2), 1.0);

    // Verify get_targets_row returns correct row values
    assert_eq!(dataset.get_targets_row(0), &[1.0, 0.0, 1.0]);
    assert_eq!(dataset.get_targets_row(1), &[0.0, 1.0, 0.0]);
    assert_eq!(dataset.get_targets_row(2), &[1.0, 1.0, 0.0]);
    assert_eq!(dataset.get_targets_row(3), &[0.0, 0.0, 1.0]);
}

/// Test backward compatibility with scalar targets
#[test]
fn test_dataset_scalar_compatibility() {
    let num_rows = 4;
    let features = vec![0u8, 1, 2, 3];
    let targets = vec![1.0, 2.0, 3.0, 4.0];
    let feature_info = vec![create_test_feature_info("f0", 4)];

    // Old-style constructor should create single-output dataset
    let dataset = BinnedDataset::new(num_rows, features, targets, feature_info);

    assert_eq!(dataset.num_target_cols(), 1);
    assert!(!dataset.is_multioutput());

    // Legacy target accessor should still work
    assert_eq!(dataset.target(0), 1.0);
    assert_eq!(dataset.target(1), 2.0);
    assert_eq!(dataset.targets(), &[1.0, 2.0, 3.0, 4.0]);

    // Multi-output accessor should also work
    assert_eq!(dataset.get_target(0, 0), 1.0);
    assert_eq!(dataset.get_target(1, 0), 2.0);
}

/// Test that targets length validation works
#[test]
#[should_panic(expected = "targets length")]
fn test_dataset_targets_validation() {
    let num_rows = 4;
    let num_target_cols = 3;
    let features = vec![0u8, 1, 2, 3];

    // Wrong number of targets (should be 4*3=12, but providing 10)
    let targets = vec![1.0; 10];
    let feature_info = vec![create_test_feature_info("f0", 4)];

    // This should panic
    let _ =
        BinnedDataset::new_multioutput(num_rows, features, targets, feature_info, num_target_cols);
}

/// Test subset_by_indices preserves multi-output structure
#[test]
fn test_dataset_subset_multioutput() {
    let num_rows = 4;
    let num_target_cols = 2;

    let features = vec![0u8, 1, 2, 3, 10, 11, 12, 13];
    let targets = vec![
        1.0, 0.0, // row 0
        0.0, 1.0, // row 1
        1.0, 1.0, // row 2
        0.0, 0.0, // row 3
    ];
    let feature_info = vec![
        create_test_feature_info("f0", 4),
        create_test_feature_info("f1", 14),
    ];

    let dataset =
        BinnedDataset::new_multioutput(num_rows, features, targets, feature_info, num_target_cols);

    // Take indices [1, 3]
    let subset = dataset.subset_by_indices(&[1, 3]);

    assert_eq!(subset.num_rows(), 2);
    assert_eq!(subset.num_target_cols(), 2);

    // Row 0 of subset = original row 1
    assert_eq!(subset.get_target(0, 0), 0.0);
    assert_eq!(subset.get_target(0, 1), 1.0);

    // Row 1 of subset = original row 3
    assert_eq!(subset.get_target(1, 0), 0.0);
    assert_eq!(subset.get_target(1, 1), 0.0);
}

/// Test with_targets for multi-output datasets
#[test]
fn test_dataset_with_targets_multioutput() {
    let num_rows = 3;
    let num_target_cols = 2;

    let features = vec![0u8, 1, 2];
    let targets = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let feature_info = vec![create_test_feature_info("f0", 3)];

    let dataset =
        BinnedDataset::new_multioutput(num_rows, features, targets, feature_info, num_target_cols);

    // Replace with new targets
    let new_targets = vec![0.0, 1.0, 1.0, 0.0, 0.5, 0.5];
    let new_dataset = dataset.with_targets_multioutput(new_targets, num_target_cols);

    assert_eq!(new_dataset.num_target_cols(), 2);
    assert_eq!(new_dataset.get_target(0, 0), 0.0);
    assert_eq!(new_dataset.get_target(0, 1), 1.0);
    assert_eq!(new_dataset.get_target(2, 0), 0.5);
    assert_eq!(new_dataset.get_target(2, 1), 0.5);
}

/// Test get_targets_column for efficient column-wise access
#[test]
fn test_dataset_targets_column_access() {
    let num_rows = 3;
    let num_target_cols = 2;

    let features = vec![0u8, 1, 2];
    let targets = vec![
        1.0, 0.0, // row 0
        0.0, 1.0, // row 1
        1.0, 1.0, // row 2
    ];
    let feature_info = vec![create_test_feature_info("f0", 3)];

    let dataset =
        BinnedDataset::new_multioutput(num_rows, features, targets, feature_info, num_target_cols);

    // Get all values for label 0
    let col0: Vec<f32> = (0..num_rows).map(|r| dataset.get_target(r, 0)).collect();
    assert_eq!(col0, vec![1.0, 0.0, 1.0]);

    // Get all values for label 1
    let col1: Vec<f32> = (0..num_rows).map(|r| dataset.get_target(r, 1)).collect();
    assert_eq!(col1, vec![0.0, 1.0, 1.0]);
}
