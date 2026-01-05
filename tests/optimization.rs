//! Integration tests for dataset optimization features
//!
//! Tests 4-bit packing and cache-aware column reordering

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::{
    AccessTracker, BinnedDataset, ColumnPermutation, FeatureInfo, FeatureType, PackedDataset,
    StorageMode,
};

/// Test 4-bit packed storage for low-cardinality features
#[test]
fn test_packed_dataset_memory_savings() {
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

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

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
    let importances = model.feature_importance();
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

    let model = GBDTModel::train_binned(&dataset, config).expect("Training should succeed");

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
