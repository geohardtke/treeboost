//! Integration tests for monitoring module (distribution shift detection)

mod common;

use treeboost::monitoring::{AlertLevel, ShiftDetector};

use common::create_synthetic_dataset;

/// Test ShiftDetector with no drift
#[test]
fn test_monitoring_no_drift() {
    let train = create_synthetic_dataset(500, 42);

    let detector = ShiftDetector::from_dataset(&train).with_thresholds(0.1, 0.25);

    // Check same dataset - should have no drift
    let result = detector.check(&train);

    assert_eq!(
        result.alert,
        AlertLevel::None,
        "Same data should have no drift"
    );
    assert!(
        result.overall_score < 0.05,
        "Score should be very low for same data: {}",
        result.overall_score
    );
    assert!(
        result.drifted_features.is_empty(),
        "No features should be flagged"
    );
}

/// Test ShiftDetector detects distribution shift
#[test]
fn test_monitoring_detects_drift() {
    let train = create_synthetic_dataset(500, 42);

    // Create shifted dataset (different seed = different distribution)
    let shifted = create_synthetic_dataset(500, 999);

    let detector = ShiftDetector::from_dataset(&train).with_thresholds(0.05, 0.15);

    let result = detector.check(&shifted);

    // Should detect some drift
    assert!(
        result.overall_score > 0.0,
        "Should detect drift between different distributions"
    );
}

/// Test ShiftDetector feature scores are computed correctly
#[test]
fn test_monitoring_feature_scores() {
    let train = create_synthetic_dataset(500, 42);

    let detector = ShiftDetector::from_dataset(&train).with_thresholds(0.1, 0.25);

    // Check with same data - should have no drift
    let result = detector.check(&train);
    assert_eq!(
        result.alert,
        AlertLevel::None,
        "Same data should have no drift"
    );
    assert!(
        result.overall_score < 0.01,
        "Score for same data should be ~0"
    );

    // Verify feature scores are computed for all features
    assert_eq!(
        result.feature_scores.len(),
        5,
        "Should have scores for 5 features"
    );

    for (name, score) in &result.feature_scores {
        assert!(!name.is_empty(), "Feature name should not be empty");
        assert!(
            score.is_finite(),
            "Score should be finite for feature {}",
            name
        );
        assert!(
            *score >= 0.0,
            "Score should be non-negative for feature {}",
            name
        );
    }
}
