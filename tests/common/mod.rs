//! Shared test utilities for TreeBoost integration tests

use treeboost::dataset::{BinnedDataset, FeatureInfo, FeatureType};

/// Create a synthetic regression dataset for testing
///
/// Generates deterministic pseudo-random data using a seed for reproducibility.
/// Target: y = f0 * 10 + f1 * 5 + noise
pub fn create_synthetic_dataset(n: usize, seed: u64) -> BinnedDataset {
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

/// Create a binary classification dataset for testing
///
/// Target: 1 if f0 + f1 > 1, else 0
pub fn create_binary_classification_dataset(n: usize, seed: u64) -> BinnedDataset {
    let num_features = 5;

    let mut state = seed;
    let mut next_rand = || -> f32 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((state >> 16) & 0x7FFF) as f32 / 32767.0
    };

    let mut features = Vec::with_capacity(n * num_features);
    for _f in 0..num_features {
        for _r in 0..n {
            features.push((next_rand() * 255.0) as u8);
        }
    }

    // Binary targets based on feature values
    let targets: Vec<f32> = (0..n)
        .map(|i| {
            let f0 = features[i] as f32 / 255.0;
            let f1 = features[n + i] as f32 / 255.0;
            if f0 + f1 > 1.0 {
                1.0
            } else {
                0.0
            }
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
