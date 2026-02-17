//! Cache-aware column reordering for improved memory access patterns
//!
//! During tree traversal, frequently accessed features should be stored
//! contiguously in memory to maximize CPU cache hits. This module provides
//! utilities to reorder columns based on feature importance or access frequency.

use crate::dataset::core::{BinnedDataset, FeatureInfo};
use rkyv::{Archive, Deserialize, Serialize};

/// Column ordering strategy
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
    Default,
)]
pub enum OrderingStrategy {
    /// Original ordering (no reordering)
    #[default]
    Original,
    /// Order by feature importance (most important first)
    ByImportance,
    /// Order by access frequency (most accessed first)
    ByAccessFrequency,
}

/// Tracks feature access patterns during tree traversal
#[derive(Debug, Clone)]
pub struct AccessTracker {
    /// Number of times each feature was accessed
    access_counts: Vec<u64>,
}

impl AccessTracker {
    /// Create a new access tracker for n features
    pub fn new(num_features: usize) -> Self {
        Self {
            access_counts: vec![0; num_features],
        }
    }

    /// Record an access to a feature
    #[inline]
    pub fn record(&mut self, feature_idx: usize) {
        if feature_idx < self.access_counts.len() {
            self.access_counts[feature_idx] += 1;
        }
    }

    /// Get access counts
    pub fn counts(&self) -> &[u64] {
        &self.access_counts
    }

    /// Reset all counts
    pub fn reset(&mut self) {
        self.access_counts.fill(0);
    }

    /// Compute optimal ordering (most accessed first)
    pub fn optimal_order(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.access_counts.len()).collect();
        indices.sort_by(|&a, &b| self.access_counts[b].cmp(&self.access_counts[a]));
        indices
    }
}

/// Column permutation mapping
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct ColumnPermutation {
    /// Maps new index -> original index
    new_to_original: Vec<usize>,
    /// Maps original index -> new index
    original_to_new: Vec<usize>,
}

impl ColumnPermutation {
    /// Create identity permutation (no reordering)
    pub fn identity(num_features: usize) -> Self {
        let indices: Vec<usize> = (0..num_features).collect();
        Self {
            new_to_original: indices.clone(),
            original_to_new: indices,
        }
    }

    /// Create permutation from importance scores (higher = more important)
    pub fn from_importances(importances: &[f32]) -> Self {
        let mut indices: Vec<usize> = (0..importances.len()).collect();
        // Sort by importance descending (most important first for cache locality)
        indices.sort_by(|&a, &b| {
            importances[b]
                .partial_cmp(&importances[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut original_to_new = vec![0; importances.len()];
        for (new_idx, &orig_idx) in indices.iter().enumerate() {
            original_to_new[orig_idx] = new_idx;
        }

        Self {
            new_to_original: indices,
            original_to_new,
        }
    }

    /// Create permutation from access tracker
    pub fn from_access_tracker(tracker: &AccessTracker) -> Self {
        let new_to_original = tracker.optimal_order();
        let mut original_to_new = vec![0; new_to_original.len()];
        for (new_idx, &orig_idx) in new_to_original.iter().enumerate() {
            original_to_new[orig_idx] = new_idx;
        }

        Self {
            new_to_original,
            original_to_new,
        }
    }

    /// Map new index to original index
    #[inline]
    pub fn to_original(&self, new_idx: usize) -> usize {
        self.new_to_original[new_idx]
    }

    /// Map original index to new index
    #[inline]
    pub fn to_new(&self, original_idx: usize) -> usize {
        self.original_to_new[original_idx]
    }

    /// Number of features
    pub fn len(&self) -> usize {
        self.new_to_original.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.new_to_original.is_empty()
    }

    /// Check if this is the identity permutation
    pub fn is_identity(&self) -> bool {
        self.new_to_original
            .iter()
            .enumerate()
            .all(|(i, &orig)| i == orig)
    }

    /// Get the new-to-original mapping
    pub fn new_to_original(&self) -> &[usize] {
        &self.new_to_original
    }
}

/// Reorder dataset columns according to a permutation
pub fn reorder_dataset(dataset: &BinnedDataset, permutation: &ColumnPermutation) -> BinnedDataset {
    if permutation.is_identity() {
        return dataset.clone();
    }

    let num_rows = dataset.num_rows();
    let num_features = dataset.num_features();

    // Reorder feature data
    let mut new_features = Vec::with_capacity(num_rows * num_features);
    for new_idx in 0..num_features {
        let orig_idx = permutation.to_original(new_idx);
        let column = dataset.feature_column(orig_idx);
        new_features.extend_from_slice(column);
    }

    // Reorder feature info
    let new_feature_info: Vec<FeatureInfo> = (0..num_features)
        .map(|new_idx| {
            let orig_idx = permutation.to_original(new_idx);
            dataset.feature_info(orig_idx).clone()
        })
        .collect();

    BinnedDataset::new(
        num_rows,
        new_features,
        dataset.targets().to_vec(),
        new_feature_info,
    )
}

/// Builder for creating optimally ordered datasets
pub struct ReorderBuilder {
    strategy: OrderingStrategy,
    importances: Option<Vec<f32>>,
    access_tracker: Option<AccessTracker>,
}

impl Default for ReorderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ReorderBuilder {
    /// Create a new reorder builder
    pub fn new() -> Self {
        Self {
            strategy: OrderingStrategy::Original,
            importances: None,
            access_tracker: None,
        }
    }

    /// Set ordering strategy
    pub fn with_strategy(mut self, strategy: OrderingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set feature importances (for ByImportance strategy)
    pub fn with_importances(mut self, importances: Vec<f32>) -> Self {
        self.importances = Some(importances);
        self
    }

    /// Set access tracker (for ByAccessFrequency strategy)
    pub fn with_access_tracker(mut self, tracker: AccessTracker) -> Self {
        self.access_tracker = Some(tracker);
        self
    }

    /// Build permutation based on strategy
    pub fn build_permutation(&self, num_features: usize) -> ColumnPermutation {
        match self.strategy {
            OrderingStrategy::Original => ColumnPermutation::identity(num_features),
            OrderingStrategy::ByImportance => {
                if let Some(ref importances) = self.importances {
                    ColumnPermutation::from_importances(importances)
                } else {
                    ColumnPermutation::identity(num_features)
                }
            }
            OrderingStrategy::ByAccessFrequency => {
                if let Some(ref tracker) = self.access_tracker {
                    ColumnPermutation::from_access_tracker(tracker)
                } else {
                    ColumnPermutation::identity(num_features)
                }
            }
        }
    }

    /// Reorder a dataset
    pub fn reorder(&self, dataset: &BinnedDataset) -> BinnedDataset {
        let permutation = self.build_permutation(dataset.num_features());
        reorder_dataset(dataset, &permutation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::FeatureType;

    fn create_test_dataset() -> BinnedDataset {
        let num_rows = 4;
        // Column-major: f0=[0,1,2,3], f1=[10,11,12,13], f2=[20,21,22,23]
        let features = vec![0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let feature_info = vec![
            FeatureInfo {
                name: "f0".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 4,
                bin_boundaries: vec![],
                impute_value: 0.0,
            },
            FeatureInfo {
                name: "f1".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 14,
                bin_boundaries: vec![],
                impute_value: 0.0,
            },
            FeatureInfo {
                name: "f2".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 24,
                bin_boundaries: vec![],
                impute_value: 0.0,
            },
        ];

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_identity_permutation() {
        let perm = ColumnPermutation::identity(5);

        assert!(perm.is_identity());
        assert_eq!(perm.len(), 5);
        for i in 0..5 {
            assert_eq!(perm.to_original(i), i);
            assert_eq!(perm.to_new(i), i);
        }
    }

    #[test]
    fn test_permutation_from_importances() {
        // Features with importances: [0.1, 0.5, 0.3, 0.05, 0.05]
        // Expected order: 1, 2, 0, 3, 4 (descending by importance)
        let importances = vec![0.1, 0.5, 0.3, 0.05, 0.05];
        let perm = ColumnPermutation::from_importances(&importances);

        assert!(!perm.is_identity());
        assert_eq!(perm.to_original(0), 1); // Most important
        assert_eq!(perm.to_original(1), 2); // Second most
        assert_eq!(perm.to_original(2), 0); // Third
    }

    #[test]
    fn test_access_tracker() {
        let mut tracker = AccessTracker::new(3);

        tracker.record(0);
        tracker.record(2);
        tracker.record(2);
        tracker.record(2);
        tracker.record(1);
        tracker.record(1);

        assert_eq!(tracker.counts(), &[1, 2, 3]);

        let order = tracker.optimal_order();
        assert_eq!(order, vec![2, 1, 0]); // Most accessed first
    }

    #[test]
    fn test_reorder_dataset() {
        let dataset = create_test_dataset();
        let importances = vec![0.1, 0.5, 0.3]; // f1 most important, then f2, then f0
        let perm = ColumnPermutation::from_importances(&importances);

        let reordered = reorder_dataset(&dataset, &perm);

        // After reordering: new f0 = old f1, new f1 = old f2, new f2 = old f0
        assert_eq!(reordered.feature_column(0), &[10, 11, 12, 13]); // old f1
        assert_eq!(reordered.feature_column(1), &[20, 21, 22, 23]); // old f2
        assert_eq!(reordered.feature_column(2), &[0, 1, 2, 3]); // old f0

        // Feature names should also be reordered
        assert_eq!(reordered.feature_info(0).name, "f1");
        assert_eq!(reordered.feature_info(1).name, "f2");
        assert_eq!(reordered.feature_info(2).name, "f0");

        // Targets unchanged
        assert_eq!(reordered.targets(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reorder_builder() {
        let dataset = create_test_dataset();
        let importances = vec![0.3, 0.1, 0.6]; // f2 most important

        let reordered = ReorderBuilder::new()
            .with_strategy(OrderingStrategy::ByImportance)
            .with_importances(importances)
            .reorder(&dataset);

        // f2 should now be first
        assert_eq!(reordered.feature_info(0).name, "f2");
    }

    #[test]
    fn test_identity_reorder_no_copy() {
        let dataset = create_test_dataset();
        let perm = ColumnPermutation::identity(3);

        // Identity permutation should return clone without reordering
        let reordered = reorder_dataset(&dataset, &perm);
        assert_eq!(reordered.feature_column(0), dataset.feature_column(0));
        assert_eq!(reordered.feature_column(1), dataset.feature_column(1));
        assert_eq!(reordered.feature_column(2), dataset.feature_column(2));
    }
}
