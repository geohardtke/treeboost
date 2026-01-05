//! Weak learner abstractions for gradient boosting
//!
//! This module provides abstractions for different base learners that can be used
//! in gradient boosting ensembles:
//!
//! - **TreeBooster**: Decision tree weak learner (works on BinnedDataset)
//! - **LinearBooster**: Linear model with Coordinate Descent (works on raw features)
//!
//! # Architecture
//!
//! ```text
//! Booster (enum)
//!     ├── Tree(TreeBooster)     - Histogram-based decision tree
//!     └── Linear(LinearBooster) - Linear model with Ridge regularization
//!
//! WeakLearner (trait) - For raw-feature learners only
//!     └── LinearBooster
//! ```
//!
//! # Design Rationale
//!
//! TreeBooster and LinearBooster have different interfaces because:
//! - **Trees** work on binned data (histogram-based split finding)
//! - **Linear models** work on raw features (need actual values for regression)
//!
//! The `Booster` enum provides type-safe dispatch while respecting these differences.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::learner::{Booster, TreeConfig, LinearConfig};
//!
//! // Create boosters
//! let tree_booster = Booster::tree(TreeConfig::default());
//! let linear_booster = Booster::linear(10, LinearConfig::default());
//!
//! // Fit and predict (using appropriate data format for each type)
//! ```

mod linear;
mod traits;
mod tree;

pub use linear::{LinearBooster, LinearConfig};
pub use traits::WeakLearner;
pub use tree::{SerializableTreeBooster, TreeBooster, TreeConfig};

use crate::dataset::BinnedDataset;
use crate::{Result, TreeBoostError};
use rkyv::{Archive, Deserialize, Serialize};

/// Unified booster enum for gradient boosting
///
/// Provides type-safe dispatch between different learner types while
/// maintaining serialization compatibility.
///
/// # Design
///
/// Uses enum dispatch instead of trait objects for:
/// - Zero-cost abstraction (no vtable overhead)
/// - rkyv serialization compatibility
/// - Compile-time optimization opportunities
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum Booster {
    /// Decision tree weak learner
    ///
    /// Works on BinnedDataset. Best for:
    /// - Capturing non-linear relationships
    /// - Feature interactions
    /// - Most tabular data problems
    Tree(SerializableTreeBooster),

    /// Linear model with Ridge regularization
    ///
    /// Works on raw features. Best for:
    /// - Capturing global trends
    /// - Extrapolation beyond training range
    /// - Time-series with drift
    Linear(LinearBooster),
}

impl Booster {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a new tree booster
    pub fn tree(config: TreeConfig) -> Self {
        Self::Tree(SerializableTreeBooster {
            tree: None,
            config,
        })
    }

    /// Create a new linear booster
    pub fn linear(num_features: usize, config: LinearConfig) -> Self {
        Self::Linear(LinearBooster::new(num_features, config))
    }

    // =========================================================================
    // Fitting
    // =========================================================================

    /// Fit tree booster on binned data
    ///
    /// # Errors
    /// Returns `TreeBoostError::Config` if called on a Linear booster.
    pub fn fit_tree(
        &mut self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
        row_indices: Option<&[usize]>,
    ) -> Result<()> {
        match self {
            Self::Tree(ser) => {
                // Create TreeBooster from serializable form, fit, then store back
                let mut booster = TreeBooster::new(ser.config.clone());
                if let Some(tree) = ser.tree.take() {
                    booster.set_tree(tree);
                }
                booster.fit_on_gradients(dataset, gradients, hessians, row_indices)?;
                *ser = booster.into();
                Ok(())
            }
            Self::Linear(_) => Err(TreeBoostError::Config(
                "Cannot fit tree on Linear booster - use fit_linear instead".to_string(),
            )),
        }
    }

    /// Fit linear booster on raw features
    ///
    /// # Errors
    /// Returns `TreeBoostError::Config` if called on a Tree booster.
    pub fn fit_linear(
        &mut self,
        features: &[f32],
        num_features: usize,
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<()> {
        match self {
            Self::Linear(l) => l.fit_on_gradients(features, num_features, gradients, hessians),
            Self::Tree(_) => Err(TreeBoostError::Config(
                "Cannot fit linear on Tree booster - use fit_tree instead".to_string(),
            )),
        }
    }

    // =========================================================================
    // Prediction
    // =========================================================================

    /// Predict using tree booster on binned data
    ///
    /// Returns zero vector if called on Linear booster.
    pub fn predict_tree(&self, dataset: &BinnedDataset) -> Vec<f32> {
        match self {
            Self::Tree(ser) => {
                // Direct tree prediction avoids unnecessary TreeBooster allocation
                match &ser.tree {
                    Some(tree) => tree.predict_all(dataset),
                    None => vec![0.0; dataset.num_rows()],
                }
            }
            Self::Linear(_) => vec![0.0; dataset.num_rows()],
        }
    }

    /// Add tree predictions to existing buffer
    ///
    /// No-op if called on Linear booster or if tree not fitted.
    pub fn predict_tree_add(&self, dataset: &BinnedDataset, predictions: &mut [f32]) {
        if let Self::Tree(ser) = self {
            // Direct tree prediction avoids unnecessary TreeBooster allocation
            if let Some(tree) = &ser.tree {
                tree.predict_batch_add(dataset, predictions);
            }
        }
    }

    /// Predict using linear booster on raw features
    ///
    /// Returns zero vector if called on Tree booster.
    pub fn predict_linear(&self, features: &[f32], num_features: usize) -> Vec<f32> {
        match self {
            Self::Linear(l) => l.predict_batch(features, num_features),
            Self::Tree(_) => {
                let num_rows = features.len() / num_features;
                vec![0.0; num_rows]
            }
        }
    }

    // =========================================================================
    // Inspection
    // =========================================================================

    /// Check if this is a tree booster
    pub fn is_tree(&self) -> bool {
        matches!(self, Self::Tree(_))
    }

    /// Check if this is a linear booster
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::Linear(_))
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        match self {
            Self::Tree(ser) => {
                ser.tree.as_ref().map(|t| t.num_leaves()).unwrap_or(0)
            }
            Self::Linear(l) => l.num_params(),
        }
    }

    /// Check if the booster is fitted
    pub fn is_fitted(&self) -> bool {
        match self {
            Self::Tree(ser) => ser.tree.is_some(),
            Self::Linear(l) => l.weights().iter().any(|&w| w.abs() > 1e-10),
        }
    }

    /// Get tree reference (if tree booster)
    pub fn as_tree(&self) -> Option<&crate::tree::Tree> {
        match self {
            Self::Tree(ser) => ser.tree.as_ref(),
            Self::Linear(_) => None,
        }
    }

    /// Get linear booster reference (if linear booster)
    pub fn as_linear(&self) -> Option<&LinearBooster> {
        match self {
            Self::Linear(l) => Some(l),
            Self::Tree(_) => None,
        }
    }

    /// Reset the booster to initial state
    pub fn reset(&mut self) {
        match self {
            Self::Tree(ser) => {
                ser.tree = None;
            }
            Self::Linear(l) => {
                l.reset();
            }
        }
    }
}

impl From<TreeBooster> for Booster {
    fn from(booster: TreeBooster) -> Self {
        Self::Tree(booster.into())
    }
}

impl From<LinearBooster> for Booster {
    fn from(booster: LinearBooster) -> Self {
        Self::Linear(booster)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeatureInfo, FeatureType};

    fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * 3 + f * 7) % 256) as u8);
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32).sin()).collect();
        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("f{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_booster_tree_creation() {
        let booster = Booster::tree(TreeConfig::default());
        assert!(booster.is_tree());
        assert!(!booster.is_linear());
        assert!(!booster.is_fitted());
    }

    #[test]
    fn test_booster_linear_creation() {
        let booster = Booster::linear(5, LinearConfig::default());
        assert!(booster.is_linear());
        assert!(!booster.is_tree());
    }

    #[test]
    fn test_booster_tree_fit_predict() {
        let dataset = create_test_dataset(100, 3);
        let gradients: Vec<f32> = (0..100).map(|i| if i < 50 { -1.0 } else { 1.0 }).collect();
        let hessians = vec![1.0; 100];

        let mut booster = Booster::tree(TreeConfig::default());
        booster.fit_tree(&dataset, &gradients, &hessians, None).unwrap();

        assert!(booster.is_fitted());

        let predictions = booster.predict_tree(&dataset);
        assert_eq!(predictions.len(), 100);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_booster_linear_fit_predict() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gradients = vec![-3.0, -5.0, -7.0, -9.0, -11.0];
        let hessians = vec![1.0; 5];

        let config = LinearConfig::default()
            .with_lambda(0.01)
            .with_learning_rate(0.5)
            .with_max_iter(100);

        let mut booster = Booster::linear(1, config);
        booster.fit_linear(&features, 1, &gradients, &hessians).unwrap();

        let predictions = booster.predict_linear(&features, 1);
        assert_eq!(predictions.len(), 5);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_booster_reset() {
        let dataset = create_test_dataset(100, 3);
        let gradients = vec![-1.0; 100];
        let hessians = vec![1.0; 100];

        let mut booster = Booster::tree(TreeConfig::default());
        booster.fit_tree(&dataset, &gradients, &hessians, None).unwrap();
        assert!(booster.is_fitted());

        booster.reset();
        assert!(!booster.is_fitted());
    }

    #[test]
    fn test_booster_from_conversions() {
        let tree_booster = TreeBooster::new(TreeConfig::default());
        let booster: Booster = tree_booster.into();
        assert!(booster.is_tree());

        let linear_booster = LinearBooster::new(5, LinearConfig::default());
        let booster: Booster = linear_booster.into();
        assert!(booster.is_linear());
    }

    #[test]
    fn test_fit_tree_on_linear_returns_error() {
        let dataset = create_test_dataset(100, 3);
        let gradients = vec![-1.0; 100];
        let hessians = vec![1.0; 100];

        let mut booster = Booster::linear(3, LinearConfig::default());
        let result = booster.fit_tree(&dataset, &gradients, &hessians, None);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Cannot fit tree on Linear booster"));
    }

    #[test]
    fn test_fit_linear_on_tree_returns_error() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gradients = vec![-1.0; 5];
        let hessians = vec![1.0; 5];

        let mut booster = Booster::tree(TreeConfig::default());
        let result = booster.fit_linear(&features, 1, &gradients, &hessians);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Cannot fit linear on Tree booster"));
    }
}
