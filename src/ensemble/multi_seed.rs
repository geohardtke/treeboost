//! Multi-seed training for variance reduction
//!
//! Trains multiple models with different random seeds to reduce variance
//! and generate out-of-fold predictions for stacking.

use crate::booster::{GBDTConfig, GBDTModel};
use crate::dataset::{split_kfold, BinnedDataset};
use crate::defaults::{ensemble as ensemble_defaults, seeds as seeds_defaults};
use crate::tuner::Metric;
use crate::Result;
use rayon::prelude::*;

/// Create a subset of a BinnedDataset by extracting only the specified row indices.
///
/// This is a thin wrapper around `BinnedDataset::subset_by_indices()` for K-fold
/// cross-validation where we need to create training and validation subsets.
#[inline]
fn subset_dataset(dataset: &BinnedDataset, indices: &[usize]) -> BinnedDataset {
    dataset.subset_by_indices(indices)
}

/// Configuration for multi-seed training
#[derive(Debug, Clone)]
pub struct MultiSeedConfig {
    /// Number of seeds to train with
    pub n_seeds: usize,
    /// Base seed (seeds will be base_seed, base_seed+1, ...)
    pub base_seed: u64,
    /// Number of folds for OOF prediction generation
    pub n_folds: usize,
    /// Whether to train seeds in parallel
    pub parallel: bool,
}

impl Default for MultiSeedConfig {
    fn default() -> Self {
        Self {
            n_seeds: ensemble_defaults::DEFAULT_N_SEEDS,
            base_seed: seeds_defaults::DEFAULT_SEED,
            n_folds: ensemble_defaults::DEFAULT_N_FOLDS,
            parallel: true,
        }
    }
}

impl MultiSeedConfig {
    /// Create a new multi-seed config
    pub fn new(n_seeds: usize) -> Self {
        Self {
            n_seeds,
            ..Default::default()
        }
    }

    /// Set the base seed
    pub fn with_base_seed(mut self, seed: u64) -> Self {
        self.base_seed = seed;
        self
    }

    /// Set the number of folds for OOF generation
    pub fn with_n_folds(mut self, n_folds: usize) -> Self {
        self.n_folds = n_folds;
        self
    }

    /// Enable or disable parallel training
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// A trained model with its out-of-fold predictions
#[derive(Debug, Clone)]
pub struct TrainedMember {
    /// The trained GBDT model (trained on full data)
    pub model: GBDTModel,
    /// Out-of-fold predictions (one per sample)
    pub oof_preds: Vec<f32>,
    /// OOF metric score
    pub oof_metric: f32,
    /// Configuration used for training
    pub config: GBDTConfig,
    /// Random seed used
    pub seed: u64,
}

impl TrainedMember {
    /// Get the model's unique ID (based on config hash and seed)
    pub fn model_id(&self) -> u64 {
        // Simple hash combining seed with some config values
        let mut id = self.seed;
        id = id
            .wrapping_mul(31)
            .wrapping_add(self.config.num_rounds as u64);
        id = id
            .wrapping_mul(31)
            .wrapping_add(self.config.max_depth as u64);
        id = id
            .wrapping_mul(31)
            .wrapping_add((self.config.learning_rate * 1000.0) as u64);
        id
    }
}

/// Trainer for multiple models with different seeds
pub struct MultiSeedTrainer {
    base_config: GBDTConfig,
    multi_seed_config: MultiSeedConfig,
    metric: Metric,
}

impl MultiSeedTrainer {
    /// Create a new multi-seed trainer
    pub fn new(base_config: GBDTConfig, multi_seed_config: MultiSeedConfig) -> Self {
        // Auto-select metric based on loss type
        let metric = match base_config.loss_type {
            crate::booster::LossType::BinaryLogLoss => Metric::BinaryLogLoss,
            crate::booster::LossType::MultiClassLogLoss { num_classes } => {
                Metric::MultiClassLogLoss {
                    n_classes: num_classes,
                }
            }
            _ => Metric::Mse,
        };

        Self {
            base_config,
            multi_seed_config,
            metric,
        }
    }

    /// Set the evaluation metric
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Train N models with different seeds
    ///
    /// Each model is trained using K-fold cross-validation to generate
    /// out-of-fold predictions, then retrained on the full dataset.
    ///
    /// # Returns
    /// Vector of trained members with their OOF predictions
    pub fn train(&self, dataset: &BinnedDataset) -> Result<Vec<TrainedMember>> {
        let seeds: Vec<u64> = (0..self.multi_seed_config.n_seeds)
            .map(|i| self.multi_seed_config.base_seed + i as u64)
            .collect();

        if self.multi_seed_config.parallel {
            seeds
                .into_par_iter()
                .map(|seed| self.train_with_seed(dataset, seed))
                .collect()
        } else {
            seeds
                .into_iter()
                .map(|seed| self.train_with_seed(dataset, seed))
                .collect()
        }
    }

    /// Train a single model with OOF predictions via K-fold
    fn train_with_seed(&self, dataset: &BinnedDataset, seed: u64) -> Result<TrainedMember> {
        let mut config = self.base_config.clone();
        config.seed = seed;

        let num_rows = dataset.num_rows();
        let targets = dataset.targets();
        let n_folds = self.multi_seed_config.n_folds;

        // Generate K-fold split
        let kfold = split_kfold(num_rows, n_folds, seed);
        let mut oof_preds = vec![0.0f32; num_rows];

        // Train on each fold and collect OOF predictions
        for fold_idx in 0..n_folds {
            let (train_idx, val_idx) = kfold.get_fold(fold_idx);

            // Create training subset
            let train_data = subset_dataset(dataset, &train_idx);

            // Train fold model
            let fold_model = GBDTModel::train_binned(&train_data, config.clone())?;

            // Predict on validation fold
            let val_data = subset_dataset(dataset, &val_idx);
            let fold_preds = fold_model.predict(&val_data);

            // Store OOF predictions at original indices
            for (i, &pred) in fold_preds.iter().enumerate() {
                oof_preds[val_idx[i]] = pred;
            }
        }

        // Compute OOF metric
        let oof_metric = self.metric.compute(&oof_preds, targets);

        // Train final model on full data
        let model = GBDTModel::train_binned(dataset, config.clone())?;

        Ok(TrainedMember {
            model,
            oof_preds,
            oof_metric,
            config,
            seed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_seed_config_default() {
        let config = MultiSeedConfig::default();
        assert_eq!(config.n_seeds, 5);
        assert_eq!(config.base_seed, 42);
        assert_eq!(config.n_folds, 5);
        assert!(config.parallel);
    }

    #[test]
    fn test_multi_seed_config_builder() {
        let config = MultiSeedConfig::new(10)
            .with_base_seed(100)
            .with_n_folds(3)
            .with_parallel(false);

        assert_eq!(config.n_seeds, 10);
        assert_eq!(config.base_seed, 100);
        assert_eq!(config.n_folds, 3);
        assert!(!config.parallel);
    }

    #[test]
    fn test_trained_member_id() {
        let config1 = GBDTConfig::new().with_num_rounds(100).with_max_depth(6);
        let config2 = GBDTConfig::new().with_num_rounds(100).with_max_depth(6);

        // Create dummy trained members
        let member1 = TrainedMember {
            model: create_dummy_model(&config1),
            oof_preds: vec![],
            oof_metric: 0.0,
            config: config1.clone(),
            seed: 42,
        };

        let member2 = TrainedMember {
            model: create_dummy_model(&config2),
            oof_preds: vec![],
            oof_metric: 0.0,
            config: config2,
            seed: 42,
        };

        // Same config and seed should have same ID
        assert_eq!(member1.model_id(), member2.model_id());

        // Different seed should have different ID
        let member3 = TrainedMember {
            model: create_dummy_model(&config1),
            oof_preds: vec![],
            oof_metric: 0.0,
            config: config1,
            seed: 43,
        };
        assert_ne!(member1.model_id(), member3.model_id());
    }

    fn create_dummy_model(config: &GBDTConfig) -> GBDTModel {
        // Create a minimal dataset for testing
        let features = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![0.5, 1.5];
        GBDTModel::train(&features, 2, &targets, config.clone(), None).unwrap()
    }
}
