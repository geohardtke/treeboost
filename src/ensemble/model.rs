//! Stacked ensemble model
//!
//! Provides the final composite model that combines multiple GBDT models
//! using a stacking strategy.

use super::multi_seed::{MultiSeedConfig, MultiSeedTrainer, TrainedMember};
use super::selection::{HillClimbingSelector, SelectionConfig};
use super::stacking::{RidgeStacker, SimpleAverageStacker, StackingConfig};
use super::traits::Stacker;
use crate::booster::GBDTConfig;
use crate::dataset::BinnedDataset;
use crate::tuner::Metric;
use crate::Result;

/// Statistics about an ensemble
#[derive(Debug, Clone)]
pub struct EnsembleStats {
    /// Number of member models
    pub n_members: usize,
    /// Stacking weights (if applicable)
    pub weights: Option<Vec<f32>>,
    /// Ensemble OOF metric
    pub oof_metric: f32,
    /// Individual member OOF metrics
    pub member_metrics: Vec<f32>,
    /// Best individual model metric
    pub best_individual: f32,
    /// Improvement over best individual
    pub improvement: f32,
}

/// A stacked ensemble of GBDT models
///
/// Combines multiple GBDT models trained with different configurations
/// and/or random seeds using a stacking strategy (Ridge regression or simple average).
pub struct StackedEnsemble {
    /// Selected member models
    members: Vec<TrainedMember>,
    /// Stacker for combining predictions
    stacker: Box<dyn Stacker>,
    /// Ensemble OOF metric
    oof_metric: f32,
    /// Metric used for evaluation
    metric: Metric,
}

impl StackedEnsemble {
    /// Create a new stacked ensemble from trained members and a fitted stacker
    pub fn new(
        members: Vec<TrainedMember>,
        stacker: Box<dyn Stacker>,
        oof_metric: f32,
        metric: Metric,
    ) -> Self {
        Self {
            members,
            stacker,
            oof_metric,
            metric,
        }
    }

    /// Get the number of member models
    pub fn n_members(&self) -> usize {
        self.members.len()
    }

    /// Get the stacking weights (if applicable)
    pub fn weights(&self) -> Option<&[f32]> {
        self.stacker.weights()
    }

    /// Get the OOF metric
    pub fn oof_metric(&self) -> f32 {
        self.oof_metric
    }

    /// Get ensemble statistics
    pub fn stats(&self) -> EnsembleStats {
        let member_metrics: Vec<f32> = self.members.iter().map(|m| m.oof_metric).collect();

        let best_individual = if self.metric.lower_is_better() {
            member_metrics
                .iter()
                .cloned()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(f32::INFINITY)
        } else {
            member_metrics
                .iter()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(f32::NEG_INFINITY)
        };

        let improvement = if self.metric.lower_is_better() {
            best_individual - self.oof_metric
        } else {
            self.oof_metric - best_individual
        };

        EnsembleStats {
            n_members: self.members.len(),
            weights: self.stacker.weights().map(|w| w.to_vec()),
            oof_metric: self.oof_metric,
            member_metrics,
            best_individual,
            improvement,
        }
    }

    /// Predict on binned dataset
    pub fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        if self.members.is_empty() {
            return vec![0.0; dataset.num_rows()];
        }

        // Get predictions from each member
        let predictions: Vec<Vec<f32>> = self
            .members
            .iter()
            .map(|m| m.model.predict(dataset))
            .collect();

        // Combine using stacker
        self.stacker.combine(&predictions)
    }

    /// Predict with raw features (f64)
    pub fn predict_raw(&self, features: &[f64]) -> Vec<f32> {
        if self.members.is_empty() {
            // Estimate num_rows from first member's feature count
            return Vec::new();
        }

        // Get predictions from each member
        let predictions: Vec<Vec<f32>> = self
            .members
            .iter()
            .map(|m| m.model.predict_raw(features))
            .collect();

        // Combine using stacker
        self.stacker.combine(&predictions)
    }

    /// Get reference to member models
    pub fn members(&self) -> &[TrainedMember] {
        &self.members
    }
}

/// Builder for constructing stacked ensembles
///
/// Provides a fluent API for configuring and building ensemble models.
///
/// # Example
///
/// ```ignore
/// let ensemble = EnsembleBuilder::new(config)
///     .with_n_seeds(5)
///     .with_ridge_alpha(10.0)
///     .with_max_models(20)
///     .build(&dataset)?;
/// ```
pub struct EnsembleBuilder {
    base_config: GBDTConfig,
    multi_seed: MultiSeedConfig,
    selection: SelectionConfig,
    stacking: StackingConfig,
    metric: Option<Metric>,
    use_ridge: bool,
}

impl EnsembleBuilder {
    /// Create a new ensemble builder with a base GBDT config
    pub fn new(base_config: GBDTConfig) -> Self {
        Self {
            base_config,
            multi_seed: MultiSeedConfig::default(),
            selection: SelectionConfig::default(),
            stacking: StackingConfig::default(),
            metric: None,
            use_ridge: true,
        }
    }

    /// Set the number of seeds to train
    pub fn with_n_seeds(mut self, n: usize) -> Self {
        self.multi_seed.n_seeds = n;
        self
    }

    /// Set the base seed
    pub fn with_base_seed(mut self, seed: u64) -> Self {
        self.multi_seed.base_seed = seed;
        self
    }

    /// Set the number of folds for OOF generation
    pub fn with_n_folds(mut self, n: usize) -> Self {
        self.multi_seed.n_folds = n;
        self
    }

    /// Enable or disable parallel training
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.multi_seed.parallel = parallel;
        self
    }

    /// Set maximum models to select
    pub fn with_max_models(mut self, max: usize) -> Self {
        self.selection.max_models = max;
        self
    }

    /// Set minimum improvement for selection
    pub fn with_min_improvement(mut self, min: f32) -> Self {
        self.selection.min_improvement = min;
        self
    }

    /// Set Ridge alpha parameter
    pub fn with_ridge_alpha(mut self, alpha: f32) -> Self {
        self.stacking.alpha = alpha;
        self
    }

    /// Enable or disable rank transformation
    pub fn with_rank_transform(mut self, enabled: bool) -> Self {
        self.stacking.rank_transform = enabled;
        self
    }

    /// Use simple average instead of Ridge stacking
    pub fn with_simple_average(mut self) -> Self {
        self.use_ridge = false;
        self
    }

    /// Set the evaluation metric
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = Some(metric);
        self
    }

    /// Set multi-seed config
    pub fn with_multi_seed_config(mut self, config: MultiSeedConfig) -> Self {
        self.multi_seed = config;
        self
    }

    /// Set selection config
    pub fn with_selection_config(mut self, config: SelectionConfig) -> Self {
        self.selection = config;
        self
    }

    /// Set stacking config
    pub fn with_stacking_config(mut self, config: StackingConfig) -> Self {
        self.stacking = config;
        self
    }

    /// Build the ensemble from a dataset
    ///
    /// # Steps
    /// 1. Train multiple models with different seeds (K-fold OOF)
    /// 2. Select models using hill climbing
    /// 3. Fit stacker on selected models' OOF predictions
    pub fn build(self, dataset: &BinnedDataset) -> Result<StackedEnsemble> {
        let targets = dataset.targets();

        // Auto-select metric if not specified
        let metric = self.metric.unwrap_or({
            match self.base_config.loss_type {
                crate::booster::LossType::BinaryLogLoss => Metric::BinaryLogLoss,
                crate::booster::LossType::MultiClassLogLoss { num_classes } => {
                    Metric::MultiClassLogLoss {
                        n_classes: num_classes,
                    }
                }
                _ => Metric::Mse,
            }
        });

        // 1. Train with multiple seeds
        let trainer =
            MultiSeedTrainer::new(self.base_config.clone(), self.multi_seed).with_metric(metric);
        let all_members = trainer.train(dataset)?;

        // 2. Hill climbing selection
        let selector = HillClimbingSelector::new(self.selection, metric);
        let selected_indices = selector.select(&all_members, targets);

        // Extract selected members
        let members: Vec<TrainedMember> = if selected_indices.is_empty() {
            // If selection returned empty, use all models
            all_members
        } else {
            selected_indices
                .iter()
                .map(|&i| all_members[i].clone())
                .collect()
        };

        // 3. Fit stacker
        let oof_preds: Vec<Vec<f32>> = members.iter().map(|m| m.oof_preds.clone()).collect();

        let mut stacker: Box<dyn Stacker> = if self.use_ridge {
            Box::new(RidgeStacker::new(self.stacking))
        } else {
            Box::new(SimpleAverageStacker::new())
        };

        stacker.fit(&oof_preds, targets);

        // 4. Compute ensemble OOF metric
        let blended = stacker.combine(&oof_preds);
        let oof_metric = metric.compute(&blended, targets);

        Ok(StackedEnsemble::new(members, stacker, oof_metric, metric))
    }

    /// Build from pre-trained members (skip training step)
    pub fn build_from_members(
        self,
        members: Vec<TrainedMember>,
        targets: &[f32],
    ) -> Result<StackedEnsemble> {
        let metric = self.metric.unwrap_or(Metric::Mse);

        // Hill climbing selection
        let selector = HillClimbingSelector::new(self.selection, metric);
        let selected_indices = selector.select(&members, targets);

        // Extract selected members
        let selected: Vec<TrainedMember> = if selected_indices.is_empty() {
            members
        } else {
            selected_indices
                .iter()
                .map(|&i| members[i].clone())
                .collect()
        };

        // Fit stacker
        let oof_preds: Vec<Vec<f32>> = selected.iter().map(|m| m.oof_preds.clone()).collect();

        let mut stacker: Box<dyn Stacker> = if self.use_ridge {
            Box::new(RidgeStacker::new(self.stacking))
        } else {
            Box::new(SimpleAverageStacker::new())
        };

        stacker.fit(&oof_preds, targets);

        // Compute ensemble OOF metric
        let blended = stacker.combine(&oof_preds);
        let oof_metric = metric.compute(&blended, targets);

        Ok(StackedEnsemble::new(selected, stacker, oof_metric, metric))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_builder_defaults() {
        let config = GBDTConfig::new();
        let builder = EnsembleBuilder::new(config);

        assert_eq!(builder.multi_seed.n_seeds, 5);
        assert!(builder.use_ridge);
    }

    #[test]
    fn test_ensemble_builder_fluent_api() {
        let config = GBDTConfig::new();
        let builder = EnsembleBuilder::new(config)
            .with_n_seeds(10)
            .with_base_seed(123)
            .with_ridge_alpha(5.0)
            .with_max_models(20)
            .with_rank_transform(true);

        assert_eq!(builder.multi_seed.n_seeds, 10);
        assert_eq!(builder.multi_seed.base_seed, 123);
        assert!((builder.stacking.alpha - 5.0).abs() < 1e-6);
        assert_eq!(builder.selection.max_models, 20);
        assert!(builder.stacking.rank_transform);
    }

    #[test]
    fn test_ensemble_builder_simple_average() {
        let config = GBDTConfig::new();
        let builder = EnsembleBuilder::new(config).with_simple_average();

        assert!(!builder.use_ridge);
    }
}
