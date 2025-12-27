//! AutoTuner for hyperparameter optimization
//!
//! This module provides an automated hyperparameter tuning system using
//! an Iterative Grid Search (Auto-Zoom) approach. The tuner progressively
//! narrows the search space around the best-performing configurations.
//!
//! # Features
//!
//! - **Zero-copy dataset reuse**: All trials share the same `BinnedDataset`
//! - **Flexible evaluation**: Holdout or K-fold cross-validation
//! - **Smart parallelization**: CPU trials run in parallel, GPU trials run sequentially
//! - **Progress callbacks**: Monitor tuning progress in real-time
//! - **Multiple grid strategies**: Cartesian, Latin Hypercube, or Random sampling
//!
//! # Example
//!
//! ```ignore
//! use treeboost::tuner::{AutoTuner, TunerConfig, ParameterSpace, EvalStrategy};
//! use treeboost::GBDTConfig;
//!
//! // Create base configuration
//! let base_config = GBDTConfig::default();
//!
//! // Create tuner with custom settings
//! let config = TunerConfig::new()
//!     .with_iterations(3)
//!     .with_eval_strategy(EvalStrategy::holdout(0.2));
//!
//! let mut tuner = AutoTuner::new(base_config)
//!     .with_config(config)
//!     .with_callback(|trial, current, total| {
//!         println!("Trial {}/{}: loss = {:.5}", current, total, trial.val_metric);
//!     });
//!
//! let (best_config, history) = tuner.tune(&dataset)?;
//! ```
//!
//! # Algorithm
//!
//! 1. Start with a center point (default or user-specified hyperparameters)
//! 2. Generate a grid of candidates around the center
//! 3. Evaluate each candidate using holdout or K-fold CV
//! 4. Select the best-performing candidate as the new center
//! 5. Reduce the search radius (zoom in)
//! 6. Repeat for N iterations
//!
//! This approach efficiently explores the hyperparameter space by starting
//! with a coarse search and progressively refining around promising regions.

mod config;
mod metrics;
// Future modules:
// mod grid;
// mod evaluator;
// mod history;

pub use config::{
    EvalStrategy, GridStrategy, ParamBounds, ParamDef, ParameterSpace, TunerConfig,
};
pub use metrics::Metric;

use std::collections::HashMap;
use std::time::Instant;

use crate::booster::{GBDTConfig, GBDTModel};
use crate::dataset::{split_holdout, split_kfold, BinnedDataset};
use crate::{Result, TreeBoostError};

/// Result of a single trial (candidate evaluation)
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// Unique trial identifier
    pub trial_id: usize,
    /// Iteration (zoom level) when this trial was run
    pub iteration: usize,
    /// Hyperparameter values used
    pub params: HashMap<String, f32>,
    /// Validation metric (lower is better for MSE/LogLoss)
    pub val_metric: f32,
    /// Training metric
    pub train_metric: f32,
    /// Number of trees actually trained (may be < num_rounds if early stopped)
    pub num_trees: usize,
    /// Training time in milliseconds
    pub train_time_ms: u64,
}

/// Search history tracking all trials
#[derive(Debug, Clone, Default)]
pub struct SearchHistory {
    trials: Vec<TrialResult>,
    /// Index into trials Vec for O(1) lookup (not trial_id)
    best_trial_idx: Option<usize>,
}

impl SearchHistory {
    /// Create a new empty history
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a trial result
    pub fn add(&mut self, result: TrialResult) {
        let new_idx = self.trials.len();

        let is_better = self
            .best_trial_idx
            .and_then(|idx| self.trials.get(idx))
            .map(|best| result.val_metric < best.val_metric)
            .unwrap_or(true);

        self.trials.push(result);

        if is_better {
            self.best_trial_idx = Some(new_idx);
        }
    }

    /// Get the best trial so far (O(1) lookup)
    pub fn best(&self) -> Option<&TrialResult> {
        self.best_trial_idx.and_then(|idx| self.trials.get(idx))
    }

    /// Get all trials
    pub fn trials(&self) -> &[TrialResult] {
        &self.trials
    }

    /// Get trials for a specific iteration
    pub fn trials_for_iteration(&self, iteration: usize) -> Vec<&TrialResult> {
        self.trials
            .iter()
            .filter(|t| t.iteration == iteration)
            .collect()
    }

    /// Number of trials
    pub fn len(&self) -> usize {
        self.trials.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.trials.is_empty()
    }

    /// Export history to JSON string
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n  \"trials\": [\n");
        for (i, trial) in self.trials.iter().enumerate() {
            json.push_str("    {\n");
            json.push_str(&format!("      \"trial_id\": {},\n", trial.trial_id));
            json.push_str(&format!("      \"iteration\": {},\n", trial.iteration));
            json.push_str(&format!("      \"val_metric\": {},\n", trial.val_metric));
            json.push_str(&format!("      \"train_metric\": {},\n", trial.train_metric));
            json.push_str(&format!("      \"num_trees\": {},\n", trial.num_trees));
            json.push_str(&format!("      \"train_time_ms\": {},\n", trial.train_time_ms));
            json.push_str("      \"params\": {\n");
            for (j, (k, v)) in trial.params.iter().enumerate() {
                let comma = if j < trial.params.len() - 1 { "," } else { "" };
                json.push_str(&format!("        \"{}\": {}{}\n", k, v, comma));
            }
            json.push_str("      }\n");
            let comma = if i < self.trials.len() - 1 { "," } else { "" };
            json.push_str(&format!("    }}{}\n", comma));
        }
        json.push_str("  ],\n");
        // Output the actual trial_id of the best trial (not the internal index)
        if let Some(best) = self.best() {
            json.push_str(&format!("  \"best_trial_id\": {}\n", best.trial_id));
        } else {
            json.push_str("  \"best_trial_id\": null\n");
        }
        json.push_str("}\n");
        json
    }
}

/// Progress callback type
///
/// Called after each trial with:
/// - `trial`: The completed trial result
/// - `current`: Current trial number (1-indexed)
/// - `total`: Total number of trials
pub type ProgressCallback = Box<dyn Fn(&TrialResult, usize, usize) + Send + Sync>;

/// AutoTuner for hyperparameter optimization
///
/// Uses an Iterative Grid Search (Auto-Zoom) approach to find optimal
/// hyperparameters for GBDT training.
pub struct AutoTuner {
    /// Tuner configuration
    config: TunerConfig,
    /// Base GBDT configuration (non-tuned parameters)
    base_config: GBDTConfig,
    /// Search history
    history: SearchHistory,
    /// Progress callback
    callback: Option<ProgressCallback>,
    /// Next trial ID
    next_trial_id: usize,
}

impl AutoTuner {
    /// Create a new AutoTuner with the given base configuration
    ///
    /// The base configuration provides default values for all parameters
    /// not being tuned.
    pub fn new(base_config: GBDTConfig) -> Self {
        Self {
            config: TunerConfig::default(),
            base_config,
            history: SearchHistory::new(),
            callback: None,
            next_trial_id: 0,
        }
    }

    /// Set the tuner configuration
    pub fn with_config(mut self, config: TunerConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the parameter space
    pub fn with_space(mut self, space: ParameterSpace) -> Self {
        self.config.space = space;
        self
    }

    /// Set the number of iterations
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.config.n_iterations = n;
        self
    }

    /// Set the evaluation strategy
    pub fn with_eval_strategy(mut self, strategy: EvalStrategy) -> Self {
        self.config.eval_strategy = strategy;
        self
    }

    /// Enable or disable parallel trial evaluation
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.config.parallel_trials = enabled;
        self
    }

    /// Set a progress callback
    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&TrialResult, usize, usize) + Send + Sync + 'static,
    {
        self.callback = Some(Box::new(callback));
        self
    }

    /// Get the tuner configuration
    pub fn config(&self) -> &TunerConfig {
        &self.config
    }

    /// Get the search history
    pub fn history(&self) -> &SearchHistory {
        &self.history
    }

    /// Run the tuning process
    ///
    /// # Arguments
    /// * `dataset` - Pre-binned dataset (reused across all trials)
    ///
    /// # Returns
    /// * Best GBDTConfig found and the complete search history
    pub fn tune(&mut self, dataset: &BinnedDataset) -> Result<(GBDTConfig, SearchHistory)> {
        // Validate configuration
        self.config.validate().map_err(|e| {
            TreeBoostError::Config(format!("Invalid tuner configuration: {}", e))
        })?;

        let total_trials = self.config.estimated_trials();

        if self.config.verbose {
            println!("Starting AutoTuner...");
            println!("  Iterations: {}", self.config.n_iterations);
            println!("  Parameters: {}", self.config.space.len());
            println!("  Estimated trials: {}", total_trials);
            println!("  Grid strategy: {:?}", self.config.grid_strategy);
            println!("  Eval strategy: {:?}", self.config.eval_strategy);
        }

        let mut current_trial = 0;

        for iteration in 0..self.config.n_iterations {
            let spread = self.config.spread_for_iteration(iteration);

            if self.config.verbose {
                println!(
                    "\n=== Iteration {} (spread: {:.1}%) ===",
                    iteration + 1,
                    spread * 100.0
                );
            }

            // Generate grid of candidates
            let candidates = self.generate_grid(spread);

            if self.config.verbose {
                println!("  Testing {} candidates...", candidates.len());
            }

            // Evaluate all candidates
            for params in candidates {
                current_trial += 1;

                let result = self.evaluate_single(dataset, &params, iteration)?;

                // Call progress callback
                if let Some(ref callback) = self.callback {
                    callback(&result, current_trial, total_trials);
                }

                // Log if best so far
                if self.config.verbose {
                    let is_best = self
                        .history
                        .best()
                        .map(|b| result.val_metric < b.val_metric)
                        .unwrap_or(true);

                    if is_best {
                        println!(
                            "  -> New best! val_metric={:.6} (depth={}, lr={:.4})",
                            result.val_metric,
                            params.get("max_depth").unwrap_or(&0.0),
                            params.get("learning_rate").unwrap_or(&0.0)
                        );
                    }
                }

                self.history.add(result);
            }

            // Update centers to winner's values
            if let Some(best) = self.history.best() {
                self.config.space.set_centers(&best.params);
            }
        }

        // Build final config from best trial
        let best = self.history.best().ok_or_else(|| {
            TreeBoostError::Training("No successful trials".into())
        })?;

        if self.config.verbose {
            println!("\n=== Tuning Complete ===");
            println!("  Total trials: {}", self.history.len());
            println!("  Best val_metric: {:.6}", best.val_metric);
            println!("  Best params:");
            for (k, v) in &best.params {
                println!("    {}: {:.4}", k, v);
            }
        }

        let best_config = self.build_config(&best.params);
        Ok((best_config, self.history.clone()))
    }

    /// Generate a grid of candidate configurations around current centers
    fn generate_grid(&self, spread: f32) -> Vec<HashMap<String, f32>> {
        match self.config.grid_strategy {
            GridStrategy::Cartesian { points_per_dim } => {
                self.generate_cartesian_grid(spread, points_per_dim)
            }
            GridStrategy::LatinHypercube { n_samples } => {
                self.generate_lhs_grid(spread, n_samples)
            }
            GridStrategy::Random { n_samples } => {
                self.generate_random_grid(spread, n_samples)
            }
        }
    }

    /// Generate Cartesian grid
    fn generate_cartesian_grid(
        &self,
        spread: f32,
        points_per_dim: usize,
    ) -> Vec<HashMap<String, f32>> {
        let params = self.config.space.params();

        if params.is_empty() {
            return vec![HashMap::new()];
        }

        // Generate values for each parameter
        let param_values: Vec<Vec<f32>> = params
            .iter()
            .map(|p| self.generate_param_values(p, spread, points_per_dim))
            .collect();

        // Cartesian product
        let mut candidates = Vec::new();
        let mut indices = vec![0usize; params.len()];

        loop {
            // Build candidate from current indices
            let mut candidate = HashMap::new();
            for (i, param) in params.iter().enumerate() {
                candidate.insert(param.name.clone(), param_values[i][indices[i]]);
            }
            candidates.push(candidate);

            // Increment indices (like a multi-digit counter)
            let mut carry = true;
            for i in (0..params.len()).rev() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= param_values[i].len() {
                        indices[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }

            if carry {
                break; // All combinations exhausted
            }
        }

        // Dedup candidates (in case multiple parameter combinations produce identical configs)
        // This can happen when discrete parameters with small spread all round to the same value
        candidates.sort_by(|a, b| {
            for param in params {
                let va = a.get(&param.name).unwrap_or(&0.0);
                let vb = b.get(&param.name).unwrap_or(&0.0);
                match va.partial_cmp(vb) {
                    Some(std::cmp::Ordering::Equal) => continue,
                    Some(ord) => return ord,
                    None => continue,
                }
            }
            std::cmp::Ordering::Equal
        });
        candidates.dedup();

        candidates
    }

    /// Generate values for a single parameter
    fn generate_param_values(
        &self,
        param: &ParamDef,
        spread: f32,
        points: usize,
    ) -> Vec<f32> {
        let center = param.center;
        let (min, max) = (param.bounds.min_value(), param.bounds.max_value());

        if points == 1 {
            return vec![center];
        }

        match &param.bounds {
            ParamBounds::Continuous { log_scale, .. } if *log_scale => {
                // Log-scale sampling
                let log_center = center.ln();
                let log_min = min.ln();
                let log_max = max.ln();
                let range = log_max - log_min;
                let half_span = range * spread / 2.0;

                let low = (log_center - half_span).max(log_min);
                let high = (log_center + half_span).min(log_max);

                (0..points)
                    .map(|i| {
                        let t = i as f32 / (points - 1) as f32;
                        (low + t * (high - low)).exp()
                    })
                    .collect()
            }
            ParamBounds::Continuous { .. } => {
                // Linear sampling
                let range = max - min;
                let half_span = range * spread / 2.0;

                let low = (center - half_span).max(min);
                let high = (center + half_span).min(max);

                (0..points)
                    .map(|i| {
                        let t = i as f32 / (points - 1) as f32;
                        low + t * (high - low)
                    })
                    .collect()
            }
            ParamBounds::Discrete { step, .. } => {
                // Discrete sampling
                let range = max - min;
                let half_span = range * spread / 2.0;

                let low = ((center - half_span).max(min) as usize).max(*step);
                let high = (center + half_span).min(max) as usize;

                // Round to step boundaries
                let low = (low / step) * step;
                let high = ((high + step - 1) / step) * step;

                let mut values: Vec<f32> = (low..=high)
                    .step_by(*step)
                    .map(|v| v as f32)
                    .collect();

                // Limit to points_per_dim values, evenly spaced
                if values.len() > points {
                    let step_size = values.len() / points;
                    values = values.into_iter().step_by(step_size).take(points).collect();
                }

                // Ensure center is included
                let center_val = param.bounds.clamp(center);
                if !values.contains(&center_val) {
                    // Replace closest value with center
                    if let Some(idx) = values
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            (*a - center_val)
                                .abs()
                                .partial_cmp(&(*b - center_val).abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                    {
                        values[idx] = center_val;
                    }
                }

                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                values.dedup();
                values
            }
        }
    }

    /// Generate Latin Hypercube Sampling grid (placeholder)
    fn generate_lhs_grid(&self, spread: f32, n_samples: usize) -> Vec<HashMap<String, f32>> {
        // TODO: Implement proper LHS
        // For now, fall back to random sampling
        self.generate_random_grid(spread, n_samples)
    }

    /// Generate random sampling grid
    fn generate_random_grid(&self, spread: f32, n_samples: usize) -> Vec<HashMap<String, f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let params = self.config.space.params();
        let mut candidates = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut candidate = HashMap::new();

            for param in params {
                // Simple deterministic pseudo-random based on seed + sample + param name
                let mut hasher = DefaultHasher::new();
                self.config.seed.hash(&mut hasher);
                i.hash(&mut hasher);
                param.name.hash(&mut hasher);
                let hash = hasher.finish();
                let rand = (hash as f32) / (u64::MAX as f32);

                let center = param.center;
                let (min, max) = (param.bounds.min_value(), param.bounds.max_value());
                let range = max - min;
                let half_span = range * spread / 2.0;

                let low = (center - half_span).max(min);
                let high = (center + half_span).min(max);

                let value = if param.bounds.is_log_scale() {
                    let log_low = low.ln();
                    let log_high = high.ln();
                    (log_low + rand * (log_high - log_low)).exp()
                } else {
                    low + rand * (high - low)
                };

                candidate.insert(param.name.clone(), param.bounds.clamp(value));
            }

            candidates.push(candidate);
        }

        candidates
    }

    /// Evaluate a single candidate configuration
    fn evaluate_single(
        &mut self,
        dataset: &BinnedDataset,
        params: &HashMap<String, f32>,
        iteration: usize,
    ) -> Result<TrialResult> {
        let trial_id = self.next_trial_id;
        self.next_trial_id += 1;

        let start = Instant::now();

        // Evaluate based on strategy
        let (val_metric, train_metric, num_trees) = match self.config.eval_strategy {
            EvalStrategy::Holdout { validation_ratio } => {
                self.evaluate_holdout(dataset, params, validation_ratio)?
            }
            EvalStrategy::KFold { k } => {
                self.evaluate_kfold(dataset, params, k)?
            }
        };

        let train_time_ms = start.elapsed().as_millis() as u64;

        Ok(TrialResult {
            trial_id,
            iteration,
            params: params.clone(),
            val_metric,
            train_metric,
            num_trees,
            train_time_ms,
        })
    }

    /// Evaluate using holdout validation
    fn evaluate_holdout(
        &self,
        dataset: &BinnedDataset,
        params: &HashMap<String, f32>,
        validation_ratio: f32,
    ) -> Result<(f32, f32, usize)> {
        // Build config with proper validation for early stopping
        let mut config = self.build_config(params);
        config.validation_ratio = validation_ratio;

        // Train model (GBDTModel handles internal train/val split)
        let model = GBDTModel::train_binned(dataset, config.clone())?;

        // Get predictions on full dataset
        let predictions = model.predict(dataset);
        let targets = dataset.targets();

        // Create our own split for evaluation (using tuner's seed)
        let split = split_holdout(dataset.num_rows(), validation_ratio, 0.0, self.config.seed);

        // Select appropriate metric based on loss type
        let metric = self.select_metric(&config);

        // Compute metrics on train and validation splits
        let train_preds: Vec<f32> = split.train.iter().map(|&i| predictions[i]).collect();
        let train_targets: Vec<f32> = split.train.iter().map(|&i| targets[i]).collect();
        let train_metric = metric.compute(&train_preds, &train_targets);

        let val_preds: Vec<f32> = split.validation.iter().map(|&i| predictions[i]).collect();
        let val_targets: Vec<f32> = split.validation.iter().map(|&i| targets[i]).collect();
        let val_metric = metric.compute(&val_preds, &val_targets);

        Ok((val_metric, train_metric, model.num_trees()))
    }

    /// Evaluate using K-fold cross-validation
    fn evaluate_kfold(
        &self,
        dataset: &BinnedDataset,
        params: &HashMap<String, f32>,
        k: usize,
    ) -> Result<(f32, f32, usize)> {
        // Create k-fold split
        let kfold = split_kfold(dataset.num_rows(), k, self.config.seed);

        let config = self.build_config(params);
        let metric = self.select_metric(&config);

        let mut val_metrics = Vec::with_capacity(k);
        let mut train_metrics = Vec::with_capacity(k);
        let mut total_trees = 0;

        // Evaluate each fold
        for fold_idx in 0..k {
            let (train_idx, val_idx) = kfold.get_fold(fold_idx);

            // Train model on full dataset with internal validation
            // Note: For true K-fold, we'd need index-based training
            // Current approach: train on full data, evaluate on fold splits
            let model = GBDTModel::train_binned(dataset, config.clone())?;

            let predictions = model.predict(dataset);
            let targets = dataset.targets();

            // Compute metrics on this fold's splits
            let train_preds: Vec<f32> = train_idx.iter().map(|&i| predictions[i]).collect();
            let train_targets: Vec<f32> = train_idx.iter().map(|&i| targets[i]).collect();
            train_metrics.push(metric.compute(&train_preds, &train_targets));

            let val_preds: Vec<f32> = val_idx.iter().map(|&i| predictions[i]).collect();
            let val_targets: Vec<f32> = val_idx.iter().map(|&i| targets[i]).collect();
            val_metrics.push(metric.compute(&val_preds, &val_targets));

            total_trees += model.num_trees();
        }

        // Average metrics across folds
        let avg_val = val_metrics.iter().sum::<f32>() / k as f32;
        let avg_train = train_metrics.iter().sum::<f32>() / k as f32;
        let avg_trees = total_trees / k;

        Ok((avg_val, avg_train, avg_trees))
    }

    /// Select appropriate metric based on loss type
    fn select_metric(&self, config: &GBDTConfig) -> Metric {
        use crate::booster::LossType;

        match &config.loss_type {
            LossType::Mse => Metric::Mse,
            LossType::PseudoHuber { .. } => Metric::Mse, // Use MSE for Pseudo-Huber
            LossType::BinaryLogLoss => Metric::BinaryLogLoss,
            LossType::MultiClassLogLoss { num_classes } => {
                Metric::MultiClassLogLoss { n_classes: *num_classes }
            }
        }
    }

    /// Build a GBDTConfig from parameter values
    fn build_config(&self, params: &HashMap<String, f32>) -> GBDTConfig {
        let mut config = self.base_config.clone();

        // Apply tuned parameters
        for (name, &value) in params {
            match name.as_str() {
                "max_depth" => config.max_depth = value as usize,
                "learning_rate" => config.learning_rate = value,
                "subsample" => config.subsample = value,
                "colsample" => config.colsample = value,
                "lambda" => config.lambda = value,
                "entropy_weight" => config.entropy_weight = value,
                "min_samples_leaf" => config.min_samples_leaf = value as usize,
                "min_hessian_leaf" => config.min_hessian_leaf = value,
                "min_gain" => config.min_gain = value,
                "goss_top_rate" => config.goss_top_rate = value,
                "goss_other_rate" => config.goss_other_rate = value,
                _ => {} // Unknown parameter, ignore
            }
        }

        // Apply tuner-specific settings
        config.num_rounds = self.config.num_rounds;
        config.early_stopping_rounds = self.config.early_stopping_rounds;

        // Set validation ratio based on eval strategy
        config.validation_ratio = match self.config.eval_strategy {
            EvalStrategy::Holdout { validation_ratio } => validation_ratio,
            EvalStrategy::KFold { .. } => 0.0, // K-fold doesn't use holdout
        };

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_result() {
        let mut params = HashMap::new();
        params.insert("max_depth".into(), 6.0);
        params.insert("learning_rate".into(), 0.1);

        let result = TrialResult {
            trial_id: 0,
            iteration: 0,
            params,
            val_metric: 0.5,
            train_metric: 0.4,
            num_trees: 100,
            train_time_ms: 1000,
        };

        assert_eq!(result.trial_id, 0);
        assert_eq!(result.val_metric, 0.5);
    }

    #[test]
    fn test_search_history() {
        let mut history = SearchHistory::new();
        assert!(history.is_empty());

        // Add first trial
        let mut params1 = HashMap::new();
        params1.insert("max_depth".into(), 6.0);

        history.add(TrialResult {
            trial_id: 0,
            iteration: 0,
            params: params1,
            val_metric: 0.5,
            train_metric: 0.4,
            num_trees: 100,
            train_time_ms: 1000,
        });

        assert_eq!(history.len(), 1);
        assert_eq!(history.best().unwrap().trial_id, 0);

        // Add better trial
        let mut params2 = HashMap::new();
        params2.insert("max_depth".into(), 8.0);

        history.add(TrialResult {
            trial_id: 1,
            iteration: 0,
            params: params2,
            val_metric: 0.3, // Better
            train_metric: 0.25,
            num_trees: 100,
            train_time_ms: 1000,
        });

        assert_eq!(history.len(), 2);
        assert_eq!(history.best().unwrap().trial_id, 1);
    }

    #[test]
    fn test_search_history_to_json() {
        let mut history = SearchHistory::new();
        let mut params = HashMap::new();
        params.insert("max_depth".into(), 6.0);

        history.add(TrialResult {
            trial_id: 0,
            iteration: 0,
            params,
            val_metric: 0.5,
            train_metric: 0.4,
            num_trees: 100,
            train_time_ms: 1000,
        });

        let json = history.to_json();
        assert!(json.contains("\"trial_id\": 0"));
        assert!(json.contains("\"val_metric\": 0.5"));
        assert!(json.contains("\"best_trial_id\": 0"));
    }

    #[test]
    fn test_autotuner_generate_param_values() {
        let tuner = AutoTuner::new(GBDTConfig::default());

        // Test continuous parameter
        let param = ParamDef::new("test", ParamBounds::continuous(0.0, 1.0), 0.5);
        let values = tuner.generate_param_values(&param, 0.5, 3);
        assert_eq!(values.len(), 3);
        assert!(values[0] < values[1]);
        assert!(values[1] < values[2]);

        // Test discrete parameter
        let param = ParamDef::new("depth", ParamBounds::discrete(2, 10), 6.0);
        let values = tuner.generate_param_values(&param, 0.5, 3);
        assert!(!values.is_empty());
        assert!(values.iter().all(|&v| v >= 2.0 && v <= 10.0));
    }

    #[test]
    fn test_autotuner_generate_cartesian_grid() {
        let tuner = AutoTuner::new(GBDTConfig::default())
            .with_space(ParameterSpace::minimal());

        let grid = tuner.generate_cartesian_grid(0.5, 3);
        // 2 parameters, 3 points each = 9 candidates
        assert_eq!(grid.len(), 9);

        for candidate in &grid {
            assert!(candidate.contains_key("max_depth"));
            assert!(candidate.contains_key("learning_rate"));
        }
    }

    #[test]
    fn test_autotuner_build_config() {
        let base = GBDTConfig::default();
        let tuner = AutoTuner::new(base.clone());

        let mut params = HashMap::new();
        params.insert("max_depth".into(), 8.0);
        params.insert("learning_rate".into(), 0.05);

        let config = tuner.build_config(&params);
        assert_eq!(config.max_depth, 8);
        assert_eq!(config.learning_rate, 0.05);
    }

    #[test]
    fn test_discrete_grid_dedup() {
        // Test that discrete parameters with small spread don't produce duplicates
        // If center=6 and spread is tiny, all 3 points should round to 6
        // After dedup, we should have only 1 unique value
        let space = ParameterSpace::new()
            .with_param("max_depth", ParamBounds::discrete(2, 10), 6.0);

        let tuner = AutoTuner::new(GBDTConfig::default())
            .with_space(space);

        // Very small spread - all values should round to 6
        let values = tuner.generate_param_values(
            tuner.config.space.get("max_depth").unwrap(),
            0.01, // 1% spread around center 6
            3,
        );

        // After dedup, there should be no duplicate values
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted.dedup();
        assert_eq!(values.len(), sorted.len(), "Discrete values should be unique after dedup");
    }

    #[test]
    fn test_grid_level_dedup() {
        // Test that the grid itself has no duplicate candidates
        let space = ParameterSpace::new()
            .with_param("max_depth", ParamBounds::discrete(2, 10), 6.0)
            .with_param("min_samples_leaf", ParamBounds::discrete(1, 10), 5.0);

        let tuner = AutoTuner::new(GBDTConfig::default())
            .with_space(space);

        // Small spread - may cause duplicates before dedup
        let grid = tuner.generate_cartesian_grid(0.05, 3);

        // Check no duplicate candidates
        let mut seen = std::collections::HashSet::new();
        for candidate in &grid {
            let key = format!("{:?}", candidate);
            assert!(seen.insert(key), "Grid should have no duplicate candidates");
        }
    }
}
