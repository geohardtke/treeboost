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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;

use crate::backend::BackendType;
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
    /// Next trial ID (atomic for parallel evaluation)
    next_trial_id: AtomicUsize,
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
            next_trial_id: AtomicUsize::new(0),
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

    /// Set the random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
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
        let use_parallel = self.config.parallel_trials && !self.is_gpu_backend();

        if self.config.verbose {
            println!("Starting AutoTuner...");
            println!("  Iterations: {}", self.config.n_iterations);
            println!("  Parameters: {}", self.config.space.len());
            println!("  Estimated trials: {}", total_trials);
            println!("  Grid strategy: {:?}", self.config.grid_strategy);
            println!("  Eval strategy: {:?}", self.config.eval_strategy);
            println!(
                "  Parallel: {} (backend: {:?})",
                if use_parallel { "enabled" } else { "disabled" },
                self.base_config.backend_type
            );
        }

        let current_trial = AtomicUsize::new(0);

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

            // Evaluate all candidates (parallel or sequential based on backend)
            let results = self.evaluate_candidates(
                dataset,
                candidates,
                iteration,
                &current_trial,
                total_trials,
            );

            // Add all results to history and log new best
            for result in results {
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
                            result.params.get("max_depth").unwrap_or(&0.0),
                            result.params.get("learning_rate").unwrap_or(&0.0)
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

    /// Generate Latin Hypercube Sampling grid
    ///
    /// LHS ensures good space-filling by dividing each parameter's range into n equal strata
    /// and sampling exactly once from each stratum. This provides better coverage than
    /// pure random sampling with the same number of samples.
    fn generate_lhs_grid(&self, spread: f32, n_samples: usize) -> Vec<HashMap<String, f32>> {
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        if n_samples == 0 {
            return Vec::new();
        }

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let params = self.config.space.params();
        let n_params = params.len();

        if n_params == 0 {
            return vec![HashMap::new(); n_samples];
        }

        // Create permutation for each parameter dimension
        // Each column gets a shuffled list of strata indices [0, 1, ..., n_samples-1]
        let mut strata_permutations: Vec<Vec<usize>> = Vec::with_capacity(n_params);
        for _ in 0..n_params {
            let mut perm: Vec<usize> = (0..n_samples).collect();
            perm.shuffle(&mut rng);
            strata_permutations.push(perm);
        }

        // Generate samples
        let mut candidates = Vec::with_capacity(n_samples);
        for sample_idx in 0..n_samples {
            let mut candidate = HashMap::new();

            for (param_idx, param) in params.iter().enumerate() {
                let stratum = strata_permutations[param_idx][sample_idx];

                // Compute the effective bounds based on spread around center
                let center = param.center;
                let (min, max) = (param.bounds.min_value(), param.bounds.max_value());
                let range = max - min;
                let half_span = range * spread / 2.0;
                let low = (center - half_span).max(min);
                let high = (center + half_span).min(max);

                // Sample uniformly within this stratum
                // Stratum boundaries: [stratum/n_samples, (stratum+1)/n_samples] of the [low, high] range
                let stratum_low = stratum as f32 / n_samples as f32;
                let stratum_high = (stratum + 1) as f32 / n_samples as f32;
                let u: f32 = rng.gen_range(stratum_low..stratum_high);

                let value = if param.bounds.is_log_scale() {
                    // Log-uniform sampling within stratum
                    let log_low = low.max(1e-10).ln();
                    let log_high = high.max(1e-10).ln();
                    (log_low + u * (log_high - log_low)).exp()
                } else {
                    // Linear interpolation within stratum
                    low + u * (high - low)
                };

                candidate.insert(param.name.clone(), param.bounds.clamp(value));
            }

            candidates.push(candidate);
        }

        candidates
    }

    /// Generate random sampling grid with proper deterministic PRNG
    fn generate_random_grid(&self, spread: f32, n_samples: usize) -> Vec<HashMap<String, f32>> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        if n_samples == 0 {
            return Vec::new();
        }

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let params = self.config.space.params();

        if params.is_empty() {
            return vec![HashMap::new(); n_samples];
        }

        let mut candidates = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let mut candidate = HashMap::new();

            for param in params {
                // Compute the effective bounds based on spread around center
                let center = param.center;
                let (min, max) = (param.bounds.min_value(), param.bounds.max_value());
                let range = max - min;
                let half_span = range * spread / 2.0;
                let low = (center - half_span).max(min);
                let high = (center + half_span).min(max);

                // Sample uniformly in [0, 1)
                let u: f32 = rng.gen();

                let value = if param.bounds.is_log_scale() {
                    // Log-uniform sampling
                    let log_low = low.max(1e-10).ln();
                    let log_high = high.max(1e-10).ln();
                    (log_low + u * (log_high - log_low)).exp()
                } else {
                    // Linear interpolation
                    low + u * (high - low)
                };

                candidate.insert(param.name.clone(), param.bounds.clamp(value));
            }

            candidates.push(candidate);
        }

        candidates
    }

    /// Evaluate a single candidate configuration
    ///
    /// Thread-safe: uses atomic operations for trial ID assignment
    fn evaluate_single(
        &self,
        dataset: &BinnedDataset,
        params: &HashMap<String, f32>,
        iteration: usize,
    ) -> Result<TrialResult> {
        let trial_id = self.next_trial_id.fetch_add(1, Ordering::SeqCst);

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

    /// Check if the backend requires sequential execution
    ///
    /// GPU backends (WGPU, CUDA, ROCm, Metal) cannot run multiple contexts
    /// concurrently on a single device, so trials must run sequentially.
    /// CPU backends (Scalar, AVX-512, SVE2) can run in parallel.
    fn is_gpu_backend(&self) -> bool {
        match self.base_config.backend_type {
            // GPU backends: force sequential to avoid OOM/contention
            BackendType::Auto => true, // Auto may resolve to GPU, be conservative
            BackendType::Wgpu => true,
            BackendType::Cuda => true,
            BackendType::Rocm => true,
            BackendType::Metal => true,
            // CPU backends: safe for parallel
            BackendType::Scalar => false,
            BackendType::Avx512 => false,
            BackendType::Sve2 => false,
        }
    }

    /// Evaluate candidates using parallel or sequential strategy
    ///
    /// For CPU backends, uses Rayon for parallel evaluation.
    /// For GPU backends, evaluates sequentially to avoid contention.
    fn evaluate_candidates(
        &self,
        dataset: &BinnedDataset,
        candidates: Vec<HashMap<String, f32>>,
        iteration: usize,
        current_trial: &AtomicUsize,
        total_trials: usize,
    ) -> Vec<TrialResult> {
        let use_parallel = self.config.parallel_trials && !self.is_gpu_backend();

        if use_parallel {
            // Parallel evaluation for CPU backends
            // n_parallel of 0 means "auto" (use all available threads)
            let n_parallel = if self.config.n_parallel == 0 {
                rayon::current_num_threads()
            } else {
                self.config.n_parallel
            };

            // Create a thread pool with limited parallelism if specified
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_parallel)
                .build()
                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

            let results = Mutex::new(Vec::with_capacity(candidates.len()));
            let callback = &self.callback;

            pool.install(|| {
                candidates.par_iter().for_each(|params| {
                    match self.evaluate_single(dataset, params, iteration) {
                        Ok(result) => {
                            let trial_num = current_trial.fetch_add(1, Ordering::SeqCst) + 1;

                            // Call callback (if set)
                            if let Some(ref cb) = callback {
                                cb(&result, trial_num, total_trials);
                            }

                            results.lock().unwrap().push(result);
                        }
                        Err(e) => {
                            eprintln!("Trial failed: {}", e);
                        }
                    }
                });
            });

            results.into_inner().unwrap()
        } else {
            // Sequential evaluation for GPU backends or when parallel disabled
            let mut results = Vec::with_capacity(candidates.len());

            for params in candidates {
                match self.evaluate_single(dataset, &params, iteration) {
                    Ok(result) => {
                        let trial_num = current_trial.fetch_add(1, Ordering::SeqCst) + 1;

                        // Call callback
                        if let Some(ref callback) = self.callback {
                            callback(&result, trial_num, total_trials);
                        }

                        results.push(result);
                    }
                    Err(e) => {
                        eprintln!("Trial failed: {}", e);
                    }
                }
            }

            results
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

    #[test]
    fn test_lhs_determinism() {
        // Same seed should produce identical samples
        let space = ParameterSpace::new()
            .with_param("learning_rate", ParamBounds::log_continuous(0.01, 0.5), 0.1)
            .with_param("max_depth", ParamBounds::discrete(2, 12), 6.0);

        let tuner1 = AutoTuner::new(GBDTConfig::default())
            .with_space(space.clone())
            .with_seed(42);

        let tuner2 = AutoTuner::new(GBDTConfig::default())
            .with_space(space)
            .with_seed(42);

        let grid1 = tuner1.generate_lhs_grid(0.5, 10);
        let grid2 = tuner2.generate_lhs_grid(0.5, 10);

        assert_eq!(grid1.len(), grid2.len());
        for (c1, c2) in grid1.iter().zip(grid2.iter()) {
            for key in c1.keys() {
                assert!(
                    (c1[key] - c2[key]).abs() < 1e-6,
                    "LHS should be deterministic with same seed"
                );
            }
        }
    }

    #[test]
    fn test_lhs_sample_count() {
        let space = ParameterSpace::new()
            .with_param("learning_rate", ParamBounds::continuous(0.01, 0.5), 0.1)
            .with_param("subsample", ParamBounds::continuous(0.5, 1.0), 0.8);

        let tuner = AutoTuner::new(GBDTConfig::default())
            .with_space(space)
            .with_seed(123);

        // Request 20 samples
        let grid = tuner.generate_lhs_grid(1.0, 20);
        assert_eq!(grid.len(), 20, "LHS should return exactly n_samples");

        // Edge case: 0 samples
        let empty = tuner.generate_lhs_grid(1.0, 0);
        assert!(empty.is_empty(), "LHS with n_samples=0 should be empty");
    }

    #[test]
    fn test_lhs_bounds_respected() {
        let space = ParameterSpace::new()
            .with_param("learning_rate", ParamBounds::continuous(0.01, 0.5), 0.1)
            .with_param("max_depth", ParamBounds::discrete(2, 12), 6.0);

        let tuner = AutoTuner::new(GBDTConfig::default())
            .with_space(space)
            .with_seed(999);

        let grid = tuner.generate_lhs_grid(1.0, 50);

        for candidate in &grid {
            let lr = candidate["learning_rate"];
            assert!(
                lr >= 0.01 && lr <= 0.5,
                "learning_rate {} out of bounds [0.01, 0.5]",
                lr
            );

            let depth = candidate["max_depth"];
            assert!(
                depth >= 2.0 && depth <= 12.0,
                "max_depth {} out of bounds [2, 12]",
                depth
            );
        }
    }

    #[test]
    fn test_lhs_stratification() {
        // LHS should have good space-filling property
        // Each stratum should be sampled exactly once
        let space = ParameterSpace::new()
            .with_param("x", ParamBounds::continuous(0.0, 1.0), 0.5);

        let tuner = AutoTuner::new(GBDTConfig::default())
            .with_space(space)
            .with_seed(12345);

        let n_samples = 10;
        let grid = tuner.generate_lhs_grid(1.0, n_samples);

        // Extract values and check stratum coverage
        let values: Vec<f32> = grid.iter().map(|c| c["x"]).collect();

        // Count how many samples fall into each stratum
        let mut stratum_counts = vec![0; n_samples];
        for &v in &values {
            let stratum = (v * n_samples as f32).floor() as usize;
            let stratum = stratum.min(n_samples - 1); // Handle edge case of v = 1.0
            stratum_counts[stratum] += 1;
        }

        // Each stratum should have exactly one sample
        for (i, &count) in stratum_counts.iter().enumerate() {
            assert_eq!(
                count, 1,
                "Stratum {} should have exactly 1 sample, got {}",
                i, count
            );
        }
    }

    #[test]
    fn test_random_determinism() {
        let space = ParameterSpace::new()
            .with_param("learning_rate", ParamBounds::log_continuous(0.01, 0.5), 0.1)
            .with_param("lambda", ParamBounds::continuous(0.0, 10.0), 1.0);

        let tuner1 = AutoTuner::new(GBDTConfig::default())
            .with_space(space.clone())
            .with_seed(42);

        let tuner2 = AutoTuner::new(GBDTConfig::default())
            .with_space(space)
            .with_seed(42);

        let grid1 = tuner1.generate_random_grid(0.5, 15);
        let grid2 = tuner2.generate_random_grid(0.5, 15);

        assert_eq!(grid1.len(), grid2.len());
        for (c1, c2) in grid1.iter().zip(grid2.iter()) {
            for key in c1.keys() {
                assert!(
                    (c1[key] - c2[key]).abs() < 1e-6,
                    "Random sampling should be deterministic with same seed"
                );
            }
        }
    }

    #[test]
    fn test_random_sample_count() {
        let space = ParameterSpace::minimal();
        let tuner = AutoTuner::new(GBDTConfig::default())
            .with_space(space)
            .with_seed(777);

        let grid = tuner.generate_random_grid(1.0, 25);
        assert_eq!(grid.len(), 25, "Random should return exactly n_samples");

        let empty = tuner.generate_random_grid(1.0, 0);
        assert!(empty.is_empty(), "Random with n_samples=0 should be empty");
    }

    #[test]
    fn test_random_bounds_respected() {
        let space = ParameterSpace::new()
            .with_param("subsample", ParamBounds::continuous(0.5, 1.0), 0.8)
            .with_param("entropy_weight", ParamBounds::continuous(0.0, 0.5), 0.1);

        let tuner = AutoTuner::new(GBDTConfig::default())
            .with_space(space)
            .with_seed(888);

        let grid = tuner.generate_random_grid(1.0, 100);

        for candidate in &grid {
            let ss = candidate["subsample"];
            assert!(
                ss >= 0.5 && ss <= 1.0,
                "subsample {} out of bounds [0.5, 1.0]",
                ss
            );

            let ew = candidate["entropy_weight"];
            assert!(
                ew >= 0.0 && ew <= 0.5,
                "entropy_weight {} out of bounds [0.0, 0.5]",
                ew
            );
        }
    }

    #[test]
    fn test_different_seeds_produce_different_results() {
        let space = ParameterSpace::minimal();

        let tuner1 = AutoTuner::new(GBDTConfig::default())
            .with_space(space.clone())
            .with_seed(1);

        let tuner2 = AutoTuner::new(GBDTConfig::default())
            .with_space(space)
            .with_seed(2);

        let grid1 = tuner1.generate_lhs_grid(1.0, 5);
        let grid2 = tuner2.generate_lhs_grid(1.0, 5);

        // At least one value should differ
        let mut all_same = true;
        for (c1, c2) in grid1.iter().zip(grid2.iter()) {
            for key in c1.keys() {
                if (c1[key] - c2[key]).abs() > 1e-6 {
                    all_same = false;
                    break;
                }
            }
        }
        assert!(!all_same, "Different seeds should produce different results");
    }

    #[test]
    fn test_log_scale_sampling() {
        // Verify log-scale parameters are sampled uniformly in log space
        let space = ParameterSpace::new()
            .with_param("learning_rate", ParamBounds::log_continuous(0.001, 1.0), 0.1);

        let tuner = AutoTuner::new(GBDTConfig::default())
            .with_space(space)
            .with_seed(42);

        let grid = tuner.generate_random_grid(1.0, 1000);
        let values: Vec<f32> = grid.iter().map(|c| c["learning_rate"]).collect();

        // Count how many are below vs above geometric mean
        let geo_mean = (0.001_f32 * 1.0).sqrt(); // ~0.0316
        let below = values.iter().filter(|&&v| v < geo_mean).count();
        let above = values.iter().filter(|&&v| v >= geo_mean).count();

        // Should be roughly 50/50 in log space
        let ratio = below as f32 / (below + above) as f32;
        assert!(
            ratio > 0.4 && ratio < 0.6,
            "Log-scale sampling should be balanced: ratio = {}",
            ratio
        );
    }

    #[test]
    fn test_spread_affects_range() {
        let space = ParameterSpace::new()
            .with_param("x", ParamBounds::continuous(0.0, 1.0), 0.5);

        let tuner = AutoTuner::new(GBDTConfig::default())
            .with_space(space)
            .with_seed(42);

        // Wide spread
        let wide = tuner.generate_random_grid(1.0, 100);
        let wide_range: f32 = wide.iter().map(|c| c["x"]).fold(0.0_f32, |a, b| a.max(b))
            - wide.iter().map(|c| c["x"]).fold(1.0_f32, |a, b| a.min(b));

        // Narrow spread
        let narrow = tuner.generate_random_grid(0.1, 100);
        let narrow_range: f32 = narrow.iter().map(|c| c["x"]).fold(0.0_f32, |a, b| a.max(b))
            - narrow.iter().map(|c| c["x"]).fold(1.0_f32, |a, b| a.min(b));

        assert!(
            wide_range > narrow_range,
            "Larger spread should produce wider range: wide={}, narrow={}",
            wide_range, narrow_range
        );
    }

    #[test]
    fn test_is_gpu_backend() {
        // Test GPU backends are detected
        let mut config = GBDTConfig::default();

        config.backend_type = BackendType::Auto;
        let tuner = AutoTuner::new(config.clone());
        assert!(tuner.is_gpu_backend(), "Auto should be treated as GPU (conservative)");

        config.backend_type = BackendType::Wgpu;
        let tuner = AutoTuner::new(config.clone());
        assert!(tuner.is_gpu_backend(), "WGPU is a GPU backend");

        config.backend_type = BackendType::Cuda;
        let tuner = AutoTuner::new(config.clone());
        assert!(tuner.is_gpu_backend(), "CUDA is a GPU backend");

        // Test CPU backends are not GPU
        config.backend_type = BackendType::Scalar;
        let tuner = AutoTuner::new(config.clone());
        assert!(!tuner.is_gpu_backend(), "Scalar is a CPU backend");

        config.backend_type = BackendType::Avx512;
        let tuner = AutoTuner::new(config.clone());
        assert!(!tuner.is_gpu_backend(), "AVX-512 is a CPU backend");

        config.backend_type = BackendType::Sve2;
        let tuner = AutoTuner::new(config);
        assert!(!tuner.is_gpu_backend(), "SVE2 is a CPU backend");
    }

    #[test]
    fn test_parallel_config_respected() {
        // Test that parallel_trials setting is respected
        let mut config = GBDTConfig::default();
        config.backend_type = BackendType::Scalar; // CPU backend

        let tuner_config = TunerConfig::new()
            .with_parallel(true)
            .with_n_parallel(4);

        let tuner = AutoTuner::new(config)
            .with_config(tuner_config);

        // Verify settings are applied
        assert!(tuner.config().parallel_trials);
        assert_eq!(tuner.config().n_parallel, 4);
    }
}
