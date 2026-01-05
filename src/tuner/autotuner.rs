//! AutoTuner for hyperparameter optimization
//!
//! Provides the main `AutoTuner` struct and implementation for
//! iterative grid search with auto-zoom.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use polars::prelude::*;
use rayon::prelude::*;

use crate::backend::BackendType;
use crate::booster::{GBDTConfig, GBDTModel};
use crate::dataset::{split_holdout, split_kfold, BinnedDataset};
use crate::{Result, TreeBoostError};

use super::config::{
    EvalStrategy, GridStrategy, ParamBounds, ParamDef, ParameterSpace, TunerConfig, TuningMode,
};
use super::history::{ProgressCallback, SearchHistory};
use super::logger::{
    finalize_logging, init_logger, log_trial, save_model_formats, start_iteration_logging,
    SharedLogger,
};
use super::metrics::Metric;
use super::realistic::{
    encode_full_dataset, encode_train_val_split, split_dataframe_by_indices, RealisticModeConfig,
};
use super::trial::TrialResult;

// =============================================================================
// Constants
// =============================================================================

/// Maximum consecutive zone switches before abandoning search
const MAX_ZONE_SWITCH_FAILS: usize = 3;

/// Minimum early stopping trees for tuner trials (lower than default for faster iteration).
/// The default GBDTConfig uses 20 trees, but tuning needs faster feedback per trial.
/// See `crate::booster::config::DEFAULT_MIN_EARLY_STOPPING_TREES` for the production default.
const TUNER_MIN_EARLY_STOPPING_TREES: usize = 5;

/// Binary classification threshold for F1 score computation
const BINARY_CLASSIFICATION_THRESHOLD: f32 = 0.5;

// =============================================================================
// Evaluation Data Types
// =============================================================================

/// Input data for evaluation (unifies optimistic and realistic modes)
enum EvalInput<'a> {
    /// Pre-binned dataset (optimistic mode - faster, may have target leakage)
    Optimistic(&'a BinnedDataset),
    /// Raw DataFrame with encoding config (realistic mode - no target leakage)
    Realistic {
        raw_data: &'a DataFrame,
        config: &'a RealisticModeConfig,
    },
}

/// Evaluation metrics tuple: (val_metric, train_metric, num_trees, f1_score, roc_auc)
type EvalMetrics = (f32, f32, usize, Option<f32>, Option<f64>);

/// Result of model evaluation
type EvalResult = Result<EvalMetrics>;

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute evaluation metrics for a trained model
fn compute_eval_metrics(
    model: &GBDTModel,
    train_dataset: &BinnedDataset,
    val_dataset: &BinnedDataset,
    val_targets: &[f32],
    metric: &Metric,
    config: &GBDTConfig,
    tuner: &AutoTuner,
) -> (f32, f32, Option<f32>, Option<f64>) {
    let train_preds = model.predict(train_dataset);
    let val_preds = model.predict(val_dataset);

    let train_metric = metric.compute(&train_preds, train_dataset.targets());
    let val_metric = metric.compute(&val_preds, val_targets);
    let f1_score = if tuner.config.task_type.is_classification() {
        tuner.compute_f1_score(config, &val_preds, val_targets)
    } else {
        None
    };

    // Compute ROC-AUC for binary classification
    let roc_auc = if tuner.config.task_type.is_binary() {
        Some(super::metrics::compute_roc_auc(&val_preds, val_targets))
    } else {
        None
    };

    (val_metric, train_metric, f1_score, roc_auc)
}

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

    // Realistic mode support (encoding per split)
    /// Raw data for realistic mode (stored as Arc for sharing)
    raw_data: Option<std::sync::Arc<DataFrame>>,
    /// Realistic mode encoding configuration
    realistic_config: Option<RealisticModeConfig>,
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
            raw_data: None,
            realistic_config: None,
        }
    }

    /// Set the tuner configuration
    pub fn with_config(mut self, config: TunerConfig) -> Self {
        // Update history to use the configured optimization metric
        self.history = SearchHistory::with_metric(config.optimization_metric);
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

    /// Run the tuning process (optimistic mode - uses pre-encoded data)
    ///
    /// This is the fast mode that uses pre-encoded `BinnedDataset`.
    /// May have target leakage if target encoding was applied before binning.
    ///
    /// For accurate F1 estimates with categorical features, use `tune_dataframe()` instead.
    ///
    /// # Arguments
    /// * `dataset` - Pre-binned dataset (reused across all trials)
    ///
    /// # Returns
    /// * Best GBDTConfig found and the complete search history
    pub fn tune(&mut self, dataset: &BinnedDataset) -> Result<(GBDTConfig, SearchHistory)> {
        // Store dataset reference for evaluation (wrapped in a temporary storage)
        // We use a simple approach: store dataset pointer and retrieve in evaluate functions
        self.run_tune_with_dataset(dataset)
    }

    /// Internal tune method that works with BinnedDataset
    fn run_tune_with_dataset(&mut self, dataset: &BinnedDataset) -> Result<(GBDTConfig, SearchHistory)> {
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
            println!("  Tuning mode: {:?}", self.config.tuning_mode);
            println!(
                "  Parallel: {} (backend: {:?})",
                if use_parallel { "enabled" } else { "disabled" },
                self.base_config.backend_type
            );
        }

        let current_trial = AtomicUsize::new(0);
        let start_time = Instant::now();

        // Initialize trial logger if output_dir is configured
        let logger = init_logger(
            &self.config.output_dir,
            self.config.space.param_names(),
            self.config.verbose,
        )?;

        // Use while loop instead of for loop to allow extending iterations when unbalanced
        let mut iteration = 0;
        let mut zoom_level = 0; // Separate from iteration - resets when we switch zones
        let mut zone_switch_fails = 0; // Count consecutive failed zone switches
        let max_iterations = self.config.n_iterations;

        while iteration < max_iterations {
            let spread = self.config.spread_for_iteration(zoom_level);

            // Start new CSV file for this iteration
            start_iteration_logging(&logger, iteration)?;

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
            // Results are logged immediately inside evaluate_candidates via the shared logger
            let results = self.evaluate_candidates(
                dataset,
                candidates,
                iteration,
                &current_trial,
                total_trials,
                logger.as_ref(),
            );

            // Add all results to history and log new best
            for result in results {
                // Log if best so far (using configured optimization metric)
                if self.config.verbose {
                    let is_best = self
                        .history
                        .best()
                        .map(|b| self.history.compare_trials(&result, b))
                        .unwrap_or(true);

                    if is_best {
                        // Show all available metrics
                        let auc_str = result
                            .roc_auc
                            .map(|auc| format!(" AUC={:.4}", auc))
                            .unwrap_or_default();
                        let f1_str = result
                            .f1_score
                            .map(|f1| format!(" F1={:.2}%", f1 * 100.0))
                            .unwrap_or_default();
                        println!(
                            "  -> New best! loss={:.5}{}{} (depth={}, lr={:.4}, trees={})",
                            result.val_metric,
                            auc_str,
                            f1_str,
                            result.params.get("max_depth").unwrap_or(&0.0),
                            result.params.get("learning_rate").unwrap_or(&0.0),
                            result.num_trees,
                        );
                    }
                }

                self.history.add(result);
            }

            // Check if we found improvement using the configured optimization metric
            let improved = if let Some(best_after) = self.history.best() {
                // Find best trial from previous iterations
                let best_before_trial = self.history.trials().iter()
                    .filter(|t| t.iteration < iteration)
                    .max_by(|a, b| {
                        if self.history.compare_trials(a, b) {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    });

                match best_before_trial {
                    Some(before) => self.history.compare_trials(best_after, before),
                    None => true, // First iteration always improves
                }
            } else {
                false
            };

            // Update centers to winner's values
            if let Some(best) = self.history.best() {
                self.config.space.set_centers(&best.params);
            }

            // OUTER LOOP STOPPING: Diminishing returns check with F1 guard
            // For classification tasks, don't stop if F1 score is too low
            let current_f1 = self.history.best().and_then(|b| b.f1_score).unwrap_or(0.0);
            let is_balanced = current_f1 >= self.config.min_f1_score;

            // ZONE SWITCHING: If no improvement and model is unbalanced, switch zone immediately
            if improved {
                zoom_level += 1; // Zoom in when improving
                zone_switch_fails = 0; // Reset fail counter on any improvement
            } else if !is_balanced {
                // No improvement found - switch zone immediately
                zone_switch_fails += 1;

                if zone_switch_fails >= MAX_ZONE_SWITCH_FAILS {
                    if self.config.verbose {
                        println!("  {} consecutive zone switches failed, stopping search.", zone_switch_fails);
                    }
                    break;
                }

                if self.config.verbose {
                    println!("  No improvement found, switching zone ({}/{} fails)...",
                             zone_switch_fails, MAX_ZONE_SWITCH_FAILS);
                }
                // Reset zoom level to explore wider
                zoom_level = 0;

                // Randomize centers to explore different region
                self.randomize_centers();
            }

            // Stop if no improvement AND model is balanced
            if !improved && is_balanced && iteration > 0 {
                if self.config.verbose {
                    println!("  No improvement found, stopping early");
                }
                break;
            }

            // Stop if we've exhausted iterations
            if iteration + 1 >= max_iterations {
                break;
            }

            iteration += 1;
        }

        // Build final config from best trial
        let best = self.history.best().ok_or_else(|| {
            TreeBoostError::Training("No successful trials".into())
        })?;

        if self.config.verbose {
            println!("\n=== Tuning Complete ===");
            println!("  Total trials: {}", self.history.len());
            println!("  Best val_metric: {:.6}", best.val_metric);
            if let Some(f1) = best.f1_score {
                println!("  F1 score: {:.2}%", f1 * 100.0);
            }
            println!("  Best params:");
            for (k, v) in &best.params {
                println!("    {}: {:.4}", k, v);
            }
        }

        // Export final results if logging is enabled
        if logger.is_some() {
            let duration_secs = start_time.elapsed().as_secs_f64();
            let run_dir = finalize_logging(&logger, &self.history, best, duration_secs)?;

            // Train and save final model if enabled (optimistic mode)
            if !self.config.save_model_formats.is_empty() {
                if self.config.verbose {
                    println!("  Training final model on full dataset...");
                }

                // Build best config and train on full dataset
                let best_gbdt_config = self.build_config(&best.params);
                let final_model = GBDTModel::train_binned(dataset, best_gbdt_config)?;

                // Save model in all specified formats
                save_model_formats(&logger, &final_model, &self.config.save_model_formats)?;

                if self.config.verbose {
                    let formats: Vec<_> = self.config.save_model_formats.iter()
                        .map(|f| f.extension())
                        .collect();
                    println!("  Model saved ({} trees) in formats: {:?}", final_model.num_trees(), formats);
                }
            }

            if self.config.verbose {
                println!("  Results saved to: {}", run_dir.display());
            }
        }

        let best_config = self.build_config(&best.params);
        Ok((best_config, self.history.clone()))
    }

    /// Run tuning with realistic mode (encoding per train/val split)
    ///
    /// This method prevents target leakage by encoding data separately for each
    /// train/validation split. Slower than `tune()` but gives accurate F1 estimates.
    ///
    /// # Arguments
    /// * `df` - Raw DataFrame with all features and target column
    /// * `realistic_config` - Configuration for encoding (pipeline, target column, categorical columns)
    ///
    /// # Returns
    /// * Best GBDTConfig found and the complete search history
    ///
    /// # Example
    /// ```ignore
    /// let realistic_config = RealisticModeConfig::new(
    ///     PipelineConfig::new().with_num_bins(255),
    ///     "Survived",
    ///     Some(vec!["Sex".into(), "Embarked".into()]),
    /// );
    ///
    /// let (best_config, history) = tuner.tune_dataframe(df, realistic_config)?;
    /// ```
    pub fn tune_dataframe(
        &mut self,
        df: DataFrame,
        realistic_config: RealisticModeConfig,
    ) -> Result<(GBDTConfig, SearchHistory)> {
        // Store raw data and config for use in evaluation
        self.raw_data = Some(std::sync::Arc::new(df));
        self.realistic_config = Some(realistic_config);

        // Force realistic mode
        self.config.tuning_mode = TuningMode::Realistic;

        // Run the tuning loop (same as tune(), but evaluate methods will use realistic encoding)
        self.tune_internal()
    }

    /// Internal tuning loop (shared by tune and tune_dataframe)
    fn tune_internal(&mut self) -> Result<(GBDTConfig, SearchHistory)> {
        // Validate configuration
        self.config.validate().map_err(|e| {
            TreeBoostError::Config(format!("Invalid tuner configuration: {}", e))
        })?;

        let total_trials = self.config.estimated_trials();
        let use_parallel = self.config.parallel_trials && !self.is_gpu_backend();

        // Parallel not supported in realistic mode (encoding is stateful)
        let use_parallel = use_parallel && !self.config.tuning_mode.is_realistic();

        if self.config.verbose {
            println!("Starting AutoTuner...");
            println!("  Iterations: {}", self.config.n_iterations);
            println!("  Parameters: {}", self.config.space.len());
            println!("  Estimated trials: {}", total_trials);
            println!("  Grid strategy: {:?}", self.config.grid_strategy);
            println!("  Eval strategy: {:?}", self.config.eval_strategy);
            println!("  Tuning mode: {:?}", self.config.tuning_mode);
            println!(
                "  Parallel: {} (backend: {:?})",
                if use_parallel { "enabled" } else { "disabled" },
                self.base_config.backend_type
            );
        }

        let current_trial = AtomicUsize::new(0);
        let start_time = Instant::now();

        // Initialize trial logger if output_dir is configured
        let logger = init_logger(
            &self.config.output_dir,
            self.config.space.param_names(),
            self.config.verbose,
        )?;

        // Use while loop instead of for loop to allow extending iterations when unbalanced
        let mut iteration = 0;
        let mut zoom_level = 0; // Separate from iteration - resets when we switch zones
        let mut zone_switch_fails = 0; // Count consecutive failed zone switches
        let max_iterations = self.config.n_iterations;

        while iteration < max_iterations {
            let spread = self.config.spread_for_iteration(zoom_level);

            // Start new CSV file for this iteration
            start_iteration_logging(&logger, iteration)?;

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
            // Results are logged immediately inside evaluate_candidates_internal via the shared logger
            let results = self.evaluate_candidates_internal(
                candidates,
                iteration,
                &current_trial,
                total_trials,
                use_parallel,
                logger.as_ref(),
            )?;

            // Add all results to history and log new best
            for result in results {
                // Log if best so far (using configured optimization metric)
                if self.config.verbose {
                    let is_best = self
                        .history
                        .best()
                        .map(|b| self.history.compare_trials(&result, b))
                        .unwrap_or(true);

                    if is_best {
                        // Show all available metrics
                        let auc_str = result
                            .roc_auc
                            .map(|auc| format!(" AUC={:.4}", auc))
                            .unwrap_or_default();
                        let f1_str = result
                            .f1_score
                            .map(|f1| format!(" F1={:.2}%", f1 * 100.0))
                            .unwrap_or_default();
                        println!(
                            "  -> New best! loss={:.5}{}{} (depth={}, lr={:.4}, trees={})",
                            result.val_metric,
                            auc_str,
                            f1_str,
                            result.params.get("max_depth").unwrap_or(&0.0),
                            result.params.get("learning_rate").unwrap_or(&0.0),
                            result.num_trees,
                        );
                    }
                }

                self.history.add(result);
            }

            // Check if we found improvement using the configured optimization metric
            let improved = if let Some(best_after) = self.history.best() {
                // Find best trial from previous iterations
                let best_before_trial = self.history.trials().iter()
                    .filter(|t| t.iteration < iteration)
                    .max_by(|a, b| {
                        if self.history.compare_trials(a, b) {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    });

                match best_before_trial {
                    Some(before) => self.history.compare_trials(best_after, before),
                    None => true, // First iteration always improves
                }
            } else {
                false
            };

            // Update centers to winner's values
            if let Some(best) = self.history.best() {
                self.config.space.set_centers(&best.params);
            }

            // OUTER LOOP STOPPING: Diminishing returns check with F1 guard
            // For classification tasks, don't stop if F1 score is too low
            let current_f1 = self.history.best().and_then(|b| b.f1_score).unwrap_or(0.0);
            let is_balanced = current_f1 >= self.config.min_f1_score;

            // ZONE SWITCHING: If no improvement and model is unbalanced, switch zone immediately
            if improved {
                zoom_level += 1; // Zoom in when improving
                zone_switch_fails = 0; // Reset fail counter on any improvement
            } else if !is_balanced {
                // No improvement found - switch zone immediately
                zone_switch_fails += 1;

                if zone_switch_fails >= MAX_ZONE_SWITCH_FAILS {
                    if self.config.verbose {
                        println!("  {} consecutive zone switches failed, stopping search.", zone_switch_fails);
                    }
                    break;
                }

                if self.config.verbose {
                    println!("  No improvement found, switching zone ({}/{} fails)...",
                             zone_switch_fails, MAX_ZONE_SWITCH_FAILS);
                }
                // Reset zoom level to explore wider
                zoom_level = 0;

                // Randomize centers to explore different region
                self.randomize_centers();
            }

            // Stop if no improvement AND model is balanced
            if !improved && is_balanced && iteration > 0 {
                if self.config.verbose {
                    println!("  No improvement found, stopping early");
                }
                break;
            }

            // Stop if we've exhausted iterations
            if iteration + 1 >= max_iterations {
                break;
            }

            iteration += 1;
        }

        // Build final config from best trial
        let best = self.history.best().ok_or_else(|| {
            TreeBoostError::Training("No successful trials".into())
        })?;

        if self.config.verbose {
            println!("\n=== Tuning Complete ===");
            println!("  Total trials: {}", self.history.len());
            println!("  Best val_metric: {:.6}", best.val_metric);
            if let Some(f1) = best.f1_score {
                println!("  F1 score: {:.2}%", f1 * 100.0);
            }
            println!("  Best params:");
            for (k, v) in &best.params {
                println!("    {}: {:.4}", k, v);
            }
        }

        // Export final results if logging is enabled
        if logger.is_some() {
            let duration_secs = start_time.elapsed().as_secs_f64();
            let run_dir = finalize_logging(&logger, &self.history, best, duration_secs)?;

            // Train and save final model if enabled
            if !self.config.save_model_formats.is_empty() {
                match (&self.raw_data, &self.realistic_config) {
                    (Some(ref raw_data), Some(ref realistic_cfg)) => {
                        if self.config.verbose {
                            println!("  Training final model on full dataset...");
                        }

                        // Encode full dataset
                        let full_df = (**raw_data).clone();
                        let full_dataset = encode_full_dataset(full_df, realistic_cfg)?;

                        // Build best config and train
                        let best_gbdt_config = self.build_config(&best.params);
                        let final_model = GBDTModel::train_binned(&full_dataset, best_gbdt_config)?;

                        // Save model in all specified formats
                        save_model_formats(&logger, &final_model, &self.config.save_model_formats)?;

                        if self.config.verbose {
                            let formats: Vec<_> = self.config.save_model_formats.iter()
                                .map(|f| f.extension())
                                .collect();
                            println!("  Model saved ({} trees) in formats: {:?}", final_model.num_trees(), formats);
                        }
                    }
                    _ => {
                        // This shouldn't happen in realistic mode, but warn if it does
                        eprintln!("  Warning: Model saving skipped - realistic mode requires raw_data and realistic_config");
                    }
                }
            }

            if self.config.verbose {
                println!("  Results saved to: {}", run_dir.display());
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
                let high = high.div_ceil(*step) * step;

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

        // Generate samples - iterate by sample index, accessing each param's permutation
        let mut candidates = Vec::with_capacity(n_samples);
        #[allow(clippy::needless_range_loop)]
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

    /// Evaluate a single candidate configuration (unified for both modes)
    ///
    /// Thread-safe: uses atomic operations for trial ID assignment.
    /// Handles both optimistic (pre-binned) and realistic (per-split encoding) modes.
    fn evaluate_single(
        &self,
        input: EvalInput<'_>,
        params: &HashMap<String, f32>,
        iteration: usize,
    ) -> Result<TrialResult> {
        let trial_id = self.next_trial_id.fetch_add(1, Ordering::SeqCst);
        let start = Instant::now();

        // Dispatch to appropriate strategy based on input mode
        let (val_metric, train_metric, num_trees, f1_score, roc_auc) = match input {
            EvalInput::Optimistic(dataset) => match self.config.eval_strategy {
                EvalStrategy::Holdout {
                    validation_ratio,
                    folds,
                } => {
                    self.evaluate_holdout_with_folds(dataset, params, validation_ratio, folds)?
                }
                EvalStrategy::Conformal {
                    calibration_ratio,
                    quantile,
                    folds,
                } => {
                    self.evaluate_conformal_with_folds(
                        dataset,
                        params,
                        calibration_ratio,
                        quantile,
                        folds,
                    )?
                }
            },
            EvalInput::Realistic { raw_data, config } => match self.config.eval_strategy {
                EvalStrategy::Holdout {
                    validation_ratio,
                    folds,
                } => {
                    self.evaluate_holdout_realistic_with_folds(
                        raw_data,
                        config,
                        params,
                        validation_ratio,
                        folds,
                    )?
                }
                EvalStrategy::Conformal {
                    calibration_ratio,
                    quantile,
                    folds,
                } => {
                    self.evaluate_conformal_realistic_with_folds(
                        raw_data,
                        config,
                        params,
                        calibration_ratio,
                        quantile,
                        folds,
                    )?
                }
            },
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
            f1_score,
            roc_auc,
        })
    }

    /// Evaluate using holdout validation with optional k-fold
    ///
    /// If folds == 1, uses simple holdout. If folds > 1, runs k-fold CV.
    fn evaluate_holdout_with_folds(
        &self,
        dataset: &BinnedDataset,
        params: &HashMap<String, f32>,
        validation_ratio: f32,
        folds: usize,
    ) -> EvalResult {
        if folds == 1 {
            self.evaluate_holdout(dataset, params, validation_ratio)
        } else {
            self.evaluate_kfold(dataset, params, folds)
        }
    }

    /// Evaluate using holdout validation (single fold)
    ///
    /// Returns: (val_metric, train_metric, num_trees, f1_score, roc_auc)
    fn evaluate_holdout(
        &self,
        dataset: &BinnedDataset,
        params: &HashMap<String, f32>,
        validation_ratio: f32,
    ) -> EvalResult {
        // Build config with proper validation for early stopping
        // Use tuner's seed for consistency between training and evaluation splits
        let mut config = self.build_config(params);
        config.validation_ratio = validation_ratio;
        config.seed = self.config.seed; // Ensure same split for training and eval

        // Train model (GBDTModel handles internal train/val split)
        let model = GBDTModel::train_binned(dataset, config.clone())?;

        // Get predictions on full dataset
        // TODO: Could optimize by only predicting on validation set
        let predictions = model.predict(dataset);
        let targets = dataset.targets();

        // Create split for evaluation (using same seed as training for consistency)
        let split = split_holdout(dataset.num_rows(), validation_ratio, 0.0, self.config.seed);
        let metric = self.select_metric(&config);

        // Compute metrics using shared helper
        let (val_metric, train_metric, f1_score, roc_auc) = self.compute_metrics_by_indices(
            &predictions, targets, &split.train, &split.validation, &metric, &config,
        );

        Ok((val_metric, train_metric, model.num_trees(), f1_score, roc_auc))
    }

    /// Evaluate using K-fold cross-validation
    ///
    /// Returns: (val_metric, train_metric, num_trees, f1_score, roc_auc)
    fn evaluate_kfold(
        &self,
        dataset: &BinnedDataset,
        params: &HashMap<String, f32>,
        k: usize,
    ) -> EvalResult {
        let kfold = split_kfold(dataset.num_rows(), k, self.config.seed);
        let config = self.build_config(params);
        let metric = self.select_metric(&config);

        let mut fold_results = Vec::with_capacity(k);

        for fold_idx in 0..k {
            let (train_idx, val_idx) = kfold.get_fold(fold_idx);

            // Train model on full dataset with internal validation
            // Note: For true K-fold, we'd need index-based training
            // Current approach: train on full data, evaluate on fold splits
            let model = GBDTModel::train_binned(dataset, config.clone())?;
            let predictions = model.predict(dataset);
            let targets = dataset.targets();

            // Compute metrics using shared helper
            let (val_metric, train_metric, f1_score, roc_auc) = self.compute_metrics_by_indices(
                &predictions, targets, &train_idx, &val_idx, &metric, &config,
            );

            fold_results.push((val_metric, train_metric, model.num_trees(), f1_score, roc_auc));
        }

        Ok(Self::aggregate_fold_results(&fold_results))
    }

    /// Evaluate using conformal prediction with optional k-fold
    ///
    /// If folds == 1, uses simple conformal. If folds > 1, runs conformal k-fold CV.
    fn evaluate_conformal_with_folds(
        &self,
        dataset: &BinnedDataset,
        params: &HashMap<String, f32>,
        calibration_ratio: f32,
        quantile: f32,
        folds: usize,
    ) -> EvalResult {
        if folds == 1 {
            self.evaluate_conformal(dataset, params, calibration_ratio, quantile)
        } else {
            // Run conformal on each fold and average
            let kfold = split_kfold(dataset.num_rows(), folds, self.config.seed);
            let mut fold_results = Vec::with_capacity(folds);

            for fold_idx in 0..folds {
                let (_train_idx, _val_idx) = kfold.get_fold(fold_idx);
                // For now, use full dataset training (same limitation as regular k-fold)
                // TODO: Add BinnedDataset::subset_by_indices for proper k-fold
                let result = self.evaluate_conformal(dataset, params, calibration_ratio, quantile)?;
                fold_results.push(result);
            }

            Ok(Self::aggregate_fold_results(&fold_results))
        }
    }

    /// Evaluate using conformal prediction (O(1) metric lookup)
    ///
    /// Instead of computing MSE over a validation set, this uses the conformal
    /// quantile `q` as the optimization metric. Lower `q` = tighter intervals
    /// = more confident model.
    ///
    /// This is O(1) because `q` is already computed during training and stored
    /// in the model. No prediction loop is needed.
    ///
    /// # Arguments
    /// * `dataset` - The binned dataset
    /// * `params` - Hyperparameters to evaluate
    /// * `calibration_ratio` - Fraction for calibration set
    /// * `quantile` - Coverage quantile (e.g., 0.9 for 90%)
    ///
    /// # Returns
    /// * `val_metric` - The conformal quantile `q` (lower = better)
    /// * `train_metric` - MSE on training set (for reference)
    /// * `num_trees` - Number of trees in the model
    /// * `f1_score` - None (conformal is typically used for regression)
    fn evaluate_conformal(
        &self,
        dataset: &BinnedDataset,
        params: &HashMap<String, f32>,
        calibration_ratio: f32,
        quantile: f32,
    ) -> EvalResult {
        // Build config with conformal calibration
        let mut config = self.build_config(params);
        config.calibration_ratio = calibration_ratio;
        config.conformal_quantile = quantile;

        // Train model (this computes conformal quantile internally)
        let model = GBDTModel::train_binned(dataset, config)?;

        // O(1) metric: just read the conformal quantile
        // Lower q = tighter intervals = better model
        let val_metric = model.conformal_quantile().unwrap_or(f32::MAX);

        // Optionally compute training MSE for reference (can skip for speed)
        // For now, use q as both metrics (training metric is less meaningful here)
        let train_metric = val_metric;

        // For classification tasks, also compute F1 and ROC-AUC on calibration set
        // Split dataset to get calibration targets for metrics
        let cal_size = (dataset.num_rows() as f32 * calibration_ratio) as usize;
        let cal_start = dataset.num_rows() - cal_size;
        let cal_targets: Vec<f32> = dataset.targets()[cal_start..].to_vec();
        let all_preds = model.predict(dataset);
        let cal_preds: Vec<f32> = all_preds[cal_start..].to_vec();

        let f1_score = if self.config.task_type.is_classification() {
            self.compute_f1_score(&self.build_config(params), &cal_preds, &cal_targets)
        } else {
            None
        };
        let roc_auc = if self.config.task_type.is_binary() {
            Some(super::metrics::compute_roc_auc(&cal_preds, &cal_targets))
        } else {
            None
        };

        Ok((val_metric, train_metric, model.num_trees(), f1_score, roc_auc))
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

    /// Compute F1 score for classification tasks
    ///
    /// F1 = 2 * (precision * recall) / (precision + recall)
    /// - Precision = TP / (TP + FP)
    /// - Recall = TP / (TP + FN)
    ///
    /// **Note:** Uses 0.5 as decision threshold, assuming binary class labels {0, 1}.
    /// For highly imbalanced datasets, consider using a custom threshold or
    /// alternative evaluation metric.
    ///
    /// Returns `None` for regression tasks or if predictions/targets are misaligned.
    fn compute_f1_score(
        &self,
        config: &GBDTConfig,
        predictions: &[f32],
        targets: &[f32],
    ) -> Option<f32> {
        use crate::booster::LossType;

        // Only compute for binary classification
        if !matches!(config.loss_type, LossType::BinaryLogLoss) {
            return None;
        }

        if predictions.is_empty() || predictions.len() != targets.len() {
            return None;
        }

        // For binary classification: predictions are log-odds, apply sigmoid
        // Then threshold at 0.5 for predicted class
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            // Convert log-odds to probability via sigmoid
            let prob = 1.0 / (1.0 + (-pred).exp());
            let pred_class = if prob >= BINARY_CLASSIFICATION_THRESHOLD { 1.0 } else { 0.0 };
            let actual_class = if target >= BINARY_CLASSIFICATION_THRESHOLD { 1.0 } else { 0.0 };

            match (pred_class as u8, actual_class as u8) {
                (1, 1) => true_positives += 1,
                (1, 0) => false_positives += 1,
                (0, 1) => false_negatives += 1,
                _ => {} // true negatives not needed for F1
            }
        }

        // Precision = TP / (TP + FP)
        let precision = if true_positives + false_positives > 0 {
            true_positives as f32 / (true_positives + false_positives) as f32
        } else {
            0.0 // No positive predictions
        };

        // Recall = TP / (TP + FN)
        let recall = if true_positives + false_negatives > 0 {
            true_positives as f32 / (true_positives + false_negatives) as f32
        } else {
            0.0 // No actual positives
        };

        // F1 = 2 * (precision * recall) / (precision + recall)
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0 // Both precision and recall are 0
        };

        Some(f1)
    }

    // =========================================================================
    // Evaluation Helpers (shared by holdout/kfold/conformal strategies)
    // =========================================================================

    /// Train model with external validation and compute metrics
    ///
    /// This is the core training loop for realistic mode evaluation.
    /// Handles config setup, training with external validation, and metric computation.
    ///
    /// Returns: (val_metric, train_metric, num_trees, f1_score)
    fn train_and_evaluate(
        &self,
        train_dataset: &BinnedDataset,
        val_dataset: &BinnedDataset,
        val_targets: &[f32],
        params: &HashMap<String, f32>,
    ) -> EvalResult {
        let mut config = self.build_config(params);
        config.validation_ratio = 0.0;
        config.min_early_stopping_trees = TUNER_MIN_EARLY_STOPPING_TREES;

        let model = GBDTModel::train_binned_with_validation(
            train_dataset,
            val_dataset,
            val_targets,
            config.clone(),
        )?;

        let metric = self.select_metric(&config);
        let (val_metric, train_metric, f1_score, roc_auc) =
            compute_eval_metrics(&model, train_dataset, val_dataset, val_targets, &metric, &config, self);

        Ok((val_metric, train_metric, model.num_trees(), f1_score, roc_auc))
    }

    /// Train model with conformal config and return quantile metric
    ///
    /// Specialized version for conformal prediction evaluation.
    /// Uses conformal quantile as the optimization metric instead of MSE/logloss.
    ///
    /// Returns: (conformal_quantile, conformal_quantile, num_trees, f1_score, roc_auc)
    fn train_and_evaluate_conformal(
        &self,
        train_dataset: &BinnedDataset,
        val_dataset: &BinnedDataset,
        val_targets: &[f32],
        params: &HashMap<String, f32>,
        quantile: f32,
    ) -> EvalResult {
        let mut config = self.build_config(params);
        config.validation_ratio = 0.0;
        config.conformal_quantile = quantile;

        let model = GBDTModel::train_binned_with_validation(
            train_dataset,
            val_dataset,
            val_targets,
            config.clone(),
        )?;

        // O(1) metric: just read the conformal quantile
        let val_metric = model.conformal_quantile().unwrap_or(f32::MAX);
        let train_metric = val_metric;

        // Compute F1 and ROC-AUC for classification tasks
        let val_preds = model.predict(val_dataset);
        let f1_score = if self.config.task_type.is_classification() {
            self.compute_f1_score(&config, &val_preds, val_targets)
        } else {
            None
        };
        let roc_auc = if self.config.task_type.is_binary() {
            Some(super::metrics::compute_roc_auc(&val_preds, val_targets))
        } else {
            None
        };

        Ok((val_metric, train_metric, model.num_trees(), f1_score, roc_auc))
    }

    /// Aggregate results from multiple folds
    ///
    /// Computes average metrics across k-fold results.
    fn aggregate_fold_results(
        results: &[EvalMetrics],
    ) -> EvalMetrics {
        let k = results.len();
        if k == 0 {
            return (f32::MAX, f32::MAX, 0, None, None);
        }

        let avg_val = results.iter().map(|r| r.0).sum::<f32>() / k as f32;
        let avg_train = results.iter().map(|r| r.1).sum::<f32>() / k as f32;
        let avg_trees = results.iter().map(|r| r.2).sum::<usize>() / k;

        let f1_scores: Vec<f32> = results.iter().filter_map(|r| r.3).collect();
        let avg_f1 = if f1_scores.is_empty() {
            None
        } else {
            Some(f1_scores.iter().sum::<f32>() / f1_scores.len() as f32)
        };

        let roc_aucs: Vec<f64> = results.iter().filter_map(|r| r.4).collect();
        let avg_roc_auc = if roc_aucs.is_empty() {
            None
        } else {
            Some(roc_aucs.iter().sum::<f64>() / roc_aucs.len() as f64)
        };

        (avg_val, avg_train, avg_trees, avg_f1, avg_roc_auc)
    }

    /// Compute metrics by splitting predictions according to indices (optimistic mode)
    ///
    /// Used when training on full dataset and splitting predictions for evaluation.
    /// Returns: (val_metric, train_metric, f1_score)
    fn compute_metrics_by_indices(
        &self,
        predictions: &[f32],
        targets: &[f32],
        train_idx: &[usize],
        val_idx: &[usize],
        metric: &Metric,
        config: &GBDTConfig,
    ) -> (f32, f32, Option<f32>, Option<f64>) {
        let train_preds: Vec<f32> = train_idx.iter().map(|&i| predictions[i]).collect();
        let train_targets: Vec<f32> = train_idx.iter().map(|&i| targets[i]).collect();
        let train_metric = metric.compute(&train_preds, &train_targets);

        let val_preds: Vec<f32> = val_idx.iter().map(|&i| predictions[i]).collect();
        let val_targets: Vec<f32> = val_idx.iter().map(|&i| targets[i]).collect();
        let val_metric = metric.compute(&val_preds, &val_targets);

        let f1_score = if self.config.task_type.is_classification() {
            self.compute_f1_score(config, &val_preds, &val_targets)
        } else {
            None
        };

        // Compute ROC-AUC for binary classification
        let roc_auc = if self.config.task_type.is_binary() {
            Some(super::metrics::compute_roc_auc(&val_preds, &val_targets))
        } else {
            None
        };

        (val_metric, train_metric, f1_score, roc_auc)
    }

    /// Randomize parameter centers to explore a different region
    ///
    /// Called when stuck in a local optimum. Shifts each parameter's center
    /// to a random position within its bounds.
    fn randomize_centers(&mut self) {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        // Use a seed derived from current iteration count for reproducibility
        let seed = self.config.seed.wrapping_add(self.history.len() as u64);
        let mut rng = StdRng::seed_from_u64(seed);

        for param in self.config.space.params_mut() {
            let (min, max) = (param.bounds.min_value(), param.bounds.max_value());

            let new_center = if param.bounds.is_log_scale() {
                // Log-uniform for log-scale parameters
                let log_min = min.max(1e-10).ln();
                let log_max = max.max(1e-10).ln();
                (log_min + rng.gen::<f32>() * (log_max - log_min)).exp()
            } else {
                // Uniform for linear parameters
                min + rng.gen::<f32>() * (max - min)
            };

            param.set_center(new_center);
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
    ///
    /// If a logger is provided, results are written immediately after each trial.
    fn evaluate_candidates(
        &self,
        dataset: &BinnedDataset,
        candidates: Vec<HashMap<String, f32>>,
        iteration: usize,
        current_trial: &AtomicUsize,
        total_trials: usize,
        logger: Option<&SharedLogger>,
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
                    match self.evaluate_single(EvalInput::Optimistic(dataset), params, iteration) {
                        Ok(result) => {
                            let trial_num = current_trial.fetch_add(1, Ordering::SeqCst) + 1;

                            // Call callback (if set)
                            if let Some(ref cb) = callback {
                                cb(&result, trial_num, total_trials);
                            }

                            // Log immediately (streaming write with flush)
                            log_trial(logger, &result);

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
                match self.evaluate_single(EvalInput::Optimistic(dataset), &params, iteration) {
                    Ok(result) => {
                        let trial_num = current_trial.fetch_add(1, Ordering::SeqCst) + 1;

                        // Call callback
                        if let Some(ref callback) = self.callback {
                            callback(&result, trial_num, total_trials);
                        }

                        // Log immediately (streaming write with flush)
                        log_trial(logger, &result);

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

    /// Evaluate candidates for realistic mode (encoding per split)
    ///
    /// For realistic mode, we cannot parallelize because encoding is stateful.
    fn evaluate_candidates_internal(
        &self,
        candidates: Vec<HashMap<String, f32>>,
        iteration: usize,
        current_trial: &AtomicUsize,
        total_trials: usize,
        use_parallel: bool,
        logger: Option<&SharedLogger>,
    ) -> Result<Vec<TrialResult>> {
        // Get raw data and config for realistic mode (should always be set by tune_dataframe)
        let raw_data = self.raw_data.as_ref().ok_or_else(|| {
            TreeBoostError::Config("raw_data must be set for realistic mode".into())
        })?;
        let realistic_cfg = self.realistic_config.as_ref().ok_or_else(|| {
            TreeBoostError::Config("realistic_config must be set for realistic mode".into())
        })?;

        // Realistic mode cannot be parallelized (encoding is stateful)
        if use_parallel {
            eprintln!("Warning: Parallel mode not supported with realistic tuning, running sequentially");
        }

        // Sequential evaluation
        let mut results = Vec::with_capacity(candidates.len());

        for params in candidates {
            let input = EvalInput::Realistic { raw_data, config: realistic_cfg };
            match self.evaluate_single(input, &params, iteration) {
                Ok(result) => {
                    let trial_num = current_trial.fetch_add(1, Ordering::SeqCst) + 1;

                    // Call callback
                    if let Some(ref callback) = self.callback {
                        callback(&result, trial_num, total_trials);
                    }

                    // Log immediately (streaming write with flush)
                    log_trial(logger, &result);

                    results.push(result);
                }
                Err(e) => {
                    eprintln!("Trial failed: {}", e);
                }
            }
        }

        Ok(results)
    }

    /// Evaluate using holdout with optional k-fold (realistic mode)
    fn evaluate_holdout_realistic_with_folds(
        &self,
        raw_data: &DataFrame,
        realistic_cfg: &RealisticModeConfig,
        params: &HashMap<String, f32>,
        validation_ratio: f32,
        folds: usize,
    ) -> EvalResult {
        if folds == 1 {
            self.evaluate_holdout_realistic(raw_data, realistic_cfg, params, validation_ratio)
        } else {
            self.evaluate_kfold_realistic(raw_data, realistic_cfg, params, folds)
        }
    }

    /// Evaluate using holdout validation with per-split encoding (realistic mode)
    ///
    /// Prevents target leakage by:
    /// 1. Splitting raw data into train/val
    /// 2. Fitting encoder on TRAIN ONLY
    /// 3. Applying encoder to both train and val
    /// 4. Training model on encoded train
    /// 5. Evaluating on encoded val
    fn evaluate_holdout_realistic(
        &self,
        raw_data: &DataFrame,
        realistic_cfg: &RealisticModeConfig,
        params: &HashMap<String, f32>,
        validation_ratio: f32,
    ) -> EvalResult {
        // Split data
        let split = split_holdout(raw_data.height(), validation_ratio, 0.0, self.config.seed);
        let (train_df, val_df) = split_dataframe_by_indices(raw_data, &split.train, &split.validation)?;

        // Encode with per-split pipeline (no target leakage)
        let (train_dataset, val_dataset, val_targets) =
            encode_train_val_split(train_df, val_df, realistic_cfg)?;

        // Train and evaluate using shared helper
        self.train_and_evaluate(&train_dataset, &val_dataset, &val_targets, params)
    }

    /// Evaluate using K-fold cross-validation with per-split encoding (realistic mode)
    fn evaluate_kfold_realistic(
        &self,
        raw_data: &DataFrame,
        realistic_cfg: &RealisticModeConfig,
        params: &HashMap<String, f32>,
        k: usize,
    ) -> EvalResult {
        let kfold = split_kfold(raw_data.height(), k, self.config.seed);
        let mut fold_results = Vec::with_capacity(k);

        for fold_idx in 0..k {
            let (train_idx, val_idx) = kfold.get_fold(fold_idx);

            // Split and encode with per-fold pipeline (no target leakage)
            let (train_df, val_df) = split_dataframe_by_indices(raw_data, &train_idx, &val_idx)?;
            let (train_dataset, val_dataset, val_targets) =
                encode_train_val_split(train_df, val_df, realistic_cfg)?;

            // Train and evaluate using shared helper
            fold_results.push(self.train_and_evaluate(&train_dataset, &val_dataset, &val_targets, params)?);
        }

        Ok(Self::aggregate_fold_results(&fold_results))
    }

    /// Evaluate using conformal with optional k-fold (realistic mode)
    fn evaluate_conformal_realistic_with_folds(
        &self,
        raw_data: &DataFrame,
        realistic_cfg: &RealisticModeConfig,
        params: &HashMap<String, f32>,
        calibration_ratio: f32,
        quantile: f32,
        folds: usize,
    ) -> EvalResult {
        if folds == 1 {
            self.evaluate_conformal_realistic(raw_data, realistic_cfg, params, calibration_ratio, quantile)
        } else {
            // Run conformal on each fold and average
            let kfold = split_kfold(raw_data.height(), folds, self.config.seed);
            let mut fold_results = Vec::with_capacity(folds);

            for fold_idx in 0..folds {
                let (train_idx, val_idx) = kfold.get_fold(fold_idx);

                // Split and encode with per-fold pipeline
                let (train_df, val_df) = split_dataframe_by_indices(raw_data, &train_idx, &val_idx)?;
                let (train_dataset, cal_dataset, cal_targets) =
                    encode_train_val_split(train_df, val_df, realistic_cfg)?;

                // Train and evaluate using conformal helper
                let result = self.train_and_evaluate_conformal(
                    &train_dataset, &cal_dataset, &cal_targets, params, quantile,
                )?;
                fold_results.push(result);
            }

            Ok(Self::aggregate_fold_results(&fold_results))
        }
    }

    /// Evaluate using conformal prediction with per-split encoding (realistic mode)
    ///
    /// Uses the conformal quantile `q` as the optimization metric.
    /// Lower `q` = tighter intervals = more confident model.
    fn evaluate_conformal_realistic(
        &self,
        raw_data: &DataFrame,
        realistic_cfg: &RealisticModeConfig,
        params: &HashMap<String, f32>,
        calibration_ratio: f32,
        quantile: f32,
    ) -> EvalResult {
        // Split data
        let split = split_holdout(raw_data.height(), calibration_ratio, 0.0, self.config.seed);
        let (train_df, cal_df) = split_dataframe_by_indices(raw_data, &split.train, &split.validation)?;

        // Encode with per-split pipeline (no target leakage)
        let (train_dataset, cal_dataset, cal_targets) =
            encode_train_val_split(train_df, cal_df, realistic_cfg)?;

        // Train and evaluate using conformal helper
        self.train_and_evaluate_conformal(&train_dataset, &cal_dataset, &cal_targets, params, quantile)
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

        // Apply early stopping for inner loop (individual model training)
        // Note: Conformal strategy doesn't use early stopping or validation_ratio
        // It uses calibration_ratio instead (set in evaluate_conformal)
        if self.config.early_stopping_rounds > 0 {
            config.early_stopping_rounds = self.config.early_stopping_rounds;
            config.validation_ratio = self.config.validation_ratio;
        } else {
            // No early stopping - use validation from eval strategy for metrics only
            config.validation_ratio = match self.config.eval_strategy {
                EvalStrategy::Holdout { validation_ratio, .. } => validation_ratio,
                EvalStrategy::Conformal { .. } => 0.0, // Conformal uses calibration_ratio instead
            };
        }

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
            f1_score: None,
            roc_auc: None,
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
            f1_score: None,
            roc_auc: None,
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
            f1_score: None,
            roc_auc: None,
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
            f1_score: None,
            roc_auc: None,
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

    // ==========================================================================
    // Realistic Mode Tests
    // ==========================================================================

    #[test]
    fn test_tuning_mode_variants() {
        // Test that TuningMode enum works correctly
        let optimistic = TuningMode::Optimistic;
        let realistic = TuningMode::Realistic;

        // Verify they are distinct variants
        assert!(matches!(optimistic, TuningMode::Optimistic));
        assert!(matches!(realistic, TuningMode::Realistic));
    }
}