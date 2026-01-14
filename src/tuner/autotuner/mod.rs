//! AutoTuner for hyperparameter optimization
//!
//! Provides the main `AutoTuner` struct and implementation for
//! iterative grid search with auto-zoom.
//!
//! The tuner is generic over `TunableModel`, allowing it to work with
//! different model types (GBDTModel, UniversalModel, etc.) without
//! code duplication.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use polars::prelude::*;

use crate::dataset::{BinnedDataset};
use crate::{Result, TreeBoostError};

use super::traits::{ParamMapExt, TunableModel};

use super::config::{
    EvalStrategy, TunerConfig, TuningMode, OptimizationMetric,
};
use super::history::{SearchHistory};
use super::logger::{
    finalize_logging, init_logger, save_model_formats, start_iteration_logging,
};
use super::realistic::RealisticModeConfig;

// Submodules
mod types;
mod grid;
mod metrics;
mod eval_optimistic;
mod eval_realistic;
mod execution;
mod builder;

// Internal use of types
use types::{SendPtr, MAX_ZONE_SWITCH_FAILS};

// =============================================================================
// AutoTuner Main Struct
// =============================================================================

/// AutoTuner for hyperparameter optimization
///
/// Uses an Iterative Grid Search (Auto-Zoom) approach to find optimal
/// hyperparameters. Generic over `TunableModel` to support different
/// model types (GBDTModel, UniversalModel, etc.).
///
/// # Type Parameters
///
/// * `M` - The model type to tune, must implement `TunableModel`
///
/// # Example
///
/// ```ignore
/// use treeboost::tuner::AutoTuner;
/// use treeboost::GBDTConfig;
///
/// // Tune GBDTModel (turbofish syntax)
/// let tuner = AutoTuner::<GBDTModel>::new(GBDTConfig::default());
/// let (best_config, history) = tuner.tune(&dataset)?;
/// ```
pub struct AutoTuner<M: TunableModel> {
    /// Tuner configuration
    config: TunerConfig,
    /// Base model configuration (non-tuned parameters)
    base_config: M::Config,
    /// Search history
    history: SearchHistory,
    /// Progress callback
    callback: Option<super::history::ProgressCallback>,
    /// Next trial ID (atomic for parallel evaluation)
    next_trial_id: AtomicUsize,

    // Realistic mode support (encoding per split)
    /// Raw data for realistic mode (stored as Arc for sharing)
    raw_data: Option<std::sync::Arc<DataFrame>>,
    /// Realistic mode encoding configuration
    realistic_config: Option<RealisticModeConfig>,

    // Custom validation support (for time-series/grouped data)
    /// Custom train/validation datasets (when provided via tune_with_validation)
    /// Stored as raw pointers since they're only valid during the tune call
    /// Wrapped in SendPtr for thread safety (pointers are read-only and valid during tune)
    custom_validation: Option<(SendPtr<BinnedDataset>, SendPtr<BinnedDataset>)>,

    /// Phantom data for generic type
    _phantom: PhantomData<M>,
}

// =============================================================================
// Public API
// =============================================================================

impl<M: TunableModel> AutoTuner<M> {
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
    /// * Best model config found and the complete search history
    pub fn tune(&mut self, dataset: &BinnedDataset) -> Result<(M::Config, SearchHistory)> {
        // Store dataset reference for evaluation (wrapped in a temporary storage)
        // We use a simple approach: store dataset pointer and retrieve in evaluate functions
        self.run_tune_with_dataset(dataset)
    }

    /// Tune hyperparameters with custom validation dataset (for time-series/grouped data)
    ///
    /// **When to use this method:**
    ///
    /// Use `tune_with_validation()` when random validation splits would cause **data leakage**.
    /// This is critical for:
    ///
    /// - **Time-series data**: Train on past dates, validate on future dates
    /// - **Panel data**: Train on some groups, validate on held-out groups
    /// - **Hierarchical data**: Prevent leakage across hierarchical levels
    /// - **Cross-validation with custom folds**: Control exact train/val composition
    ///
    /// For standard cross-sectional (i.i.d.) data where rows are independent,
    /// use [`AutoTuner::tune()`] instead, which performs random splits internally.
    ///
    /// **Key difference from `tune()`:**
    ///
    /// - [`AutoTuner::tune()`]: Accepts single dataset, performs random internal splits
    /// - `tune_with_validation()`: Accepts pre-split train/val datasets, no further splitting
    ///
    /// # Arguments
    ///
    /// * `train_dataset` - Training data (will NOT be split further)
    /// * `val_dataset` - Validation data (for hyperparameter evaluation)
    ///
    /// # Returns
    ///
    /// * Best model config found and the complete search history
    ///
    /// # Example 1: Time-Series (Date-Based Split)
    ///
    /// ```ignore
    /// use treeboost::{AutoTuner, TunerConfig, ParameterSpace, SpacePreset};
    /// use treeboost::dataset::{DatasetLoader, BinnedDataset};
    /// use polars::prelude::*;
    ///
    /// // Load time-series data
    /// let df = CsvReadOptions::default()
    ///     .try_into_reader_with_file_path(Some("stock_prices.csv".into()))?
    ///     .finish()?;
    ///
    /// // CRITICAL: Split by date BEFORE binning (prevents target leakage)
    /// let train_df = df.filter(col("date").lt(lit("2024-01-01")))?;
    /// let val_df = df.filter(col("date").gte(lit("2024-01-01")))?;
    ///
    /// // Encode train and validation datasets separately
    /// let loader = DatasetLoader::new(255);
    /// let train_dataset = loader.load_dataframe(train_df, "return", None)?;
    /// let val_dataset = loader.load_dataframe(val_df, "return", None)?;
    ///
    /// // Configure tuner
    /// let base_config = GBDTConfig::default();
    /// let tuner_config = TunerConfig::new()
    ///     .with_eval_strategy(EvalStrategy::holdout(0.0));  // No further splits!
    ///
    /// let mut tuner = AutoTuner::new(base_config)
    ///     .with_config(tuner_config)
    ///     .with_space(ParameterSpace::from_preset(SpacePreset::Moderate));
    ///
    /// // Tune using pre-split data (no random splits)
    /// let (best_config, history) = tuner.tune_with_validation(&train_dataset, &val_dataset)?;
    ///
    /// // Train final model on full training set with best config
    /// let final_model = GBDTModel::train(&train_dataset, &best_config)?;
    /// ```
    ///
    /// # Example 2: Panel Data (Group-Based Split)
    ///
    /// ```ignore
    /// use treeboost::{AutoTuner, GBDTConfig};
    /// use polars::prelude::*;
    ///
    /// // Panel data: stocks × dates
    /// let df = CsvReadOptions::default()
    ///     .try_into_reader_with_file_path(Some("panel_data.csv".into()))?
    ///     .finish()?;
    ///
    /// // Hold out specific stocks for validation (no group leakage)
    /// let train_stocks = vec!["AAPL", "MSFT", "GOOGL"];
    /// let val_stocks = vec!["TSLA", "AMZN"];
    ///
    /// let train_df = df.filter(col("stock_id").is_in(lit(train_stocks)))?;
    /// let val_df = df.filter(col("stock_id").is_in(lit(val_stocks)))?;
    ///
    /// // Encode separately
    /// let loader = DatasetLoader::new(255);
    /// let train_dataset = loader.load_dataframe(train_df, "return", None)?;
    /// let val_dataset = loader.load_dataframe(val_df, "return", None)?;
    ///
    /// // Tune with group-aware validation
    /// let mut tuner = AutoTuner::new(GBDTConfig::default());
    /// let (best_config, history) = tuner.tune_with_validation(&train_dataset, &val_dataset)?;
    /// ```
    ///
    /// # Integration with AutoBuilder
    ///
    /// For high-level AutoML workflows, use [`crate::model::AutoBuilder::with_presplit_validation`]
    /// which handles preprocessing and feature engineering automatically:
    ///
    /// ```ignore
    /// use treeboost::model::AutoBuilder;
    ///
    /// // Split raw DataFrame by date
    /// let train_df = df.filter(col("date").lt(lit("2024-01-01")))?;
    /// let val_df = df.filter(col("date").gte(lit("2024-01-01")))?;
    ///
    /// // AutoBuilder handles preprocessing + tuning with pre-split validation
    /// let model = AutoBuilder::new()
    ///     .with_presplit_validation(val_df)
    ///     .fit(&train_df, "target")?;
    /// ```
    ///
    /// # See Also
    ///
    /// * [`AutoTuner::tune`] - For cross-sectional (i.i.d.) data with random splits
    /// * [`crate::model::AutoBuilder::with_presplit_validation`] - High-level API for AutoML workflows
    /// * [`crate::model::AutoBuilder::with_random_validation_split`] - For cross-sectional data
    pub fn tune_with_validation(
        &mut self,
        train_dataset: &BinnedDataset,
        val_dataset: &BinnedDataset,
    ) -> Result<(M::Config, SearchHistory)> {
        // Store raw pointers for custom validation (valid only during this call)
        self.custom_validation = Some((
            SendPtr::new(train_dataset as *const _),
            SendPtr::new(val_dataset as *const _),
        ));

        // Use catch_unwind for panic safety - ensures cleanup even if evaluation panics
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.run_tune_with_dataset(train_dataset)
        }));

        // Always clean up pointers
        self.custom_validation = None;

        // Propagate result or panic
        match result {
            Ok(r) => r,
            Err(panic_payload) => std::panic::resume_unwind(panic_payload),
        }
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
    ) -> Result<(M::Config, SearchHistory)> {
        // Store raw data and config for use in evaluation
        self.raw_data = Some(std::sync::Arc::new(df));
        self.realistic_config = Some(realistic_config);

        // Force realistic mode
        self.config.tuning_mode = TuningMode::Realistic;

        // Run the tuning loop (same as tune(), but evaluate methods will use realistic encoding)
        self.tune_internal()
    }

    // Getters for internal use by submodules (pub(super))
    pub(super) fn base_config(&self) -> &M::Config {
        &self.base_config
    }

    pub(super) fn build_config(&self, params: &HashMap<String, f32>) -> M::Config {
        let mut config = self.base_config.clone();

        // Convert f32 params to ParamValue with proper categorical handling
        let param_values = params.to_param_values_with_space(&self.config.space);
        M::apply_params(&mut config, &param_values);

        // Apply tuner-specific settings
        M::set_num_rounds(&mut config, self.config.num_rounds);

        // Apply early stopping for inner loop (individual model training)
        // Note: Conformal strategy doesn't use early stopping or validation_ratio
        // It uses calibration_ratio instead (set in evaluate_conformal)
        if self.config.early_stopping_rounds > 0 {
            M::configure_validation(
                &mut config,
                self.config.validation_ratio,
                self.config.early_stopping_rounds,
            );
        } else {
            // No early stopping - use validation from eval strategy for metrics only
            let validation_ratio = match self.config.eval_strategy {
                EvalStrategy::Holdout {
                    validation_ratio, ..
                } => validation_ratio,
                EvalStrategy::Conformal { .. } => 0.0, // Conformal uses calibration_ratio instead
            };
            M::configure_validation(&mut config, validation_ratio, 0);
        }

        config
    }

    pub(super) fn task_type(&self) -> super::config::TaskType {
        self.config.task_type
    }

    pub(super) fn optimization_metric(&self) -> &OptimizationMetric {
        &self.config.optimization_metric
    }

    pub(super) fn seed(&self) -> u64 {
        self.config.seed
    }

    pub(super) fn eval_strategy(&self) -> &EvalStrategy {
        &self.config.eval_strategy
    }

    pub(super) fn early_stopping_rounds(&self) -> usize {
        self.config.early_stopping_rounds
    }

    pub(super) fn parallel_trials(&self) -> bool {
        self.config.parallel_trials
    }

    pub(super) fn n_parallel(&self) -> usize {
        self.config.n_parallel
    }

    pub(super) fn custom_validation(&self) -> Option<&(SendPtr<BinnedDataset>, SendPtr<BinnedDataset>)> {
        self.custom_validation.as_ref()
    }

    pub(super) fn callback(&self) -> &Option<super::history::ProgressCallback> {
        &self.callback
    }

    pub(super) fn next_trial_id_fetch_add(&self, delta: usize) -> usize {
        self.next_trial_id.fetch_add(delta, Ordering::SeqCst)
    }

    pub(super) fn space_params_mut(&mut self) -> &mut [super::config::ParamDef] {
        self.config.space.params_mut()
    }

    pub(super) fn history_len(&self) -> usize {
        self.history.len()
    }

    pub(super) fn raw_data(&self) -> Option<&std::sync::Arc<DataFrame>> {
        self.raw_data.as_ref()
    }

    pub(super) fn realistic_config(&self) -> Option<&RealisticModeConfig> {
        self.realistic_config.as_ref()
    }
}

// =============================================================================
// Internal Implementation
// =============================================================================

/// Format trial metrics for display based on eval strategy and task type.
///
/// This helper function centralizes metric formatting logic to avoid duplication
/// when displaying "New best!" trial results during tuning.
///
/// # Arguments
/// * `result` - The trial result containing metrics
/// * `eval_strategy` - Evaluation strategy (determines conformal formatting)
/// * `task_type` - Task type (determines metric type)
///
/// # Returns
/// A formatted string displaying the most relevant metrics for this trial.
fn format_trial_metrics(
    result: &crate::tuner::trial::TrialResult,
    eval_strategy: &EvalStrategy,
    task_type: &crate::tuner::config::TaskType,
) -> String {
    let is_conformal = matches!(eval_strategy, EvalStrategy::Conformal { .. });

    if is_conformal {
        // Conformal: val_metric is interval width (quantile q)
        format!("q={:.5} (interval width)", result.val_loss)
    } else if task_type.is_regression() {
        // Regression: show MSE and RMSE
        format!(
            "MSE={:.5} RMSE={:.4}",
            result.val_loss,
            result.val_loss.sqrt()
        )
    } else {
        // Classification: show LogLoss, AUC, F1
        let auc_str = result
            .roc_auc
            .map(|auc| format!(" AUC={:.4}", auc))
            .unwrap_or_default();
        let f1_str = result
            .f1_score
            .map(|f1| format!(" F1={:.2}%", f1 * 100.0))
            .unwrap_or_default();
        format!("LogLoss={:.5}{}{}", result.val_loss, auc_str, f1_str)
    }
}

impl<M: TunableModel> AutoTuner<M> {
    /// Internal tune method that works with BinnedDataset
    fn run_tune_with_dataset(
        &mut self,
        dataset: &BinnedDataset,
    ) -> Result<(M::Config, SearchHistory)> {
        // Validate configuration
        self.config
            .validate()
            .map_err(|e| TreeBoostError::Config(format!("Invalid tuner configuration: {}", e)))?;

        let total_trials = self.config.estimated_trials();
        // For GPU backends, run sequentially to avoid CUDA context contention
        // GPU trials are fast (~1-2s each), so sequential is fine
        let use_parallel = self.config.parallel_trials && !execution::is_gpu_backend::<M>(&self.base_config);

        if self.config.verbose {
            println!("Starting AutoTuner...");
            println!("  Iterations: {}", self.config.n_iterations);
            println!("  Parameters: {}", self.config.space.len());
            println!("  Estimated trials: {}", total_trials);
            println!("  Grid strategy: {:?}", self.config.grid_strategy);
            println!("  Eval strategy: {:?}", self.config.eval_strategy);
            println!("  Tuning mode: {:?}", self.config.tuning_mode);
            let parallel_reason = if execution::is_gpu_backend::<M>(&self.base_config) {
                "disabled (GPU: sequential trials avoid context contention)"
            } else if use_parallel {
                "enabled (CPU: parallel trials for faster tuning)"
            } else {
                "disabled by config"
            };
            println!("  Parallel: {}", parallel_reason);
        }

        let current_trial = AtomicUsize::new(0);
        let start_time = Instant::now();

        // Initialize trial logger if output_dir is configured
        let logger = init_logger(
            &self.config.output_dir,
            self.config
                .space
                .param_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
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
            let candidates = grid::generate_grid(&self.config.space, &self.config.grid_strategy, self.config.seed, spread);

            if self.config.verbose {
                println!("  Testing {} candidates...", candidates.len());
            }

            // Evaluate all candidates (parallel or sequential based on backend)
            // Results are logged immediately inside evaluate_candidates via the shared logger
            let results = execution::evaluate_candidates(
                self,
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
                        // Show learning_rate from params if tuned, otherwise from base_config
                        let lr = result
                            .params
                            .get("learning_rate")
                            .copied()
                            .unwrap_or(M::get_learning_rate(&self.base_config));
                        let metric_str = format_trial_metrics(&result, &self.config.eval_strategy, &self.config.task_type);
                        println!(
                            "  -> New best! {} (depth={}, lr={:.4}, trees={})",
                            metric_str,
                            result.params.get("max_depth").unwrap_or(&0.0),
                            lr,
                            result.num_trees,
                        );
                    }
                }

                self.history.add(result);
            }

            // Check if we found improvement using the configured optimization metric
            let improved = if let Some(best_after) = self.history.best() {
                // Find best trial from previous iterations
                let best_before_trial = self
                    .history
                    .trials()
                    .iter()
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
                        println!(
                            "  {} consecutive zone switches failed, stopping search.",
                            zone_switch_fails
                        );
                    }
                    break;
                }

                if self.config.verbose {
                    println!(
                        "  No improvement found, switching zone ({}/{} fails)...",
                        zone_switch_fails, MAX_ZONE_SWITCH_FAILS
                    );
                }
                // Reset zoom level to explore wider
                zoom_level = 0;

                // Randomize centers to explore different region
                execution::randomize_centers(self);
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
        let best = self
            .history
            .best()
            .ok_or_else(|| TreeBoostError::Training("No successful trials".into()))?;

        if self.config.verbose {
            println!("\n=== Tuning Complete ===");
            println!("  Total trials: {}", self.history.len());
            // Show metrics based on eval strategy and task type
            let is_conformal = matches!(self.config.eval_strategy, EvalStrategy::Conformal { .. });
            if is_conformal {
                println!("  Best interval width (q): {:.6}", best.val_loss);
            } else if self.config.task_type.is_regression() {
                println!(
                    "  Best MSE: {:.6} (RMSE: {:.4})",
                    best.val_loss,
                    best.val_loss.sqrt()
                );
            } else {
                println!("  Best LogLoss: {:.6}", best.val_loss);
                if let Some(auc) = best.roc_auc {
                    println!("  ROC-AUC: {:.4}", auc);
                }
                if let Some(f1) = best.f1_score {
                    println!("  F1 score: {:.2}%", f1 * 100.0);
                }
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
                let best_config = self.build_config(&best.params);
                let final_model = M::train(dataset, &best_config)?;

                if self.config.verbose {
                    println!("  Model trained ({} trees)", final_model.num_trees());
                }

                // Save model in requested formats
                save_model_formats(&logger, &final_model, &self.config.save_model_formats)?;

                if self.config.verbose {
                    println!(
                        "  Model saved in {} format(s)",
                        self.config.save_model_formats.len()
                    );
                }
            }

            if self.config.verbose {
                println!("  Results saved to: {}", run_dir.display());
            }
        }

        let best_config = self.build_config(&best.params);
        Ok((best_config, self.history.clone()))
    }

    /// Internal tuning loop (shared by tune and tune_dataframe)
    fn tune_internal(&mut self) -> Result<(M::Config, SearchHistory)> {
        // Validate configuration
        self.config
            .validate()
            .map_err(|e| TreeBoostError::Config(format!("Invalid tuner configuration: {}", e)))?;

        let total_trials = self.config.estimated_trials();
        // For GPU backends, run sequentially to avoid CUDA context contention
        // GPU trials are fast (~1-2s each), so sequential is fine
        let use_parallel = self.config.parallel_trials && !execution::is_gpu_backend::<M>(&self.base_config);

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
            let parallel_reason = if execution::is_gpu_backend::<M>(&self.base_config) {
                "disabled (GPU: sequential trials avoid context contention)"
            } else if use_parallel {
                "enabled (CPU: parallel trials for faster tuning)"
            } else {
                "disabled by config"
            };
            println!("  Parallel: {}", parallel_reason);
        }

        let current_trial = AtomicUsize::new(0);
        let start_time = Instant::now();

        // Initialize trial logger if output_dir is configured
        let logger = init_logger(
            &self.config.output_dir,
            self.config
                .space
                .param_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
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
            let candidates = grid::generate_grid(&self.config.space, &self.config.grid_strategy, self.config.seed, spread);

            if self.config.verbose {
                println!("  Testing {} candidates...", candidates.len());
            }

            // Evaluate all candidates (parallel or sequential based on backend)
            // Results are logged immediately inside evaluate_candidates_internal via the shared logger
            let results = if let (Some(raw_data), Some(realistic_cfg)) = (self.raw_data.as_ref(), self.realistic_config.as_ref()) {
                execution::evaluate_candidates_internal(
                    self,
                    raw_data,
                    realistic_cfg,
                    candidates,
                    iteration,
                    &current_trial,
                    total_trials,
                    use_parallel,
                    logger.as_ref(),
                )?
            } else {
                return Err(TreeBoostError::Config(
                    "raw_data and realistic_config must be set for realistic mode".into(),
                ));
            };

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
                        // Show learning_rate from params if tuned, otherwise from base_config
                        let lr = result
                            .params
                            .get("learning_rate")
                            .copied()
                            .unwrap_or(M::get_learning_rate(&self.base_config));
                        let metric_str = format_trial_metrics(&result, &self.config.eval_strategy, &self.config.task_type);
                        println!(
                            "  -> New best! {} (depth={}, lr={:.4}, trees={})",
                            metric_str,
                            result.params.get("max_depth").unwrap_or(&0.0),
                            lr,
                            result.num_trees,
                        );
                    }
                }

                self.history.add(result);
            }

            // Check if we found improvement using the configured optimization metric
            let improved = if let Some(best_after) = self.history.best() {
                // Find best trial from previous iterations
                let best_before_trial = self
                    .history
                    .trials()
                    .iter()
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
                        println!(
                            "  {} consecutive zone switches failed, stopping search.",
                            zone_switch_fails
                        );
                    }
                    break;
                }

                if self.config.verbose {
                    println!(
                        "  No improvement found, switching zone ({}/{} fails)...",
                        zone_switch_fails, MAX_ZONE_SWITCH_FAILS
                    );
                }
                // Reset zoom level to explore wider
                zoom_level = 0;

                // Randomize centers to explore different region
                execution::randomize_centers(self);
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
        let best = self
            .history
            .best()
            .ok_or_else(|| TreeBoostError::Training("No successful trials".into()))?;

        if self.config.verbose {
            println!("\n=== Tuning Complete ===");
            println!("  Total trials: {}", self.history.len());
            // Show metrics based on eval strategy and task type
            let is_conformal = matches!(self.config.eval_strategy, EvalStrategy::Conformal { .. });
            if is_conformal {
                println!("  Best interval width (q): {:.6}", best.val_loss);
            } else if self.config.task_type.is_regression() {
                println!(
                    "  Best MSE: {:.6} (RMSE: {:.4})",
                    best.val_loss,
                    best.val_loss.sqrt()
                );
            } else {
                println!("  Best LogLoss: {:.6}", best.val_loss);
                if let Some(auc) = best.roc_auc {
                    println!("  ROC-AUC: {:.4}", auc);
                }
                if let Some(f1) = best.f1_score {
                    println!("  F1 score: {:.2}%", f1 * 100.0);
                }
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
                        let full_dataset = super::realistic::encode_full_dataset(full_df, realistic_cfg)?;

                        // Build best config and train
                        let best_config = self.build_config(&best.params);
                        let final_model = M::train(&full_dataset, &best_config)?;

                        if self.config.verbose {
                            println!("  Model trained ({} trees)", final_model.num_trees());
                        }

                        // Save model in requested formats
                        save_model_formats(&logger, &final_model, &self.config.save_model_formats)?;

                        if self.config.verbose {
                            println!(
                                "  Model saved in {} format(s)",
                                self.config.save_model_formats.len()
                            );
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

    // =============================================================================
    // Test helper methods (not part of public API)
    // =============================================================================

    #[cfg(test)]
    pub(crate) fn generate_param_values(&self, param: &crate::tuner::config::ParamDef, spread: f32, points: usize) -> Vec<f32> {
        grid::generate_param_values(param, spread, points)
    }

    #[cfg(test)]
    pub(crate) fn generate_cartesian_grid(&self, spread: f32, points_per_dim: usize) -> Vec<HashMap<String, f32>> {
        grid::generate_cartesian_grid(&self.config.space, spread, points_per_dim)
    }

    #[cfg(test)]
    pub(crate) fn is_gpu_backend(&self) -> bool {
        execution::is_gpu_backend::<M>(&self.base_config)
    }

    #[cfg(test)]
    pub(crate) fn generate_lhs_grid(&self, spread: f32, n_samples: usize) -> Vec<HashMap<String, f32>> {
        grid::generate_lhs_grid(&self.config.space, spread, n_samples, self.config.seed)
    }

    #[cfg(test)]
    pub(crate) fn generate_random_grid(&self, spread: f32, n_samples: usize) -> Vec<HashMap<String, f32>> {
        grid::generate_random_grid(&self.config.space, spread, n_samples, self.config.seed)
    }
}

#[cfg(test)]
mod tests;
