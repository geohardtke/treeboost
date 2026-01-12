//! AutoBuilder: High-level AutoML interface for TreeBoost
//!
//! Provides a simplified, opinionated API that automatically:
//! 1. Profiles your data (identifies column types, drops useless columns)
//! 2. Selects preprocessing (scalers, encoders, imputers)
//! 3. Engineers features (polynomials, interactions)
//! 4. Tunes hyperparameters (LTT dual-phase, tree params)
//! 5. Trains the optimal model
//!
//! # Design Philosophy
//!
//! AutoBuilder follows the principle of **analyze first, train once**. Unlike
//! AutoML systems that try every combination, TreeBoost analyzes dataset
//! characteristics and makes informed decisions.
//!
//! # Architecture
//!
//! This module is split into three files for maintainability:
//! - `config.rs` - Configuration types (AutoConfig, BuildResult, etc.)
//! - `tuning.rs` - Hyperparameter tuning methods
//! - `builder.rs` - Core AutoBuilder and orchestration logic (this file)
//!
//! # Example
//!
//! ```ignore
//! use treeboost::model::{AutoBuilder, AutoConfig};
//! use polars::prelude::*;
//!
//! // Load your data
//! let df = CsvReader::from_path("data.csv")?.finish()?;
//!
//! // Build with defaults (analyze and train)
//! let model = AutoBuilder::new()
//!     .fit(&df, "target_column")?;
//!
//! // Or customize the process
//! let model = AutoBuilder::new()
//!     .with_tuning(TuningLevel::Thorough)
//!     .with_random_validation_split(0.2)
//!     .fit(&df, "target_column")?;
//!
//! // Predict on new data
//! let predictions = model.predict(&test_df)?;
//! ```

use crate::analysis::{Confidence, DataFrameProfile, DatasetAnalysis, PanelDataInfo};
use crate::dataset::feature_extractor::LinearFeatureConfig;
use crate::dataset::{BinnedDataset, DataPipeline};
use crate::defaults::auto as auto_defaults;
use crate::features::{FeaturePlan, SmartFeatureEngine};
use crate::model::config::{AutoConfig, BuildPhaseTimes, BuildResult, TuningLevel};
use crate::model::progress::{ProgressCallback, ProgressUpdate, TrainingPhase};
use crate::model::universal::ModeSelection;
use crate::model::{BoostingMode, UniversalConfig, UniversalModel};
use crate::preprocessing::{ModelType, PreprocessingPlan, SmartPreprocessor};
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Import tuning functions
use super::tuning;

/// AutoBuilder: High-level AutoML interface
pub struct AutoBuilder {
    config: AutoConfig,
    /// Optional custom validation DataFrame (for time-series or grouped data)
    validation_df: Option<DataFrame>,
}

impl AutoBuilder {
    /// Create a new AutoBuilder with default configuration
    pub fn new() -> Self {
        Self {
            config: AutoConfig::default(),
            validation_df: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AutoConfig) -> Self {
        Self {
            config,
            validation_df: None,
        }
    }

    /// Set tuning level
    pub fn with_tuning(mut self, level: TuningLevel) -> Self {
        self.config.tuning_level = level;
        self
    }

    /// Set validation split ratio for **random** train/validation split.
    ///
    /// **WARNING**: Only use this for cross-sectional (i.i.d.) data where rows are independent.
    /// For time-series or panel data, use `Self::with_presplit_validation` instead to avoid data leakage.
    ///
    /// The library will randomly split your data into:
    /// - Training set: (1 - ratio) * num_rows
    /// - Validation set: ratio * num_rows
    ///
    /// # Arguments
    ///
    /// * `ratio` - Fraction of data to use for validation (typically 0.2 for 80/20 split)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Cross-sectional data (rows are independent)
    /// let model = AutoBuilder::new()
    ///     .with_random_validation_split(0.2)  // Random 80/20 split
    ///     .fit(&df, "target")?;
    /// ```
    ///
    /// # See Also
    ///
    /// * `Self::with_presplit_validation` - For time-series or panel data
    pub fn with_random_validation_split(mut self, ratio: f32) -> Self {
        self.config.val_ratio = ratio;
        self
    }

    /// Set feature engineering mode
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::{AutoBuilder, FeatureEngineeringMode};
    /// use treeboost::features::SmartFeatureConfig;
    ///
    /// // Disable feature engineering
    /// let builder = AutoBuilder::new()
    ///     .with_feature_engineering(FeatureEngineeringMode::None);
    ///
    /// // Use aggressive features
    /// let builder = AutoBuilder::new()
    ///     .with_feature_engineering(FeatureEngineeringMode::Aggressive);
    ///
    /// // Custom configuration
    /// let builder = AutoBuilder::new()
    ///     .with_feature_engineering(FeatureEngineeringMode::Custom(
    ///         SmartFeatureConfig::default()
    ///             .with_enable_polynomial(false)
    ///             .with_top_n_interactions(10)
    ///     ));
    /// ```
    pub fn with_feature_engineering(
        mut self,
        mode: crate::model::config::FeatureEngineeringMode,
    ) -> Self {
        self.config.feature_engineering = mode;
        self
    }

    /// Set preprocessing mode
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::{AutoBuilder, PreprocessingMode};
    ///
    /// // No preprocessing
    /// let builder = AutoBuilder::new()
    ///     .with_preprocessing(PreprocessingMode::None);
    ///
    /// // Strict preprocessing
    /// let builder = AutoBuilder::new()
    ///     .with_preprocessing(PreprocessingMode::Strict);
    /// ```
    pub fn with_preprocessing(mut self, mode: crate::model::config::PreprocessingMode) -> Self {
        self.config.preprocessing = mode;
        self
    }

    /// Use pre-split validation data for time-series, panel, or grouped data.
    ///
    /// **Use this method when** random splits would cause data leakage:
    /// - **Time-series data**: Validate on future dates (date-based split)
    /// - **Panel data**: Validate on held-out groups (no group leakage)
    /// - **Cross-validation**: Custom fold splits
    /// - **Any non-i.i.d. data**: Where rows have dependencies
    ///
    /// When you provide pre-split validation data, the library will:
    /// 1. Disable internal random splitting (sets `val_ratio = 0.0` automatically)
    /// 2. Use your training DataFrame for training
    /// 3. Use your validation DataFrame for hyperparameter tuning and early stopping
    ///
    /// # Arguments
    ///
    /// * `validation_df` - Pre-split validation DataFrame with same schema as training data
    ///
    /// # Example: Time-Series (Date-Based Split)
    ///
    /// ```ignore
    /// // Split by date for time-series forecasting
    /// let train_df = df.filter(col("date").lt(lit("2024-01-01")))?;
    /// let val_df = df.filter(col("date").gte(lit("2024-01-01")))?;
    ///
    /// let model = AutoBuilder::new()
    ///     .with_presplit_validation(val_df)  // Correct: no leakage!
    ///     .fit(&train_df, "target")?;
    /// ```
    ///
    /// # Example: Panel Data (Group-Based Split)
    ///
    /// ```ignore
    /// // Hold out specific stocks for validation
    /// let train_df = df.filter(col("stock_id").is_in(train_stocks))?;
    /// let val_df = df.filter(col("stock_id").is_in(val_stocks))?;
    ///
    /// let model = AutoBuilder::new()
    ///     .with_presplit_validation(val_df)
    ///     .fit(&train_df, "target")?;
    /// ```
    ///
    /// # See Also
    ///
    /// * `Self::with_random_validation_split` - For cross-sectional (i.i.d.) data
    pub fn with_presplit_validation(mut self, validation_df: DataFrame) -> Self {
        // Disable internal validation split when custom data is provided
        self.config.val_ratio = 0.0;
        self.validation_df = Some(validation_df);
        self
    }

    /// Force a specific mode
    pub fn with_mode(mut self, mode: BoostingMode) -> Self {
        self.config.mode_selection = ModeSelection::Fixed(mode);
        self
    }

    /// Set ensemble mode (PureTree only)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use treeboost::{AutoBuilder, EnsembleMode, AutoEnsembleMethod, AutoEnsembleConfig};
    ///
    /// // No ensemble
    /// let builder = AutoBuilder::new()
    ///     .with_ensemble(EnsembleMode::Disabled);
    ///
    /// // Default ensemble
    /// let builder = AutoBuilder::new()
    ///     .with_ensemble(EnsembleMode::Default);
    ///
    /// // With specific method
    /// let builder = AutoBuilder::new()
    ///     .with_ensemble(EnsembleMode::WithMethod(AutoEnsembleMethod::SimpleAverage));
    ///
    /// // Custom configuration
    /// let builder = AutoBuilder::new()
    ///     .with_ensemble(EnsembleMode::Custom(AutoEnsembleConfig::new()));
    /// ```
    pub fn with_ensemble(mut self, mode: crate::model::config::EnsembleMode) -> Self {
        self.config.ensemble = mode;
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set time budget for training
    ///
    /// AutoBuilder will adapt its behavior to fit within this time limit:
    /// - Skips feature engineering if time is tight (< 20s remaining)
    /// - Reduces tuning intensity if time is limited (< 30s → Quick tuning)
    /// - Skips tuning entirely if very low on time (< 10s)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::time::Duration;
    /// let builder = AutoBuilder::new()
    ///     .with_time_budget(Duration::from_secs(60));  // 1 minute max
    /// ```
    pub fn with_time_budget(mut self, budget: Duration) -> Self {
        self.config.time_budget = Some(budget);
        self
    }

    /// Set progress callback for tracking training phases
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::model::ConsoleProgress;
    /// use std::sync::Arc;
    ///
    /// let builder = AutoBuilder::new()
    ///     .with_progress_callback(Arc::new(ConsoleProgress::detailed()));
    /// ```
    pub fn with_progress_callback(mut self, callback: Arc<dyn ProgressCallback>) -> Self {
        self.config.progress_callback = callback;
        self
    }

    /// Set custom feature extractor configuration for LinearThenTree mode
    ///
    /// This controls which features are extracted for the linear component.
    /// By default, all features are included (no exclusions).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::dataset::feature_extractor::LinearFeatureConfig;
    ///
    /// let feature_config = LinearFeatureConfig::default()
    ///     .with_exclude_categorical(true)  // Exclude categorical features from linear model
    ///     .with_exclude_id(true);          // Exclude ID columns
    ///
    /// let builder = AutoBuilder::new()
    ///     .with_linear_feature_config(feature_config);
    /// ```
    pub fn with_linear_feature_config(mut self, config: LinearFeatureConfig) -> Self {
        self.config.linear_feature_config = config;
        self
    }

    /// Fit the model on a Polars DataFrame
    ///
    /// This is the main entry point. It:
    /// 1. Profiles the DataFrame
    /// 2. Determines preprocessing strategy
    /// 3. Plans feature engineering
    /// 4. Analyzes for mode selection
    /// 5. Tunes hyperparameters
    /// 6. Trains the final model
    pub fn fit(&self, df: &DataFrame, target_col: &str) -> Result<BuildResult> {
        let start = Instant::now();
        let mut phase_times = BuildPhaseTimes::default();

        // Adapt configuration based on time budget
        let mut adapted_config = self.config.clone();
        if let Some(budget) = self.config.time_budget {
            if self.config.verbose {
                println!("AutoBuilder: Time budget set to {:?}", budget);
            }
        }

        if self.config.verbose {
            println!("AutoBuilder: Starting build process...");
        }

        // === Phase 1: Column Profiling ===
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Profiling,
            progress_pct: TrainingPhase::Profiling.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!("Analyzing {} columns", df.width())),
        });

        let phase_start = Instant::now();
        let profile = self.profile_dataframe(df, target_col)?;
        phase_times.profiling = phase_start.elapsed();

        if self.config.verbose {
            println!(
                "  [Profile] {} columns analyzed, {} dropped, task: {:?}",
                profile.columns.len(),
                profile.drop_columns.len(),
                profile.task_type
            );
        }

        // === Phase 2: Determine Model Type and Preprocessing ===
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Preprocessing,
            progress_pct: TrainingPhase::Preprocessing.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!(
                "{} columns retained",
                profile
                    .columns
                    .len()
                    .saturating_sub(profile.drop_columns.len())
            )),
        });

        let phase_start = Instant::now();
        let (model_type, preprocessing_plan) = self.plan_preprocessing(&profile)?;
        phase_times.preprocessing = phase_start.elapsed();

        if self.config.verbose {
            println!(
                "  [Preprocess] Model type: {:?}, {} steps planned",
                model_type,
                preprocessing_plan.steps.len()
            );
        }

        // === Phase 3: Feature Engineering ===
        // Adapt feature engineering based on remaining time
        let mut skip_features = !adapted_config.feature_engineering.is_enabled();
        if let Some(budget) = self.config.time_budget {
            let elapsed = start.elapsed();
            let remaining = budget.saturating_sub(elapsed);

            // Skip feature engineering if time is tight
            if remaining < Duration::from_secs(20) {
                skip_features = true;
                if self.config.verbose && adapted_config.feature_engineering.is_enabled() {
                    println!("  [Budget] Low time remaining, skipping feature engineering");
                }
            }
        }

        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::FeatureEngineering,
            progress_pct: TrainingPhase::FeatureEngineering.progress_pct(),
            elapsed: start.elapsed(),
            message: if skip_features {
                Some("Skipped".to_string())
            } else {
                Some("Planning features".to_string())
            },
        });

        let phase_start = Instant::now();
        let mut feature_plan = if !skip_features {
            Some(self.plan_features(&profile)?)
        } else {
            None
        };
        phase_times.feature_engineering = phase_start.elapsed();

        if self.config.verbose {
            if let Some(ref plan) = feature_plan {
                println!(
                    "  [Features] Planning: {} polynomial, {} ratio, {} interaction features",
                    plan.polynomial_features.len(),
                    plan.ratio_pairs.len(),
                    plan.interaction_pairs.len()
                );
            }
        }

        // === Phase 3b: Apply Cross-Sectional Features (polynomial, ratio, interaction) ===
        let mut working_df = df.clone();
        let initial_cols = working_df.width();

        // Apply cross-sectional features (polynomial, ratio, interaction) using canonical helper
        working_df = crate::features::apply_feature_plan(
            working_df,
            feature_plan.as_ref(),
            None, // panel_info will be added for time-series features below
        )?;

        if self.config.verbose && feature_plan.as_ref().map_or(false, |p| !p.is_empty()) {
            let added = working_df.width() - initial_cols;
            println!(
                "  [Features] Applied cross-sectional: {} → {} columns (+{})",
                initial_cols,
                working_df.width(),
                added
            );
        }

        // === Phase 3c: Time-Series Feature Engineering ===
        // ALWAYS detect panel data structure (needed for era indices, even if not generating features)
        let panel_info = self.detect_panel_structure(&profile, df);

        if let Some(ref info) = panel_info {
            if self.config.verbose {
                println!(
                    "  [Panel] Detected: {} groups by '{}', ordered by '{}'",
                    info.num_groups, info.group_column, info.date_column
                );
            }

            // Only generate time-series features if feature engineering is enabled
            if !skip_features {
                // Plan time-series features using user config (or default if not specified)
                let default_config = crate::features::SmartFeatureConfig::default();
                let config_option = self.config.feature_engineering.get_config();
                let feature_config = config_option.as_ref().unwrap_or(&default_config);
                let ts_plan =
                    SmartFeatureEngine::plan_timeseries_features(&profile, info, feature_config);

                if !ts_plan.is_empty() {
                    if self.config.verbose {
                        println!(
                            "  [TimeSeries] Generating {} features (lag, rolling, EWMA)",
                            ts_plan.estimated_features
                        );
                    }

                    // Apply time-series features to the DataFrame
                    working_df = crate::features::apply_timeseries_features(
                        working_df, &ts_plan, info, true,
                    )?;

                    if self.config.verbose {
                        println!(
                            "  [TimeSeries] New shape: {} rows × {} columns",
                            working_df.height(),
                            working_df.width()
                        );
                    }

                    // Update feature plan with time-series info
                    if let Some(ref mut plan) = feature_plan {
                        plan.timeseries_features = Some(ts_plan);
                    }
                }
            }
        }

        // === Phase 4: Prepare Dataset ===
        // Convert DataFrame to BinnedDataset (and capture pipeline state for inference!)
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::DatasetPreparation,
            progress_pct: TrainingPhase::DatasetPreparation.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!(
                "{} rows × {} cols",
                working_df.height(),
                working_df.width()
            )),
        });

        let (dataset, pipeline_state, filtered_df) =
            self.prepare_dataset_and_state(&working_df, target_col, panel_info.as_ref())?;

        // Prepare validation dataset if custom validation data provided
        let validation_dataset = if let Some(ref val_df) = self.validation_df {
            // Apply same time-series features to validation data
            let val_working_df = if let (Some(ref info), Some(ref plan)) =
                (&panel_info, &feature_plan)
            {
                if let Some(ref ts_plan) = plan.timeseries_features {
                    crate::features::apply_timeseries_features(val_df.clone(), ts_plan, info, true)?
                } else {
                    val_df.clone()
                }
            } else {
                val_df.clone()
            };

            // CRITICAL: Reuse era_column from training data's panel_info!
            // Don't re-detect panel structure on validation set (may fail due to different cardinality ratio)
            let era_column = panel_info.as_ref().map(|info| info.date_column.as_str());
            let (val_dataset, _, _) =
                self.prepare_dataset_with_era_column(&val_working_df, target_col, era_column)?;
            Some(val_dataset)
        } else {
            None
        };

        // === Phase 5: Dataset Analysis (for mode selection) ===
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Analysis,
            progress_pct: TrainingPhase::Analysis.progress_pct(),
            elapsed: start.elapsed(),
            message: Some("Selecting optimal mode".to_string()),
        });

        let phase_start = Instant::now();
        let (mode, analysis, mode_confidence) = self.select_mode(&dataset)?;
        phase_times.analysis = phase_start.elapsed();

        if self.config.verbose {
            println!(
                "  [Analysis] Selected mode: {:?}, confidence: {:?}",
                mode, mode_confidence
            );
        }

        // === Phase 6: Hyperparameter Tuning ===
        // Adapt tuning level based on remaining time
        if let Some(budget) = self.config.time_budget {
            let elapsed = start.elapsed();
            let remaining = budget.saturating_sub(elapsed);

            // Adjust tuning level based on remaining time
            if remaining < Duration::from_secs(10) {
                // Less than 10 seconds - skip tuning
                adapted_config.tuning_level = TuningLevel::None;
                if self.config.verbose {
                    println!("  [Budget] Low time remaining, skipping tuning");
                }
            } else if remaining < Duration::from_secs(30) {
                // Less than 30 seconds - use quick tuning
                if adapted_config.tuning_level != TuningLevel::None {
                    adapted_config.tuning_level = TuningLevel::Quick;
                    if self.config.verbose {
                        println!("  [Budget] Limited time, using quick tuning");
                    }
                }
            }
        }

        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Tuning,
            progress_pct: TrainingPhase::Tuning.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!("{:?} - {:?}", adapted_config.tuning_level, mode)),
        });

        let phase_start = Instant::now();
        let (universal_config, ltt_tuning, tree_tuning) =
            // CRITICAL: If user provided custom_config, use it directly (bypasses all tuning)
            if let Some(ref custom) = self.config.custom_config {
                (custom.clone(), None, None)
            } else if adapted_config.tuning_level == TuningLevel::None {
                // Skip tuning, use defaults
                let config = tuning::create_config_for_mode(mode, self.config.tuning_level);
                (config, None, None)
            } else {
                // CRITICAL: Pass filtered_df (preprocessed) to tuning for LTT raw feature extraction
                // Also pass custom validation dataset if provided
                tuning::tune_hyperparameters(
                    &self.config,
                    &dataset,
                    validation_dataset.as_ref(),
                    mode,
                    &filtered_df,
                    target_col,
                    &profile,
                )?
            };
        phase_times.tuning = phase_start.elapsed();

        if self.config.verbose {
            let num_trials = ltt_tuning
                .as_ref()
                .map(|t| t.history.linear_trials.len())
                .or_else(|| tree_tuning.as_ref().map(|t| t.num_trials))
                .unwrap_or(0);
            println!("  [Tuning] {} trials completed", num_trials);
        }

        // === Phase 7: Train Final Model ===
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Training,
            progress_pct: TrainingPhase::Training.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!("{:?} model", mode)),
        });

        let phase_start = Instant::now();
        // LinearThenTree Feature Extraction Strategy
        // ==========================================
        //
        // Critical design for LinearThenTree mode:
        //
        // 1. TREE COMPONENT needs ALL features (including categoricals, IDs, etc.)
        //    - BinnedDataset contains all preprocessed features
        //    - Trees make splits on the full feature space
        //
        // 2. LINEAR COMPONENT may want a SUBSET (e.g., exclude IDs, categoricals)
        //    - Linear models work best with numeric predictors
        //    - User can configure which features to exclude via LinearFeatureConfig
        //
        // Implementation approach:
        // - Extract ALL features as raw_features array (matches BinnedDataset)
        // - Build linear_feature_indices to select subset for linear model
        // - During training: linear_booster uses selected indices
        // - During prediction: same indices applied to maintain consistency
        //
        // This ensures:
        // - Training and inference use identical feature selection
        // - Trees use full feature space (no information loss)
        // - Linear model uses only appropriate features (better generalization)
        //
        let (feature_extractor, raw_features, linear_indices) =
            if matches!(mode, BoostingMode::LinearThenTree) {
                // Step 1: Extract ALL features (no filtering) to match BinnedDataset
                let all_config = LinearFeatureConfig {
                    exclude_columns: std::collections::HashSet::new(),
                    exclude_categorical: false,
                    exclude_id: false,
                    exclude_constant: false,
                    exclude_boolean: false,
                    exclude_datetime: false,
                    exclude_text: false,
                };
                let all_extractor =
                    crate::dataset::feature_extractor::FeatureExtractor::with_config(all_config);
                let (all_features, _num_all_features) =
                    all_extractor.extract(&filtered_df, target_col)?;

                // Step 2: Build linear_feature_indices based on filter config
                let filter_config = &self.config.linear_feature_config;
                let filter_extractor =
                    crate::dataset::feature_extractor::FeatureExtractor::with_config(
                        filter_config.clone(),
                    );
                let mut linear_feature_indices = Vec::new();

                let col_names: Vec<String> = filtered_df
                    .get_column_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();

                // Build indices relative to EXTRACTED features (target excluded)
                let mut feature_idx = 0;
                for col_name in &col_names {
                    if col_name == target_col {
                        continue; // Skip target
                    }
                    // Check if this column should be used by linear (based on filter config)
                    if !filter_extractor.should_exclude_column(&filtered_df, col_name, target_col) {
                        linear_feature_indices.push(feature_idx);
                    }
                    feature_idx += 1; // Increment for each non-target feature
                }

                (
                    Some(all_extractor),
                    Some(all_features),
                    Some(linear_feature_indices),
                )
            } else {
                (None, None, None)
            };
        // Handle ensemble configuration by setting ensemble_seeds in UniversalConfig
        let final_config = if let Some(ref ensemble_config) = adapted_config.ensemble.get_config() {
            // Only PureTree and LinearThenTree support ensembles
            if matches!(mode, BoostingMode::PureTree | BoostingMode::LinearThenTree) {
                // Generate ensemble seeds from multi_seed config
                let seeds: Vec<u64> = (0..ensemble_config.multi_seed.n_seeds)
                    .map(|i| ensemble_config.multi_seed.base_seed + i as u64)
                    .collect();

                if self.config.verbose {
                    println!("  [Ensemble] Training with {} seeds", seeds.len());
                }

                // Convert StackingConfig to StackingStrategy
                let stacking_strategy = crate::model::universal::config::StackingStrategy::Ridge {
                    alpha: ensemble_config.stacking.alpha,
                    rank_transform: ensemble_config.stacking.rank_transform,
                    fit_intercept: ensemble_config.stacking.fit_intercept,
                    min_weight: ensemble_config.stacking.min_weight,
                };

                // Set ensemble_seeds and stacking_strategy in UniversalConfig
                universal_config
                    .with_ensemble_seeds(seeds)
                    .with_stacking_strategy(stacking_strategy)
            } else {
                if self.config.verbose {
                    println!(
                        "  [Ensemble] Skipped (mode {:?} does not support ensembles)",
                        mode
                    );
                }
                universal_config
            }
        } else {
            universal_config
        };

        // Check if skip_training is enabled (discovery mode)
        if self.config.skip_training {
            // Save discovered config without training
            if let Some(ref output_dir) = self.config.model_output_dir {
                use std::fs;
                fs::create_dir_all(output_dir)?;

                // Build enriched config with all discovered settings
                let mut enriched_config = final_config.clone();
                enriched_config.feature_plan = feature_plan.clone();
                enriched_config.preprocessing_plan = Some(preprocessing_plan.clone());
                enriched_config.target_column = Some(target_col.to_string());

                // Save config.json only (no model.rkyv)
                let config_path = output_dir.join("config.json");
                let config_json = serde_json::to_string_pretty(&enriched_config).map_err(|e| {
                    TreeBoostError::Serialization(format!("Failed to serialize config: {}", e))
                })?;
                fs::write(&config_path, config_json)?;

                if self.config.verbose {
                    println!("  [Discovery] Config saved: {}", config_path.display());
                    println!("  [Discovery] Skip training enabled - no model trained");
                    println!("  [Discovery] Use UniversalModel::train() with this config for production training");
                }
            }

            // Return BuildResult with skip_training=true and no model
            return Ok(BuildResult {
                model: None,
                skip_training: true,
                mode,
                target_column: target_col.to_string(),
                mode_confidence: Some(mode_confidence),
                preprocessing_plan: Some(preprocessing_plan),
                feature_plan,
                ltt_tuning: None,
                tree_tuning: None,
                column_profile: Some(profile),
                analysis,
                pipeline_state: Some(pipeline_state),
                build_time: start.elapsed(),
                phase_times,
            });
        }

        // Train UniversalModel (handles both single and ensemble internally)
        let max_rounds = final_config.num_rounds; // Save before moving
        let model = self.train_model(
            &dataset,
            final_config,
            feature_extractor,
            raw_features,
            linear_indices,
        )?;
        phase_times.training = phase_start.elapsed();

        if self.config.verbose {
            let num_trees = model.num_trees();
            let stopped_early = num_trees < max_rounds;

            println!("  [Train] Model trained in {:?}", phase_times.training);
            println!(
                "  [Train] Trees: {}/{} {}",
                num_trees,
                max_rounds,
                if stopped_early {
                    "(early stopped)"
                } else {
                    "(full rounds)"
                }
            );
        }

        // Emit completion progress
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Complete,
            progress_pct: TrainingPhase::Complete.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!("Total: {:?}", start.elapsed())),
        });

        let build_result = BuildResult {
            model: Some(model),
            skip_training: false,
            mode,
            target_column: target_col.to_string(),
            mode_confidence: Some(mode_confidence),
            preprocessing_plan: Some(preprocessing_plan),
            feature_plan,
            ltt_tuning,
            tree_tuning,
            column_profile: Some(profile),
            analysis,
            pipeline_state: Some(pipeline_state), // CRITICAL for inference!
            build_time: start.elapsed(),
            phase_times,
        };

        // Auto-save artifacts if model_output_dir is configured
        if let Some(ref output_dir) = self.config.model_output_dir {
            self.auto_save_artifacts(&build_result, output_dir)?;
        }

        Ok(build_result)
    }

    /// Profile a DataFrame to understand column types
    fn profile_dataframe(&self, df: &DataFrame, target_col: &str) -> Result<DataFrameProfile> {
        // Skip correlations when mode is fixed and feature engineering disabled
        let skip_correlations =
            !self.config.mode_selection.is_auto() && !self.config.feature_engineering.is_enabled();
        DataFrameProfile::analyze_with_options(df, target_col, skip_correlations)
    }

    /// Plan preprocessing based on profile and model type
    fn plan_preprocessing(
        &self,
        profile: &DataFrameProfile,
    ) -> Result<(ModelType, PreprocessingPlan)> {
        // For now, use a simple heuristic based on profile
        // If the data looks linear (high correlation with target), use LinearThenTree
        // Otherwise use Tree
        let has_linear_signal = profile.columns.iter().any(|c| {
            c.target_correlation
                .map(|r| r.abs() > auto_defaults::LINEAR_SIGNAL_THRESHOLD)
                .unwrap_or(false)
        });

        let model_type = if has_linear_signal {
            ModelType::LinearThenTree
        } else {
            ModelType::Tree
        };

        let plan = SmartPreprocessor::infer(profile, model_type);
        Ok((model_type, plan))
    }

    /// Plan feature engineering based on profile
    fn plan_features(&self, profile: &DataFrameProfile) -> Result<FeaturePlan> {
        // Use user-provided feature config, or default if not specified
        let default_config = crate::features::SmartFeatureConfig::default();
        let config_option = self.config.feature_engineering.get_config();
        let config = config_option.as_ref().unwrap_or(&default_config);

        let plan = SmartFeatureEngine::infer_with_config(profile, None, config);
        Ok(plan)
    }

    /// Prepare dataset and return pipeline state (for AutoModel inference)
    fn prepare_dataset_and_state(
        &self,
        df: &DataFrame,
        target_col: &str,
        panel_info: Option<&crate::analysis::PanelDataInfo>,
    ) -> Result<(BinnedDataset, crate::dataset::PipelineState, DataFrame)> {
        // Use DataPipeline to create BinnedDataset
        let pipeline = DataPipeline::with_defaults();

        // Pass None for categorical_columns to enable auto-detection
        // (String/Categorical dtypes will be treated as categorical)
        // Pass date_column as era_column if panel data detected
        let era_column = panel_info.map(|info| info.date_column.as_str());

        let (dataset, pipeline_state, filtered_df) =
            pipeline.process_for_training(df.clone(), target_col, None, era_column)?;

        Ok((dataset, pipeline_state, filtered_df))
    }

    /// Prepare dataset with explicit era column (for validation sets)
    ///
    /// This method accepts an explicit era_column parameter instead of re-detecting panel structure.
    /// Use this when preparing validation data to reuse the era column from training data,
    /// avoiding panel detection failures due to different cardinality ratios.
    fn prepare_dataset_with_era_column(
        &self,
        df: &DataFrame,
        target_col: &str,
        era_column: Option<&str>,
    ) -> Result<(BinnedDataset, crate::dataset::PipelineState, DataFrame)> {
        // Use DataPipeline to create BinnedDataset
        let pipeline = DataPipeline::with_defaults();

        // Pass None for categorical_columns to enable auto-detection
        // Use provided era_column directly (no panel detection)
        let (dataset, pipeline_state, filtered_df) =
            pipeline.process_for_training(df.clone(), target_col, None, era_column)?;

        Ok((dataset, pipeline_state, filtered_df))
    }

    /// Select boosting mode based on analysis
    fn select_mode(
        &self,
        dataset: &BinnedDataset,
    ) -> Result<(BoostingMode, Option<DatasetAnalysis>, Confidence)> {
        // If custom config provided, use its mode
        if let Some(ref custom) = self.config.custom_config {
            return Ok((custom.mode, None, Confidence::High));
        }

        // If mode is fixed, use it
        if let Some(mode) = self.config.mode_selection.fixed_mode() {
            return Ok((mode, None, Confidence::High));
        }

        // Auto mode is enabled - run analysis

        // Run analysis
        let analysis = DatasetAnalysis::analyze(dataset)?;

        let mode = analysis.recommend_mode();
        let confidence = analysis.confidence();

        Ok((mode, Some(analysis), confidence))
    }

    /// Train the final model
    fn train_model(
        &self,
        dataset: &BinnedDataset,
        mut config: UniversalConfig,
        feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
        raw_features: Option<Vec<f32>>,
        linear_indices: Option<Vec<usize>>,
    ) -> Result<UniversalModel> {
        // Use MSE loss as default (could be made configurable)
        let loss = crate::loss::MseLoss;

        // Add feature_extractor to config
        config.feature_extractor = feature_extractor;

        // Pass raw features AND linear_feature_indices to training if we're in LTT mode
        // Use pattern matching instead of is_some() + unwrap() for idiomatic Rust
        match (config.mode, raw_features, linear_indices) {
            (crate::model::BoostingMode::LinearThenTree, Some(features), Some(indices)) => {
                UniversalModel::train_with_linear_feature_selection(
                    dataset, &features, &indices, config, &loss,
                )
            }
            _ => UniversalModel::train(dataset, config, &loss),
        }
    }

    /// Detect panel data structure in the DataFrame
    ///
    /// Tries automatic detection first, then falls back to heuristics
    /// for common patterns (e.g., numeric date columns).
    fn detect_panel_structure(
        &self,
        profile: &DataFrameProfile,
        df: &DataFrame,
    ) -> Option<PanelDataInfo> {
        // Try automatic detection first
        if let Some(info) = profile.detect_panel_structure(df) {
            return Some(info);
        }

        // Heuristic: Look for categorical column + numeric column named "date"
        // This handles cases where date is stored as integer (e.g., YYYYMMDD or ordinal)
        // CRITICAL: Exclude low-cardinality categoricals (e.g., "time_of_day" with 3 values = categorical period, not timestamp)

        // Find the date column (must be numeric OR high-cardinality categorical)
        let date_col_name = df
            .get_column_names()
            .iter()
            .find(|c| {
                let lower = c.to_lowercase();
                if !lower.contains("date") && !lower.contains("time") && lower != "dt" {
                    return false;
                }

                // Check the column's dtype and cardinality
                if let Ok(series) = df.column(c) {
                    let dtype = series.dtype();
                    // Accept numeric columns (integer timestamps)
                    if dtype.is_numeric() {
                        return true;
                    }
                    // For categorical: only accept if many unique values (>MIN_DATE_CARDINALITY)
                    // Low cardinality = categorical time period (morning/afternoon), not timestamps
                    if matches!(
                        dtype,
                        polars::prelude::DataType::String
                            | polars::prelude::DataType::Categorical(_, _)
                    ) {
                        if let Ok(n_unique) = series.n_unique() {
                            return n_unique > crate::defaults::analysis::MIN_DATE_CARDINALITY;
                        }
                    }
                }
                false
            })?
            .to_string();

        // Find a categorical column that could be the group
        let group_col = profile.columns.iter().find(|c| {
            c.dtype == crate::analysis::ColumnDataType::Categorical
                && c.cardinality >= 2
                && c.cardinality <= 10000
                && c.cardinality_ratio < 0.5
        })?;

        // Compute statistics
        let num_groups = group_col.cardinality;
        let avg_obs = df.height() as f32 / num_groups as f32;

        Some(PanelDataInfo {
            group_column: group_col.name.clone(),
            date_column: date_col_name,
            num_groups,
            avg_observations_per_group: avg_obs,
            min_observations_per_group: (avg_obs * 0.5) as usize,
            max_observations_per_group: (avg_obs * 1.5) as usize,
            confidence: 0.7, // Lower confidence for heuristic detection
            time_granularity: crate::analysis::TimeGranularity::Daily, // Default assumption
        })
    }

    /// Automatically save all artifacts (model, config, metadata) to output directory
    fn auto_save_artifacts(
        &self,
        build_result: &BuildResult,
        output_dir: &std::path::Path,
    ) -> Result<()> {
        use std::fs;

        // Create output directory if it doesn't exist
        fs::create_dir_all(output_dir)?;

        // If skip_training mode, the config was already saved inline during fit()
        // No model to save, only config was produced
        if build_result.skip_training {
            return Ok(());
        }

        // Extract model (safe to unwrap since skip_training is false)
        let model = build_result
            .model
            .as_ref()
            .expect("model must be Some when skip_training is false");

        // Update model config to include feature plan, preprocessing plan, and target column
        // This makes UniversalConfig truly universal - it contains everything needed to replicate the pipeline
        let mut enriched_config = model.config().clone();
        enriched_config.feature_plan = build_result.feature_plan.clone();
        enriched_config.preprocessing_plan = build_result.preprocessing_plan.clone();
        enriched_config.target_column = Some(build_result.target_column.clone());

        // Save trained model (rkyv format for fast loading)
        // Note: The model internally stores its config, which now includes feature_plan/preprocessing_plan
        let model_path = output_dir.join("model.rkyv");
        model.save(&model_path)?;
        if self.config.verbose {
            println!("  [AutoSave] Model: {}", model_path.display());
        }

        // Save UniversalConfig as config.json
        // This is the SINGLE SOURCE OF TRUTH containing everything:
        // - Model hyperparameters
        // - Feature engineering plan (polynomial, interactions, ratios)
        // - Preprocessing plan (encoders, scalers, imputers)
        // - Target column name
        let config_path = output_dir.join("config.json");
        let config_json = serde_json::to_string_pretty(&enriched_config).map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to serialize config: {}", e))
        })?;
        fs::write(&config_path, config_json)?;
        if self.config.verbose {
            println!("  [AutoSave] Config: {}", config_path.display());
        }

        Ok(())
    }
}

impl Default for AutoBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_config_defaults() {
        let config = AutoConfig::default();
        assert_eq!(config.tuning_level, TuningLevel::Standard);
        assert!((config.val_ratio - 0.2).abs() < 0.01);
        assert!(config.feature_engineering.is_enabled());
        assert!(config.preprocessing.is_enabled());
        assert_eq!(config.mode_selection, ModeSelection::Auto);
    }

    #[test]
    fn test_auto_builder_creation() {
        let builder = AutoBuilder::new()
            .with_tuning(TuningLevel::Quick)
            .with_verbose(true);

        assert_eq!(builder.config.tuning_level, TuningLevel::Quick);
        assert!(builder.config.verbose);
    }

    #[test]
    fn test_tuning_level_variants() {
        assert_eq!(TuningLevel::default(), TuningLevel::Standard);
    }
}
