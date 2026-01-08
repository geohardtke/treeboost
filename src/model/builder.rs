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
//!     .with_validation_split(0.2)
//!     .fit(&df, "target_column")?;
//!
//! // Predict on new data
//! let predictions = model.predict(&test_df)?;
//! ```

use crate::analysis::{Confidence, DataFrameProfile, DatasetAnalysis};
use crate::dataset::feature_extractor::LinearFeatureConfig;
use crate::dataset::{BinnedDataset, DataPipeline};
use crate::features::{FeaturePlan, SmartFeatureEngine};
use crate::model::config::{AutoConfig, BuildPhaseTimes, BuildResult, TuningLevel};
use crate::model::progress::{ProgressCallback, ProgressUpdate, TrainingPhase};
use crate::model::{BoostingMode, UniversalConfig, UniversalModel};
use crate::preprocessing::{ModelType, PreprocessingPlan, SmartPreprocessor};
use crate::Result;
use polars::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Import tuning functions
use super::tuning;

/// AutoBuilder: High-level AutoML interface
pub struct AutoBuilder {
    config: AutoConfig,
}

impl AutoBuilder {
    /// Create a new AutoBuilder with default configuration
    pub fn new() -> Self {
        Self {
            config: AutoConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AutoConfig) -> Self {
        Self { config }
    }

    /// Set tuning level
    pub fn with_tuning(mut self, level: TuningLevel) -> Self {
        self.config.tuning_level = level;
        self
    }

    /// Set validation split ratio
    pub fn with_validation_split(mut self, ratio: f32) -> Self {
        self.config.val_ratio = ratio;
        self
    }

    /// Enable/disable automatic features
    pub fn with_auto_features(mut self, enabled: bool) -> Self {
        self.config.auto_features = enabled;
        self
    }

    /// Force a specific mode
    pub fn with_mode(mut self, mode: BoostingMode) -> Self {
        self.config.force_mode = Some(mode);
        self.config.auto_mode = false;
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
        let mut skip_features = !adapted_config.auto_features;
        if let Some(budget) = self.config.time_budget {
            let elapsed = start.elapsed();
            let remaining = budget.saturating_sub(elapsed);

            // Skip feature engineering if time is tight
            if remaining < Duration::from_secs(20) {
                skip_features = true;
                if self.config.verbose && adapted_config.auto_features {
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
        let feature_plan = if !skip_features {
            Some(self.plan_features(&profile)?)
        } else {
            None
        };
        phase_times.feature_engineering = phase_start.elapsed();

        if self.config.verbose {
            if let Some(ref plan) = feature_plan {
                println!(
                    "  [Features] {} polynomial, {} ratio, {} interaction features",
                    plan.polynomial_features.len(),
                    plan.ratio_pairs.len(),
                    plan.interaction_pairs.len()
                );
            }
        }

        // === Phase 4: Prepare Dataset ===
        // Convert DataFrame to BinnedDataset (and capture pipeline state for inference!)
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::DatasetPreparation,
            progress_pct: TrainingPhase::DatasetPreparation.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!("{} rows", df.height())),
        });

        let (dataset, pipeline_state, filtered_df) =
            self.prepare_dataset_and_state(df, target_col)?;

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
                tuning::tune_hyperparameters(
                    &self.config,
                    &dataset,
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
        let model = self.train_model(
            &dataset,
            universal_config,
            feature_extractor,
            raw_features,
            linear_indices,
        )?;
        phase_times.training = phase_start.elapsed();

        if self.config.verbose {
            println!("  [Train] Model trained in {:?}", phase_times.training);
        }

        // Emit completion progress
        self.config.progress_callback.on_progress(&ProgressUpdate {
            phase: TrainingPhase::Complete,
            progress_pct: TrainingPhase::Complete.progress_pct(),
            elapsed: start.elapsed(),
            message: Some(format!("Total: {:?}", start.elapsed())),
        });

        Ok(BuildResult {
            model,
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
        })
    }

    /// Profile a DataFrame to understand column types
    fn profile_dataframe(&self, df: &DataFrame, target_col: &str) -> Result<DataFrameProfile> {
        DataFrameProfile::analyze(df, target_col)
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
                .map(|r| r.abs() > crate::model::config::LINEAR_SIGNAL_THRESHOLD)
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
        let plan = SmartFeatureEngine::infer(profile, None);
        Ok(plan)
    }

    /// Prepare dataset and return pipeline state (for AutoModel inference)
    fn prepare_dataset_and_state(
        &self,
        df: &DataFrame,
        target_col: &str,
    ) -> Result<(BinnedDataset, crate::dataset::PipelineState, DataFrame)> {
        // Use DataPipeline to create BinnedDataset
        let pipeline = DataPipeline::with_defaults();

        // Pass None for categorical_columns to enable auto-detection
        // (String/Categorical dtypes will be treated as categorical)
        let (dataset, pipeline_state, filtered_df) =
            pipeline.process_for_training(df.clone(), target_col, None)?;

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

        // If mode is forced, use it
        if let Some(mode) = self.config.force_mode {
            return Ok((mode, None, Confidence::High));
        }

        // If auto mode is disabled, use PureTree
        if !self.config.auto_mode {
            return Ok((BoostingMode::PureTree, None, Confidence::Medium));
        }

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
        assert!(config.auto_features);
        assert!(config.auto_preprocessing);
        assert!(config.auto_mode);
    }

    #[test]
    fn test_auto_config_builder() {
        let config = AutoConfig::new()
            .with_tuning(TuningLevel::Thorough)
            .with_validation_split(0.3)
            .with_auto_features(false)
            .with_mode(BoostingMode::LinearThenTree);

        assert_eq!(config.tuning_level, TuningLevel::Thorough);
        assert!((config.val_ratio - 0.3).abs() < 0.01);
        assert!(!config.auto_features);
        assert_eq!(config.force_mode, Some(BoostingMode::LinearThenTree));
        assert!(!config.auto_mode);
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
