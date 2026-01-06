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

use crate::analysis::{
    Confidence, DatasetAnalysis, DataFrameProfile,
};
use crate::dataset::{BinnedDataset, DataPipeline, PipelineConfig};
use crate::features::{SmartFeatureEngine, FeaturePlan};
use crate::learner::TreeConfig;
use crate::model::{BoostingMode, UniversalConfig, UniversalModel};
use crate::preprocessing::{
    SmartPreprocessor, PreprocessingPlan, ModelType,
};
use crate::tuner::ltt::{LttTuner, LttTunerConfig, LttTuningResult};
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use std::time::{Duration, Instant};

/// Tuning intensity level
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TuningLevel {
    /// Minimal tuning - use sensible defaults
    /// Best for: Quick experiments, small datasets
    Quick,

    /// Moderate tuning - good balance of speed and quality
    /// Best for: Most production use cases
    #[default]
    Standard,

    /// Extensive tuning - thorough hyperparameter search
    /// Best for: Maximum accuracy when time is not a constraint
    Thorough,

    /// No tuning - use provided hyperparameters
    None,
}

/// AutoBuilder configuration
#[derive(Debug, Clone)]
pub struct AutoConfig {
    /// Tuning intensity
    pub tuning_level: TuningLevel,

    /// Validation split ratio (default: 0.2)
    pub val_ratio: f32,

    /// Whether to enable automatic feature engineering
    pub auto_features: bool,

    /// Whether to enable automatic preprocessing
    pub auto_preprocessing: bool,

    /// Whether to enable automatic mode selection
    pub auto_mode: bool,

    /// Force a specific mode (overrides auto_mode if set)
    pub force_mode: Option<BoostingMode>,

    /// Maximum number of features to generate
    pub max_generated_features: usize,

    /// Random seed for reproducibility
    pub seed: u64,

    /// Verbose output
    pub verbose: bool,
}

impl Default for AutoConfig {
    fn default() -> Self {
        Self {
            tuning_level: TuningLevel::Standard,
            val_ratio: 0.2,
            auto_features: true,
            auto_preprocessing: true,
            auto_mode: true,
            force_mode: None,
            max_generated_features: 50,
            seed: 42,
            verbose: false,
        }
    }
}

impl AutoConfig {
    /// Create a new AutoConfig with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tuning level
    pub fn with_tuning(mut self, level: TuningLevel) -> Self {
        self.tuning_level = level;
        self
    }

    /// Set validation split ratio
    pub fn with_validation_split(mut self, ratio: f32) -> Self {
        self.val_ratio = ratio.clamp(0.1, 0.4);
        self
    }

    /// Enable/disable automatic feature engineering
    pub fn with_auto_features(mut self, enabled: bool) -> Self {
        self.auto_features = enabled;
        self
    }

    /// Enable/disable automatic preprocessing
    pub fn with_auto_preprocessing(mut self, enabled: bool) -> Self {
        self.auto_preprocessing = enabled;
        self
    }

    /// Enable/disable automatic mode selection
    pub fn with_auto_mode(mut self, enabled: bool) -> Self {
        self.auto_mode = enabled;
        self
    }

    /// Force a specific boosting mode
    pub fn with_mode(mut self, mode: BoostingMode) -> Self {
        self.force_mode = Some(mode);
        self.auto_mode = false;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Build result containing the trained model and metadata
#[derive(Debug)]
pub struct BuildResult {
    /// The trained model
    pub model: UniversalModel,

    /// The boosting mode used
    pub mode: BoostingMode,

    /// Mode selection confidence (if auto mode was used)
    pub mode_confidence: Option<Confidence>,

    /// Preprocessing plan that was applied
    pub preprocessing_plan: Option<PreprocessingPlan>,

    /// Feature engineering plan that was applied
    pub feature_plan: Option<FeaturePlan>,

    /// LTT tuning result (if LTT mode was used)
    pub ltt_tuning: Option<LttTuningResult>,

    /// Column profile from analysis
    pub column_profile: Option<DataFrameProfile>,

    /// Dataset analysis result
    pub analysis: Option<DatasetAnalysis>,

    /// Total build time
    pub build_time: Duration,

    /// Time breakdown by phase
    pub phase_times: BuildPhaseTimes,
}

/// Time breakdown for build phases
#[derive(Debug, Clone, Default)]
pub struct BuildPhaseTimes {
    pub profiling: Duration,
    pub preprocessing: Duration,
    pub feature_engineering: Duration,
    pub analysis: Duration,
    pub tuning: Duration,
    pub training: Duration,
}

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

        if self.config.verbose {
            println!("AutoBuilder: Starting build process...");
        }

        // === Phase 1: Column Profiling ===
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
        let phase_start = Instant::now();
        let feature_plan = if self.config.auto_features {
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
        // Convert DataFrame to BinnedDataset
        let dataset = self.prepare_dataset(df, target_col, &profile)?;

        // === Phase 5: Dataset Analysis (for mode selection) ===
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
        let phase_start = Instant::now();
        let (universal_config, ltt_tuning) = self.tune_hyperparameters(
            &dataset,
            mode,
            df,
            target_col,
            &profile,
        )?;
        phase_times.tuning = phase_start.elapsed();

        if self.config.verbose {
            println!(
                "  [Tuning] {} trials completed",
                ltt_tuning.as_ref().map(|t| t.history.linear_trials.len()).unwrap_or(0)
            );
        }

        // === Phase 7: Train Final Model ===
        let phase_start = Instant::now();
        let model = self.train_model(&dataset, universal_config)?;
        phase_times.training = phase_start.elapsed();

        if self.config.verbose {
            println!(
                "  [Train] Model trained in {:?}",
                phase_times.training
            );
        }

        Ok(BuildResult {
            model,
            mode,
            mode_confidence: Some(mode_confidence),
            preprocessing_plan: Some(preprocessing_plan),
            feature_plan,
            ltt_tuning,
            column_profile: Some(profile),
            analysis,
            build_time: start.elapsed(),
            phase_times,
        })
    }

    /// Profile a DataFrame to understand column types
    fn profile_dataframe(&self, df: &DataFrame, target_col: &str) -> Result<DataFrameProfile> {
        DataFrameProfile::analyze(df, target_col)
    }

    /// Plan preprocessing based on profile and model type
    fn plan_preprocessing(&self, profile: &DataFrameProfile) -> Result<(ModelType, PreprocessingPlan)> {
        // For now, use a simple heuristic based on profile
        // If the data looks linear (high correlation with target), use LinearThenTree
        // Otherwise use Tree
        let has_linear_signal = profile.columns.iter().any(|c| {
            c.target_correlation.map(|r| r.abs() > 0.3).unwrap_or(false)
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

    /// Prepare dataset from DataFrame
    fn prepare_dataset(
        &self,
        df: &DataFrame,
        target_col: &str,
        _profile: &DataFrameProfile,
    ) -> Result<BinnedDataset> {
        // Get feature column names (all except target)
        let feature_cols: Vec<String> = df
            .get_column_names()
            .iter()
            .filter(|&name| name.as_str() != target_col)
            .map(|s| s.to_string())
            .collect();

        // Convert to &str slice for DataPipeline
        let feature_col_refs: Vec<&str> = feature_cols.iter().map(|s| s.as_str()).collect();

        // Use DataPipeline to create BinnedDataset
        let pipeline_config = PipelineConfig::default();
        let pipeline = DataPipeline::new(pipeline_config);

        let (dataset, _state) = pipeline.process_for_training(
            df.clone(),
            target_col,
            Some(&feature_col_refs),
        )?;

        Ok(dataset)
    }

    /// Select boosting mode based on analysis
    fn select_mode(
        &self,
        dataset: &BinnedDataset,
    ) -> Result<(BoostingMode, Option<DatasetAnalysis>, Confidence)> {
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

    /// Tune hyperparameters based on mode
    fn tune_hyperparameters(
        &self,
        dataset: &BinnedDataset,
        mode: BoostingMode,
        df: &DataFrame,
        target_col: &str,
        profile: &DataFrameProfile,
    ) -> Result<(UniversalConfig, Option<LttTuningResult>)> {
        match self.config.tuning_level {
            TuningLevel::None => {
                // No tuning - use defaults
                let config = UniversalConfig::default().with_mode(mode);
                Ok((config, None))
            }
            _ => {
                // Tune based on mode
                match mode {
                    BoostingMode::LinearThenTree => {
                        self.tune_ltt(dataset, df, target_col, profile)
                    }
                    _ => {
                        // For PureTree and RandomForest, use standard config
                        let config = self.create_config_for_mode(mode, dataset);
                        Ok((config, None))
                    }
                }
            }
        }
    }

    /// Tune LinearThenTree mode
    fn tune_ltt(
        &self,
        _dataset: &BinnedDataset,
        df: &DataFrame,
        target_col: &str,
        profile: &DataFrameProfile,
    ) -> Result<(UniversalConfig, Option<LttTuningResult>)> {
        // Extract raw features for linear tuning
        let (features, targets, num_features) = self.extract_raw_features(df, target_col, profile)?;

        // Create tuner config based on tuning level
        let tuner_config = match self.config.tuning_level {
            TuningLevel::Quick => LttTunerConfig::quick(),
            TuningLevel::Standard => LttTunerConfig::default(),
            TuningLevel::Thorough => LttTunerConfig::thorough(),
            TuningLevel::None => LttTunerConfig::quick(), // Should not reach here
        };

        let tuner = LttTuner::new(tuner_config);
        let result = tuner.tune(&features, num_features, &targets)?;

        // Build UniversalConfig from tuning result
        let tree_config = TreeConfig::default()
            .with_max_depth(result.tree_params.max_depth as usize);

        let config = UniversalConfig::default()
            .with_mode(BoostingMode::LinearThenTree)
            .with_learning_rate(result.tree_params.learning_rate)
            .with_num_rounds(result.tree_params.num_rounds as usize)
            .with_tree_config(tree_config)
            .with_linear_config(result.linear_params.to_config());

        Ok((config, Some(result)))
    }

    /// Extract raw features from DataFrame for linear model tuning
    ///
    /// NOTE: This is a simplified extraction for LTT hyperparameter tuning.
    /// - Only numeric features are used (categorical features need encoding first)
    /// - Columns marked for dropping in the profile are excluded
    /// - For production prediction, use the full DataPipeline with preprocessing
    fn extract_raw_features(
        &self,
        df: &DataFrame,
        target_col: &str,
        profile: &DataFrameProfile,
    ) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        let num_rows = df.height();

        // Build set of columns to drop based on profile
        let drop_cols: std::collections::HashSet<String> = profile
            .drop_columns
            .iter()
            .map(|d| d.name.clone())
            .collect();

        // Get numeric columns only, excluding dropped columns
        let numeric_cols: Vec<String> = df
            .get_column_names()
            .iter()
            .filter(|&name| {
                let name_str = name.as_str();

                // Skip target column
                if name_str == target_col {
                    return false;
                }

                // Skip columns marked for dropping
                if drop_cols.contains(name_str) {
                    return false;
                }

                // Keep only numeric types
                match df.column(name_str) {
                    Ok(col) => matches!(
                        col.dtype(),
                        DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::Int64
                    ),
                    Err(_) => false,
                }
            })
            .map(|s| s.to_string())
            .collect();

        let num_features = numeric_cols.len();
        if num_features == 0 {
            return Err(TreeBoostError::Data(
                format!(
                    "No numeric features found for linear model tuning. \
                     Dataset has {} columns, {} dropped (constant/ID/text), {} remaining. \
                     Categorical features need encoding first - consider using DataPipeline.",
                    df.width(),
                    drop_cols.len(),
                    df.width() - drop_cols.len() - 1 // -1 for target
                )
            ));
        }

        // Extract features (row-major)
        let mut features = Vec::with_capacity(num_rows * num_features);
        for row_idx in 0..num_rows {
            for col_name in &numeric_cols {
                let col = df.column(col_name)?;
                let val = col.get(row_idx).map_err(|e| TreeBoostError::Data(e.to_string()))?;
                let f_val = match val {
                    AnyValue::Float32(v) => v,
                    AnyValue::Float64(v) => v as f32,
                    AnyValue::Int32(v) => v as f32,
                    AnyValue::Int64(v) => v as f32,
                    AnyValue::Null => 0.0, // Simple imputation
                    _ => 0.0,
                };
                features.push(f_val);
            }
        }

        // Extract targets
        let target_col_series = df.column(target_col)?;
        let targets: Vec<f32> = (0..num_rows)
            .map(|i| {
                match target_col_series.get(i) {
                    Ok(val) => match val {
                        AnyValue::Float32(v) => v,
                        AnyValue::Float64(v) => v as f32,
                        AnyValue::Int32(v) => v as f32,
                        AnyValue::Int64(v) => v as f32,
                        _ => 0.0,
                    },
                    Err(_) => 0.0,
                }
            })
            .collect();

        Ok((features, targets, num_features))
    }

    /// Create config for non-LTT modes
    fn create_config_for_mode(&self, mode: BoostingMode, _dataset: &BinnedDataset) -> UniversalConfig {
        let (num_rounds, learning_rate, max_depth) = match self.config.tuning_level {
            TuningLevel::Quick => (100, 0.1, 4),
            TuningLevel::Standard => (500, 0.1, 6),
            TuningLevel::Thorough => (1000, 0.05, 8),
            TuningLevel::None => (100, 0.1, 6),
        };

        let tree_config = TreeConfig::default()
            .with_max_depth(max_depth);

        UniversalConfig::default()
            .with_mode(mode)
            .with_num_rounds(num_rounds)
            .with_learning_rate(learning_rate)
            .with_tree_config(tree_config)
    }

    /// Train the final model
    fn train_model(
        &self,
        dataset: &BinnedDataset,
        config: UniversalConfig,
    ) -> Result<UniversalModel> {
        // Use MSE loss as default (could be made configurable)
        let loss = crate::loss::MseLoss;
        UniversalModel::train(dataset, config, &loss)
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
