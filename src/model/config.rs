//! Configuration types for AutoBuilder
//!
//! This module contains all configuration structs, enums, and result types
//! used by the AutoBuilder interface.

use crate::analysis::{Confidence, DataFrameProfile, DatasetAnalysis};
use crate::dataset::feature_extractor::LinearFeatureConfig;
use crate::defaults::{auto as auto_defaults, seeds as seeds_defaults};
use crate::ensemble::{MultiSeedConfig, SelectionConfig, StackingConfig};
use crate::features::FeaturePlan;
use crate::model::progress::{ProgressCallback, QuietProgress};
use crate::model::{BoostingMode, UniversalModel};
use crate::preprocessing::PreprocessingPlan;
use crate::tuner::ltt::LttTuningResult;
use crate::tuner::{OptimizationMetric, TaskType as TunerTaskType};
use std::sync::Arc;
use std::time::Duration;

/// Tuning intensity level for AutoBuilder's hyperparameter optimization.
///
/// ## When to Use TuningLevel
///
/// Use `TuningLevel` when working with **`AutoBuilder`** or **`AutoConfig`**.
/// This enum controls AutoBuilder's automatic tuning pipeline intensity.
///
/// ## Relationship to Other Preset Enums
///
/// TreeBoost has several tuning-related preset enums for different contexts:
///
/// | Enum | Context | Purpose |
/// |------|---------|---------|
/// | **`TuningLevel`** | `AutoBuilder`, `AutoConfig` | High-level: AutoML tuning intensity |
/// | **`TunerPreset`** | `AutoTuner`, `TunerConfig` | Mid-level: Manual tuning intensity |
/// | **`TreeTunerPreset`** | `TreeTunerConfig` | Low-level: Tree-only tuning intensity |
///
/// **Mapping between presets:**
/// - `TuningLevel::Quick` ≈ `TunerPreset::Quick` ≈ `TreeTunerPreset::Quick`
/// - `TuningLevel::Standard` ≈ `TunerPreset::Balanced` ≈ `TreeTunerPreset::Standard`
/// - `TuningLevel::Thorough` ≈ `TunerPreset::Thorough` ≈ `TreeTunerPreset::Thorough`
/// - `TuningLevel::None` = No tuning (use provided hyperparameters)
///
/// ## Variants
///
/// ### `Quick`
/// Minimal tuning with sensible defaults.
/// - **Best for**: Quick experiments, small datasets, CI/debugging
/// - **Time**: Seconds to minutes
/// - **Quality**: Good baseline, not optimized
///
/// ### `Standard` (Default)
/// Moderate tuning with balanced speed and quality.
/// - **Best for**: Most real-world use cases
/// - **Time**: Minutes to tens of minutes
/// - **Quality**: Well-tuned, production-ready
///
/// ### `Thorough`
/// Extensive hyperparameter search for maximum accuracy.
/// - **Best for**: Competitions, research, when accuracy > time
/// - **Time**: Tens of minutes to hours
/// - **Quality**: Highly optimized
///
/// ### `None`
/// No tuning - use provided hyperparameters as-is.
/// - **Best for**: Already-tuned configs, reproducibility
/// - **Time**: Instant (no tuning overhead)
/// - **Quality**: Depends on provided hyperparameters
///
/// ## Examples
///
/// ```ignore
/// use treeboost::{AutoBuilder, TuningLevel};
///
/// // Quick experiments
/// let model = AutoBuilder::new()
///     .with_tuning_level(TuningLevel::Quick)
///     .fit(&df, "target")?;
///
/// // Production model (default)
/// let model = AutoBuilder::new()
///     .with_tuning_level(TuningLevel::Standard)  // or omit (default)
///     .fit(&df, "target")?;
///
/// // Maximum accuracy
/// let model = AutoBuilder::new()
///     .with_tuning_level(TuningLevel::Thorough)
///     .fit(&df, "target")?;
///
/// // No tuning (use specific hyperparameters)
/// let config = UniversalConfig::new().with_num_rounds(100).with_learning_rate(0.1);
/// let model = AutoBuilder::new()
///     .with_tuning_level(TuningLevel::None)
///     .with_config(config)
///     .fit(&df, "target")?;
/// ```
///
/// ## See Also
///
/// - [`TunerPreset`] - For manual tuning with `AutoTuner`
/// - [`TreeTunerPreset`] - For tree-specific tuning with `TreeTunerConfig`
/// - [`AutoBuilder::with_tuning_level`] - Set the tuning level
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TuningLevel {
    /// Minimal tuning with sensible defaults
    Quick,

    /// Moderate tuning with balanced speed and quality
    #[default]
    Standard,

    /// Extensive tuning for maximum accuracy
    Thorough,

    /// No tuning - use provided hyperparameters as-is
    None,
}

impl TuningLevel {
    /// Convert to the equivalent `TunerPreset` for use with `AutoTuner`.
    ///
    /// This provides a sensible mapping between `TuningLevel` (used in `AutoBuilder`)
    /// and `TunerPreset` (used in `AutoTuner`/`TunerConfig`).
    ///
    /// # Mapping
    ///
    /// - `Quick` → `TunerPreset::Quick`
    /// - `Standard` → `TunerPreset::Balanced`
    /// - `Thorough` → `TunerPreset::Thorough`
    /// - `None` → `TunerPreset::Quick` (minimal tuning as fallback)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::{TuningLevel, TunerPreset};
    ///
    /// let level = TuningLevel::Standard;
    /// let tuner_preset = level.to_tuner_preset();
    /// assert_eq!(tuner_preset, TunerPreset::Balanced);
    /// ```
    pub fn to_tuner_preset(self) -> crate::tuner::TunerPreset {
        use crate::tuner::TunerPreset;
        match self {
            TuningLevel::Quick => TunerPreset::Quick,
            TuningLevel::Standard => TunerPreset::Balanced,
            TuningLevel::Thorough => TunerPreset::Thorough,
            TuningLevel::None => TunerPreset::Quick, // Fallback to minimal tuning
        }
    }

    /// Convert to the equivalent `TreeTunerPreset` for use with `TreeTunerConfig`.
    ///
    /// This provides a sensible mapping between `TuningLevel` (used in `AutoBuilder`)
    /// and `TreeTunerPreset` (used in `TreeTunerConfig`).
    ///
    /// # Mapping
    ///
    /// - `Quick` → `TreeTunerPreset::Quick`
    /// - `Standard` → `TreeTunerPreset::Standard`
    /// - `Thorough` → `TreeTunerPreset::Thorough`
    /// - `None` → `TreeTunerPreset::Quick` (minimal tuning as fallback)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::{TuningLevel, TreeTunerPreset};
    ///
    /// let level = TuningLevel::Thorough;
    /// let tree_preset = level.to_tree_tuner_preset();
    /// assert_eq!(tree_preset, TreeTunerPreset::Thorough);
    /// ```
    pub fn to_tree_tuner_preset(self) -> TreeTunerPreset {
        match self {
            TuningLevel::Quick => TreeTunerPreset::Quick,
            TuningLevel::Standard => TreeTunerPreset::Standard,
            TuningLevel::Thorough => TreeTunerPreset::Thorough,
            TuningLevel::None => TreeTunerPreset::Quick, // Fallback to minimal tuning
        }
    }
}

/// Ensemble strategy for AutoBuilder (PureTree only)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoEnsembleMethod {
    /// Simple average of selected models
    SimpleAverage,
    /// Ridge stacking of selected models (default)
    RidgeStacking,
}

/// Ensemble configuration for AutoBuilder
#[derive(Debug, Clone)]
pub struct AutoEnsembleConfig {
    pub method: AutoEnsembleMethod,
    pub multi_seed: MultiSeedConfig,
    pub selection: SelectionConfig,
    pub stacking: StackingConfig,
}

impl Default for AutoEnsembleConfig {
    fn default() -> Self {
        Self {
            method: AutoEnsembleMethod::RidgeStacking,
            multi_seed: MultiSeedConfig::default(),
            selection: SelectionConfig::default(),
            stacking: StackingConfig::default(),
        }
    }
}

impl AutoEnsembleConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_method(mut self, method: AutoEnsembleMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_multi_seed_config(mut self, config: MultiSeedConfig) -> Self {
        self.multi_seed = config;
        self
    }

    pub fn with_selection_config(mut self, config: SelectionConfig) -> Self {
        self.selection = config;
        self
    }

    pub fn with_stacking_config(mut self, config: StackingConfig) -> Self {
        self.stacking = config;
        self
    }
}

/// AutoBuilder configuration
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

    /// Maximum time budget for training (None = no limit)
    /// AutoBuilder will adapt tuning intensity to fit within this budget
    pub time_budget: Option<Duration>,

    /// Progress callback for tracking training phases
    pub progress_callback: Arc<dyn ProgressCallback>,

    /// Configuration for extracting features for linear models (LinearThenTree)
    pub linear_feature_config: LinearFeatureConfig,

    /// Custom UniversalConfig to use (bypasses tuning if provided with TuningLevel::None)
    pub custom_config: Option<crate::model::UniversalConfig>,

    /// Optional ensemble configuration (PureTree only)
    pub ensemble: Option<AutoEnsembleConfig>,

    /// Custom tree tuner configuration (overrides tuning_level for tree-based modes)
    pub tree_tuner_config: Option<TreeTunerConfig>,

    /// Backend type for computation (Auto, Cuda, Wgpu, Scalar, etc.)
    pub backend_type: crate::backend::BackendType,
}

impl Clone for AutoConfig {
    fn clone(&self) -> Self {
        Self {
            tuning_level: self.tuning_level,
            val_ratio: self.val_ratio,
            auto_features: self.auto_features,
            auto_preprocessing: self.auto_preprocessing,
            auto_mode: self.auto_mode,
            force_mode: self.force_mode,
            max_generated_features: self.max_generated_features,
            seed: self.seed,
            verbose: self.verbose,
            time_budget: self.time_budget,
            progress_callback: Arc::clone(&self.progress_callback),
            linear_feature_config: self.linear_feature_config.clone(),
            custom_config: self.custom_config.clone(),
            ensemble: self.ensemble.clone(),
            tree_tuner_config: self.tree_tuner_config.clone(),
            backend_type: self.backend_type,
        }
    }
}

impl std::fmt::Debug for AutoConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoConfig")
            .field("tuning_level", &self.tuning_level)
            .field("val_ratio", &self.val_ratio)
            .field("auto_features", &self.auto_features)
            .field("auto_preprocessing", &self.auto_preprocessing)
            .field("auto_mode", &self.auto_mode)
            .field("force_mode", &self.force_mode)
            .field("max_generated_features", &self.max_generated_features)
            .field("seed", &self.seed)
            .field("verbose", &self.verbose)
            .field("time_budget", &self.time_budget)
            .field("progress_callback", &"<callback>")
            .field("linear_feature_config", &self.linear_feature_config)
            .field("ensemble", &self.ensemble)
            .field("tree_tuner_config", &self.tree_tuner_config)
            .field("backend_type", &self.backend_type)
            .finish()
    }
}

impl Default for AutoConfig {
    fn default() -> Self {
        Self {
            tuning_level: TuningLevel::Standard,
            val_ratio: auto_defaults::DEFAULT_VALIDATION_RATIO,
            auto_features: true,
            auto_preprocessing: true,
            auto_mode: true,
            force_mode: None,
            max_generated_features: auto_defaults::AUTO_FEATURES_DEFAULT_COUNT,
            seed: seeds_defaults::DEFAULT_SEED,
            verbose: false,
            time_budget: None,
            progress_callback: Arc::new(QuietProgress),
            linear_feature_config: LinearFeatureConfig::default(),
            custom_config: None,
            ensemble: None,
            tree_tuner_config: None,
            backend_type: crate::backend::BackendType::Auto,
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

    /// Set validation split ratio for **random** train/validation split.
    ///
    /// **For cross-sectional (i.i.d.) data only** where rows are independent.
    /// For time-series or panel data, use `AutoBuilder::with_presplit_validation` instead.
    pub fn with_random_validation_split(mut self, ratio: f32) -> Self {
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

    /// Enable ensemble training with default settings (PureTree only)
    pub fn with_ensemble(mut self) -> Self {
        self.ensemble = Some(AutoEnsembleConfig::default());
        self
    }

    /// Enable ensemble training with a specific method (PureTree only)
    pub fn with_ensemble_method(mut self, method: AutoEnsembleMethod) -> Self {
        self.ensemble = Some(AutoEnsembleConfig::default().with_method(method));
        self
    }

    /// Set full ensemble configuration (PureTree only)
    pub fn with_ensemble_config(mut self, config: AutoEnsembleConfig) -> Self {
        self.ensemble = Some(config);
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

    /// Set backend type for computation
    ///
    /// Use BackendType::Auto (default) for automatic detection: CUDA > WGPU > Scalar
    pub fn with_backend(mut self, backend_type: crate::backend::BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }

    /// Set time budget for training
    ///
    /// AutoBuilder will adapt tuning intensity to fit within this budget.
    /// For example, if 60 seconds is allocated and profiling takes 10 seconds,
    /// the remaining 50 seconds will be split between tuning and training.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::time::Duration;
    /// let config = AutoConfig::new()
    ///     .with_time_budget(Duration::from_secs(60)); // 1 minute max
    /// ```
    pub fn with_time_budget(mut self, budget: Duration) -> Self {
        self.time_budget = Some(budget);
        self
    }

    /// Set progress callback for tracking training phases
    ///
    /// Use this to receive updates as training progresses through each phase.
    /// Useful for long-running training to show progress to users.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::model::ConsoleProgress;
    /// use std::sync::Arc;
    ///
    /// let config = AutoConfig::new()
    ///     .with_progress_callback(Arc::new(ConsoleProgress::detailed()));
    /// ```
    pub fn with_progress_callback(mut self, callback: Arc<dyn ProgressCallback>) -> Self {
        self.progress_callback = callback;
        self
    }

    /// Set configuration for linear model features (LinearThenTree mode)
    ///
    /// Use this to control which features are used for the linear component
    /// in LinearThenTree mode. This can improve accuracy by excluding features
    /// that are not useful for linear regression (e.g., high-cardinality categoricals).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::dataset::feature_extractor::LinearFeatureConfig;
    ///
    /// let config = AutoConfig::new()
    ///     .with_linear_feature_config(LinearFeatureConfig::default()
    ///         .with_exclude_patterns(&vec!["id_".to_string(), "rank_".to_string()]));
    /// ```
    pub fn with_linear_feature_config(mut self, config: LinearFeatureConfig) -> Self {
        self.linear_feature_config = config;
        self
    }

    /// Set custom UniversalConfig (overrides tuning)
    ///
    /// Use this to bypass tuning and provide your own hyperparameters.
    /// The custom config will be used regardless of tuning level.
    pub fn with_custom_config(mut self, config: crate::model::UniversalConfig) -> Self {
        self.custom_config = Some(config);
        self
    }

    /// Set custom tree tuner configuration (overrides tuning_level for tree-based modes)
    ///
    /// Use this to provide custom hyperparameter search configuration for PureTree
    /// and RandomForest modes. This allows fine-grained control over the number of
    /// samples, iterations, depth ranges, and other tuning parameters.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::model::{AutoConfig, TreeTunerConfig};
    /// use treeboost::defaults::auto as auto_defaults;
    ///
    /// let custom_tuner = TreeTunerConfig {
    ///     max_depth_range: (3, 8),
    ///     learning_rate_range: auto_defaults::STANDARD_LR_RANGE,
    ///     n_samples: 50,
    ///     n_iterations: 1,
    ///     max_rounds: 200,
    ///     early_stopping_rounds: 10,
    ///     validation_ratio: 0.2,
    ///     improvement_threshold: 0.001,
    ///     min_f1_score: 0.85,
    /// };
    ///
    /// let config = AutoConfig::new()
    ///     .with_tree_tuner_config(custom_tuner);
    /// ```
    pub fn with_tree_tuner_config(mut self, config: TreeTunerConfig) -> Self {
        self.tree_tuner_config = Some(config);
        self
    }
}

/// Tuning result for tree-based models (PureTree, RandomForest)
#[derive(Debug, Clone)]
pub struct TreeTuningResult {
    /// Number of tuning trials evaluated
    pub num_trials: usize,
    /// Best validation metric achieved
    pub best_metric: f32,
    /// Best hyperparameters found
    pub best_params: std::collections::HashMap<String, f32>,
}

/// Configuration for tree-based model tuning (PureTree, RandomForest)
#[derive(Debug, Clone)]
pub struct TreeTunerConfig {
    /// Maximum tree depth range (min, max)
    pub max_depth_range: (usize, usize),
    /// Learning rate range (min, max) - log scale
    pub learning_rate_range: (f32, f32),
    /// Number of Latin Hypercube samples per iteration
    pub n_samples: usize,
    /// Number of zoom iterations (iterative refinement)
    pub n_iterations: usize,
    /// Maximum boosting rounds per trial
    pub max_rounds: usize,
    /// Early stopping rounds
    pub early_stopping_rounds: usize,
    /// Validation ratio for early stopping
    pub validation_ratio: f32,
    /// Improvement threshold for stopping iterations (e.g., 0.001 = 0.1%)
    pub improvement_threshold: f32,
    /// Minimum F1 score required before stopping
    pub min_f1_score: f32,
    /// Optional output directory for CSV logging (None = no logging)
    pub output_dir: Option<std::path::PathBuf>,
    /// Metric to optimize (ValidationLoss, F1Score, RocAuc, RankIc)
    ///
    /// - `ValidationLoss`: Use val_metric (lower is better) - default
    /// - `F1Score`: Use F1 score (higher is better) - classification
    /// - `RocAuc`: Use ROC-AUC (higher is better) - binary classification
    /// - `RankIc`: Use Rank IC (higher is better) - regression
    pub optimization_metric: OptimizationMetric,
    /// Task type (Regression, BinaryClassification, MultiClassClassification)
    ///
    /// When set to Some, overrides the profile-detected task type.
    /// Use this to explicitly set regression for stock/quant data where
    /// Rank IC should be computed.
    pub task_type: Option<TunerTaskType>,
}

/// Tuning intensity preset for tree-specific hyperparameter optimization.
///
/// ## When to Use TreeTunerPreset
///
/// Use `TreeTunerPreset` when working with **`TreeTunerConfig`**.
/// This enum controls tree-specific tuning intensity for PureTree and RandomForest modes.
///
/// ## Relationship to Other Preset Enums
///
/// TreeBoost has several tuning-related preset enums for different contexts:
///
/// | Enum | Context | Purpose |
/// |------|---------|---------|
/// | **`TuningLevel`** | `AutoBuilder`, `AutoConfig` | High-level: AutoML tuning intensity |
/// | **`TunerPreset`** | `AutoTuner`, `TunerConfig` | Mid-level: Manual tuning intensity |
/// | **`TreeTunerPreset`** | `TreeTunerConfig` | Low-level: Tree-only tuning intensity |
///
/// **Mapping between presets:**
/// - `TreeTunerPreset::Quick` ≈ `TuningLevel::Quick` ≈ `TunerPreset::Quick`
/// - `TreeTunerPreset::Standard` ≈ `TuningLevel::Standard` ≈ `TunerPreset::Balanced`
/// - `TreeTunerPreset::Thorough` ≈ `TuningLevel::Thorough` ≈ `TunerPreset::Thorough`
///
/// ## Variants
///
/// ### `Quick`
/// Fast tree tuning with shallow depth range.
/// - **Best for**: Prototyping, debugging, quick experiments
/// - **Depth range**: [3-6]
/// - **Samples**: 30 Latin Hypercube samples
/// - **Iterations**: 1 (no zoom)
/// - **Time**: Minutes
/// - **Quality**: Good starting point, not fully optimized
///
/// ### `Standard` (Default)
/// Balanced tree tuning with moderate depth range.
/// - **Best for**: Most production use cases
/// - **Depth range**: [3-8]
/// - **Samples**: 100 Latin Hypercube samples
/// - **Iterations**: 3 (with zoom refinement)
/// - **Time**: Tens of minutes
/// - **Quality**: Well-tuned, production-ready
///
/// ### `Thorough`
/// Extensive tree tuning with deep depth range.
/// - **Best for**: Maximum accuracy, competitions, research
/// - **Depth range**: [3-10]
/// - **Samples**: 150 Latin Hypercube samples
/// - **Iterations**: 15 (aggressive zoom)
/// - **Time**: Hours
/// - **Quality**: Near-optimal hyperparameters
///
/// ## Examples
///
/// ```ignore
/// use treeboost::{TreeTunerConfig, TreeTunerPreset};
///
/// // Quick tree tuning
/// let config = TreeTunerConfig::with_preset(TreeTunerPreset::Quick);
///
/// // Standard tree tuning (default)
/// let config = TreeTunerConfig::with_preset(TreeTunerPreset::Standard);
///
/// // Thorough tree tuning
/// let config = TreeTunerConfig::with_preset(TreeTunerPreset::Thorough);
/// ```
///
/// ## See Also
///
/// - [`TuningLevel`] - For high-level AutoML tuning with `AutoBuilder`
/// - [`TunerPreset`] - For manual tuning with `AutoTuner`
/// - [`TreeTunerConfig::with_preset`] - Apply a preset to tree tuner configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeTunerPreset {
    /// Fast tree tuning with shallow depth range [3-6]
    Quick,
    /// Balanced tree tuning with moderate depth range [3-8]
    Standard,
    /// Extensive tree tuning with deep depth range [3-10]
    Thorough,
}

impl TreeTunerConfig {
    fn preset_quick() -> Self {
        Self {
            max_depth_range: auto_defaults::QUICK_DEPTH_RANGE,
            learning_rate_range: auto_defaults::QUICK_LR_RANGE,
            n_samples: 30, // Default for Quick preset
            n_iterations: 1,
            max_rounds: 100,
            early_stopping_rounds: 10,
            validation_ratio: auto_defaults::DEFAULT_VALIDATION_RATIO,
            improvement_threshold: 0.001,
            min_f1_score: 0.80,
            output_dir: None,
            optimization_metric: OptimizationMetric::ValidationLoss,
            task_type: None, // Use profile detection
        }
    }

    fn preset_standard() -> Self {
        Self {
            max_depth_range: auto_defaults::STANDARD_DEPTH_RANGE,
            learning_rate_range: auto_defaults::STANDARD_LR_RANGE,
            n_samples: 100,
            n_iterations: 3,
            max_rounds: 200,
            early_stopping_rounds: 10,
            validation_ratio: auto_defaults::DEFAULT_VALIDATION_RATIO,
            improvement_threshold: 0.001,
            min_f1_score: 0.85,
            output_dir: None,
            optimization_metric: OptimizationMetric::ValidationLoss,
            task_type: None, // Use profile detection
        }
    }

    fn preset_thorough() -> Self {
        Self {
            max_depth_range: auto_defaults::THOROUGH_DEPTH_RANGE,
            learning_rate_range: auto_defaults::STANDARD_LR_RANGE,
            n_samples: 150,
            n_iterations: 15,
            max_rounds: 200,
            early_stopping_rounds: 10,
            validation_ratio: auto_defaults::DEFAULT_VALIDATION_RATIO,
            improvement_threshold: 0.001,
            min_f1_score: 0.85,
            output_dir: None,
            optimization_metric: OptimizationMetric::ValidationLoss,
            task_type: None, // Use profile detection
        }
    }

    /// Set the optimization metric
    pub fn with_optimization_metric(mut self, metric: OptimizationMetric) -> Self {
        self.optimization_metric = metric;
        self
    }

    /// Set the task type explicitly (overrides profile detection)
    pub fn with_task_type(mut self, task_type: TunerTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    /// Apply a preset configuration.
    pub fn with_preset(preset: TreeTunerPreset) -> Self {
        match preset {
            TreeTunerPreset::Quick => Self::preset_quick(),
            TreeTunerPreset::Standard => Self::preset_standard(),
            TreeTunerPreset::Thorough => Self::preset_thorough(),
        }
    }
}

/// Build result containing the trained model and metadata
#[derive(Debug)]
pub struct BuildResult {
    /// The trained model (UniversalModel handles all modes and ensembles)
    pub model: UniversalModel,

    /// The boosting mode used
    pub mode: BoostingMode,

    /// Target column name used during training
    pub target_column: String,

    /// Mode selection confidence (if auto mode was used)
    pub mode_confidence: Option<Confidence>,

    /// Preprocessing plan that was applied
    pub preprocessing_plan: Option<PreprocessingPlan>,

    /// Feature engineering plan that was applied
    pub feature_plan: Option<FeaturePlan>,

    /// LTT tuning result (if LTT mode was used)
    pub ltt_tuning: Option<LttTuningResult>,

    /// Tree tuning result (if PureTree/RandomForest mode was used)
    pub tree_tuning: Option<TreeTuningResult>,

    /// Column profile from analysis
    pub column_profile: Option<DataFrameProfile>,

    /// Dataset analysis result
    pub analysis: Option<DatasetAnalysis>,

    /// Fitted pipeline state for inference (CRITICAL for prediction!)
    pub pipeline_state: Option<crate::dataset::PipelineState>,

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
