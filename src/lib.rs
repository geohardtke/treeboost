//! TreeBoost: Universal Tabular Learning Engine
//!
//! Combines linear models, gradient boosted trees, and random forests in a
//! single unified interface. Pick the right tool for your data—or let the
//! AutoTuner figure it out.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      UniversalModel                         │
//! ├──────────────┬──────────────────────┬───────────────────────┤
//! │   PureTree   │   LinearThenTree     │    RandomForest       │
//! │   (GBDT)     │   (Hybrid)           │    (Bagging)          │
//! └──────────────┴──────────────────────┴───────────────────────┘
//! ```
//!
//! # Quick Start (AutoML - Recommended)
//!
//! ```ignore
//! use polars::prelude::*;
//! use treeboost::auto_train;
//!
//! // Load data
//! let df = CsvReadOptions::default()
//!     .try_into_reader_with_file_path(Some("housing.csv".into()))?
//!     .finish()?;
//!
//! // One-line training - analyzes data, selects mode, tunes params
//! let model = auto_train(&df, "price")?;
//!
//! // Predict
//! let predictions = model.predict(&test_df)?;
//!
//! // See what AutoML did
//! println!("{}", model.summary());
//! ```
//!
//! # Manual Configuration (Advanced)
//!
//! ```ignore
//! use treeboost::{UniversalConfig, UniversalModel, BoostingMode};
//! use treeboost::dataset::DatasetLoader;
//! use treeboost::loss::MseLoss;
//!
//! let loader = DatasetLoader::new(255);
//! let dataset = loader.load_parquet("data.parquet", "target", None)?;
//!
//! let config = UniversalConfig::new()
//!     .with_mode(BoostingMode::LinearThenTree)  // Hybrid mode
//!     .with_num_rounds(100)
//!     .with_linear_rounds(10);
//!
//! let model = UniversalModel::train(&dataset, config, &MseLoss)?;
//! let predictions = model.predict(&dataset);
//! ```
//!
//! # Boosting Modes
//!
//! | Mode | Best For |
//! |------|----------|
//! | [`BoostingMode::PureTree`] | General tabular, categorical features |
//! | [`BoostingMode::LinearThenTree`] | Time-series, trending data, extrapolation |
//! | [`BoostingMode::RandomForest`] | Noisy data, variance reduction |
//!
//! # Weak Learners
//!
//! - [`LinearBooster`]: Ridge/LASSO/ElasticNet via Coordinate Descent
//! - [`LinearTreeBooster`]: Decision trees with linear regression in leaves
//! - [`TreeBooster`]: Standard histogram-based GBDT trees
//!
//! # Preprocessing
//!
//! The [`preprocessing`] module provides transforms that serialize with your model:
//!
//! - Scalers: [`StandardScaler`], [`MinMaxScaler`], [`RobustScaler`]
//! - Encoders: [`FrequencyEncoder`], [`LabelEncoder`], [`OneHotEncoder`]
//! - Imputers: [`SimpleImputer`], [`IndicatorImputer`]
//! - Time-series: [`LagGenerator`], [`RollingGenerator`], [`EwmaGenerator`]
//!
//! # Additional Features
//!
//! - **Histogram-based training**: u8 bins for memory efficiency
//! - **Shannon Entropy regularized splits**: Drift-resilient objective
//! - **Pseudo-Huber loss**: Robust to outliers
//! - **Split Conformal Prediction**: Distribution-free prediction intervals
//! - **Zero-copy serialization**: Fast model loading via rkyv
//! - **GPU acceleration**: WGPU (all GPUs), CUDA (NVIDIA)
//!
//! # Validation Modes
//!
//! TreeBoost supports two validation strategies to accommodate different data structures:
//!
//! ## 1. Random Validation Split (Cross-Sectional Data)
//!
//! **When to use:** Cross-sectional tabular data where rows are independent (i.i.d.).
//!
//! The library performs internal random splitting into train/validation sets.
//! Simple and appropriate for most classification and regression tasks.
//!
//! **API Entry Points:**
//!
//! ```ignore
//! // High-level AutoML
//! let model = AutoBuilder::new()
//!     .with_random_validation_split(0.2)  // 20% for validation
//!     .fit(&df, "target")?;
//!
//! // Manual tuning
//! let tuner_config = TunerConfig::new()
//!     .with_eval_strategy(EvalStrategy::holdout(0.2));
//! let mut tuner = AutoTuner::new(base_config).with_config(tuner_config);
//! let (best_config, history) = tuner.tune(&dataset)?;
//! ```
//!
//! ## 2. Pre-Split Validation (Time-Series / Panel Data)
//!
//! **When to use:** Data with temporal or group structure where random splits cause leakage:
//! - **Time-series:** Train on past, validate on future (date-based boundary)
//! - **Panel data:** Train on some groups, validate on held-out groups (no group leakage)
//! - **Hierarchical data:** Prevent information leakage across hierarchy levels
//!
//! You manually split the data BEFORE training, and the library uses your custom split.
//!
//! **API Entry Points:**
//!
//! ```ignore
//! // High-level AutoML (with DataFrame)
//! let train_df = df.filter(col("date").lt(lit("2024-01-01")))?;
//! let val_df = df.filter(col("date").gte(lit("2024-01-01")))?;
//!
//! let model = AutoBuilder::new()
//!     .with_presplit_validation(val_df)  // Custom validation DataFrame
//!     .fit(&train_df, "target")?;
//!
//! // Manual tuning (with BinnedDataset)
//! let train_dataset = loader.load_dataframe(train_df, "target", None)?;
//! let val_dataset = loader.load_dataframe(val_df, "target", None)?;
//!
//! let mut tuner = AutoTuner::new(base_config);
//! let (best_config, history) = tuner.tune_with_validation(&train_dataset, &val_dataset)?;
//! ```
//!
//! ## Choosing the Right Mode
//!
//! | Data Type | Validation Mode | Rationale |
//! |-----------|-----------------|-----------|
//! | Cross-sectional tabular | Random split | Rows are independent (i.i.d.) |
//! | Time-series forecasting | Pre-split (date) | Temporal boundary prevents leakage |
//! | Panel data (stocks × dates) | Pre-split (group/date) | Group and temporal structure |
//! | Hierarchical data | Pre-split (hierarchy) | Cross-level leakage prevention |
//! | Cross-validation with custom folds | Pre-split (fold-specific) | Full control over splits |
//!
//! **Critical Rule:** For time-series or panel data, ALWAYS split BEFORE encoding.
//! Encoding (especially target encoding) must be fit on training data only to prevent leakage.
//!
//! See also:
//! - [`AutoBuilder::with_random_validation_split`] - Random split API
//! - [`AutoBuilder::with_presplit_validation`] - Pre-split API
//! - [`AutoTuner::tune`] - Tuner with random split
//! - [`AutoTuner::tune_with_validation`] - Tuner with pre-split validation
//!
//! # Production Features
//!
//! TreeBoost includes production-ready capabilities for real-world deployment scenarios.
//! These features are fully implemented and battle-tested - not experimental.
//!
//! ## Incremental Learning & Model Updates
//!
//! Update models with new data without full retraining using the TRB (TreeBoost) format:
//!
//! ```ignore
//! use treeboost::{UniversalModel, AutoModel};
//! use treeboost::loss::MseLoss;
//!
//! // Initial training with AutoModel
//! let auto = AutoModel::train(&df, "target")?;
//! auto.inner().save_trb("model.trb", "Initial training")?;
//!
//! // Later: Load and incrementally update
//! let mut model = UniversalModel::load_trb("model.trb")?;
//! let report = model.update(&new_dataset, &MseLoss, 10)?;
//! println!("Added {} trees", report.trees_added);
//!
//! // Append update to file (O(1), no rewrite)
//! model.save_trb_update("model.trb", new_dataset.num_rows(), "Update batch")?;
//! ```
//!
//! **Key capabilities:**
//! - O(1) append updates (base model never rewritten)
//! - CRC32 checksums per segment
//! - Crash recovery (truncated writes auto-detected)
//! - Memory-mapped I/O with `mmap` feature
//!
//! See [`UniversalModel::update`], [`UniversalModel::load_trb`], [`UniversalModel::save_trb`]
//!
//! ## Drift Detection & Monitoring
//!
//! Monitor distribution shifts before updating models:
//!
//! ```ignore
//! use treeboost::monitoring::{IncrementalDriftDetector, DriftRecommendation};
//!
//! // Create detector from training distribution
//! let detector = IncrementalDriftDetector::from_dataset(&train_data)
//!     .with_thresholds(0.1, 0.25);  // warning, critical
//!
//! // Check new data before updating
//! let result = detector.check_update(&new_data);
//! match result.recommendation {
//!     DriftRecommendation::ProceedNormally => {
//!         model.update(&new_data, &loss, 10)?;
//!     }
//!     DriftRecommendation::RetrainRecommended => {
//!         // Critical drift detected - consider full retrain
//!         eprintln!("Critical drift: {}", result);
//!     }
//!     _ => { /* Handle other cases */ }
//! }
//! ```
//!
//! **Metrics:** Population Stability Index (PSI), KL divergence, KS test
//!
//! See [`monitoring::IncrementalDriftDetector`], [`monitoring::DriftHistory`]
//!
//! ## Ensemble Methods
//!
//! Multi-seed training with stacked blending for reduced variance:
//!
//! ```ignore
//! use treeboost::{UniversalConfig, StackingStrategy};
//!
//! let config = UniversalConfig::new()
//!     .with_ensemble_seeds(vec![1, 2, 3, 4, 5])  // Train 5 models
//!     .with_stacking_strategy(StackingStrategy::Ridge {
//!         alpha: 0.01,
//!         rank_transform: false,
//!         fit_intercept: true,
//!         min_weight: 0.01,
//!     });
//! ```
//!
//! See [`UniversalConfig::with_ensemble_seeds`], [`StackingStrategy`], [`ensemble::StackedEnsemble`]
//!
//! ## Monotonic & Interaction Constraints
//!
//! Enforce domain knowledge in tree splits:
//!
//! ```ignore
//! use treeboost::{TreeConfig, MonotonicConstraint};
//! use treeboost::tree::InteractionConstraints;
//!
//! // Monotonic constraints (age → risk increases, income → risk decreases)
//! let tree_config = TreeConfig::default()
//!     .with_monotonic_constraints(vec![
//!         MonotonicConstraint::Increasing,  // feature 0: age
//!         MonotonicConstraint::Decreasing,  // feature 1: income
//!         MonotonicConstraint::None,        // feature 2: no constraint
//!     ]);
//!
//! // Interaction constraints (features can only split with allowed partners)
//! let constraints = InteractionConstraints::from_groups(vec![
//!     vec![0, 1, 2],  // Group 1: features 0, 1, 2 can interact
//!     vec![3, 4],     // Group 2: features 3, 4 can interact
//! ]);
//! let tree_config = tree_config.with_interaction_constraints(constraints);
//! ```
//!
//! See [`MonotonicConstraint`], [`InteractionConstraints`]
//!
//! ## High-Cardinality Categorical Encoding
//!
//! Ordered target encoding with Count-Min Sketch for rare category filtering:
//!
//! ```ignore
//! use treeboost::encoding::{OrderedTargetEncoder, CategoryFilter};
//!
//! // Target encoding (prevents leakage via sequential encoding)
//! let mut encoder = OrderedTargetEncoder::new(10.0);  // smoothing=10
//! for (category, target) in training_pairs {
//!     let encoded = encoder.encode_and_update(&category, target);
//! }
//!
//! // Rare category filtering (Count-Min Sketch - no full hash map)
//! let filter = CategoryFilter::default_for_high_cardinality();
//! filter.count_batch(&categories);  // First pass
//! filter.finalize();
//! let filtered = filter.filter_batch(&categories);  // "rare" → "unknown"
//! ```
//!
//! See [`OrderedTargetEncoder`], [`CategoryFilter`]
//!
//! ## Time-Series & Cross-Sectional Feature Engineering
//!
//! Automatic feature generation for panel data and time series:
//!
//! ```ignore
//! use treeboost::preprocessing::{LagGenerator, RollingGenerator, EwmaGenerator};
//! use treeboost::features::{PolynomialGenerator, RatioGenerator};
//!
//! // Time-series features (lag, rolling, EWMA)
//! let lag_gen = LagGenerator::new(vec![1, 2, 7]);  // 1-day, 2-day, 7-day lags
//! let rolling = RollingGenerator::new(7, vec![RollingStat::Mean, RollingStat::Std]);
//! let ewma = EwmaGenerator::new(vec![0.1, 0.3]);   // α=0.1, α=0.3
//!
//! // Cross-sectional features (polynomial, ratio, interaction)
//! let poly_gen = PolynomialGenerator::new(2);       // x², x³
//! let ratio_gen = RatioGenerator::new();            // x₁/x₂, x₂/x₁
//! ```
//!
//! See [`preprocessing::LagGenerator`], [`preprocessing::RollingGenerator`],
//! [`features::PolynomialGenerator`], [`features::RatioGenerator`]
//!
//! ## Outlier Detection & Robust Preprocessing
//!
//! IQR and Z-score based outlier detection:
//!
//! ```ignore
//! use treeboost::preprocessing::{RobustScaler, OutlierDetector};
//!
//! // RobustScaler: Uses median/IQR instead of mean/std (robust to outliers)
//! let mut scaler = RobustScaler::new();
//! scaler.fit(&features, num_features)?;
//!
//! // OutlierDetector: Identify outliers via IQR or Z-score
//! let detector = OutlierDetector::iqr(1.5);  // IQR with k=1.5
//! let mask = detector.detect(&features);     // true = outlier
//! ```
//!
//! See [`RobustScaler`], [`OutlierDetector`]
//!
//! ## Incremental Preprocessing
//!
//! Welford's algorithm for online mean/variance updates:
//!
//! ```ignore
//! use treeboost::preprocessing::StandardScaler;
//!
//! // EMA-based scaler (adapts to drift over time)
//! let mut scaler = StandardScaler::with_forget_factor(0.1);  // α=0.1
//! scaler.partial_fit(&batch1, num_features)?;  // Update with batch1
//! scaler.partial_fit(&batch2, num_features)?;  // Blend with batch2 (90% old, 10% new)
//! ```
//!
//! See [`StandardScaler::with_forget_factor`], [`StandardScaler::partial_fit`]
//!
//! ## Split Conformal Prediction
//!
//! Distribution-free prediction intervals (valid for any distribution):
//!
//! ```ignore
//! use treeboost::GBDTConfig;
//!
//! let config = GBDTConfig::default()
//!     .with_conformal(0.2, 0.9)?;  // 20% calibration, 90% coverage
//!
//! let model = GBDTModel::train(&dataset, &config)?;
//! let quantile = model.conformal_quantile();  // ~1.28 for 90% coverage
//! ```
//!
//! See [`GBDTConfig::with_conformal`], [`GBDTModel::conformal_quantile`]
//!
//! ---
//!
//! **For detailed examples and API documentation, see:**
//! - [API Reference](https://docs.rs/treeboost)
//! - [`docs/API.md`](https://github.com/ml-rust/TreeBoost/blob/main/docs/API.md) - Complete API guide
//!

pub mod analysis;
pub mod backend;
pub mod booster;
pub mod dataset;
pub mod defaults;
pub mod encoding;
pub mod ensemble;
pub mod features;
pub mod histogram;
pub mod inference;
pub mod learner;
pub mod loss;
pub mod model;
pub mod monitoring;
pub mod prelude;
pub mod preprocessing;
pub mod serialize;
pub mod tree;
pub mod tuner;
pub mod utils;

#[cfg(feature = "python")]
mod python;

// Re-exports for convenience
pub use backend::{
    BackendConfig, BackendPreset, BackendSelector, BackendType, GpuMode, HistogramBackend,
};
pub use booster::{GBDTConfig, GBDTModel, GbdtPreset};
pub use dataset::{BinnedDataset, FeatureInfo, FeatureType, QuantileBinner};
pub use ensemble::{
    EnsembleBuilder, MultiSeedConfig, SelectionConfig as EnsembleSelectionConfig, StackedEnsemble,
    StackingConfig,
};
pub use features::{
    FeatureGenerationConfig, FeatureGenerator, FeatureSelector, PolynomialGenerator,
    RatioGenerator, SelectionConfig, SmartFeatureConfig, SmartFeaturePreset,
};
pub use histogram::HistogramBuilder;
pub use inference::Prediction;
pub use learner::{
    Booster, LeafLinearModel, LinearBooster, LinearConfig, LinearPreset, LinearTreeBooster,
    LinearTreeConfig, TreeBooster, TreeConfig, TreePreset, WeakLearner,
};
pub use loss::{
    sigmoid, softmax, BinaryLogLoss, LossFunction, MseLoss, MultiClassLogLoss, PseudoHuberLoss,
};
pub use model::{
    AutoBuilder, AutoConfig, AutoEnsembleConfig, AutoEnsembleMethod, AutoModel,
    AutoModelUpdateReport, BoostingMode, BuildPhaseTimes, BuildResult, ConsoleProgress,
    IncrementalUpdateReport, ModeSelection, ProgressCallback, ProgressUpdate, QuietProgress,
    StackingStrategy, TrainingPhase, TreeTunerPreset, TuningLevel, UniversalConfig, UniversalModel,
    UniversalPreset,
};
pub use monitoring::{AlertLevel, CVHoldoutTracker, ShiftDetector, ShiftResult};

// Analysis module exports
pub use analysis::{
    compute_correlation, compute_r2, compute_variance, AnalysisConfig, AnalysisPreset,
    AnalysisReport, Confidence, DatasetAnalysis, Recommendation,
};
pub use preprocessing::{
    EncodingMap, FrequencyEncoder, ImputeStrategy, IndicatorImputer, LabelEncoder, MinMaxScaler,
    OneHotEncoder, OrderedTargetEncoder, PipelineBuilder, Preprocessor, RobustScaler, Scaler,
    SimpleImputer, SmartPreprocessConfig, SmartPreprocessPreset, StandardScaler, UnknownStrategy,
    YeoJohnsonTransform,
};
pub use tree::{InteractionConstraints, MonotonicConstraint};
pub use tuner::{
    AutoTuner, EvalStrategy, GridStrategy, ModelFormat, ParameterSpace, SearchHistory, SpacePreset,
    TunerConfig, TunerPreset,
};

/// Library error type
#[derive(Debug, thiserror::Error)]
pub enum TreeBoostError {
    #[error("Data error: {0}")]
    Data(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, TreeBoostError>;

//=============================================================================
// Convenience Entry Points (Level 0 API)
//=============================================================================

/// Train a model with automatic configuration (the simplest API)
///
/// This is the recommended entry point for most users. It automatically:
/// - Profiles the dataset to understand column types and characteristics
/// - Applies smart preprocessing based on data patterns
/// - Generates useful features (polynomial, ratio, interactions)
/// - Analyzes data to recommend the optimal boosting mode
/// - Tunes hyperparameters for the selected mode
/// - Trains the final model
///
/// # Arguments
///
/// * `df` - Input DataFrame with features and target
/// * `target_col` - Name of the target column
///
/// # Returns
///
/// A trained [`AutoModel`] ready for prediction, or an error if training fails
///
/// # Example
///
/// ```ignore
/// use polars::prelude::*;
/// use treeboost::auto_train;
///
/// // Load your data
/// let df = LazyCsvReader::new("data.csv")
///     .finish()?
///     .collect()?;
///
/// // Train with defaults
/// let model = auto_train(&df, "price")?;
///
/// // Predict
/// let predictions = model.predict(&test_df)?;
///
/// // See what happened
/// println!("{}", model.summary());
/// ```
pub fn auto_train(df: &polars::prelude::DataFrame, target_col: &str) -> Result<AutoModel> {
    AutoModel::train(df, target_col)
}

/// Train a model from a CSV file with automatic configuration
///
/// Convenience wrapper that loads a CSV file and trains a model.
/// Equivalent to loading the CSV with Polars and calling [`auto_train()`].
///
/// # Arguments
///
/// * `csv_path` - Path to CSV file
/// * `target_col` - Name of the target column
///
/// # Returns
///
/// A trained [`AutoModel`] ready for prediction, or an error if loading or training fails
///
/// # Example
///
/// ```ignore
/// use treeboost::auto_train_csv;
///
/// // One-liner training
/// let model = auto_train_csv("housing.csv", "price")?;
///
/// // Load test data and predict
/// let test_df = CsvReadOptions::default()
///     .try_into_reader_with_file_path(Some("test.csv".into()))?
///     .finish()?;
/// let predictions = model.predict(&test_df)?;
/// ```
pub fn auto_train_csv(
    csv_path: impl AsRef<std::path::Path>,
    target_col: &str,
) -> Result<AutoModel> {
    use polars::prelude::*;

    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(csv_path.as_ref().into()))
        .map_err(|e| TreeBoostError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))?
        .finish()?;

    auto_train(&df, target_col)
}

/// Train quickly with minimal tuning (for fast experimentation)
///
/// Uses [`TuningLevel::Quick`] which performs minimal hyperparameter search.
/// Ideal for rapid prototyping or when you want results in seconds rather than minutes.
///
/// # Example
///
/// ```ignore
/// use treeboost::auto_train_quick;
///
/// // Fast training for experimentation
/// let model = auto_train_quick(&df, "target")?;
/// ```
pub fn auto_train_quick(df: &polars::prelude::DataFrame, target_col: &str) -> Result<AutoModel> {
    AutoModel::train_quick(df, target_col)
}

/// Train thoroughly with extensive tuning (for best accuracy)
///
/// Uses [`TuningLevel::Thorough`] which performs comprehensive hyperparameter search.
/// Takes longer but may find better configurations, especially for complex datasets.
///
/// # Example
///
/// ```ignore
/// use treeboost::auto_train_thorough;
///
/// // Extensive search for production model
/// let model = auto_train_thorough(&df, "target")?;
/// ```
pub fn auto_train_thorough(df: &polars::prelude::DataFrame, target_col: &str) -> Result<AutoModel> {
    AutoModel::train_thorough(df, target_col)
}

/// Train with a specific boosting mode (bypass auto-selection)
///
/// Use this when you know which mode you want (e.g., from domain knowledge
/// or previous experiments) and want to skip the analysis phase.
///
/// # Example
///
/// ```ignore
/// use treeboost::{auto_train_with_mode, BoostingMode};
///
/// // Force LinearThenTree for time-series data
/// let model = auto_train_with_mode(&df, "target", BoostingMode::LinearThenTree)?;
/// ```
pub fn auto_train_with_mode(
    df: &polars::prelude::DataFrame,
    target_col: &str,
    mode: BoostingMode,
) -> Result<AutoModel> {
    AutoModel::train_with_mode(df, target_col, mode)
}

// Python module entry point
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)
}
