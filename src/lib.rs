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

pub mod analysis;
pub mod backend;
pub mod booster;
pub mod dataset;
pub mod encoding;
pub mod ensemble;
pub mod features;
pub mod histogram;
pub mod inference;
pub mod learner;
pub mod loss;
pub mod model;
pub mod monitoring;
pub mod preprocessing;
pub mod serialize;
pub mod tree;
pub mod tuner;
pub(crate) mod utils;

// Kernel re-exported from scalar backend (canonical location for CPU kernels)
pub use backend::scalar::kernel;

#[cfg(feature = "python")]
mod python;

// Re-exports for convenience
pub use backend::{BackendConfig, BackendSelector, BackendType, GpuMode, HistogramBackend};
pub use booster::{GBDTConfig, GBDTModel};
pub use dataset::{BinnedDataset, FeatureInfo, FeatureType, QuantileBinner};
pub use ensemble::{EnsembleBuilder, StackedEnsemble, MultiSeedConfig, SelectionConfig as EnsembleSelectionConfig, StackingConfig};
pub use features::{FeatureGenerationConfig, FeatureGenerator, FeatureSelector, PolynomialGenerator, RatioGenerator, SelectionConfig};
pub use histogram::HistogramBuilder;
pub use inference::Prediction;
pub use learner::{
    Booster, LeafLinearModel, LinearBooster, LinearConfig, LinearTreeBooster, LinearTreeConfig,
    TreeBooster, TreeConfig, WeakLearner,
};
pub use loss::{sigmoid, softmax, BinaryLogLoss, LossFunction, MseLoss, MultiClassLogLoss, PseudoHuberLoss};
pub use model::{AutoBuilder, AutoConfig, AutoModel, BoostingMode, BuildPhaseTimes, BuildResult, ModeSelection, TuningLevel, UniversalConfig, UniversalModel};
pub use monitoring::{AlertLevel, CVHoldoutTracker, ShiftDetector, ShiftResult};

// Analysis module exports
pub use analysis::{
    AnalysisConfig, AnalysisReport, Confidence, DatasetAnalysis, Recommendation,
    compute_correlation, compute_r2, compute_variance,
};
pub use preprocessing::{
    EncodingMap, FrequencyEncoder, ImputeStrategy, IndicatorImputer, LabelEncoder, MinMaxScaler,
    OneHotEncoder, OrderedTargetEncoder, PipelineBuilder, Preprocessor, RobustScaler, Scaler,
    SimpleImputer, StandardScaler, UnknownStrategy, YeoJohnsonTransform,
};
pub use tree::{InteractionConstraints, MonotonicConstraint};
pub use tuner::{AutoTuner, EvalStrategy, GridStrategy, ModelFormat, ParameterSpace, SearchHistory, TunerConfig};

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
pub fn auto_train_csv(csv_path: impl AsRef<std::path::Path>, target_col: &str) -> Result<AutoModel> {
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
