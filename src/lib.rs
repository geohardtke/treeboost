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
//! # Quick Start
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
pub use model::{BoostingMode, ModeSelection, UniversalConfig, UniversalModel};
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

// Python module entry point
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)
}
