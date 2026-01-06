//! Preprocessing transformations for data preparation
//!
//! This module provides standard preprocessing operations:
//!
//! ## Scaling
//! - **StandardScaler**: Zero mean, unit variance (most common)
//! - **MinMaxScaler**: Scale to fixed range [min, max]
//! - **RobustScaler**: Median/IQR scaling (robust to outliers)
//!
//! ## Categorical Encoding
//! - **FrequencyEncoder**: Category → count (optimal for trees)
//! - **LabelEncoder**: String → integer label (CSV loading essential)
//! - **OneHotEncoder**: Category → binary columns (for linear models)
//! - **OrderedTargetEncoder**: Target-based encoding with M-estimate smoothing (high-cardinality)
//!
//! ## Missing Value Imputation
//! - **SimpleImputer**: Mean/Median/Mode/Constant strategies
//! - **IndicatorImputer**: Binary flags for missing values
//!
//! ## Power Transforms
//! - **YeoJohnsonTransform**: Normalize skewed distributions (handles negatives)
//!
//! ## Time-Series Features
//! - **LagGenerator**: Create lagged features (x_{t-1}, x_{t-2}, etc.)
//! - **RollingGenerator**: Rolling statistics (mean, std, min, max, sum, median)
//! - **EwmaGenerator**: Exponentially weighted moving average
//! - **SeasonalGenerator**: Extract datetime components with cyclical encoding
//!
//! ## Outlier Detection
//! - **OutlierDetector**: Detect and handle outliers with IQR or Z-score methods
//!   - Actions: Cap (winsorize), Flag (indicator columns), Remove (filter rows)
//!
//! ## Design Philosophy
//!
//! All preprocessors follow the fit-transform pattern:
//! 1. `fit()` on training data to learn parameters
//! 2. `transform()` on train/test data using learned parameters
//! 3. Serialize fitted state with model for inference consistency
//!
//! ## GBDT vs Linear Model Considerations
//!
//! **For Trees (GBDT)**:
//! - Prefer FrequencyEncoder or LabelEncoder
//! - Scalers help with regularization fairness and binning
//! - Missing values handled via bin 0 (implicit indicator)
//! - Power transforms have minimal impact (non-parametric)
//!
//! **For Linear Models**:
//! - Scalers are ESSENTIAL (sensitive to feature scales)
//! - OneHotEncoder for categorical (need binary indicators)
//! - SimpleImputer required (linear models can't handle NaN)
//! - YeoJohnsonTransform critical (Gaussian residuals assumption)
//!
//! **For Mixed Ensembles (Linear + Tree)**:
//! - Use all preprocessing types
//! - Different encodings for different model components
//!
//! ## Polars Integration
//!
//! The `polars_ext` module provides ergonomic helpers for working with
//! Polars DataFrames:
//!
//! ```ignore
//! use treeboost::preprocessing::polars_ext::*;
//!
//! let (data, num_features) = df_to_features(&df, &["col1", "col2"])?;
//! scaler.fit_transform(&mut data, num_features)?;
//! let scaled_df = features_to_df(&data, num_features, &["col1", "col2"])?;
//! ```

pub mod encoding;
pub mod imputer;
pub mod outliers;
pub mod pipeline;
pub mod polars_ext;
pub mod scaler;
pub mod smart;
pub mod timeseries;
pub mod transforms;

pub use encoding::{FrequencyEncoder, LabelEncoder, OneHotEncoder, UnknownStrategy};
pub use imputer::{ImputeStrategy, IndicatorImputer, SimpleImputer};

// Re-export production-grade encoders from encoding module for convenience
pub use crate::encoding::{EncodingMap, OrderedTargetEncoder};
pub use outliers::{FeatureBounds, OutlierAction, OutlierDetector, OutlierMethod, TransformResult};
pub use pipeline::{PipelineBuilder, Preprocessor};
pub use polars_ext::{
    column_to_f32, column_to_strings, df_column_names, df_to_features, df_to_target, features_to_df,
    is_categorical, is_numeric, series_to_f32, series_to_strings, split_by_dtype,
};
pub use scaler::{MinMaxScaler, RobustScaler, Scaler, StandardScaler};
pub use timeseries::{
    EwmaGenerator, LagGenerator, NaNStrategy, RollingGenerator, RollingStat, SeasonalComponent,
    SeasonalGenerator,
};
pub use transforms::YeoJohnsonTransform;
pub use smart::{
    SmartPreprocessor, SmartPreprocessConfig, PreprocessingPlan, PreprocessingStep,
    LttPreprocessingPlan, ModelType, EncodingType, ScalerType,
};
