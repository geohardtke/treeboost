//! Preprocessing pipeline with enum dispatch pattern
//!
//! Provides a unified `Preprocessor` enum that wraps all preprocessing transformations.
//! This enables:
//! - **Serialization**: Easy to serialize/deserialize with serde/rkyv
//! - **Static dispatch**: No trait object overhead
//! - **Pipeline composition**: Chain multiple preprocessors
//!
//! # Example
//!
//! ```rust
//! use treeboost::preprocessing::{Preprocessor, StandardScaler, FrequencyEncoder};
//!
//! // Create a pipeline
//! let pipeline = Preprocessor::Pipeline(vec![
//!     Preprocessor::Standard(StandardScaler::new()),
//!     // FrequencyEncoder would be added for categorical columns
//! ]);
//!
//! // Serialize the entire pipeline
//! let json = serde_json::to_string(&pipeline).unwrap();
//! ```

use crate::{Result, TreeBoostError};

use super::encoding::{FrequencyEncoder, LabelEncoder, OneHotEncoder};
use super::imputer::SimpleImputer;
use super::scaler::{MinMaxScaler, RobustScaler, Scaler, StandardScaler};
use super::transforms::YeoJohnsonTransform;

// =============================================================================
// Preprocessor Enum (Dispatch Pattern)
// =============================================================================

/// Unified preprocessor enum for pipeline composition and serialization
///
/// This enum wraps all preprocessing transformations, enabling:
/// - Static dispatch (better performance than trait objects)
/// - Easy serialization (no `dyn` complications)
/// - Pipeline composition (via `Preprocessor::Pipeline`)
///
/// # Design Rationale
///
/// In Rust, `Box<dyn Trait>` is problematic for serialization (especially with rkyv).
/// The enum dispatch pattern provides the same polymorphism with trivial serialization.
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{Preprocessor, StandardScaler};
///
/// let mut scaler = Preprocessor::Standard(StandardScaler::new());
///
/// // Fit on training data
/// let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 features
/// scaler.fit_numerical(&data, 3).unwrap();
/// scaler.transform_numerical(&mut data, 3).unwrap();
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Preprocessor {
    // Scalers (numerical)
    /// StandardScaler: (x - μ) / σ
    Standard(StandardScaler),
    /// MinMaxScaler: scale to [min, max] range
    MinMax(MinMaxScaler),
    /// RobustScaler: (x - median) / IQR
    Robust(RobustScaler),

    // Encoders (categorical)
    /// FrequencyEncoder: category → count (optimal for trees)
    Frequency(FrequencyEncoder),
    /// LabelEncoder: category → integer label
    Label(LabelEncoder),
    /// OneHotEncoder: category → binary columns (for linear models)
    OneHot(OneHotEncoder),

    // Imputers (missing values)
    /// SimpleImputer: Mean/Median/Mode/Constant strategies
    Imputer(SimpleImputer),

    // Power transforms
    /// YeoJohnsonTransform: Normalize skewed distributions
    YeoJohnson(YeoJohnsonTransform),

    // Composition
    /// Pipeline: chain multiple preprocessors sequentially
    Pipeline(Vec<Preprocessor>),
}

impl Preprocessor {
    // -------------------------------------------------------------------------
    // Numerical Operations (for scalers)
    // -------------------------------------------------------------------------

    /// Fit on numerical data (for scalers)
    ///
    /// This operation is valid for:
    /// - StandardScaler, MinMaxScaler, RobustScaler
    /// - Pipeline (fits all numerical preprocessors in sequence)
    ///
    /// Returns error for categorical encoders (use `fit_categorical` instead).
    pub fn fit_numerical(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        match self {
            Preprocessor::Standard(s) => s.fit(data, num_features),
            Preprocessor::MinMax(s) => s.fit(data, num_features),
            Preprocessor::Robust(s) => s.fit(data, num_features),
            Preprocessor::Imputer(i) => i.fit(data, num_features),
            Preprocessor::YeoJohnson(t) => t.fit(data, num_features),
            Preprocessor::Pipeline(steps) => {
                // For pipelines, fit each step in sequence
                // Note: This creates a copy for intermediate transformations
                let mut current_data = data.to_vec();
                for step in steps.iter_mut() {
                    step.fit_numerical(&current_data, num_features)?;
                    step.transform_numerical(&mut current_data, num_features)?;
                }
                Ok(())
            }
            _ => Err(TreeBoostError::Config(
                "fit_numerical not supported for categorical encoders".into(),
            )),
        }
    }

    /// Transform numerical data in-place (for scalers, imputers, and transforms)
    pub fn transform_numerical(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        match self {
            Preprocessor::Standard(s) => s.transform(data, num_features),
            Preprocessor::MinMax(s) => s.transform(data, num_features),
            Preprocessor::Robust(s) => s.transform(data, num_features),
            Preprocessor::Imputer(i) => i.transform(data, num_features),
            Preprocessor::YeoJohnson(t) => t.transform(data, num_features),
            Preprocessor::Pipeline(steps) => {
                for step in steps {
                    step.transform_numerical(data, num_features)?;
                }
                Ok(())
            }
            _ => Err(TreeBoostError::Config(
                "transform_numerical not supported for categorical encoders".into(),
            )),
        }
    }

    /// Fit and transform numerical data (convenience)
    pub fn fit_transform_numerical(
        &mut self,
        data: &mut [f32],
        num_features: usize,
    ) -> Result<()> {
        self.fit_numerical(data, num_features)?;
        self.transform_numerical(data, num_features)?;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Categorical Operations (for encoders)
    // -------------------------------------------------------------------------

    /// Fit on categorical data (for encoders)
    ///
    /// This operation is valid for:
    /// - FrequencyEncoder, LabelEncoder, OneHotEncoder
    ///
    /// Returns error for scalers (use `fit_numerical` instead).
    pub fn fit_categorical(&mut self, categories: &[impl AsRef<str>]) -> Result<()> {
        match self {
            Preprocessor::Frequency(e) => {
                e.fit(categories);
                Ok(())
            }
            Preprocessor::Label(e) => {
                e.fit(categories);
                Ok(())
            }
            Preprocessor::OneHot(e) => {
                e.fit(categories);
                Ok(())
            }
            _ => Err(TreeBoostError::Config(
                "fit_categorical not supported for numerical scalers".into(),
            )),
        }
    }

    /// Transform categorical data to f32 (for tree input)
    ///
    /// - FrequencyEncoder: category → count (f32)
    /// - LabelEncoder: category → label (f32)
    /// - OneHotEncoder: category → flattened binary columns (f32)
    pub fn transform_categorical(&self, categories: &[impl AsRef<str>]) -> Result<Vec<f32>> {
        match self {
            Preprocessor::Frequency(e) => e.transform(categories),
            Preprocessor::Label(e) => e.transform_f32(categories),
            Preprocessor::OneHot(e) => e.transform(categories),
            _ => Err(TreeBoostError::Config(
                "transform_categorical not supported for numerical scalers".into(),
            )),
        }
    }

    /// Fit and transform categorical data (convenience)
    pub fn fit_transform_categorical(
        &mut self,
        categories: &[impl AsRef<str>],
    ) -> Result<Vec<f32>> {
        self.fit_categorical(categories)?;
        self.transform_categorical(categories)
    }

    // -------------------------------------------------------------------------
    // Utility Methods
    // -------------------------------------------------------------------------

    /// Check if the preprocessor has been fitted
    pub fn is_fitted(&self) -> bool {
        match self {
            Preprocessor::Standard(s) => s.is_fitted(),
            Preprocessor::MinMax(s) => s.is_fitted(),
            Preprocessor::Robust(s) => s.is_fitted(),
            Preprocessor::Frequency(e) => e.is_fitted(),
            Preprocessor::Label(e) => e.is_fitted(),
            Preprocessor::OneHot(e) => e.is_fitted(),
            Preprocessor::Imputer(i) => i.is_fitted(),
            Preprocessor::YeoJohnson(t) => t.is_fitted(),
            Preprocessor::Pipeline(steps) => steps.iter().all(|s| s.is_fitted()),
        }
    }

    /// Check if this is a numerical preprocessor (scaler, imputer, or transform)
    pub fn is_numerical(&self) -> bool {
        matches!(
            self,
            Preprocessor::Standard(_)
                | Preprocessor::MinMax(_)
                | Preprocessor::Robust(_)
                | Preprocessor::Imputer(_)
                | Preprocessor::YeoJohnson(_)
        )
    }

    /// Check if this is a categorical preprocessor (encoder)
    pub fn is_categorical(&self) -> bool {
        matches!(
            self,
            Preprocessor::Frequency(_) | Preprocessor::Label(_) | Preprocessor::OneHot(_)
        )
    }

    /// Check if this is an imputer
    pub fn is_imputer(&self) -> bool {
        matches!(self, Preprocessor::Imputer(_))
    }

    /// Check if this is a power transform
    pub fn is_transform(&self) -> bool {
        matches!(self, Preprocessor::YeoJohnson(_))
    }

    /// Check if this is a pipeline
    pub fn is_pipeline(&self) -> bool {
        matches!(self, Preprocessor::Pipeline(_))
    }

    /// Get the number of output columns for categorical encoders
    ///
    /// Returns `None` for scalers (output columns = input columns).
    pub fn num_output_columns(&self) -> Option<usize> {
        match self {
            Preprocessor::Frequency(_) => Some(1), // Single column output
            Preprocessor::Label(_) => Some(1),     // Single column output
            Preprocessor::OneHot(e) => Some(e.num_columns()),
            _ => None,
        }
    }
}

// =============================================================================
// Pipeline Builder
// =============================================================================

/// Builder for creating preprocessing pipelines
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{PipelineBuilder, StandardScaler, FrequencyEncoder};
///
/// let pipeline = PipelineBuilder::new()
///     .add_scaler(StandardScaler::new())
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct PipelineBuilder {
    steps: Vec<Preprocessor>,
}

impl PipelineBuilder {
    /// Create a new empty pipeline builder
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a StandardScaler to the pipeline
    pub fn add_standard_scaler(mut self) -> Self {
        self.steps.push(Preprocessor::Standard(StandardScaler::new()));
        self
    }

    /// Add a MinMaxScaler to the pipeline
    pub fn add_minmax_scaler(mut self) -> Self {
        self.steps.push(Preprocessor::MinMax(MinMaxScaler::new()));
        self
    }

    /// Add a RobustScaler to the pipeline
    pub fn add_robust_scaler(mut self) -> Self {
        self.steps.push(Preprocessor::Robust(RobustScaler::new()));
        self
    }

    /// Add a mean imputer to the pipeline
    pub fn add_mean_imputer(mut self) -> Self {
        self.steps.push(Preprocessor::Imputer(SimpleImputer::mean()));
        self
    }

    /// Add a median imputer to the pipeline
    pub fn add_median_imputer(mut self) -> Self {
        self.steps.push(Preprocessor::Imputer(SimpleImputer::median()));
        self
    }

    /// Add a custom imputer to the pipeline
    pub fn add_imputer(mut self, imputer: SimpleImputer) -> Self {
        self.steps.push(Preprocessor::Imputer(imputer));
        self
    }

    /// Add a Yeo-Johnson transform to the pipeline
    pub fn add_yeo_johnson(mut self) -> Self {
        self.steps
            .push(Preprocessor::YeoJohnson(YeoJohnsonTransform::new()));
        self
    }

    /// Add a custom scaler to the pipeline
    pub fn add_scaler(mut self, scaler: impl Into<Preprocessor>) -> Self {
        self.steps.push(scaler.into());
        self
    }

    /// Add a preprocessor to the pipeline
    pub fn add(mut self, preprocessor: Preprocessor) -> Self {
        self.steps.push(preprocessor);
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Preprocessor {
        if self.steps.len() == 1 {
            self.steps.into_iter().next().unwrap()
        } else {
            Preprocessor::Pipeline(self.steps)
        }
    }
}

// =============================================================================
// Conversion Traits
// =============================================================================

impl From<StandardScaler> for Preprocessor {
    fn from(scaler: StandardScaler) -> Self {
        Preprocessor::Standard(scaler)
    }
}

impl From<MinMaxScaler> for Preprocessor {
    fn from(scaler: MinMaxScaler) -> Self {
        Preprocessor::MinMax(scaler)
    }
}

impl From<RobustScaler> for Preprocessor {
    fn from(scaler: RobustScaler) -> Self {
        Preprocessor::Robust(scaler)
    }
}

impl From<FrequencyEncoder> for Preprocessor {
    fn from(encoder: FrequencyEncoder) -> Self {
        Preprocessor::Frequency(encoder)
    }
}

impl From<LabelEncoder> for Preprocessor {
    fn from(encoder: LabelEncoder) -> Self {
        Preprocessor::Label(encoder)
    }
}

impl From<OneHotEncoder> for Preprocessor {
    fn from(encoder: OneHotEncoder) -> Self {
        Preprocessor::OneHot(encoder)
    }
}

impl From<SimpleImputer> for Preprocessor {
    fn from(imputer: SimpleImputer) -> Self {
        Preprocessor::Imputer(imputer)
    }
}

impl From<YeoJohnsonTransform> for Preprocessor {
    fn from(transform: YeoJohnsonTransform) -> Self {
        Preprocessor::YeoJohnson(transform)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_standard_scaler() {
        let mut preprocessor = Preprocessor::Standard(StandardScaler::new());
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 features

        preprocessor.fit_numerical(&data, 3).unwrap();
        assert!(preprocessor.is_fitted());
        assert!(preprocessor.is_numerical());
        assert!(!preprocessor.is_categorical());

        preprocessor.transform_numerical(&mut data, 3).unwrap();
    }

    #[test]
    fn test_preprocessor_frequency_encoder() {
        let mut preprocessor = Preprocessor::Frequency(FrequencyEncoder::new());
        let categories = vec!["A", "B", "A", "C"];

        preprocessor.fit_categorical(&categories).unwrap();
        assert!(preprocessor.is_fitted());
        assert!(preprocessor.is_categorical());
        assert!(!preprocessor.is_numerical());

        let encoded = preprocessor.transform_categorical(&categories).unwrap();
        assert_eq!(encoded, vec![2.0, 1.0, 2.0, 1.0]); // A=2, B=1, C=1
    }

    #[test]
    fn test_preprocessor_pipeline() {
        let mut pipeline = Preprocessor::Pipeline(vec![
            Preprocessor::Standard(StandardScaler::new()),
            Preprocessor::MinMax(MinMaxScaler::new()),
        ]);

        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        pipeline.fit_numerical(&data, 3).unwrap();

        assert!(pipeline.is_fitted());
        assert!(pipeline.is_pipeline());

        pipeline.transform_numerical(&mut data, 3).unwrap();
    }

    #[test]
    fn test_preprocessor_serialization() {
        let mut preprocessor = Preprocessor::Standard(StandardScaler::new());
        let data = vec![1.0, 2.0, 3.0, 4.0];
        preprocessor.fit_numerical(&data, 2).unwrap();

        // Serialize
        let json = serde_json::to_string(&preprocessor).unwrap();
        assert!(!json.is_empty());

        // Deserialize
        let loaded: Preprocessor = serde_json::from_str(&json).unwrap();
        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new()
            .add_standard_scaler()
            .add_minmax_scaler()
            .build();

        assert!(pipeline.is_pipeline());
    }

    #[test]
    fn test_from_conversions() {
        let p1: Preprocessor = StandardScaler::new().into();
        assert!(matches!(p1, Preprocessor::Standard(_)));

        let p2: Preprocessor = FrequencyEncoder::new().into();
        assert!(matches!(p2, Preprocessor::Frequency(_)));

        let p3: Preprocessor = OneHotEncoder::new().into();
        assert!(matches!(p3, Preprocessor::OneHot(_)));
    }

    #[test]
    fn test_numerical_on_categorical_error() {
        let mut preprocessor = Preprocessor::Frequency(FrequencyEncoder::new());
        let data = vec![1.0, 2.0, 3.0];

        let result = preprocessor.fit_numerical(&data, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_categorical_on_numerical_error() {
        let mut preprocessor = Preprocessor::Standard(StandardScaler::new());
        let categories = vec!["A", "B"];

        let result = preprocessor.fit_categorical(&categories);
        assert!(result.is_err());
    }
}
