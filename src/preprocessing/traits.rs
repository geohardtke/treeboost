//! Traits for extensible preprocessing
//!
//! This module provides the `PreprocessorTrait` for creating custom preprocessors
//! that can be serialized with TreeBoost models.
//!
//! # Example: Creating a Custom Preprocessor
//!
//! ```ignore
//! use treeboost::preprocessing::PreprocessorTrait;
//! use treeboost::Result;
//!
//! #[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
//! struct MyScaler {
//!     means: Vec<f32>,
//!     stds: Vec<f32>,
//!     fitted: bool,
//! }
//!
//! impl PreprocessorTrait for MyScaler {
//!     fn type_id(&self) -> &'static str {
//!         "my_crate::MyScaler"
//!     }
//!
//!     fn supports_numerical(&self) -> bool { true }
//!
//!     fn fit_numerical(&mut self, data: &[f32], num_features: usize) -> Result<()> {
//!         // Compute means and stds...
//!         self.fitted = true;
//!         Ok(())
//!     }
//!
//!     fn transform_numerical(&self, data: &mut [f32], num_features: usize) -> Result<()> {
//!         // Apply scaling...
//!         Ok(())
//!     }
//!
//!     fn is_fitted(&self) -> bool { self.fitted }
//!
//!     fn serialize_state(&self) -> Result<Vec<u8>> {
//!         serde_json::to_vec(self)
//!             .map_err(|e| treeboost::TreeBoostError::Serialization(e.to_string()))
//!     }
//!
//!     fn deserialize_state(&mut self, data: &[u8]) -> Result<()> {
//!         *self = serde_json::from_slice(data)
//!             .map_err(|e| treeboost::TreeBoostError::Serialization(e.to_string()))?;
//!         Ok(())
//!     }
//! }
//! ```

use crate::{Result, TreeBoostError};

// =============================================================================
// PreprocessorTrait - Core trait for custom preprocessors
// =============================================================================

/// Trait for custom preprocessors that can be registered and serialized with models.
///
/// # Design
///
/// This trait is **NOT object-safe** by design (due to Clone requirement).
/// Instead, we use a type-erased wrapper ([`CustomPreprocessorDyn`]) for dynamic
/// dispatch while maintaining serialization capabilities.
///
/// # Type ID
///
/// The `type_id()` method must return a stable, unique identifier for your type.
/// Use a fully-qualified name like `"my_crate::MyScaler"` to avoid collisions.
/// This ID is used for registry lookup during deserialization.
///
/// # Serialization
///
/// Custom preprocessors must implement `serialize_state()` and `deserialize_state()`
/// to control their wire format. The recommended approach is to use serde_json:
///
/// ```ignore
/// fn serialize_state(&self) -> Result<Vec<u8>> {
///     serde_json::to_vec(self)
///         .map_err(|e| TreeBoostError::Serialization(e.to_string()))
/// }
///
/// fn deserialize_state(&mut self, data: &[u8]) -> Result<()> {
///     *self = serde_json::from_slice(data)
///         .map_err(|e| TreeBoostError::Serialization(e.to_string()))?;
///     Ok(())
/// }
/// ```
pub trait PreprocessorTrait: Send + Sync + Clone + 'static {
    /// Unique type identifier for serialization/deserialization.
    ///
    /// Must be stable across versions. Use fully-qualified names like
    /// `"my_crate::v1::MyScaler"` to avoid collisions.
    fn type_id(&self) -> &'static str;

    // =========================================================================
    // Numerical Operations
    // =========================================================================

    /// Fit on numerical data (for scalers/transforms).
    ///
    /// Data is in row-major format: `data[row * num_features + col]`.
    ///
    /// Default implementation returns an error. Override if your preprocessor
    /// supports numerical operations.
    fn fit_numerical(&mut self, _data: &[f32], _num_features: usize) -> Result<()> {
        Err(TreeBoostError::Config(format!(
            "{} does not support numerical operations",
            self.type_id()
        )))
    }

    /// Transform numerical data in-place.
    ///
    /// Data is in row-major format: `data[row * num_features + col]`.
    ///
    /// Default implementation returns an error. Override if your preprocessor
    /// supports numerical operations.
    fn transform_numerical(&self, _data: &mut [f32], _num_features: usize) -> Result<()> {
        Err(TreeBoostError::Config(format!(
            "{} does not support numerical operations",
            self.type_id()
        )))
    }

    /// Check if this preprocessor supports numerical operations.
    ///
    /// Default: false. Override to return true if you implement
    /// `fit_numerical()` and `transform_numerical()`.
    fn supports_numerical(&self) -> bool {
        false
    }

    // =========================================================================
    // Categorical Operations
    // =========================================================================

    /// Fit on categorical data (for encoders).
    ///
    /// Default implementation returns an error. Override if your preprocessor
    /// supports categorical operations.
    fn fit_categorical(&mut self, _categories: &[&str]) -> Result<()> {
        Err(TreeBoostError::Config(format!(
            "{} does not support categorical operations",
            self.type_id()
        )))
    }

    /// Transform categorical data to f32 values.
    ///
    /// Default implementation returns an error. Override if your preprocessor
    /// supports categorical operations.
    fn transform_categorical(&self, _categories: &[&str]) -> Result<Vec<f32>> {
        Err(TreeBoostError::Config(format!(
            "{} does not support categorical operations",
            self.type_id()
        )))
    }

    /// Check if this preprocessor supports categorical operations.
    ///
    /// Default: false. Override to return true if you implement
    /// `fit_categorical()` and `transform_categorical()`.
    fn supports_categorical(&self) -> bool {
        false
    }

    // =========================================================================
    // Common Operations
    // =========================================================================

    /// Check if the preprocessor has been fitted.
    fn is_fitted(&self) -> bool;

    /// Serialize the preprocessor state to bytes.
    ///
    /// This is called when saving a model that contains this custom preprocessor.
    /// The bytes will be stored in the model file and passed to `deserialize_state()`
    /// when loading.
    fn serialize_state(&self) -> Result<Vec<u8>>;

    /// Deserialize the preprocessor state from bytes.
    ///
    /// This is called when loading a model that contains this custom preprocessor.
    /// The bytes were produced by a previous call to `serialize_state()`.
    fn deserialize_state(&mut self, data: &[u8]) -> Result<()>;
}

// =============================================================================
// CustomPreprocessorDyn - Object-safe wrapper for dynamic dispatch
// =============================================================================

/// Object-safe trait for type-erased custom preprocessors.
///
/// This trait wraps [`PreprocessorTrait`] implementations to enable dynamic dispatch
/// via `Box<dyn CustomPreprocessorDyn>`. It's used internally by the `Custom` variant
/// of the [`Preprocessor`](super::Preprocessor) enum.
///
/// You don't need to implement this trait directly - it's automatically implemented
/// for any type that implements [`PreprocessorTrait`] via the registry system.
pub trait CustomPreprocessorDyn: Send + Sync {
    /// Get the type identifier for registry lookup.
    fn type_id(&self) -> &'static str;

    /// Fit on numerical data.
    fn fit_numerical(&mut self, data: &[f32], num_features: usize) -> Result<()>;

    /// Transform numerical data in-place.
    fn transform_numerical(&self, data: &mut [f32], num_features: usize) -> Result<()>;

    /// Check if numerical operations are supported.
    fn supports_numerical(&self) -> bool;

    /// Fit on categorical data.
    fn fit_categorical(&mut self, categories: &[&str]) -> Result<()>;

    /// Transform categorical data to f32.
    fn transform_categorical(&self, categories: &[&str]) -> Result<Vec<f32>>;

    /// Check if categorical operations are supported.
    fn supports_categorical(&self) -> bool;

    /// Check if fitted.
    fn is_fitted(&self) -> bool;

    /// Serialize state to bytes.
    fn serialize_state(&self) -> Result<Vec<u8>>;

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn CustomPreprocessorDyn>;
}

impl Clone for Box<dyn CustomPreprocessorDyn> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl std::fmt::Debug for dyn CustomPreprocessorDyn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CustomPreprocessor({})", self.type_id())
    }
}

// =============================================================================
// PreprocessorWrapper - Bridges PreprocessorTrait to CustomPreprocessorDyn
// =============================================================================

/// Internal wrapper that implements [`CustomPreprocessorDyn`] for any [`PreprocessorTrait`].
///
/// This is used by the registry to create trait objects from concrete types.
pub(crate) struct PreprocessorWrapper<T: PreprocessorTrait>(pub T);

impl<T: PreprocessorTrait> CustomPreprocessorDyn for PreprocessorWrapper<T> {
    fn type_id(&self) -> &'static str {
        self.0.type_id()
    }

    fn fit_numerical(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        self.0.fit_numerical(data, num_features)
    }

    fn transform_numerical(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        self.0.transform_numerical(data, num_features)
    }

    fn supports_numerical(&self) -> bool {
        self.0.supports_numerical()
    }

    fn fit_categorical(&mut self, categories: &[&str]) -> Result<()> {
        self.0.fit_categorical(categories)
    }

    fn transform_categorical(&self, categories: &[&str]) -> Result<Vec<f32>> {
        self.0.transform_categorical(categories)
    }

    fn supports_categorical(&self) -> bool {
        self.0.supports_categorical()
    }

    fn is_fitted(&self) -> bool {
        self.0.is_fitted()
    }

    fn serialize_state(&self) -> Result<Vec<u8>> {
        self.0.serialize_state()
    }

    fn clone_box(&self) -> Box<dyn CustomPreprocessorDyn> {
        Box::new(PreprocessorWrapper(self.0.clone()))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct TestScaler {
        mean: f32,
        fitted: bool,
    }

    impl PreprocessorTrait for TestScaler {
        fn type_id(&self) -> &'static str {
            "test::TestScaler"
        }

        fn supports_numerical(&self) -> bool {
            true
        }

        fn fit_numerical(&mut self, data: &[f32], _num_features: usize) -> Result<()> {
            if data.is_empty() {
                return Err(TreeBoostError::Config("Empty data".into()));
            }
            self.mean = data.iter().sum::<f32>() / data.len() as f32;
            self.fitted = true;
            Ok(())
        }

        fn transform_numerical(&self, data: &mut [f32], _num_features: usize) -> Result<()> {
            if !self.fitted {
                return Err(TreeBoostError::Config("Not fitted".into()));
            }
            for x in data.iter_mut() {
                *x -= self.mean;
            }
            Ok(())
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn serialize_state(&self) -> Result<Vec<u8>> {
            Ok(format!("{}|{}", self.mean, self.fitted).into_bytes())
        }

        fn deserialize_state(&mut self, data: &[u8]) -> Result<()> {
            let s = std::str::from_utf8(data)
                .map_err(|e| TreeBoostError::Serialization(e.to_string()))?;
            let parts: Vec<&str> = s.split('|').collect();
            if parts.len() != 2 {
                return Err(TreeBoostError::Serialization("Invalid format".into()));
            }
            self.mean = parts[0].parse().map_err(|e: std::num::ParseFloatError| {
                TreeBoostError::Serialization(e.to_string())
            })?;
            self.fitted = parts[1] == "true";
            Ok(())
        }
    }

    #[test]
    fn test_preprocessor_trait_implementation() {
        let mut scaler = TestScaler::default();
        assert!(!scaler.is_fitted());
        assert!(scaler.supports_numerical());
        assert!(!scaler.supports_categorical());

        scaler.fit_numerical(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        assert!(scaler.is_fitted());
        assert!((scaler.mean - 2.5).abs() < 0.001);

        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        scaler.transform_numerical(&mut data, 2).unwrap();
        assert!((data[0] - (-1.5)).abs() < 0.001);
    }

    #[test]
    fn test_preprocessor_wrapper() {
        let scaler = TestScaler {
            mean: 2.5,
            fitted: true,
        };
        let wrapper = PreprocessorWrapper(scaler);

        assert_eq!(wrapper.type_id(), "test::TestScaler");
        assert!(wrapper.is_fitted());
        assert!(wrapper.supports_numerical());

        // Test clone_box
        let cloned = wrapper.clone_box();
        assert_eq!(cloned.type_id(), "test::TestScaler");
        assert!(cloned.is_fitted());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut scaler = TestScaler::default();
        scaler.fit_numerical(&[1.0, 2.0, 3.0], 1).unwrap();

        let serialized = scaler.serialize_state().unwrap();
        let mut loaded = TestScaler::default();
        loaded.deserialize_state(&serialized).unwrap();

        assert!(loaded.is_fitted());
        assert!((loaded.mean - scaler.mean).abs() < 0.001);
    }

    #[test]
    fn test_categorical_not_supported() {
        let scaler = TestScaler::default();
        assert!(!scaler.supports_categorical());

        let result = scaler.transform_categorical(&["A", "B"]);
        assert!(result.is_err());
    }
}
