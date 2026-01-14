//! Target transformations for bounded regression problems
//!
//! This module provides transformations that map bounded targets to unbounded space,
//! allowing standard regression losses (MSE, MAE) to work properly with bounded targets
//! like percentages, probabilities, or scores.

use crate::{Result, TreeBoostError};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

/// Trait for target transformations
///
/// Transforms bounded targets to unbounded space for training, then
/// inverse-transforms predictions back to the original bounded range.
pub trait TargetTransform: Send + Sync {
    /// Transform targets from bounded to unbounded space
    ///
    /// # Arguments
    /// * `targets` - Mutable slice of target values to transform
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if transformation fails (e.g., out-of-bounds values)
    fn transform(&self, targets: &mut [f32]) -> Result<()>;

    /// Inverse transform predictions from unbounded to bounded space
    ///
    /// # Arguments
    /// * `predictions` - Mutable slice of predictions to inverse-transform
    ///
    /// # Returns
    /// * `Ok(())` on success
    fn inverse_transform(&self, predictions: &mut [f32]) -> Result<()>;

    /// Get the expected bounds for the target values
    ///
    /// Returns (min, max) for the bounded space
    fn bounds(&self) -> (f32, f32);

    /// Clone the transform as a boxed trait object
    fn clone_box(&self) -> Box<dyn TargetTransform>;
}

/// Enum wrapper for serializable target transforms
///
/// This enum wraps all target transform types to enable serialization with rkyv and serde.
/// Use this type when you need to store transforms in configs or save them with models.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, PartialEq)]
pub enum TargetTransformKind {
    /// No transformation (identity)
    Identity,
    /// Logit/sigmoid transformation for bounded [min, max] targets
    Logit(LogitTransformParams),
    /// Clamp-only: No transform during training, clamp predictions to [min, max] at inference
    /// Use when Logit causes issues or for simpler bounded regression
    Clamp(ClampParams),
}

/// Parameters for ClampTransform (for serialization)
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, PartialEq)]
pub struct ClampParams {
    pub min: f32,
    pub max: f32,
}

/// Parameters for LogitTransform (for serialization)
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize, PartialEq)]
pub struct LogitTransformParams {
    pub min: f32,
    pub max: f32,
    pub epsilon: f32,
}

impl TargetTransformKind {
    /// Create identity transform (no-op)
    pub fn identity() -> Self {
        Self::Identity
    }

    /// Create logit transform for bounded [min, max] targets
    pub fn logit(min: f32, max: f32) -> Result<Self> {
        if min >= max {
            return Err(TreeBoostError::Config(format!(
                "LogitTransform requires min < max, got min={}, max={}",
                min, max
            )));
        }
        Ok(Self::Logit(LogitTransformParams {
            min,
            max,
            epsilon: 1e-7,
        }))
    }

    /// Create logit transform with custom epsilon
    pub fn logit_with_epsilon(min: f32, max: f32, epsilon: f32) -> Result<Self> {
        if min >= max {
            return Err(TreeBoostError::Config(format!(
                "LogitTransform requires min < max, got min={}, max={}",
                min, max
            )));
        }
        Ok(Self::Logit(LogitTransformParams { min, max, epsilon }))
    }

    /// Create clamp transform for bounded [min, max] targets
    ///
    /// Unlike Logit, this does NOT transform targets during training.
    /// It only clamps predictions to [min, max] at inference time.
    /// Use this when Logit causes issues or for simpler bounded regression.
    pub fn clamp(min: f32, max: f32) -> Result<Self> {
        if min >= max {
            return Err(TreeBoostError::Config(format!(
                "ClampTransform requires min < max, got min={}, max={}",
                min, max
            )));
        }
        Ok(Self::Clamp(ClampParams { min, max }))
    }
}

impl TargetTransform for TargetTransformKind {
    fn transform(&self, targets: &mut [f32]) -> Result<()> {
        match self {
            Self::Identity => Ok(()),
            Self::Logit(params) => {
                let transform = LogitTransform {
                    min: params.min,
                    max: params.max,
                    epsilon: params.epsilon,
                };
                transform.transform(targets)
            }
            // Clamp: no transform during training (identity)
            Self::Clamp(_) => Ok(()),
        }
    }

    fn inverse_transform(&self, predictions: &mut [f32]) -> Result<()> {
        match self {
            Self::Identity => Ok(()),
            Self::Logit(params) => {
                let transform = LogitTransform {
                    min: params.min,
                    max: params.max,
                    epsilon: params.epsilon,
                };
                transform.inverse_transform(predictions)
            }
            // Clamp: just clamp predictions to [min, max]
            Self::Clamp(params) => {
                for pred in predictions.iter_mut() {
                    *pred = pred.clamp(params.min, params.max);
                }
                Ok(())
            }
        }
    }

    fn bounds(&self) -> (f32, f32) {
        match self {
            Self::Identity => (f32::NEG_INFINITY, f32::INFINITY),
            Self::Logit(params) => (params.min, params.max),
            Self::Clamp(params) => (params.min, params.max),
        }
    }

    fn clone_box(&self) -> Box<dyn TargetTransform> {
        Box::new(self.clone())
    }
}

/// No-op transform (identity function)
///
/// Use this when targets are already unbounded or when you don't want
/// any target transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityTransform;

impl IdentityTransform {
    /// Create a new identity transform
    pub fn new() -> Self {
        Self
    }
}

impl Default for IdentityTransform {
    fn default() -> Self {
        Self::new()
    }
}

impl TargetTransform for IdentityTransform {
    fn transform(&self, _targets: &mut [f32]) -> Result<()> {
        Ok(()) // No-op
    }

    fn inverse_transform(&self, _predictions: &mut [f32]) -> Result<()> {
        Ok(()) // No-op
    }

    fn bounds(&self) -> (f32, f32) {
        (f32::NEG_INFINITY, f32::INFINITY)
    }

    fn clone_box(&self) -> Box<dyn TargetTransform> {
        Box::new(self.clone())
    }
}

/// Logit/Sigmoid transform for bounded [min, max] targets
///
/// Transforms bounded targets to unbounded space using:
/// 1. Scale to [0, 1]: `y_scaled = (y - min) / (max - min)`
/// 2. Logit transform: `y_transformed = log(y_scaled / (1 - y_scaled))`
///
/// Inverse transform (for predictions):
/// 1. Sigmoid: `y_scaled = 1 / (1 + exp(-pred))`
/// 2. Scale back: `y = y_scaled * (max - min) + min`
///
/// # Example
/// ```
/// use treeboost::preprocessing::LogitTransform;
///
/// // Exam scores in [0, 100]
/// let transform = LogitTransform::new(0.0, 100.0).unwrap();
/// let mut targets = vec![10.0, 50.0, 90.0];
///
/// // Transform to unbounded space
/// transform.transform(&mut targets).unwrap();
/// // targets are now in (-∞, +∞)
///
/// // Inverse transform predictions back to [0, 100]
/// transform.inverse_transform(&mut targets).unwrap();
/// // targets are back in [0, 100]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitTransform {
    /// Minimum value of the bounded range
    min: f32,
    /// Maximum value of the bounded range
    max: f32,
    /// Epsilon to avoid log(0) and log(1)
    epsilon: f32,
}

impl LogitTransform {
    /// Create a new logit transform for bounded [min, max] targets
    ///
    /// # Arguments
    /// * `min` - Minimum value of the bounded range
    /// * `max` - Maximum value of the bounded range
    ///
    /// # Returns
    /// * `Ok(LogitTransform)` on success
    /// * `Err` if min >= max
    ///
    /// # Example
    /// ```
    /// use treeboost::preprocessing::LogitTransform;
    ///
    /// // For exam scores [0, 100]
    /// let transform = LogitTransform::new(0.0, 100.0).unwrap();
    ///
    /// // For probabilities [0, 1]
    /// let transform = LogitTransform::new(0.0, 1.0).unwrap();
    /// ```
    pub fn new(min: f32, max: f32) -> Result<Self> {
        if min >= max {
            return Err(TreeBoostError::Config(format!(
                "LogitTransform requires min < max, got min={}, max={}",
                min, max
            )));
        }

        Ok(Self {
            min,
            max,
            epsilon: 1e-7,
        })
    }

    /// Set the epsilon value for numerical stability
    ///
    /// Epsilon is used to avoid log(0) and log(1) by clamping to [epsilon, 1-epsilon].
    /// Default is 1e-7.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl TargetTransform for LogitTransform {
    fn transform(&self, targets: &mut [f32]) -> Result<()> {
        let range = self.max - self.min;

        for target in targets.iter_mut() {
            // Validate bounds
            if *target < self.min || *target > self.max {
                return Err(TreeBoostError::Data(format!(
                    "Target value {} is outside bounds [{}, {}]. \
                     LogitTransform requires all targets to be within the specified range.",
                    target, self.min, self.max
                )));
            }

            // Scale to [0, 1]
            let scaled = (*target - self.min) / range;

            // Clamp to [epsilon, 1-epsilon] to avoid log(0) and log(1)
            let clamped = scaled.clamp(self.epsilon, 1.0 - self.epsilon);

            // Logit: log(p / (1-p))
            *target = (clamped / (1.0 - clamped)).ln();
        }

        Ok(())
    }

    fn inverse_transform(&self, predictions: &mut [f32]) -> Result<()> {
        let range = self.max - self.min;

        for pred in predictions.iter_mut() {
            // Sigmoid: 1 / (1 + exp(-x))
            let scaled = 1.0 / (1.0 + (-*pred).exp());

            // Scale back to [min, max]
            *pred = scaled * range + self.min;
        }

        Ok(())
    }

    fn bounds(&self) -> (f32, f32) {
        (self.min, self.max)
    }

    fn clone_box(&self) -> Box<dyn TargetTransform> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let transform = IdentityTransform::new();
        let mut data = vec![1.0, 2.0, 3.0];
        let original = data.clone();

        transform.transform(&mut data).unwrap();
        assert_eq!(data, original); // No change

        transform.inverse_transform(&mut data).unwrap();
        assert_eq!(data, original); // No change
    }

    #[test]
    fn test_logit_transform_bounds() {
        // Invalid: min >= max
        assert!(LogitTransform::new(100.0, 0.0).is_err());
        assert!(LogitTransform::new(50.0, 50.0).is_err());

        // Valid
        assert!(LogitTransform::new(0.0, 100.0).is_ok());
        assert!(LogitTransform::new(-1.0, 1.0).is_ok());
    }

    #[test]
    fn test_logit_transform_round_trip() {
        let transform = LogitTransform::new(0.0, 100.0).unwrap();
        let original = vec![10.0, 50.0, 90.0];
        let mut data = original.clone();

        // Transform to unbounded
        transform.transform(&mut data).unwrap();

        // Values should be different
        assert_ne!(data, original);

        // Inverse transform back
        transform.inverse_transform(&mut data).unwrap();

        // Should recover original (within tolerance)
        for (orig, recovered) in original.iter().zip(data.iter()) {
            assert!(
                (orig - recovered).abs() < 0.01,
                "Expected {}, got {}",
                orig,
                recovered
            );
        }
    }

    #[test]
    fn test_logit_transform_edge_cases() {
        let transform = LogitTransform::new(0.0, 100.0).unwrap();

        // Test near boundaries
        let mut data = vec![0.0, 0.1, 99.9, 100.0];
        transform.transform(&mut data).unwrap();

        // Should not be infinite
        for val in &data {
            assert!(val.is_finite(), "Transformed value should be finite");
        }

        // Inverse should recover original
        transform.inverse_transform(&mut data).unwrap();
        assert!((data[0] - 0.0).abs() < 0.1);
        assert!((data[1] - 0.1).abs() < 0.1);
        assert!((data[2] - 99.9).abs() < 0.1);
        assert!((data[3] - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_logit_transform_out_of_bounds() {
        let transform = LogitTransform::new(0.0, 100.0).unwrap();

        // Values outside bounds should fail
        let mut data = vec![-10.0, 50.0];
        assert!(transform.transform(&mut data).is_err());

        let mut data = vec![50.0, 110.0];
        assert!(transform.transform(&mut data).is_err());
    }

    #[test]
    fn test_logit_transform_different_ranges() {
        // Test [0, 1] range (probabilities)
        let transform = LogitTransform::new(0.0, 1.0).unwrap();
        let mut data = vec![0.1, 0.5, 0.9];
        let original = data.clone();

        transform.transform(&mut data).unwrap();
        transform.inverse_transform(&mut data).unwrap();

        for (orig, recovered) in original.iter().zip(data.iter()) {
            assert!((orig - recovered).abs() < 0.001);
        }

        // Test [-1, 1] range
        let transform = LogitTransform::new(-1.0, 1.0).unwrap();
        let mut data = vec![-0.5, 0.0, 0.5];
        let original = data.clone();

        transform.transform(&mut data).unwrap();
        transform.inverse_transform(&mut data).unwrap();

        for (orig, recovered) in original.iter().zip(data.iter()) {
            assert!((orig - recovered).abs() < 0.001);
        }
    }
}
