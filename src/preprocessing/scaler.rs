//! Scaling transformations for numerical features
//!
//! Scalers normalize feature distributions to improve model performance:
//! - **StandardScaler**: Zero mean, unit variance (most common)
//! - **MinMaxScaler**: Scale to fixed range [min, max]
//! - **RobustScaler**: Use median/IQR (robust to outliers)
//!
//! # Example
//!
//! ```rust
//! use treeboost::preprocessing::{StandardScaler, Scaler};
//!
//! let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 features
//! let num_features = 3;
//!
//! let mut scaler = StandardScaler::new();
//! scaler.fit(&data, num_features);
//! scaler.transform(&mut data, num_features);
//!
//! // data is now standardized: (x - mean) / std
//! ```

use crate::{Result, TreeBoostError};

// =============================================================================
// Scaler Trait
// =============================================================================

/// Trait for all scalers (fit-transform pattern)
///
/// All scalers must implement:
/// - `fit()`: Learn parameters from training data
/// - `transform()`: Apply learned parameters to data
/// - Serialization for train/test consistency
pub trait Scaler {
    /// Fit scaler on training data (row-major: num_rows × num_features)
    ///
    /// # Arguments
    /// - `data`: Row-major flat array (row0_feat0, row0_feat1, ..., row1_feat0, ...)
    /// - `num_features`: Number of features per row
    fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()>;

    /// Transform data in-place using fitted parameters
    ///
    /// # Arguments
    /// - `data`: Row-major flat array to transform in-place
    /// - `num_features`: Number of features per row (must match fit)
    fn transform(&self, data: &mut [f32], num_features: usize) -> Result<()>;

    /// Fit and transform in one step (convenience)
    fn fit_transform(&mut self, data: &mut [f32], num_features: usize) -> Result<()> {
        self.fit(data, num_features)?;
        self.transform(data, num_features)?;
        Ok(())
    }

    /// Check if scaler has been fitted
    fn is_fitted(&self) -> bool;
}

// =============================================================================
// StandardScaler
// =============================================================================

/// StandardScaler: (x - μ) / σ
///
/// Transforms features to have zero mean and unit variance.
///
/// # Why it helps GBDTs
/// Even though trees are scale-invariant, scaling improves:
/// - **Regularization fairness**: L1/L2 penalties applied uniformly
/// - **Binning uniformity**: Quantiles distributed evenly
/// - **Numerical stability**: Gradient/Hessian calculations
/// - **Mixed ensembles**: Combining linear + tree models
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{StandardScaler, Scaler};
///
/// let mut train = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]; // 3 rows × 2 features
/// let mut test = vec![1.5, 15.0, 2.5, 25.0]; // 2 rows × 2 features
///
/// let mut scaler = StandardScaler::new();
/// scaler.fit(&train, 2)?;
/// scaler.transform(&mut train, 2)?;
/// scaler.transform(&mut test, 2)?; // Use same mean/std from training
/// # Ok::<(), treeboost::TreeBoostError>(())
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StandardScaler {
    /// Mean of each feature (learned during fit)
    pub means: Vec<f32>,
    /// Standard deviation of each feature (learned during fit)
    pub stds: Vec<f32>,
    /// Whether fit() has been called
    fitted: bool,
}

impl StandardScaler {
    /// Create a new unfitted StandardScaler
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            stds: Vec::new(),
            fitted: false,
        }
    }

    /// Get the means (only valid after fit)
    pub fn means(&self) -> &[f32] {
        &self.means
    }

    /// Get the standard deviations (only valid after fit)
    pub fn stds(&self) -> &[f32] {
        &self.stds
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scaler for StandardScaler {
    fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if num_features == 0 {
            return Err(TreeBoostError::Data("num_features must be > 0".into()));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        if num_rows == 0 {
            return Err(TreeBoostError::Data("No rows to fit".into()));
        }

        self.means = vec![0.0; num_features];
        self.stds = vec![0.0; num_features];

        // Compute means
        for feat in 0..num_features {
            let mut sum = 0.0;
            for row in 0..num_rows {
                sum += data[row * num_features + feat];
            }
            self.means[feat] = sum / num_rows as f32;
        }

        // Compute standard deviations
        for feat in 0..num_features {
            let mean = self.means[feat];
            let mut variance = 0.0;
            for row in 0..num_rows {
                let x = data[row * num_features + feat];
                variance += (x - mean).powi(2);
            }
            let std = (variance / num_rows as f32).sqrt();

            // Handle zero-variance features (constant column)
            self.stds[feat] = if std < 1e-8 { 1.0 } else { std };
        }

        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "StandardScaler not fitted. Call fit() first.".into(),
            ));
        }

        if num_features != self.means.len() {
            return Err(TreeBoostError::Data(format!(
                "num_features mismatch: fit with {}, transform with {}",
                self.means.len(),
                num_features
            )));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        // Apply standardization: (x - mean) / std
        for feat in 0..num_features {
            let mean = self.means[feat];
            let std = self.stds[feat];
            for row in 0..num_rows {
                let idx = row * num_features + feat;
                data[idx] = (data[idx] - mean) / std;
            }
        }

        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// =============================================================================
// MinMaxScaler
// =============================================================================

/// MinMaxScaler: (x - min) / (max - min) * (b - a) + a
///
/// Scales features to a fixed range [a, b] (default [0, 1]).
///
/// # Use cases
/// - When you need features in a specific range (e.g., [0, 1] for neural nets)
/// - When you know the expected min/max bounds
///
/// # Warning
/// - Sensitive to outliers (one extreme value affects entire scale)
/// - Consider RobustScaler if outliers are present
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{MinMaxScaler, Scaler};
///
/// let mut data = vec![1.0, 5.0, 2.0, 10.0, 3.0, 15.0]; // 3 rows × 2 features
///
/// let mut scaler = MinMaxScaler::new().with_range(0.0, 1.0);
/// scaler.fit(&data, 2)?;
/// scaler.transform(&mut data, 2)?;
///
/// // data is now in [0, 1] range
/// # Ok::<(), treeboost::TreeBoostError>(())
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinMaxScaler {
    /// Minimum of each feature (learned during fit)
    pub mins: Vec<f32>,
    /// Maximum of each feature (learned during fit)
    pub maxs: Vec<f32>,
    /// Output range (a, b)
    pub feature_range: (f32, f32),
    /// Whether fit() has been called
    fitted: bool,
}

impl MinMaxScaler {
    /// Create a new unfitted MinMaxScaler with default range [0, 1]
    pub fn new() -> Self {
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            feature_range: (0.0, 1.0),
            fitted: false,
        }
    }

    /// Set the output range (default is [0, 1])
    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.feature_range = (min, max);
        self
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scaler for MinMaxScaler {
    fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if num_features == 0 {
            return Err(TreeBoostError::Data("num_features must be > 0".into()));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        if num_rows == 0 {
            return Err(TreeBoostError::Data("No rows to fit".into()));
        }

        self.mins = vec![f32::INFINITY; num_features];
        self.maxs = vec![f32::NEG_INFINITY; num_features];

        // Find min and max for each feature
        for feat in 0..num_features {
            for row in 0..num_rows {
                let val = data[row * num_features + feat];
                self.mins[feat] = self.mins[feat].min(val);
                self.maxs[feat] = self.maxs[feat].max(val);
            }

            // Handle constant features (min == max)
            if (self.maxs[feat] - self.mins[feat]).abs() < 1e-8 {
                self.maxs[feat] = self.mins[feat] + 1.0;
            }
        }

        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "MinMaxScaler not fitted. Call fit() first.".into(),
            ));
        }

        if num_features != self.mins.len() {
            return Err(TreeBoostError::Data(format!(
                "num_features mismatch: fit with {}, transform with {}",
                self.mins.len(),
                num_features
            )));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;
        let (a, b) = self.feature_range;

        // Apply scaling: (x - min) / (max - min) * (b - a) + a
        for feat in 0..num_features {
            let min = self.mins[feat];
            let max = self.maxs[feat];
            let scale = b - a;

            for row in 0..num_rows {
                let idx = row * num_features + feat;
                data[idx] = (data[idx] - min) / (max - min) * scale + a;

                // Clip to range (handles out-of-bound values in test set)
                data[idx] = data[idx].clamp(a, b);
            }
        }

        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// =============================================================================
// RobustScaler
// =============================================================================

/// RobustScaler: (x - median) / IQR
///
/// Scales features using statistics robust to outliers:
/// - Center: median (instead of mean)
/// - Scale: IQR = Q3 - Q1 (instead of std)
///
/// # Use cases
/// - Data with outliers or heavy-tailed distributions
/// - When mean/std are unreliable due to extreme values
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{RobustScaler, Scaler};
///
/// let mut data = vec![1.0, 2.0, 3.0, 100.0, 5.0, 6.0]; // 3 rows × 2 features (outlier: 100)
///
/// let mut scaler = RobustScaler::new();
/// scaler.fit(&data, 2)?;
/// scaler.transform(&mut data, 2)?;
///
/// // Outlier (100) has less impact than with StandardScaler
/// # Ok::<(), treeboost::TreeBoostError>(())
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RobustScaler {
    /// Median of each feature (learned during fit)
    pub medians: Vec<f32>,
    /// IQR (Q3 - Q1) of each feature (learned during fit)
    pub iqrs: Vec<f32>,
    /// Whether fit() has been called
    fitted: bool,
}

impl RobustScaler {
    /// Create a new unfitted RobustScaler
    pub fn new() -> Self {
        Self {
            medians: Vec::new(),
            iqrs: Vec::new(),
            fitted: false,
        }
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scaler for RobustScaler {
    fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if num_features == 0 {
            return Err(TreeBoostError::Data("num_features must be > 0".into()));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        if num_rows == 0 {
            return Err(TreeBoostError::Data("No rows to fit".into()));
        }

        self.medians = vec![0.0; num_features];
        self.iqrs = vec![0.0; num_features];

        // Use T-Digest for O(n) quantile estimation instead of O(n log n) sorting
        // This is critical for large datasets (100M+ rows)
        use tdigest::TDigest;

        // Compute median and IQR for each feature using approximate quantiles
        for feat in 0..num_features {
            // Build T-Digest for this feature column
            let mut digest = TDigest::new_with_size(100); // 100 centroids is accurate enough

            for row in 0..num_rows {
                let value = data[row * num_features + feat] as f64;
                if value.is_finite() {
                    digest = digest.merge_unsorted(vec![value]);
                }
            }

            // Get approximate quantiles from T-Digest
            let q1 = digest.estimate_quantile(0.25) as f32;
            let median = digest.estimate_quantile(0.50) as f32;
            let q3 = digest.estimate_quantile(0.75) as f32;

            self.medians[feat] = median;

            // IQR = Q3 - Q1
            let iqr = q3 - q1;

            // Handle zero IQR (all values in Q1-Q3 range are same)
            self.iqrs[feat] = if iqr < 1e-8 { 1.0 } else { iqr };
        }

        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "RobustScaler not fitted. Call fit() first.".into(),
            ));
        }

        if num_features != self.medians.len() {
            return Err(TreeBoostError::Data(format!(
                "num_features mismatch: fit with {}, transform with {}",
                self.medians.len(),
                num_features
            )));
        }

        if !data.len().is_multiple_of(num_features) {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        // Apply robust scaling: (x - median) / IQR
        for feat in 0..num_features {
            let median = self.medians[feat];
            let iqr = self.iqrs[feat];
            for row in 0..num_rows {
                let idx = row * num_features + feat;
                data[idx] = (data[idx] - median) / iqr;
            }
        }

        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // 2 rows × 3 features
        // Row 0: [1.0, 2.0, 3.0]
        // Row 1: [4.0, 5.0, 6.0]
        let num_features = 3;

        let mut scaler = StandardScaler::new();
        assert!(!scaler.is_fitted());

        scaler.fit(&data, num_features).unwrap();
        assert!(scaler.is_fitted());

        // Check means (column averages)
        // Feature 0: (1.0 + 4.0) / 2 = 2.5
        // Feature 1: (2.0 + 5.0) / 2 = 3.5
        // Feature 2: (3.0 + 6.0) / 2 = 4.5
        assert_eq!(scaler.means(), &[2.5, 3.5, 4.5]);

        scaler.transform(&mut data, num_features).unwrap();

        // After standardization, mean should be ~0, std should be ~1
    }

    #[test]
    fn test_standard_scaler_zero_variance() {
        let mut data = vec![5.0, 1.0, 2.0, 5.0, 3.0, 4.0];
        // 2 rows × 3 features
        // Row 0: [5.0, 1.0, 2.0]
        // Row 1: [5.0, 3.0, 4.0]
        // Feature 0 is constant: [5.0, 5.0]
        let num_features = 3;

        let mut scaler = StandardScaler::new();
        scaler.fit(&data, num_features).unwrap();

        // Zero-variance feature should have std = 1.0 (fallback)
        assert_eq!(scaler.stds[0], 1.0);
        assert_eq!(scaler.means[0], 5.0);

        // Transform should not panic
        scaler.transform(&mut data, num_features).unwrap();
    }

    #[test]
    fn test_minmax_scaler_basic() {
        let mut data = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]; // 3 rows × 2 features
        let num_features = 2;

        let mut scaler = MinMaxScaler::new();
        scaler.fit(&data, num_features).unwrap();

        assert_eq!(scaler.mins, vec![1.0, 10.0]);
        assert_eq!(scaler.maxs, vec![3.0, 30.0]);

        scaler.transform(&mut data, num_features).unwrap();

        // First feature: [1, 2, 3] → [0.0, 0.5, 1.0]
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[2] - 0.5).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6);

        // Second feature: [10, 20, 30] → [0.0, 0.5, 1.0]
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[3] - 0.5).abs() < 1e-6);
        assert!((data[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_scaler_custom_range() {
        let mut data = vec![1.0, 2.0, 3.0]; // 3 rows × 1 feature
        let num_features = 1;

        let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
        scaler.fit(&data, num_features).unwrap();
        scaler.transform(&mut data, num_features).unwrap();

        // [1, 2, 3] → [-1.0, 0.0, 1.0]
        assert!((data[0] - (-1.0)).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_robust_scaler_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 100.0]; // 2 rows × 2 features (outlier: 100)
        let num_features = 2;

        let mut scaler = RobustScaler::new();
        scaler.fit(&data, num_features).unwrap();

        // Medians: [2.0, 51.0] (avg of middle two values)
        assert!((scaler.medians[0] - 2.0).abs() < 1e-6);

        scaler.transform(&mut data, num_features).unwrap();

        // Check that outlier doesn't dominate (median-based scaling)
    }

    #[test]
    fn test_scaler_not_fitted_error() {
        let mut data = vec![1.0, 2.0, 3.0];
        let scaler = StandardScaler::new();

        let result = scaler.transform(&mut data, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not fitted"));
    }

    #[test]
    fn test_scaler_feature_mismatch_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut scaler = StandardScaler::new();

        scaler.fit(&data, 2).unwrap(); // Fit with 2 features

        let mut test_data = vec![5.0, 6.0, 7.0];
        let result = scaler.transform(&mut test_data, 3); // Try to transform with 3 features

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }
}
