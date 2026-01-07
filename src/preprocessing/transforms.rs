//! Power transformations for normalizing distributions
//!
//! This module provides power transforms that make data more Gaussian-like:
//!
//! ## Yeo-Johnson Transform
//! - Handles **both positive and negative values** (unlike Box-Cox or log)
//! - Makes data more normally distributed (critical for linear models)
//! - For trees: minimal impact, but helpful for mixed ensembles
//!
//! ## Design Philosophy
//!
//! Power transforms are **ESSENTIAL** for linear model components in mixed ensembles.
//! Linear models (OLS, Ridge, Lasso) assume Gaussian residuals, which requires
//! approximately Gaussian features. Yeo-Johnson normalizes skewed distributions.

use crate::{Result, TreeBoostError};

// =============================================================================
// Yeo-Johnson Transform
// =============================================================================

/// Yeo-Johnson power transform for normalizing distributions
///
/// Unlike Box-Cox, Yeo-Johnson handles **both positive and negative values**.
/// It applies a power transformation controlled by parameter λ (lambda).
///
/// # Transform Definition
///
/// For input x and parameter λ:
/// - If x ≥ 0 and λ ≠ 0: y = ((x + 1)^λ - 1) / λ
/// - If x ≥ 0 and λ = 0: y = log(x + 1)
/// - If x < 0 and λ ≠ 2: y = -((-x + 1)^(2-λ) - 1) / (2 - λ)
/// - If x < 0 and λ = 2: y = -log(-x + 1)
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::YeoJohnsonTransform;
///
/// let mut transform = YeoJohnsonTransform::new();
///
/// // Data with skewed distribution
/// let mut data = vec![0.1, 1.0, 10.0, 100.0, -5.0, -1.0]; // 2 rows × 3 features
///
/// transform.fit(&data, 3).unwrap();
/// transform.transform(&mut data, 3).unwrap();
/// // Data is now more Gaussian-like
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct YeoJohnsonTransform {
    /// Lambda parameter per feature (learned during fit)
    lambdas: Vec<f32>,
    /// Whether the transform has been fitted
    fitted: bool,
    /// Max iterations for lambda optimization
    max_iter: usize,
    /// Tolerance for lambda optimization
    tolerance: f32,
}

impl Default for YeoJohnsonTransform {
    fn default() -> Self {
        Self::new()
    }
}

impl YeoJohnsonTransform {
    /// Create a new Yeo-Johnson transform
    pub fn new() -> Self {
        Self {
            lambdas: Vec::new(),
            fitted: false,
            max_iter: 100,
            tolerance: 1e-6,
        }
    }

    /// Set maximum iterations for lambda optimization
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance for lambda optimization
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Create a transform with fixed lambdas (skip fitting)
    pub fn with_lambdas(lambdas: Vec<f32>) -> Self {
        Self {
            lambdas,
            fitted: true,
            max_iter: 100,
            tolerance: 1e-6,
        }
    }

    /// Fit the transform by finding optimal lambda for each feature
    ///
    /// Uses Brent's method to maximize log-likelihood.
    /// Data is in row-major format: `data[row * num_features + col]`
    pub fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if data.is_empty() {
            return Err(TreeBoostError::Data("Cannot fit on empty data".into()));
        }

        let num_rows = data.len() / num_features;
        if data.len() != num_rows * num_features {
            return Err(TreeBoostError::Data(format!(
                "Data length {} is not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        self.lambdas = Vec::with_capacity(num_features);

        for col in 0..num_features {
            // Extract column values (skip NaN)
            let values: Vec<f32> = (0..num_rows)
                .map(|row| data[row * num_features + col])
                .filter(|v| !v.is_nan())
                .collect();

            if values.is_empty() {
                // All NaN - use lambda = 1 (identity-ish)
                self.lambdas.push(1.0);
                continue;
            }

            // Find optimal lambda using Brent's method
            let optimal_lambda = self.find_optimal_lambda(&values);
            self.lambdas.push(optimal_lambda);
        }

        self.fitted = true;
        Ok(())
    }

    /// Transform data in-place using fitted lambdas
    pub fn transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Config(
                "YeoJohnsonTransform not fitted. Call fit() first.".into(),
            ));
        }

        if self.lambdas.len() != num_features {
            return Err(TreeBoostError::Config(format!(
                "Feature count mismatch: fitted with {} features, got {}",
                self.lambdas.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        for row in 0..num_rows {
            for col in 0..num_features {
                let idx = row * num_features + col;
                if !data[idx].is_nan() {
                    data[idx] = yeo_johnson_transform(data[idx], self.lambdas[col]);
                }
            }
        }

        Ok(())
    }

    /// Inverse transform (convert back to original scale)
    pub fn inverse_transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Config(
                "YeoJohnsonTransform not fitted. Call fit() first.".into(),
            ));
        }

        if self.lambdas.len() != num_features {
            return Err(TreeBoostError::Config(format!(
                "Feature count mismatch: fitted with {} features, got {}",
                self.lambdas.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        for row in 0..num_rows {
            for col in 0..num_features {
                let idx = row * num_features + col;
                if !data[idx].is_nan() {
                    data[idx] = yeo_johnson_inverse(data[idx], self.lambdas[col]);
                }
            }
        }

        Ok(())
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &mut [f32], num_features: usize) -> Result<()> {
        self.fit(data, num_features)?;
        self.transform(data, num_features)?;
        Ok(())
    }

    /// Check if the transform has been fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the fitted lambdas
    pub fn lambdas(&self) -> &[f32] {
        &self.lambdas
    }

    /// Find optimal lambda for a single feature using golden section search
    fn find_optimal_lambda(&self, values: &[f32]) -> f32 {
        // Search bounds for lambda
        let mut a = -5.0f32;
        let mut b = 5.0f32;

        // Golden ratio
        let phi = (1.0 + 5.0f32.sqrt()) / 2.0;
        let resphi = 2.0 - phi;

        let mut x1 = a + resphi * (b - a);
        let mut x2 = b - resphi * (b - a);

        let mut f1 = -self.log_likelihood(values, x1);
        let mut f2 = -self.log_likelihood(values, x2);

        for _ in 0..self.max_iter {
            if (b - a).abs() < self.tolerance {
                break;
            }

            if f1 < f2 {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = a + resphi * (b - a);
                f1 = -self.log_likelihood(values, x1);
            } else {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = b - resphi * (b - a);
                f2 = -self.log_likelihood(values, x2);
            }
        }

        (a + b) / 2.0
    }

    /// Compute log-likelihood for a given lambda (higher is better)
    fn log_likelihood(&self, values: &[f32], lambda: f32) -> f32 {
        let n = values.len() as f32;
        if n == 0.0 {
            return f32::NEG_INFINITY;
        }

        // Transform values
        let transformed: Vec<f32> = values
            .iter()
            .map(|&x| yeo_johnson_transform(x, lambda))
            .collect();

        // Compute variance of transformed data
        let mean: f32 = transformed.iter().sum::<f32>() / n;
        let variance: f32 = transformed.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;

        if variance <= 0.0 || variance.is_nan() {
            return f32::NEG_INFINITY;
        }

        // Log-likelihood (simplified, ignoring constant terms)
        // LL = -n/2 * log(var) + (lambda - 1) * sum(sign(x) * log(|x| + 1))
        let jacobian_term: f32 = values
            .iter()
            .map(|&x| {
                let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                (lambda - 1.0) * sign * (x.abs() + 1.0).ln()
            })
            .sum();

        -0.5 * n * variance.ln() + jacobian_term
    }
}

// =============================================================================
// Transform Functions
// =============================================================================

/// Apply Yeo-Johnson transform to a single value
#[inline]
pub fn yeo_johnson_transform(x: f32, lambda: f32) -> f32 {
    if x >= 0.0 {
        if lambda.abs() > 1e-10 {
            // y = ((x + 1)^λ - 1) / λ
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        } else {
            // y = log(x + 1)
            (x + 1.0).ln()
        }
    } else {
        // x < 0
        let neg_x = -x;
        if (lambda - 2.0).abs() > 1e-10 {
            // y = -((-x + 1)^(2-λ) - 1) / (2 - λ)
            -((neg_x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
        } else {
            // y = -log(-x + 1)
            -(neg_x + 1.0).ln()
        }
    }
}

/// Apply inverse Yeo-Johnson transform to a single value
#[inline]
pub fn yeo_johnson_inverse(y: f32, lambda: f32) -> f32 {
    if y >= 0.0 {
        if lambda.abs() > 1e-10 {
            // x = (λ*y + 1)^(1/λ) - 1
            (lambda * y + 1.0).powf(1.0 / lambda) - 1.0
        } else {
            // x = exp(y) - 1
            y.exp() - 1.0
        }
    } else {
        // y < 0
        if (lambda - 2.0).abs() > 1e-10 {
            // x = 1 - ((2-λ)*(-y) + 1)^(1/(2-λ))
            1.0 - ((2.0 - lambda) * (-y) + 1.0).powf(1.0 / (2.0 - lambda))
        } else {
            // x = 1 - exp(-y)
            1.0 - (-y).exp()
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yeo_johnson_transform_positive() {
        // For positive values, should behave like Box-Cox with shift
        let x = 5.0;
        let lambda = 0.5;

        let y = yeo_johnson_transform(x, lambda);

        // Should be a positive, reasonable value
        assert!(y > 0.0);
        assert!(y.is_finite());
    }

    #[test]
    fn test_yeo_johnson_transform_negative() {
        // Should handle negative values
        let x = -5.0;
        let lambda = 0.5;

        let y = yeo_johnson_transform(x, lambda);

        // Should be negative
        assert!(y < 0.0);
        assert!(y.is_finite());
    }

    #[test]
    fn test_yeo_johnson_transform_zero_lambda() {
        // Lambda = 0 should be log transform for positive
        let x = 5.0;
        let y = yeo_johnson_transform(x, 0.0);

        let expected = (x + 1.0).ln();
        assert!((y - expected).abs() < 1e-5);
    }

    #[test]
    fn test_yeo_johnson_transform_lambda_two() {
        // Lambda = 2 should be -log for negative
        let x = -5.0;
        let y = yeo_johnson_transform(x, 2.0);

        let expected = -((-x) + 1.0).ln();
        assert!((y - expected).abs() < 1e-5);
    }

    #[test]
    fn test_yeo_johnson_inverse_positive() {
        let x = 5.0;
        let lambda = 0.5;

        let y = yeo_johnson_transform(x, lambda);
        let x_recovered = yeo_johnson_inverse(y, lambda);

        assert!((x - x_recovered).abs() < 1e-4);
    }

    #[test]
    fn test_yeo_johnson_inverse_negative() {
        let x = -5.0;
        let lambda = 0.5;

        let y = yeo_johnson_transform(x, lambda);
        let x_recovered = yeo_johnson_inverse(y, lambda);

        assert!((x - x_recovered).abs() < 1e-4);
    }

    #[test]
    fn test_yeo_johnson_fit() {
        let mut transform = YeoJohnsonTransform::new();

        // Skewed data - should find lambda that normalizes it
        let data = vec![0.1, 10.0, 1.0, 100.0, 5.0, 1000.0]; // 3 rows × 2 features

        transform.fit(&data, 2).unwrap();

        assert!(transform.is_fitted());
        assert_eq!(transform.lambdas().len(), 2);

        // Lambdas should be reasonable
        for &lambda in transform.lambdas() {
            assert!(lambda > -5.0 && lambda < 5.0);
        }
    }

    #[test]
    fn test_yeo_johnson_fit_transform() {
        let mut transform = YeoJohnsonTransform::new();

        let mut data = vec![0.1, 1.0, 10.0, 100.0]; // 2 rows × 2 features

        transform.fit_transform(&mut data, 2).unwrap();

        // Data should be transformed
        assert!(transform.is_fitted());

        // Values should be finite
        for &v in &data {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_yeo_johnson_inverse_transform() {
        let mut transform = YeoJohnsonTransform::new();

        let original = vec![0.5, 2.0, 5.0, 10.0]; // 2 rows × 2 features
        let mut data = original.clone();

        transform.fit_transform(&mut data, 2).unwrap();
        transform.inverse_transform(&mut data, 2).unwrap();

        // Should recover original (within tolerance)
        for (orig, recovered) in original.iter().zip(data.iter()) {
            assert!(
                (orig - recovered).abs() < 0.01,
                "orig={}, recovered={}",
                orig,
                recovered
            );
        }
    }

    #[test]
    fn test_yeo_johnson_with_lambdas() {
        // Use fixed lambdas
        let transform = YeoJohnsonTransform::with_lambdas(vec![0.5, 1.0]);

        assert!(transform.is_fitted());
        assert_eq!(transform.lambdas(), &[0.5, 1.0]);
    }

    #[test]
    fn test_yeo_johnson_not_fitted_error() {
        let transform = YeoJohnsonTransform::new();
        let mut data = vec![1.0, 2.0];

        let result = transform.transform(&mut data, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not fitted"));
    }

    #[test]
    fn test_yeo_johnson_with_nan() {
        let mut transform = YeoJohnsonTransform::new();

        // Data with NaN
        let mut data = vec![1.0, f32::NAN, 5.0, 10.0];

        transform.fit_transform(&mut data, 2).unwrap();

        // NaN should remain NaN
        assert!(!data[0].is_nan()); // 1.0 transformed
        assert!(data[1].is_nan()); // Still NaN
        assert!(!data[2].is_nan()); // 5.0 transformed
        assert!(!data[3].is_nan()); // 10.0 transformed
    }

    #[test]
    fn test_yeo_johnson_serialization() {
        let mut transform = YeoJohnsonTransform::new();
        transform.fit(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();

        let json = serde_json::to_string(&transform).unwrap();
        let loaded: YeoJohnsonTransform = serde_json::from_str(&json).unwrap();

        assert!(loaded.is_fitted());
        assert_eq!(loaded.lambdas(), transform.lambdas());
    }

    #[test]
    fn test_yeo_johnson_identity_lambda_one() {
        // Lambda = 1 should be close to identity-ish
        let x = 5.0;
        let y = yeo_johnson_transform(x, 1.0);

        // For lambda=1: y = (x+1)^1 - 1 = x
        assert!((y - x).abs() < 1e-5);
    }

    #[test]
    fn test_yeo_johnson_all_nan_column() {
        let mut transform = YeoJohnsonTransform::new();

        // Column 0 all NaN
        let data = vec![f32::NAN, 1.0, f32::NAN, 2.0];

        transform.fit(&data, 2).unwrap();

        // Should handle gracefully (lambda = 1.0 for NaN column)
        assert!(transform.is_fitted());
        assert_eq!(transform.lambdas().len(), 2);
    }
}
