//! Mean Absolute Percentage Error (MAPE) Loss
//!
//! Scale-independent loss for forecasting and regression.

use super::LossFunction;

/// Mean Absolute Percentage Error Loss
///
/// L(y, ŷ) = |y - ŷ| / |y|
///
/// Properties:
/// - Scale-independent (useful for comparing across different magnitudes)
/// - Commonly used in forecasting and demand prediction
/// - Asymmetric: penalizes over-prediction less than under-prediction for large y
/// - Undefined for y = 0 (handled with epsilon)
///
/// # Smoothed Implementation
///
/// Since MAPE is not differentiable at ŷ = y, we use a smoothed approximation
/// similar to Pseudo-Huber loss:
///
/// L_smooth = sqrt((y - ŷ)² + ε²) / max(|y|, floor)
///
/// This provides smooth gradients while preserving the scale-invariance property.
#[derive(Debug, Clone, Copy)]
pub struct MapeLoss {
    /// Smoothing parameter for numerical stability
    epsilon: f32,
    /// Floor for denominator to handle near-zero targets
    floor: f32,
}

impl Default for MapeLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl MapeLoss {
    /// Create a MAPE loss with default parameters
    pub fn new() -> Self {
        Self {
            epsilon: 1e-4,
            floor: 1e-3,
        }
    }

    /// Create a MAPE loss with custom parameters
    ///
    /// # Arguments
    /// * `epsilon` - Smoothing parameter for gradient (default: 1e-4)
    /// * `floor` - Minimum denominator to avoid division by zero (default: 1e-3)
    ///
    /// # Returns
    /// * `Result<Self>` - Returns error if validation fails
    ///
    /// # Errors
    /// * Returns error if `epsilon <= 0.0`
    /// * Returns error if `floor <= 0.0`
    pub fn with_params(epsilon: f32, floor: f32) -> crate::Result<Self> {
        if epsilon <= 0.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "epsilon must be positive, got {}",
                epsilon
            )));
        }
        if floor <= 0.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "floor must be positive, got {}",
                floor
            )));
        }
        Ok(Self { epsilon, floor })
    }

    /// Symmetric MAPE (sMAPE) variant
    ///
    /// Uses (|y| + |ŷ|) / 2 in denominator for better symmetry
    pub fn symmetric() -> SymmetricMapeLoss {
        SymmetricMapeLoss::new()
    }
}

impl LossFunction for MapeLoss {
    #[inline]
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        let residual = target - prediction;
        let abs_residual = (residual * residual + self.epsilon * self.epsilon).sqrt();
        let denominator = target.abs().max(self.floor);
        abs_residual / denominator
    }

    #[inline]
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        // d/d(pred) of sqrt((y - pred)² + ε²) / |y|
        // = -(y - pred) / (sqrt((y - pred)² + ε²) * |y|)
        // = (pred - y) / (sqrt((y - pred)² + ε²) * |y|)
        let residual = target - prediction;
        let sqrt_term = (residual * residual + self.epsilon * self.epsilon).sqrt();
        let denominator = target.abs().max(self.floor);
        -residual / (sqrt_term * denominator)
    }

    #[inline]
    fn hessian(&self, target: f32, prediction: f32) -> f32 {
        // d²/d(pred)² of sqrt((y - pred)² + ε²) / |y|
        // = ε² / ((y - pred)² + ε²)^(3/2) / |y|
        let residual = target - prediction;
        let sq_term = residual * residual + self.epsilon * self.epsilon;
        let denominator = target.abs().max(self.floor);
        self.epsilon * self.epsilon / (sq_term.sqrt() * sq_term * denominator)
    }

    #[inline]
    fn gradient_hessian(&self, target: f32, prediction: f32) -> (f32, f32) {
        let residual = target - prediction;
        let sq_term = residual * residual + self.epsilon * self.epsilon;
        let sqrt_term = sq_term.sqrt();
        let denominator = target.abs().max(self.floor);

        let gradient = -residual / (sqrt_term * denominator);
        let hessian = self.epsilon * self.epsilon / (sqrt_term * sq_term * denominator);

        (gradient, hessian)
    }

    fn initial_prediction(&self, targets: &[f32]) -> f32 {
        if targets.is_empty() {
            return 0.0;
        }
        // For MAPE, median is often a better initial guess than mean
        let mut sorted: Vec<f32> = targets.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    }

    fn name(&self) -> &'static str {
        "mape"
    }
}

/// Symmetric MAPE (sMAPE) Loss
///
/// L(y, ŷ) = |y - ŷ| / ((|y| + |ŷ|) / 2)
///
/// Properties:
/// - Bounded between 0 and 2 (unlike MAPE which is unbounded)
/// - More symmetric treatment of over/under-prediction
/// - Still undefined when both y and ŷ are zero (handled with floor)
#[derive(Debug, Clone, Copy)]
pub struct SymmetricMapeLoss {
    epsilon: f32,
    floor: f32,
}

impl Default for SymmetricMapeLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl SymmetricMapeLoss {
    pub fn new() -> Self {
        Self {
            epsilon: 1e-4,
            floor: 1e-3,
        }
    }

    /// Create a sMAPE loss with custom parameters
    ///
    /// # Arguments
    /// * `epsilon` - Smoothing parameter for gradient (default: 1e-4)
    /// * `floor` - Minimum denominator to avoid division by zero (default: 1e-3)
    ///
    /// # Returns
    /// * `Result<Self>` - Returns error if validation fails
    ///
    /// # Errors
    /// * Returns error if `epsilon <= 0.0`
    /// * Returns error if `floor <= 0.0`
    pub fn with_params(epsilon: f32, floor: f32) -> crate::Result<Self> {
        if epsilon <= 0.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "epsilon must be positive, got {}",
                epsilon
            )));
        }
        if floor <= 0.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "floor must be positive, got {}",
                floor
            )));
        }
        Ok(Self { epsilon, floor })
    }
}

impl LossFunction for SymmetricMapeLoss {
    #[inline]
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        let residual = target - prediction;
        let abs_residual = (residual * residual + self.epsilon * self.epsilon).sqrt();
        let denominator = ((target.abs() + prediction.abs()) / 2.0).max(self.floor);
        abs_residual / denominator
    }

    #[inline]
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        let residual = target - prediction;
        let sqrt_term = (residual * residual + self.epsilon * self.epsilon).sqrt();
        let sum_abs = target.abs() + prediction.abs();
        let denominator = (sum_abs / 2.0).max(self.floor);

        // Gradient has two parts due to |ŷ| in denominator
        let numerator_grad = -residual / sqrt_term;

        // d/d(pred) of 1/((|y| + |ŷ|)/2) = -sign(pred) / ((|y| + |ŷ|)/2)²
        let pred_sign = if prediction >= 0.0 { 1.0 } else { -1.0 };
        let denom_grad = -pred_sign / (2.0 * denominator * denominator);

        // Product rule: d(N/D) = (dN * D - N * dD) / D²
        // Simplified: dN/D + N * (-dD/D²) = dN/D - N * dD / D²
        let n = sqrt_term;
        numerator_grad / denominator + n * denom_grad
    }

    #[inline]
    fn hessian(&self, target: f32, prediction: f32) -> f32 {
        // Simplified hessian approximation
        let residual = target - prediction;
        let sq_term = residual * residual + self.epsilon * self.epsilon;
        let denominator = ((target.abs() + prediction.abs()) / 2.0).max(self.floor);

        // Main contribution from numerator curvature
        self.epsilon * self.epsilon / (sq_term.sqrt() * sq_term * denominator)
    }

    fn initial_prediction(&self, targets: &[f32]) -> f32 {
        if targets.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f32> = targets.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    }

    fn name(&self) -> &'static str {
        "smape"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mape_loss_scale_invariance() {
        let loss = MapeLoss::new();

        // MAPE should be similar regardless of scale
        let l1 = loss.loss(100.0, 110.0); // 10% error on scale 100
        let l2 = loss.loss(1000.0, 1100.0); // 10% error on scale 1000

        // Both should be approximately 0.1 (10%)
        assert!((l1 - l2).abs() < 0.01);
    }

    #[test]
    fn test_mape_loss_values() {
        let loss = MapeLoss::new();

        // 10% error
        let l = loss.loss(100.0, 110.0);
        assert!(l > 0.09 && l < 0.11);

        // Perfect prediction
        let l_perfect = loss.loss(100.0, 100.0);
        assert!(l_perfect < 0.01);
    }

    #[test]
    fn test_mape_gradient_direction() {
        let loss = MapeLoss::new();

        // Over-prediction: gradient should be positive
        let g_over = loss.gradient(100.0, 120.0);
        assert!(g_over > 0.0);

        // Under-prediction: gradient should be negative
        let g_under = loss.gradient(100.0, 80.0);
        assert!(g_under < 0.0);
    }

    #[test]
    fn test_mape_hessian_positive() {
        let loss = MapeLoss::new();

        assert!(loss.hessian(100.0, 100.0) > 0.0);
        assert!(loss.hessian(100.0, 120.0) > 0.0);
        assert!(loss.hessian(100.0, 80.0) > 0.0);
    }

    #[test]
    fn test_mape_near_zero_target() {
        let loss = MapeLoss::new();

        // Should handle near-zero targets without exploding
        let l = loss.loss(0.001, 0.002);
        assert!(l.is_finite());
        assert!(l >= 0.0);

        let g = loss.gradient(0.001, 0.002);
        assert!(g.is_finite());
    }

    #[test]
    fn test_smape_bounded() {
        let loss = SymmetricMapeLoss::new();

        // sMAPE should be bounded (approximately 0 to 2)
        let l1 = loss.loss(100.0, 0.01);
        let l2 = loss.loss(0.01, 100.0);

        assert!(l1 < 3.0); // Should be close to 2
        assert!(l2 < 3.0);
    }

    #[test]
    fn test_smape_symmetry() {
        let loss = SymmetricMapeLoss::new();

        // sMAPE should be more symmetric than MAPE
        let l1 = loss.loss(100.0, 50.0);
        let l2 = loss.loss(50.0, 100.0);

        // These should be similar (though not exactly equal due to smoothing)
        assert!((l1 - l2).abs() < 0.2);
    }

    #[test]
    fn test_initial_prediction() {
        let loss = MapeLoss::new();
        let targets = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        // Should return median
        let init = loss.initial_prediction(&targets);
        assert!((init - 30.0).abs() < 1e-6);
    }
}
