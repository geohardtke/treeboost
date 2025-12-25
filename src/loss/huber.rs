//! Pseudo-Huber Loss function
//!
//! Smooth approximation to Huber loss that is differentiable everywhere.

use super::LossFunction;

/// Pseudo-Huber Loss
///
/// L(a) = δ² * (√(1 + (a/δ)²) - 1)  where a = y - ŷ
///
/// Properties:
/// - Smooth (twice differentiable everywhere)
/// - Behaves like L2 for small errors (efficient convergence)
/// - Behaves like L1 for large errors (robust to outliers)
/// - δ parameter controls the transition point
///
/// Gradient: a / √(1 + (a/δ)²)
/// Hessian: 1 / (1 + (a/δ)²)^(3/2)
#[derive(Debug, Clone, Copy)]
pub struct PseudoHuberLoss {
    /// Transition parameter (larger = more L2-like behavior)
    delta: f32,
    /// Precomputed delta squared
    delta_sq: f32,
}

impl Default for PseudoHuberLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl PseudoHuberLoss {
    /// Create a new Pseudo-Huber loss with given delta
    ///
    /// - Small delta (e.g., 0.1): More robust, slower convergence
    /// - Large delta (e.g., 10.0): More like MSE, faster convergence
    /// - Typical values: 1.0 to 5.0
    pub fn new(delta: f32) -> Self {
        assert!(delta > 0.0, "delta must be positive");
        Self {
            delta,
            delta_sq: delta * delta,
        }
    }

    /// Get the delta parameter
    pub fn delta(&self) -> f32 {
        self.delta
    }

    /// Compute the scaled residual squared: (a/δ)²
    #[inline]
    fn scaled_residual_sq(&self, residual: f32) -> f32 {
        (residual * residual) / self.delta_sq
    }
}

impl LossFunction for PseudoHuberLoss {
    #[inline]
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        let a = target - prediction;
        let scaled_sq = self.scaled_residual_sq(a);
        self.delta_sq * ((1.0 + scaled_sq).sqrt() - 1.0)
    }

    #[inline]
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        let a = prediction - target; // Note: gradient is negated residual direction
        let scaled_sq = self.scaled_residual_sq(a);
        a / (1.0 + scaled_sq).sqrt()
    }

    #[inline]
    fn hessian(&self, target: f32, prediction: f32) -> f32 {
        let a = target - prediction;
        let scaled_sq = self.scaled_residual_sq(a);
        let denom = (1.0 + scaled_sq).sqrt();
        1.0 / (denom * denom * denom)
    }

    #[inline]
    fn gradient_hessian(&self, target: f32, prediction: f32) -> (f32, f32) {
        let a_pos = prediction - target; // For gradient (positive direction)
        let a_neg = target - prediction; // For hessian (original residual)
        let scaled_sq = self.scaled_residual_sq(a_neg);
        let sqrt_term = (1.0 + scaled_sq).sqrt();

        let gradient = a_pos / sqrt_term;
        let hessian = 1.0 / (sqrt_term * sqrt_term * sqrt_term);

        (gradient, hessian)
    }

    fn name(&self) -> &'static str {
        "pseudo_huber"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pseudo_huber_loss() {
        let loss = PseudoHuberLoss::new(1.0);

        // Perfect prediction
        assert!((loss.loss(10.0, 10.0) - 0.0).abs() < 1e-6);

        // Small error - should be close to 0.5 * a² (like MSE)
        let l = loss.loss(10.0, 10.1);
        let mse_approx = 0.5 * 0.1 * 0.1;
        assert!((l - mse_approx).abs() < 0.01);
    }

    #[test]
    fn test_pseudo_huber_robustness() {
        let loss = PseudoHuberLoss::new(1.0);
        let mse = super::super::MseLoss::new();

        // Large error - Pseudo-Huber should grow slower than MSE
        let ph_loss = loss.loss(0.0, 100.0);
        let mse_loss = mse.loss(0.0, 100.0);

        // Pseudo-Huber loss grows approximately linearly for large errors
        // MSE grows quadratically
        assert!(ph_loss < mse_loss);
    }

    #[test]
    fn test_pseudo_huber_gradient() {
        let loss = PseudoHuberLoss::new(1.0);

        // Small error - gradient should be close to (pred - target) like MSE
        let g = loss.gradient(10.0, 10.1);
        assert!((g - 0.1).abs() < 0.01);

        // Large error - gradient is bounded by delta
        let g_large = loss.gradient(0.0, 100.0);
        // For large errors, gradient approaches sign(a) * delta = delta
        assert!(g_large < 100.0); // Much smaller than MSE gradient
        assert!(g_large > 0.9); // But still positive and close to delta
    }

    #[test]
    fn test_pseudo_huber_hessian() {
        let loss = PseudoHuberLoss::new(1.0);

        // At zero error, hessian should be 1.0
        let h = loss.hessian(10.0, 10.0);
        assert!((h - 1.0).abs() < 1e-6);

        // For large errors, hessian approaches 0
        let h_large = loss.hessian(0.0, 100.0);
        assert!(h_large < 0.01);
        assert!(h_large > 0.0);
    }

    #[test]
    fn test_delta_effect() {
        let small_delta = PseudoHuberLoss::new(0.1);
        let large_delta = PseudoHuberLoss::new(10.0);

        // With large delta, behavior is more MSE-like
        let g_small = small_delta.gradient(0.0, 5.0);
        let g_large = large_delta.gradient(0.0, 5.0);

        // Large delta should have gradient closer to MSE gradient (5.0)
        assert!((g_large - 5.0).abs() < (g_small - 5.0).abs());
    }
}
