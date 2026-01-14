//! Tweedie Loss for count data and insurance claims
//!
//! Handles data with mass at zero and positive continuous values.

use super::LossFunction;

/// Tweedie Deviance Loss
///
/// The Tweedie distribution is a family of distributions useful for:
/// - Insurance claims (many zeros, positive amounts when non-zero)
/// - Count data with overdispersion
/// - Positive continuous data with point mass at zero
///
/// The variance function is V(μ) = μ^p where p is the power parameter.
///
/// # Power Parameter
///
/// - p = 0: Normal distribution (MSE)
/// - p = 1: Poisson distribution
/// - 1 < p < 2: Compound Poisson-Gamma (typical for insurance)
/// - p = 2: Gamma distribution
/// - p = 3: Inverse Gaussian
///
/// # Deviance Loss
///
/// For 1 < p < 2 (compound Poisson-Gamma):
/// D(y, μ) = 2 * (y^(2-p)/((1-p)(2-p)) - y*μ^(1-p)/(1-p) + μ^(2-p)/(2-p))
///
/// # Note
///
/// Predictions (μ) are on log scale internally: μ = exp(raw_prediction)
/// This ensures predictions are always positive.
#[derive(Debug, Clone, Copy)]
pub struct TweedieLoss {
    /// Power parameter (1 < p < 2 for compound Poisson-Gamma)
    power: f32,
}

impl Default for TweedieLoss {
    fn default() -> Self {
        // SAFETY: 1.5 is always in (1, 2)
        Self::new(1.5).unwrap()
    }
}

impl TweedieLoss {
    /// Create a Tweedie loss with the given power parameter
    ///
    /// # Arguments
    /// * `power` - Power parameter p (must be in (1, 2) for typical use)
    ///
    /// # Returns
    /// * `Result<Self>` - Returns error if power is not in (1, 2)
    ///
    /// # Errors
    /// * Returns error if `power <= 1.0` or `power >= 2.0`
    ///
    /// # Common Values
    /// - p = 1.5: Good default for insurance/count data
    /// - p = 1.1-1.3: More Poisson-like (frequent small values)
    /// - p = 1.7-1.9: More Gamma-like (less mass at zero)
    pub fn new(power: f32) -> crate::Result<Self> {
        if power <= 1.0 || power >= 2.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "power must be in (1, 2) for compound Poisson-Gamma, got {}",
                power
            )));
        }
        Ok(Self { power })
    }

    /// Compound Poisson-Gamma with p = 1.5 (common default)
    ///
    /// This is infallible since 1.5 is always valid.
    pub fn compound_poisson_gamma() -> Self {
        // SAFETY: 1.5 is always in (1, 2)
        Self::new(1.5).unwrap()
    }

    /// Get the power parameter
    pub fn power(&self) -> f32 {
        self.power
    }
}

impl LossFunction for TweedieLoss {
    #[inline]
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        // prediction is raw score, convert to mean: μ = exp(prediction)
        let mu = prediction.exp().max(1e-10);
        let y = target.max(0.0); // Tweedie targets must be >= 0
        let p = self.power;

        // Tweedie deviance for 1 < p < 2
        // D = 2 * (y^(2-p)/((1-p)(2-p)) - y*μ^(1-p)/(1-p) + μ^(2-p)/(2-p))
        let term1 = if y > 0.0 {
            y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
        } else {
            0.0
        };
        let term2 = y * mu.powf(1.0 - p) / (1.0 - p);
        let term3 = mu.powf(2.0 - p) / (2.0 - p);

        2.0 * (term1 - term2 + term3)
    }

    #[inline]
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        // μ = exp(prediction), dμ/d(pred) = μ
        // gradient = dL/d(pred) = dL/dμ * dμ/d(pred)
        //          = (μ^(1-p) - y*μ^(-p)) * μ
        //          = μ^(2-p) - y*μ^(1-p)
        let mu = prediction.exp().max(1e-10);
        let y = target.max(0.0);
        let p = self.power;

        mu.powf(2.0 - p) - y * mu.powf(1.0 - p)
    }

    #[inline]
    fn hessian(&self, target: f32, prediction: f32) -> f32 {
        // hessian = d²L/d(pred)²
        // Using chain rule and the fact that predictions are on log scale
        // H = (2-p)*μ^(2-p) - (1-p)*y*μ^(1-p)
        // For numerical stability, we use a simplified positive approximation
        let mu = prediction.exp().max(1e-10);
        let y = target.max(0.0);
        let p = self.power;

        // Simplified: variance function V(μ) = μ^p on log scale gives μ^(2-p)
        // Adding small contribution from target for stability
        let base_hessian = (2.0 - p) * mu.powf(2.0 - p);
        let target_term = (p - 1.0) * y * mu.powf(1.0 - p);

        // Ensure positive hessian
        (base_hessian + target_term).max(1e-6)
    }

    fn initial_prediction(&self, targets: &[f32]) -> f32 {
        if targets.is_empty() {
            return 0.0; // log(1) = 0
        }
        // Mean of targets on log scale
        let mean = targets.iter().map(|&t| t.max(1e-10)).sum::<f32>() / targets.len() as f32;
        mean.ln()
    }

    fn name(&self) -> &'static str {
        "tweedie"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tweedie_loss_zero_target() {
        let loss = TweedieLoss::default();

        // For zero target, loss should still be valid
        let l = loss.loss(0.0, 0.0); // μ = exp(0) = 1
        assert!(l.is_finite());
        assert!(l >= 0.0);
    }

    #[test]
    fn test_tweedie_loss_positive() {
        let loss = TweedieLoss::new(1.5).unwrap();

        // Loss should be non-negative
        let l1 = loss.loss(5.0, 1.5); // μ = exp(1.5) ≈ 4.48
        let l2 = loss.loss(5.0, 2.0); // μ = exp(2.0) ≈ 7.39

        assert!(l1 >= 0.0);
        assert!(l2 >= 0.0);
    }

    #[test]
    fn test_tweedie_gradient_direction() {
        let loss = TweedieLoss::default();

        // When μ > target, gradient should be positive (reduce prediction)
        let g_over = loss.gradient(1.0, 2.0); // μ = exp(2) ≈ 7.4 > 1
        assert!(g_over > 0.0);

        // When μ < target, gradient should be negative (increase prediction)
        let g_under = loss.gradient(100.0, 2.0); // μ = exp(2) ≈ 7.4 < 100
        assert!(g_under < 0.0);
    }

    #[test]
    fn test_tweedie_hessian_positive() {
        let loss = TweedieLoss::new(1.5).unwrap();

        // Hessian should always be positive
        assert!(loss.hessian(0.0, 0.0) > 0.0);
        assert!(loss.hessian(5.0, 1.0) > 0.0);
        assert!(loss.hessian(100.0, 3.0) > 0.0);
    }

    #[test]
    fn test_tweedie_initial_prediction() {
        let loss = TweedieLoss::default();
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let init = loss.initial_prediction(&targets);
        // Should be log of mean, approximately log(3) ≈ 1.1
        assert!(init > 0.0 && init < 2.0);
    }

    #[test]
    fn test_power_parameter() {
        let loss_15 = TweedieLoss::new(1.5).unwrap();
        let loss_12 = TweedieLoss::new(1.2).unwrap();

        assert!((loss_15.power() - 1.5).abs() < 1e-6);
        assert!((loss_12.power() - 1.2).abs() < 1e-6);
    }
}
