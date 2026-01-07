//! Binary Log Loss (Cross-Entropy) for binary classification

use super::activation::sigmoid;
use super::LossFunction;

/// Binary Log Loss (Binary Cross-Entropy)
///
/// For binary classification with targets in {0, 1}.
///
/// L(y, ŷ) = -[y * log(p) + (1-y) * log(1-p)]
///
/// where p = sigmoid(ŷ) = 1 / (1 + exp(-ŷ))
///
/// Properties:
/// - Gradient: p - y (where p = sigmoid(raw_prediction))
/// - Hessian: p * (1 - p)
/// - Initial prediction: log(mean(y) / (1 - mean(y))) (log-odds)
///
/// # Example
///
/// ```
/// use treeboost::loss::BinaryLogLoss;
/// use treeboost::{GBDTConfig, GBDTModel};
///
/// let config = GBDTConfig::new()
///     .with_binary_logloss()
///     .with_num_rounds(100);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryLogLoss {
    /// Small epsilon for numerical stability
    eps: f32,
}

impl BinaryLogLoss {
    /// Create a new Binary Log Loss
    pub fn new() -> Self {
        Self { eps: 1e-7 }
    }

    /// Create with custom epsilon for numerical stability
    pub fn with_eps(eps: f32) -> Self {
        Self { eps }
    }

    /// Convert raw prediction to probability via sigmoid
    #[inline]
    pub fn to_probability(&self, raw: f32) -> f32 {
        sigmoid(raw)
    }

    /// Convert probability to class (0 or 1) with threshold
    #[inline]
    pub fn to_class(&self, prob: f32, threshold: f32) -> u32 {
        if prob >= threshold {
            1
        } else {
            0
        }
    }
}

impl LossFunction for BinaryLogLoss {
    #[inline]
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        let p = sigmoid(prediction).clamp(self.eps, 1.0 - self.eps);
        -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
    }

    #[inline]
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        // gradient = p - y where p = sigmoid(prediction)
        sigmoid(prediction) - target
    }

    #[inline]
    fn hessian(&self, _target: f32, prediction: f32) -> f32 {
        // hessian = p * (1 - p)
        let p = sigmoid(prediction);
        let h = p * (1.0 - p);
        // Clip hessian to avoid division issues in leaf weight computation
        h.max(self.eps)
    }

    #[inline]
    fn gradient_hessian(&self, target: f32, prediction: f32) -> (f32, f32) {
        let p = sigmoid(prediction);
        let g = p - target;
        let h = (p * (1.0 - p)).max(self.eps);
        (g, h)
    }

    /// Compute gradients and hessians for a batch (optimized)
    fn compute_gradients(
        &self,
        targets: &[f32],
        predictions: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        debug_assert_eq!(targets.len(), predictions.len());
        debug_assert_eq!(targets.len(), gradients.len());
        debug_assert_eq!(targets.len(), hessians.len());

        for i in 0..targets.len() {
            let p = sigmoid(predictions[i]);
            gradients[i] = p - targets[i];
            hessians[i] = (p * (1.0 - p)).max(self.eps);
        }
    }

    /// Initial prediction is log-odds of positive class
    fn initial_prediction(&self, targets: &[f32]) -> f32 {
        if targets.is_empty() {
            return 0.0;
        }

        let positive: f32 = targets.iter().filter(|&&t| t > 0.5).count() as f32;
        let total = targets.len() as f32;
        let p = (positive / total).clamp(self.eps, 1.0 - self.eps);

        // log-odds: log(p / (1 - p))
        (p / (1.0 - p)).ln()
    }

    fn name(&self) -> &'static str {
        "binary_logloss"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss() {
        let loss = BinaryLogLoss::new();

        // Perfect prediction for positive class
        // When target=1 and p≈1, loss ≈ 0
        assert!(loss.loss(1.0, 10.0) < 0.001);

        // Perfect prediction for negative class
        // When target=0 and p≈0, loss ≈ 0
        assert!(loss.loss(0.0, -10.0) < 0.001);

        // Wrong prediction is penalized
        // When target=1 and p≈0, loss is high
        assert!(loss.loss(1.0, -10.0) > 5.0);

        // When target=0 and p≈1, loss is high
        assert!(loss.loss(0.0, 10.0) > 5.0);
    }

    #[test]
    fn test_gradient() {
        let loss = BinaryLogLoss::new();

        // When prediction = 0 (p = 0.5)
        // gradient = 0.5 - target
        assert!((loss.gradient(1.0, 0.0) - (-0.5)).abs() < 1e-6);
        assert!((loss.gradient(0.0, 0.0) - 0.5).abs() < 1e-6);

        // Large positive prediction (p ≈ 1)
        // For target=1: gradient ≈ 0 (correct prediction)
        assert!(loss.gradient(1.0, 10.0).abs() < 0.001);
        // For target=0: gradient ≈ 1 (wrong prediction)
        assert!((loss.gradient(0.0, 10.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hessian() {
        let loss = BinaryLogLoss::new();

        // Hessian = p * (1 - p)
        // Maximum at p = 0.5 (prediction = 0)
        assert!((loss.hessian(0.0, 0.0) - 0.25).abs() < 1e-6);

        // Minimum near 0 at extreme predictions
        assert!(loss.hessian(0.0, 10.0) < 0.01);
        assert!(loss.hessian(0.0, -10.0) < 0.01);

        // Hessian is always positive (clipped)
        assert!(loss.hessian(0.0, 100.0) > 0.0);
    }

    #[test]
    fn test_initial_prediction() {
        let loss = BinaryLogLoss::new();

        // 50% positive: log-odds = 0
        let targets = vec![0.0, 1.0, 0.0, 1.0];
        assert!(loss.initial_prediction(&targets).abs() < 1e-6);

        // 75% positive: log-odds = log(0.75/0.25) = log(3) ≈ 1.1
        let targets = vec![1.0, 1.0, 1.0, 0.0];
        assert!((loss.initial_prediction(&targets) - 3.0_f32.ln()).abs() < 1e-5);

        // 25% positive: log-odds = log(0.25/0.75) = log(1/3) ≈ -1.1
        let targets = vec![0.0, 0.0, 0.0, 1.0];
        assert!((loss.initial_prediction(&targets) - (1.0_f32 / 3.0).ln()).abs() < 1e-5);
    }

    #[test]
    fn test_to_probability() {
        let loss = BinaryLogLoss::new();

        assert!((loss.to_probability(0.0) - 0.5).abs() < 1e-6);
        assert!(loss.to_probability(10.0) > 0.99);
        assert!(loss.to_probability(-10.0) < 0.01);
    }

    #[test]
    fn test_to_class() {
        let loss = BinaryLogLoss::new();

        assert_eq!(loss.to_class(0.6, 0.5), 1);
        assert_eq!(loss.to_class(0.4, 0.5), 0);
        assert_eq!(loss.to_class(0.5, 0.5), 1);

        // Custom threshold
        assert_eq!(loss.to_class(0.6, 0.7), 0);
        assert_eq!(loss.to_class(0.8, 0.7), 1);
    }

    #[test]
    fn test_numerical_stability() {
        let loss = BinaryLogLoss::new();

        // Extreme predictions should not cause NaN or Inf
        let extreme_preds = [-1000.0, -100.0, 100.0, 1000.0];

        for pred in extreme_preds {
            let l = loss.loss(0.0, pred);
            let g = loss.gradient(0.0, pred);
            let h = loss.hessian(0.0, pred);

            assert!(l.is_finite(), "Loss not finite for pred={}", pred);
            assert!(g.is_finite(), "Gradient not finite for pred={}", pred);
            assert!(h.is_finite(), "Hessian not finite for pred={}", pred);
            assert!(h > 0.0, "Hessian not positive for pred={}", pred);
        }
    }
}
