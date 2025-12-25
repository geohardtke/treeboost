//! Mean Squared Error loss function

use super::LossFunction;

/// Mean Squared Error (L2) loss
///
/// L(y, ŷ) = 0.5 * (y - ŷ)²
///
/// Properties:
/// - Gradient: ŷ - y
/// - Hessian: 1.0 (constant)
/// - Sensitive to outliers (gradient grows linearly with error)
///
/// Use `PseudoHuberLoss` for data with outliers.
#[derive(Debug, Clone, Copy, Default)]
pub struct MseLoss;

impl MseLoss {
    pub fn new() -> Self {
        Self
    }
}

impl LossFunction for MseLoss {
    #[inline]
    fn loss(&self, target: f32, prediction: f32) -> f32 {
        let residual = target - prediction;
        0.5 * residual * residual
    }

    #[inline]
    fn gradient(&self, target: f32, prediction: f32) -> f32 {
        prediction - target
    }

    #[inline]
    fn hessian(&self, _target: f32, _prediction: f32) -> f32 {
        1.0
    }

    #[inline]
    fn gradient_hessian(&self, target: f32, prediction: f32) -> (f32, f32) {
        (prediction - target, 1.0)
    }

    fn name(&self) -> &'static str {
        "mse"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let loss = MseLoss::new();

        // Perfect prediction
        assert_eq!(loss.loss(10.0, 10.0), 0.0);

        // Error of 2
        assert_eq!(loss.loss(10.0, 12.0), 2.0); // 0.5 * 4 = 2
        assert_eq!(loss.loss(12.0, 10.0), 2.0);
    }

    #[test]
    fn test_mse_gradient() {
        let loss = MseLoss::new();

        // Gradient points toward reducing prediction when pred > target
        assert_eq!(loss.gradient(10.0, 12.0), 2.0);
        // Gradient points toward increasing prediction when pred < target
        assert_eq!(loss.gradient(10.0, 8.0), -2.0);
    }

    #[test]
    fn test_mse_hessian() {
        let loss = MseLoss::new();

        // Hessian is constant
        assert_eq!(loss.hessian(10.0, 12.0), 1.0);
        assert_eq!(loss.hessian(0.0, 100.0), 1.0);
    }

    #[test]
    fn test_initial_prediction() {
        let loss = MseLoss::new();
        let targets = vec![10.0, 20.0, 30.0];

        assert_eq!(loss.initial_prediction(&targets), 20.0);
    }
}
