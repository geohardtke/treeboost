//! Loss function trait definition

/// Objective function for gradient boosting
///
/// Provides first and second derivatives needed for the Newton-Raphson update.
/// The loss function must be twice differentiable for histogram-based GBDT.
pub trait LossFunction: Send + Sync {
    /// Compute the loss value for a single prediction
    fn loss(&self, target: f32, prediction: f32) -> f32;

    /// Compute the negative gradient (residual direction)
    ///
    /// For MSE: gradient = prediction - target
    /// The tree fits the negative gradient, so we return (pred - target)
    fn gradient(&self, target: f32, prediction: f32) -> f32;

    /// Compute the second derivative (hessian)
    ///
    /// For MSE: hessian = 1.0 (constant)
    /// Used for Newton step scaling in leaf weight computation
    fn hessian(&self, target: f32, prediction: f32) -> f32;

    /// Compute gradient and hessian together (may be more efficient)
    #[inline]
    fn gradient_hessian(&self, target: f32, prediction: f32) -> (f32, f32) {
        (self.gradient(target, prediction), self.hessian(target, prediction))
    }

    /// Compute gradients and hessians for a batch of samples
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
            let (g, h) = self.gradient_hessian(targets[i], predictions[i]);
            gradients[i] = g;
            hessians[i] = h;
        }
    }

    /// Initial prediction (base score) for the ensemble
    ///
    /// For regression, typically the mean of targets.
    fn initial_prediction(&self, targets: &[f32]) -> f32 {
        if targets.is_empty() {
            return 0.0;
        }
        targets.iter().sum::<f32>() / targets.len() as f32
    }

    /// Name of the loss function
    fn name(&self) -> &'static str;
}
