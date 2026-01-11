//! Loss function trait definition
//!
//! Supports both single-output (scalar) and multi-output (vector) losses.
//! Single-output losses implement the required methods.
//! Multi-output losses can override the `compute_gradients_multi` method.

/// Objective function for gradient boosting
///
/// Provides first and second derivatives needed for the Newton-Raphson update.
/// The loss function must be twice differentiable for histogram-based GBDT.
///
/// # Single-Output vs Multi-Output
///
/// By default, this trait supports single-output (scalar) losses. For multi-output
/// losses (multi-label, multi-target), override the `*_multi` methods.
///
/// The data layout for multi-output is row-wise flattened:
/// `[row0_out0, row0_out1, ..., row0_outK, row1_out0, ...]`
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
        (
            self.gradient(target, prediction),
            self.hessian(target, prediction),
        )
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

    /// Compute gradients and hessians for multi-output data
    ///
    /// # Arguments
    ///
    /// * `targets` - Row-wise flattened targets, length = num_rows * num_outputs
    /// * `predictions` - Row-wise flattened predictions, same length
    /// * `gradients` - Output gradients, same length
    /// * `hessians` - Output hessians, same length
    /// * `num_outputs` - Number of output dimensions
    ///
    /// # Default Implementation
    ///
    /// For `num_outputs == 1`, delegates to `compute_gradients`.
    /// Override for true multi-output losses.
    fn compute_gradients_multi(
        &self,
        targets: &[f32],
        predictions: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
        num_outputs: usize,
    ) {
        if num_outputs == 1 {
            // Single-output: use the standard method
            self.compute_gradients(targets, predictions, gradients, hessians);
        } else {
            // Default: treat each output independently
            // This works for element-wise losses but may not be optimal
            self.compute_gradients(targets, predictions, gradients, hessians);
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

    /// Initial predictions for multi-output data
    ///
    /// # Arguments
    ///
    /// * `targets` - Row-wise flattened targets
    /// * `num_outputs` - Number of output dimensions
    ///
    /// # Returns
    ///
    /// Vector of length `num_outputs` with initial prediction per output.
    ///
    /// # Default Implementation
    ///
    /// For `num_outputs == 1`, returns a single-element vector with `initial_prediction`.
    /// Override for true multi-output losses.
    fn initial_predictions_multi(&self, targets: &[f32], num_outputs: usize) -> Vec<f32> {
        if num_outputs == 1 {
            vec![self.initial_prediction(targets)]
        } else {
            // Default: compute mean per output dimension
            if targets.is_empty() || num_outputs == 0 {
                return vec![0.0; num_outputs];
            }

            let num_rows = targets.len() / num_outputs;
            let mut initial = vec![0.0; num_outputs];

            for output_idx in 0..num_outputs {
                let mut sum = 0.0;
                for row in 0..num_rows {
                    sum += targets[row * num_outputs + output_idx];
                }
                initial[output_idx] = sum / num_rows as f32;
            }

            initial
        }
    }

    /// Number of outputs this loss function supports
    ///
    /// Returns `None` for losses that work with any number of outputs.
    /// Returns `Some(n)` for losses designed for exactly `n` outputs.
    ///
    /// Default is `Some(1)` for backward compatibility with scalar losses.
    fn num_outputs(&self) -> Option<usize> {
        Some(1)
    }

    /// Whether this loss supports multi-output (vector) targets
    ///
    /// Returns `true` if the loss can handle `num_outputs > 1`.
    fn supports_multi_output(&self) -> bool {
        self.num_outputs().is_none() || self.num_outputs() == Some(1)
    }

    /// Name of the loss function
    fn name(&self) -> &'static str;
}
