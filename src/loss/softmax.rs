//! Multi-class LogLoss (Softmax Cross-Entropy) for multi-class classification
//!
//! This loss function is used for multi-class classification problems where
//! the target is one of K mutually exclusive classes.
//!
//! For multi-class GBDT, we train K trees per round (one per class).
//! Each tree predicts the gradient for its class.
//!
//! # Math
//!
//! For K classes with raw predictions `f_k` for class k:
//! - `p_k = exp(f_k) / sum(exp(f_j))` (softmax)
//! - `loss = -sum(y_k * log(p_k))` where y_k is 1 for true class
//! - `gradient_k = p_k - y_k`
//! - `hessian_k = p_k * (1 - p_k)`

/// Softmax function with numerical stability
///
/// Subtracts max value before exp to avoid overflow.
/// Returns probabilities for each class.
#[inline]
pub fn softmax(raw_scores: &[f32]) -> Vec<f32> {
    if raw_scores.is_empty() {
        return vec![];
    }

    // Find max for numerical stability
    let max_score = raw_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) for each class
    let exp_scores: Vec<f32> = raw_scores.iter().map(|&x| (x - max_score).exp()).collect();

    // Sum of all exp scores
    let sum_exp: f32 = exp_scores.iter().sum();

    // Normalize to get probabilities
    exp_scores.iter().map(|&e| e / sum_exp).collect()
}

/// Multi-class LogLoss (Softmax Cross-Entropy)
///
/// Used for multi-class classification where targets are class indices (0, 1, ..., K-1).
///
/// # Note
///
/// This loss function works differently from regression/binary losses:
/// - `gradient()` and `hessian()` compute for a single class given all raw predictions
/// - Training builds K trees per round (one per class)
/// - `compute_gradient_for_class()` is the primary method for multi-class training
#[derive(Debug, Clone)]
pub struct MultiClassLogLoss {
    /// Number of classes
    pub num_classes: usize,
    /// Epsilon for numerical stability in hessian
    eps: f32,
}

impl MultiClassLogLoss {
    /// Create a new multi-class log loss
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            eps: 1e-7,
        }
    }

    /// Compute gradient and hessian for a specific class
    ///
    /// # Arguments
    /// * `target_class` - The true class label (0 to K-1)
    /// * `class_idx` - The class we're computing gradient for
    /// * `raw_predictions` - Raw predictions for all K classes
    ///
    /// # Returns
    /// (gradient, hessian) for the specified class
    #[inline]
    pub fn gradient_hessian_for_class(
        &self,
        target_class: usize,
        class_idx: usize,
        raw_predictions: &[f32],
    ) -> (f32, f32) {
        let probs = softmax(raw_predictions);
        let p = probs[class_idx];

        // y_k is 1 if this is the true class, 0 otherwise
        let y = if target_class == class_idx { 1.0 } else { 0.0 };

        let gradient = p - y;
        let hessian = (p * (1.0 - p)).max(self.eps);

        (gradient, hessian)
    }

    /// Compute gradients and hessians for all classes at once
    ///
    /// More efficient when you need all K gradients for a sample.
    #[inline]
    pub fn gradient_hessian_all_classes(
        &self,
        target_class: usize,
        raw_predictions: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let probs = softmax(raw_predictions);
        let mut gradients = Vec::with_capacity(self.num_classes);
        let mut hessians = Vec::with_capacity(self.num_classes);

        for (k, &p) in probs.iter().enumerate() {
            let y = if target_class == k { 1.0 } else { 0.0 };
            gradients.push(p - y);
            hessians.push((p * (1.0 - p)).max(self.eps));
        }

        (gradients, hessians)
    }

    /// Compute gradients and hessians for a batch of samples for a specific class
    ///
    /// This is optimized for the training loop where we compute gradients for all
    /// training samples for one class at a time.
    ///
    /// # Arguments
    /// * `class_idx` - The class we're computing gradients for
    /// * `targets` - Target class indices for all samples
    /// * `predictions` - Flattened predictions matrix: predictions[sample * num_classes + class]
    /// * `sample_indices` - Indices of samples to process
    /// * `gradients` - Output buffer for gradients (indexed by sample)
    /// * `hessians` - Output buffer for hessians (indexed by sample)
    pub fn compute_gradients_batch(
        &self,
        class_idx: usize,
        targets: &[f32],
        predictions: &[f32],
        sample_indices: &[usize],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        let num_classes = self.num_classes;
        let eps = self.eps;

        for &idx in sample_indices {
            let target_class = targets[idx] as usize;
            let row_start = idx * num_classes;
            let row_preds = &predictions[row_start..row_start + num_classes];

            // Compute softmax for this row
            let probs = softmax(row_preds);
            let p = probs[class_idx];

            // y_k is 1 if this is the true class, 0 otherwise
            let y = if target_class == class_idx { 1.0 } else { 0.0 };

            gradients[idx] = p - y;
            hessians[idx] = (p * (1.0 - p)).max(eps);
        }
    }

    /// Compute initial predictions for all classes (log of class priors)
    ///
    /// Uses the class frequencies as initial probabilities.
    /// Sets f_k = log(p_k) so that softmax(f) = class frequencies.
    ///
    /// # Math
    /// If f_k = log(p_k), then:
    /// - exp(f_k) = p_k
    /// - softmax_k = p_k / sum(p_j) = p_k / 1 = p_k
    ///
    /// This ensures the model starts with predictions matching the class distribution.
    pub fn initial_predictions(&self, targets: &[f32]) -> Vec<f32> {
        let mut class_counts = vec![0usize; self.num_classes];

        for &t in targets {
            let class_idx = t as usize;
            if class_idx < self.num_classes {
                class_counts[class_idx] += 1;
            }
        }

        let n = targets.len() as f32;

        // Compute log(p_k) for each class
        // This ensures softmax(initial_predictions) = class frequencies
        class_counts
            .iter()
            .map(|&count| {
                let p = (count as f32 / n).clamp(self.eps, 1.0 - self.eps);
                p.ln()
            })
            .collect()
    }

    /// Get number of classes
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

// Note: MultiClassLogLoss does NOT implement LossFunction trait because:
// 1. Per-sample loss/gradient/hessian methods don't make sense for multi-class
// 2. Multi-class requires all K class predictions to compute gradients
// 3. The training loop uses gradient_hessian_for_class() and gradient_hessian_all_classes() directly
//
// If you need the LossFunction trait for type compatibility, use a wrapper or
// restructure your code to handle multi-class separately.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let scores = vec![1.0, 2.0, 3.0];
        let probs = softmax(&scores);

        // Probabilities should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher score should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without stability trick
        let scores = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&scores);

        // Should not be NaN or Inf
        for p in &probs {
            assert!(p.is_finite());
            assert!(*p >= 0.0 && *p <= 1.0);
        }

        // Probabilities should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_for_true_class() {
        let loss = MultiClassLogLoss::new(3);
        let raw = vec![1.0, 2.0, 0.5];

        // Gradient for true class should be p_k - 1 (negative)
        let (grad, hess) = loss.gradient_hessian_for_class(1, 1, &raw);

        let probs = softmax(&raw);
        assert!((grad - (probs[1] - 1.0)).abs() < 1e-6);
        assert!(hess > 0.0);
    }

    #[test]
    fn test_gradient_for_other_class() {
        let loss = MultiClassLogLoss::new(3);
        let raw = vec![1.0, 2.0, 0.5];

        // Gradient for non-true class should be p_k (positive)
        let (grad, _hess) = loss.gradient_hessian_for_class(1, 0, &raw);

        let probs = softmax(&raw);
        assert!((grad - probs[0]).abs() < 1e-6);
    }

    #[test]
    fn test_all_classes_gradient() {
        let loss = MultiClassLogLoss::new(3);
        let raw = vec![1.0, 2.0, 0.5];
        let target_class = 1;

        let (grads, hess) = loss.gradient_hessian_all_classes(target_class, &raw);

        assert_eq!(grads.len(), 3);
        assert_eq!(hess.len(), 3);

        // Sum of gradients should be 0 (since sum of probs = 1 and one y = 1)
        let grad_sum: f32 = grads.iter().sum();
        assert!(grad_sum.abs() < 1e-6);
    }

    #[test]
    fn test_initial_predictions() {
        let loss = MultiClassLogLoss::new(3);
        // 50% class 0, 30% class 1, 20% class 2
        let targets: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, // 5 class 0
            1.0, 1.0, 1.0,           // 3 class 1
            2.0, 2.0,                // 2 class 2
        ];

        let init = loss.initial_predictions(&targets);

        assert_eq!(init.len(), 3);
        // Class 0 should have highest initial prediction (most samples)
        assert!(init[0] > init[1]);
        assert!(init[1] > init[2]);
    }
}
