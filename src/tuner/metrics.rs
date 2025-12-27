//! Evaluation metrics for hyperparameter tuning
//!
//! Provides metrics for assessing model performance during tuning.
//! All metrics are computed in a numerically stable manner.

use crate::loss::LossFunction;

/// Metric types for evaluating model performance
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    /// Mean Squared Error (regression)
    Mse,
    /// Root Mean Squared Error (regression)
    Rmse,
    /// Mean Absolute Error (regression)
    Mae,
    /// Binary log loss (binary classification)
    BinaryLogLoss,
    /// Multi-class log loss (multi-class classification)
    MultiClassLogLoss { n_classes: usize },
    /// Accuracy (classification)
    Accuracy { threshold: f32 },
}

impl Default for Metric {
    fn default() -> Self {
        Metric::Mse
    }
}

impl Metric {
    /// Create MSE metric
    pub fn mse() -> Self {
        Metric::Mse
    }

    /// Create RMSE metric
    pub fn rmse() -> Self {
        Metric::Rmse
    }

    /// Create MAE metric
    pub fn mae() -> Self {
        Metric::Mae
    }

    /// Create binary log loss metric
    pub fn binary_log_loss() -> Self {
        Metric::BinaryLogLoss
    }

    /// Create multi-class log loss metric
    pub fn multi_class_log_loss(n_classes: usize) -> Self {
        Metric::MultiClassLogLoss { n_classes }
    }

    /// Create accuracy metric with default threshold (0.5)
    pub fn accuracy() -> Self {
        Metric::Accuracy { threshold: 0.5 }
    }

    /// Create accuracy metric with custom threshold
    pub fn accuracy_with_threshold(threshold: f32) -> Self {
        Metric::Accuracy { threshold }
    }

    /// Auto-select metric from loss function type
    pub fn from_loss_type(loss: &dyn LossFunction) -> Self {
        let name = loss.name();
        match name {
            "mse" | "pseudo_huber" => Metric::Mse,
            "binary_log_loss" => Metric::BinaryLogLoss,
            "multi_class_log_loss" => Metric::MultiClassLogLoss { n_classes: 2 },
            _ => Metric::Mse,
        }
    }

    /// Whether lower values are better for this metric
    pub fn lower_is_better(&self) -> bool {
        match self {
            Metric::Mse | Metric::Rmse | Metric::Mae => true,
            Metric::BinaryLogLoss | Metric::MultiClassLogLoss { .. } => true,
            Metric::Accuracy { .. } => false,
        }
    }

    /// Compute the metric value
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (raw scores for classification)
    ///   - For regression/binary: one prediction per sample
    ///   - For multi-class: n_classes predictions per sample (logits)
    /// * `targets` - Ground truth values (0/1 for binary, class indices for multi)
    ///
    /// # Returns
    /// The metric value, or f32::INFINITY on error
    pub fn compute(&self, predictions: &[f32], targets: &[f32]) -> f32 {
        if targets.is_empty() {
            return f32::INFINITY;
        }

        // For multi-class, predictions has n_samples * n_classes elements
        // For other metrics, predictions.len() == targets.len()
        match self {
            Metric::MultiClassLogLoss { n_classes } => {
                // Multi-class: predictions has n_samples * n_classes elements
                if predictions.len() != targets.len() * n_classes {
                    return f32::INFINITY;
                }
                compute_multi_class_log_loss(predictions, targets, *n_classes)
            }
            _ => {
                // Other metrics: predictions and targets have same length
                if predictions.len() != targets.len() {
                    return f32::INFINITY;
                }
                match self {
                    Metric::Mse => compute_mse(predictions, targets),
                    Metric::Rmse => compute_rmse(predictions, targets),
                    Metric::Mae => compute_mae(predictions, targets),
                    Metric::BinaryLogLoss => compute_binary_log_loss(predictions, targets),
                    Metric::Accuracy { threshold } => compute_accuracy(predictions, targets, *threshold),
                    Metric::MultiClassLogLoss { .. } => unreachable!(),
                }
            }
        }
    }

    /// Return the name of the metric
    pub fn name(&self) -> &'static str {
        match self {
            Metric::Mse => "mse",
            Metric::Rmse => "rmse",
            Metric::Mae => "mae",
            Metric::BinaryLogLoss => "binary_log_loss",
            Metric::MultiClassLogLoss { .. } => "multi_class_log_loss",
            Metric::Accuracy { .. } => "accuracy",
        }
    }
}

/// Compute Mean Squared Error
fn compute_mse(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = predictions.len() as f32;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / n
}

/// Compute Root Mean Squared Error
fn compute_rmse(predictions: &[f32], targets: &[f32]) -> f32 {
    compute_mse(predictions, targets).sqrt()
}

/// Compute Mean Absolute Error
fn compute_mae(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = predictions.len() as f32;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f32>()
        / n
}

/// Compute binary log loss (cross-entropy)
///
/// Uses numerically stable implementation with clamping to avoid log(0)
fn compute_binary_log_loss(predictions: &[f32], targets: &[f32]) -> f32 {
    const EPSILON: f32 = 1e-7;

    let n = predictions.len() as f32;
    let sum: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| {
            // Apply sigmoid to raw predictions
            let prob = sigmoid(pred);
            // Clamp to avoid log(0)
            let prob = prob.clamp(EPSILON, 1.0 - EPSILON);
            // Binary cross-entropy
            -(target * prob.ln() + (1.0 - target) * (1.0 - prob).ln())
        })
        .sum();

    sum / n
}

/// Compute multi-class log loss
///
/// Predictions should be arranged as: [class0_sample0, class1_sample0, ..., class0_sample1, ...]
fn compute_multi_class_log_loss(predictions: &[f32], targets: &[f32], n_classes: usize) -> f32 {
    if n_classes < 2 {
        return f32::INFINITY;
    }

    const EPSILON: f32 = 1e-7;

    let n_samples = targets.len();
    if predictions.len() != n_samples * n_classes {
        return f32::INFINITY;
    }

    let mut sum = 0.0f32;

    for (i, &target) in targets.iter().enumerate() {
        let class_idx = target as usize;
        if class_idx >= n_classes {
            return f32::INFINITY;
        }

        // Get logits for this sample
        let logits = &predictions[i * n_classes..(i + 1) * n_classes];

        // Softmax with numerical stability (log-sum-exp trick)
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_prob = logits[class_idx] - max_logit - exp_sum.ln();

        // Clamp for numerical stability
        sum -= log_prob.max(EPSILON.ln());
    }

    sum / n_samples as f32
}

/// Compute accuracy
fn compute_accuracy(predictions: &[f32], targets: &[f32], threshold: f32) -> f32 {
    let n = predictions.len() as f32;
    let correct: usize = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| {
            // Apply sigmoid to get probability
            let prob = sigmoid(pred);
            let predicted_class = if prob >= threshold { 1.0 } else { 0.0 };
            if (predicted_class - target).abs() < 0.5 {
                1
            } else {
                0
            }
        })
        .sum();

    correct as f32 / n
}

/// Sigmoid function
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_lower_is_better() {
        assert!(Metric::Mse.lower_is_better());
        assert!(Metric::Rmse.lower_is_better());
        assert!(Metric::Mae.lower_is_better());
        assert!(Metric::BinaryLogLoss.lower_is_better());
        assert!(Metric::multi_class_log_loss(3).lower_is_better());
        assert!(!Metric::accuracy().lower_is_better());
    }

    #[test]
    fn test_mse() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let mse = Metric::Mse.compute(&predictions, &targets);
        assert!((mse - 0.0).abs() < 1e-6);

        let predictions = vec![2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let mse = Metric::Mse.compute(&predictions, &targets);
        assert!((mse - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rmse() {
        let predictions = vec![2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let rmse = Metric::Rmse.compute(&predictions, &targets);
        assert!((rmse - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mae() {
        let predictions = vec![2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let mae = Metric::Mae.compute(&predictions, &targets);
        assert!((mae - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_log_loss() {
        // Perfect predictions should have very low loss
        let predictions = vec![10.0, 10.0, -10.0, -10.0]; // After sigmoid: ~1, ~1, ~0, ~0
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let loss = Metric::BinaryLogLoss.compute(&predictions, &targets);
        assert!(loss < 0.001);

        // Wrong predictions should have high loss
        let predictions = vec![-10.0, -10.0, 10.0, 10.0]; // After sigmoid: ~0, ~0, ~1, ~1
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let loss = Metric::BinaryLogLoss.compute(&predictions, &targets);
        assert!(loss > 5.0);
    }

    #[test]
    fn test_binary_log_loss_numerical_stability() {
        // Extreme values should not produce NaN or Inf
        let predictions = vec![1000.0, -1000.0, 0.0];
        let targets = vec![1.0, 0.0, 0.5];
        let loss = Metric::BinaryLogLoss.compute(&predictions, &targets);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_multi_class_log_loss() {
        // 3 classes, 2 samples
        // Sample 0: true class = 0, logits = [10, 0, 0] -> should predict class 0
        // Sample 1: true class = 2, logits = [0, 0, 10] -> should predict class 2
        let predictions = vec![
            10.0, 0.0, 0.0, // Sample 0
            0.0, 0.0, 10.0, // Sample 1
        ];
        let targets = vec![0.0, 2.0];
        let loss = Metric::multi_class_log_loss(3).compute(&predictions, &targets);
        assert!(loss < 0.001, "Expected loss < 0.001, got {}", loss);

        // Wrong predictions
        let predictions = vec![
            0.0, 0.0, 10.0, // Sample 0: predicts class 2
            10.0, 0.0, 0.0, // Sample 1: predicts class 0
        ];
        let targets = vec![0.0, 2.0];
        let loss = Metric::multi_class_log_loss(3).compute(&predictions, &targets);
        assert!(loss > 5.0);
    }

    #[test]
    fn test_accuracy() {
        // Perfect predictions
        let predictions = vec![10.0, 10.0, -10.0, -10.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let acc = Metric::accuracy().compute(&predictions, &targets);
        assert!((acc - 1.0).abs() < 1e-6);

        // All wrong
        let predictions = vec![-10.0, -10.0, 10.0, 10.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let acc = Metric::accuracy().compute(&predictions, &targets);
        assert!((acc - 0.0).abs() < 1e-6);

        // 50% correct
        let predictions = vec![10.0, -10.0, 10.0, -10.0];
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let acc = Metric::accuracy().compute(&predictions, &targets);
        assert!((acc - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<f32> = vec![];
        assert_eq!(Metric::Mse.compute(&empty, &empty), f32::INFINITY);
    }

    #[test]
    fn test_mismatched_lengths() {
        let predictions = vec![1.0, 2.0];
        let targets = vec![1.0];
        assert_eq!(Metric::Mse.compute(&predictions, &targets), f32::INFINITY);
    }

    #[test]
    fn test_metric_name() {
        assert_eq!(Metric::Mse.name(), "mse");
        assert_eq!(Metric::Rmse.name(), "rmse");
        assert_eq!(Metric::Mae.name(), "mae");
        assert_eq!(Metric::BinaryLogLoss.name(), "binary_log_loss");
        assert_eq!(Metric::multi_class_log_loss(3).name(), "multi_class_log_loss");
        assert_eq!(Metric::accuracy().name(), "accuracy");
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
        // Test numerical stability with extreme values
        assert!(sigmoid(1000.0).is_finite());
        assert!(sigmoid(-1000.0).is_finite());
    }
}
