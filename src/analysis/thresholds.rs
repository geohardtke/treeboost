//! Threshold Tuning for Multi-Label Classification
//!
//! This module provides threshold optimization for multi-label models.
//! Instead of using the default 0.5 threshold, we sweep thresholds to
//! maximize F1 score per label on a validation set.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::analysis::thresholds::{ThresholdTuner, TuneResult};
//!
//! let tuner = ThresholdTuner::new();
//! let result = tuner.tune(&probabilities, &targets, num_labels);
//!
//! // Use optimal thresholds for prediction
//! for (label_idx, &threshold) in result.thresholds.iter().enumerate() {
//!     println!("Label {}: optimal threshold = {:.2}", label_idx, threshold);
//! }
//! ```
//!
//! # Leakage Warning
//!
//! Always use a **held-out validation set** for threshold tuning, never the training set.
//! Using training data to tune thresholds causes data leakage and overfit thresholds.

use crate::Result;

/// Configuration for threshold tuning
#[derive(Debug, Clone)]
pub struct ThresholdTunerConfig {
    /// Minimum threshold to try
    pub min_threshold: f32,
    /// Maximum threshold to try
    pub max_threshold: f32,
    /// Step size for threshold sweep
    pub step: f32,
}

impl Default for ThresholdTunerConfig {
    fn default() -> Self {
        Self {
            min_threshold: 0.01,
            max_threshold: 0.99,
            step: 0.01,
        }
    }
}

/// Result of threshold tuning
#[derive(Debug, Clone)]
pub struct TuneResult {
    /// Optimal threshold per label
    pub thresholds: Vec<f32>,
    /// Best F1 score per label
    pub f1_scores: Vec<f32>,
    /// Precision at optimal threshold per label
    pub precisions: Vec<f32>,
    /// Recall at optimal threshold per label
    pub recalls: Vec<f32>,
}

/// Threshold tuner for multi-label classification
#[derive(Debug, Clone)]
pub struct ThresholdTuner {
    config: ThresholdTunerConfig,
}

impl ThresholdTuner {
    /// Create a new threshold tuner with default config
    pub fn new() -> Self {
        Self {
            config: ThresholdTunerConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ThresholdTunerConfig) -> Self {
        Self { config }
    }

    /// Tune thresholds for each label to maximize F1 score
    ///
    /// # Arguments
    /// * `probabilities` - Predicted probabilities, shape: `[num_samples][num_labels]`
    /// * `targets` - True binary labels, shape: `[num_samples][num_labels]`
    /// * `num_labels` - Number of labels
    ///
    /// # Returns
    /// `TuneResult` containing optimal thresholds and metrics per label
    pub fn tune(
        &self,
        probabilities: &[Vec<f32>],
        targets: &[Vec<f32>],
        num_labels: usize,
    ) -> Result<TuneResult> {
        if probabilities.is_empty() || targets.is_empty() {
            return Err(crate::TreeBoostError::Data(
                "Empty probabilities or targets".to_string(),
            ));
        }

        if probabilities.len() != targets.len() {
            return Err(crate::TreeBoostError::Data(format!(
                "Mismatched lengths: probabilities={}, targets={}",
                probabilities.len(),
                targets.len()
            )));
        }

        let _num_samples = probabilities.len();

        let mut thresholds = Vec::with_capacity(num_labels);
        let mut f1_scores = Vec::with_capacity(num_labels);
        let mut precisions = Vec::with_capacity(num_labels);
        let mut recalls = Vec::with_capacity(num_labels);

        for k in 0..num_labels {
            // Extract per-label data
            let probs_k: Vec<f32> = probabilities.iter().map(|row| row[k]).collect();
            let targets_k: Vec<f32> = targets.iter().map(|row| row[k]).collect();

            // Find optimal threshold for this label
            let (best_threshold, best_f1, best_precision, best_recall) =
                self.find_optimal_threshold(&probs_k, &targets_k);

            thresholds.push(best_threshold);
            f1_scores.push(best_f1);
            precisions.push(best_precision);
            recalls.push(best_recall);
        }

        Ok(TuneResult {
            thresholds,
            f1_scores,
            precisions,
            recalls,
        })
    }

    /// Find optimal threshold for a single label
    fn find_optimal_threshold(
        &self,
        probs: &[f32],
        targets: &[f32],
    ) -> (f32, f32, f32, f32) {
        let mut best_threshold = 0.5;
        let mut best_f1 = 0.0;
        let mut best_precision = 0.0;
        let mut best_recall = 0.0;

        let mut threshold = self.config.min_threshold;
        while threshold <= self.config.max_threshold {
            let (f1, precision, recall) = self.compute_f1(probs, targets, threshold);

            if f1 > best_f1 {
                best_f1 = f1;
                best_threshold = threshold;
                best_precision = precision;
                best_recall = recall;
            }

            threshold += self.config.step;
        }

        (best_threshold, best_f1, best_precision, best_recall)
    }

    /// Compute F1 score (and precision/recall) for a given threshold
    fn compute_f1(&self, probs: &[f32], targets: &[f32], threshold: f32) -> (f32, f32, f32) {
        let mut tp = 0u32;
        let mut fp = 0u32;
        let mut fn_ = 0u32;

        for (prob, target) in probs.iter().zip(targets.iter()) {
            let pred = if *prob >= threshold { 1.0 } else { 0.0 };
            let actual = *target;

            if pred == 1.0 && actual == 1.0 {
                tp += 1;
            } else if pred == 1.0 && actual == 0.0 {
                fp += 1;
            } else if pred == 0.0 && actual == 1.0 {
                fn_ += 1;
            }
        }

        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };

        let recall = if tp + fn_ > 0 {
            tp as f32 / (tp + fn_) as f32
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        (f1, precision, recall)
    }

    /// Tune thresholds from flat arrays (row-wise flattened)
    ///
    /// # Arguments
    /// * `probabilities_flat` - Flat array [row0_label0, row0_label1, ..., row1_label0, ...]
    /// * `targets_flat` - Flat array with same layout
    /// * `num_rows` - Number of samples
    /// * `num_labels` - Number of labels
    pub fn tune_flat(
        &self,
        probabilities_flat: &[f32],
        targets_flat: &[f32],
        num_rows: usize,
        num_labels: usize,
    ) -> Result<TuneResult> {
        // Convert flat arrays to nested format
        let probabilities: Vec<Vec<f32>> = (0..num_rows)
            .map(|i| {
                (0..num_labels)
                    .map(|k| probabilities_flat[i * num_labels + k])
                    .collect()
            })
            .collect();

        let targets: Vec<Vec<f32>> = (0..num_rows)
            .map(|i| {
                (0..num_labels)
                    .map(|k| targets_flat[i * num_labels + k])
                    .collect()
            })
            .collect();

        self.tune(&probabilities, &targets, num_labels)
    }
}

impl Default for ThresholdTuner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_tuner_basic() {
        let tuner = ThresholdTuner::new();

        // Create simple test data
        let probabilities = vec![
            vec![0.9, 0.1], // Sample 0: high prob for label 0, low for label 1
            vec![0.8, 0.2],
            vec![0.7, 0.3],
            vec![0.2, 0.8], // Sample 3: low prob for label 0, high for label 1
            vec![0.3, 0.9],
        ];

        let targets = vec![
            vec![1.0, 0.0], // True for label 0, false for label 1
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0], // False for label 0, true for label 1
            vec![0.0, 1.0],
        ];

        let result = tuner.tune(&probabilities, &targets, 2).unwrap();

        // Check we got thresholds for both labels
        assert_eq!(result.thresholds.len(), 2);
        assert_eq!(result.f1_scores.len(), 2);

        // F1 should be good for this clean data
        assert!(result.f1_scores[0] > 0.8, "Label 0 F1 should be high");
        assert!(result.f1_scores[1] > 0.8, "Label 1 F1 should be high");
    }

    #[test]
    fn test_threshold_tuner_imbalanced() {
        let tuner = ThresholdTuner::new();

        // Imbalanced data: most samples are negative
        let probabilities = vec![
            vec![0.9],
            vec![0.1],
            vec![0.1],
            vec![0.1],
            vec![0.1],
        ];

        let targets = vec![
            vec![1.0], // Only one positive
            vec![0.0],
            vec![0.0],
            vec![0.0],
            vec![0.0],
        ];

        let result = tuner.tune(&probabilities, &targets, 1).unwrap();

        // Threshold should be higher than 0.5 for imbalanced data
        // (to avoid false positives)
        assert!(
            result.thresholds[0] > 0.1,
            "Threshold should adjust for imbalance"
        );
    }

    #[test]
    fn test_threshold_tuner_flat() {
        let tuner = ThresholdTuner::new();

        // Flat arrays (row-wise: [r0_l0, r0_l1, r1_l0, r1_l1, ...])
        let probabilities_flat = vec![
            0.9, 0.1, // Row 0
            0.8, 0.2, // Row 1
            0.2, 0.8, // Row 2
            0.3, 0.9, // Row 3
        ];

        let targets_flat = vec![
            1.0, 0.0, // Row 0
            1.0, 0.0, // Row 1
            0.0, 1.0, // Row 2
            0.0, 1.0, // Row 3
        ];

        let result = tuner.tune_flat(&probabilities_flat, &targets_flat, 4, 2).unwrap();

        assert_eq!(result.thresholds.len(), 2);
        assert!(result.f1_scores[0] > 0.8);
        assert!(result.f1_scores[1] > 0.8);
    }
}
