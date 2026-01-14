//! Metric computation and selection for model evaluation
//!
//! This module handles computation of various evaluation metrics including
//! F1 scores, ROC-AUC, Rank IC, and metric selection based on task type.

use crate::tuner::config::TaskType;

// Re-export Metric for use within autotuner submodules
pub(super) use crate::tuner::metrics::Metric;

// Import binary classification threshold from types module (single source of truth)
use super::types::BINARY_CLASSIFICATION_THRESHOLD;

/// Compute F1 score for classification tasks
///
/// F1 = 2 * (precision * recall) / (precision + recall)
/// - Precision = TP / (TP + FP)
/// - Recall = TP / (TP + FN)
///
/// **Note:** Uses 0.5 as decision threshold, assuming binary class labels {0, 1}.
/// For highly imbalanced datasets, consider using a custom threshold or
/// alternative evaluation metric.
///
/// Returns `None` for regression tasks or if predictions/targets are misaligned.
pub(super) fn compute_f1_score(task_type: &TaskType, predictions: &[f32], targets: &[f32]) -> Option<f32> {
    // Only compute for binary classification (use TunerConfig's task_type)
    if !task_type.is_binary() {
        return None;
    }

    if predictions.is_empty() || predictions.len() != targets.len() {
        return None;
    }

    // For binary classification: predictions are log-odds, apply sigmoid
    // Then threshold at 0.5 for predicted class
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    for (&pred, &target) in predictions.iter().zip(targets.iter()) {
        // Convert log-odds to probability via sigmoid
        let prob = 1.0 / (1.0 + (-pred).exp());
        let pred_class = if prob >= BINARY_CLASSIFICATION_THRESHOLD {
            1.0
        } else {
            0.0
        };
        let actual_class = if target >= BINARY_CLASSIFICATION_THRESHOLD {
            1.0
        } else {
            0.0
        };

        match (pred_class as u8, actual_class as u8) {
            (1, 1) => true_positives += 1,
            (1, 0) => false_positives += 1,
            (0, 1) => false_negatives += 1,
            _ => {} // true negatives not needed for F1
        }
    }

    // Precision = TP / (TP + FP)
    let precision = if true_positives + false_positives > 0 {
        true_positives as f32 / (true_positives + false_positives) as f32
    } else {
        0.0 // No positive predictions
    };

    // Recall = TP / (TP + FN)
    let recall = if true_positives + false_negatives > 0 {
        true_positives as f32 / (true_positives + false_negatives) as f32
    } else {
        0.0 // No actual positives
    };

    // F1 = 2 * (precision * recall) / (precision + recall)
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0 // Both precision and recall are 0
    };

    Some(f1)
}

/// Select appropriate metric based on task type
pub(super) fn select_metric(task_type: &TaskType) -> Metric {
    match task_type {
        TaskType::Regression => Metric::Mse,
        TaskType::BinaryClassification => Metric::BinaryLogLoss,
        TaskType::MultiClassClassification => {
            // Default to 3 classes for multi-class; use MSE as primary metric
            // since MultiClassLogLoss requires knowing the exact number of classes
            Metric::Mse
        }
    }
}

/// Compute additional evaluation metrics (F1, ROC-AUC, Rank IC) for model validation
///
/// This helper consolidates metric computation logic that was previously duplicated
/// across multiple evaluation paths (holdout, conformal, custom validation).
///
/// # Arguments
/// * `task_type` - Classification or regression task
/// * `predictions` - Model predictions on validation data
/// * `targets` - Ground truth labels/values
/// * `era_indices` - Optional era/time group indices for panel data (used for Rank IC)
///
/// # Returns
/// Tuple of (F1 score, ROC-AUC, Rank IC), each Optional based on task type:
/// - F1: Classification tasks only
/// - ROC-AUC: Binary classification only
/// - Rank IC: Regression tasks only (cross-sectional correlation within eras)
///
/// # Industry Standard Practice
/// Metric computation should be centralized to ensure:
/// - Consistency across all evaluation strategies
/// - Single source of truth for metric definitions
/// - Easy maintenance and bug fixes
/// - Prevention of metric divergence across code paths
pub(super) fn compute_additional_metrics(
    task_type: &TaskType,
    predictions: &[f32],
    targets: &[f32],
    era_indices: Option<&[u16]>,
) -> (Option<f32>, Option<f64>, Option<f64>) {
    // F1 score for classification
    let f1_score = if task_type.is_classification() {
        compute_f1_score(task_type, predictions, targets)
    } else {
        None
    };

    // ROC-AUC for binary classification
    let roc_auc = if task_type.is_binary() {
        Some(super::super::metrics::compute_roc_auc(predictions, targets))
    } else {
        None
    };

    // Rank IC for regression (cross-sectional correlation within eras)
    let rank_ic = if task_type.is_regression() {
        let ic = super::super::metrics::compute_rank_ic(predictions, targets, era_indices);
        Some(ic)
    } else {
        None
    };

    (f1_score, roc_auc, rank_ic)
}
