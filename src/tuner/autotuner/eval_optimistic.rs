//! Optimistic mode evaluation (pre-binned datasets without target leakage concerns)
//!
//! This module handles evaluation using pre-binned BinnedDataset with various strategies:
//! - Holdout validation (single fold)
//! - K-fold cross-validation
//! - Conformal prediction with optional k-fold

use std::collections::HashMap;

use crate::dataset::{split_holdout, split_kfold, BinnedDataset};
use crate::tuner::traits::TunableModel;
use crate::tuner::config::OptimizationMetric;
use crate::Result;

use super::types::{EvalMetrics, EvalResult, check_conformal_support, aggregate_fold_results};
use super::metrics::{compute_additional_metrics, select_metric};

/// Evaluate using holdout validation with optional k-fold
///
/// If folds == 1, uses simple holdout. If folds > 1, runs k-fold CV.
pub(super) fn evaluate_holdout_with_folds<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    dataset: &BinnedDataset,
    params: &HashMap<String, f32>,
    validation_ratio: f32,
    folds: usize,
) -> EvalResult {
    if folds == 1 {
        evaluate_holdout(tuner, dataset, params, validation_ratio)
    } else {
        evaluate_kfold(tuner, dataset, params, folds)
    }
}

/// Evaluate using holdout validation (single fold)
///
/// Returns: (val_metric, train_metric, num_trees, f1_score, roc_auc, rank_ic)
fn evaluate_holdout<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    dataset: &BinnedDataset,
    params: &HashMap<String, f32>,
    validation_ratio: f32,
) -> EvalResult {
    // Check if custom validation is provided (for time-series/grouped data)
    if let Some((train_ptr, val_ptr)) = tuner.custom_validation() {
        // SAFETY: Pointers are valid for the duration of tune_with_validation() call
        let train_dataset = unsafe { train_ptr.as_ref() };
        let val_dataset = unsafe { val_ptr.as_ref() };

        // Build config - NO internal validation split (already pre-split!)
        let mut config = tuner.build_config(params);
        M::configure_validation(&mut config, 0.0, 0); // Disable internal split!

        // Train on custom train dataset
        let model = M::train(train_dataset, &config)?;

        // Predict on both datasets for metrics
        let train_predictions = model.predict(train_dataset);
        let val_predictions = model.predict(val_dataset);

        let metric = select_metric(&tuner.task_type());

        // Compute metrics on the respective datasets
        let train_metric = metric.compute(&train_predictions, train_dataset.targets());
        let val_metric = metric.compute(&val_predictions, val_dataset.targets());

        // Compute additional metrics using centralized helper
        let (f1_score, roc_auc, rank_ic) = compute_additional_metrics(
            &tuner.task_type(),
            &val_predictions,
            val_dataset.targets(),
            val_dataset.era_indices(),
        );

        return Ok(EvalMetrics {
            val_metric,
            train_metric,
            num_trees: model.num_trees(),
            f1_score,
            roc_auc,
            rank_ic,
        });
    }

    // Standard holdout: split the dataset internally
    // Build config with proper validation for early stopping
    // Use tuner's seed for consistency between training and evaluation splits
    let mut config = tuner.build_config(params);
    M::configure_validation(&mut config, validation_ratio, 0);

    // Train model (handles internal train/val split)
    let model = M::train(dataset, &config)?;

    // Create split for evaluation
    // Use era-based split ONLY when optimizing RankIC on panel data
    // BUT: only split internally if custom validation wasn't already provided
    let should_use_era_split = tuner.optimization_metric() == &OptimizationMetric::RankIc
        && dataset.has_eras()
        && tuner.custom_validation().is_none(); // Don't double-split!

    let split = if should_use_era_split {
        crate::dataset::split_holdout_by_era(dataset.era_indices().unwrap(), validation_ratio)
    } else {
        split_holdout(dataset.num_rows(), validation_ratio, 0.0, tuner.seed())
    };
    let metric = select_metric(&tuner.task_type());

    // Optimization: Only predict on validation set
    // Instead of predicting on full dataset and extracting both train/val indices,
    // we predict only on validation subset. This saves computation since validation
    // is typically only 20-30% of the data.
    let targets = dataset.targets();

    // Predict on validation subset (avoids predicting on training rows)
    let val_dataset = dataset.subset_by_indices(&split.validation);
    let val_predictions = model.predict(&val_dataset);

    let val_targets: Vec<f32> = split.validation.iter()
        .map(|&i| targets[i])
        .collect();
    let val_metric = metric.compute(&val_predictions, &val_targets);

    // Still need training metric for overfitting detection
    // For train metric, compute using all data but extract only training portion
    let all_predictions = model.predict(dataset);
    let train_predictions: Vec<f32> = split.train.iter()
        .map(|&i| all_predictions[i])
        .collect();
    let train_targets: Vec<f32> = split.train.iter()
        .map(|&i| targets[i])
        .collect();
    let train_metric = metric.compute(&train_predictions, &train_targets);

    // Extract validation era indices if available (for Rank IC computation)
    let val_era_indices =
        dataset.era_indices().map(|eras| split.validation.iter().map(|&i| eras[i]).collect::<Vec<u16>>());

    // Compute additional metrics (F1, ROC-AUC, Rank IC)
    let (f1_score, roc_auc, rank_ic) =
        compute_additional_metrics(&tuner.task_type(), &val_predictions, &val_targets, val_era_indices.as_deref());

    Ok(EvalMetrics {
        val_metric,
        train_metric,
        num_trees: model.num_trees(),
        f1_score,
        roc_auc,
        rank_ic,
    })
}

/// Evaluate using K-fold cross-validation
///
/// Each fold trains on (k-1)/k of the data and validates on 1/k.
/// Returns the average metrics across all folds.
///
/// Returns: (val_metric, train_metric, num_trees, f1_score, roc_auc, rank_ic)
fn evaluate_kfold<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    dataset: &BinnedDataset,
    params: &HashMap<String, f32>,
    k: usize,
) -> EvalResult {
    let kfold = split_kfold(dataset.num_rows(), k, tuner.seed())?;
    let config = tuner.build_config(params);
    let metric = select_metric(&tuner.task_type());

    let mut fold_results = Vec::with_capacity(k);

    for fold_idx in 0..k {
        let (train_idx, val_idx) = kfold.get_fold(fold_idx);

        // Create subset datasets for training and validation
        let train_dataset = dataset.subset_by_indices(&train_idx);
        let val_dataset = dataset.subset_by_indices(&val_idx);

        // Train on training fold only
        let model = M::train(&train_dataset, &config)?;

        // Get predictions on both train and validation sets
        let train_predictions = model.predict(&train_dataset);
        let val_predictions = model.predict(&val_dataset);

        // Compute metrics on respective sets
        let train_targets = train_dataset.targets();
        let val_targets = val_dataset.targets();

        // Compute train metric
        let train_metric = metric.compute(&train_predictions, train_targets);

        // Compute validation metric
        let val_metric = metric.compute(&val_predictions, val_targets);

        // Compute additional metrics using centralized helper
        let (f1_score, roc_auc, rank_ic) = compute_additional_metrics(
            &tuner.task_type(),
            &val_predictions,
            val_targets,
            val_dataset.era_indices(),
        );

        fold_results.push(EvalMetrics {
            val_metric,
            train_metric,
            num_trees: model.num_trees(),
            f1_score,
            roc_auc,
            rank_ic,
        });
    }

    Ok(aggregate_fold_results(&fold_results))
}

/// Evaluate using conformal prediction with optional k-fold
///
/// If folds == 1, uses simple conformal. If folds > 1, runs conformal k-fold CV
/// where each fold trains on the training subset and computes conformal quantile
/// from predictions on the validation subset.
pub(super) fn evaluate_conformal_with_folds<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    dataset: &BinnedDataset,
    params: &HashMap<String, f32>,
    calibration_ratio: f32,
    quantile: f32,
    folds: usize,
) -> EvalResult {
    check_conformal_support::<M>()?;

    if folds == 1 {
        evaluate_conformal(tuner, dataset, params, calibration_ratio, quantile)
    } else {
        // Run conformal on each fold and average
        let kfold = split_kfold(dataset.num_rows(), folds, tuner.seed())?;
        let mut fold_results = Vec::with_capacity(folds);

        for fold_idx in 0..folds {
            let (train_idx, val_idx) = kfold.get_fold(fold_idx);

            // Create subset datasets for training and validation
            let train_dataset = dataset.subset_by_indices(&train_idx);
            let val_dataset = dataset.subset_by_indices(&val_idx);

            // Build config with conformal settings and train
            let mut config = tuner.build_config(params);
            M::configure_conformal(&mut config, calibration_ratio, quantile);
            let model = M::train(&train_dataset, &config)?;

            // Extract conformal metrics
            fold_results.push(extract_conformal_result(
                tuner,
                &model,
                &val_dataset,
                val_dataset.targets(),
            ));
        }

        Ok(aggregate_fold_results(&fold_results))
    }
}

/// Evaluate using conformal prediction (O(1) metric lookup)
///
/// Instead of computing MSE over a validation set, this uses the conformal
/// quantile `q` as the optimization metric. Lower `q` = tighter intervals
/// = more confident model.
///
/// This is O(1) because `q` is already computed during training and stored
/// in the model. No prediction loop is needed.
///
/// # Arguments
/// * `dataset` - The binned dataset
/// * `params` - Hyperparameters to evaluate
/// * `calibration_ratio` - Fraction for calibration set
/// * `quantile` - Coverage quantile (e.g., 0.9 for 90%)
///
/// # Returns
/// * `val_metric` - The conformal quantile `q` (lower = better)
/// * `train_metric` - MSE on training set (for reference)
/// * `num_trees` - Number of trees in the model
/// * `f1_score` - None (conformal is typically used for regression)
fn evaluate_conformal<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    dataset: &BinnedDataset,
    params: &HashMap<String, f32>,
    calibration_ratio: f32,
    quantile: f32,
) -> EvalResult {
    check_conformal_support::<M>()?;

    // Check if custom validation is provided (for time-series/grouped data)
    // If so, use the same logic as evaluate_holdout (train on train_dataset, validate on val_dataset)
    if let Some((train_ptr, val_ptr)) = tuner.custom_validation() {
        // SAFETY: Pointers are valid for the duration of tune_with_validation() call
        let train_dataset = unsafe { train_ptr.as_ref() };
        let val_dataset = unsafe { val_ptr.as_ref() };

        // Build config - NO internal validation split (already pre-split!)
        let mut config = tuner.build_config(params);
        M::configure_validation(&mut config, 0.0, 0); // Disable internal split!

        // Train on custom train dataset
        let model = M::train(train_dataset, &config)?;

        // Predict on both datasets for metrics
        let train_predictions = model.predict(train_dataset);
        let val_predictions = model.predict(val_dataset);

        let metric = select_metric(&tuner.task_type());

        // Compute metrics on the respective datasets
        let train_metric = metric.compute(&train_predictions, train_dataset.targets());
        let val_metric = metric.compute(&val_predictions, val_dataset.targets());

        // Compute additional metrics using centralized helper
        let (f1_score, roc_auc, rank_ic) = compute_additional_metrics(
            &tuner.task_type(),
            &val_predictions,
            val_dataset.targets(),
            val_dataset.era_indices(),
        );

        return Ok(EvalMetrics {
            val_metric,
            train_metric,
            num_trees: model.num_trees(),
            f1_score,
            roc_auc,
            rank_ic,
        });
    }

    // Build config with conformal settings and train
    let mut config = tuner.build_config(params);
    M::configure_conformal(&mut config, calibration_ratio, quantile);
    let model = M::train(dataset, &config)?;

    // Extract conformal metrics (evaluate on training set)
    // For RankIC optimization with panel data, we need a proper validation split
    // BUT: only split internally if custom validation wasn't already provided
    let should_use_validation_split = tuner.optimization_metric() == &OptimizationMetric::RankIc
        && dataset.has_eras()
        && tuner.custom_validation().is_none(); // Don't double-split!

    if should_use_validation_split {
        // Use validation split for RankIC to avoid inflated training IC
        let validation_ratio = 0.2;

        let split = crate::dataset::split_holdout_by_era(
            dataset.era_indices().unwrap(),
            validation_ratio,
        );

        let predictions = model.predict(dataset);
        let metric = select_metric(&tuner.task_type());

        let (val_metric, train_metric, f1_score, roc_auc, rank_ic) = compute_metrics_by_indices(
            tuner,
            &predictions,
            dataset.targets(),
            &split.train,
            &split.validation,
            &metric,
            dataset.era_indices(),
        );

        Ok(EvalMetrics {
            val_metric,
            train_metric,
            num_trees: model.num_trees(),
            f1_score,
            roc_auc,
            rank_ic,
        })
    } else {
        // Standard conformal evaluation (on full training set)
        Ok(extract_conformal_result(tuner, &model, dataset, dataset.targets()))
    }
}

/// Extract conformal metrics from a trained model.
///
/// Returns (conformal_quantile, mse_on_eval_set, num_trees, f1_score, roc_auc, rank_ic)
fn extract_conformal_result<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    model: &M,
    eval_dataset: &BinnedDataset,
    eval_targets: &[f32],
) -> EvalMetrics {
    let conformal_q = model.conformal_quantile().unwrap_or(f32::MAX);
    let predictions = model.predict(eval_dataset);
    let mse = super::metrics::select_metric(&tuner.task_type()).compute(&predictions, eval_targets);

    // Compute additional metrics based on task type
    let (f1_score, roc_auc, rank_ic) =
        compute_additional_metrics(&tuner.task_type(), &predictions, eval_targets, eval_dataset.era_indices());

    EvalMetrics {
        val_metric: conformal_q,
        train_metric: mse,
        num_trees: model.num_trees(),
        f1_score,
        roc_auc,
        rank_ic,
    }
}

/// Compute metrics by splitting predictions according to indices (optimistic mode)
///
/// Used when training on full dataset and splitting predictions for evaluation.
/// Returns: (val_metric, train_metric, f1_score, roc_auc, rank_ic)
fn compute_metrics_by_indices<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    predictions: &[f32],
    targets: &[f32],
    train_idx: &[usize],
    val_idx: &[usize],
    metric: &super::metrics::Metric,
    era_indices: Option<&[u16]>,
) -> (f32, f32, Option<f32>, Option<f64>, Option<f64>) {
    let train_preds: Vec<f32> = train_idx.iter().map(|&i| predictions[i]).collect();
    let train_targets: Vec<f32> = train_idx.iter().map(|&i| targets[i]).collect();
    let train_metric = metric.compute(&train_preds, &train_targets);

    let val_preds: Vec<f32> = val_idx.iter().map(|&i| predictions[i]).collect();
    let val_targets: Vec<f32> = val_idx.iter().map(|&i| targets[i]).collect();
    let val_metric = metric.compute(&val_preds, &val_targets);

    // Extract validation era indices if available (for Rank IC computation)
    let val_era_indices =
        era_indices.map(|eras| val_idx.iter().map(|&i| eras[i]).collect::<Vec<u16>>());

    // Compute additional metrics (F1, ROC-AUC, Rank IC)
    let (f1_score, roc_auc, rank_ic) =
        compute_additional_metrics(&tuner.task_type(), &val_preds, &val_targets, val_era_indices.as_deref());

    (val_metric, train_metric, f1_score, roc_auc, rank_ic)
}

