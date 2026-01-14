//! Realistic mode evaluation (per-split encoding to prevent target leakage)
//!
//! This module handles evaluation using raw DataFrames with per-split encoding
//! to ensure no target information leaks into the encoding/binning process.

use std::collections::HashMap;

use polars::prelude::DataFrame;

use crate::dataset::{split_holdout, split_kfold, BinnedDataset};
use crate::tuner::realistic::{
    encode_train_val_split, split_dataframe_by_indices, RealisticModeConfig,
};
use crate::tuner::traits::TunableModel;
use crate::Result;

use super::types::{aggregate_fold_results, check_conformal_support, EvalMetrics};

/// Evaluate using holdout with optional k-fold (realistic mode)
pub(super) fn evaluate_holdout_realistic_with_folds<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    raw_data: &DataFrame,
    realistic_cfg: &RealisticModeConfig,
    params: &HashMap<String, f32>,
    validation_ratio: f32,
    folds: usize,
) -> Result<EvalMetrics> {
    if folds == 1 {
        evaluate_holdout_realistic(tuner, raw_data, realistic_cfg, params, validation_ratio)
    } else {
        evaluate_kfold_realistic(tuner, raw_data, realistic_cfg, params, folds)
    }
}

/// Evaluate using holdout validation with per-split encoding (realistic mode)
///
/// Prevents target leakage by:
/// 1. Splitting raw data into train/val
/// 2. Fitting encoder on TRAIN ONLY
/// 3. Applying encoder to both train and val
/// 4. Training model on encoded train
/// 5. Evaluating on encoded val
fn evaluate_holdout_realistic<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    raw_data: &DataFrame,
    realistic_cfg: &RealisticModeConfig,
    params: &HashMap<String, f32>,
    validation_ratio: f32,
) -> Result<EvalMetrics> {
    // Split data
    let split = split_holdout(raw_data.height(), validation_ratio, 0.0, tuner.seed());
    let (train_df, val_df) = split_dataframe_by_indices(raw_data, &split.train, &split.validation)?;

    // Encode with per-split pipeline (no target leakage)
    let (train_dataset, val_dataset, val_targets) =
        encode_train_val_split(train_df, val_df, realistic_cfg)?;

    // Train and evaluate using shared helper
    train_and_evaluate(tuner, &train_dataset, &val_dataset, &val_targets, params)
}

/// Evaluate using K-fold cross-validation with per-split encoding (realistic mode)
fn evaluate_kfold_realistic<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    raw_data: &DataFrame,
    realistic_cfg: &RealisticModeConfig,
    params: &HashMap<String, f32>,
    k: usize,
) -> Result<EvalMetrics> {
    let kfold = split_kfold(raw_data.height(), k, tuner.seed())?;
    let mut fold_results = Vec::with_capacity(k);

    for fold_idx in 0..k {
        let (train_idx, val_idx) = kfold.get_fold(fold_idx);

        // Split and encode with per-fold pipeline (no target leakage)
        let (train_df, val_df) = split_dataframe_by_indices(raw_data, &train_idx, &val_idx)?;
        let (train_dataset, val_dataset, val_targets) =
            encode_train_val_split(train_df, val_df, realistic_cfg)?;

        // Train and evaluate using shared helper
        fold_results.push(train_and_evaluate(
            tuner,
            &train_dataset,
            &val_dataset,
            &val_targets,
            params,
        )?);
    }

    Ok(aggregate_fold_results(&fold_results))
}

/// Evaluate using conformal with optional k-fold (realistic mode)
pub(super) fn evaluate_conformal_realistic_with_folds<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    raw_data: &DataFrame,
    realistic_cfg: &RealisticModeConfig,
    params: &HashMap<String, f32>,
    calibration_ratio: f32,
    quantile: f32,
    folds: usize,
) -> Result<EvalMetrics> {
    // Check conformal support early (before expensive operations)
    check_conformal_support::<M>()?;

    if folds == 1 {
        evaluate_conformal_realistic(
            tuner,
            raw_data,
            realistic_cfg,
            params,
            calibration_ratio,
            quantile,
        )
    } else {
        // Run conformal on each fold and average
        let kfold = split_kfold(raw_data.height(), folds, tuner.seed())?;
        let mut fold_results = Vec::with_capacity(folds);

        for fold_idx in 0..folds {
            let (train_idx, val_idx) = kfold.get_fold(fold_idx);

            // Split and encode with per-fold pipeline
            let (train_df, val_df) = split_dataframe_by_indices(raw_data, &train_idx, &val_idx)?;
            let (train_dataset, cal_dataset, cal_targets) =
                encode_train_val_split(train_df, val_df, realistic_cfg)?;

            // Train and evaluate using conformal helper
            let result = train_and_evaluate_conformal(
                tuner,
                &train_dataset,
                &cal_dataset,
                &cal_targets,
                params,
                quantile,
            )?;
            fold_results.push(result);
        }

        Ok(aggregate_fold_results(&fold_results))
    }
}

/// Evaluate using conformal prediction with per-split encoding (realistic mode)
///
/// Uses the conformal quantile `q` as the optimization metric.
/// Lower `q` = tighter intervals = more confident model.
fn evaluate_conformal_realistic<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    raw_data: &DataFrame,
    realistic_cfg: &RealisticModeConfig,
    params: &HashMap<String, f32>,
    calibration_ratio: f32,
    quantile: f32,
) -> Result<EvalMetrics> {
    // Split data
    let split = split_holdout(raw_data.height(), calibration_ratio, 0.0, tuner.seed());
    let (train_df, cal_df) = split_dataframe_by_indices(raw_data, &split.train, &split.validation)?;

    // Encode with per-split pipeline (no target leakage)
    let (train_dataset, cal_dataset, cal_targets) =
        encode_train_val_split(train_df, cal_df, realistic_cfg)?;

    // Train and evaluate using conformal helper
    train_and_evaluate_conformal(
        tuner,
        &train_dataset,
        &cal_dataset,
        &cal_targets,
        params,
        quantile,
    )
}

// =============================================================================
// Helpers for realistic mode evaluation
// =============================================================================

/// Train model with external validation and compute metrics
///
/// This is the core training loop for realistic mode evaluation.
/// Handles config setup, training with external validation, and metric computation.
///
/// Returns: (val_metric, train_metric, num_trees, f1_score)
fn train_and_evaluate<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    train_dataset: &BinnedDataset,
    val_dataset: &BinnedDataset,
    val_targets: &[f32],
    params: &HashMap<String, f32>,
) -> Result<EvalMetrics> {
    let mut config = tuner.build_config(params);
    M::configure_validation(&mut config, 0.0, tuner.early_stopping_rounds());

    let model = M::train_with_validation(train_dataset, val_dataset, val_targets, &config)?;

    let metric = super::metrics::select_metric(&tuner.task_type());
    let eval_metrics = compute_eval_metrics(
        tuner,
        &model,
        train_dataset,
        val_dataset,
        val_targets,
        &metric,
    );

    Ok(eval_metrics)
}

/// Train model with conformal config and return quantile metric
///
/// Specialized version for conformal prediction evaluation.
/// Uses conformal quantile as the optimization metric instead of MSE/logloss.
///
/// Returns: (conformal_quantile, val_mse, num_trees, None, None, None)
fn train_and_evaluate_conformal<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    train_dataset: &BinnedDataset,
    val_dataset: &BinnedDataset,
    val_targets: &[f32],
    params: &HashMap<String, f32>,
    quantile: f32,
) -> Result<EvalMetrics> {
    if !M::supports_conformal() {
        return Err(crate::TreeBoostError::Config(
            "Conformal evaluation is not supported for this model type. \
             Use EvalStrategy::Holdout for generic model tuning."
                .to_string(),
        ));
    }

    // Build config with conformal settings (20% of train for calibration)
    let mut config = tuner.build_config(params);
    M::configure_conformal(&mut config, 0.2, quantile);
    let model = M::train(train_dataset, &config)?;

    // Extract conformal metrics (evaluate on validation set)
    Ok(extract_conformal_result(
        tuner,
        &model,
        val_dataset,
        val_targets,
    ))
}

/// Compute evaluation metrics for a trained model (realistic mode)
fn compute_eval_metrics<M: TunableModel>(
    tuner: &crate::tuner::AutoTuner<M>,
    model: &M,
    train_dataset: &BinnedDataset,
    val_dataset: &BinnedDataset,
    val_targets: &[f32],
    metric: &super::metrics::Metric,
) -> EvalMetrics {
    let train_preds = model.predict(train_dataset);
    let val_preds = model.predict(val_dataset);

    let train_metric = metric.compute(&train_preds, train_dataset.targets());
    let val_metric = metric.compute(&val_preds, val_targets);

    // Compute additional metrics (F1, ROC-AUC, Rank IC) using centralized helper
    let (f1_score, roc_auc, rank_ic) = super::metrics::compute_additional_metrics(
        &tuner.task_type(),
        &val_preds,
        val_targets,
        val_dataset.era_indices(),
    );

    EvalMetrics {
        val_metric,
        train_metric,
        num_trees: model.num_trees(),
        f1_score,
        roc_auc,
        rank_ic,
    }
}

/// Extract conformal metrics from a trained model.
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
    let (f1_score, roc_auc, rank_ic) = super::metrics::compute_additional_metrics(
        &tuner.task_type(),
        &predictions,
        eval_targets,
        eval_dataset.era_indices(),
    );

    EvalMetrics {
        val_metric: conformal_q,
        train_metric: mse,
        num_trees: model.num_trees(),
        f1_score,
        roc_auc,
        rank_ic,
    }
}
