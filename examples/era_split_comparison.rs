//! Era Split Comparison: Standard vs Directional Era Splitting (DES)
//!
//! This example compares TreeBoost's standard global split finding against
//! Directional Era Splitting (DES), inspired by WarpGBM's approach for
//! learning invariant/robust features across different time periods or environments.
//!
//! **Key Concept:**
//! - Standard GBDT learns ANY correlation in the data (including spurious ones)
//! - DES only learns correlations that are consistent across all eras/environments
//!
//! **Use Cases:**
//! - Financial ML: Market regimes shift over time
//! - Time series: Distribution changes across periods
//! - Multi-environment data: Different experimental conditions
//! - Competition data (Numerai): Era labels track time periods
//!
//! Run with:
//!   cargo run --release --example era_split_comparison

use rand::prelude::*;
use std::collections::HashMap;

use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::histogram::{average_era_gain, has_directional_agreement, EraSplitStats};

/// Era configuration
const NUM_ERAS: usize = 7;
const TRAIN_ERAS: usize = 5; // First 5 eras for training
const TEST_ERAS: usize = 2;  // Last 2 eras for testing (OOD evaluation)
const SAMPLES_PER_ERA: usize = 2000;
const NUM_FEATURES: usize = 10;

/// Features 0-4 are "robust" (consistent across eras)
/// Features 5-9 are "spurious" (flip direction every 2 eras)
const NUM_ROBUST_FEATURES: usize = 5;

/// Generate synthetic data with robust and spurious features
///
/// Robust features: Same relationship with target in all eras
/// Spurious features: Relationship flips or disappears in different eras
fn generate_era_data(seed: u64) -> EraDataset {
    let mut rng = StdRng::seed_from_u64(seed);

    let total_samples = NUM_ERAS * SAMPLES_PER_ERA;
    let mut features = Vec::with_capacity(total_samples * NUM_FEATURES);
    let mut targets = Vec::with_capacity(total_samples);
    let mut era_indices = Vec::with_capacity(total_samples);

    for era in 0..NUM_ERAS {
        for _ in 0..SAMPLES_PER_ERA {
            let mut target = 0.0f32;

            // Generate features
            for f in 0..NUM_FEATURES {
                let value: f32 = rng.gen_range(-1.0..1.0);
                features.push(value);

                if f < NUM_ROBUST_FEATURES {
                    // Robust feature: consistent positive relationship
                    // weight = 0.5 for each robust feature
                    target += 0.5 * value;
                } else {
                    // Spurious feature: relationship depends on era
                    // Flips sign every 2 eras
                    let era_group = era / 2;
                    let sign = if era_group % 2 == 0 { 1.0 } else { -1.0 };
                    // Stronger weight to make spurious correlations tempting
                    target += 0.8 * sign * value;
                }
            }

            // Add noise
            target += rng.gen_range(-0.3..0.3);

            targets.push(target);
            era_indices.push(era);
        }
    }

    EraDataset {
        features,
        targets,
        era_indices,
        num_samples: total_samples,
        num_features: NUM_FEATURES,
    }
}

/// Dataset with era annotations
struct EraDataset {
    features: Vec<f32>,
    targets: Vec<f32>,
    era_indices: Vec<usize>,
    num_samples: usize,
    num_features: usize,
}

impl EraDataset {
    /// Split into training and test sets based on eras
    fn split_by_era(&self) -> (TrainTestSplit, TrainTestSplit) {
        let mut train_features = Vec::new();
        let mut train_targets = Vec::new();
        let mut train_eras = Vec::new();

        let mut test_features = Vec::new();
        let mut test_targets = Vec::new();
        let mut test_eras = Vec::new();

        for i in 0..self.num_samples {
            let era = self.era_indices[i];
            let feat_start = i * self.num_features;
            let feat_end = feat_start + self.num_features;

            if era < TRAIN_ERAS {
                train_features.extend_from_slice(&self.features[feat_start..feat_end]);
                train_targets.push(self.targets[i]);
                train_eras.push(era);
            } else {
                test_features.extend_from_slice(&self.features[feat_start..feat_end]);
                test_targets.push(self.targets[i]);
                test_eras.push(era);
            }
        }

        (
            TrainTestSplit {
                features: train_features,
                targets: train_targets,
                era_indices: train_eras,
                num_samples: TRAIN_ERAS * SAMPLES_PER_ERA,
            },
            TrainTestSplit {
                features: test_features,
                targets: test_targets,
                era_indices: test_eras,
                num_samples: TEST_ERAS * SAMPLES_PER_ERA,
            },
        )
    }
}

struct TrainTestSplit {
    features: Vec<f32>,
    targets: Vec<f32>,
    era_indices: Vec<usize>,
    num_samples: usize,
}

/// Compute per-era histograms for a feature
/// Returns: (era_grad_left, era_hess_left, era_direction) for each split point
fn compute_era_histograms(
    features: &[f32],
    targets: &[f32],
    predictions: &[f32],
    era_indices: &[usize],
    feature_idx: usize,
    num_features: usize,
    num_bins: usize,
) -> Vec<EraHistogram> {
    let num_samples = targets.len();

    // Initialize per-era histograms
    let mut era_hists: HashMap<usize, Vec<(f32, f32)>> =
        HashMap::new(); // era -> [(grad_sum, hess_sum) per bin]

    for era in 0..NUM_ERAS {
        era_hists.insert(era, vec![(0.0, 0.0); num_bins]);
    }

    // Compute gradients and accumulate per-era histograms
    for i in 0..num_samples {
        let era = era_indices[i];
        let gradient = predictions[i] - targets[i]; // MSE gradient
        let hessian = 1.0f32;

        // Get feature value and bin it
        let value = features[i * num_features + feature_idx];
        let bin = ((value + 1.0) / 2.0 * (num_bins - 1) as f32)
            .clamp(0.0, (num_bins - 1) as f32) as usize;

        if let Some(hist) = era_hists.get_mut(&era) {
            hist[bin].0 += gradient;
            hist[bin].1 += hessian;
        }
    }

    // Convert to EraHistogram format with cumulative sums
    let mut result = Vec::new();
    let lambda = 1.0; // L2 regularization for EraSplitStats::compute

    for bin in 0..(num_bins - 1) {
        let mut era_data = Vec::new();

        for era in 0..TRAIN_ERAS {
            if let Some(hist) = era_hists.get(&era) {
                // Cumulative sum up to this bin
                let mut grad_left = 0.0f32;
                let mut hess_left = 0.0f32;
                let mut grad_total = 0.0f32;
                let mut hess_total = 0.0f32;

                for b in 0..num_bins {
                    if b <= bin {
                        grad_left += hist[b].0;
                        hess_left += hist[b].1;
                    }
                    grad_total += hist[b].0;
                    hess_total += hist[b].1;
                }

                // Use library's EraSplitStats::compute for direction and gain
                era_data.push(EraSplitStats::compute(
                    era,
                    grad_left,
                    hess_left,
                    grad_total,
                    hess_total,
                    lambda,
                ));
            }
        }

        result.push(EraHistogram {
            feature_idx,
            bin,
            era_stats: era_data,
        });
    }

    result
}

struct EraHistogram {
    #[allow(dead_code)]
    feature_idx: usize,
    #[allow(dead_code)]
    bin: usize,
    era_stats: Vec<EraSplitStats>,
}

/// Simple DES-style training: finds features with directional agreement
fn analyze_directional_splits(
    train: &TrainTestSplit,
    num_bins: usize,
) -> Vec<(usize, bool, f32)> {
    // Initialize predictions to zero
    let predictions = vec![0.0f32; train.num_samples];

    let mut feature_analysis = Vec::new();

    for f in 0..NUM_FEATURES {
        let era_hists = compute_era_histograms(
            &train.features,
            &train.targets,
            &predictions,
            &train.era_indices,
            f,
            NUM_FEATURES,
            num_bins,
        );

        // Find best split with directional agreement
        let mut best_des_gain = 0.0f32;
        let mut has_valid_des_split = false;

        // Find best global split (ignoring directional agreement)
        let mut best_global_gain = 0.0f32;

        for hist in &era_hists {
            let avg_gain = average_era_gain(&hist.era_stats);
            if avg_gain > best_global_gain {
                best_global_gain = avg_gain;
            }

            if has_directional_agreement(&hist.era_stats) && avg_gain > best_des_gain {
                best_des_gain = avg_gain;
                has_valid_des_split = true;
            }
        }

        feature_analysis.push((f, has_valid_des_split, best_des_gain));
    }

    feature_analysis
}

/// Compute RMSE
fn compute_rmse(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = predictions.len() as f32;
    let mse: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / n;
    mse.sqrt()
}

/// Compute correlation
fn compute_correlation(predictions: &[f32], targets: &[f32]) -> f32 {
    let n = predictions.len() as f32;
    let mean_p: f32 = predictions.iter().sum::<f32>() / n;
    let mean_t: f32 = targets.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_p = 0.0f32;
    let mut var_t = 0.0f32;

    for (p, t) in predictions.iter().zip(targets.iter()) {
        let dp = p - mean_p;
        let dt = t - mean_t;
        cov += dp * dt;
        var_p += dp * dp;
        var_t += dt * dt;
    }

    if var_p < 1e-10 || var_t < 1e-10 {
        return 0.0;
    }

    cov / (var_p.sqrt() * var_t.sqrt())
}

fn print_header() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║        Era Split Comparison: Standard vs Directional Era Splitting (DES)        ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║  Compares accuracy when learning robust vs spurious features across eras        ║"
    );
    println!(
        "║                                                                                  ║"
    );
    println!(
        "║  Data Design:                                                                   ║"
    );
    println!(
        "║    - Features 0-4: ROBUST (consistent relationship across all eras)             ║"
    );
    println!(
        "║    - Features 5-9: SPURIOUS (relationship flips every 2 eras)                   ║"
    );
    println!(
        "║    - Training: Eras 0-4 | Testing: Eras 5-6 (out-of-distribution)              ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();
}

fn main() {
    print_header();

    // Generate data
    println!("Generating synthetic data with {} eras...", NUM_ERAS);
    println!(
        "  {} samples/era x {} eras = {} total samples",
        SAMPLES_PER_ERA,
        NUM_ERAS,
        NUM_ERAS * SAMPLES_PER_ERA
    );
    println!("  {} robust features, {} spurious features", NUM_ROBUST_FEATURES, NUM_FEATURES - NUM_ROBUST_FEATURES);
    println!();

    let dataset = generate_era_data(42);
    let (train, test) = dataset.split_by_era();

    println!(
        "Train: {} samples from eras 0-{} (in-distribution)",
        train.num_samples,
        TRAIN_ERAS - 1
    );
    println!(
        "Test:  {} samples from eras {}-{} (out-of-distribution)",
        test.num_samples,
        TRAIN_ERAS,
        NUM_ERAS - 1
    );
    println!();

    // Analyze directional agreement
    println!("═══════════════════════════════════════════════════════════════════════════════════");
    println!("  Directional Era Splitting Analysis");
    println!("═══════════════════════════════════════════════════════════════════════════════════");
    println!();

    let feature_analysis = analyze_directional_splits(&train, 32);

    println!(
        "  {:>8} │ {:>12} │ {:>10} │ {:>12}",
        "Feature", "Type", "DES Valid", "Best Gain"
    );
    println!("  {}", "─".repeat(50));

    for (f, has_des, gain) in &feature_analysis {
        let feat_type = if *f < NUM_ROBUST_FEATURES {
            "Robust"
        } else {
            "Spurious"
        };
        let des_status = if *has_des { "Yes" } else { "No" };
        println!(
            "  {:>8} │ {:>12} │ {:>10} │ {:>12.4}",
            f, feat_type, des_status, gain
        );
    }
    println!();

    // Count how many robust vs spurious features pass DES
    let robust_des_count = feature_analysis
        .iter()
        .filter(|(f, has_des, _)| *f < NUM_ROBUST_FEATURES && *has_des)
        .count();
    let spurious_des_count = feature_analysis
        .iter()
        .filter(|(f, has_des, _)| *f >= NUM_ROBUST_FEATURES && *has_des)
        .count();

    println!(
        "  Summary: {}/{} robust features pass DES, {}/{} spurious features pass DES",
        robust_des_count,
        NUM_ROBUST_FEATURES,
        spurious_des_count,
        NUM_FEATURES - NUM_ROBUST_FEATURES
    );
    println!();

    // Train standard TreeBoost model (uses all features, global histograms)
    println!("═══════════════════════════════════════════════════════════════════════════════════");
    println!("  Training Models");
    println!("═══════════════════════════════════════════════════════════════════════════════════");
    println!();

    let config = GBDTConfig::new()
        .with_num_rounds(50)
        .with_max_depth(4)
        .with_learning_rate(0.1)
        .with_min_samples_leaf(10);

    println!("Training Standard TreeBoost (uses all features)...");
    let model_standard = GBDTModel::train(
        &train.features,
        NUM_FEATURES,
        &train.targets,
        config.clone(),
        None,
    )
    .expect("Training failed");

    // Get feature importance from standard model
    let importance = model_standard.feature_importances(NUM_FEATURES);
    println!("  Feature Importance (Standard Model):");
    for (f, imp) in importance.iter().enumerate() {
        let feat_type = if f < NUM_ROBUST_FEATURES {
            "Robust"
        } else {
            "Spurious"
        };
        println!("    Feature {} ({}): {:.4}", f, feat_type, imp);
    }
    println!();

    // Train DES-filtered model (only uses features that pass directional agreement)
    let des_features: Vec<usize> = feature_analysis
        .iter()
        .filter(|(_, has_des, _)| *has_des)
        .map(|(f, _, _)| *f)
        .collect();

    println!(
        "Training DES-Filtered TreeBoost (uses only {} DES-valid features: {:?})...",
        des_features.len(),
        des_features
    );

    // Create filtered feature set
    let mut train_features_filtered = Vec::with_capacity(train.num_samples * des_features.len());
    for i in 0..train.num_samples {
        for &f in &des_features {
            train_features_filtered.push(train.features[i * NUM_FEATURES + f]);
        }
    }

    let mut test_features_filtered = Vec::with_capacity(test.num_samples * des_features.len());
    for i in 0..test.num_samples {
        for &f in &des_features {
            test_features_filtered.push(test.features[i * NUM_FEATURES + f]);
        }
    }

    let model_des = if !des_features.is_empty() {
        Some(
            GBDTModel::train(
                &train_features_filtered,
                des_features.len(),
                &train.targets,
                config.clone(),
                None,
            )
            .expect("Training failed"),
        )
    } else {
        println!("  Warning: No features pass DES, skipping DES model");
        None
    };

    // Evaluate both models
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════");
    println!("  Evaluation Results");
    println!("═══════════════════════════════════════════════════════════════════════════════════");
    println!();

    // Standard model predictions
    let train_preds_std: Vec<f32> = model_standard
        .predict_raw(
            &train
                .features
                .iter()
                .map(|&x| x as f64)
                .collect::<Vec<_>>(),
        )
        .iter()
        .map(|&x| x as f32)
        .collect();

    let test_preds_std: Vec<f32> = model_standard
        .predict_raw(
            &test
                .features
                .iter()
                .map(|&x| x as f64)
                .collect::<Vec<_>>(),
        )
        .iter()
        .map(|&x| x as f32)
        .collect();

    let train_rmse_std = compute_rmse(&train_preds_std, &train.targets);
    let test_rmse_std = compute_rmse(&test_preds_std, &test.targets);
    let train_corr_std = compute_correlation(&train_preds_std, &train.targets);
    let test_corr_std = compute_correlation(&test_preds_std, &test.targets);

    println!("  Standard TreeBoost (all {} features):", NUM_FEATURES);
    println!(
        "    Train RMSE: {:.4}  | Test RMSE: {:.4}  | Generalization Gap: {:.4}",
        train_rmse_std,
        test_rmse_std,
        test_rmse_std - train_rmse_std
    );
    println!(
        "    Train Corr: {:.4}  | Test Corr: {:.4}  | Correlation Drop: {:.4}",
        train_corr_std,
        test_corr_std,
        train_corr_std - test_corr_std
    );
    println!();

    // DES model predictions (if available)
    if let Some(ref model) = model_des {
        let train_preds_des: Vec<f32> = model
            .predict_raw(
                &train_features_filtered
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
            )
            .iter()
            .map(|&x| x as f32)
            .collect();

        let test_preds_des: Vec<f32> = model
            .predict_raw(
                &test_features_filtered
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
            )
            .iter()
            .map(|&x| x as f32)
            .collect();

        let train_rmse_des = compute_rmse(&train_preds_des, &train.targets);
        let test_rmse_des = compute_rmse(&test_preds_des, &test.targets);
        let train_corr_des = compute_correlation(&train_preds_des, &train.targets);
        let test_corr_des = compute_correlation(&test_preds_des, &test.targets);

        println!(
            "  DES-Filtered TreeBoost ({} robust features):",
            des_features.len()
        );
        println!(
            "    Train RMSE: {:.4}  | Test RMSE: {:.4}  | Generalization Gap: {:.4}",
            train_rmse_des,
            test_rmse_des,
            test_rmse_des - train_rmse_des
        );
        println!(
            "    Train Corr: {:.4}  | Test Corr: {:.4}  | Correlation Drop: {:.4}",
            train_corr_des,
            test_corr_des,
            train_corr_des - test_corr_des
        );
        println!();

        // Summary comparison
        println!("═══════════════════════════════════════════════════════════════════════════════════");
        println!("  Summary");
        println!("═══════════════════════════════════════════════════════════════════════════════════");
        println!();

        let rmse_improvement = test_rmse_std - test_rmse_des;
        let corr_improvement = test_corr_des - test_corr_std;

        if rmse_improvement > 0.0 || corr_improvement > 0.0 {
            println!("  DES IMPROVES out-of-distribution generalization:");
            if rmse_improvement > 0.0 {
                println!(
                    "    - Test RMSE reduced by {:.4} ({:.1}% improvement)",
                    rmse_improvement,
                    rmse_improvement / test_rmse_std * 100.0
                );
            }
            if corr_improvement > 0.0 {
                println!(
                    "    - Test Correlation increased by {:.4}",
                    corr_improvement
                );
            }
        } else {
            println!("  Note: In this run, standard model performed similarly or better.");
            println!("  This can happen with random seeds - try different seeds to see the effect.");
        }

        println!();
        println!("  Key Insight:");
        println!("    Standard GBDT picks up spurious features that happen to correlate");
        println!("    with the target in training eras, but these relationships flip in");
        println!("    test eras, hurting OOD generalization.");
        println!();
        println!("    DES filters to only use features with CONSISTENT directional");
        println!("    relationships across all training eras, learning invariant patterns");
        println!("    that generalize better to unseen eras.");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════");
    println!("  Legend:");
    println!("    Robust:   Features with consistent target relationship across eras");
    println!("    Spurious: Features with relationship that flips between eras");
    println!("    DES:      Directional Era Splitting (accepts only consistent splits)");
    println!("    OOD:      Out-Of-Distribution (test eras not seen during training)");
    println!("═══════════════════════════════════════════════════════════════════════════════════");
}
