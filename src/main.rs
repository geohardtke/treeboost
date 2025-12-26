//! TreeBoost CLI
//!
//! Command-line interface for training, prediction, and model inspection.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use treeboost::booster::{GBDTConfig, GBDTModel};
use treeboost::dataset::DatasetLoader;
use treeboost::serialize::{load_model, save_model};
use treeboost::tree::MonotonicConstraint;
use treeboost::Result;

#[derive(Parser)]
#[command(name = "treeboost")]
#[command(about = "High-performance Gradient Boosted Decision Tree engine", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new model
    Train {
        /// Input data file (Parquet or CSV)
        #[arg(short, long)]
        data: PathBuf,

        /// Target column name
        #[arg(short, long)]
        target: String,

        /// Output model path
        #[arg(short, long)]
        output: PathBuf,

        /// Number of boosting rounds
        #[arg(long, default_value_t = 100)]
        rounds: usize,

        /// Maximum tree depth
        #[arg(long, default_value_t = 6)]
        max_depth: usize,

        /// Maximum number of leaves
        #[arg(long, default_value_t = 31)]
        max_leaves: usize,

        /// Learning rate
        #[arg(long, default_value_t = 0.1)]
        learning_rate: f32,

        /// Minimum samples per leaf
        #[arg(long, default_value_t = 20)]
        min_samples_leaf: usize,

        /// L2 regularization (lambda)
        #[arg(long, default_value_t = 1.0)]
        lambda: f32,

        /// Shannon Entropy regularization weight
        #[arg(long, default_value_t = 0.0)]
        entropy_weight: f32,

        /// Row subsampling ratio (0.0 to 1.0)
        #[arg(long, default_value_t = 1.0)]
        subsample: f32,

        /// Column subsampling ratio (0.0 to 1.0)
        #[arg(long, default_value_t = 1.0)]
        colsample: f32,

        /// Early stopping rounds (0 to disable)
        #[arg(long, default_value_t = 0)]
        early_stopping: usize,

        /// Validation ratio for early stopping (e.g., 0.1 for 10%)
        #[arg(long, default_value_t = 0.1)]
        validation_ratio: f32,

        /// Loss function: mse or huber
        #[arg(long, default_value = "mse")]
        loss: String,

        /// Pseudo-Huber delta (only if loss=huber)
        #[arg(long, default_value_t = 1.0)]
        huber_delta: f32,

        /// Enable conformal prediction with calibration ratio
        #[arg(long)]
        conformal: Option<f32>,

        /// Conformal quantile level (e.g., 0.9 for 90% coverage)
        #[arg(long, default_value_t = 0.9)]
        conformal_quantile: f32,

        /// Number of bins for feature discretization
        #[arg(long, default_value_t = 255)]
        num_bins: usize,

        /// Feature columns (comma-separated, or all if not specified)
        #[arg(long)]
        features: Option<String>,

        /// Disable parallel prediction (use sequential instead)
        #[arg(long)]
        no_parallel: bool,

        /// Disable column reordering optimization
        #[arg(long)]
        no_reorder: bool,

        /// Disable 4-bit packed dataset optimization
        #[arg(long)]
        no_pack: bool,

        /// Disable all performance optimizations
        #[arg(long)]
        no_optimizations: bool,

        /// Monotonic constraints (comma-separated: +1=increasing, -1=decreasing, 0=none)
        /// Example: "--monotonic +1,-1,0" for 3 features
        #[arg(long)]
        monotonic: Option<String>,

        /// Feature interaction groups (semicolon-separated groups of comma-separated indices)
        /// Example: "--interactions 0,1,2;3,4" means features 0-2 can interact, features 3-4 can interact
        #[arg(long)]
        interactions: Option<String>,
    },

    /// Make predictions using a trained model
    Predict {
        /// Trained model path
        #[arg(short, long)]
        model: PathBuf,

        /// Input data file (Parquet or CSV)
        #[arg(short, long)]
        data: PathBuf,

        /// Target column name (optional, for evaluation)
        #[arg(short, long)]
        target: Option<String>,

        /// Output predictions file (JSON)
        #[arg(short, long)]
        output: PathBuf,

        /// Include prediction intervals (if model was calibrated)
        #[arg(long)]
        intervals: bool,
    },

    /// Display model information
    Info {
        /// Model path
        #[arg(short, long)]
        model: PathBuf,

        /// Show feature importances
        #[arg(long)]
        importances: bool,

        /// Number of features (required if showing importances)
        #[arg(long)]
        num_features: Option<usize>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            data,
            target,
            output,
            rounds,
            max_depth,
            max_leaves,
            learning_rate,
            min_samples_leaf,
            lambda,
            entropy_weight,
            subsample,
            colsample,
            early_stopping,
            validation_ratio,
            loss,
            huber_delta,
            conformal,
            conformal_quantile,
            num_bins,
            features,
            no_parallel,
            no_reorder,
            no_pack,
            no_optimizations,
            monotonic,
            interactions,
        } => {
            println!("Loading data from {:?}...", data);
            let loader = DatasetLoader::new(num_bins);

            let feature_cols: Option<Vec<&str>> = features.as_ref().map(|f| {
                f.split(',').map(|s| s.trim()).collect()
            });

            let dataset = if data.extension().and_then(|s| s.to_str()) == Some("parquet") {
                loader.load_parquet(&data, &target, feature_cols.as_deref())?
            } else {
                loader.load_csv(&data, &target, feature_cols.as_deref())?
            };

            println!("Loaded {} rows, {} features", dataset.num_rows(), dataset.num_features());

            // Build config
            let mut config = GBDTConfig::new()
                .with_num_rounds(rounds)
                .with_max_depth(max_depth)
                .with_max_leaves(max_leaves)
                .with_learning_rate(learning_rate)
                .with_min_samples_leaf(min_samples_leaf)
                .with_lambda(lambda)
                .with_entropy_weight(entropy_weight)
                .with_subsample(subsample)
                .with_colsample(colsample);

            // Enable early stopping if requested
            if early_stopping > 0 {
                config = config.with_early_stopping(early_stopping, validation_ratio);
                println!("Early stopping enabled: {} rounds, {:.1}% validation",
                         early_stopping, validation_ratio * 100.0);
            }

            // Set loss function
            config = match loss.as_str() {
                "mse" => config.with_mse_loss(),
                "huber" => config.with_pseudo_huber_loss(huber_delta),
                _ => {
                    eprintln!("Unknown loss function: {}. Using MSE.", loss);
                    config.with_mse_loss()
                }
            };

            // Enable conformal prediction if requested
            if let Some(calib_ratio) = conformal {
                config = config.with_conformal(calib_ratio, conformal_quantile);
                println!("Conformal prediction enabled: {:.1}% calibration, {:.1}% coverage",
                         calib_ratio * 100.0, conformal_quantile * 100.0);
            }

            // Parse and apply monotonic constraints
            if let Some(ref mono_str) = monotonic {
                let constraints: Vec<MonotonicConstraint> = mono_str
                    .split(',')
                    .map(|s| {
                        let s = s.trim();
                        match s {
                            "+1" | "1" => MonotonicConstraint::Increasing,
                            "-1" => MonotonicConstraint::Decreasing,
                            "0" | "" => MonotonicConstraint::None,
                            _ => {
                                eprintln!("Warning: Unknown monotonic value '{}', using None", s);
                                MonotonicConstraint::None
                            }
                        }
                    })
                    .collect();

                let num_constrained = constraints.iter()
                    .filter(|c| **c != MonotonicConstraint::None)
                    .count();
                if num_constrained > 0 {
                    println!("Monotonic constraints: {} features constrained", num_constrained);
                }
                config = config.with_monotonic_constraints(constraints);
            }

            // Parse and apply interaction constraints
            if let Some(ref interact_str) = interactions {
                let groups: Vec<Vec<usize>> = interact_str
                    .split(';')
                    .filter(|g| !g.trim().is_empty())
                    .map(|group| {
                        group
                            .split(',')
                            .filter_map(|s| s.trim().parse::<usize>().ok())
                            .collect()
                    })
                    .filter(|g: &Vec<usize>| !g.is_empty())
                    .collect();

                if !groups.is_empty() {
                    println!("Interaction constraints: {} groups", groups.len());
                    config = config.with_interaction_groups(groups);
                }
            }

            // Apply optimization opt-outs
            if no_optimizations {
                config = config.without_optimizations();
            } else {
                if no_parallel {
                    config = config.with_parallel_prediction(false);
                }
                if no_reorder {
                    config = config.with_column_reordering(false);
                }
                if no_pack {
                    config = config.with_packed_dataset(false);
                }
            }

            println!("\nTraining configuration:");
            println!("  Rounds: {}", rounds);
            println!("  Max depth: {}", max_depth);
            println!("  Max leaves: {}", max_leaves);
            println!("  Learning rate: {}", learning_rate);
            println!("  Loss: {}", loss);
            if entropy_weight > 0.0 {
                println!("  Entropy weight: {}", entropy_weight);
            }
            if subsample < 1.0 {
                println!("  Row subsampling: {:.0}%", subsample * 100.0);
            }
            if colsample < 1.0 {
                println!("  Column subsampling: {:.0}%", colsample * 100.0);
            }
            if early_stopping > 0 {
                println!("  Early stopping: {} rounds, {:.0}% validation", early_stopping, validation_ratio * 100.0);
            }
            println!("  Optimizations:");
            println!("    Parallel prediction: {}", config.parallel_prediction);
            println!("    Column reordering: {}", config.column_reordering);
            println!("    4-bit packing: {}", config.packed_dataset);

            println!("\nTraining model...");
            let model = GBDTModel::train_binned(&dataset, config)?;

            println!("Training complete: {} trees built", model.num_trees());

            if let Some(q) = model.conformal_quantile() {
                println!("Conformal quantile: {:.4}", q);
            }

            println!("\nSaving model to {:?}...", output);
            save_model(&model, &output)?;

            println!("Model saved successfully.");
            Ok(())
        }

        Commands::Predict {
            model,
            data,
            target,
            output,
            intervals,
        } => {
            println!("Loading model from {:?}...", model);
            let model = load_model(&model)?;

            println!("Model loaded: {} trees, {} features", model.num_trees(), model.num_features());

            println!("Loading data from {:?}...", data);
            let loader = DatasetLoader::new(255);

            // Use model's feature info for consistent binning
            let dataset = if data.extension().and_then(|s| s.to_str()) == Some("parquet") {
                loader.load_parquet_for_prediction(&data, model.feature_info())?
            } else {
                loader.load_csv_for_prediction(&data, model.feature_info())?
            };

            println!("Loaded {} rows", dataset.num_rows());

            // Load actual targets if specified (for evaluation)
            let actual_targets: Option<Vec<f32>> = if let Some(ref target_col) = target {
                use polars::prelude::*;
                let df = if data.extension().and_then(|s| s.to_str()) == Some("parquet") {
                    LazyFrame::scan_parquet(&data, Default::default())?.collect()?
                } else {
                    CsvReadOptions::default()
                        .try_into_reader_with_file_path(Some(data.clone()))?
                        .finish()?
                };
                let col = df.column(target_col)?;
                let series = col.as_materialized_series();
                let targets: Vec<f32> = series
                    .cast(&DataType::Float32)?
                    .f32()?
                    .into_iter()
                    .map(|opt| opt.unwrap_or(f32::NAN))
                    .collect();
                Some(targets)
            } else {
                None
            };

            println!("Making predictions...");
            if intervals && model.conformal_quantile().is_some() {
                let (predictions, lower, upper) = model.predict_with_intervals(&dataset);

                // Create JSON output
                let results: Vec<serde_json::Value> = predictions
                    .iter()
                    .zip(lower.iter())
                    .zip(upper.iter())
                    .enumerate()
                    .map(|(i, ((p, l), u))| {
                        serde_json::json!({
                            "row": i,
                            "prediction": p,
                            "lower": l,
                            "upper": u,
                        })
                    })
                    .collect();

                let json = serde_json::to_string_pretty(&results)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                std::fs::write(&output, json)?;
                println!("Predictions with intervals saved to {:?}", output);

                // Compute metrics if target available
                if let Some(ref targets) = actual_targets {
                    let mse: f32 = predictions
                        .iter()
                        .zip(targets.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f32>()
                        / predictions.len() as f32;

                    let coverage: f32 = targets
                        .iter()
                        .zip(lower.iter())
                        .zip(upper.iter())
                        .filter(|((t, l), u)| **t >= **l && **t <= **u)
                        .count() as f32
                        / targets.len() as f32;

                    println!("\nEvaluation:");
                    println!("  MSE: {:.4}", mse);
                    println!("  RMSE: {:.4}", mse.sqrt());
                    println!("  Coverage: {:.2}%", coverage * 100.0);
                }
            } else {
                let predictions = model.predict(&dataset);

                // Create JSON output
                let results: Vec<serde_json::Value> = predictions
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        serde_json::json!({
                            "row": i,
                            "prediction": p,
                        })
                    })
                    .collect();

                let json = serde_json::to_string_pretty(&results)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                std::fs::write(&output, json)?;
                println!("Predictions saved to {:?}", output);

                // Compute metrics if target available
                if let Some(ref targets) = actual_targets {
                    let mse: f32 = predictions
                        .iter()
                        .zip(targets.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f32>()
                        / predictions.len() as f32;

                    println!("\nEvaluation:");
                    println!("  MSE: {:.4}", mse);
                    println!("  RMSE: {:.4}", mse.sqrt());
                }
            }

            Ok(())
        }

        Commands::Info {
            model,
            importances,
            num_features,
        } => {
            println!("Loading model from {:?}...", model);
            let model = load_model(&model)?;

            println!("\nModel Information:");
            println!("  Number of trees: {}", model.num_trees());
            println!("  Base prediction: {:.4}", model.base_prediction());

            let config = model.config();
            println!("\nTraining Configuration:");
            println!("  Rounds: {}", config.num_rounds);
            println!("  Max depth: {}", config.max_depth);
            println!("  Max leaves: {}", config.max_leaves);
            println!("  Learning rate: {}", config.learning_rate);
            println!("  Lambda: {}", config.lambda);
            println!("  Min samples/leaf: {}", config.min_samples_leaf);
            println!("  Subsample: {}", config.subsample);
            println!("  Loss: {:?}", config.loss_type);

            if config.entropy_weight > 0.0 {
                println!("  Entropy weight: {}", config.entropy_weight);
            }

            if let Some(q) = model.conformal_quantile() {
                println!("\nConformal Prediction:");
                println!("  Quantile: {:.4}", q);
                println!("  Coverage: {:.1}%", config.conformal_quantile * 100.0);
            }

            if importances {
                if let Some(n_feat) = num_features {
                    println!("\nFeature Importances:");
                    let imps = model.feature_importances(n_feat);
                    for (i, imp) in imps.iter().enumerate() {
                        println!("  Feature {}: {:.4}", i, imp);
                    }
                } else {
                    println!("\nWarning: --num-features required to show importances");
                }
            }

            Ok(())
        }
    }
}
