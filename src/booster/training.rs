//! GBDT training implementation
//!
//! Contains all training-related methods for GBDTModel.

use super::GBDTModel;
#[cfg(any(feature = "cuda", feature = "gpu"))]
use crate::backend::{BackendType, GpuMode};
use crate::booster::GBDTConfig;
use crate::dataset::{
    split_holdout, BinnedDataset, ColumnPermutation, FeatureInfo, FeatureType, QuantileBinner,
};
use crate::loss::{softmax, MultiClassLogLoss};
use crate::tree::{InteractionConstraints, Tree, TreeGrower};
use crate::tuner::ModelFormat;
use crate::{Result, TreeBoostError};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::path::Path;

#[cfg(feature = "cuda")]
use crate::backend::cuda::FullCudaTreeBuilder;

#[cfg(feature = "gpu")]
use crate::backend::wgpu::FullGpuTreeBuilder;

/// Check if early stopping should trigger
#[inline]
pub(crate) fn should_early_stop(
    rounds_without_improvement: usize,
    current_count: usize,
    early_stopping_rounds: usize,
    min_early_stopping: usize,
) -> bool {
    rounds_without_improvement >= early_stopping_rounds && current_count >= min_early_stopping
}

/// Calculate how many trees/rounds to keep after early stopping
#[inline]
pub(crate) fn early_stop_keep_count(best_count: usize, min_early_stopping: usize) -> usize {
    best_count.max(min_early_stopping)
}

impl GBDTModel {
    /// Train a GBDT model from raw feature data (high-level API)
    ///
    /// This is the primary training API that handles binning automatically.
    /// Features are discretized using T-Digest quantile binning with parallelization.
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix: `features[row * num_features + feature]`
    ///   Shape: `(num_rows, num_features)` flattened to 1D
    /// * `num_features` - Number of features (columns)
    /// * `targets` - Target values, one per row
    /// * `config` - Training configuration
    /// * `feature_names` - Optional feature names (defaults to "feature_0", "feature_1", ...)
    ///
    /// # Example
    /// ```ignore
    /// let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 features
    /// let targets = vec![0.5, 1.5];
    /// let config = GBDTConfig::new().with_num_rounds(100);
    /// let model = GBDTModel::train(&features, 3, &targets, config, None)?;
    /// ```
    pub fn train(
        features: &[f32],
        num_features: usize,
        targets: &[f32],
        config: GBDTConfig,
        feature_names: Option<Vec<String>>,
    ) -> Result<Self> {
        let num_rows = if num_features > 0 {
            features.len() / num_features
        } else {
            0
        };

        if num_rows == 0 || num_features == 0 {
            return Err(TreeBoostError::Config("Empty dataset".to_string()));
        }

        if features.len() != num_rows * num_features {
            return Err(TreeBoostError::Config(format!(
                "Feature array length {} doesn't match num_rows * num_features ({} * {} = {})",
                features.len(),
                num_rows,
                num_features,
                num_rows * num_features
            )));
        }

        if targets.len() != num_rows {
            return Err(TreeBoostError::Config(format!(
                "Target length {} doesn't match num_rows {}",
                targets.len(),
                num_rows
            )));
        }

        // Create binner
        let binner = QuantileBinner::new(config.num_bins);

        // Parallel binning: process each feature column in parallel
        let binned_results: Vec<(Vec<u8>, FeatureInfo)> = (0..num_features)
            .into_par_iter()
            .map(|f| {
                // Extract column (row-major to column values)
                let column: Vec<f64> = (0..num_rows)
                    .map(|r| features[r * num_features + f] as f64)
                    .collect();

                // Compute boundaries and bin
                let boundaries = binner.compute_boundaries(&column);
                let binned = binner.bin_column(&column, &boundaries);

                // Create feature info
                let name = feature_names
                    .as_ref()
                    .and_then(|names| names.get(f).cloned())
                    .unwrap_or_else(|| format!("feature_{}", f));

                let info = FeatureInfo {
                    name,
                    feature_type: FeatureType::Numeric,
                    num_bins: (boundaries.len() + 1).min(255) as u8,
                    bin_boundaries: boundaries,
                };

                (binned, info)
            })
            .collect();

        // Combine results into column-major storage
        let mut binned_data = Vec::with_capacity(num_rows * num_features);
        let mut feature_info = Vec::with_capacity(num_features);

        for (binned_col, info) in binned_results {
            binned_data.extend(binned_col);
            feature_info.push(info);
        }

        // Create BinnedDataset and train
        let dataset = BinnedDataset::new(num_rows, binned_data, targets.to_vec(), feature_info);

        Self::train_binned(&dataset, config)
    }

    /// Train a GBDT model and save to output directory
    ///
    /// This is a convenience method that trains a model and automatically saves:
    /// - The trained model in the specified format(s)
    /// - `config.json` with the training configuration for reproducibility
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix: `features[row * num_features + feature]`
    /// * `num_features` - Number of features (columns)
    /// * `targets` - Target values, one per row
    /// * `config` - Training configuration
    /// * `feature_names` - Optional feature names
    /// * `output_dir` - Directory to save the model and config
    /// * `formats` - Model formats to save (e.g., `[ModelFormat::Rkyv]`)
    ///
    /// # Example
    /// ```ignore
    /// let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 features
    /// let targets = vec![0.5, 1.5];
    /// let config = GBDTConfig::new().with_num_rounds(100);
    ///
    /// let model = GBDTModel::train_with_output(
    ///     &features, 3, &targets, config, None,
    ///     "output/my_model",
    ///     &[ModelFormat::Rkyv],
    /// )?;
    /// // Creates: output/my_model/model.rkyv and output/my_model/config.json
    /// ```
    pub fn train_with_output(
        features: &[f32],
        num_features: usize,
        targets: &[f32],
        config: GBDTConfig,
        feature_names: Option<Vec<String>>,
        output_dir: impl AsRef<Path>,
        formats: &[ModelFormat],
    ) -> Result<Self> {
        // Train the model
        let model = Self::train(
            features,
            num_features,
            targets,
            config.clone(),
            feature_names,
        )?;

        // Save to output directory
        model.save_to_directory(output_dir, &config, formats)?;

        Ok(model)
    }

    /// Save a trained model to a directory
    ///
    /// Creates the directory if it doesn't exist and saves:
    /// - The model in each specified format
    /// - `config.json` with the training configuration
    ///
    /// # Arguments
    /// * `output_dir` - Directory to save the model
    /// * `config` - Training configuration (for config.json)
    /// * `formats` - Model formats to save (must not be empty)
    ///
    /// # Errors
    /// Returns an error if `formats` is empty or if I/O operations fail.
    pub fn save_to_directory(
        &self,
        output_dir: impl AsRef<Path>,
        config: &GBDTConfig,
        formats: &[ModelFormat],
    ) -> Result<()> {
        use std::fs;
        use std::io::Write;

        // Validate formats is not empty
        if formats.is_empty() {
            return Err(TreeBoostError::Config(
                "formats must not be empty - specify at least one model format".to_string(),
            ));
        }

        let dir = output_dir.as_ref();

        // Create directory if it doesn't exist
        fs::create_dir_all(dir)?;

        // Save config.json for reproducibility
        let config_path = dir.join("config.json");
        let config_json = serde_json::to_string_pretty(config).map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to serialize config: {}", e))
        })?;
        let mut file = fs::File::create(&config_path)?;
        file.write_all(config_json.as_bytes())?;

        // Save model in each format
        for format in formats {
            let model_path = dir.join(format!("model.{}", format.extension()));
            match format {
                ModelFormat::Rkyv => {
                    crate::serialize::save_model(self, &model_path)?;
                }
                ModelFormat::Bincode => {
                    crate::serialize::save_model_bincode(self, &model_path)?;
                }
            }
        }

        Ok(())
    }

    /// Train a GBDT model with Directional Era Splitting (DES)
    ///
    /// Era splitting filters out spurious correlations by requiring all eras
    /// to agree on split direction. This is useful for time-series or financial
    /// data where patterns may not generalize across time periods.
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix: `features[row * num_features + feature]`
    /// * `num_features` - Number of features (columns)
    /// * `targets` - Target values, one per row
    /// * `era_indices` - Era index (0-based) for each row
    /// * `config` - Training configuration (era_splitting must be enabled)
    /// * `feature_names` - Optional feature names
    ///
    /// # Example
    /// ```ignore
    /// let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 features
    /// let targets = vec![0.5, 1.5];
    /// let era_indices = vec![0, 1]; // Row 0 in era 0, row 1 in era 1
    /// let config = GBDTConfig::new()
    ///     .with_num_rounds(100)
    ///     .with_era_splitting(true);
    /// let model = GBDTModel::train_with_eras(&features, 3, &targets, &era_indices, config, None)?;
    /// ```
    pub fn train_with_eras(
        features: &[f32],
        num_features: usize,
        targets: &[f32],
        era_indices: &[u16],
        config: GBDTConfig,
        feature_names: Option<Vec<String>>,
    ) -> Result<Self> {
        let num_rows = if num_features > 0 {
            features.len() / num_features
        } else {
            0
        };

        if num_rows == 0 || num_features == 0 {
            return Err(TreeBoostError::Config("Empty dataset".to_string()));
        }

        if features.len() != num_rows * num_features {
            return Err(TreeBoostError::Config(format!(
                "Feature array length {} doesn't match num_rows * num_features ({} * {} = {})",
                features.len(),
                num_rows,
                num_features,
                num_rows * num_features
            )));
        }

        if targets.len() != num_rows {
            return Err(TreeBoostError::Config(format!(
                "Target length {} doesn't match num_rows {}",
                targets.len(),
                num_rows
            )));
        }

        if era_indices.len() != num_rows {
            return Err(TreeBoostError::Config(format!(
                "era_indices length {} doesn't match num_rows {}",
                era_indices.len(),
                num_rows
            )));
        }

        if !config.era_splitting {
            return Err(TreeBoostError::Config(
                "era_splitting must be enabled in config when using train_with_eras".to_string(),
            ));
        }

        // Create binner
        let binner = QuantileBinner::new(config.num_bins);

        // Parallel binning: process each feature column in parallel
        let binned_results: Vec<(Vec<u8>, FeatureInfo)> = (0..num_features)
            .into_par_iter()
            .map(|f| {
                // Extract column (row-major to column values)
                let column: Vec<f64> = (0..num_rows)
                    .map(|r| features[r * num_features + f] as f64)
                    .collect();

                // Compute boundaries and bin
                let boundaries = binner.compute_boundaries(&column);
                let binned = binner.bin_column(&column, &boundaries);

                // Create feature info
                let name = feature_names
                    .as_ref()
                    .and_then(|names| names.get(f).cloned())
                    .unwrap_or_else(|| format!("feature_{}", f));

                let info = FeatureInfo {
                    name,
                    feature_type: FeatureType::Numeric,
                    num_bins: (boundaries.len() + 1).min(255) as u8,
                    bin_boundaries: boundaries,
                };

                (binned, info)
            })
            .collect();

        // Combine results into column-major storage
        let mut binned_data = Vec::with_capacity(num_rows * num_features);
        let mut feature_info = Vec::with_capacity(num_features);

        for (binned_col, info) in binned_results {
            binned_data.extend(binned_col);
            feature_info.push(info);
        }

        // Create BinnedDataset with era indices
        let mut dataset = BinnedDataset::new(num_rows, binned_data, targets.to_vec(), feature_info);
        dataset.set_era_indices(era_indices.to_vec());

        Self::train_binned(&dataset, config)
    }

    /// Train a GBDT model from pre-binned data (low-level API)
    ///
    /// Use this when you have already binned your data (e.g., for repeated training
    /// with different hyperparameters on the same binned dataset).
    ///
    /// For most use cases, prefer `train()` which handles binning automatically.
    pub fn train_binned(dataset: &BinnedDataset, mut config: GBDTConfig) -> Result<Self> {
        // Dispatch to multi-class training if using multi-class loss
        if let Some(num_classes) = config.loss_type.num_classes() {
            return Self::train_binned_multiclass(dataset, config, num_classes);
        }

        // Dispatch to multi-label training if using multi-label loss
        if config.loss_type.is_multilabel() {
            if let Some(num_outputs) = config.loss_type.num_outputs() {
                return Self::train_binned_multilabel(dataset, config, num_outputs);
            }
        }

        config.validate().map_err(TreeBoostError::Config)?;

        // CRITICAL: Resolve BackendType::Auto to concrete backend ONCE based on full dataset
        // This prevents mixing CPU/GPU during train/val splits (bad for accuracy)
        if matches!(config.backend_type, crate::backend::BackendType::Auto) {
            let resolved =
                crate::backend::BackendSelector::with_config(crate::backend::BackendConfig {
                    preferred: crate::backend::BackendType::Auto,
                    ..Default::default()
                })
                .select(dataset.num_rows())?;

            // Update config with resolved backend (CUDA/WGPU/Scalar/etc)
            let resolved_type = match resolved.name() {
                "CUDA" => crate::backend::BackendType::Cuda,
                "WGPU" => crate::backend::BackendType::Wgpu,
                "Scalar (AVX2)" | "Scalar (NEON)" | "Scalar" => crate::backend::BackendType::Scalar,
                _ => crate::backend::BackendType::Scalar, // Fallback
            };
            config.backend_type = resolved_type;
        }

        let loss_fn = config.loss_type.create();
        let targets = dataset.targets();

        // Split data for validation (early stopping) and conformal calibration
        let split = split_holdout(
            dataset.num_rows(),
            config.validation_ratio,
            config.calibration_ratio,
            config.seed,
        );
        let (train_indices, validation_indices, calibration_indices) =
            (split.train, split.validation, split.calibration);

        // Compute base prediction from training data only
        let train_targets: Vec<f32> = train_indices.iter().map(|&i| targets[i]).collect();
        let base_prediction = loss_fn.initial_prediction(&train_targets);

        // Initialize predictions for all rows
        let mut predictions = vec![base_prediction; dataset.num_rows()];

        // Gradient and hessian buffers
        let mut gradients = vec![0.0f32; dataset.num_rows()];
        let mut hessians = vec![0.0f32; dataset.num_rows()];

        // Build interaction constraints from groups
        let interaction_constraints = if config.interaction_groups.is_empty() {
            InteractionConstraints::new()
        } else {
            InteractionConstraints::from_groups(
                config.interaction_groups.clone(),
                dataset.num_features(),
            )
        };

        // Create tree grower
        let tree_grower = TreeGrower::new()
            .with_max_depth(config.max_depth)
            .with_max_leaves(config.max_leaves)
            .with_lambda(config.lambda)
            .with_min_samples_leaf(config.min_samples_leaf)
            .with_min_hessian_leaf(config.min_hessian_leaf)
            .with_entropy_weight(config.entropy_weight)
            .with_min_gain(config.min_gain)
            .with_learning_rate(config.learning_rate)
            .with_colsample(config.colsample)
            .with_monotonic_constraints(config.monotonic_constraints.clone())
            .with_interaction_constraints(interaction_constraints)
            .with_backend(config.backend_type)
            .with_gpu_subgroups(config.use_gpu_subgroups)
            .with_era_splitting(config.era_splitting);

        let mut trees = Vec::with_capacity(config.num_rounds);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Early stopping state
        let early_stopping_enabled =
            config.early_stopping_rounds > 0 && !validation_indices.is_empty();
        let mut best_val_loss = f32::MAX;
        let mut rounds_without_improvement = 0;
        let mut best_num_trees = 0;

        // Pre-allocate reusable buffers for subsampling (avoid per-round allocation)
        let mut sample_indices: Vec<usize> = Vec::with_capacity(train_indices.len());
        let mut shuffle_buffer: Vec<usize> = if config.subsample < 1.0 && !config.goss_enabled {
            train_indices.clone() // Pre-allocate for random subsampling
        } else {
            Vec::new()
        };
        let mut goss_indexed: Vec<(usize, f32)> = if config.goss_enabled {
            Vec::with_capacity(train_indices.len())
        } else {
            Vec::new()
        };

        // Determine if we can use fused gradient+histogram (no subsampling)
        let use_fused = !config.goss_enabled && config.subsample >= 1.0;

        // Create Full GPU builders if applicable
        // For BackendType::Auto, we try CUDA first, then WGPU
        #[cfg(feature = "cuda")]
        let mut cuda_builder: Option<FullCudaTreeBuilder> =
            if use_fused && matches!(config.backend_type, BackendType::Cuda | BackendType::Auto) {
                use crate::backend::cuda::CudaDevice;
                CudaDevice::new().and_then(|d| {
                    // Resolve gpu_mode knowing we have CUDA available
                    let resolved = config.gpu_mode.resolve(BackendType::Cuda);
                    if matches!(resolved, GpuMode::Full) {
                        Some(FullCudaTreeBuilder::new(std::sync::Arc::new(d)))
                    } else {
                        None
                    }
                })
            } else {
                None
            };

        #[cfg(feature = "gpu")]
        let mut wgpu_builder: Option<FullGpuTreeBuilder> = if use_fused
            && matches!(config.backend_type, BackendType::Wgpu | BackendType::Auto)
            && {
                #[cfg(feature = "cuda")]
                {
                    cuda_builder.is_none() // Only use WGPU if CUDA not available/chosen
                }
                #[cfg(not(feature = "cuda"))]
                {
                    true
                }
            } {
            use crate::backend::wgpu::GpuDevice;
            GpuDevice::new().and_then(|d| {
                // Resolve gpu_mode knowing we have WGPU
                let resolved = config.gpu_mode.resolve(BackendType::Wgpu);
                if matches!(resolved, GpuMode::Full) {
                    Some(FullGpuTreeBuilder::new(std::sync::Arc::new(d)))
                } else {
                    None
                }
            })
        } else {
            None
        };

        Self::train_binned_impl(
            dataset,
            config,
            loss_fn,
            targets,
            &train_indices,
            &validation_indices,
            &calibration_indices,
            base_prediction,
            &mut predictions,
            &mut gradients,
            &mut hessians,
            tree_grower,
            &mut trees,
            &mut rng,
            early_stopping_enabled,
            &mut best_val_loss,
            &mut rounds_without_improvement,
            &mut best_num_trees,
            &mut sample_indices,
            &mut shuffle_buffer,
            &mut goss_indexed,
            use_fused,
            #[cfg(feature = "cuda")]
            &mut cuda_builder,
            #[cfg(feature = "gpu")]
            &mut wgpu_builder,
        )
    }

    /// Internal implementation of train_binned that handles tree growing loop
    #[allow(clippy::too_many_arguments)]
    fn train_binned_impl(
        dataset: &BinnedDataset,
        config: GBDTConfig,
        loss_fn: std::boxed::Box<dyn crate::loss::LossFunction>,
        targets: &[f32],
        train_indices: &[usize],
        validation_indices: &[usize],
        calibration_indices: &[usize],
        base_prediction: f32,
        predictions: &mut Vec<f32>,
        gradients: &mut Vec<f32>,
        hessians: &mut Vec<f32>,
        tree_grower: TreeGrower,
        trees: &mut Vec<Tree>,
        rng: &mut rand::rngs::StdRng,
        early_stopping_enabled: bool,
        best_val_loss: &mut f32,
        rounds_without_improvement: &mut usize,
        best_num_trees: &mut usize,
        sample_indices: &mut Vec<usize>,
        shuffle_buffer: &mut Vec<usize>,
        goss_indexed: &mut Vec<(usize, f32)>,
        use_fused: bool,
        #[cfg(feature = "cuda")] cuda_builder: &mut Option<FullCudaTreeBuilder>,
        #[cfg(feature = "gpu")] wgpu_builder: &mut Option<FullGpuTreeBuilder>,
    ) -> Result<Self> {
        let train_dataset = dataset;
        let val_targets: Vec<f32> = validation_indices.iter().map(|&i| targets[i]).collect();
        let mut val_predictions = vec![base_prediction; validation_indices.len()];

        for _round in 0..config.num_rounds {
            // Grow tree - either fused, Full GPU, or separate gradient+histogram paths
            #[allow(unused_mut, unused_assignments)]
            let mut tree: Option<Tree> = None;

            // Try Full GPU builders first (level-wise growth, all-GPU pipeline)
            #[cfg(feature = "cuda")]
            if tree.is_none() {
                if let Some(ref mut builder) = cuda_builder {
                    // Compute gradients for this round
                    for &idx in train_indices {
                        let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                        gradients[idx] = g;
                        hessians[idx] = h;
                    }
                    tree = Some(builder.build_tree(
                        dataset,
                        gradients,
                        hessians,
                        train_indices,
                        config.max_depth,
                        config.max_leaves,
                        config.lambda,
                        config.min_samples_leaf,
                        config.min_hessian_leaf,
                        config.min_gain,
                        config.learning_rate,
                    ));
                }
            }

            #[cfg(feature = "gpu")]
            if tree.is_none() {
                if let Some(ref mut builder) = wgpu_builder {
                    // Compute gradients for this round
                    for &idx in train_indices {
                        let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                        gradients[idx] = g;
                        hessians[idx] = h;
                    }
                    tree = Some(builder.build_tree(
                        dataset,
                        gradients,
                        hessians,
                        train_indices,
                        config.max_depth,
                        config.max_leaves,
                        config.lambda,
                        config.min_samples_leaf,
                        config.min_hessian_leaf,
                        config.min_gain,
                        config.learning_rate,
                    ));
                }
            }

            // Fall back to TreeGrower (Hybrid mode or CPU)
            let tree = match tree {
                Some(t) => t,
                None => {
                    if use_fused {
                        // FUSED PATH: Compute gradients AND build root histogram in single pass
                        tree_grower.grow_fused(
                            dataset,
                            train_indices,
                            targets,
                            predictions,
                            loss_fn.as_ref(),
                            gradients,
                            hessians,
                        )?
                    } else {
                        // SEPARATE PATH: Compute gradients first, then build histogram
                        // Compute gradients and hessians
                        if config.parallel_gradient {
                            train_indices.par_iter().for_each(|&idx| {
                                let (g, h) =
                                    loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                                // SAFETY: Each idx is unique within train_indices, so no data races
                                unsafe {
                                    let grad_ptr = gradients.as_ptr() as *mut f32;
                                    let hess_ptr = hessians.as_ptr() as *mut f32;
                                    *grad_ptr.add(idx) = g;
                                    *hess_ptr.add(idx) = h;
                                }
                            });
                        } else {
                            for &idx in train_indices {
                                let (g, h) =
                                    loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                                gradients[idx] = g;
                                hessians[idx] = h;
                            }
                        }

                        // GOSS or random subsampling
                        let tree_indices: &[usize] = if config.goss_enabled {
                            // GOSS: Gradient-based One-Side Sampling
                            sample_indices.clear();
                            Self::goss_sample_into(
                                train_indices,
                                gradients,
                                hessians,
                                config.goss_top_rate,
                                config.goss_other_rate,
                                rng,
                                goss_indexed,
                                sample_indices,
                            );
                            sample_indices
                        } else if config.subsample < 1.0 {
                            // Random subsampling (Stochastic Gradient Boosting)
                            sample_indices.clear();
                            let n_samples =
                                ((train_indices.len() as f32) * config.subsample).ceil() as usize;
                            shuffle_buffer.shuffle(rng);
                            sample_indices.extend_from_slice(&shuffle_buffer[..n_samples]);
                            sample_indices
                        } else {
                            train_indices
                        };

                        // Grow tree using the selected training indices
                        tree_grower.grow_with_indices(dataset, gradients, hessians, tree_indices)?
                    }
                }
            };

            // Update predictions with the new tree
            for idx in 0..predictions.len() {
                predictions[idx] += tree.predict(|f| dataset.get_bin(idx, f));
            }

            trees.push(tree);

            // Early stopping on validation set
            if early_stopping_enabled {
                // Compute loss on validation set
                for (i, &val_idx) in validation_indices.iter().enumerate() {
                    val_predictions[i] = predictions[val_idx];
                }

                let mut val_loss = 0.0f32;
                for (&pred, &target) in val_predictions.iter().zip(val_targets.iter()) {
                    let (g, _h) = loss_fn.gradient_hessian(target, pred);
                    val_loss += g.abs();
                }
                val_loss /= validation_indices.len().max(1) as f32;

                if val_loss < *best_val_loss {
                    *best_val_loss = val_loss;
                    *best_num_trees = trees.len();
                    *rounds_without_improvement = 0;
                } else {
                    *rounds_without_improvement += 1;
                    if should_early_stop(
                        *rounds_without_improvement,
                        trees.len(),
                        config.early_stopping_rounds,
                        config.min_early_stopping_trees,
                    ) {
                        trees.truncate(early_stop_keep_count(
                            *best_num_trees,
                            config.min_early_stopping_trees,
                        ));
                        break;
                    }
                }
            }
        }

        // Truncate if we finished all rounds but best was earlier
        if early_stopping_enabled && *best_num_trees > 0 && *best_num_trees < trees.len() {
            trees.truncate(early_stop_keep_count(
                *best_num_trees,
                config.min_early_stopping_trees,
            ));
        }

        // Column reordering
        let column_permutation = if config.column_reordering && !trees.is_empty() {
            let importances =
                Self::compute_importances_from_trees(trees, train_dataset.num_features());
            Some(ColumnPermutation::from_importances(&importances))
        } else {
            None
        };

        // Compute conformal quantile from calibration set residuals
        // Use calibration set if available, otherwise fall back to validation set
        let conformal_q = if !calibration_indices.is_empty() {
            let calib_residuals: Vec<f32> = calibration_indices
                .iter()
                .map(|&idx| (targets[idx] - predictions[idx]).abs())
                .collect();
            Some(Self::compute_quantile(
                &calib_residuals,
                config.conformal_quantile,
            ))
        } else if !val_targets.is_empty() {
            // Fall back to validation set if no calibration set
            let residuals: Vec<f32> = val_targets
                .iter()
                .zip(val_predictions.iter())
                .map(|(&target, &pred)| (target - pred).abs())
                .collect();
            Some(Self::compute_quantile(
                &residuals,
                config.conformal_quantile,
            ))
        } else {
            None
        };

        // Determine output type from loss configuration
        let output_type = config.loss_type.output_type();

        #[allow(deprecated)]
        Ok(Self {
            config,
            // New unified fields
            base_predictions: vec![base_prediction],
            trees: trees.clone(),
            num_outputs: 1,
            output_type,
            conformal_q,
            feature_info: train_dataset.all_feature_info().to_vec(),
            column_permutation,
            // Legacy fields (for backward compatibility)
            base_prediction,
            base_predictions_multiclass: Vec::new(),
            num_classes: 0,
        })
    }

    /// Train a multi-class classification model from pre-binned data
    ///
    /// This trains K trees per round (one per class) and combines predictions
    /// via softmax for final class probabilities.
    fn train_binned_multiclass(
        dataset: &BinnedDataset,
        config: GBDTConfig,
        num_classes: usize,
    ) -> Result<Self> {
        config.validate().map_err(TreeBoostError::Config)?;

        let targets = dataset.targets();
        let multiclass_loss = MultiClassLogLoss::new(num_classes);

        // Split data for validation and calibration
        let split = split_holdout(
            dataset.num_rows(),
            config.validation_ratio,
            config.calibration_ratio,
            config.seed,
        );
        let (train_indices, validation_indices, _calibration_indices) =
            (split.train, split.validation, split.calibration);

        // Compute initial predictions per class from training data
        let train_targets: Vec<f32> = train_indices.iter().map(|&i| targets[i]).collect();
        let base_predictions = multiclass_loss.initial_predictions(&train_targets);

        // Initialize predictions for all rows: predictions[row * num_classes + class]
        let num_rows = dataset.num_rows();
        let mut predictions: Vec<f32> = Vec::with_capacity(num_rows * num_classes);
        for _ in 0..num_rows {
            predictions.extend_from_slice(&base_predictions);
        }

        // Gradient and hessian buffers (per sample, used for one class at a time)
        let mut gradients = vec![0.0f32; num_rows];
        let mut hessians = vec![0.0f32; num_rows];

        // Build interaction constraints
        let interaction_constraints = if config.interaction_groups.is_empty() {
            InteractionConstraints::new()
        } else {
            InteractionConstraints::from_groups(
                config.interaction_groups.clone(),
                dataset.num_features(),
            )
        };

        // Create tree grower
        let tree_grower = TreeGrower::new()
            .with_max_depth(config.max_depth)
            .with_max_leaves(config.max_leaves)
            .with_lambda(config.lambda)
            .with_min_samples_leaf(config.min_samples_leaf)
            .with_min_hessian_leaf(config.min_hessian_leaf)
            .with_entropy_weight(config.entropy_weight)
            .with_min_gain(config.min_gain)
            .with_learning_rate(config.learning_rate)
            .with_colsample(config.colsample)
            .with_monotonic_constraints(config.monotonic_constraints.clone())
            .with_interaction_constraints(interaction_constraints)
            .with_backend(config.backend_type)
            .with_gpu_subgroups(config.use_gpu_subgroups)
            .with_era_splitting(config.era_splitting);

        // Trees stored as: [round0_class0, round0_class1, ..., round0_classK, round1_class0, ...]
        let mut trees = Vec::with_capacity(config.num_rounds * num_classes);

        // Early stopping state
        let early_stopping_enabled =
            config.early_stopping_rounds > 0 && !validation_indices.is_empty();
        let mut best_val_loss = f32::MAX;
        let mut rounds_without_improvement = 0;
        let mut best_num_rounds = 0;

        for round in 0..config.num_rounds {
            // Train K trees for this round (one per class)
            for class_idx in 0..num_classes {
                // Compute gradients and hessians for this class using batch method
                multiclass_loss.compute_gradients_batch(
                    class_idx,
                    targets,
                    &predictions,
                    &train_indices,
                    &mut gradients,
                    &mut hessians,
                );

                // Grow tree for this class
                let tree = tree_grower.grow_with_indices(
                    dataset,
                    &gradients,
                    &hessians,
                    &train_indices,
                )?;

                // Update predictions for this class
                for idx in 0..num_rows {
                    let delta = tree.predict(|f| dataset.get_bin(idx, f));
                    predictions[idx * num_classes + class_idx] += delta;
                }

                trees.push(tree);
            }

            // Early stopping check on validation set
            if early_stopping_enabled {
                // Compute multi-class log loss on validation set
                let mut val_loss = 0.0f32;
                for &idx in &validation_indices {
                    let target_class = targets[idx] as usize;
                    let row_preds = &predictions[idx * num_classes..(idx + 1) * num_classes];
                    let probs = softmax(row_preds);
                    // Negative log likelihood for true class
                    val_loss -= probs[target_class].max(1e-15).ln();
                }
                val_loss /= validation_indices.len() as f32;

                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    best_num_rounds = round + 1;
                    rounds_without_improvement = 0;
                } else {
                    rounds_without_improvement += 1;
                    // Use actual tree count (not round count) for consistency with binary/regression
                    if should_early_stop(
                        rounds_without_improvement,
                        trees.len(),
                        config.early_stopping_rounds,
                        config.min_early_stopping_trees,
                    ) {
                        let keep_rounds = early_stop_keep_count(
                            best_num_rounds,
                            config.min_early_stopping_trees / num_classes.max(1),
                        );
                        trees.truncate(keep_rounds * num_classes);
                        break;
                    }
                }
            }
        }

        // Truncate if early stopping finished all rounds but best was earlier
        if early_stopping_enabled
            && best_num_rounds > 0
            && best_num_rounds * num_classes < trees.len()
        {
            let keep_rounds = early_stop_keep_count(
                best_num_rounds,
                config.min_early_stopping_trees / num_classes.max(1),
            );
            trees.truncate(keep_rounds * num_classes);
        }

        // Compute column permutation if enabled
        let column_permutation = if config.column_reordering && !trees.is_empty() {
            let importances = Self::compute_importances_from_trees(&trees, dataset.num_features());
            Some(ColumnPermutation::from_importances(&importances))
        } else {
            None
        };

        #[allow(deprecated)]
        Ok(Self {
            config,
            // New unified fields
            base_predictions: base_predictions.clone(),
            trees,
            num_outputs: num_classes,
            output_type: crate::booster::OutputType::MultiClass,
            conformal_q: None, // Conformal not supported for multi-class yet
            feature_info: dataset.all_feature_info().to_vec(),
            column_permutation,
            // Legacy fields (for backward compatibility)
            base_prediction: 0.0,
            base_predictions_multiclass: base_predictions,
            num_classes,
        })
    }

    /// Train a multi-label classification model from pre-binned data
    ///
    /// Multi-label differs from multi-class in that each sample can belong to
    /// multiple labels simultaneously. Uses sigmoid per label (not softmax).
    ///
    /// # Architecture
    ///
    /// This follows the same K-trees-per-round pattern as multi-class training:
    /// - Trains N trees per round (one per label)
    /// - Trees stored as: `[round0_label0, round0_label1, ..., roundR_labelN]`
    /// - Each tree produces scalar predictions for its specific label
    /// - Final prediction: sigmoid applied per label for probabilities
    ///
    /// # Arguments
    /// * `dataset` - Multi-output binned dataset (targets row-wise flattened)
    /// * `config` - Training configuration with multi-label loss
    /// * `num_outputs` - Number of labels/outputs
    fn train_binned_multilabel(
        dataset: &BinnedDataset,
        config: GBDTConfig,
        num_outputs: usize,
    ) -> Result<Self> {
        config.validate().map_err(TreeBoostError::Config)?;

        let targets = dataset.targets();
        let num_rows = dataset.num_rows();

        // Verify dataset has correct target dimensions
        if targets.len() != num_rows * num_outputs {
            return Err(TreeBoostError::Config(format!(
                "Dataset targets length {} doesn't match num_rows * num_outputs ({} * {} = {}). \
                 Use BinnedDataset::new_multioutput() for multi-label data.",
                targets.len(),
                num_rows,
                num_outputs,
                num_rows * num_outputs
            )));
        }

        // Create the loss function
        let loss_fn = config.loss_type.create();

        // Split data for validation and calibration
        let split = split_holdout(
            num_rows,
            config.validation_ratio,
            config.calibration_ratio,
            config.seed,
        );
        let (train_indices, validation_indices, _calibration_indices) =
            (split.train, split.validation, split.calibration);

        // Compute initial predictions per label from training data
        let train_targets: Vec<f32> = train_indices
            .iter()
            .flat_map(|&i| {
                (0..num_outputs).map(move |k| targets[i * num_outputs + k])
            })
            .collect();
        let base_predictions = loss_fn.initial_predictions_multi(&train_targets, num_outputs);

        // Initialize predictions for all rows: predictions[row * num_outputs + label]
        let mut predictions: Vec<f32> = Vec::with_capacity(num_rows * num_outputs);
        for _ in 0..num_rows {
            predictions.extend_from_slice(&base_predictions);
        }

        // Gradient and hessian buffers (per sample, used for one label at a time)
        let mut gradients = vec![0.0f32; num_rows];
        let mut hessians = vec![0.0f32; num_rows];

        // Build interaction constraints
        let interaction_constraints = if config.interaction_groups.is_empty() {
            InteractionConstraints::new()
        } else {
            InteractionConstraints::from_groups(
                config.interaction_groups.clone(),
                dataset.num_features(),
            )
        };

        // Create tree grower
        let tree_grower = TreeGrower::new()
            .with_max_depth(config.max_depth)
            .with_max_leaves(config.max_leaves)
            .with_lambda(config.lambda)
            .with_min_samples_leaf(config.min_samples_leaf)
            .with_min_hessian_leaf(config.min_hessian_leaf)
            .with_entropy_weight(config.entropy_weight)
            .with_min_gain(config.min_gain)
            .with_learning_rate(config.learning_rate)
            .with_colsample(config.colsample)
            .with_monotonic_constraints(config.monotonic_constraints.clone())
            .with_interaction_constraints(interaction_constraints)
            .with_backend(config.backend_type)
            .with_gpu_subgroups(config.use_gpu_subgroups)
            .with_era_splitting(config.era_splitting);

        // Trees stored as: [round0_label0, round0_label1, ..., round0_labelN, round1_label0, ...]
        let mut trees = Vec::with_capacity(config.num_rounds * num_outputs);

        // Early stopping state
        let early_stopping_enabled =
            config.early_stopping_rounds > 0 && !validation_indices.is_empty();
        let mut best_val_loss = f32::MAX;
        let mut rounds_without_improvement = 0;
        let mut best_num_rounds = 0;

        for round in 0..config.num_rounds {
            // Train N trees for this round (one per label)
            for label_idx in 0..num_outputs {
                // Compute gradients and hessians for this label
                // For multi-label, each label is independent binary classification
                for &idx in &train_indices {
                    let target = targets[idx * num_outputs + label_idx];
                    let pred = predictions[idx * num_outputs + label_idx];
                    let (g, h) = loss_fn.gradient_hessian(target, pred);
                    gradients[idx] = g;
                    hessians[idx] = h;
                }

                // Grow tree for this label
                let tree = tree_grower.grow_with_indices(
                    dataset,
                    &gradients,
                    &hessians,
                    &train_indices,
                )?;

                // Update predictions for this label
                for idx in 0..num_rows {
                    let delta = tree.predict(|f| dataset.get_bin(idx, f));
                    predictions[idx * num_outputs + label_idx] += delta;
                }

                trees.push(tree);
            }

            // Early stopping check on validation set
            if early_stopping_enabled {
                // Compute multi-label loss on validation set (sum of per-label losses)
                let mut val_loss = 0.0f32;
                for &idx in &validation_indices {
                    for label_idx in 0..num_outputs {
                        let target = targets[idx * num_outputs + label_idx];
                        let pred = predictions[idx * num_outputs + label_idx];
                        val_loss += loss_fn.loss(target, pred);
                    }
                }
                val_loss /= (validation_indices.len() * num_outputs) as f32;

                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    best_num_rounds = round + 1;
                    rounds_without_improvement = 0;
                } else {
                    rounds_without_improvement += 1;
                    if should_early_stop(
                        rounds_without_improvement,
                        trees.len(),
                        config.early_stopping_rounds,
                        config.min_early_stopping_trees,
                    ) {
                        let keep_rounds = early_stop_keep_count(
                            best_num_rounds,
                            config.min_early_stopping_trees / num_outputs.max(1),
                        );
                        trees.truncate(keep_rounds * num_outputs);
                        break;
                    }
                }
            }
        }

        // Truncate if early stopping finished all rounds but best was earlier
        if early_stopping_enabled
            && best_num_rounds > 0
            && best_num_rounds * num_outputs < trees.len()
        {
            let keep_rounds = early_stop_keep_count(
                best_num_rounds,
                config.min_early_stopping_trees / num_outputs.max(1),
            );
            trees.truncate(keep_rounds * num_outputs);
        }

        // Compute column permutation if enabled
        let column_permutation = if config.column_reordering && !trees.is_empty() {
            let importances = Self::compute_importances_from_trees(&trees, dataset.num_features());
            Some(ColumnPermutation::from_importances(&importances))
        } else {
            None
        };

        #[allow(deprecated)]
        Ok(Self {
            config,
            // New unified fields
            base_predictions: base_predictions.clone(),
            trees,
            num_outputs,
            output_type: crate::booster::OutputType::MultiLabel,
            conformal_q: None, // Conformal not supported for multi-label yet
            feature_info: dataset.all_feature_info().to_vec(),
            column_permutation,
            // Legacy fields (for backward compatibility)
            base_prediction: 0.0,
            base_predictions_multiclass: base_predictions,
            num_classes: 0, // Not multi-class
        })
    }

    /// Compute feature importances from a collection of trees (internal helper)
    fn compute_importances_from_trees(trees: &[Tree], num_features: usize) -> Vec<f32> {
        let mut importances = vec![0.0f32; num_features];

        for tree in trees {
            for (_, node) in tree.internal_nodes() {
                if let Some((feature_idx, _, _, _, _)) = node.split_info() {
                    importances[feature_idx] += node.sum_hessians;
                }
            }
        }

        // Normalize
        let total: f32 = importances.iter().sum();
        if total > 0.0 {
            for imp in &mut importances {
                *imp /= total;
            }
        }

        importances
    }

    /// GOSS (Gradient-based One-Side Sampling) with buffer reuse
    ///
    /// Selects samples based on gradient magnitude:
    /// 1. Keep all top `top_rate` samples with largest |gradient|
    /// 2. Randomly sample `other_rate` from the remaining samples
    /// 3. Apply weight correction (1 - top_rate) / other_rate to sampled small-gradient samples
    ///
    /// Weight correction is applied in-place to gradients and hessians.
    /// Uses partial sorting (select_nth_unstable) for O(n) instead of O(n log n).
    ///
    /// This version reuses pre-allocated buffers to avoid per-round allocation.
    #[allow(clippy::too_many_arguments)]
    fn goss_sample_into(
        train_indices: &[usize],
        gradients: &mut [f32],
        hessians: &mut [f32],
        top_rate: f32,
        other_rate: f32,
        rng: &mut rand::rngs::StdRng,
        indexed_buffer: &mut Vec<(usize, f32)>,
        result: &mut Vec<usize>,
    ) {
        let n = train_indices.len();
        if n == 0 {
            return;
        }

        // Number of top-gradient samples to keep
        let n_top = ((n as f32) * top_rate).ceil() as usize;
        let n_top = n_top.min(n);
        // Number of other samples to randomly select
        let n_other = ((n as f32) * other_rate).ceil() as usize;

        // Reuse indexed buffer - clear and repopulate
        indexed_buffer.clear();
        indexed_buffer.extend(train_indices.iter().map(|&idx| (idx, gradients[idx].abs())));

        // Partition around the n_top-th largest element (descending order)
        if n_top < n {
            indexed_buffer.select_nth_unstable_by(n_top, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Add top n_top samples (large gradients) - no weight modification needed
        result.extend(indexed_buffer[..n_top].iter().map(|(idx, _)| *idx));

        // For small gradients: shuffle the rest portion in-place and take n_other
        let rest_slice = &mut indexed_buffer[n_top..];
        rest_slice.shuffle(rng);
        let n_rest = rest_slice.len().min(n_other);

        // Weight correction factor for small-gradient samples
        let weight = (1.0 - top_rate) / other_rate;

        // Apply weight correction and add to result
        for &(idx, _) in &rest_slice[..n_rest] {
            gradients[idx] *= weight;
            hessians[idx] *= weight;
            result.push(idx);
        }
    }

    /// Compute quantile of a sorted slice
    fn compute_quantile(values: &[f32], q: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        // Filter out NaN values before sorting
        let mut sorted: Vec<f32> = values.iter().copied().filter(|v| !v.is_nan()).collect();

        if sorted.is_empty() {
            return 0.0;
        }

        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((sorted.len() as f32) * q).ceil() as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Train a GBDT model with separate train/validation sets
    ///
    /// This is useful when you have pre-split data or want more control over validation.
    pub fn train_binned_with_validation(
        train_data: &BinnedDataset,
        _val_data: &BinnedDataset,
        val_targets: &[f32],
        config: GBDTConfig,
    ) -> Result<Self> {
        // For now, just train on the combined data and validate
        // In a full implementation, this would handle separate datasets properly
        // This maintains backward compatibility with TunableModel trait

        // Create a combined dataset (not ideal, but maintains API compatibility)
        let mut train_targets_vec = train_data.targets().to_vec();
        train_targets_vec.extend_from_slice(val_targets);

        // For this simple implementation, just train on training data
        Self::train_binned(train_data, config)
    }
}
