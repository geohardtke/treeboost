//! UniversalModel core implementation
//!
//! This module contains the UniversalModel struct and all its implementations.
//! Types like BoostingMode, ModeSelection, and UniversalConfig are defined in
//! separate submodules (mode.rs, config.rs) for better organization.

use super::{BoostingMode, ModeSelection, UniversalConfig};
use crate::analysis::{AnalysisConfig, Confidence, DatasetAnalysis};
use crate::booster::{GBDTConfig, GBDTModel};
use crate::dataset::BinnedDataset;
use crate::learner::{LinearBooster, TreeBooster, WeakLearner};
use crate::loss::LossFunction;
use crate::tree::Tree;
use crate::utils::features::extract_selected_features;
use crate::Result;
use rand::SeedableRng;
use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

// Note: BoostingMode, ModeSelection, and UniversalConfig are now defined in
// separate modules (mode.rs and config.rs) and imported via super::.

// =============================================================================
// UniversalModel
// =============================================================================

/// Use [`UniversalModel::auto()`] to let TreeBoost analyze your data and pick the best mode.
/// The analysis result is stored and can be retrieved with [`UniversalModel::analysis()`].
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct UniversalModel {
    /// Training configuration
    config: UniversalConfig,

    /// GBDTModel for PureTree mode (wraps the mature implementation)
    /// Used when ensemble_seeds is None
    gbdt_model: Option<GBDTModel>,

    /// Multi-seed GBDT ensemble (when ensemble_seeds is Some)
    /// - For PureTree mode: Multiple GBDTs trained directly
    /// - For LinearThenTree mode: Multiple GBDTs trained on linear residuals
    /// - For RandomForest mode: Not used (RF already uses multiple trees)
    gbdt_ensemble: Option<Vec<GBDTModel>>,

    /// Stacker weights for ensemble combination
    /// Only populated when gbdt_ensemble.is_some()
    stacker_weights: Option<Vec<f32>>,

    /// Stacker intercept for Ridge stacking
    /// Only populated when using Ridge stacking strategy
    stacker_intercept: Option<f32>,

    /// Linear booster (for LinearThenTree mode)
    linear_booster: Option<LinearBooster>,

    /// Ensemble of trained trees (for LinearThenTree and RandomForest modes)
    /// Used when NOT using gbdt_model or gbdt_ensemble
    trees: Vec<Tree>,

    /// Base prediction (for LinearThenTree and RandomForest modes)
    base_prediction: f32,

    /// Number of features
    num_features: usize,

    /// Analysis result (if auto mode was used)
    ///
    /// Stores the dataset analysis that led to mode selection.
    /// Use `analysis()` to retrieve and `analysis_report()` to get a formatted report.
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    analysis: Option<DatasetAnalysis>,

    /// Raw features for LinearThenTree prediction (optional)
    ///
    /// When LTT is trained with raw features, we store them for prediction.
    /// This avoids the lossy bin-center approximation.
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    raw_features_for_linear: Option<Vec<f32>>,

    /// Feature indices to use for linear model (optional)
    ///
    /// When set, only these feature indices from raw_features are used for
    /// the linear model. This allows feature selection for linear while
    /// trees use all features.
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    linear_feature_indices: Option<Vec<usize>>,

    /// Number of features used by linear model (may differ from num_features)
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    num_linear_features: Option<usize>,

    /// Feature extractor for LinearThenTree inference (optional)
    ///
    /// Stores feature extraction configuration (which columns to exclude,
    /// auto-exclusion settings) used during training. This ensures consistent
    /// feature extraction during inference for linear component.
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
}

impl UniversalModel {
    /// Set feature extractor for LinearThenTree inference
    ///
    /// This is called after training to store the feature extraction configuration
    /// (which columns to exclude, auto-exclusion settings). This ensures consistent
    /// feature extraction during inference for the linear component.
    pub fn with_feature_extractor(
        mut self,
        extractor: crate::dataset::feature_extractor::FeatureExtractor,
    ) -> Self {
        self.feature_extractor = Some(extractor);
        self
    }

    /// Get the feature extractor (if set)
    pub fn feature_extractor(
        &self,
    ) -> Option<&crate::dataset::feature_extractor::FeatureExtractor> {
        self.feature_extractor.as_ref()
    }

    /// Apply linear model shrinkage factor to predictions (batch)
    ///
    /// In LinearThenTree mode, the linear model's contribution to the ensemble
    /// is weighted by `shrinkage_factor`:
    ///
    /// ```text
    /// ensemble_pred = base + shrinkage_factor * linear_pred + tree_pred
    /// ```
    ///
    /// This shrinkage factor controls ensemble weighting (not optimization step size).
    /// See `LinearConfig::shrinkage_factor` for details.
    ///
    /// # Arguments
    /// * `predictions` - Mutable slice of predictions to update
    /// * `linear_preds` - Linear model predictions to add (with shrinkage applied)
    #[inline]
    fn apply_linear_shrinkage(&self, predictions: &mut [f32], linear_preds: &[f32]) {
        let shrinkage = self.config.linear_config.shrinkage_factor;
        let len = predictions.len().min(linear_preds.len());
        for i in 0..len {
            predictions[i] += shrinkage * linear_preds[i];
        }
    }

    /// Apply linear model shrinkage factor to a single prediction
    ///
    /// Applies the same ensemble weighting logic as `apply_linear_shrinkage`
    /// but for a single scalar value.
    ///
    /// # Arguments
    /// * `linear_pred` - Raw linear prediction
    ///
    /// # Returns
    /// Linear prediction scaled by shrinkage factor
    #[inline]
    fn apply_linear_shrinkage_scalar(&self, linear_pred: f32) -> f32 {
        self.config.linear_config.shrinkage_factor * linear_pred
    }

    /// Train a UniversalModel on binned data
    ///
    /// # Arguments
    /// * `dataset` - Binned training data
    /// * `config` - Training configuration (fully serializable and persisted)
    /// * `loss_fn` - Loss function for computing gradients during training
    ///
    /// # Important Notes
    ///
    /// - The **loss function is NOT persisted** in the saved model. It is used only
    ///   during training to compute gradients. The trained model is loss-function-agnostic
    ///   and will work correctly with any data processed with the same preprocessing.
    /// - The **config is fully persisted** and can be exported via [`config()`] or
    ///   saved to JSON for inspection and reuse.
    /// - For reproducibility, store your loss function choice separately if needed.
    ///
    /// # Example
    /// ```ignore
    /// // Train with MSE loss
    /// let model = UniversalModel::train(&dataset, config, &MseLoss)?;
    ///
    /// // Save config and model
    /// let config_json = serde_json::to_string_pretty(model.config())?;
    /// std::fs::write("config.json", config_json)?;
    /// model.save("model.rkyv")?;
    ///
    /// // Later: Load and use (loss function is already baked into the model)
    /// let loaded = UniversalModel::load("model.rkyv")?;
    /// let preds = loaded.predict(&test_dataset)?;  // No need to specify loss again
    /// ```
    pub fn train(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        // Validate configuration
        if config.num_rounds == 0 {
            return Err(crate::TreeBoostError::Config(
                "num_rounds must be greater than 0".to_string(),
            ));
        }
        if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "learning_rate must be in (0, 1], got {}",
                config.learning_rate
            )));
        }
        if config.subsample <= 0.0 || config.subsample > 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "subsample must be in (0, 1], got {}",
                config.subsample
            )));
        }

        let feature_extractor = config.feature_extractor.clone();
        match config.mode {
            BoostingMode::PureTree => {
                Self::train_pure_tree(dataset, config, loss_fn, None, feature_extractor)
            }
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(
                dataset,
                None,
                None,
                config,
                loss_fn,
                None,
                feature_extractor,
            ),
            BoostingMode::RandomForest => {
                Self::train_random_forest(dataset, config, loss_fn, None, feature_extractor)
            }
        }
    }

    /// Train LinearThenTree with raw features (recommended for best accuracy)
    ///
    /// For LinearThenTree mode, passing raw (unbinned) features significantly improves
    /// the linear model's accuracy. Without raw features, LTT uses bin-center approximations
    /// which lose precision that linear models need.
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset (for tree training)
    /// * `raw_features` - Original features, row-major f32 array (num_rows * num_features)
    /// * `config` - Training configuration
    /// * `loss_fn` - Loss function
    ///
    /// # Example
    /// ```ignore
    /// let model = UniversalModel::train_with_raw_features(
    ///     &binned_dataset,
    ///     &scaled_features,  // Original StandardScaler'd features
    ///     config,
    ///     &MseLoss,
    /// )?;
    /// ```
    pub fn train_with_raw_features(
        dataset: &BinnedDataset,
        raw_features: &[f32],
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        let feature_extractor = config.feature_extractor.clone();
        match config.mode {
            BoostingMode::PureTree => {
                Self::train_pure_tree(dataset, config, loss_fn, None, feature_extractor)
            }
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(
                dataset,
                Some(raw_features),
                None,
                config,
                loss_fn,
                None,
                feature_extractor,
            ),
            BoostingMode::RandomForest => {
                Self::train_random_forest(dataset, config, loss_fn, None, feature_extractor)
            }
        }
    }

    /// Train LinearThenTree with feature selection for linear model
    ///
    /// This allows using a curated subset of features for the linear model
    /// while trees use all features. This can improve linear generalization
    /// by excluding meaningless features (like row IDs) from linear.
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset (for tree training with all features)
    /// * `raw_features` - All features, row-major f32 array
    /// * `linear_feature_indices` - Which feature indices to use for linear model
    /// * `config` - Training configuration
    /// * `loss_fn` - Loss function
    pub fn train_with_linear_feature_selection(
        dataset: &BinnedDataset,
        raw_features: &[f32],
        linear_feature_indices: &[usize],
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        // Validate configuration
        if config.num_rounds == 0 {
            return Err(crate::TreeBoostError::Config(
                "num_rounds must be greater than 0".to_string(),
            ));
        }
        if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "learning_rate must be in (0, 1], got {}",
                config.learning_rate
            )));
        }
        if config.subsample <= 0.0 || config.subsample > 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "subsample must be in (0, 1], got {}",
                config.subsample
            )));
        }

        let feature_extractor = config.feature_extractor.clone();
        match config.mode {
            BoostingMode::PureTree => {
                Self::train_pure_tree(dataset, config, loss_fn, None, feature_extractor)
            }
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(
                dataset,
                Some(raw_features),
                Some(linear_feature_indices),
                config,
                loss_fn,
                None,
                feature_extractor,
            ),
            BoostingMode::RandomForest => {
                Self::train_random_forest(dataset, config, loss_fn, None, feature_extractor)
            }
        }
    }

    // =========================================================================
    // Automatic Mode Selection
    // =========================================================================

    /// Train with automatic mode selection
    ///
    /// This is TreeBoost's "smart" entry point. It:
    /// 1. Analyzes your dataset (lightweight probes on subsamples)
    /// 2. Picks the best boosting mode with confidence score
    /// 3. Trains the model with optimal settings
    /// 4. Stores the analysis for inspection
    ///
    /// # Example
    ///
    /// ```ignore
    /// use treeboost::{UniversalModel, MseLoss};
    ///
    /// let model = UniversalModel::auto(&dataset, &MseLoss)?;
    ///
    /// // See what mode was selected and why
    /// println!("Mode: {:?}", model.mode());
    /// println!("Confidence: {:?}", model.selection_confidence());
    /// println!("{}", model.analysis_report().unwrap());
    /// ```
    ///
    /// # When to Use
    ///
    /// Use `auto()` when:
    /// - You're not sure which mode is best for your data
    /// - You want TreeBoost to explain its decision
    /// - You want a simple one-liner that "just works"
    ///
    /// Use `train()` when:
    /// - You know the best mode for your data
    /// - You need fine-grained control over configuration
    /// - You're running benchmarks and want deterministic mode
    pub fn auto(dataset: &BinnedDataset, loss_fn: &dyn LossFunction) -> Result<Self> {
        Self::auto_with_config(dataset, UniversalConfig::default(), loss_fn)
    }

    /// Train with automatic mode selection and custom configuration
    ///
    /// Like `auto()`, but lets you customize other settings (num_rounds, tree config, etc.).
    /// The mode will be overridden by the analysis recommendation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_num_rounds(200)
    ///     .with_learning_rate(0.05);
    ///
    /// let model = UniversalModel::auto_with_config(&dataset, config, &MseLoss)?;
    /// ```
    pub fn auto_with_config(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        Self::auto_with_analysis_config(dataset, config, AnalysisConfig::default(), loss_fn)
    }

    /// Train with automatic mode selection and custom analysis configuration
    ///
    /// Full control over both model config and analysis settings.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = UniversalConfig::new().with_num_rounds(200);
    /// let analysis_config = AnalysisConfig::fast(); // Quick analysis
    ///
    /// let model = UniversalModel::auto_with_analysis_config(
    ///     &dataset, config, analysis_config, &MseLoss
    /// )?;
    /// ```
    pub fn auto_with_analysis_config(
        dataset: &BinnedDataset,
        mut config: UniversalConfig,
        analysis_config: AnalysisConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        // Step 1: Analyze the dataset
        let analysis = DatasetAnalysis::analyze_with_config(dataset, analysis_config)?;

        // Step 2: Get the recommended mode
        let recommended_mode = analysis.recommend_mode();

        // Step 3: Update config with recommended mode
        config.mode = recommended_mode;

        // Step 4: Train with the recommended mode
        let model = match config.mode {
            BoostingMode::PureTree => {
                Self::train_pure_tree(dataset, config, loss_fn, Some(analysis), None)
            }
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(
                dataset,
                None,
                None,
                config,
                loss_fn,
                Some(analysis),
                None,
            ),
            BoostingMode::RandomForest => {
                Self::train_random_forest(dataset, config, loss_fn, Some(analysis), None)
            }
        }?;

        Ok(model)
    }

    /// Train using a ModeSelection strategy
    ///
    /// This is the most flexible entry point, supporting:
    /// - `ModeSelection::Auto` - Automatic analysis and selection
    /// - `ModeSelection::AutoWithConfig(config)` - Auto with custom analysis
    /// - `ModeSelection::Fixed(mode)` - Explicit mode specification
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Auto mode
    /// let model = UniversalModel::train_with_selection(
    ///     &dataset, config, ModeSelection::Auto, &MseLoss
    /// )?;
    ///
    /// // Fixed mode
    /// let model = UniversalModel::train_with_selection(
    ///     &dataset, config, ModeSelection::Fixed(BoostingMode::LinearThenTree), &MseLoss
    /// )?;
    /// ```
    pub fn train_with_selection(
        dataset: &BinnedDataset,
        mut config: UniversalConfig,
        selection: ModeSelection,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        match selection {
            ModeSelection::Auto => Self::auto_with_config(dataset, config, loss_fn),
            ModeSelection::AutoWithConfig(analysis_config) => {
                Self::auto_with_analysis_config(dataset, config, analysis_config, loss_fn)
            }
            ModeSelection::Fixed(mode) => {
                config.mode = mode;
                Self::train(dataset, config, loss_fn)
            }
        }
    }

    // =========================================================================
    // Config Conversion
    // =========================================================================

    /// Convert UniversalConfig to GBDTConfig for delegation to GBDTModel
    pub(crate) fn to_gbdt_config(
        config: &UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<GBDTConfig> {
        let mut gbdt_config = GBDTConfig::new()
            .with_num_rounds(config.num_rounds)
            .with_learning_rate(config.learning_rate)
            .with_max_depth(config.tree_config.max_depth)
            .with_max_leaves(config.tree_config.max_leaves)
            .with_lambda(config.tree_config.lambda)
            .with_entropy_weight(config.tree_config.entropy_weight) // Pass entropy regularization
            .with_subsample(config.subsample)?
            .with_backend(config.backend_type) // Pass backend type
            .with_seed(config.seed);

        // Early stopping
        if config.early_stopping_rounds > 0 && config.validation_ratio > 0.0 {
            gbdt_config = gbdt_config
                .with_early_stopping(config.early_stopping_rounds, config.validation_ratio)?;
        }

        // Conformal calibration
        if config.calibration_ratio > 0.0 {
            gbdt_config.calibration_ratio = config.calibration_ratio;
            gbdt_config.conformal_quantile = config.conformal_quantile;
        }

        // Set loss type based on loss_fn type name (best effort)
        let loss_name = std::any::type_name_of_val(loss_fn);
        if loss_name.contains("PseudoHuber") {
            gbdt_config = gbdt_config.with_pseudo_huber_loss(1.0);
        } else if loss_name.contains("BinaryLogLoss") || loss_name.contains("LogLoss") {
            gbdt_config = gbdt_config.with_binary_logloss();
        }
        // Default is MSE which is already the default

        Ok(gbdt_config)
    }

    // =========================================================================
    // PureTree Mode - Delegates to GBDTModel
    // =========================================================================

    fn train_pure_tree(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        analysis: Option<DatasetAnalysis>,
        feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
    ) -> Result<Self> {
        let num_features = dataset.num_features();

        // Check if ensemble training is requested
        let (gbdt_model, gbdt_ensemble, stacker_weights, stacker_intercept) =
            if let Some(ref seeds) = config.ensemble_seeds {
                // Multi-seed ensemble training
                Self::train_gbdt_ensemble(dataset, &config, loss_fn, seeds)?
            } else {
                // Single GBDT training (standard path)
                let gbdt_config = Self::to_gbdt_config(&config, loss_fn)?;
                let gbdt_model = GBDTModel::train_binned(dataset, gbdt_config)?;
                (Some(gbdt_model), None, None, None)
            };

        Ok(Self {
            config,
            gbdt_model,
            gbdt_ensemble,
            stacker_weights,
            stacker_intercept,
            linear_booster: None,
            trees: Vec::new(),
            base_prediction: 0.0, // Not used - GBDTModel handles this
            num_features,
            analysis,
            raw_features_for_linear: None,
            linear_feature_indices: None,
            num_linear_features: None,
            feature_extractor,
        })
    }

    // =========================================================================
    // LinearThenTree Mode - Linear phase + GBDTModel on residuals
    // =========================================================================
    // Uses Newton-step Coordinate Descent for the linear phase (gradient/hessian).
    // This provides implicit regularization via learning_rate and captures global
    // trends that trees cannot extrapolate.
    //
    // NOTE ON INTENTIONAL CODE DUPLICATION:
    // The linear training loop in this method (Phase 1, lines 580-606) is duplicated
    // in `AutoBuilder::train_ltt_ensemble()` (src/model/builder.rs). This duplication
    // is intentional and acceptable because:
    // - This method is part of UniversalModel (core training API)
    // - AutoBuilder::train_ltt_ensemble is high-level AutoML orchestration
    // - Extracting shared logic would require complex trait/closure patterns
    // - The duplication is ~25 lines and stable (rarely changes)
    // - Clear architectural separation outweighs DRY principle
    //
    // See: AutoBuilder::train_ltt_ensemble() in src/model/builder.rs for the other copy.

    fn train_linear_then_tree(
        dataset: &BinnedDataset,
        raw_features_opt: Option<&[f32]>,
        linear_feature_indices_opt: Option<&[usize]>,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        analysis: Option<DatasetAnalysis>,
        feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
    ) -> Result<Self> {
        let targets = dataset.targets();
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        // Determine which features to use for linear model
        let linear_indices: Option<Vec<usize>> = linear_feature_indices_opt.map(|v| v.to_vec());

        // Get raw features: use provided ones or extract from bins (lossy fallback)
        let raw_features: Vec<f32> = if let Some(provided) = raw_features_opt {
            provided.to_vec()
        } else {
            // Memory safety check for bin-center extraction fallback
            let estimated_bytes = config.estimate_linear_memory(num_rows, num_features);
            let estimated_mb = estimated_bytes / (1024 * 1024);

            if config.max_linear_memory_mb > 0 && estimated_mb > config.max_linear_memory_mb {
                return Err(crate::TreeBoostError::Config(format!(
                    "LinearThenTree mode would require ~{}MB for raw feature extraction \
                     ({}rows × {}features × 4bytes), exceeding limit of {}MB. \
                     Options: (1) Increase max_linear_memory_mb, (2) Use PureTree mode, \
                     (3) Reduce dataset size, (4) Use fewer features.",
                    estimated_mb, num_rows, num_features, config.max_linear_memory_mb
                )));
            }

            // Warn on very large allocations (>1GB) even without explicit limit
            if estimated_mb > 1024 {
                eprintln!(
                    "Warning: LinearThenTree will allocate ~{}MB for raw features. \
                     Consider setting max_linear_memory_mb to prevent OOM.",
                    estimated_mb
                );
            }

            // Fallback: extract from bins (lossy - linear model will be less accurate)
            Self::extract_raw_features(dataset)
        };

        // Calculate actual number of features in raw_features
        // (may differ from num_features if FeatureExtractor was used)
        let num_raw_features = if num_rows > 0 {
            raw_features.len() / num_rows
        } else {
            num_features
        };

        // Determine number of features for linear model
        let num_linear_features = linear_indices
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(num_raw_features);

        // Base prediction (mean target)
        let base_prediction = loss_fn.initial_prediction(targets);

        // =====================================================================
        // Phase 1: Fit Linear Model using Newton-step Coordinate Descent
        // =====================================================================
        // Iterative gradient-based fitting with learning_rate provides implicit
        // regularization. This is more robust than closed-form Ridge for
        // generalization.

        // Extract selected features for linear model if indices are specified
        let linear_features = extract_selected_features(
            &raw_features,
            num_rows,
            num_raw_features,
            linear_indices.as_deref(),
        );

        let mut linear_booster =
            LinearBooster::new(num_linear_features, config.linear_config.clone());

        // Current predictions start from base
        let mut predictions = vec![base_prediction; num_rows];

        // Iteratively fit linear model
        for _round in 0..config.linear_rounds {
            // Compute gradients and hessians
            let mut gradients = vec![0.0f32; num_rows];
            let mut hessians = vec![0.0f32; num_rows];

            for i in 0..num_rows {
                let (g, h) = loss_fn.gradient_hessian(targets[i], predictions[i]);
                gradients[i] = g;
                hessians[i] = h;
            }

            // Fit linear model on gradients (Newton step)
            linear_booster.fit_on_gradients(
                &linear_features,
                num_linear_features,
                &gradients,
                &hessians,
            )?;

            // Update predictions with shrinkage factor (ensemble weighting)
            let shrinkage = config.linear_config.shrinkage_factor;
            for (i, pred) in predictions.iter_mut().enumerate().take(num_rows) {
                let linear_pred =
                    linear_booster.predict_row(&linear_features, num_linear_features, i);
                *pred += shrinkage * linear_pred;
            }
        }

        // =====================================================================
        // Phase 2: Train GBDTModel on Residuals
        // =====================================================================

        // Clone dataset and modify targets to residuals
        let mut residual_dataset = dataset.clone();
        {
            let residual_targets = residual_dataset.targets_mut();
            for i in 0..num_rows {
                residual_targets[i] = targets[i] - predictions[i];
            }
        }

        // Check if ensemble training is requested
        let (gbdt_model, gbdt_ensemble, stacker_weights, stacker_intercept) =
            if let Some(ref seeds) = config.ensemble_seeds {
                // Multi-seed ensemble training
                Self::train_gbdt_ensemble(&residual_dataset, &config, loss_fn, seeds)?
            } else {
                // Single GBDT training (standard path)
                let gbdt_config = Self::to_gbdt_config(&config, loss_fn)?;
                let gbdt_model = GBDTModel::train_binned(&residual_dataset, gbdt_config)?;
                (Some(gbdt_model), None, None, None)
            };

        Ok(Self {
            config,
            gbdt_model,
            gbdt_ensemble,
            stacker_weights,
            stacker_intercept,
            linear_booster: Some(linear_booster),
            trees: Vec::new(), // Not used - GBDTModel stores trees
            base_prediction,
            num_features,
            analysis,
            raw_features_for_linear: Some(raw_features),
            linear_feature_indices: linear_indices.map(|v| v.to_vec()),
            num_linear_features: Some(num_linear_features),
            feature_extractor,
        })
    }

    // =========================================================================
    // RandomForest Mode
    // =========================================================================

    fn train_random_forest(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        analysis: Option<DatasetAnalysis>,
        feature_extractor: Option<crate::dataset::feature_extractor::FeatureExtractor>,
    ) -> Result<Self> {
        let targets = dataset.targets();
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        // Initial prediction (mean for RF)
        let base_prediction = loss_fn.initial_prediction(targets);

        // RF uses learning_rate = 1.0 (each tree contributes fully)
        let tree_config = config.tree_config.clone().with_learning_rate(1.0);

        // Train trees in parallel with bootstrap samples
        let trees: Vec<Tree> = (0..config.num_rounds)
            .into_par_iter()
            .filter_map(|seed_offset| {
                // Bootstrap sample
                let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed + seed_offset as u64);
                let bootstrap_indices: Vec<usize> = (0..num_rows)
                    .map(|_| {
                        use rand::Rng;
                        rng.gen_range(0..num_rows)
                    })
                    .collect();

                // Compute gradients for this bootstrap sample
                // For RF, we fit to residuals from base prediction
                let mut gradients = vec![0.0f32; num_rows];
                let mut hessians = vec![0.0f32; num_rows];

                for &idx in &bootstrap_indices {
                    let (g, h) = loss_fn.gradient_hessian(targets[idx], base_prediction);
                    gradients[idx] = g;
                    hessians[idx] = h;
                }

                // Grow tree on bootstrap sample
                let mut booster = TreeBooster::new(tree_config.clone());
                if booster
                    .fit_on_gradients(dataset, &gradients, &hessians, Some(&bootstrap_indices))
                    .is_ok()
                {
                    booster.take_tree()
                } else {
                    None
                }
            })
            .collect();

        Ok(Self {
            config,
            gbdt_model: None, // RandomForest uses self.trees, not GBDTModel
            gbdt_ensemble: None,
            stacker_weights: None,
            stacker_intercept: None,
            linear_booster: None,
            trees,
            base_prediction,
            num_features,
            analysis,
            raw_features_for_linear: None,
            linear_feature_indices: None,
            num_linear_features: None,
            feature_extractor,
        })
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Predict using GBDT ensemble with stacker weights
    fn predict_ensemble(&self, dataset: &BinnedDataset, ensemble: &[GBDTModel]) -> Vec<f32> {
        let num_rows = dataset.num_rows();

        // Get predictions from all ensemble members
        let member_predictions: Vec<Vec<f32>> = ensemble
            .iter()
            .map(|model| model.predict(dataset))
            .collect();

        // Combine using stacker weights
        if let Some(ref weights) = self.stacker_weights {
            let mut combined = vec![0.0; num_rows];

            // Weighted sum of member predictions
            for (pred, &weight) in member_predictions.iter().zip(weights.iter()) {
                for i in 0..num_rows {
                    combined[i] += pred[i] * weight;
                }
            }

            // Add intercept if present (Ridge stacking)
            if let Some(intercept) = self.stacker_intercept {
                for i in 0..num_rows {
                    combined[i] += intercept;
                }
            }

            combined
        } else {
            // Fallback to equal-weight averaging if weights not available
            let mut combined = vec![0.0; num_rows];
            let scale = 1.0 / ensemble.len() as f32;

            for pred in &member_predictions {
                for i in 0..num_rows {
                    combined[i] += pred[i] * scale;
                }
            }

            combined
        }
    }

    /// Train multi-seed GBDT ensemble and fit stacker
    ///
    /// Returns (None, Some(ensemble), Some(weights), Some(intercept))
    fn train_gbdt_ensemble(
        dataset: &BinnedDataset,
        config: &UniversalConfig,
        loss_fn: &dyn LossFunction,
        seeds: &[u64],
    ) -> Result<(
        Option<GBDTModel>,
        Option<Vec<GBDTModel>>,
        Option<Vec<f32>>,
        Option<f32>,
    )> {
        use crate::ensemble::{RidgeStacker, Stacker, StackingConfig};

        // Train multiple GBDTs with different seeds
        let mut models = Vec::with_capacity(seeds.len());
        let mut oof_predictions = Vec::with_capacity(seeds.len());

        for &seed in seeds {
            let mut gbdt_config = Self::to_gbdt_config(config, loss_fn)?;
            gbdt_config.seed = seed;

            let model = GBDTModel::train_binned(dataset, gbdt_config)?;

            // Get OOF predictions for stacking
            let preds = model.predict(dataset);
            oof_predictions.push(preds);

            models.push(model);
        }

        // Fit stacker on OOF predictions
        let (weights, intercept) = match &config.stacking_strategy {
            super::config::StackingStrategy::Ridge {
                alpha,
                rank_transform,
                fit_intercept,
                min_weight,
            } => {
                let stacking_config = StackingConfig {
                    alpha: *alpha,
                    rank_transform: *rank_transform,
                    fit_intercept: *fit_intercept,
                    min_weight: *min_weight,
                };

                let mut stacker = RidgeStacker::new(stacking_config);
                stacker.fit(&oof_predictions, dataset.targets());

                let weights = stacker.weights().map(|w| w.to_vec());
                let intercept = if *fit_intercept {
                    Some(stacker.intercept())
                } else {
                    None
                };

                (weights, intercept)
            }
            super::config::StackingStrategy::Average => {
                // Equal weights for all models
                let weights = vec![1.0 / seeds.len() as f32; seeds.len()];
                (Some(weights), None)
            }
        };

        Ok((None, Some(models), weights, intercept))
    }

    /// Extract raw feature values from BinnedDataset using bin-center approximation
    ///
    /// **⚠️ Note**: This is a fallback method with accuracy limitations. For best results,
    /// use `FeatureExtractor` with the original DataFrame instead. See the
    /// `feature_extractor` module documentation for a detailed comparison.
    ///
    /// This method approximates raw values by using the midpoint of bin boundaries.
    /// While functional, this loses precision compared to extracting from the original
    /// DataFrame with `FeatureExtractor`.
    ///
    /// Returns row-major f32 array for linear model training.
    ///
    /// # When to Use
    /// - Only when you have a BinnedDataset but not the original DataFrame
    /// - When using UniversalModel directly (advanced usage)
    /// - When slight accuracy loss is acceptable
    ///
    /// # Alternative (Recommended)
    /// ```rust,ignore
    /// use treeboost::dataset::feature_extractor::FeatureExtractor;
    ///
    /// let extractor = FeatureExtractor::new();
    /// let (raw_features, num_features) = extractor.extract(&df, "target")?;
    /// // Use raw_features with train_with_raw_features()
    /// ```
    /// Extract raw features from bins (lossy approximation)
    ///
    /// **Deprecated approach**: This method is kept for compatibility but delegates
    /// to `BinnedDataset::extract_raw_features_from_bins()` for consistency across
    /// the codebase. For best accuracy in LinearThenTree mode, pass actual raw features
    /// to training instead of relying on bin-center approximation.
    ///
    /// # Historical Note
    ///
    /// Previous implementation used bin-center approximation (midpoint between boundaries)
    /// for improved accuracy. Current implementation uses bin split values for consistency.
    /// The difference is negligible in practice, and users needing high accuracy should
    /// pass raw features directly rather than relying on either approximation.
    pub fn extract_raw_features(dataset: &BinnedDataset) -> Vec<f32> {
        dataset.extract_raw_features_from_bins()
    }

    /// Extract linear features for LinearThenTree mode
    ///
    /// Combines raw feature extraction with optional feature selection.
    /// Returns (linear_features, num_linear_features).
    fn get_linear_features(&self, dataset: &BinnedDataset) -> (Vec<f32>, usize) {
        let num_rows = dataset.num_rows();

        // Get raw features (from stored or extract from bins)
        let raw_features = self
            .raw_features_for_linear
            .clone()
            .unwrap_or_else(|| Self::extract_raw_features(dataset));

        let num_raw_features = if num_rows > 0 {
            raw_features.len() / num_rows
        } else {
            self.num_features
        };

        // Extract selected features if feature selection was used
        let linear_features = crate::utils::features::extract_selected_features(
            &raw_features,
            num_rows,
            num_raw_features,
            self.linear_feature_indices.as_deref(),
        );

        let num_linear_features = self.num_linear_features.unwrap_or(num_raw_features);

        (linear_features, num_linear_features)
    }

    /// Extract raw feature values for a single row
    ///
    /// More efficient than extract_raw_features() when only predicting one row.
    fn extract_raw_features_row(dataset: &BinnedDataset, row_idx: usize) -> Vec<f32> {
        let feature_info = dataset.all_feature_info();
        let mut raw_features = Vec::with_capacity(feature_info.len());

        for (f, info) in feature_info.iter().enumerate() {
            let boundaries = &info.bin_boundaries;
            let bin = dataset.get_bin(row_idx, f) as usize;

            // Convert bin back to approximate raw value using bin center
            let raw_value = if boundaries.is_empty() {
                bin as f32
            } else if bin == 0 {
                boundaries.first().copied().unwrap_or(0.0) as f32
            } else if bin >= boundaries.len() {
                boundaries.last().copied().unwrap_or(0.0) as f32
            } else {
                // Midpoint between bin boundaries
                ((boundaries[bin - 1] + boundaries[bin.min(boundaries.len() - 1)]) / 2.0) as f32
            };

            raw_features.push(raw_value);
        }

        raw_features
    }

    // =========================================================================
    // Prediction
    // =========================================================================

    /// Predict for all rows in dataset
    pub fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        match self.config.mode {
            BoostingMode::PureTree => {
                // Check for ensemble first, then single model
                if let Some(ref ensemble) = self.gbdt_ensemble {
                    self.predict_ensemble(dataset, ensemble)
                } else {
                    // Delegate to single GBDTModel
                    self.gbdt_model
                        .as_ref()
                        .map(|m| m.predict(dataset))
                        .unwrap_or_else(|| vec![0.0; dataset.num_rows()])
                }
            }
            BoostingMode::LinearThenTree => {
                // Linear contribution + GBDTModel (trained on residuals)
                let num_rows = dataset.num_rows();
                let mut predictions = vec![self.base_prediction; num_rows];

                // Add linear contribution with shrinkage
                if let Some(ref linear) = self.linear_booster {
                    let raw_features = Self::extract_raw_features(dataset);
                    let linear_preds = linear.predict_batch(&raw_features, self.num_features);
                    self.apply_linear_shrinkage(&mut predictions, &linear_preds);
                }

                // Add tree contribution (either ensemble or single GBDT, trained on residuals)
                // IMPORTANT: Subtract gbdt's base_prediction to avoid double-counting
                // gbdt.predict() returns (gbdt_base + tree_sum), but we already have ltt_base
                if let Some(ref ensemble) = self.gbdt_ensemble {
                    let tree_preds = self.predict_ensemble(dataset, ensemble);
                    // Ensemble already accounts for base predictions internally
                    for i in 0..num_rows {
                        predictions[i] += tree_preds[i];
                    }
                } else if let Some(ref gbdt) = self.gbdt_model {
                    let tree_preds = gbdt.predict(dataset);
                    let gbdt_base = gbdt.base_prediction();
                    for i in 0..num_rows {
                        predictions[i] += tree_preds[i] - gbdt_base;
                    }
                }

                predictions
            }
            BoostingMode::RandomForest => {
                // RandomForest: Each tree is independent, predictions averaged
                let num_rows = dataset.num_rows();
                let mut predictions = vec![self.base_prediction; num_rows];

                if !self.trees.is_empty() {
                    let mut tree_sum = vec![0.0f32; num_rows];
                    for tree in &self.trees {
                        tree.predict_batch_add(dataset, &mut tree_sum);
                    }
                    let scale = 1.0 / self.trees.len() as f32;
                    for i in 0..num_rows {
                        predictions[i] += tree_sum[i] * scale;
                    }
                }

                predictions
            }
        }
    }

    /// Predict for all rows using raw features (recommended for LinearThenTree)
    ///
    /// For LinearThenTree mode, using raw (unbinned) features for the linear model
    /// gives significantly better accuracy than the bin-center approximation.
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset (used for tree predictions)
    /// * `raw_features` - Original features, row-major f32 array (num_rows * num_features)
    ///
    /// # Note
    /// For PureTree and RandomForest, `raw_features` is ignored (trees use binned data).
    pub fn predict_with_raw_features(
        &self,
        dataset: &BinnedDataset,
        raw_features: &[f32],
    ) -> Vec<f32> {
        match self.config.mode {
            BoostingMode::PureTree => {
                // Check for ensemble first, then single model (raw features not used for trees)
                if let Some(ref ensemble) = self.gbdt_ensemble {
                    self.predict_ensemble(dataset, ensemble)
                } else {
                    self.gbdt_model
                        .as_ref()
                        .map(|m| m.predict(dataset))
                        .unwrap_or_else(|| vec![0.0; dataset.num_rows()])
                }
            }
            BoostingMode::LinearThenTree => {
                // Linear contribution uses raw features + GBDTModel uses binned
                let num_rows = dataset.num_rows();
                let mut predictions = vec![self.base_prediction; num_rows];

                // Add linear contribution using raw features (possibly selected subset)
                if let Some(ref linear) = self.linear_booster {
                    // Calculate actual number of features in raw_features array
                    // (may differ from self.num_features if FeatureExtractor was used)
                    let num_raw_features = if num_rows > 0 {
                        raw_features.len() / num_rows
                    } else {
                        self.num_features
                    };

                    let num_lin_feats = self.num_linear_features.unwrap_or(num_raw_features);

                    // Extract selected features if indices are specified
                    let linear_features = extract_selected_features(
                        raw_features,
                        num_rows,
                        num_raw_features,
                        self.linear_feature_indices.as_deref(),
                    );

                    let linear_preds = linear.predict_batch(&linear_features, num_lin_feats);

                    // Apply shrinkage factor (ensemble weighting)
                    self.apply_linear_shrinkage(&mut predictions, &linear_preds);
                }

                // Add tree contribution (either ensemble or single GBDT, trees use binned data)
                // IMPORTANT: Subtract gbdt's base_prediction to avoid double-counting
                if let Some(ref ensemble) = self.gbdt_ensemble {
                    let tree_preds = self.predict_ensemble(dataset, ensemble);
                    for i in 0..num_rows {
                        predictions[i] += tree_preds[i];
                    }
                } else if let Some(ref gbdt) = self.gbdt_model {
                    let tree_preds = gbdt.predict(dataset);
                    let gbdt_base = gbdt.base_prediction();
                    for i in 0..num_rows {
                        predictions[i] += tree_preds[i] - gbdt_base;
                    }
                }

                predictions
            }
            BoostingMode::RandomForest => {
                // RandomForest: trees don't use raw features
                self.predict(dataset)
            }
        }
    }

    /// Predict using only linear component (LinearThenTree mode only)
    ///
    /// Returns predictions from base + linear model, without tree contribution.
    /// Useful for comparing linear-only vs full LinearThenTree performance.
    pub fn predict_linear_only(
        &self,
        dataset: &BinnedDataset,
        raw_features: &[f32],
    ) -> Result<Vec<f32>> {
        if !matches!(self.config.mode, BoostingMode::LinearThenTree) {
            return Err(crate::TreeBoostError::Config(
                "predict_linear_only() only available for LinearThenTree mode".to_string(),
            ));
        }

        let num_rows = dataset.num_rows();
        let mut predictions = vec![self.base_prediction; num_rows];

        // Add only linear contribution (no trees)
        if let Some(ref linear) = self.linear_booster {
            let num_raw_features = if num_rows > 0 {
                raw_features.len() / num_rows
            } else {
                self.num_features
            };

            let num_lin_feats = self.num_linear_features.unwrap_or(num_raw_features);

            // Extract selected features if indices are specified
            let linear_features = extract_selected_features(
                raw_features,
                num_rows,
                num_raw_features,
                self.linear_feature_indices.as_deref(),
            );

            let linear_preds = linear.predict_batch(&linear_features, num_lin_feats);

            // NOTE: For linear-only, we return the full linear model prediction
            // WITHOUT the boosting shrinkage (shrinkage_factor). This shows the
            // standalone linear model's performance for fair comparison.
            for i in 0..num_rows.min(linear_preds.len()) {
                predictions[i] += linear_preds[i];
            }
        }

        Ok(predictions)
    }

    /// Predict for a single row
    pub fn predict_row(&self, dataset: &BinnedDataset, row_idx: usize) -> f32 {
        match self.config.mode {
            BoostingMode::PureTree => {
                // Delegate to GBDTModel
                self.gbdt_model
                    .as_ref()
                    .map(|m| m.predict(dataset)[row_idx]) // GBDTModel doesn't have predict_row
                    .unwrap_or(0.0)
            }
            BoostingMode::LinearThenTree => {
                let mut pred = self.base_prediction;

                // Add linear contribution with shrinkage
                if let Some(ref linear) = self.linear_booster {
                    let raw_features = Self::extract_raw_features_row(dataset, row_idx);
                    let linear_pred = linear.predict_row(&raw_features, self.num_features, 0);
                    pred += self.apply_linear_shrinkage_scalar(linear_pred);
                }

                // Add tree contribution from GBDTModel
                // Subtract gbdt's base_prediction to avoid double-counting
                if let Some(ref gbdt) = self.gbdt_model {
                    pred += gbdt.predict(dataset)[row_idx] - gbdt.base_prediction();
                }

                pred
            }
            BoostingMode::RandomForest => {
                let mut pred = self.base_prediction;

                if !self.trees.is_empty() {
                    let tree_sum: f32 = self
                        .trees
                        .iter()
                        .map(|t| t.predict_row(dataset, row_idx))
                        .sum();
                    pred += tree_sum / self.trees.len() as f32;
                }

                pred
            }
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the boosting mode
    pub fn mode(&self) -> BoostingMode {
        self.config.mode
    }

    /// Get training configuration
    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    /// Save the model to a file using zero-copy rkyv serialization
    ///
    /// # Example
    /// ```ignore
    /// model.save("model.rkyv")?;
    /// let loaded = UniversalModel::load("model.rkyv")?;
    /// ```
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        crate::serialize::save_universal_model(self, path)
    }

    /// Load a model from a file using zero-copy rkyv deserialization
    ///
    /// # Example
    /// ```ignore
    /// let model = UniversalModel::load("model.rkyv")?;
    /// let predictions = model.predict(&dataset);
    /// ```
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        crate::serialize::load_universal_model(path)
    }

    /// Get number of trees
    pub fn num_trees(&self) -> usize {
        // Check GBDTModel first (PureTree and LinearThenTree modes)
        if let Some(ref gbdt) = self.gbdt_model {
            gbdt.num_trees()
        } else {
            // RandomForest uses self.trees
            self.trees.len()
        }
    }

    /// Get base prediction
    ///
    /// For PureTree, this delegates to GBDTModel.
    /// For LinearThenTree, returns the original base prediction (GBDTModel was trained on residuals).
    /// For RandomForest, returns the stored base prediction.
    pub fn base_prediction(&self) -> f32 {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| m.base_prediction())
                .unwrap_or(self.base_prediction),
            // LinearThenTree and RandomForest: use stored base_prediction
            // (GBDTModel in LinearThenTree was trained on residuals, so its base is ~0)
            BoostingMode::LinearThenTree | BoostingMode::RandomForest => self.base_prediction,
        }
    }

    /// Check if model has linear component
    pub fn has_linear(&self) -> bool {
        self.linear_booster.is_some()
    }

    /// Get linear booster reference (if present)
    pub fn linear_booster(&self) -> Option<&LinearBooster> {
        self.linear_booster.as_ref()
    }

    /// Get underlying GBDTModel (for PureTree and LinearThenTree modes)
    pub fn gbdt_model(&self) -> Option<&GBDTModel> {
        self.gbdt_model.as_ref()
    }

    /// Get trees (only for RandomForest mode; PureTree/LinearThenTree use GBDTModel)
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    // =========================================================================
    // Analysis and Mode Selection Info
    // =========================================================================

    /// Get the dataset analysis that led to mode selection (if auto mode was used)
    ///
    /// Returns `Some(analysis)` if the model was trained with `auto()` or `train_with_selection(Auto)`.
    /// Returns `None` if a fixed mode was specified.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = UniversalModel::auto(&dataset, &MseLoss)?;
    ///
    /// if let Some(analysis) = model.analysis() {
    ///     println!("Linear R²: {:.2}", analysis.linear_r2);
    ///     println!("Tree gain: {:.2}", analysis.tree_gain);
    ///     println!("Noise floor: {:.2}", analysis.noise_floor);
    /// }
    /// ```
    pub fn analysis(&self) -> Option<&DatasetAnalysis> {
        self.analysis.as_ref()
    }

    /// Get the confidence in the mode selection (if auto mode was used)
    ///
    /// Returns the confidence level from the analysis:
    /// - `High`: Very clear signal, strongly recommend this mode
    /// - `Medium`: Reasonable signal, this mode is likely best
    /// - `Low`: Weak signal, consider validating with cross-validation
    ///
    /// Returns `None` if a fixed mode was specified.
    pub fn selection_confidence(&self) -> Option<Confidence> {
        self.analysis.as_ref().map(|a| a.confidence())
    }

    /// Check if the mode was automatically selected
    pub fn was_auto_selected(&self) -> bool {
        self.analysis.is_some()
    }

    /// Get a formatted analysis report (if auto mode was used)
    ///
    /// Returns a human-readable report explaining:
    /// - Dataset characteristics (linear signal, tree gain, noise floor)
    /// - Mode scores for each option
    /// - Why the selected mode was chosen
    /// - Alternative modes to consider
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = UniversalModel::auto(&dataset, &MseLoss)?;
    ///
    /// if let Some(report) = model.analysis_report() {
    ///     println!("{}", report);
    /// }
    /// ```
    pub fn analysis_report(&self) -> Option<crate::analysis::AnalysisReport<'_>> {
        self.analysis.as_ref().map(|a| a.report())
    }

    /// Get a compact single-line summary of the analysis (if auto mode was used)
    ///
    /// Useful for logging or progress output.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = UniversalModel::auto(&dataset, &MseLoss)?;
    ///
    /// if let Some(summary) = model.analysis_summary() {
    ///     log::info!("{}", summary);
    /// }
    /// ```
    pub fn analysis_summary(&self) -> Option<String> {
        self.analysis.as_ref().map(crate::analysis::compact_summary)
    }

    // =========================================================================
    // Conformal Prediction (PureTree only)
    // =========================================================================

    /// Predict with conformal prediction intervals
    ///
    /// Returns (predictions, lower_bounds, upper_bounds) for uncertainty quantification.
    ///
    /// # Note
    /// Only supported in PureTree mode (delegates to GBDTModel).
    /// LinearThenTree and RandomForest modes return an error.
    pub fn predict_with_intervals(
        &self,
        dataset: &BinnedDataset,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_with_intervals(dataset)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            BoostingMode::LinearThenTree | BoostingMode::RandomForest => {
                Err(crate::TreeBoostError::Config(
                    "Conformal prediction only supported in PureTree mode".to_string(),
                ))
            }
        }
    }

    /// Get the calibrated conformal quantile (if available)
    pub fn conformal_quantile(&self) -> Option<f32> {
        self.gbdt_model
            .as_ref()
            .and_then(|m| m.conformal_quantile())
    }

    // =========================================================================
    // Classification (PureTree only)
    // =========================================================================

    /// Binary classification: predict probabilities
    ///
    /// Returns probabilities in [0, 1] for binary classification.
    /// Requires model trained with binary log loss.
    pub fn predict_proba(&self, dataset: &BinnedDataset) -> Result<Vec<f32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_proba(dataset)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Classification only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Binary classification: predict classes
    ///
    /// Returns 0 or 1 based on threshold (default: 0.5).
    pub fn predict_class(&self, dataset: &BinnedDataset, threshold: f32) -> Result<Vec<u32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_class(dataset, threshold)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Classification only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Check if model is multi-class
    pub fn is_multiclass(&self) -> bool {
        self.gbdt_model.as_ref().is_some_and(|m| m.is_multiclass())
    }

    /// Get number of classes (1 for regression, 2+ for classification)
    pub fn get_num_classes(&self) -> usize {
        self.gbdt_model
            .as_ref()
            .map(|m| m.get_num_classes())
            .unwrap_or(1)
    }

    /// Multi-class classification: predict probabilities for each class
    ///
    /// Returns Vec<Vec<f32>> where each inner vec contains probabilities for all classes.
    pub fn predict_proba_multiclass(&self, dataset: &BinnedDataset) -> Result<Vec<Vec<f32>>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_proba_multiclass(dataset)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Multi-class only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class classification: predict class labels
    pub fn predict_class_multiclass(&self, dataset: &BinnedDataset) -> Result<Vec<u32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_class_multiclass(dataset)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Multi-class only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class classification: predict raw logits
    pub fn predict_raw_multiclass(&self, dataset: &BinnedDataset) -> Result<Vec<Vec<f32>>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_raw_multiclass(dataset)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Multi-class only supported in PureTree mode".to_string(),
            )),
        }
    }

    // =========================================================================
    // Feature Importance
    // =========================================================================

    /// Get feature importance scores
    ///
    /// Returns importance scores for each feature based on gain/split frequency.
    /// For PureTree/LinearThenTree, delegates to GBDTModel.
    /// For RandomForest, computes importance from split frequencies across all trees.
    pub fn feature_importance(&self) -> Vec<f32> {
        use crate::tree::NodeType;

        if let Some(ref gbdt) = self.gbdt_model {
            gbdt.feature_importance()
        } else if !self.trees.is_empty() {
            // RandomForest: count split frequencies per feature across all trees
            let mut importances = vec![0.0f32; self.num_features];
            for tree in &self.trees {
                for node in tree.nodes() {
                    if let NodeType::Internal { feature_idx, .. } = node.node_type {
                        if feature_idx < importances.len() {
                            // Count splits per feature (weighted by samples in node)
                            importances[feature_idx] += node.num_samples as f32;
                        }
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
        } else {
            vec![0.0; self.num_features]
        }
    }

    // =========================================================================
    // Raw Prediction (from unbinned features)
    // =========================================================================

    /// Predict from raw (unbinned) feature values
    ///
    /// Useful when you have raw feature values and don't want to create a BinnedDataset.
    /// Only supported in PureTree mode.
    ///
    /// # Arguments
    /// * `features` - Raw feature values for one or more rows, flattened row-major
    pub fn predict_raw(&self, features: &[f64]) -> Result<Vec<f32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_raw(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Predict with intervals from raw (unbinned) feature values
    pub fn predict_raw_with_intervals(
        &self,
        features: &[f64],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_raw_with_intervals(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Binary classification probability from raw features
    pub fn predict_proba_raw(&self, features: &[f64]) -> Result<Vec<f32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_proba_raw(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Binary classification class from raw features
    pub fn predict_class_raw(&self, features: &[f64], threshold: f32) -> Result<Vec<u32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_class_raw(features, threshold)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class probabilities from raw features
    pub fn predict_proba_multiclass_raw(&self, features: &[f64]) -> Result<Vec<Vec<f32>>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_proba_multiclass_raw(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class class labels from raw features
    pub fn predict_class_multiclass_raw(&self, features: &[f64]) -> Result<Vec<u32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_class_multiclass_raw(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class raw logits from raw features
    pub fn predict_raw_multiclass_raw(&self, features: &[f64]) -> Result<Vec<Vec<f32>>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_raw_multiclass_raw(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    // =========================================================================
    // Incremental Learning Support
    // =========================================================================

    /// Update the model with new training data (incremental learning)
    ///
    /// This method continues training from the current model state:
    /// - **PureTree**: Appends new trees trained on residuals
    /// - **LinearThenTree**: Updates linear weights (warm_fit) + appends new trees
    /// - **RandomForest**: Appends new bootstrap trees
    ///
    /// # Arguments
    /// * `dataset` - New data to train on (must have same feature schema)
    /// * `loss_fn` - Loss function (should match original training)
    /// * `additional_rounds` - Number of new boosting rounds (trees) to add
    ///
    /// # Example
    /// ```ignore
    /// // Train on January data
    /// let model = UniversalModel::train(&jan_data, config, &MseLoss)?;
    ///
    /// // Update with February data (10 more trees)
    /// model.update(&feb_data, &MseLoss, 10)?;
    /// ```
    pub fn update(
        &mut self,
        dataset: &BinnedDataset,
        loss_fn: &dyn LossFunction,
        additional_rounds: usize,
    ) -> Result<IncrementalUpdateReport> {
        // Validate feature compatibility
        if dataset.num_features() != self.num_features {
            return Err(crate::TreeBoostError::Config(format!(
                "Feature count mismatch: model has {} features, dataset has {}",
                self.num_features,
                dataset.num_features()
            )));
        }

        let num_rows = dataset.num_rows();
        let trees_before = self.num_trees();

        match self.config.mode {
            BoostingMode::PureTree => {
                self.update_pure_tree(dataset, loss_fn, additional_rounds)?;
            }
            BoostingMode::LinearThenTree => {
                self.update_linear_then_tree(dataset, loss_fn, additional_rounds)?;
            }
            BoostingMode::RandomForest => {
                self.update_random_forest(dataset, loss_fn, additional_rounds)?;
            }
        }

        let trees_after = self.num_trees();

        Ok(IncrementalUpdateReport {
            rows_trained: num_rows,
            trees_before,
            trees_after,
            trees_added: trees_after - trees_before,
            mode: self.config.mode,
        })
    }

    /// Update PureTree model by appending trees trained on residuals
    fn update_pure_tree(
        &mut self,
        dataset: &BinnedDataset,
        loss_fn: &dyn LossFunction,
        additional_rounds: usize,
    ) -> Result<()> {
        let gbdt = self.gbdt_model.as_mut().ok_or_else(|| {
            crate::TreeBoostError::Config("No GBDTModel available for update".to_string())
        })?;

        // Compute current predictions
        let predictions = gbdt.predict(dataset);

        // Create residual targets (avoids cloning targets only to overwrite)
        let residual_targets: Vec<f32> = dataset
            .targets()
            .iter()
            .zip(predictions.iter())
            .map(|(target, pred)| target - pred)
            .collect();

        // Create residual dataset with new targets
        let residual_dataset = dataset.with_targets(residual_targets);

        // Train new trees on residuals using same config
        let mut update_config = self.config.clone();
        update_config.num_rounds = additional_rounds;
        let gbdt_config = Self::to_gbdt_config(&update_config, loss_fn)?;
        let new_model = GBDTModel::train_binned(&residual_dataset, gbdt_config)?;

        // Append new trees
        gbdt.append_trees(new_model.trees().to_vec());

        Ok(())
    }

    /// Update LinearThenTree model
    fn update_linear_then_tree(
        &mut self,
        dataset: &BinnedDataset,
        loss_fn: &dyn LossFunction,
        additional_rounds: usize,
    ) -> Result<()> {
        use crate::learner::incremental::IncrementalLearner;

        let num_rows = dataset.num_rows();
        let targets = dataset.targets();

        // Compute current predictions BEFORE borrowing linear_booster mutably
        let current_preds = self.predict(dataset);

        // Extract linear features once (avoids duplication)
        let (linear_features, num_linear_features) = self.get_linear_features(dataset);

        // Update linear booster with warm_fit if present
        if let Some(ref mut linear_booster) = self.linear_booster {
            // Compute gradients
            let (gradients, hessians): (Vec<f32>, Vec<f32>) = targets
                .iter()
                .zip(current_preds.iter())
                .map(|(t, p)| loss_fn.gradient_hessian(*t, *p))
                .unzip();

            // Warm fit linear booster
            linear_booster.warm_fit(
                &linear_features,
                num_linear_features,
                &gradients,
                &hessians,
            )?;
        }

        // Now update trees on residuals from full ensemble
        if let Some(ref mut gbdt) = self.gbdt_model {
            // Get linear predictions (using already-extracted features)
            let linear_preds = if let Some(ref linear_booster) = self.linear_booster {
                linear_booster.predict_batch(&linear_features, num_linear_features)
            } else {
                vec![0.0; num_rows]
            };

            // Get existing tree predictions
            let tree_preds = gbdt.predict(dataset);

            // Compute residual targets: target - base - shrinkage*linear - tree
            let shrinkage = self.config.linear_config.shrinkage_factor;
            let residual_targets: Vec<f32> = targets
                .iter()
                .zip(linear_preds.iter())
                .zip(tree_preds.iter())
                .map(|((t, lp), tp)| t - self.base_prediction - shrinkage * lp - tp)
                .collect();

            // Create residual dataset with new targets (avoids full clone)
            let residual_dataset = dataset.with_targets(residual_targets);

            // Train new trees on residuals
            let mut update_config = self.config.clone();
            update_config.num_rounds = additional_rounds;
            let gbdt_config = Self::to_gbdt_config(&update_config, loss_fn)?;
            let new_model = GBDTModel::train_binned(&residual_dataset, gbdt_config)?;

            gbdt.append_trees(new_model.trees().to_vec());
        }

        Ok(())
    }

    /// Update RandomForest by adding more bootstrap trees
    fn update_random_forest(
        &mut self,
        dataset: &BinnedDataset,
        loss_fn: &dyn LossFunction,
        additional_rounds: usize,
    ) -> Result<()> {
        use crate::learner::TreeBooster;

        let num_rows = dataset.num_rows();
        let targets = dataset.targets();

        // RF trees are trained on bootstrap samples from base prediction
        let tree_config = self.config.tree_config.clone().with_learning_rate(1.0);

        // Use next seed offset based on current tree count
        let seed_offset_start = self.trees.len() as u64;

        let new_trees: Vec<Tree> = (0..additional_rounds)
            .into_par_iter()
            .filter_map(|i| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(
                    self.config.seed + seed_offset_start + i as u64,
                );

                // Bootstrap sample
                let bootstrap_indices: Vec<usize> = (0..num_rows)
                    .map(|_| {
                        use rand::Rng;
                        rng.gen_range(0..num_rows)
                    })
                    .collect();

                // Compute gradients for bootstrap sample
                let mut gradients = vec![0.0f32; num_rows];
                let mut hessians = vec![0.0f32; num_rows];

                for &idx in &bootstrap_indices {
                    let (g, h) = loss_fn.gradient_hessian(targets[idx], self.base_prediction);
                    gradients[idx] = g;
                    hessians[idx] = h;
                }

                // Grow tree
                let mut booster = TreeBooster::new(tree_config.clone());
                if booster
                    .fit_on_gradients(dataset, &gradients, &hessians, Some(&bootstrap_indices))
                    .is_ok()
                {
                    booster.take_tree()
                } else {
                    None
                }
            })
            .collect();

        self.trees.extend(new_trees);

        Ok(())
    }

    /// Save model to TRB (TreeBoost) incremental format
    ///
    /// TRB format supports incremental updates without rewriting the entire file.
    /// Use this format when you plan to update the model with new data.
    ///
    /// # Example
    /// ```ignore
    /// model.save_trb("model.trb", "Initial training on January data")?;
    ///
    /// // Later, after updating:
    /// model.save_trb_update("model.trb", 1000, "February update")?;
    /// ```
    pub fn save_trb(&self, path: impl AsRef<std::path::Path>, description: &str) -> Result<()> {
        use crate::serialize::{TrbHeader, TrbWriter, FORMAT_VERSION};
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let header = TrbHeader {
            format_version: FORMAT_VERSION,
            model_type: "universal".to_string(),
            created_at: timestamp,
            boosting_mode: format!("{:?}", self.config.mode),
            num_features: self.num_features,
            base_blob_size: 0, // Will be filled by TrbWriter
            metadata: description.to_string(),
        };

        // Serialize the model to rkyv bytes
        let blob = crate::serialize::rkyv_io::serialize_universal_model(self)?;

        TrbWriter::new(path, header, &blob)?;

        Ok(())
    }

    /// Append an update to an existing TRB file
    ///
    /// This appends a new segment without rewriting the base model.
    /// The model must be loaded with `load_trb()` before calling this.
    ///
    /// # Arguments
    /// * `path` - Path to existing .trb file
    /// * `rows_trained` - Number of rows used in this update
    /// * `description` - Description of this update
    pub fn save_trb_update(
        &self,
        path: impl AsRef<std::path::Path>,
        rows_trained: usize,
        description: &str,
    ) -> Result<()> {
        use crate::serialize::{open_for_append, TrbUpdateHeader, UpdateType};
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Use Snapshot type for all modes - stores full model state
        // This is simpler and more robust than incremental tree-only updates
        let update_type = UpdateType::Snapshot;

        let header = TrbUpdateHeader {
            update_type,
            created_at: timestamp,
            rows_trained,
            description: description.to_string(),
        };

        // Serialize the updated model
        let blob = crate::serialize::rkyv_io::serialize_universal_model(self)?;

        // Open and append
        let mut writer = open_for_append(path)?;
        writer.append_update(&header, &blob)?;

        Ok(())
    }

    /// Load model from TRB format
    ///
    /// Loads the base model and applies all update segments.
    ///
    /// # Example
    /// ```ignore
    /// let model = UniversalModel::load_trb("model.trb")?;
    /// ```
    pub fn load_trb(path: impl AsRef<std::path::Path>) -> Result<Self> {
        use crate::serialize::{TrbReader, TrbSegment, UpdateType};

        let mut reader = TrbReader::open(path)?;

        // Load all segments
        let segments = reader.load_all_segments()?;

        // The last snapshot-type segment (or base if no snapshots) is the current state
        let mut model: Option<Self> = None;

        for segment in segments {
            match segment {
                TrbSegment::Base { blob, .. } => {
                    model = Some(crate::serialize::rkyv_io::deserialize_universal_model(
                        &blob,
                    )?);
                }
                TrbSegment::Update { header, blob } => {
                    match header.update_type {
                        UpdateType::Snapshot => {
                            // Replace with snapshot
                            model = Some(crate::serialize::rkyv_io::deserialize_universal_model(
                                &blob,
                            )?);
                        }
                        UpdateType::Trees => {
                            // For trees-only updates, we'd need to deserialize and merge
                            // For now, we just use snapshots for simplicity
                            if model.is_none() {
                                model = Some(
                                    crate::serialize::rkyv_io::deserialize_universal_model(&blob)?,
                                );
                            }
                        }
                        UpdateType::Linear | UpdateType::Preprocessor => {
                            // These would need specialized handling
                            // For now, fall through
                        }
                    }
                }
            }
        }

        model.ok_or_else(|| {
            crate::TreeBoostError::Serialization("No valid model found in TRB file".to_string())
        })
    }

    /// Check if model is compatible with dataset for incremental update
    pub fn is_compatible_for_update(&self, dataset: &BinnedDataset) -> bool {
        self.num_features == dataset.num_features()
    }

    /// Get mutable reference to underlying GBDTModel (for advanced manipulation)
    pub fn gbdt_model_mut(&mut self) -> Option<&mut GBDTModel> {
        self.gbdt_model.as_mut()
    }

    /// Get mutable reference to linear booster (for advanced manipulation)
    pub fn linear_booster_mut(&mut self) -> Option<&mut LinearBooster> {
        self.linear_booster.as_mut()
    }

    /// Get mutable reference to trees (for RandomForest mode)
    pub fn trees_mut(&mut self) -> &mut Vec<Tree> {
        &mut self.trees
    }
}

/// Report from an incremental training update
#[derive(Debug, Clone)]
pub struct IncrementalUpdateReport {
    /// Number of rows in the new training data
    pub rows_trained: usize,
    /// Number of trees before update
    pub trees_before: usize,
    /// Number of trees after update
    pub trees_after: usize,
    /// Number of trees added
    pub trees_added: usize,
    /// Boosting mode
    pub mode: BoostingMode,
}

impl std::fmt::Display for IncrementalUpdateReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Incremental Update: {} rows, {} trees added ({} -> {}), mode={:?}",
            self.rows_trained, self.trees_added, self.trees_before, self.trees_after, self.mode
        )
    }
}

// =============================================================================
// TunableModel Implementation
// =============================================================================

use crate::tuner::{ParamValue, TunableModel};
use std::collections::HashMap;

impl TunableModel for UniversalModel {
    type Config = UniversalConfig;

    fn train(dataset: &BinnedDataset, config: &Self::Config) -> crate::Result<Self> {
        // Create a default MSE loss for tuning (loss type could be parameterized later)
        let loss_fn = crate::loss::MseLoss::new();
        Self::train(dataset, config.clone(), &loss_fn)
    }

    fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        UniversalModel::predict(self, dataset)
    }

    fn num_trees(&self) -> usize {
        // Delegate to UniversalModel::num_trees() which handles all modes correctly
        UniversalModel::num_trees(self)
    }

    fn apply_params(config: &mut Self::Config, params: &HashMap<String, ParamValue>) {
        for (name, value) in params {
            match (name.as_str(), value) {
                // Categorical: boosting mode
                ("mode", ParamValue::Categorical(v)) => {
                    config.mode = match v.as_str() {
                        "PureTree" => BoostingMode::PureTree,
                        "LinearThenTree" => BoostingMode::LinearThenTree,
                        "RandomForest" => BoostingMode::RandomForest,
                        _ => BoostingMode::PureTree, // Default fallback
                    };
                }
                // Numeric parameters
                ("num_rounds", ParamValue::Numeric(v)) => config.num_rounds = *v as usize,
                ("learning_rate", ParamValue::Numeric(v)) => config.learning_rate = *v,
                ("subsample", ParamValue::Numeric(v)) => config.subsample = *v,
                ("validation_ratio", ParamValue::Numeric(v)) => config.validation_ratio = *v,
                ("early_stopping_rounds", ParamValue::Numeric(v)) => {
                    config.early_stopping_rounds = *v as usize
                }
                ("linear_rounds", ParamValue::Numeric(v)) => config.linear_rounds = *v as usize,
                // Tree config parameters (prefixed with tree_)
                ("tree_max_depth", ParamValue::Numeric(v)) => {
                    config.tree_config = config.tree_config.clone().with_max_depth(*v as usize)
                }
                ("tree_max_leaves", ParamValue::Numeric(v)) => {
                    config.tree_config = config.tree_config.clone().with_max_leaves(*v as usize)
                }
                ("tree_lambda", ParamValue::Numeric(v)) => {
                    config.tree_config = config.tree_config.clone().with_lambda(*v)
                }
                // Linear config parameters (prefixed with linear_)
                ("linear_lambda", ParamValue::Numeric(v)) => {
                    config.linear_config = config.linear_config.clone().with_lambda(*v)
                }
                ("linear_max_iter", ParamValue::Numeric(v)) => {
                    config.linear_config = config.linear_config.clone().with_max_iter(*v as usize)
                }
                _ => {} // Unknown params are ignored
            }
        }
    }

    fn valid_params() -> &'static [&'static str] {
        &[
            // Categorical
            "mode",
            // Numeric
            "num_rounds",
            "learning_rate",
            "subsample",
            "validation_ratio",
            "early_stopping_rounds",
            "linear_rounds",
            // Tree config
            "tree_max_depth",
            "tree_max_leaves",
            "tree_lambda",
            // Linear config
            "linear_lambda",
            "linear_max_iter",
        ]
    }

    fn default_config() -> Self::Config {
        UniversalConfig::default()
    }

    fn get_learning_rate(config: &Self::Config) -> f32 {
        config.learning_rate
    }

    fn configure_validation(
        config: &mut Self::Config,
        validation_ratio: f32,
        early_stopping_rounds: usize,
    ) {
        config.validation_ratio = validation_ratio;
        config.early_stopping_rounds = early_stopping_rounds;
    }

    fn set_num_rounds(config: &mut Self::Config, num_rounds: usize) {
        config.num_rounds = num_rounds;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeatureInfo, FeatureType};
    use crate::loss::MseLoss;
    use rkyv::rancor::Error as RkyvError;

    fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * 3 + f * 7) % 256) as u8);
            }
        }

        // Linear relationship with some noise
        let targets: Vec<f32> = (0..num_rows)
            .map(|i| (i as f32) * 0.1 + (i % 10) as f32 * 0.01)
            .collect();

        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("f{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: (0..255).map(|b| b as f64).collect(),
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    // ========================================
    // Test Helper Functions
    // ========================================

    /// Test helper: Verify serde serialization roundtrip for BoostingMode
    fn assert_serde_roundtrip_mode(mode: &BoostingMode) {
        let json = serde_json::to_string(mode).expect("Failed to serialize");
        assert!(!json.is_empty(), "Serialized JSON should not be empty");

        let loaded: BoostingMode = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(loaded, *mode, "Deserialized value should match original");
    }

    /// Test helper: Verify model predictions match after serialization
    fn assert_model_predictions_match(
        original: &UniversalModel,
        loaded: &UniversalModel,
        dataset: &BinnedDataset,
        tolerance: f32,
    ) {
        let original_preds = original.predict(dataset);
        let loaded_preds = loaded.predict(dataset);

        assert_eq!(
            original_preds.len(),
            loaded_preds.len(),
            "Prediction count mismatch"
        );

        for (i, (&orig, &load)) in original_preds.iter().zip(loaded_preds.iter()).enumerate() {
            assert!(
                (orig - load).abs() < tolerance,
                "Prediction mismatch at index {}: {} vs {} (diff: {})",
                i,
                orig,
                load,
                (orig - load).abs()
            );
        }
    }

    /// Test helper: Train a model with specified mode
    fn train_test_model(mode: BoostingMode, num_rounds: usize) -> (UniversalModel, BinnedDataset) {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(mode)
            .with_num_rounds(num_rounds)
            .with_linear_rounds(1);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();
        (model, dataset)
    }

    #[test]
    fn test_universal_config_defaults() {
        let config = UniversalConfig::default();
        assert_eq!(config.mode, BoostingMode::PureTree);
        assert_eq!(config.num_rounds, 100);
        assert_eq!(config.learning_rate, 0.1);
    }

    #[test]
    fn test_universal_config_builder() {
        let config = UniversalConfig::new()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(50)
            .with_learning_rate(0.05)
            .with_linear_rounds(5);

        assert_eq!(config.mode, BoostingMode::LinearThenTree);
        assert_eq!(config.num_rounds, 50);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.linear_rounds, 5);
    }

    #[test]
    fn test_pure_tree_training() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();

        assert_eq!(model.mode(), BoostingMode::PureTree);
        assert_eq!(model.num_trees(), 5);
        assert!(!model.has_linear());
    }

    #[test]
    fn test_pure_tree_prediction() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();
        let predictions = model.predict(&dataset);

        assert_eq!(predictions.len(), 100);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_linear_then_tree_training() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(5)
            .with_linear_rounds(3);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();

        assert_eq!(model.mode(), BoostingMode::LinearThenTree);
        assert_eq!(model.num_trees(), 5);
        assert!(model.has_linear());
    }

    #[test]
    fn test_linear_then_tree_prediction() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(5)
            .with_linear_rounds(3);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();
        let predictions = model.predict(&dataset);

        assert_eq!(predictions.len(), 100);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_random_forest_training() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::RandomForest)
            .with_num_rounds(10);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();

        assert_eq!(model.mode(), BoostingMode::RandomForest);
        assert!(model.num_trees() <= 10); // Some trees might fail to grow
        assert!(!model.has_linear());
    }

    #[test]
    fn test_random_forest_prediction() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::RandomForest)
            .with_num_rounds(10);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();
        let predictions = model.predict(&dataset);

        assert_eq!(predictions.len(), 100);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_single_row_prediction_matches_batch() {
        let dataset = create_test_dataset(50, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();

        let batch_preds = model.predict(&dataset);
        for i in 0..50 {
            let single_pred = model.predict_row(&dataset, i);
            assert!((batch_preds[i] - single_pred).abs() < 1e-5);
        }
    }

    // =========================================================================
    // Auto Mode Selection Tests
    // =========================================================================

    #[test]
    fn test_auto_selects_mode_and_trains() {
        let dataset = create_test_dataset(200, 5);
        let loss = MseLoss;

        // Auto mode should analyze and train
        let model = UniversalModel::auto(&dataset, &loss).unwrap();

        // Model should have selected a mode
        assert!(matches!(
            model.mode(),
            BoostingMode::PureTree | BoostingMode::LinearThenTree | BoostingMode::RandomForest
        ));

        // Should have analysis attached
        assert!(model.was_auto_selected());
        assert!(model.analysis().is_some());
        assert!(model.selection_confidence().is_some());

        // Should predict successfully
        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), 200);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_auto_with_config() {
        let dataset = create_test_dataset(200, 5);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_num_rounds(10)
            .with_learning_rate(0.05);

        let model = UniversalModel::auto_with_config(&dataset, config, &loss).unwrap();

        // Should use custom config settings
        assert_eq!(model.config().num_rounds, 10);
        assert_eq!(model.config().learning_rate, 0.05);

        // Should still be auto-selected
        assert!(model.was_auto_selected());
    }

    #[test]
    fn test_analysis_report_generation() {
        let dataset = create_test_dataset(200, 5);
        let loss = MseLoss;

        let model = UniversalModel::auto(&dataset, &loss).unwrap();

        // Report should be available
        let report = model.analysis_report();
        assert!(report.is_some());

        // Report should be displayable (non-empty)
        let report_string = format!("{}", report.unwrap());
        assert!(!report_string.is_empty());
        assert!(report_string.contains("TreeBoost"));
    }

    #[test]
    fn test_analysis_summary() {
        let dataset = create_test_dataset(200, 5);
        let loss = MseLoss;

        let model = UniversalModel::auto(&dataset, &loss).unwrap();

        let summary = model.analysis_summary();
        assert!(summary.is_some());

        let summary_str = summary.unwrap();
        assert!(summary_str.contains("TreeBoost"));
        assert!(summary_str.contains("Recommended"));
    }

    #[test]
    fn test_train_with_selection_fixed() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new().with_num_rounds(5);

        // Fixed mode should use the specified mode
        let model = UniversalModel::train_with_selection(
            &dataset,
            config,
            ModeSelection::Fixed(BoostingMode::RandomForest),
            &loss,
        )
        .unwrap();

        assert_eq!(model.mode(), BoostingMode::RandomForest);
        // Fixed mode should NOT have analysis attached
        assert!(!model.was_auto_selected());
    }

    #[test]
    fn test_train_with_selection_auto() {
        let dataset = create_test_dataset(200, 5);
        let loss = MseLoss;

        let config = UniversalConfig::new().with_num_rounds(10);

        let model =
            UniversalModel::train_with_selection(&dataset, config, ModeSelection::Auto, &loss)
                .unwrap();

        // Auto selection should have analysis
        assert!(model.was_auto_selected());
        assert!(model.analysis().is_some());
    }

    #[test]
    fn test_mode_selection_default() {
        // Default should be Fixed(PureTree) for backwards compatibility
        let selection = ModeSelection::default();
        assert!(matches!(
            selection,
            ModeSelection::Fixed(BoostingMode::PureTree)
        ));
    }

    #[test]
    fn test_analysis_contains_metrics() {
        let dataset = create_test_dataset(200, 5);
        let loss = MseLoss;

        let model = UniversalModel::auto(&dataset, &loss).unwrap();
        let analysis = model.analysis().unwrap();

        // Analysis should have valid metrics
        assert!(analysis.linear_r2 >= 0.0 && analysis.linear_r2 <= 1.0);
        assert!(analysis.tree_gain >= 0.0);
        assert!(analysis.categorical_ratio >= 0.0 && analysis.categorical_ratio <= 1.0);
        assert!(analysis.noise_floor >= 0.0 && analysis.noise_floor <= 1.0);
        assert_eq!(analysis.num_rows, 200);
        assert_eq!(analysis.num_features, 5);
    }

    // ========================================
    // Serialization Tests (serde)
    // ========================================

    #[test]
    fn test_universal_config_serde_serialization() {
        let config = UniversalConfig::new()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(150)
            .with_learning_rate(0.05);

        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.is_empty());

        let loaded: UniversalConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.mode, config.mode);
        assert_eq!(loaded.num_rounds, config.num_rounds);
        assert!((loaded.learning_rate - config.learning_rate).abs() < 1e-6);
    }

    #[test]
    fn test_boosting_mode_serde_serialization() {
        assert_serde_roundtrip_mode(&BoostingMode::PureTree);
        assert_serde_roundtrip_mode(&BoostingMode::LinearThenTree);
        assert_serde_roundtrip_mode(&BoostingMode::RandomForest);
    }

    #[test]
    fn test_puretree_model_serde_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::PureTree, 10);

        let json = serde_json::to_string(&model).unwrap();
        assert!(!json.is_empty());

        let loaded: UniversalModel = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.mode(), BoostingMode::PureTree);
        assert_eq!(loaded.num_features(), 3);

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }

    #[test]
    fn test_linear_then_tree_model_serde_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::LinearThenTree, 10);

        let json = serde_json::to_string(&model).unwrap();
        assert!(!json.is_empty());

        let loaded: UniversalModel = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.mode(), BoostingMode::LinearThenTree);

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }

    #[test]
    fn test_random_forest_model_serde_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::RandomForest, 5);

        let json = serde_json::to_string(&model).unwrap();
        assert!(!json.is_empty());

        let loaded: UniversalModel = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.mode(), BoostingMode::RandomForest);

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }

    // ========================================
    // Serialization Tests (rkyv)
    // ========================================

    #[test]
    fn test_universal_config_rkyv_serialization() {
        let config = UniversalConfig::new()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(150)
            .with_learning_rate(0.05);

        let bytes = rkyv::to_bytes::<RkyvError>(&config).unwrap();
        assert!(!bytes.is_empty());

        let loaded: UniversalConfig = rkyv::from_bytes::<_, RkyvError>(&bytes).unwrap();
        assert_eq!(loaded.mode, config.mode);
        assert_eq!(loaded.num_rounds, config.num_rounds);
        assert!((loaded.learning_rate - config.learning_rate).abs() < 1e-6);
    }

    #[test]
    fn test_puretree_model_rkyv_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::PureTree, 10);

        let bytes = rkyv::to_bytes::<RkyvError>(&model).unwrap();
        assert!(!bytes.is_empty());

        let loaded: UniversalModel = rkyv::from_bytes::<_, RkyvError>(&bytes).unwrap();
        assert_eq!(loaded.mode(), BoostingMode::PureTree);
        assert_eq!(loaded.num_features(), 3);

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }

    #[test]
    fn test_linear_then_tree_model_rkyv_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::LinearThenTree, 10);

        let bytes = rkyv::to_bytes::<RkyvError>(&model).unwrap();
        assert!(!bytes.is_empty());

        let loaded: UniversalModel = rkyv::from_bytes::<_, RkyvError>(&bytes).unwrap();
        assert_eq!(loaded.mode(), BoostingMode::LinearThenTree);

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }

    #[test]
    fn test_random_forest_model_rkyv_serialization() {
        let (model, dataset) = train_test_model(BoostingMode::RandomForest, 5);

        let bytes = rkyv::to_bytes::<RkyvError>(&model).unwrap();
        assert!(!bytes.is_empty());

        let loaded: UniversalModel = rkyv::from_bytes::<_, RkyvError>(&bytes).unwrap();
        assert_eq!(loaded.mode(), BoostingMode::RandomForest);

        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }

    // ========================================
    // Incremental Learning Tests
    // ========================================

    #[test]
    fn test_puretree_incremental_update() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let mut model = UniversalModel::train(&dataset, config, &loss).unwrap();
        let trees_before = model.num_trees();
        assert_eq!(trees_before, 5);

        // Update with 5 more trees
        let report = model.update(&dataset, &loss, 5).unwrap();

        assert_eq!(report.trees_before, 5);
        assert_eq!(report.trees_after, 10);
        assert_eq!(report.trees_added, 5);
        assert_eq!(model.num_trees(), 10);
    }

    #[test]
    fn test_linear_then_tree_incremental_update() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(5)
            .with_linear_rounds(2);

        let mut model = UniversalModel::train(&dataset, config, &loss).unwrap();
        let trees_before = model.num_trees();
        assert_eq!(trees_before, 5);
        assert!(model.has_linear());

        // Update with 5 more trees
        let report = model.update(&dataset, &loss, 5).unwrap();

        assert_eq!(report.trees_before, 5);
        assert_eq!(report.trees_after, 10);
        assert_eq!(report.trees_added, 5);
    }

    #[test]
    fn test_random_forest_incremental_update() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::RandomForest)
            .with_num_rounds(5);

        let mut model = UniversalModel::train(&dataset, config, &loss).unwrap();
        let trees_before = model.num_trees();
        assert_eq!(trees_before, 5);

        // Update with 5 more trees
        let report = model.update(&dataset, &loss, 5).unwrap();

        assert_eq!(report.trees_before, 5);
        assert_eq!(report.trees_after, 10);
        assert_eq!(report.trees_added, 5);
    }

    #[test]
    fn test_incremental_update_report_display() {
        let report = IncrementalUpdateReport {
            rows_trained: 1000,
            trees_before: 10,
            trees_after: 20,
            trees_added: 10,
            mode: BoostingMode::PureTree,
        };

        let display = format!("{}", report);
        assert!(display.contains("1000 rows"));
        assert!(display.contains("10 trees added"));
    }

    #[test]
    fn test_incremental_update_feature_mismatch() {
        let dataset1 = create_test_dataset(100, 3);
        let dataset2 = create_test_dataset(100, 5); // Different feature count
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let mut model = UniversalModel::train(&dataset1, config, &loss).unwrap();

        // Update with mismatched features should fail
        let result = model.update(&dataset2, &loss, 5);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Feature count mismatch"));
    }

    #[test]
    fn test_is_compatible_for_update() {
        let dataset1 = create_test_dataset(100, 3);
        let dataset2 = create_test_dataset(50, 3); // Same features, different rows
        let dataset3 = create_test_dataset(100, 5); // Different features
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let model = UniversalModel::train(&dataset1, config, &loss).unwrap();

        assert!(model.is_compatible_for_update(&dataset1));
        assert!(model.is_compatible_for_update(&dataset2));
        assert!(!model.is_compatible_for_update(&dataset3));
    }

    #[test]
    fn test_trb_save_and_load() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();

        // Save to temp file
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.trb");

        model.save_trb(&path, "Test model").unwrap();

        // Load back
        let loaded = UniversalModel::load_trb(&path).unwrap();

        assert_eq!(loaded.mode(), BoostingMode::PureTree);
        assert_eq!(loaded.num_features(), 3);
        assert_eq!(loaded.num_trees(), 5);

        // Predictions should match
        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }

    #[test]
    fn test_trb_save_update_and_load() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let mut model = UniversalModel::train(&dataset, config, &loss).unwrap();

        // Save initial model
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.trb");
        model.save_trb(&path, "Initial training").unwrap();

        let initial_size = std::fs::metadata(&path).unwrap().len();

        // Update model
        model.update(&dataset, &loss, 5).unwrap();
        assert_eq!(model.num_trees(), 10);

        // Save update
        model.save_trb_update(&path, 100, "Update batch").unwrap();

        // File should be larger
        let updated_size = std::fs::metadata(&path).unwrap().len();
        assert!(
            updated_size > initial_size,
            "TRB file should grow after update"
        );

        // Load and verify
        let loaded = UniversalModel::load_trb(&path).unwrap();
        assert_eq!(loaded.num_trees(), 10);

        // Predictions should match
        assert_model_predictions_match(&model, &loaded, &dataset, 1e-4);
    }
}
