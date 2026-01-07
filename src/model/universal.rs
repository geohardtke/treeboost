//! UniversalModel: Unified boosting framework
//!
//! Supports multiple boosting modes:
//! - **PureTree**: Standard GBDT (delegates to GBDTModel - GPU, conformal, multi-class)
//! - **LinearThenTree**: Linear model captures trend, trees capture residuals
//! - **RandomForest**: Parallel trees with bootstrap sampling
//!
//! # Design Rationale
//!
//! Most tabular problems are solved by Linear, Tree, or their combination.
//! UniversalModel provides a single interface for all three patterns.
//!
//! ## Architecture
//!
//! - **PureTree** wraps `GBDTModel` directly - you get GPU acceleration, conformal
//!   prediction, multi-class, and all mature GBDTModel features.
//! - **LinearThenTree** and **RandomForest** are specialized modes that don't
//!   fit the standard GBDT pattern.
//!
//! ## When to Use Each Mode
//!
//! - **PureTree**: Most tabular problems, categorical-heavy data
//! - **LinearThenTree**: Time-series with trends, extrapolation beyond training range
//! - **RandomForest**: When robustness and variance reduction are priorities
//!
//! # Automatic Mode Selection
//!
//! TreeBoost can automatically analyze your dataset and pick the best mode:
//!
//! ```ignore
//! use treeboost::{UniversalModel, MseLoss};
//!
//! // Let TreeBoost analyze the data and pick the best mode
//! let model = UniversalModel::auto(&dataset, &MseLoss)?;
//!
//! // See why it picked this mode
//! println!("{}", model.analysis_report().unwrap());
//! ```
//!
//! The analysis runs lightweight "probes" (quick linear and tree models on subsamples)
//! to measure signal strength WITHOUT the cost of full training. A 5-second analysis
//! beats a 4-hour search.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::model::{UniversalModel, BoostingMode, UniversalConfig};
//!
//! let config = UniversalConfig::default()
//!     .with_mode(BoostingMode::LinearThenTree)
//!     .with_num_rounds(100);
//!
//! let model = UniversalModel::train(&dataset, config)?;
//! let predictions = model.predict(&dataset);
//! ```

use crate::analysis::{AnalysisConfig, Confidence, DatasetAnalysis};
use crate::booster::{GBDTConfig, GBDTModel};
use crate::dataset::BinnedDataset;
use crate::learner::{LinearBooster, LinearConfig, TreeBooster, TreeConfig, WeakLearner};
use crate::loss::LossFunction;
use crate::tree::Tree;
use crate::Result;
use rand::SeedableRng;
use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

// =============================================================================
// BoostingMode
// =============================================================================

/// Boosting mode for UniversalModel
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum BoostingMode {
    /// Pure GBDT: Standard histogram-based tree boosting
    ///
    /// Best for: Most tabular problems, categorical-heavy data
    #[default]
    PureTree,

    /// Linear + Tree: Linear model first, then trees on residuals
    ///
    /// Best for: Time-series with trends, extrapolation beyond training range
    ///
    /// How it works:
    /// 1. Train LinearBooster to capture global trend
    /// 2. Compute residuals: r = y - linear_pred
    /// 3. Train TreeBoosters on residuals (non-linear nuances)
    /// 4. Final: linear_pred + tree_pred
    LinearThenTree,

    /// Random Forest: Parallel trees with bootstrap sampling
    ///
    /// Best for: Robustness, variance reduction, when overfitting is a concern
    ///
    /// How it works:
    /// - learning_rate = 1.0 (full contribution per tree)
    /// - Each tree trained on bootstrap sample
    /// - Trees trained in parallel (independent)
    /// - Average predictions
    RandomForest,
}

// =============================================================================
// ModeSelection - How to choose the boosting mode
// =============================================================================

/// How to select the boosting mode
///
/// TreeBoost provides three ways to select the boosting mode:
///
/// 1. **Auto** (recommended): Let TreeBoost analyze the data and pick the best mode
/// 2. **AutoWithConfig**: Auto with custom analysis configuration
/// 3. **Fixed**: You explicitly specify the mode
///
/// # Example: Auto Selection
///
/// ```ignore
/// use treeboost::{UniversalModel, UniversalConfig, ModeSelection, MseLoss};
///
/// let config = UniversalConfig::new()
///     .with_mode_selection(ModeSelection::Auto)
///     .with_num_rounds(100);
///
/// let model = UniversalModel::train_smart(&dataset, config, &MseLoss)?;
/// println!("Selected mode: {:?}", model.mode());
/// println!("Confidence: {:?}", model.selection_confidence());
/// ```
///
/// # Example: Fixed Mode
///
/// ```ignore
/// use treeboost::{UniversalModel, UniversalConfig, ModeSelection, BoostingMode};
///
/// let config = UniversalConfig::new()
///     .with_mode_selection(ModeSelection::Fixed(BoostingMode::LinearThenTree))
///     .with_num_rounds(100);
/// ```
#[derive(Debug, Clone)]
pub enum ModeSelection {
    /// Automatically analyze the dataset and pick the best mode
    ///
    /// This runs lightweight "probes" (quick models on subsamples) to measure:
    /// - Linear signal strength (R²)
    /// - Non-linear structure (tree gain on residuals)
    /// - Categorical feature ratio
    /// - Noise floor
    /// - Monotonicity of relationships
    ///
    /// Based on these metrics, TreeBoost picks the mode with the highest score
    /// and provides confidence level and full explanation.
    Auto,

    /// Auto mode selection with custom analysis configuration
    ///
    /// Use this to control:
    /// - Sample size for analysis
    /// - Probe depth and iterations
    /// - Number of features to analyze
    AutoWithConfig(AnalysisConfig),

    /// Explicitly specify the boosting mode
    ///
    /// Use this when you know what mode works best for your data,
    /// or when you want to override the automatic selection.
    Fixed(BoostingMode),
}

impl Default for ModeSelection {
    fn default() -> Self {
        // Default is Fixed(PureTree) for backwards compatibility
        // Users who want auto should explicitly opt in
        ModeSelection::Fixed(BoostingMode::PureTree)
    }
}

// =============================================================================
// UniversalConfig
// =============================================================================

/// Configuration for UniversalModel
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct UniversalConfig {
    /// Boosting mode
    pub mode: BoostingMode,

    /// Number of boosting rounds (trees for PureTree/LinearThenTree, or total trees for RF)
    pub num_rounds: usize,

    /// Tree configuration
    pub tree_config: TreeConfig,

    /// Linear configuration (for LinearThenTree mode)
    pub linear_config: LinearConfig,

    /// Learning rate for gradient boosting (not used in RandomForest mode)
    pub learning_rate: f32,

    /// Row subsampling ratio (0.0-1.0)
    pub subsample: f32,

    /// Validation ratio for early stopping (0.0 to disable)
    pub validation_ratio: f32,

    /// Early stopping rounds (0 to disable)
    pub early_stopping_rounds: usize,

    /// Random seed
    pub seed: u64,

    /// Number of linear boosting rounds before trees (LinearThenTree mode)
    pub linear_rounds: usize,

    /// Maximum memory (MB) for LinearThenTree raw feature extraction
    ///
    /// LinearThenTree mode unpacks binned u8 data to f32 (4x expansion).
    /// Set this to limit memory usage. If exceeded:
    /// - `0` = No limit (default, may cause OOM on large datasets)
    /// - `> 0` = Error if estimated memory exceeds this limit
    ///
    /// **Rule of thumb**: 100M rows × 100 features = ~40GB memory
    pub max_linear_memory_mb: usize,
}

impl Default for UniversalConfig {
    fn default() -> Self {
        Self {
            mode: BoostingMode::PureTree,
            num_rounds: 100,
            tree_config: TreeConfig::default(),
            linear_config: LinearConfig::default(),
            learning_rate: 0.1,
            subsample: 1.0,
            validation_ratio: 0.0,
            early_stopping_rounds: 0,
            seed: 42,
            linear_rounds: 1,  // Single round with many CD iterations (fit once)
            max_linear_memory_mb: 0, // No limit by default
        }
    }
}

impl UniversalConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mode(mut self, mode: BoostingMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_num_rounds(mut self, rounds: usize) -> Self {
        self.num_rounds = rounds;
        self
    }

    pub fn with_tree_config(mut self, config: TreeConfig) -> Self {
        self.tree_config = config;
        self
    }

    pub fn with_linear_config(mut self, config: LinearConfig) -> Self {
        self.linear_config = config;
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr.clamp(0.0, 1.0);
        self
    }

    pub fn with_subsample(mut self, ratio: f32) -> Self {
        self.subsample = ratio.clamp(0.0, 1.0);
        self
    }

    pub fn with_validation_ratio(mut self, ratio: f32) -> Self {
        self.validation_ratio = ratio.clamp(0.0, 0.5);
        self
    }

    pub fn with_early_stopping_rounds(mut self, rounds: usize) -> Self {
        self.early_stopping_rounds = rounds;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_linear_rounds(mut self, rounds: usize) -> Self {
        self.linear_rounds = rounds;
        self
    }

    /// Set maximum memory (MB) for LinearThenTree raw feature extraction
    ///
    /// # Arguments
    /// - `mb`: Maximum memory in megabytes (0 = no limit)
    ///
    /// # Example
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_mode(BoostingMode::LinearThenTree)
    ///     .with_max_linear_memory_mb(4096); // 4GB limit
    /// ```
    pub fn with_max_linear_memory_mb(mut self, mb: usize) -> Self {
        self.max_linear_memory_mb = mb;
        self
    }

    /// Estimate memory usage (bytes) for LinearThenTree raw feature extraction
    pub fn estimate_linear_memory(&self, num_rows: usize, num_features: usize) -> usize {
        // f32 = 4 bytes per element
        num_rows * num_features * 4
    }
}

// =============================================================================
// UniversalModel
// =============================================================================

/// Unified boosting model supporting multiple modes
///
/// # Modes
///
/// - **PureTree**: Wraps `GBDTModel` - full GPU, conformal, multi-class support
/// - **LinearThenTree**: Linear model captures trend, trees capture residuals
/// - **RandomForest**: Parallel trees with bootstrap sampling
///
/// # Automatic Mode Selection
///
/// Use [`UniversalModel::auto()`] to let TreeBoost analyze your data and pick the best mode.
/// The analysis result is stored and can be retrieved with [`UniversalModel::analysis()`].
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct UniversalModel {
    /// Training configuration
    config: UniversalConfig,

    /// GBDTModel for PureTree mode (wraps the mature implementation)
    gbdt_model: Option<GBDTModel>,

    /// Linear booster (for LinearThenTree mode)
    linear_booster: Option<LinearBooster>,

    /// Ensemble of trained trees (for LinearThenTree and RandomForest modes)
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
}

impl UniversalModel {
    /// Train a UniversalModel on binned data
    pub fn train(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        match config.mode {
            BoostingMode::PureTree => Self::train_pure_tree(dataset, config, loss_fn, None),
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(dataset, None, None, config, loss_fn, None),
            BoostingMode::RandomForest => Self::train_random_forest(dataset, config, loss_fn, None),
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
        match config.mode {
            BoostingMode::PureTree => Self::train_pure_tree(dataset, config, loss_fn, None),
            BoostingMode::LinearThenTree => {
                Self::train_linear_then_tree(dataset, Some(raw_features), None, config, loss_fn, None)
            }
            BoostingMode::RandomForest => Self::train_random_forest(dataset, config, loss_fn, None),
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
        match config.mode {
            BoostingMode::PureTree => Self::train_pure_tree(dataset, config, loss_fn, None),
            BoostingMode::LinearThenTree => {
                Self::train_linear_then_tree(
                    dataset,
                    Some(raw_features),
                    Some(linear_feature_indices),
                    config,
                    loss_fn,
                    None,
                )
            }
            BoostingMode::RandomForest => Self::train_random_forest(dataset, config, loss_fn, None),
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
            BoostingMode::PureTree => Self::train_pure_tree(dataset, config, loss_fn, Some(analysis)),
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(dataset, None, None, config, loss_fn, Some(analysis)),
            BoostingMode::RandomForest => Self::train_random_forest(dataset, config, loss_fn, Some(analysis)),
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
    fn to_gbdt_config(config: &UniversalConfig, loss_fn: &dyn LossFunction) -> GBDTConfig {
        let mut gbdt_config = GBDTConfig::new()
            .with_num_rounds(config.num_rounds)
            .with_learning_rate(config.learning_rate)
            .with_max_depth(config.tree_config.max_depth)
            .with_max_leaves(config.tree_config.max_leaves)
            .with_lambda(config.tree_config.lambda)
            .with_entropy_weight(config.tree_config.entropy_weight)  // Pass entropy regularization
            .with_subsample(config.subsample)
            .with_seed(config.seed);

        // Early stopping
        if config.early_stopping_rounds > 0 && config.validation_ratio > 0.0 {
            gbdt_config = gbdt_config.with_early_stopping(
                config.early_stopping_rounds,
                config.validation_ratio,
            );
        }

        // Set loss type based on loss_fn type name (best effort)
        let loss_name = std::any::type_name_of_val(loss_fn);
        if loss_name.contains("PseudoHuber") {
            gbdt_config = gbdt_config.with_pseudo_huber_loss(1.0);
        } else if loss_name.contains("BinaryLogLoss") || loss_name.contains("LogLoss") {
            gbdt_config = gbdt_config.with_binary_logloss();
        }
        // Default is MSE which is already the default

        gbdt_config
    }

    // =========================================================================
    // PureTree Mode - Delegates to GBDTModel
    // =========================================================================

    fn train_pure_tree(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        analysis: Option<DatasetAnalysis>,
    ) -> Result<Self> {
        let num_features = dataset.num_features();

        // Convert config and delegate to GBDTModel
        let gbdt_config = Self::to_gbdt_config(&config, loss_fn);
        let gbdt_model = GBDTModel::train_binned(dataset, gbdt_config)?;

        Ok(Self {
            config,
            gbdt_model: Some(gbdt_model),
            linear_booster: None,
            trees: Vec::new(),
            base_prediction: 0.0, // Not used - GBDTModel handles this
            num_features,
            analysis,
            raw_features_for_linear: None,
            linear_feature_indices: None,
            num_linear_features: None,
        })
    }

    // =========================================================================
    // LinearThenTree Mode - Linear phase + GBDTModel on residuals
    // =========================================================================
    // Uses Newton-step Coordinate Descent for the linear phase (gradient/hessian).
    // This provides implicit regularization via learning_rate and captures global
    // trends that trees cannot extrapolate.

    fn train_linear_then_tree(
        dataset: &BinnedDataset,
        raw_features_opt: Option<&[f32]>,
        linear_feature_indices_opt: Option<&[usize]>,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
        analysis: Option<DatasetAnalysis>,
    ) -> Result<Self> {
        let targets = dataset.targets();
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        // Determine which features to use for linear model
        let linear_indices: Option<Vec<usize>> = linear_feature_indices_opt.map(|v| v.to_vec());
        let num_linear_features = linear_indices.as_ref().map(|v| v.len()).unwrap_or(num_features);

        // Get raw features: use provided ones or extract from bins (lossy fallback)
        let raw_features: Vec<f32> = if let Some(provided) = raw_features_opt {
            // Validate size
            if provided.len() != num_rows * num_features {
                return Err(crate::TreeBoostError::Data(format!(
                    "Raw features size mismatch: expected {} ({}×{}), got {}",
                    num_rows * num_features, num_rows, num_features, provided.len()
                )));
            }
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

        // Base prediction (mean target)
        let base_prediction = loss_fn.initial_prediction(targets);

        // =====================================================================
        // Phase 1: Fit Linear Model using Newton-step Coordinate Descent
        // =====================================================================
        // Iterative gradient-based fitting with learning_rate provides implicit
        // regularization. This is more robust than closed-form Ridge for
        // generalization.

        // Extract selected features for linear model if indices are specified
        let linear_features: Vec<f32> = if let Some(ref indices) = linear_indices {
            // Extract only selected features
            let mut selected = Vec::with_capacity(num_rows * indices.len());
            for row in 0..num_rows {
                for &feat_idx in indices {
                    selected.push(raw_features[row * num_features + feat_idx]);
                }
            }
            selected
        } else {
            raw_features.clone()
        };

        let mut linear_booster = LinearBooster::new(num_linear_features, config.linear_config.clone());

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
            linear_booster.fit_on_gradients(&linear_features, num_linear_features, &gradients, &hessians)?;

            // Update predictions with learning rate
            let lr = config.linear_config.learning_rate;
            for (i, pred) in predictions.iter_mut().enumerate().take(num_rows) {
                let linear_pred = linear_booster.predict_row(&linear_features, num_linear_features, i);
                *pred += lr * linear_pred;
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

        // Train GBDTModel on residuals (gets GPU, early stopping, etc.)
        let gbdt_config = Self::to_gbdt_config(&config, loss_fn);
        let gbdt_model = GBDTModel::train_binned(&residual_dataset, gbdt_config)?;

        Ok(Self {
            config,
            gbdt_model: Some(gbdt_model),
            linear_booster: Some(linear_booster),
            trees: Vec::new(), // Not used - GBDTModel stores trees
            base_prediction,
            num_features,
            analysis,
            raw_features_for_linear: None, // Training features not needed for prediction (model is fitted)
            linear_feature_indices: linear_indices,
            num_linear_features: Some(num_linear_features),
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
            linear_booster: None,
            trees,
            base_prediction,
            num_features,
            analysis,
            raw_features_for_linear: None,
            linear_feature_indices: None,
            num_linear_features: None,
        })
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Extract raw feature values from BinnedDataset
    ///
    /// Returns row-major f32 array for linear model training.
    /// Uses bin-center approximation (midpoint of bin boundaries).
    pub fn extract_raw_features(dataset: &BinnedDataset) -> Vec<f32> {
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();
        let feature_info = dataset.all_feature_info();

        let mut raw_features = vec![0.0f32; num_rows * num_features];

        for f in 0..num_features {
            let info = &feature_info[f];
            let boundaries = &info.bin_boundaries;

            for r in 0..num_rows {
                let bin = dataset.get_bin(r, f) as usize;

                // Convert bin back to approximate raw value
                // Use bin center as the raw value approximation
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

                raw_features[r * num_features + f] = raw_value;
            }
        }

        raw_features
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
                // Delegate entirely to GBDTModel
                self.gbdt_model
                    .as_ref()
                    .map(|m| m.predict(dataset))
                    .unwrap_or_else(|| vec![0.0; dataset.num_rows()])
            }
            BoostingMode::LinearThenTree => {
                // Linear contribution + GBDTModel (trained on residuals)
                let num_rows = dataset.num_rows();
                let mut predictions = vec![self.base_prediction; num_rows];

                // Add linear contribution
                if let Some(ref linear) = self.linear_booster {
                    let raw_features = Self::extract_raw_features(dataset);
                    let linear_preds = linear.predict_batch(&raw_features, self.num_features);
                    for i in 0..num_rows {
                        predictions[i] += linear_preds[i];
                    }
                }

                // Add tree contribution (GBDTModel was trained on residuals)
                // IMPORTANT: Subtract gbdt's base_prediction to avoid double-counting
                // gbdt.predict() returns (gbdt_base + tree_sum), but we already have ltt_base
                if let Some(ref gbdt) = self.gbdt_model {
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
    pub fn predict_with_raw_features(&self, dataset: &BinnedDataset, raw_features: &[f32]) -> Vec<f32> {
        match self.config.mode {
            BoostingMode::PureTree => {
                // Delegate entirely to GBDTModel (raw features not used for trees)
                self.gbdt_model
                    .as_ref()
                    .map(|m| m.predict(dataset))
                    .unwrap_or_else(|| vec![0.0; dataset.num_rows()])
            }
            BoostingMode::LinearThenTree => {
                // Linear contribution uses raw features + GBDTModel uses binned
                let num_rows = dataset.num_rows();
                let mut predictions = vec![self.base_prediction; num_rows];

                // Add linear contribution using raw features (possibly selected subset)
                if let Some(ref linear) = self.linear_booster {
                    let num_lin_feats = self.num_linear_features.unwrap_or(self.num_features);

                    // Extract selected features if indices are specified
                    let linear_features: Vec<f32> = if let Some(ref indices) = self.linear_feature_indices {
                        let mut selected = Vec::with_capacity(num_rows * indices.len());
                        for row in 0..num_rows {
                            for &feat_idx in indices {
                                selected.push(raw_features[row * self.num_features + feat_idx]);
                            }
                        }
                        selected
                    } else {
                        raw_features.to_vec()
                    };

                    let linear_preds = linear.predict_batch(&linear_features, num_lin_feats);
                    for i in 0..num_rows {
                        predictions[i] += linear_preds[i];
                    }
                }

                // Add tree contribution (trees use binned data)
                // IMPORTANT: Subtract gbdt's base_prediction to avoid double-counting
                if let Some(ref gbdt) = self.gbdt_model {
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

                // Add linear contribution
                if let Some(ref linear) = self.linear_booster {
                    let raw_features = Self::extract_raw_features_row(dataset, row_idx);
                    pred += linear.predict_row(&raw_features, self.num_features, 0);
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
                    let tree_sum: f32 = self.trees.iter().map(|t| t.predict_row(dataset, row_idx)).sum();
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
            BoostingMode::PureTree => {
                self.gbdt_model
                    .as_ref()
                    .map(|m| m.base_prediction())
                    .unwrap_or(self.base_prediction)
            }
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
        self.gbdt_model.as_ref().and_then(|m| m.conformal_quantile())
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
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
                    Err(crate::TreeBoostError::Config("No model trained".to_string()))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
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

        let model = UniversalModel::train_with_selection(
            &dataset,
            config,
            ModeSelection::Auto,
            &loss,
        )
        .unwrap();

        // Auto selection should have analysis
        assert!(model.was_auto_selected());
        assert!(model.analysis().is_some());
    }

    #[test]
    fn test_mode_selection_default() {
        // Default should be Fixed(PureTree) for backwards compatibility
        let selection = ModeSelection::default();
        assert!(matches!(selection, ModeSelection::Fixed(BoostingMode::PureTree)));
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
}
