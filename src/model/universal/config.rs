//! Configuration for UniversalModel

use crate::dataset::feature_extractor::FeatureExtractor;
use crate::defaults::{
    gbdt as gbdt_defaults, seeds as seeds_defaults, tree as tree_defaults,
    universal as universal_defaults,
};
use crate::learner::{LinearConfig, LinearPreset, TreeConfig, TreePreset};
use crate::model::universal::mode::BoostingMode;
use rkyv::{Archive, Deserialize, Serialize};

// =============================================================================
// UniversalConfig
// =============================================================================

/// Configuration for UniversalModel
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
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

    /// Calibration set ratio for conformal prediction (0.0 to disable)
    pub calibration_ratio: f32,

    /// Conformal prediction quantile (e.g., 0.9 for 90% coverage)
    pub conformal_quantile: f32,

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

    /// Feature extractor for LinearThenTree mode
    ///
    /// Used to extract raw numeric features from DataFrame for the linear component.
    /// If None, LinearThenTree expects pre-extracted raw features.
    #[rkyv(with = rkyv::with::Skip)]
    #[serde(skip)]
    pub feature_extractor: Option<FeatureExtractor>,
}

impl Default for UniversalConfig {
    fn default() -> Self {
        Self {
            mode: BoostingMode::PureTree,
            num_rounds: gbdt_defaults::DEFAULT_NUM_ROUNDS,
            tree_config: TreeConfig::default(),
            linear_config: LinearConfig::default(),
            learning_rate: tree_defaults::DEFAULT_LEARNING_RATE,
            subsample: gbdt_defaults::DEFAULT_SUBSAMPLE,
            validation_ratio: gbdt_defaults::DEFAULT_VALIDATION_RATIO,
            early_stopping_rounds: gbdt_defaults::DEFAULT_EARLY_STOPPING_ROUNDS,
            calibration_ratio: gbdt_defaults::DEFAULT_CALIBRATION_RATIO,
            conformal_quantile: gbdt_defaults::DEFAULT_CONFORMAL_QUANTILE,
            seed: seeds_defaults::DEFAULT_SEED,
            linear_rounds: universal_defaults::DEFAULT_LINEAR_ROUNDS, // Single round with many CD iterations (fit once)
            max_linear_memory_mb: universal_defaults::DEFAULT_MAX_LINEAR_MEMORY_MB, // No limit by default
            feature_extractor: None,
        }
    }
}

/// Presets for common UniversalModel configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniversalPreset {
    /// Standard GBDT (GBDTConfig wrapper).
    PureTree,
    /// Linear first, then tree residuals.
    LinearThenTree,
    /// Bagged trees for variance reduction.
    RandomForest,
    /// LinearThenTree + aggressive linear shrinkage.
    TimeSeries,
    /// RandomForest + regularized trees.
    NoisyTabular,
    /// PureTree + conformal calibration.
    UncertaintyAware,
}

impl UniversalConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: UniversalPreset) -> Self {
        match preset {
            UniversalPreset::PureTree => {
                self.mode = BoostingMode::PureTree;
            }
            UniversalPreset::LinearThenTree => {
                self.mode = BoostingMode::LinearThenTree;
            }
            UniversalPreset::RandomForest => {
                self.mode = BoostingMode::RandomForest;
            }
            UniversalPreset::TimeSeries => {
                self.mode = BoostingMode::LinearThenTree;
                self.linear_config = self.linear_config.with_preset(LinearPreset::Aggressive);
            }
            UniversalPreset::NoisyTabular => {
                self.mode = BoostingMode::RandomForest;
                self.tree_config = self.tree_config.with_preset(TreePreset::Regularized);
            }
            UniversalPreset::UncertaintyAware => {
                self.mode = BoostingMode::PureTree;
                self.calibration_ratio = gbdt_defaults::CONFORMAL_CALIBRATION_RATIO;
                self.conformal_quantile = gbdt_defaults::DEFAULT_CONFORMAL_QUANTILE;
            }
        }
        self
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

    /// Enable conformal calibration for uncertainty estimates.
    pub fn with_conformal_calibration(mut self, ratio: f32, quantile: f32) -> Self {
        self.calibration_ratio = ratio.clamp(0.0, 0.5);
        self.conformal_quantile = quantile.clamp(0.5, 0.99);
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

    /// Set the feature extractor for LinearThenTree mode
    ///
    /// The feature extractor is used to extract raw numeric features from DataFrames
    /// for the linear component in LinearThenTree mode. This allows automatic feature
    /// selection and handling of non-numeric columns.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_mode(BoostingMode::LinearThenTree)
    ///     .with_feature_extractor(Some(FeatureExtractor::default()));
    /// ```
    pub fn with_feature_extractor(mut self, extractor: Option<FeatureExtractor>) -> Self {
        self.feature_extractor = extractor;
        self
    }

    /// Estimate memory usage (bytes) for LinearThenTree raw feature extraction
    pub fn estimate_linear_memory(&self, num_rows: usize, num_features: usize) -> usize {
        // f32 = 4 bytes per element
        num_rows * num_features * 4
    }
}
