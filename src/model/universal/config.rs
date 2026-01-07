//! Configuration for UniversalModel

use crate::learner::{LinearConfig, TreeConfig};
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
            linear_rounds: 1, // Single round with many CD iterations (fit once)
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
