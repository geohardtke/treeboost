//! Boosting mode selection for UniversalModel

use crate::analysis::AnalysisConfig;
use rkyv::{Archive, Deserialize, Serialize};

// =============================================================================
// BoostingMode
// =============================================================================

/// Boosting mode for UniversalModel
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    serde::Serialize,
    serde::Deserialize,
)]
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
#[derive(Debug, Clone, PartialEq)]
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

impl ModeSelection {
    /// Check if this is automatic mode selection
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto | Self::AutoWithConfig(_))
    }

    /// Get the fixed mode, if any
    pub fn fixed_mode(&self) -> Option<BoostingMode> {
        match self {
            Self::Fixed(mode) => Some(*mode),
            _ => None,
        }
    }
}
