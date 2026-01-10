//! Integration tests for configuration presets
//!
//! Tests all preset configurations work correctly:
//! - Backend presets (GpuRequired, CpuOnly)
//! - GBDT presets (Speed, Accuracy, Conformal)
//! - Linear presets (Ridge, Lasso, ElasticNet, Aggressive, Conservative)
//! - Tree presets (Deep, Shallow, Regularized, Expressive)
//! - Universal presets (TimeSeries, NoisyTabular, UncertaintyAware)
//! - Tuner presets (Quick, Thorough)
//! - LTT presets (Quick, LinearHyperparams, TreeHyperparams)

use treeboost::defaults::learners::{gbdt as gbdt_defaults, linear as linear_defaults, tree as tree_defaults};
use treeboost::defaults::tuning::{ltt as ltt_defaults, tuner as tuner_defaults};
use treeboost::tuner::ltt::{
    LinearHyperparams, LinearHyperparamsPreset, LttTunerConfig, LttTunerPreset, TreeHyperparams,
    TreeHyperparamsPreset,
};
use treeboost::tuner::{ParameterSpace, SpacePreset, TunerConfig, TunerPreset};
use treeboost::{
    BackendConfig, BackendPreset, BackendType, BoostingMode, GBDTConfig, GbdtPreset, LinearConfig,
    LinearPreset, TreeConfig, TreePreset, UniversalConfig, UniversalPreset,
};

// =============================================================================
// Backend Presets
// =============================================================================

#[test]
fn gpu_required_disables_fallback() {
    let cfg = BackendConfig::default().with_preset(BackendPreset::GpuRequired);
    assert_eq!(cfg.preferred, BackendType::Wgpu);
    assert!(!cfg.fallback_to_scalar);
}

#[test]
fn cpu_only_uses_scalar() {
    let cfg = BackendConfig::default().with_preset(BackendPreset::CpuOnly);
    assert_eq!(cfg.preferred, BackendType::Scalar);
    assert!(cfg.fallback_to_scalar);
}

// =============================================================================
// GBDT Presets
// =============================================================================

#[test]
fn speed_preset_enables_goss() {
    let cfg = GBDTConfig::default().with_preset(GbdtPreset::Speed);
    assert!(cfg.goss_enabled);
    assert_eq!(cfg.max_depth, tree_defaults::SHALLOW_MAX_DEPTH);
}

#[test]
fn accuracy_preset_deepens_trees() {
    let cfg = GBDTConfig::default().with_preset(GbdtPreset::Accuracy);
    assert_eq!(cfg.max_depth, tree_defaults::DEEP_MAX_DEPTH);
    assert_eq!(cfg.num_rounds, gbdt_defaults::DEFAULT_NUM_ROUNDS * 2);
}

#[test]
fn conformal_preset_enables_calibration() {
    let cfg = GBDTConfig::default().with_preset(GbdtPreset::Conformal);
    assert_eq!(
        cfg.calibration_ratio,
        gbdt_defaults::CONFORMAL_CALIBRATION_RATIO
    );
    assert_eq!(
        cfg.conformal_quantile,
        gbdt_defaults::DEFAULT_CONFORMAL_QUANTILE
    );
}

// =============================================================================
// Linear Presets
// =============================================================================

#[test]
fn ridge_preset_sets_l1_ratio() {
    let cfg = LinearConfig::default().with_preset(LinearPreset::Ridge);
    assert_eq!(cfg.l1_ratio, linear_defaults::DEFAULT_L1_RATIO);
}

#[test]
fn lasso_preset_sets_l1_ratio() {
    let cfg = LinearConfig::default().with_preset(LinearPreset::Lasso);
    assert_eq!(cfg.l1_ratio, linear_defaults::LASSO_L1_RATIO);
}

#[test]
fn elastic_net_preset_sets_l1_ratio() {
    let cfg = LinearConfig::default().with_preset(LinearPreset::ElasticNet);
    assert_eq!(cfg.l1_ratio, linear_defaults::ELASTIC_NET_L1_RATIO);
}

#[test]
fn aggressive_preset_sets_shrinkage() {
    let cfg = LinearConfig::default().with_preset(LinearPreset::Aggressive);
    assert_eq!(
        cfg.shrinkage_factor,
        linear_defaults::AGGRESSIVE_SHRINKAGE
    );
}

#[test]
fn conservative_preset_sets_shrinkage() {
    let cfg = LinearConfig::default().with_preset(LinearPreset::Conservative);
    assert_eq!(
        cfg.shrinkage_factor,
        linear_defaults::CONSERVATIVE_SHRINKAGE
    );
}

#[test]
fn safe_ridge_sets_damping() {
    let cfg = LinearConfig::default().with_preset(LinearPreset::SafeRidge);
    assert_eq!(
        cfg.extrapolation_damping,
        linear_defaults::SAFE_EXTRAPOLATION_DAMPING
    );
}

// =============================================================================
// Tree Presets
// =============================================================================

#[test]
fn deep_preset_sets_depth() {
    let cfg = TreeConfig::default().with_preset(TreePreset::Deep);
    assert_eq!(cfg.max_depth, tree_defaults::DEEP_MAX_DEPTH);
}

#[test]
fn shallow_preset_sets_depth() {
    let cfg = TreeConfig::default().with_preset(TreePreset::Shallow);
    assert_eq!(cfg.max_depth, tree_defaults::SHALLOW_MAX_DEPTH);
}

#[test]
fn regularized_preset_sets_entropy_weight() {
    let cfg = TreeConfig::default().with_preset(TreePreset::Regularized);
    assert_eq!(
        cfg.entropy_weight,
        tree_defaults::REGULARIZED_ENTROPY_WEIGHT
    );
    assert_eq!(cfg.lambda, tree_defaults::REGULARIZED_TREE_LAMBDA);
}

#[test]
fn expressive_preset_sets_lambda() {
    let cfg = TreeConfig::default().with_preset(TreePreset::Expressive);
    assert_eq!(cfg.lambda, tree_defaults::EXPRESSIVE_TREE_LAMBDA);
}

// =============================================================================
// Universal Presets
// =============================================================================

#[test]
fn time_series_preset_sets_mode_and_shrinkage() {
    let cfg = UniversalConfig::default().with_preset(UniversalPreset::TimeSeries);
    assert_eq!(cfg.mode, BoostingMode::LinearThenTree);
    assert_eq!(
        cfg.linear_config.shrinkage_factor,
        linear_defaults::AGGRESSIVE_SHRINKAGE
    );
}

#[test]
fn noisy_tabular_preset_sets_mode_and_regularization() {
    let cfg = UniversalConfig::default().with_preset(UniversalPreset::NoisyTabular);
    assert_eq!(cfg.mode, BoostingMode::RandomForest);
    assert_eq!(
        cfg.tree_config.entropy_weight,
        tree_defaults::REGULARIZED_ENTROPY_WEIGHT
    );
}

#[test]
fn uncertainty_aware_preset_enables_conformal() {
    let cfg = UniversalConfig::default().with_preset(UniversalPreset::UncertaintyAware);
    assert_eq!(cfg.mode, BoostingMode::PureTree);
    assert_eq!(
        cfg.calibration_ratio,
        gbdt_defaults::CONFORMAL_CALIBRATION_RATIO
    );
    assert_eq!(
        cfg.conformal_quantile,
        gbdt_defaults::DEFAULT_CONFORMAL_QUANTILE
    );
}

// =============================================================================
// Tuner Presets
// =============================================================================

#[test]
fn quick_preset_sets_iterations() {
    let cfg = TunerConfig::default().with_preset(TunerPreset::Quick);
    assert_eq!(cfg.n_iterations, tuner_defaults::QUICK_N_ITERATIONS);
    assert_eq!(cfg.num_rounds, tuner_defaults::QUICK_TUNER_ROUNDS);
}

#[test]
fn thorough_preset_sets_iterations() {
    let cfg = TunerConfig::default().with_preset(TunerPreset::Thorough);
    assert_eq!(cfg.n_iterations, tuner_defaults::THOROUGH_N_ITERATIONS);
    assert_eq!(cfg.num_rounds, tuner_defaults::THOROUGH_TUNER_ROUNDS);
}

#[test]
fn exhaustive_space_includes_goss() {
    let space = ParameterSpace::with_preset(SpacePreset::Exhaustive);
    let names = space.param_names();
    assert!(names.contains(&"goss_top_rate"));
    assert!(names.contains(&"goss_other_rate"));
}

// =============================================================================
// LTT Presets
// =============================================================================

#[test]
fn quick_preset_disables_joint_refinement() {
    let cfg = LttTunerConfig::default().with_preset(LttTunerPreset::Quick);
    assert!(!cfg.enable_joint_refinement);
    assert_eq!(cfg.lambda_values, ltt_defaults::QUICK_LAMBDA_GRID.to_vec());
}

#[test]
fn linear_hyperparams_preset_sets_l1_ratio() {
    let cfg = LinearHyperparams::default().with_preset(LinearHyperparamsPreset::Lasso);
    assert_eq!(cfg.l1_ratio, 1.0);
}

#[test]
fn tree_hyperparams_preset_sets_depth() {
    let cfg = TreeHyperparams::default().with_preset(TreeHyperparamsPreset::Aggressive);
    assert_eq!(cfg.max_depth, 8);
}
