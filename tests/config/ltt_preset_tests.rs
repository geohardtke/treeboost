use treeboost::defaults::ltt as ltt_defaults;
use treeboost::tuner::ltt::{
    LinearHyperparams, LinearHyperparamsPreset, LttTunerConfig, LttTunerPreset, TreeHyperparams,
    TreeHyperparamsPreset,
};

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
