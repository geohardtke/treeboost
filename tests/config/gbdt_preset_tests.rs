use treeboost::defaults::{gbdt as gbdt_defaults, tree as tree_defaults};
use treeboost::{GBDTConfig, GbdtPreset};

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
    assert_eq!(cfg.conformal_quantile, gbdt_defaults::DEFAULT_CONFORMAL_QUANTILE);
}
