use treeboost::defaults::{gbdt as gbdt_defaults, linear as linear_defaults, tree as tree_defaults};
use treeboost::{BoostingMode, UniversalConfig, UniversalPreset};

#[test]
fn time_series_preset_sets_mode_and_shrinkage() {
    let cfg = UniversalConfig::default().with_preset(UniversalPreset::TimeSeries);
    assert_eq!(cfg.mode, BoostingMode::LinearThenTree);
    assert_eq!(cfg.linear_config.shrinkage_factor, linear_defaults::AGGRESSIVE_SHRINKAGE);
}

#[test]
fn noisy_tabular_preset_sets_mode_and_regularization() {
    let cfg = UniversalConfig::default().with_preset(UniversalPreset::NoisyTabular);
    assert_eq!(cfg.mode, BoostingMode::RandomForest);
    assert_eq!(cfg.tree_config.entropy_weight, tree_defaults::REGULARIZED_ENTROPY_WEIGHT);
}

#[test]
fn uncertainty_aware_preset_enables_conformal() {
    let cfg = UniversalConfig::default().with_preset(UniversalPreset::UncertaintyAware);
    assert_eq!(cfg.mode, BoostingMode::PureTree);
    assert_eq!(cfg.calibration_ratio, gbdt_defaults::CONFORMAL_CALIBRATION_RATIO);
    assert_eq!(cfg.conformal_quantile, gbdt_defaults::DEFAULT_CONFORMAL_QUANTILE);
}
