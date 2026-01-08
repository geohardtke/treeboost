use treeboost::defaults::linear as linear_defaults;
use treeboost::{LinearConfig, LinearPreset};

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
    assert_eq!(cfg.shrinkage_factor, linear_defaults::AGGRESSIVE_SHRINKAGE);
}

#[test]
fn conservative_preset_sets_shrinkage() {
    let cfg = LinearConfig::default().with_preset(LinearPreset::Conservative);
    assert_eq!(cfg.shrinkage_factor, linear_defaults::CONSERVATIVE_SHRINKAGE);
}

#[test]
fn safe_ridge_sets_damping() {
    let cfg = LinearConfig::default().with_preset(LinearPreset::SafeRidge);
    assert_eq!(
        cfg.extrapolation_damping,
        linear_defaults::SAFE_EXTRAPOLATION_DAMPING
    );
}
