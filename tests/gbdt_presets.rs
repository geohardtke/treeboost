//! Test GBDT preset configurations

use treeboost::{GBDTConfig, GbdtPreset};

#[test]
fn test_robust_preset() {
    let config = GBDTConfig::default().with_preset(GbdtPreset::Robust);

    // Verify feature/row sampling is enabled
    assert_eq!(
        config.colsample, 0.8,
        "Robust preset should have colsample=0.8"
    );
    assert_eq!(
        config.subsample, 0.8,
        "Robust preset should have subsample=0.8"
    );
    assert!(!config.goss_enabled, "Robust preset should disable GOSS");
}

#[test]
fn test_accuracy_preset_updated() {
    let config = GBDTConfig::default().with_preset(GbdtPreset::Accuracy);

    // Verify NEW feature/row sampling (added in v0.2)
    assert_eq!(
        config.colsample, 0.8,
        "Accuracy preset should now have colsample=0.8"
    );
    assert_eq!(
        config.subsample, 0.8,
        "Accuracy preset should now have subsample=0.8"
    );

    // Verify existing Accuracy characteristics
    assert_eq!(
        config.max_depth, 10,
        "Accuracy preset should have deep trees"
    );
    assert_eq!(
        config.learning_rate, 0.05,
        "Accuracy preset should have lower LR"
    );
    assert_eq!(
        config.num_rounds, 200,
        "Accuracy preset should have more rounds"
    );
}

#[test]
fn test_standard_preset_no_sampling() {
    let config = GBDTConfig::default().with_preset(GbdtPreset::Standard);

    // Verify no sampling (may overfit on noisy data!)
    assert_eq!(
        config.colsample, 1.0,
        "Standard preset should have no feature sampling"
    );
    assert_eq!(
        config.subsample, 1.0,
        "Standard preset should have no row sampling"
    );
}

#[test]
fn test_all_presets_compile() {
    // Verify all presets can be constructed without panic
    let _ = GBDTConfig::default().with_preset(GbdtPreset::Standard);
    let _ = GBDTConfig::default().with_preset(GbdtPreset::Speed);
    let _ = GBDTConfig::default().with_preset(GbdtPreset::Accuracy);
    let _ = GBDTConfig::default().with_preset(GbdtPreset::SmallData);
    let _ = GBDTConfig::default().with_preset(GbdtPreset::LargeData);
    let _ = GBDTConfig::default().with_preset(GbdtPreset::Robust);
    let _ = GBDTConfig::default().with_preset(GbdtPreset::Conformal);
}
