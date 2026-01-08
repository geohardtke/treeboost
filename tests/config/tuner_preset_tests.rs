use treeboost::defaults::tuner as tuner_defaults;
use treeboost::tuner::{ParameterSpace, SpacePreset, TunerConfig, TunerPreset};

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
    assert!(names.contains(&"goss_top_rate".to_string()));
    assert!(names.contains(&"goss_other_rate".to_string()));
}
