#[allow(deprecated)]
#[test]
fn deprecated_builders_still_work() {
    let linear = treeboost::LinearConfig::ridge(1.0);
    assert_eq!(linear.l1_ratio, 0.0);

    let tuner = treeboost::tuner::TunerConfig::quick();
    assert!(tuner.n_iterations > 0);

    let space = treeboost::tuner::ParameterSpace::default_regression();
    assert!(!space.param_names().is_empty());
}
