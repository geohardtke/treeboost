use treeboost::defaults::tree as tree_defaults;
use treeboost::{TreeConfig, TreePreset};

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
    assert_eq!(cfg.entropy_weight, tree_defaults::REGULARIZED_ENTROPY_WEIGHT);
    assert_eq!(cfg.lambda, tree_defaults::REGULARIZED_TREE_LAMBDA);
}

#[test]
fn expressive_preset_sets_lambda() {
    let cfg = TreeConfig::default().with_preset(TreePreset::Expressive);
    assert_eq!(cfg.lambda, tree_defaults::EXPRESSIVE_TREE_LAMBDA);
}
