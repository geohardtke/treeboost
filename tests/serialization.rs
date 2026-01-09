//! Serialization tests for UniversalModel and AutoModel
//!
//! Tests verify that:
//! - Models can be saved and loaded correctly
//! - Configurations can be exported to JSON
//! - Exported configs can be inspected and reused

use polars::prelude::*;
use std::fs;
use tempfile::NamedTempFile;
use treeboost::model::{AutoModel, BoostingMode, UniversalConfig, UniversalModel};

fn create_test_dataframe() -> DataFrame {
    df!(
        "x1" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "x2" => &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
        "y" => &[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0],
    )
    .unwrap()
}

#[test]
fn test_universal_model_save_load_roundtrip() {
    let df = create_test_dataframe();

    // Train with AutoML (which handles preprocessing internally)
    let auto = AutoModel::train(&df, "y").unwrap();
    let model = auto.inner().clone();

    // Save to temp file
    let temp = NamedTempFile::new().unwrap();
    let path = temp.path();

    model.save(path).unwrap();
    assert!(path.exists());

    // Load from file
    let loaded = UniversalModel::load(path).unwrap();

    // Verify loaded model properties
    assert_eq!(loaded.mode(), model.mode());
    assert_eq!(loaded.num_features(), model.num_features());
    assert_eq!(loaded.num_trees(), model.num_trees());

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_auto_model_save_config_json() {
    let df = create_test_dataframe();
    let target = "y";

    // Train with AutoML
    let auto = AutoModel::train(&df, target).unwrap();

    // Save config to temp file
    let temp = NamedTempFile::new().unwrap();
    let path = temp.path();

    auto.save_config(path).unwrap();
    assert!(path.exists());

    // Read and verify it's valid JSON
    let json_content = fs::read_to_string(path).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&json_content).expect("Config should be valid JSON");

    // Verify expected fields exist
    assert!(
        json.get("mode").is_some(),
        "Config should have 'mode' field"
    );
    // Config contains various sub-fields like tree_config, linear_config, etc.
    // Just verify the JSON is non-empty and has expected structure
    assert!(!json.is_null(), "Config should be a valid JSON object");

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_config_roundtrip_puretree() {
    let df = create_test_dataframe();

    // Train PureTree model
    let auto = AutoModel::train(&df, "y").unwrap();
    let original_config = auto.config().clone();

    // Export to JSON
    let config_json =
        serde_json::to_string_pretty(&original_config).expect("Config should serialize to JSON");

    // Re-import from JSON
    let reimported: UniversalConfig =
        serde_json::from_str(&config_json).expect("Config should deserialize from JSON");

    // Verify mode matches (TreeConfig doesn't implement PartialEq, so only check mode)
    assert_eq!(reimported.mode, original_config.mode);
}

#[test]
fn test_ensemble_config_serialization() {
    use treeboost::StackingStrategy;

    let config = UniversalConfig::new()
        .with_mode(BoostingMode::LinearThenTree)
        .with_ensemble_seeds(vec![1, 2, 3, 4, 5])
        .with_stacking_strategy(StackingStrategy::Ridge {
            alpha: 0.01,
            rank_transform: false,
            fit_intercept: true,
            min_weight: 0.01,
        });

    // Serialize to JSON
    let json =
        serde_json::to_string_pretty(&config).expect("Config with ensemble should serialize");

    // Deserialize back
    let loaded: UniversalConfig = serde_json::from_str(&json).expect("Config should deserialize");

    // Verify ensemble config persists
    assert!(
        loaded.ensemble_seeds.is_some(),
        "Ensemble seeds should be preserved"
    );
    assert_eq!(
        loaded.ensemble_seeds.as_ref().unwrap().len(),
        5,
        "All 5 seeds should be present"
    );

    // Verify stacking strategy persists
    match loaded.stacking_strategy {
        StackingStrategy::Ridge {
            alpha,
            fit_intercept,
            ..
        } => {
            assert!((alpha - 0.01).abs() < 0.001);
            assert!(fit_intercept);
        }
        _ => panic!("Stacking strategy should be Ridge"),
    }
}

#[test]
fn test_config_json_is_human_readable() {
    let config = UniversalConfig::new()
        .with_mode(BoostingMode::PureTree)
        .with_ensemble_seeds(vec![42, 43, 44]);

    let json = serde_json::to_string_pretty(&config).unwrap();

    // Verify it's pretty-printed (has newlines and indentation)
    assert!(
        json.contains('\n'),
        "JSON should be pretty-printed with newlines"
    );
    assert!(json.contains("  "), "JSON should have indentation");

    // Verify it contains readable field names
    assert!(json.contains("ensemble_seeds"));
    assert!(json.contains("mode"));
}

#[test]
fn test_model_and_config_save_together() {
    let df = create_test_dataframe();
    let target = "y";

    // Train with AutoML
    let auto = AutoModel::train(&df, target).unwrap();

    // Use temp directory
    let temp_dir = tempfile::tempdir().unwrap();

    // Save both
    let model_path = temp_dir.path().join("model.rkyv");
    let config_path = temp_dir.path().join("config.json");

    auto.save(&model_path).unwrap();
    auto.save_config(&config_path).unwrap();

    // Verify both exist
    assert!(model_path.exists(), "Model file should exist");
    assert!(config_path.exists(), "Config file should exist");

    // Verify config is JSON
    let config_json = fs::read_to_string(&config_path).unwrap();
    serde_json::from_str::<serde_json::Value>(&config_json).expect("Config should be valid JSON");

    // Verify model can be loaded
    let loaded_model = UniversalModel::load(&model_path).unwrap();
    assert_eq!(loaded_model.config().mode, auto.config().mode);
}
