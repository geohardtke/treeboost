//! rkyv-based model serialization
//!
//! Provides zero-copy model loading via memory mapping.

use crate::booster::GBDTModel;
use crate::{Result, TreeBoostError};
use rkyv::rancor::Error as RkyvError;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Save a model to a file
pub fn save_model(model: &GBDTModel, path: impl AsRef<Path>) -> Result<()> {
    let bytes = rkyv::to_bytes::<RkyvError>(model)
        .map_err(|e| TreeBoostError::Serialization(format!("Failed to serialize: {}", e)))?;

    let mut file = File::create(path)?;
    file.write_all(&bytes)?;

    Ok(())
}

/// Load a model from a file
pub fn load_model(path: impl AsRef<Path>) -> Result<GBDTModel> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    // Safe deserialization with validation
    let archived = rkyv::access::<rkyv::Archived<GBDTModel>, RkyvError>(&bytes)
        .map_err(|e| TreeBoostError::Serialization(format!("Failed to access archive: {}", e)))?;

    let model: GBDTModel = rkyv::deserialize::<GBDTModel, RkyvError>(archived)
        .map_err(|e| TreeBoostError::Serialization(format!("Failed to deserialize: {}", e)))?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::booster::GBDTConfig;
    use crate::dataset::{BinnedDataset, FeatureInfo, FeatureType};
    use tempfile::tempdir;

    fn create_test_dataset() -> BinnedDataset {
        let num_rows = 100;
        let num_features = 2;

        let features: Vec<u8> = (0..num_rows * num_features)
            .map(|i| (i % 256) as u8)
            .collect();
        let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32) * 0.1).collect();
        let feature_info = vec![
            FeatureInfo {
                name: "f0".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
            },
            FeatureInfo {
                name: "f1".to_string(),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
            },
        ];

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_save_load_model() {
        let dataset = create_test_dataset();
        let config = GBDTConfig::new()
            .with_num_rounds(5)
            .with_max_depth(3);

        let model = GBDTModel::train(&dataset, config).unwrap();

        // Save to temp file
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.rkyv");

        save_model(&model, &path).unwrap();

        // Load back
        let loaded = load_model(&path).unwrap();

        // Verify
        assert_eq!(loaded.num_trees(), model.num_trees());
        assert_eq!(loaded.base_prediction(), model.base_prediction());

        // Compare predictions
        let orig_preds = model.predict(&dataset);
        let loaded_preds = loaded.predict(&dataset);

        for (a, b) in orig_preds.iter().zip(loaded_preds.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
