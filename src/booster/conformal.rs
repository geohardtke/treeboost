//! Conformal prediction support for GBDT models
//!
//! Provides methods for computing prediction intervals using split conformal prediction.
//! These intervals are valid under the assumption of exchangeable samples with asymptotic coverage guarantees.

use super::GBDTModel;

impl GBDTModel {
    /// Get the conformal quantile (if model was trained with conformal prediction)
    ///
    /// The quantile determines the width of prediction intervals via:
    /// - Lower bound = prediction - conformal_q
    /// - Upper bound = prediction + conformal_q
    ///
    /// Returns None if the model was not trained with conformal prediction enabled.
    pub fn conformal_quantile(&self) -> Option<f32> {
        self.conformal_q
    }

    /// Check if this model supports conformal prediction
    ///
    /// Conformal prediction is enabled when the model was trained with
    /// calibration ratio > 0 and conformal_quantile > 0 in config.
    pub fn has_conformal_quantile(&self) -> bool {
        self.conformal_q.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::booster::GBDTConfig;
    use crate::dataset::FeatureInfo;
    use crate::dataset::FeatureType;

    fn create_regression_dataset(num_rows: usize, noise: f32) -> BinnedDataset {
        let num_features = 3;

        // Generate features
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * (f + 1) * 17) % 256) as u8);
            }
        }

        // Generate targets with some pattern
        let targets: Vec<f32> = (0..num_rows)
            .map(|i| {
                let f0 = features[i] as f32 / 255.0;
                let f1 = features[num_rows + i] as f32 / 255.0;
                f0 * 10.0 + f1 * 5.0 + noise * (i as f32 % 10.0 - 5.0) / 5.0
            })
            .collect();

        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("feature_{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_train_with_conformal() {
        let dataset = create_regression_dataset(500, 0.5);

        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_conformal(0.2, 0.9)
            .unwrap();

        let model = GBDTModel::train_binned(&dataset, config).unwrap();

        assert!(model.conformal_quantile().is_some());
        assert!(model.conformal_quantile().unwrap() > 0.0);

        // Test interval prediction
        let (preds, lower, upper) = model.predict_with_intervals(&dataset);
        assert_eq!(preds.len(), dataset.num_rows());
        assert_eq!(lower.len(), dataset.num_rows());
        assert_eq!(upper.len(), dataset.num_rows());

        // Verify interval bounds
        let q = model.conformal_quantile().unwrap();
        for i in 0..preds.len() {
            assert!((lower[i] - (preds[i] - q)).abs() < 1e-5);
            assert!((upper[i] - (preds[i] + q)).abs() < 1e-5);
        }
    }

    #[test]
    fn test_conformal_quantile_basic() {
        let dataset = create_regression_dataset(300, 0.2);

        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_conformal(0.25, 0.95)
            .unwrap();

        let model = GBDTModel::train_binned(&dataset, config).unwrap();

        // Check that conformal quantile is set
        assert!(model.has_conformal_quantile());
        let q = model.conformal_quantile().unwrap();
        assert!(q > 0.0, "Conformal quantile should be positive");
    }

    #[test]
    fn test_no_conformal_by_default() {
        let dataset = create_regression_dataset(200, 0.1);

        let config = GBDTConfig::new().with_num_rounds(5);

        let model = GBDTModel::train_binned(&dataset, config).unwrap();

        // By default, conformal prediction should not be enabled
        assert!(!model.has_conformal_quantile());
        assert!(model.conformal_quantile().is_none());
    }
}
