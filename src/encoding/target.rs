//! Ordered Target Encoding with M-Estimate Smoothing
//!
//! Encodes categorical features using target statistics while preventing leakage.
//! Uses streaming/ordered approach where each row only sees prior statistics.

use rkyv::{Archive, Deserialize, Serialize};
use rustc_hash::FxHashMap;
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

/// Statistics for a single category
#[derive(Debug, Clone, Default)]
struct CategoryStats {
    /// Running sum of target values
    sum: f64,
    /// Running count of observations
    count: u64,
}

impl CategoryStats {
    #[inline]
    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

/// Ordered Target Encoder with M-Estimate Smoothing
///
/// Encodes categorical features by replacing them with smoothed target means.
/// Uses ordered/streaming approach to prevent target leakage.
///
/// Smoothing formula (M-Estimate):
/// encoded = (n × μ_category + m × μ_global) / (n + m)
///
/// Where:
/// - n = count of category
/// - μ_category = mean target for category
/// - m = smoothing parameter
/// - μ_global = global target mean
pub struct OrderedTargetEncoder {
    /// Smoothing parameter (higher = more regularization toward global mean)
    smoothing: f64,
    /// Running global statistics
    global_stats: CategoryStats,
    /// Per-category running statistics
    category_stats: FxHashMap<String, CategoryStats>,
}

impl OrderedTargetEncoder {
    /// Create a new ordered target encoder
    ///
    /// # Arguments
    /// * `smoothing` - M-estimate smoothing parameter (typically 1.0 to 10.0)
    pub fn new(smoothing: f64) -> Self {
        assert!(smoothing >= 0.0, "smoothing must be non-negative");
        Self {
            smoothing,
            global_stats: CategoryStats::default(),
            category_stats: FxHashMap::default(),
        }
    }

    /// Reset the encoder state
    pub fn reset(&mut self) {
        self.global_stats = CategoryStats::default();
        self.category_stats.clear();
    }

    /// Encode a single value using current statistics, then update
    ///
    /// This is the ordered/streaming approach:
    /// 1. Compute encoding using statistics from prior rows only
    /// 2. Update statistics with this row's values
    ///
    /// Returns the encoded value.
    pub fn encode_and_update(&mut self, category: &str, target: f64) -> f64 {
        // Get current statistics (before seeing this row)
        let global_mean = self.global_stats.mean();
        let cat_stats = self.category_stats.get(category);

        // Compute smoothed encoding
        let encoded = match cat_stats {
            Some(stats) if stats.count > 0 => {
                let n = stats.count as f64;
                let m = self.smoothing;
                (n * stats.mean() + m * global_mean) / (n + m)
            }
            _ => global_mean, // Fall back to global mean for unseen categories
        };

        // Update statistics with this observation
        self.global_stats.sum += target;
        self.global_stats.count += 1;

        let cat_stats = self.category_stats.entry(category.to_string()).or_default();
        cat_stats.sum += target;
        cat_stats.count += 1;

        encoded
    }

    /// Encode an entire column in streaming order
    ///
    /// # Arguments
    /// * `categories` - Category values for each row
    /// * `targets` - Target values for each row
    ///
    /// Returns encoded values with same length as input.
    pub fn encode_column(&mut self, categories: &[String], targets: &[f64]) -> Vec<f64> {
        assert_eq!(categories.len(), targets.len());

        self.reset();
        let mut encoded = Vec::with_capacity(categories.len());

        for (cat, &target) in categories.iter().zip(targets.iter()) {
            encoded.push(self.encode_and_update(cat, target));
        }

        encoded
    }

    /// Encode using final statistics (for inference)
    ///
    /// After training, use this to encode new data using the learned statistics.
    pub fn encode_inference(&self, category: &str) -> f64 {
        let global_mean = self.global_stats.mean();

        match self.category_stats.get(category) {
            Some(stats) if stats.count > 0 => {
                let n = stats.count as f64;
                let m = self.smoothing;
                (n * stats.mean() + m * global_mean) / (n + m)
            }
            _ => global_mean,
        }
    }

    /// Get the learned encoding map for serialization
    pub fn get_encoding_map(&self) -> EncodingMap {
        let global_mean = self.global_stats.mean();

        let mut encodings: Vec<(String, f64)> = self
            .category_stats
            .iter()
            .map(|(cat, stats)| {
                let n = stats.count as f64;
                let m = self.smoothing;
                let encoded = (n * stats.mean() + m * global_mean) / (n + m);
                (cat.clone(), encoded)
            })
            .collect();

        // Sort for deterministic serialization
        encodings.sort_by(|a, b| a.0.cmp(&b.0));

        EncodingMap {
            encodings,
            default_value: global_mean,
            smoothing: self.smoothing,
        }
    }
}

/// Serializable encoding map for inference
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
pub struct EncodingMap {
    /// Category to encoded value mapping
    pub encodings: Vec<(String, f64)>,
    /// Default value for unknown categories
    pub default_value: f64,
    /// Smoothing parameter used during training
    pub smoothing: f64,
}

impl EncodingMap {
    /// Encode a category using the stored mapping
    pub fn encode(&self, category: &str) -> f64 {
        match self
            .encodings
            .binary_search_by(|(cat, _)| cat.as_str().cmp(category))
        {
            Ok(pos) => self.encodings[pos].1,
            Err(_) => self.default_value,
        }
    }

    /// Encode a batch of categories
    pub fn encode_batch(&self, categories: &[String]) -> Vec<f64> {
        categories.iter().map(|c| self.encode(c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordered_target_encoding() {
        let mut encoder = OrderedTargetEncoder::new(1.0);

        // First observation: no prior stats, should return 0 (global mean of empty)
        let enc1 = encoder.encode_and_update("cat_a", 100.0);
        assert_eq!(enc1, 0.0);

        // Second observation of same category: has prior stat
        let enc2 = encoder.encode_and_update("cat_a", 200.0);
        // Prior: cat_a mean = 100, global mean = 100, n=1, m=1
        // encoded = (1*100 + 1*100) / 2 = 100
        assert!((enc2 - 100.0).abs() < 1e-6);

        // New category: falls back to global mean
        let enc3 = encoder.encode_and_update("cat_b", 50.0);
        // Global mean after 2 obs: (100+200)/2 = 150
        assert!((enc3 - 150.0).abs() < 1e-6);
    }

    #[test]
    fn test_smoothing_effect() {
        // High smoothing pulls toward global mean more
        let mut encoder_low = OrderedTargetEncoder::new(0.1);
        let mut encoder_high = OrderedTargetEncoder::new(10.0);

        // Setup: global mean = 50, cat_a mean = 100
        encoder_low.encode_and_update("global", 0.0);
        encoder_low.encode_and_update("global", 100.0);
        encoder_low.encode_and_update("cat_a", 100.0);

        encoder_high.encode_and_update("global", 0.0);
        encoder_high.encode_and_update("global", 100.0);
        encoder_high.encode_and_update("cat_a", 100.0);

        // cat_a has 1 observation with value 100
        // global mean ≈ 66.67

        let enc_low = encoder_low.encode_and_update("cat_a", 100.0);
        let enc_high = encoder_high.encode_and_update("cat_a", 100.0);

        // Low smoothing: closer to category mean (100)
        // High smoothing: closer to global mean (~67)
        assert!(enc_low > enc_high);
    }

    #[test]
    fn test_encode_column() {
        let mut encoder = OrderedTargetEncoder::new(1.0);

        let categories = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
        ];
        let targets = vec![10.0, 20.0, 30.0, 40.0];

        let encoded = encoder.encode_column(&categories, &targets);

        assert_eq!(encoded.len(), 4);
        // First is 0 (no prior)
        assert_eq!(encoded[0], 0.0);
    }

    #[test]
    fn test_encoding_map() {
        let mut encoder = OrderedTargetEncoder::new(1.0);

        let categories = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "b".to_string(),
        ];
        let targets = vec![10.0, 100.0, 20.0, 200.0];

        encoder.encode_column(&categories, &targets);
        let map = encoder.get_encoding_map();

        // Test inference encoding
        let enc_a = map.encode("a");
        let enc_b = map.encode("b");
        let enc_unknown = map.encode("unknown");

        // a's mean ≈ 15, b's mean ≈ 150, global ≈ 82.5
        assert!(enc_a < enc_b);
        // Unknown falls back to global mean
        assert!((enc_unknown - map.default_value).abs() < 1e-6);
    }
}
