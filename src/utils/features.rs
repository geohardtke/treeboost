//! Feature selection and extraction utilities
//!
//! This module provides shared utility functions for feature selection and extraction
//! that are used across different parts of the codebase (AutoBuilder, UniversalModel, etc.).

/// Extract selected features from raw feature array based on feature indices.
///
/// This utility is used in LinearThenTree mode to select a subset of features for
/// the linear model while trees use all features.
///
/// # Arguments
///
/// * `raw_features` - Flat array of features in row-major order [row0_feat0, row0_feat1, ..., row1_feat0, ...]
/// * `num_rows` - Number of rows in the dataset
/// * `num_raw_features` - Total number of features per row in raw_features
/// * `indices` - Optional feature indices to select. If None, returns all features.
///
/// # Returns
///
/// A new feature array containing only the selected features in row-major order.
///
/// # Examples
///
/// ```ignore
/// let raw = vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0];  // 2 rows, 3 features
/// let selected = extract_selected_features(&raw, 2, 3, Some(&[0, 2]));
/// // Result: [1.0, 3.0,  4.0, 6.0]  // Features 0 and 2 for each row
/// ```
pub fn extract_selected_features(
    raw_features: &[f32],
    num_rows: usize,
    num_raw_features: usize,
    indices: Option<&[usize]>,
) -> Vec<f32> {
    if let Some(indices) = indices {
        let mut selected = Vec::with_capacity(num_rows * indices.len());
        for row in 0..num_rows {
            let row_offset = row * num_raw_features;
            for &feat_idx in indices {
                selected.push(raw_features[row_offset + feat_idx]);
            }
        }
        selected
    } else {
        raw_features.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_all_features() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = extract_selected_features(&raw, 2, 3, None);
        assert_eq!(result, raw);
    }

    #[test]
    fn test_extract_selected_features() {
        // 2 rows, 3 features per row
        let raw = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];

        // Select features 0 and 2
        let indices = vec![0, 2];
        let result = extract_selected_features(&raw, 2, 3, Some(&indices));

        assert_eq!(
            result,
            vec![
                1.0, 3.0, // row 0: features 0, 2
                4.0, 6.0, // row 1: features 0, 2
            ]
        );
    }

    #[test]
    fn test_extract_single_feature() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![1];
        let result = extract_selected_features(&raw, 2, 3, Some(&indices));
        assert_eq!(result, vec![2.0, 5.0]); // Feature 1 from each row
    }

    #[test]
    fn test_extract_reordered_features() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![2, 0]; // Reverse order
        let result = extract_selected_features(&raw, 2, 3, Some(&indices));
        assert_eq!(result, vec![3.0, 1.0, 6.0, 4.0]);
    }
}
