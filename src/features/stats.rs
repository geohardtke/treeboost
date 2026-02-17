//! Statistical utilities for feature generation
//!
//! Shared correlation and statistical functions used by multiple feature generators.

/// Compute Pearson correlation between two arrays
///
/// Returns 0.0 if either array has zero variance (constant values).
pub(crate) fn correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f32;

    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let std_x = var_x.sqrt();
    let std_y = var_y.sqrt();

    if std_x < 1e-10 || std_y < 1e-10 {
        return 0.0;
    }

    cov / (std_x * std_y)
}

/// Compute correlation matrix for feature columns
///
/// # Arguments
/// * `data` - Row-major feature matrix (num_rows × num_features)
/// * `num_features` - Number of features (columns)
/// * `num_rows` - Number of rows
///
/// # Returns
/// Flattened correlation matrix of size num_features × num_features
pub(super) fn compute_correlation_matrix(
    data: &[f32],
    num_features: usize,
    num_rows: usize,
) -> Vec<f32> {
    let mut correlations = vec![0.0f32; num_features * num_features];

    // Compute means
    let means: Vec<f32> = (0..num_features)
        .map(|f| {
            let sum: f32 = (0..num_rows).map(|r| data[r * num_features + f]).sum();
            sum / num_rows as f32
        })
        .collect();

    // Compute standard deviations
    let stds: Vec<f32> = (0..num_features)
        .map(|f| {
            let var: f32 = (0..num_rows)
                .map(|r| {
                    let diff = data[r * num_features + f] - means[f];
                    diff * diff
                })
                .sum::<f32>()
                / num_rows as f32;
            var.sqrt().max(1e-10)
        })
        .collect();

    // Compute correlations
    for i in 0..num_features {
        for j in 0..num_features {
            if i == j {
                correlations[i * num_features + j] = 1.0;
            } else if j > i {
                // Only compute once (matrix is symmetric)
                let covar: f32 = (0..num_rows)
                    .map(|r| {
                        let xi = data[r * num_features + i] - means[i];
                        let xj = data[r * num_features + j] - means[j];
                        xi * xj
                    })
                    .sum::<f32>()
                    / num_rows as f32;

                let corr = covar / (stds[i] * stds[j]);
                correlations[i * num_features + j] = corr;
                correlations[j * num_features + i] = corr;
            }
        }
    }

    correlations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let corr = correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![8.0, 6.0, 4.0, 2.0];
        let corr = correlation(&x, &y);
        assert!((corr - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_zero_variance() {
        let x = vec![5.0, 5.0, 5.0, 5.0]; // constant
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let corr = correlation(&x, &y);
        assert!((corr - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_matrix() {
        // Perfect positive correlation
        let data = vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0];
        let corr = compute_correlation_matrix(&data, 2, 3);

        assert!((corr[0] - 1.0).abs() < 1e-6); // self-correlation
        assert!((corr[1] - 1.0).abs() < 1e-6); // perfect correlation
        assert!((corr[2] - 1.0).abs() < 1e-6); // symmetric
        assert!((corr[3] - 1.0).abs() < 1e-6); // self-correlation
    }

    #[test]
    fn test_correlation_matrix_uncorrelated() {
        // Uncorrelated features
        let data = vec![
            1.0, 0.0, // row 0
            0.0, 1.0, // row 1
            1.0, 0.0, // row 2
            0.0, 1.0, // row 3
        ];
        let corr = compute_correlation_matrix(&data, 2, 4);

        assert!((corr[0] - 1.0).abs() < 1e-6); // self
        assert!((corr[3] - 1.0).abs() < 1e-6); // self
        assert!((corr[1] - (-1.0)).abs() < 1e-6); // negative correlation
    }
}
